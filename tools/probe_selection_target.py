#!/usr/bin/env python3
"""Freeze the historical G5 and test the existing selection supervision alone."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core import YAMLConfig
from src.misc import dist_utils
from tools.gt_query_selection import candidate_pool
from src.zoo.dfine.boundary_query_selection import (
    association, gather, selection_losses, quality_targets, quality_pairs)
from src.zoo.dfine.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, box_iou


class FrozenSelectionProbe(nn.Module):
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        for name, parameter in detector.named_parameters():
            parameter.requires_grad_(name.startswith('query_selector.'))
        self.train()

    def train(self, mode=True):
        super().train(mode)
        # This also freezes all BN buffers and disables detector dropout/DN.
        self.detector.eval()
        self.detector.query_selector.train(mode)
        return self

    def forward(self, images):
        if not self.training:
            return self.detector(images)
        with torch.no_grad():
            features = self.detector.encoder(self.detector.backbone(images))
            decoder = self.detector.decoder
            memory, shapes = decoder._get_encoder_input(features)

            def select(memory, logits, anchors, bbox_head, count):
                with torch.enable_grad():
                    return self.detector.query_selector(
                        memory.detach(), logits.detach(), anchors.detach(), bbox_head,
                        features[0].detach(), images.shape[-2:], count)

            # Only encoder candidates are needed for these auxiliary objectives.
            # The native input builder provides identical anchors and candidate content.
            return decoder._get_decoder_input(memory, shapes, candidate_selector=select)[-1]


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if not name.startswith('query_selector.'):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def oracle_selection(inputs, aux, targets, count):
    """GT-assisted diagnostic using the configured supervision.

    Legacy mode prioritizes representative positives. Quality mode substitutes
    clipped GT quality logits into the native score fusion. Neither policy is
    deployable or a guaranteed upper bound on AP.
    """
    memory, logits, anchors, bbox_head = inputs[:4]
    pool, scores = candidate_pool(logits, count, aux['boxes'].shape[1])
    chosen, priorities = [], []
    for i, target in enumerate(targets):
        if 'quality_logits' in aux:
            quality, _, _ = quality_targets(aux['boxes'][i], aux['candidate_logits'][i], target)
            # GT quality through the exact deployment score fusion; diagnostic only.
            priority = scores[i] + .5 * torch.logit(quality.clamp(.01, .99))
        else:
            positive = association(aux['boxes'][i].float(), target['boxes'].float())[2]
            priority = scores[i] + positive * (scores[i].max() - scores[i].min() + 1)
        chosen.append(priority.topk(count).indices)
        priorities.append(priority)
    chosen = torch.stack(chosen)
    indices = pool.gather(1, chosen)
    selected_memory = gather(memory, indices)
    adjusted_aux = {**aux, 'selected': chosen}
    if 'ranking_scores' in aux:
        adjusted_aux['ranking_scores'] = torch.stack(priorities)
    return (selected_memory, gather(logits, indices),
            bbox_head(selected_memory) + gather(anchors, indices), adjusted_aux)


def coverage_record(aux, logits, gt, labels, small, count, image_id):
    pool, _ = candidate_pool(logits, count, aux['boxes'].shape[1])
    classes = logits.argmax(-1).gather(1, pool)
    records = []
    for i, boxes in enumerate(aux['boxes']):
        owner, ambiguous, representative, _ = association(boxes.float(), gt[i])
        iou = box_iou(box_cxcywh_to_xyxy(boxes.float()), box_cxcywh_to_xyxy(gt[i]))[0]
        class_iou = iou * (classes[i, :, None] == labels[i][None])
        chosen = aux['selected'][i]
        record = dict(image_id=image_id[i], gt=len(gt[i]), small_gt=int(small[i].sum()),
                      replaced=int((chosen >= count).sum()))
        variants = dict(native=torch.arange(count, device=boxes.device), selected=chosen,
                        pool=torch.arange(len(boxes), device=boxes.device))
        for name, idx in variants.items():
            record[name + '_representatives'] = int(representative[idx].sum())
            record[name + '_owners'] = int(owner[idx][owner[idx] >= 0].unique().numel())
            for threshold in (.5, .75):
                key = str(threshold)
                hits = class_iou[idx].max(0).values >= threshold if len(gt[i]) else torch.zeros(0, dtype=torch.bool, device=boxes.device)
                record[name + '_class_' + key] = int(hits.sum())
                record[name + '_small_class_' + key] = int((hits & small[i]).sum())
        if 'quality_logits' in aux:
            quality, quality_owner, ambiguous = quality_targets(
                boxes, aux['candidate_logits'][i], {'boxes': gt[i], 'labels': labels[i]})
            better, worse = quality_pairs(quality, quality_owner, ~ambiguous)
            record['rank_pairs'] = len(better)
            scores = aux['ranking_scores'][i]
            record['rank_correct'] = int((scores[better] > scores[worse]).sum())
        records.append(record)
    return records


def gather_records(records):
    if dist_utils.is_dist_available_and_initialized():
        gathered = [None] * dist_utils.get_world_size()
        torch.distributed.all_gather_object(gathered, records)
        records = [record for chunk in gathered for record in chunk]
    # DistributedSampler pads validation: count each real image exactly once.
    return list({record['image_id']: record for record in records}.values())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


@torch.no_grad()
def validate(probe, cfg, loader, output, name, oracle=False):
    probe.eval()
    selector = probe.detector.query_selector
    context = {}
    count = probe.detector.decoder.num_queries

    def hook(module, inputs, result):
        if oracle:
            result = oracle_selection(inputs, result[-1], context['targets'], count)
        context['aux'] = result[-1]
        context['logits'] = inputs[1]
        return result

    handle = selector.register_forward_hook(hook)
    evaluator = cfg.evaluator
    evaluator.cleanup()
    post = cfg.postprocessor.cuda()
    records = []
    try:
        for images, targets in loader:
            images = images.cuda()
            h, w = images.shape[-2:]
            norm = images.new_tensor([w, h, w, h])
            gt = [box_xyxy_to_cxcywh(t['boxes'].as_subclass(torch.Tensor).cuda() / norm) for t in targets]
            labels = [t['labels'].cuda() for t in targets]
            context['targets'] = [{'boxes': box, 'labels': label} for box, label in zip(gt, labels)]
            outputs = probe(images)
            predictions = post(outputs, torch.stack([t['orig_size'] for t in targets]).cuda())
            evaluator.update({int(t['image_id']): result for t, result in zip(targets, predictions)})
            batch_records = coverage_record(
                context['aux'], context['logits'], gt, labels,
                [t['area'].cuda() < 32**2 for t in targets], count,
                [int(t['image_id']) for t in targets])
            # Per-image losses survive validation padding deduplication too.
            for i, record in enumerate(batch_records):
                aux = {k: v[i:i+1] for k, v in context['aux'].items()}
                record.update({k: float(v) for k, v in selection_losses(aux, context['targets'][i:i+1]).items()})
            records.extend(batch_records)
    finally:
        handle.remove()
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    metrics = evaluator.coco_eval['bbox'].stats.tolist()
    records = gather_records(records)
    assert len(records) == 548
    total_gt = sum(r['gt'] for r in records)
    small_gt = sum(r['small_gt'] for r in records)
    coverage = {}
    for variant in ('native', 'selected', 'pool'):
        coverage[variant] = {metric: sum(r[variant + '_' + metric] for r in records) /
                            (small_gt if metric.startswith('small') else total_gt)
                            for metric in ('representatives', 'owners', 'class_0.5', 'class_0.75',
                                           'small_class_0.5', 'small_class_0.75')}
    losses = {k: sum(r[k] for r in records) / len(records) for k in records[0] if k.startswith('loss_query')}
    result = dict(name=name, oracle_diagnostic=oracle, metrics=metrics,
                  replaced_per_image=sum(r['replaced'] for r in records) / len(records),
                  coverage=coverage, validation_losses=losses)
    if 'rank_pairs' in records[0]:
        result['rank_pair_accuracy'] = sum(r['rank_correct'] for r in records) / max(1, sum(r['rank_pairs'] for r in records))
    if dist_utils.is_main_process():
        write_json(output / (name + '.json'), {**result, 'per_image': records})
        print('VALIDATION ' + json.dumps(result), flush=True)
    return result


def build_probe(config, checkpoint):
    cfg = YAMLConfig(config)
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        detector = cfg.model
    state = torch.load(checkpoint, map_location='cpu')['ema']['module']
    incompatible = detector.load_state_dict(state, strict=False)
    assert not incompatible.unexpected_keys, incompatible
    assert set(incompatible.missing_keys) == {k for k in detector.state_dict() if k.startswith('query_selector.')}, incompatible
    probe = FrozenSelectionProbe(detector).cuda()
    return cfg, probe


def preflight(args):
    dist_utils.setup_seed(0)
    cfg, probe = build_probe(args.config, args.checkpoint)
    cfg.yaml_cfg['train_dataloader']['total_batch_size'] = 4
    cfg.yaml_cfg['train_dataloader']['num_workers'] = 0
    images, targets = next(iter(cfg.train_dataloader))
    images = images.cuda()
    targets = [{k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
    before = frozen_digest(probe.detector)
    probe.eval()
    with torch.no_grad():
        actual = probe(images[:1])
        selector = probe.detector.query_selector
        probe.detector.query_selector = None
        native = probe(images[:1])
        probe.detector.query_selector = selector
        differences = {k: float((actual[k] - native[k]).abs().max()) for k in ('pred_boxes', 'pred_logits')}
        for k in differences:
            torch.testing.assert_close(actual[k], native[k], atol=2e-5, rtol=1e-5)
    probe.train()
    optimizer = torch.optim.AdamW(selector.parameters(), lr=cfg.yaml_cfg['probe_lr'], weight_decay=1.25e-4)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024)
    records = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            aux = probe(images)
        aux['existence'].retain_grad()
        with torch.autocast('cuda', enabled=False):
            losses = selection_losses(aux, targets)
            loss = sum(losses.values())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert all(p.grad is None for n, p in probe.detector.named_parameters() if not n.startswith('query_selector.'))
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in selector.parameters())
        if step:
            assert selector.projection.weight.grad.abs().sum() > 0
        mask = torch.ones_like(aux['existence'], dtype=torch.bool).scatter(1, aux['selected'], False)
        assert (aux['existence'].grad[mask].abs() > 0).all()
        grad = torch.nn.utils.clip_grad_norm_(selector.parameters(), .1)
        scale = scaler.get_scale()
        scaler.step(optimizer); scaler.update()
        assert scaler.get_scale() >= scale
        records.append(dict(step=step, loss=float(loss), grad_norm=float(grad)))
    assert frozen_digest(probe.detector) == before
    write_json(Path(args.output), dict(status='passed', frozen_sha256=before,
               zero_init_max_abs=differences, updates=records,
               peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.preflight:
        preflight(args)
        return
    dist_utils.setup_distributed(seed=0)
    assert dist_utils.get_world_size() == 6, 'This experiment uses exactly six GPUs'
    cfg, probe = build_probe(args.config, args.checkpoint)
    output = Path(args.output)
    if dist_utils.is_main_process():
        output.mkdir(parents=True, exist_ok=False)
    torch.distributed.barrier()
    train_loader = dist_utils.warp_loader(cfg.train_dataloader, shuffle=True)
    val_loader = dist_utils.warp_loader(cfg.val_dataloader, shuffle=False)
    assert len(train_loader.dataset) == 6471 and len(val_loader.dataset) == 548
    model = dist_utils.warp_model(probe, find_unused_parameters=False)
    selector = probe.detector.query_selector
    optimizer = torch.optim.AdamW(selector.parameters(), lr=cfg.yaml_cfg['probe_lr'], weight_decay=1.25e-4)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024)
    digest = frozen_digest(probe.detector)
    if dist_utils.is_main_process():
        import yaml
        effective = {k: v for k, v in cfg.yaml_cfg.items() if k != '__include__'}
        (output / 'effective_config.yml').write_text(yaml.safe_dump(effective, sort_keys=False))
        write_json(output / 'status.json', dict(status='initial_validation', frozen_sha256=digest))
    baseline = validate(probe, cfg, val_loader, output, 'initial')
    expected = cfg.yaml_cfg['expected_baseline_ap']
    assert abs(baseline['metrics'][0] - expected) < 1e-6, (baseline['metrics'][0], expected)
    oracle_name = 'gt_quality_diagnostic' if selector.supervision == 'quality_rank' else 'gt_representative_diagnostic'
    oracle = validate(probe, cfg, val_loader, output, oracle_name, oracle=True)
    best = baseline['metrics'][0]
    best_trained = float('-inf')
    if dist_utils.is_main_process():
        torch.save(dict(selector=selector.state_dict(), checkpoint=args.checkpoint, epoch=-1,
                        metrics=baseline, frozen_sha256=digest), output / 'best_selector.pth')
    start = time.time()
    for epoch in range(cfg.yaml_cfg['probe_epochs']):
        model.train()
        train_loader.set_epoch(epoch)
        train_loader.sampler.set_epoch(epoch)
        totals, updates, skipped = {}, 0, 0
        if dist_utils.is_main_process():
            write_json(output / 'status.json', dict(status='training', epoch=epoch, completed_epochs=epoch,
                       best_ap=best, baseline_ap=baseline['metrics'][0]))
        for step, (images, targets) in enumerate(train_loader):
            images = images.cuda()
            targets = [{k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                aux = model(images)
            with torch.autocast('cuda', enabled=False):
                losses = selection_losses(aux, targets)
                loss = sum(losses.values())
            assert torch.isfinite(loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(selector.parameters(), .1)
            scale = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            stepped = scaler.get_scale() >= scale
            updates += int(stepped); skipped += int(not stepped)
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.) + float(value)
            if step % 50 == 0 and dist_utils.is_main_process():
                print(json.dumps(dict(epoch=epoch, step=step, steps=len(train_loader),
                                      loss=float(loss), grad_norm=float(norm), elapsed=time.time()-start)), flush=True)
        metrics = validate(probe, cfg, val_loader, output, f'epoch_{epoch+1:02}')
        # Check parameters AND buffers, not just requires_grad flags.
        assert frozen_digest(probe.detector) == digest, 'Frozen detector state changed'
        reduced = dist_utils.reduce_dict({k: torch.tensor(v / len(train_loader), device='cuda') for k, v in totals.items()})
        counts = torch.tensor([updates, skipped], device='cuda', dtype=torch.long)
        minimum, maximum = counts.clone(), counts.clone()
        torch.distributed.all_reduce(minimum, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
        assert torch.equal(minimum, maximum), 'Ranks disagree on successful updates'
        record = dict(epoch=epoch, completed_epochs=epoch+1, train_losses={k: float(v) for k, v in reduced.items()},
                      validation=metrics, optimizer_updates=updates, skipped_updates=skipped,
                      frozen_detector_unchanged=True)
        improved = metrics['metrics'][0] > best
        best = max(best, metrics['metrics'][0])
        improved_trained = metrics['metrics'][0] > best_trained
        best_trained = max(best_trained, metrics['metrics'][0])
        if dist_utils.is_main_process():
            state = dict(selector=selector.state_dict(), optimizer=optimizer.state_dict(),
                         scaler=scaler.state_dict(), checkpoint=args.checkpoint, epoch=epoch,
                         metrics=metrics, frozen_sha256=digest)
            torch.save(state, output / 'last_selector.pth')
            if improved:
                torch.save(state, output / 'best_selector.pth')
            if improved_trained:
                torch.save(state, output / 'best_trained_selector.pth')
            with (output / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            write_json(output / 'status.json', dict(status='training' if epoch+1 < cfg.yaml_cfg['probe_epochs'] else 'completed',
                       completed_epochs=epoch+1, best_ap=best, baseline_ap=baseline['metrics'][0],
                       last_ap=metrics['metrics'][0], best_trained_ap=best_trained, frozen_detector_unchanged=True))
    dist_utils.cleanup()


if __name__ == '__main__':
    main()
