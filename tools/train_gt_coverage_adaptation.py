#!/usr/bin/env python3
"""One 12-epoch decoder adaptation run with GT selection in train AND val."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import time
from types import MethodType

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core import YAMLConfig
from src.misc import dist_utils
from src.zoo.dfine.box_ops import box_xyxy_to_cxcywh
from tools.gt_query_selection import gt_candidate_selection
from tools.probe_selection_target import gather_records, write_json
from tools.train_boundary_selection import Float32Criterion
from tools.train_sparse_strip_query import guard_raw_losses, require_finite


FROZEN_DECODER = ('input_proj', 'enc_output', 'enc_score_head', 'enc_bbox_head')


def fp32_encoder_input(decoder, feats):
    with torch.autocast(feats[0].device.type, enabled=False):
        return type(decoder)._get_encoder_input(decoder, [f.float() for f in feats])


def fp32_decoder_input(decoder, memory, *args, **kwargs):
    # Keep original DN concatenation differentiable; only precision is changed.
    with torch.autocast(memory.device.type, enabled=False):
        return type(decoder)._get_decoder_input(decoder, memory.float(), *args, **kwargs)


class GTCoverageAdaptation(nn.Module):
    requires_gt = True
    oracle_evaluation = True

    def __init__(self, detector, candidates=1200, budget=40):
        super().__init__()
        assert detector.query_selector is None
        assert detector.decoder.query_select_method == 'default'
        assert not detector.decoder.learn_query_content
        self.detector = detector
        self.candidates, self.budget = candidates, budget
        self.validation_cache = {}
        self.last_records = []
        for name, p in detector.named_parameters():
            keep = name.startswith('decoder.') and not name.startswith(tuple('decoder.'+n+'.' for n in FROZEN_DECODER))
            p.requires_grad_(p.requires_grad and keep)
        self.trainable_names = tuple(n for n, p in detector.named_parameters() if p.requires_grad)
        decoder = detector.decoder
        decoder._get_encoder_input = MethodType(fp32_encoder_input, decoder)
        decoder._get_decoder_input = MethodType(fp32_decoder_input, decoder)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.detector.eval()
        self.detector.decoder.train(mode)
        for name in FROZEN_DECODER:
            getattr(self.detector.decoder, name).eval()
        # LayerStripHeads uses its own training flag solely for checkpointing.
        # Children stay in eval and every original strip parameter is frozen.
        self.detector.refiner.training = mode
        return self

    def forward(self, images, targets=None):
        if targets is None:
            raise ValueError('This GT diagnostic requires normalized GT in training AND validation')
        with torch.no_grad(), torch.autocast(images.device.type, enabled=False):
            features = self.detector.encoder(self.detector.backbone(images.float()))

        def select(memory, logits, anchors, bbox_head, count):
            result, chosen, records = gt_candidate_selection(
                memory, logits, anchors, bbox_head, count, targets, self.candidates, self.budget,
                cached=None if self.training else self.validation_cache)
            self.last_records = records
            return result

        def refine(layer_index, queries, boxes, distribution):
            return self.detector.refiner(layer_index, features[self.detector.feature_level],
                queries, boxes, images.shape[-2:], distribution, self.detector.checkpoint_strips)

        return self.detector.decoder(features, targets, candidate_selector=select, distribution_refiner=refine)


def fixed_digest(model):
    trainable = set(model.trainable_names)
    digest = hashlib.sha256()
    for name, tensor in model.detector.state_dict().items():
        if name not in trainable:
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def optimizer_groups(model, lr, decay):
    norm_ids = {id(p) for m in model.modules() if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.modules.batchnorm._BatchNorm))
                for p in m.parameters(recurse=False)}
    groups = {True: [], False: []}
    for p in model.parameters():
        if p.requires_grad:
            groups[id(p) in norm_ids].append(p)
    return [dict(params=ps, lr=lr, weight_decay=0. if norm else decay) for norm, ps in groups.items() if ps]


class DecoderEMA:
    """Average only trainable parameters; never round or modify frozen tensors."""
    def __init__(self, model, decay=.999):
        self.decay, self.updates = decay, 0
        self.values = {n: p.detach().clone() for n, p in model.detector.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        params = dict(model.detector.named_parameters())
        for name, value in self.values.items():
            value.mul_(self.decay).add_(params[name].detach(), alpha=1-self.decay)
        self.updates += 1

    @contextmanager
    def apply(self, model):
        params = dict(model.detector.named_parameters())
        backup = {n: params[n].detach().clone() for n in self.values}
        with torch.no_grad():
            for n, value in self.values.items():
                params[n].copy_(value)
        try:
            yield
        finally:
            with torch.no_grad():
                for n, value in backup.items():
                    params[n].copy_(value)


def build(args):
    cfg = YAMLConfig(args.config)
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        detector = cfg.model
    state = torch.load(args.checkpoint, map_location='cpu')['ema']['module']
    detector.load_state_dict(state, strict=True)
    model = GTCoverageAdaptation(detector, cfg.yaml_cfg['selection_candidates'], cfg.yaml_cfg['selection_budget']).cuda()
    raw_criterion = cfg.criterion.cuda()
    guard_raw_losses(raw_criterion)
    criterion = Float32Criterion(raw_criterion)
    return cfg, model, criterion


@torch.no_grad()
def validate(model, cfg, loader, output, name):
    model.eval()
    evaluator = cfg.evaluator
    evaluator.cleanup()
    post = cfg.postprocessor.cuda()
    records = []
    for images, targets in loader:
        images = images.cuda()
        h, w = images.shape[-2:]
        norm = images.new_tensor([w, h, w, h])
        normalized = [dict(boxes=box_xyxy_to_cxcywh(t['boxes'].as_subclass(torch.Tensor).cuda()/norm),
                           labels=t['labels'].cuda(), image_id=t['image_id']) for t in targets]
        outputs = model(images, normalized)
        require_finite(outputs)
        predictions = post(outputs, torch.stack([t['orig_size'] for t in targets]).cuda())
        evaluator.update({int(t['image_id']): p for t, p in zip(targets, predictions)})
        records.extend(model.last_records)
    evaluator.synchronize_between_processes()
    evaluator.accumulate(); evaluator.summarize()
    records = gather_records(records)
    assert len(records) == 548
    gt = sum(r['gt'] for r in records)
    result = dict(name=name, metrics=evaluator.coco_eval['bbox'].stats.tolist(), oracle_evaluation=True,
        images=len(records), gt=gt, replaced_per_image=sum(r['replaced'] for r in records)/len(records),
        coverage={k: sum(r[k+'_covered'] for r in records)/gt for k in ('native', 'selected', 'pool')})
    if dist_utils.is_main_process():
        write_json(output/(name+'.json'), result)
        print('VALIDATION '+json.dumps(result), flush=True)
    return result


def train_step(model, raw, criterion, optimizer, scaler, images, targets):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda'):
        outputs = model(images, targets)
    with torch.autocast('cuda', enabled=False):
        losses = criterion(outputs, targets)
        require_finite(losses, 'loss')
        assert not any(k.startswith('loss_query') for k in losses)
        assert all(float(v.detach()) == 0 for k, v in losses.items() if k.startswith('loss_ddf'))
        loss = sum(losses.values())
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_([p for p in raw.parameters() if p.requires_grad], .1)
    scale = scaler.get_scale()
    scaler.step(optimizer); scaler.update()
    return losses, float(norm), scaler.get_scale() >= scale


def preflight(args):
    dist_utils.setup_seed(0)
    cfg, model, criterion = build(args)
    cfg.yaml_cfg['train_dataloader']['total_batch_size'] = 4
    cfg.yaml_cfg['train_dataloader']['num_workers'] = 0
    images, targets = next(iter(cfg.train_dataloader))
    images = images.cuda()
    targets = [{k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
    before = fixed_digest(model)
    first = next(p for p in model.parameters() if p.requires_grad).detach().clone()
    optimizer = torch.optim.AdamW(optimizer_groups(model, cfg.yaml_cfg['adapt_lr'], cfg.yaml_cfg['adapt_weight_decay']))
    scaler = torch.cuda.amp.GradScaler(init_scale=1024)
    records = []
    for step in range(3):
        model.train()
        losses, norm, stepped = train_step(model, model, criterion, optimizer, scaler, images, targets)
        assert stepped
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.detector.decoder.decoder.parameters())
        assert fixed_digest(model) == before
        records.append(dict(step=step, loss=float(sum(losses.values())), grad_norm=norm, replaced=[r['replaced'] for r in model.last_records]))
    assert not torch.equal(first, next(p for p in model.parameters() if p.requires_grad))
    ema = DecoderEMA(model)
    ema.update(model)
    with ema.apply(model):
        assert fixed_digest(model) == before
    write_json(Path(args.output), dict(status='passed', frozen_sha256=before, steps=records,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        trainable_names=model.trainable_names, peak_memory_mib=torch.cuda.max_memory_allocated()/2**20))


def run(args):
    dist_utils.setup_distributed(seed=0)
    assert dist_utils.get_world_size() == 6
    cfg, raw, criterion = build(args)
    assert cfg.yaml_cfg['epochs'] == 12
    output = Path(args.output)
    if dist_utils.is_main_process():
        output.mkdir(parents=True, exist_ok=False)
    torch.distributed.barrier()
    train_loader = dist_utils.warp_loader(cfg.train_dataloader, shuffle=True)
    val_loader = dist_utils.warp_loader(cfg.val_dataloader, shuffle=False)
    assert len(train_loader.dataset) == 6471 and len(val_loader.dataset) == 548
    digest = fixed_digest(raw)
    optimizer = torch.optim.AdamW(optimizer_groups(raw, cfg.yaml_cfg['adapt_lr'], cfg.yaml_cfg['adapt_weight_decay']))
    scaler = torch.cuda.amp.GradScaler(init_scale=1024)
    ema = DecoderEMA(raw, cfg.yaml_cfg['adapt_ema_decay'])
    model = dist_utils.warp_model(raw, find_unused_parameters=True)
    start = time.time()
    if dist_utils.is_main_process():
        import yaml
        (output/'effective_config.yml').write_text(yaml.safe_dump(cfg.yaml_cfg, sort_keys=False))
        write_json(output/'protocol.json', dict(checkpoint=args.checkpoint, checkpoint_weights='ema.module',
            train_and_val_selection='GT coverage first, Top1200, max40 exchanges, preserve300',
            trainable_names=raw.trainable_names, frozen_sha256=digest,
            trainable_parameters=sum(p.numel() for p in raw.parameters() if p.requires_grad),
            fixed_candidates_fp32=True, original_G5_strip_heads_frozen=True,
            epochs=12, total_batch_size=24, lr=cfg.yaml_cfg['adapt_lr'], ema_decay=ema.decay,
            augmentation='fixed 640x640 + random horizontal flip', gt_diagnostic_not_deployable=True))
        write_json(output/'status.json', dict(status='initial_gt_validation', completed_epochs=0))
    initial = validate(raw, cfg, val_loader, output, 'initial_gt')
    assert abs(initial['metrics'][0]-cfg.yaml_cfg['expected_gt_initial_ap']) < 1e-6, initial
    best = initial['metrics'][0]
    best_trained = None
    total_updates = total_skipped = 0
    if dist_utils.is_main_process():
        torch.save(dict(model=raw.detector.state_dict(), metrics=initial, epoch=-1, oracle_evaluation=True), output/'best.pth')
    for epoch in range(12):
        train_loader.set_epoch(epoch)
        train_loader.sampler.set_epoch(epoch)
        model.train(); criterion.train()
        losses_sum = {}; updates = skipped = 0
        for step, (images, targets) in enumerate(train_loader):
            images = images.cuda()
            targets = [{k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            losses, norm, stepped = train_step(model, raw, criterion, optimizer, scaler, images, targets)
            if stepped:
                ema.update(raw)
            updates += int(stepped); skipped += int(not stepped)
            for k, v in losses.items():
                losses_sum[k] = losses_sum.get(k, 0.) + float(v)
            if step % 20 == 0 and dist_utils.is_main_process():
                progress = dict(status='training', epoch=epoch+1, completed_epochs=epoch, step=step+1,
                    steps_per_epoch=len(train_loader), optimizer_updates=total_updates+updates,
                    skipped_updates=total_skipped+skipped, best_ap=best,
                    loss=float(sum(losses.values())), grad_norm=norm, elapsed_seconds=time.time()-start,
                    replaced_per_image=float(np.mean([r['replaced'] for r in raw.last_records])))
                print(json.dumps(progress), flush=True)
                write_json(output/'status.json', progress)
        counts = torch.tensor([updates, skipped], device='cuda')
        low, high = counts.clone(), counts.clone()
        torch.distributed.all_reduce(low, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(high, op=torch.distributed.ReduceOp.MAX)
        assert torch.equal(low, high)
        total_updates += updates; total_skipped += skipped
        assert fixed_digest(raw) == digest
        with ema.apply(raw):
            assert fixed_digest(raw) == digest
            metrics = validate(raw, cfg, val_loader, output, f'epoch_{epoch+1:02}_gt')
        assert metrics['coverage'] == initial['coverage']
        averages = dist_utils.reduce_dict({k: torch.tensor(v/len(train_loader), device='cuda') for k, v in losses_sum.items()})
        ap = metrics['metrics'][0]
        improved, improved_trained = ap > best, best_trained is None or ap > best_trained
        best = max(best, ap); best_trained = ap if best_trained is None else max(best_trained, ap)
        record = dict(epoch=epoch+1, completed_epochs=epoch+1, validation=metrics,
            train_losses={k:float(v) for k,v in averages.items()}, updates=updates, skipped=skipped,
            total_updates=total_updates, total_skipped=total_skipped, frozen_state_unchanged=True,
            best_ap=best, best_trained_ap=best_trained, elapsed_seconds=time.time()-start)
        if dist_utils.is_main_process():
            model_state = raw.detector.state_dict()
            ema_state = {k: ema.values[k] if k in ema.values else v for k,v in model_state.items()}
            state = dict(model=model_state, ema=dict(module=ema_state, updates=ema.updates),
                optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), epoch=epoch,
                completed_epochs=epoch+1, metrics=metrics, initial_metrics=initial,
                checkpoint=args.checkpoint, frozen_sha256=digest, oracle_evaluation=True,
                trainable_names=raw.trainable_names)
            torch.save(state, output/'last.pth')
            if improved:
                torch.save(state, output/'best.pth')
            if improved_trained:
                torch.save(state, output/'best_trained.pth')
            with (output/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            write_json(output/'status.json', dict(status='completed' if epoch==11 else 'training', **record))
            print('EPOCH '+json.dumps(record), flush=True)
    dist_utils.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/dfine/boundary_selection/gt_coverage_adapt12.yml')
    parser.add_argument('--checkpoint', default='output/visdrone_g5_no_distill_6gpu_20261019_232603/results/best_stg2.pth')
    parser.add_argument('--output', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    preflight(args) if args.preflight else run(args)
