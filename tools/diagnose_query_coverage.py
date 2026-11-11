#!/usr/bin/env python3
"""GT-assisted, frozen-G5 set-coverage diagnostic; never a deployable selector."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.probe_selection_target import (
    build_probe, candidate_pool, frozen_digest, gather_records, write_json)
from src.misc import dist_utils
from tools.gt_query_selection import exchange_plans
from src.zoo.dfine.boundary_query_selection import gather, association
from src.zoo.dfine.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, box_iou


def cardinality_hits(quality, threshold):
    """Maximum one-to-one query/GT cardinality, with IoU only a tie breaker."""
    m, g = quality.shape
    hits = np.zeros(g, dtype=bool)
    if m and g:
        valid = quality >= threshold
        rows, cols = linear_sum_assignment(-(valid.astype(float) + quality / (min(m, g) + 1)))
        hits[cols[valid[rows, cols]]] = True
    return hits


def set_record(q, geometric, indices):
    part = q[indices]
    maximum = part.max(0) if len(part) else np.zeros(q.shape[1])
    result = {'quality_sum': float(maximum.sum())}
    for threshold, label in ((.5, '50'), (.75, '75')):
        result['class_' + label] = np.flatnonzero(maximum >= threshold).tolist()
        result['one_to_one_' + label] = np.flatnonzero(cardinality_hits(part, threshold)).tolist()
        result['geometric_' + label] = np.flatnonzero(geometric[indices].max(0) >= threshold).tolist()
        eligible = int((part.max(1) >= threshold).sum()) if q.shape[1] else 0
        result['eligible_queries_' + label] = eligible
        result['surplus_queries_' + label] = eligible - len(result['one_to_one_' + label])
    return result


def annotation_ids(coco, image_id, target):
    w, h = target['orig_size'].tolist()
    anns = [a for a in coco.imgToAnns[image_id] if not a.get('iscrowd', 0)]
    keep, boxes = [], []
    for a in anns:
        x, y, bw, bh = a['bbox']
        box = np.clip([x, y, x + bw, y + bh], [0, 0, 0, 0], [w, h, w, h])
        if box[2] > box[0] and box[3] > box[1]:
            keep.append(a); boxes.append(box)
    assert [a['category_id'] for a in keep] == target['labels'].tolist()
    transformed = target['boxes'].as_subclass(torch.Tensor).numpy() / np.array([640, 640, 640, 640])
    np.testing.assert_allclose(np.asarray(boxes).reshape(-1, 4) / [w, h, w, h], transformed, atol=2e-6)
    return [a['id'] for a in keep]


def detection_hits(coco, image_id, prediction, gt_ids, max_dets):
    """Official COCO Python per-image matching, including ignored/crowd GT.

    AP is computed independently by the existing faster-coco evaluator. Here
    we retain nonignored matched GT identities at IoU .50/.75 and maxDets=100.
    """
    matcher = COCOeval(iouType='bbox')
    matcher.params.iouThrs = np.array([.5, .75])
    matcher.params.maxDets = [max_dets]
    matcher.params.useCats = 1
    anns = copy.deepcopy(coco.imgToAnns[image_id])
    for a in anns:
        a['ignore'] = a.get('iscrowd', 0)  # official bbox COCO _prepare behavior
        matcher._gts[image_id, a['category_id']].append(a)
    boxes = prediction['boxes'].detach().cpu().numpy().copy()
    boxes[:, 2:] -= boxes[:, :2]
    for i, (box, label, score) in enumerate(zip(boxes, prediction['labels'].tolist(), prediction['scores'].tolist())):
        matcher._dts[image_id, label].append(dict(id=i+1, image_id=image_id,
            category_id=label, bbox=box.tolist(), score=score, area=float(box[2]*box[3])))
    matched = [set(), set()]
    for category in coco.getCatIds():
        matcher.ious[image_id, category] = matcher.computeIoU(image_id, category)
        result = matcher.evaluateImg(image_id, category, [0, 1e10], max_dets)
        if result is not None:
            for i in range(2):
                matched[i].update(g for g, hit, ignore in zip(result['gtIds'], result['gtMatches'][i], result['gtIgnore']) if hit and not ignore)
    return {label: [i for i, g in enumerate(gt_ids) if g in matched[k]] for k, label in enumerate(('50', '75'))}


def summarize(records, metrics, name, max_dets):
    gt = sum(r['gt'] for r in records)
    small = sum(len(r['small']) for r in records)
    dense = sum(r['gt'] for r in records if r['gt'] >= 100)
    result = dict(name=name, metrics=metrics, images=len(records), gt=gt,
                  small_gt=small, dense_image_gt=dense, coco_max_dets=max_dets,
                  replaced_per_image=np.mean([r['replaced'] for r in records]).item())
    for variant in ('native', 'selected', 'pool'):
        section = {}
        for key in ('class_50', 'class_75', 'one_to_one_50', 'one_to_one_75', 'geometric_50', 'geometric_75'):
            section[key] = sum(len(r[variant][key]) for r in records) / gt
            section['small_' + key] = sum(len(set(r[variant][key]) & set(r['small'])) for r in records) / small
            section['dense_' + key] = sum(len(r[variant][key]) for r in records if r['gt'] >= 100) / max(1, dense)
        for key in ('eligible_queries_50', 'surplus_queries_50', 'quality_sum', 'owners'):
            section[key + '_per_image'] = float(np.mean([r[variant][key] for r in records]))
        result[variant] = section
    transitions = {}
    for label in ('50', '75'):
        counts = {key: 0 for key in ('coverage_added', 'coverage_lost', 'coverage_added_small',
            'added_already_detected', 'added_now_detected', 'added_newly_detected',
            'added_still_missed', 'added_newly_detected_small', 'detected_gained', 'detected_lost',
            'detected_gained_small', 'detected_lost_small', 'native_detected', 'selected_detected')}
        for r in records:
            native = set(r['native']['class_' + label]); chosen = set(r['selected']['class_' + label])
            before = set(r['baseline_detected'][label]); after = set(r['detected'][label])
            added, lost = chosen - native, native - chosen
            small_ids = set(r['small'])
            values = dict(coverage_added=added, coverage_lost=lost, coverage_added_small=added & small_ids,
                added_already_detected=added & before, added_now_detected=added & after,
                added_newly_detected=(added & after) - before, added_still_missed=added - after,
                added_newly_detected_small=((added & after) - before) & small_ids,
                detected_gained=after-before, detected_lost=before-after,
                detected_gained_small=(after-before) & small_ids, detected_lost_small=(before-after) & small_ids,
                native_detected=before, selected_detected=after)
            for key, value in values.items():
                counts[key] += len(value)
        transitions[label] = counts
    result['transitions'] = transitions
    return result


@torch.no_grad()
def run(args):
    torch.set_num_threads(4)
    dist_utils.setup_distributed(seed=0)
    assert dist_utils.get_world_size() == 6
    cfg, probe = build_probe(args.config, args.checkpoint)
    probe.eval().requires_grad_(False)
    digest = frozen_digest(probe.detector)
    loader = dist_utils.warp_loader(cfg.val_dataloader, shuffle=False)
    assert len(loader.dataset) == 548
    coco = loader.dataset.coco
    output = Path(args.output)
    if dist_utils.is_main_process():
        output.mkdir(parents=True, exist_ok=False)
    torch.distributed.barrier()
    count = probe.detector.decoder.num_queries
    assert count == 300
    cache, context = {}, {}
    budgets = (10, 20, 40)
    variants = ['native'] + [f'{policy}_{budget}' for policy in ('quality', 'coverage') for budget in budgets]
    post, evaluator = cfg.postprocessor.cuda(), cfg.evaluator
    max_dets = evaluator.coco_eval['bbox'].params.maxDets[-1]

    def hook(module, inputs, result):
        memory, logits, anchors, bbox_head = inputs[:4]
        aux = result[-1]
        pool, _ = candidate_pool(logits, count, aux['boxes'].shape[1])
        classes = gather(logits, pool).argmax(-1)
        chosen, batch_records = [], []
        for i, (target, gt) in enumerate(zip(context['targets'], context['gt'])):
            image_id = int(target['image_id'])
            if image_id not in cache:
                boxes = aux['boxes'][i].float()
                geo = box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(gt))[0].cpu().numpy()
                q = geo * (classes[i].cpu().numpy()[:, None] == target['labels'].numpy()[None])
                plans, traces = {}, {}
                for policy in ('quality', 'coverage'):
                    objective = q if policy == 'quality' else (q >= .5).astype(float) + .01 / max(1, len(gt)) * q
                    picks, trace = exchange_plans(objective, count, budgets)
                    plans.update({f'{policy}_{b}': pick for b, pick in picks.items()})
                    traces[policy] = trace
                plans['native'] = np.arange(count)
                owner = association(boxes, gt)[0].cpu().numpy()
                sets = {}
                for name, indices in {**plans, 'pool': np.arange(len(boxes))}.items():
                    sets[name] = set_record(q, geo, indices)
                    sets[name]['owners'] = len(np.unique(owner[indices][owner[indices] >= 0]))
                cache[image_id] = dict(plans=plans, sets=sets, traces=traces,
                    gt_ids=annotation_ids(coco, image_id, target),
                    small=np.flatnonzero(target['area'].numpy() < 32**2).tolist(), gt=len(gt))
            stored = cache[image_id]
            name = context['variant']
            pick = stored['plans'][name]
            chosen.append(torch.as_tensor(pick, device=memory.device))
            batch_records.append(dict(image_id=image_id, gt=stored['gt'], gt_ids=stored['gt_ids'],
                small=stored['small'], replaced=int((pick >= count).sum()),
                native=stored['sets']['native'], selected=stored['sets'][name], pool=stored['sets']['pool'],
                selected_pool_indices=pick.tolist()))
        chosen = torch.stack(chosen)
        indices = pool.gather(1, chosen)
        selected_memory = gather(memory, indices)
        context['records'] = batch_records
        return selected_memory, gather(logits, indices), bbox_head(selected_memory) + gather(anchors, indices), {**aux, 'selected': chosen}

    handle = probe.detector.query_selector.register_forward_hook(hook)
    results = []
    start = time.time()
    try:
        for variant in variants:
            context['variant'] = variant
            evaluator.cleanup()
            records = []
            if dist_utils.is_main_process():
                write_json(output / 'status.json', dict(status='evaluating', variant=variant,
                    completed=[r['name'] for r in results], elapsed_seconds=time.time()-start))
            for step, (images, targets) in enumerate(loader):
                images = images.cuda()
                h, w = images.shape[-2:]
                norm = images.new_tensor([w, h, w, h])
                context['targets'] = targets
                context['gt'] = [box_xyxy_to_cxcywh(t['boxes'].as_subclass(torch.Tensor).cuda()/norm) for t in targets]
                predictions = post(probe(images), torch.stack([t['orig_size'] for t in targets]).cuda())
                evaluator.update({int(t['image_id']): p for t, p in zip(targets, predictions)})
                for record, prediction in zip(context['records'], predictions):
                    record['detected'] = detection_hits(coco, record['image_id'], prediction, record['gt_ids'], max_dets)
                    stored = cache[record['image_id']]
                    if variant == 'native':
                        stored['baseline_detected'] = record['detected']
                    record['baseline_detected'] = stored['baseline_detected']
                    records.append(record)
                if step % 5 == 0 and dist_utils.is_main_process():
                    print(json.dumps(dict(variant=variant, batch=step, batches=len(loader), elapsed=time.time()-start)), flush=True)
            evaluator.synchronize_between_processes()
            evaluator.accumulate(); evaluator.summarize()
            metrics = evaluator.coco_eval['bbox'].stats.tolist()
            records = gather_records(records)
            assert len(records) == 548
            if variant == 'native':
                assert abs(metrics[0] - cfg.yaml_cfg['expected_baseline_ap']) < 1e-6
            assert frozen_digest(probe.detector) == digest
            summary = summarize(records, metrics, variant, max_dets)
            results.append(summary)
            if dist_utils.is_main_process():
                write_json(output / (variant + '.json'), {**summary, 'per_image': records})
                print('RESULT ' + json.dumps(summary), flush=True)
    finally:
        handle.remove()
    plans = gather_records([dict(image_id=k, traces=v['traces']) for k, v in cache.items()])
    if dist_utils.is_main_process():
        write_json(output / 'exchange_traces.json', plans)
        write_json(output / 'summary.json', dict(results=results, frozen_sha256=digest,
            frozen_detector_unchanged=True, checkpoint=args.checkpoint, gpu_ids=[0, 2, 3, 4, 6, 7],
            diagnostic_only=True, budgets=list(budgets), elapsed_seconds=time.time()-start,
            quality_objective='sum_g max_i class-correct coarse IoU',
            coverage_objective='sum_g max_i (1[class-correct IoU >= .5] + .01/num_gt * class-correct IoU)',
            limits='GT-assisted constrained greedy exchanges; not a deployable method or mathematical AP upper bound; one-to-one coverage is separately measured'))
        write_json(output / 'status.json', dict(status='completed', variants=variants,
            frozen_detector_unchanged=True, elapsed_seconds=time.time()-start))
    dist_utils.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/dfine/boundary_selection/frozen_target_probe.yml')
    parser.add_argument('--checkpoint', default='output/visdrone_g5_no_distill_6gpu_20261019_232603/results/best_stg2.pth')
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
