"""Measure each strip head's immediate effect on fixed query/GT pairs."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.analysis.all_layer_strip_trace import trace_all_layer_strip
from tools.analysis.evaluate_detector_geometry import load_ema_model
from tools.analysis.layer_strip_effect_metrics import LayerEffectMetrics


@torch.inference_mode()
def run(args):
    from src.core import YAMLConfig
    from src.zoo.dfine.gt_refinement import DFINEGTRefiner
    from src.zoo.dfine.matcher import HungarianMatcher
    from src.zoo.dfine.pred_refinement_criterion import PredRefinementCriterion

    output, cache = Path(args.output_file), Path(args.trace_cache)
    for path in (output, cache):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    cfg = YAMLConfig(args.config, HGNetv2={'pretrained': False},
                     val_dataloader={'total_batch_size': args.batch_size,
                                     'num_workers': args.workers})
    model, metadata = load_ema_model(cfg, args.checkpoint, args.device)
    if type(model).__name__ != 'DFINEAllLayerStrip':
        raise ValueError('This diagnostic expects the jointly trained all-layer strip model')
    loader = cfg.val_dataloader
    layer_count = model.decoder.eval_idx + 1
    costs = {'cost_class': 0, 'cost_bbox': 5, 'cost_giou': 2}
    matcher = PredRefinementCriterion(HungarianMatcher(costs, use_focal_loss=True)).to(args.device).eval()
    policies = ('final_geometry_fixed', 'layer_before_geometry', 'layer_after_geometry')
    meters = {policy: {i: LayerEffectMetrics() for i in range(layer_count)} for policy in policies}
    records, images_seen, gt_seen, fixed_pairs = [], 0, 0, 0
    input_sizes = set()
    started = time.monotonic()
    for batch_index, (images, original_targets) in enumerate(loader):
        images = images.to(args.device)
        traced = trace_all_layer_strip(model, images)
        outputs = traced['outputs']
        if not torch.isfinite(outputs['pred_logits']).all():
            raise ValueError('Non-finite final class logits')
        if batch_index == 0:
            normal = model(images)
            for key in ('pred_boxes', 'pred_logits'):
                torch.testing.assert_close(outputs[key], normal[key], atol=0, rtol=0)
            print('First batch: traced and ordinary inference are exactly equal.', flush=True)
        targets, ignored = DFINEGTRefiner._prepare_targets(
            original_targets, images.shape[-2:], images.device, 'xyxy_absolute',
            outputs['pred_logits'].shape[-1],
        )
        for original, target, count in zip(original_targets, targets, ignored):
            if count or len(original['area']) != len(target['boxes']):
                raise ValueError('Canonical GT filtering changed original area row alignment')
            target['area_original'] = original['area'].detach().cpu()
        # Classes and confidence never contribute to geometric assignment.
        neutral_logits = torch.zeros_like(outputs['pred_logits'])
        final_indices = matcher._match_baseline(outputs['pred_boxes'], neutral_logits, targets)
        batch_matches = {policy: {} for policy in policies}
        for index, layer in traced['layers'].items():
            matches = {
                'final_geometry_fixed': final_indices,
                'layer_before_geometry': matcher._match_baseline(layer['boxes_before'], neutral_logits, targets),
                'layer_after_geometry': matcher._match_baseline(layer['boxes_after'], neutral_logits, targets),
            }
            for policy, indices in matches.items():
                meters[policy][index].update(
                    layer['boxes_before'], layer['boxes_after'], targets, indices, images.shape[-2:],
                    valid=layer['valid'], half_widths_px=layer['half_widths_px'],
                )
                batch_matches[policy][index] = indices
        for b, (original, target) in enumerate(zip(original_targets, targets)):
            records.append({
                'image_id': int(original['image_id'].item()),
                'input_size': list(images.shape[-2:]),
                'target': {k: v.detach().cpu() for k, v in target.items()},
                'pred_logits': outputs['pred_logits'][b].detach().float().cpu(),
                'layers': {i: {k: v[b].detach().cpu() for k, v in layer.items()}
                           for i, layer in traced['layers'].items()},
                'matches': {policy: {i: tuple(v.detach().cpu() for v in indices[b])
                                     for i, indices in by_layer.items()}
                            for policy, by_layer in batch_matches.items()},
            })
        images_seen += len(images)
        gt_seen += sum(len(t['boxes']) for t in targets)
        fixed_pairs += sum(len(q) for q, _ in final_indices)
        input_sizes.add(tuple(images.shape[-2:]))
        if batch_index % 25 == 0 or images_seen == len(loader.dataset):
            print(f'Layers: {images_seen}/{len(loader.dataset)} images; '
                  f'{time.monotonic() - started:.1f}s', flush=True)
    if not images_seen:
        raise ValueError('No validation images processed')
    summary = {
        'model': {**metadata, 'config': str(Path(args.config).resolve())},
        'dataset': cfg.yaml_cfg['val_dataloader']['dataset'],
        'images': images_seen, 'valid_gt': gt_seen, 'fixed_pairs': fixed_pairs,
        'full_validation': images_seen == len(loader.dataset),
        'input_sizes_hw': sorted(input_sizes), 'inference_uses_gt': False,
        'inference_precision': 'float32', 'first_batch_trace_exact': True,
        'matching_costs': costs, 'clip_boxes': False, 'score_threshold': None,
        'matching_policies': {
            'final_geometry_fixed': 'Final output geometry assignment is reused unchanged across all six layers and both sides of each comparison.',
            'layer_before_geometry': 'Each layer uses its before-strip geometry assignment, held fixed for before/after.',
            'layer_after_geometry': 'Each layer uses its after-strip geometry assignment, held fixed for before/after; sensitivity check only.',
        },
        'interpretation': 'Immediate local change along the full enabled trajectory; not a decoder-layer removal ablation and not AP. Final/post matching and before-IoU grouping condition on model outputs.',
        'bandwidth': {k: getattr(model.refiner.heads[0], k) for k in (
            'bandwidth_mode', 'bandwidth_ratio', 'min_offset_pixels', 'offset_pixels')},
        'layers': {policy: {str(i): meter.compute() for i, meter in by_layer.items()}
                   for policy, by_layer in meters.items()},
        'elapsed_seconds': time.monotonic() - started,
        'trace_cache': str(cache.resolve()),
    }
    torch.save({'schema_version': 1, 'model': summary['model'],
                'inference_uses_gt': False, 'records': records}, cache)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    for i, layer in summary['layers']['final_geometry_fixed'].items():
        print(f'Layer {int(i)+1}: {json.dumps(layer["groups"]["all"])}', flush=True)
    print(f'Saved {output}; cache {cache}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--trace-cache', required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
