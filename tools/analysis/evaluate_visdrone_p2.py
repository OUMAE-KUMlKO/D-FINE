"""Evaluate one trained P2 head with all-query and GT-matched-only refinement."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

METRIC_NAMES = ('AP', 'AP50', 'AP75', 'APsmall', 'APmedium', 'APlarge',
                'AR1', 'AR10', 'AR100', 'ARsmall', 'ARmedium', 'ARlarge')


@torch.inference_mode()
def evaluate(model, cfg, args):
    from src.data import CocoEvaluator
    from src.solver.gt_refinement_metrics import GTRefinementMetrics
    from src.zoo.dfine.gt_refinement import DFINEGTRefiner
    from src.zoo.dfine.pred_refinement import DFINEPredRefiner
    from src.zoo.dfine.pred_refinement_criterion import PredRefinementCriterion

    if type(model) is not DFINEPredRefiner or model.refiner.sampling_mode != 'distribution':
        raise ValueError('Expected the trained distribution-guided P2 model')
    if model.training or model.requires_gt:
        raise ValueError('P2 evaluation requires eval mode and GT-free model inference')
    criterion = cfg.criterion.to(args.device).eval()
    if type(criterion) is not PredRefinementCriterion:
        raise ValueError('Expected the P2 baseline-matching criterion')
    matcher = criterion.matcher
    costs = {'cost_class': matcher.cost_class, 'cost_bbox': matcher.cost_bbox,
             'cost_giou': matcher.cost_giou}
    if costs != {'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2} or not matcher.use_focal_loss:
        raise ValueError('Comparison requires the oracle focal class/bbox/GIoU 2/5/2 matcher')
    loader = cfg.val_dataloader
    expected_ids = set(loader.dataset.coco.getImgIds())
    if len(loader.dataset) != args.expected_images or len(expected_ids) != args.expected_images:
        raise ValueError(f'Expected exactly {args.expected_images} validation images')
    postprocessor = cfg.postprocessor.to(args.device).eval()
    modes = ('all_queries', 'matched_only')
    evaluators = {mode: CocoEvaluator(loader.dataset.coco, ['bbox']) for mode in modes}
    meters = {mode: GTRefinementMetrics() for mode in modes}
    seen = set()
    refined_queries = {mode: 0 for mode in modes}
    started = time.monotonic()
    for batch_index, (images, originals) in enumerate(loader):
        images = images.to(args.device)
        # GT is used only after this shared, ordinary prediction-only forward.
        outputs = model(images)
        for key in ('pred_boxes', 'baseline_boxes', 'pred_logits'):
            if not torch.isfinite(outputs[key]).all():
                raise ValueError(f'Nonfinite P2 output: {key}')
        targets, ignored = DFINEGTRefiner._prepare_targets(
            originals, images.shape[-2:], images.device, 'xyxy_absolute',
            outputs['pred_logits'].shape[-1],
        )
        indices = criterion._match_baseline(outputs['baseline_boxes'], outputs['pred_logits'], targets)
        matched_mask = torch.zeros_like(outputs['refinement_mask'])
        for batch_id, (queries, _) in enumerate(indices):
            matched_mask[batch_id, queries] = outputs['refinement_mask'][batch_id, queries]
        baseline = outputs['baseline_boxes']
        matched_counts = baseline.new_tensor([len(q) for q, _ in indices], dtype=torch.long)
        refined_counts = matched_mask.sum(1)
        gt_counts = baseline.new_tensor([len(t['boxes']) for t in targets], dtype=torch.long)
        invalid_boxes = (~(torch.isfinite(baseline).all(-1) & (baseline[..., 2:] > 0).all(-1))).sum(1)
        per_image_counts = {
            'matched': matched_counts, 'refined': refined_counts,
            'ignored_gt': baseline.new_tensor(ignored, dtype=torch.long),
            'total_gt': gt_counts, 'unmatched_gt': gt_counts - matched_counts,
            'unmatched_queries': baseline.shape[1] - matched_counts,
            'invalid_gt_samples': matched_counts - refined_counts,
            'invalid_baseline_boxes': invalid_boxes,
        }
        common = {
            **outputs, 'matched_indices': indices, 'refinement_targets': targets,
            'refinement_counts_per_image': per_image_counts,
        }
        predictions = {
            'all_queries': common,
            'matched_only': {
                **common,
                'pred_boxes': torch.where(matched_mask[..., None], outputs['pred_boxes'], baseline),
                'refinement_mask': matched_mask,
            },
        }
        sizes = torch.stack([target['orig_size'] for target in originals]).to(args.device)
        for mode, prediction in predictions.items():
            # Both use the original logits; only the box selection changes.
            if prediction['pred_logits'] is not outputs['pred_logits']:
                raise AssertionError('Evaluation changed classification logits')
            meters[mode].update(prediction, images.shape[-2:])
            refined_queries[mode] += int(prediction['refinement_mask'].sum())
            detections = postprocessor(prediction, sizes)
            evaluators[mode].update({int(t['image_id'].item()): d for t, d in zip(originals, detections)})
        for target in originals:
            image_id = int(target['image_id'].item())
            if image_id in seen:
                raise ValueError(f'Duplicate validation image ID {image_id}')
            seen.add(image_id)
        if batch_index % 25 == 0 or len(seen) == args.expected_images:
            print(f'P2 paired evaluation: {len(seen)}/{args.expected_images}; '
                  f'{time.monotonic() - started:.1f}s', flush=True)
    if seen != expected_ids:
        raise ValueError(f'Validation coverage mismatch: processed {len(seen)} of {len(expected_ids)} IDs')
    results = {}
    for mode in modes:
        evaluators[mode].synchronize_between_processes()
        evaluators[mode].accumulate()
        evaluators[mode].summarize()
        meters[mode].synchronize_between_processes()
        results[mode] = {
            'coco_percent': dict(zip(METRIC_NAMES, (
                float(value) * 100 for value in evaluators[mode].coco_eval['bbox'].stats))),
            'fixed_geometry': meters[mode].compute(),
            'refined_queries': refined_queries[mode],
            'inference_uses_gt': mode == 'matched_only',
        }
    return {
        'images': len(seen), 'expected_images': args.expected_images, 'full_validation': True,
        'results': results, 'matching_costs': costs, 'matching_focal_alpha': matcher.alpha,
        'matching_focal_gamma': matcher.gamma,
        'matching_policy': 'Frozen baseline logits and boxes matched once per image; class/bbox/GIoU 2/5/2.',
        'classification_identical': True, 'model_forwards_per_batch': 1,
        'interpretation': (
            'all_queries is normal GT-free inference. matched_only uses GT to choose which query '
            'refinements to retain; every other box stays at the baseline. Fixed-pair geometry is '
            'identical in both modes by construction; COCO AP measures the effect of unmatched-query refinement.'
        ),
    }


@torch.inference_mode()
def run(args):
    from src.core import YAMLConfig

    output = Path(args.output_file)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    cfg = YAMLConfig(
        args.config, HGNetv2={'pretrained': False},
        DFINEPredRefiner={'baseline_checkpoint': None},
        val_dataloader={'total_batch_size': args.batch_size, 'num_workers': args.workers},
    )
    started = time.monotonic()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = cfg.model
    model.load_state_dict(checkpoint['model'], strict=True)
    model = model.to(args.device).float().eval()
    summary = evaluate(model, cfg, args)
    summary.update({
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'checkpoint_last_epoch': checkpoint.get('last_epoch'), 'weights': 'model',
        'model_class': type(model).__name__, 'config': str(Path(args.config).resolve()),
        'dataset': cfg.yaml_cfg['val_dataloader']['dataset'],
        'precision': 'float32', 'batch_size': args.batch_size,
        'training': False, 'checkpoint_modified': False,
        'elapsed_seconds': time.monotonic() - started,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(f'Saved {output}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--expected-images', type=int, default=548)
    args = parser.parse_args()
    if min(args.batch_size, args.threads, args.expected_images) < 1 or args.workers < 0:
        parser.error('batch size, threads and expected images must be positive; workers nonnegative')
    run(args)


if __name__ == '__main__':
    main()
