"""Diagnose existing G5 candidate scores and GT-oracle feature pooling, without training."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.analysis.candidate_pooling import candidate_pooling
from tools.analysis.diagnose_candidate_features import METRIC_NAMES, validate_outputs
from tools.analysis.evaluate_detector_geometry import load_ema_model


@torch.inference_mode()
def evaluate_oracle(model, cfg, args):
    from src.data import CocoEvaluator
    from src.solver.gt_refinement_metrics import GTRefinementMetrics

    modes = ('gt_nearest', 'prior_argmax', 'prior_weighted')
    loader = cfg.val_dataloader
    postprocessor = cfg.postprocessor.to(args.device).eval()
    evaluators = {mode: CocoEvaluator(loader.dataset.coco, ['bbox']) for mode in modes}
    meters = {mode: GTRefinementMetrics() for mode in modes}
    records, seen = [], set()
    started = time.monotonic()
    first_batch_restore_exact = False
    for batch_index, (images, originals) in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        images = images.to(args.device)
        predictions = {}
        for mode in modes:
            with candidate_pooling(model, mode):
                predictions[mode] = model(images, originals, target_format='xyxy_absolute')
            validate_outputs(predictions[mode], mode)
        reference = predictions['gt_nearest']
        for mode, prediction in predictions.items():
            for key in ('baseline_boxes', 'pred_logits', 'refinement_mask'):
                torch.testing.assert_close(prediction[key], reference[key], rtol=0, atol=0)
            for before, after in zip(reference['matched_indices'], prediction['matched_indices']):
                for left, right in zip(before, after):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
            unchanged = ~reference['refinement_mask']
            torch.testing.assert_close(prediction['pred_boxes'][unchanged],
                                       reference['baseline_boxes'][unchanged], rtol=0, atol=0)
            meters[mode].update(prediction, images.shape[-2:])
            sizes = torch.stack([t['orig_size'] for t in originals]).to(args.device)
            detections = postprocessor(prediction, sizes)
            evaluators[mode].update({int(t['image_id'].item()): d
                                     for t, d in zip(originals, detections)})
        if batch_index == 0:
            restored = model(images, originals, target_format='xyxy_absolute')
            for key in ('pred_boxes', 'pred_logits'):
                torch.testing.assert_close(restored[key], reference[key], rtol=0, atol=0)
            first_batch_restore_exact = True
            print('First batch: oracle pooling hooks restore exact ordinary output.', flush=True)
        for b, target in enumerate(originals):
            image_id = int(target['image_id'].item())
            if image_id in seen:
                raise ValueError(f'Duplicate image ID {image_id}')
            seen.add(image_id)
            records.append({
                'image_id': image_id,
                'baseline_boxes': reference['baseline_boxes'][b].cpu(),
                'pred_logits': reference['pred_logits'][b].cpu(),
                'matched_indices': tuple(v.cpu() for v in reference['matched_indices'][b]),
                'refinement_mask': reference['refinement_mask'][b].cpu(),
                'target': {k: v.cpu() for k, v in reference['refinement_targets'][b].items()},
                'pred_boxes': {mode: value['pred_boxes'][b].cpu() for mode, value in predictions.items()},
            })
        if batch_index % 25 == 0 or len(seen) == len(loader.dataset):
            print(f'Oracle pooling: {len(seen)}/{len(loader.dataset)}; {time.monotonic()-started:.1f}s', flush=True)
    if not seen:
        raise ValueError('No validation images processed')
    results = {}
    for mode, evaluator in evaluators.items():
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()
        results[mode] = {
            'coco_percent': dict(zip(METRIC_NAMES, (float(x)*100 for x in evaluator.coco_eval['bbox'].stats))),
            'fixed_geometry': meters[mode].compute(),
        }
    full = len(seen) == len(loader.dataset)
    difference = (results['gt_nearest']['coco_percent']['AP'] - args.expected_ap
                  if args.expected_ap is not None and full else None)
    torch.save({'schema_version': 1, 'records': records}, args.prediction_cache)
    return {
        'images': len(seen), 'full_validation': full, 'first_batch_restore_exact': first_batch_restore_exact,
        'results': results, 'gt_nearest_ap_difference_pp': difference,
        'reference_check_passed': difference is None or abs(difference) <= args.ap_tolerance,
        'classification_matching_and_valid_mask_preserved': True,
        'matching_policy': 'Frozen G0 class+bbox+GIoU Hungarian matching; identical query/GT pairs across modes.',
        'inference_uses_gt': True, 'prediction_cache': str(Path(args.prediction_cache).resolve()),
        'interpretation': 'Inference sensitivity of a head trained with GT one-hot selection; weighted pooling changes its input distribution. Not a G5 selector ablation.',
    }


@torch.inference_mode()
def evaluate_g5(model, cfg, args):
    from src.zoo.dfine.gt_refinement import DFINEGTRefiner
    from src.zoo.dfine.matcher import HungarianMatcher
    from src.zoo.dfine.pred_refinement_criterion import PredRefinementCriterion
    from tools.analysis.g5_candidate_selection import trace_candidate_selection, CandidateSelectionMetrics

    loader = cfg.val_dataloader
    costs = {'cost_class': 0, 'cost_bbox': 5, 'cost_giou': 2}
    matcher = PredRefinementCriterion(HungarianMatcher(costs, use_focal_loss=True)).to(args.device).eval()
    meters = {i: CandidateSelectionMetrics() for i in range(len(model.refiner.heads))}
    seen, pairs, gt_count = set(), 0, 0
    started = time.monotonic()
    for batch_index, (images, originals) in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        images = images.to(args.device)
        trace = trace_candidate_selection(model, images)
        outputs = trace['outputs']
        validate_outputs(outputs, 'g5')
        if batch_index == 0:
            ordinary = model(images)
            for key in ('pred_boxes', 'pred_logits'):
                torch.testing.assert_close(outputs[key], ordinary[key], rtol=0, atol=0)
            print('First batch: G5 observer preserves exact ordinary output.', flush=True)
        targets, _ = DFINEGTRefiner._prepare_targets(
            originals, images.shape[-2:], images.device, 'xyxy_absolute',
            outputs['pred_logits'].shape[-1])
        indices = matcher._match_baseline(outputs['pred_boxes'], torch.zeros_like(outputs['pred_logits']), targets)
        for i, layer in trace['layers'].items():
            meters[i].update(layer, targets, indices, images.shape[-2:])
        for target in originals:
            image_id = int(target['image_id'].item())
            if image_id in seen:
                raise ValueError(f'Duplicate image ID {image_id}')
            seen.add(image_id)
        pairs += sum(len(q) for q, _ in indices)
        gt_count += sum(len(t['boxes']) for t in targets)
        if batch_index % 25 == 0 or len(seen) == len(loader.dataset):
            print(f'G5 candidate scores: {len(seen)}/{len(loader.dataset)}; {time.monotonic()-started:.1f}s', flush=True)
    if not seen:
        raise ValueError('No validation images processed')
    return {
        'images': len(seen), 'valid_gt': gt_count, 'fixed_pairs': pairs,
        'full_validation': len(seen) == len(loader.dataset), 'first_batch_trace_exact': True,
        'matching_costs': costs, 'matching_policy': 'Final ordinary G5 geometry assignment fixed across all layers and modes.',
        'inference_uses_gt': False, 'layers': {str(i): m.compute() for i, m in meters.items()},
        'interpretation': 'Local score replay on the same trajectory; zero_visual only zeros visual score inputs, with fixed query/offset/prior. Not propagated ablation. Nearest-bin probability alone is not accuracy because continuous GT can lie between bins.',
    }


@torch.inference_mode()
def run(args):
    from src.core import YAMLConfig

    output = Path(args.output_file)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.stage == 'oracle':
        if not args.prediction_cache:
            raise ValueError('Oracle stage requires --prediction-cache')
        cache = Path(args.prediction_cache)
        if cache.exists() or cache.resolve() == output.resolve():
            raise FileExistsError(cache)
        cache.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    cfg = YAMLConfig(args.config, HGNetv2={'pretrained': False}, val_dataloader={
        'total_batch_size': args.batch_size, 'num_workers': args.workers,
        'dataset': {'ann_file': args.validation_annotations},
    })
    started = time.monotonic()
    if args.stage == 'g5':
        model, metadata = load_ema_model(cfg, args.checkpoint, args.device)
        summary = evaluate_g5(model, cfg, args)
    else:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model = cfg.model
        model.load_state_dict(checkpoint['model'], strict=True)
        model = model.to(args.device).eval()
        metadata = {'checkpoint': str(Path(args.checkpoint).resolve()),
                    'checkpoint_last_epoch': checkpoint.get('last_epoch'),
                    'weights': 'model', 'model_class': type(model).__name__}
        summary = evaluate_oracle(model, cfg, args)
    summary.update({
        'stage': args.stage, 'model': metadata, 'config': str(Path(args.config).resolve()),
        'dataset': cfg.yaml_cfg['val_dataloader']['dataset'], 'precision': 'float32',
        'training': False, 'checkpoint_modified': False, 'seed': 0,
        'batch_size': args.batch_size, 'elapsed_seconds': time.monotonic()-started,
    })
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(f'Saved {output}', flush=True)
    if summary.get('reference_check_passed') is False:
        raise ValueError(f'GT-nearest AP differs from reference by {summary["gt_nearest_ap_difference_pp"]} pp')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('g5', 'oracle'), required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--validation-annotations', required=True)
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--prediction-cache')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-batches', type=int)
    parser.add_argument('--expected-ap', type=float, default=36.90383272084086)
    parser.add_argument('--ap-tolerance', type=float, default=.02)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
