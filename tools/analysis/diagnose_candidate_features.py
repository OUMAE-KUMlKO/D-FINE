"""Evaluate normal, zero and Gaussian candidate features at one fixed EMA."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.analysis.evaluate_detector_geometry import load_ema_model
from tools.analysis.layer_strip_effect_metrics import LayerEffectMetrics


MODES = ('normal', 'zero', 'noise')
METRIC_NAMES = ('AP', 'AP50', 'AP75', 'APsmall', 'APmedium', 'APlarge',
                'AR1', 'AR10', 'AR100', 'ARsmall', 'ARmedium', 'ARlarge')


def validate_outputs(outputs, mode):
    for key in ('pred_boxes', 'pred_logits'):
        if not torch.isfinite(outputs[key]).all():
            raise ValueError(f'{mode}: non-finite {key}')
    if not (outputs['pred_boxes'][..., 2:] > 0).all():
        raise ValueError(f'{mode}: non-positive predicted width/height')


def update_fixed_geometry(outputs, targets, image_size, matcher, meters):
    """Match only the normal predictions; every mode keeps these query IDs."""
    for mode, prediction in outputs.items():
        validate_outputs(prediction, mode)
    normal = outputs['normal']
    indices = matcher._match_baseline(
        normal['pred_boxes'], torch.zeros_like(normal['pred_logits']), targets
    )
    for mode, prediction in outputs.items():
        meters[mode].update(normal['pred_boxes'], prediction['pred_boxes'],
                            targets, indices, image_size)
    return indices


@torch.inference_mode()
def run(args):
    from src.core import YAMLConfig
    from src.data import CocoEvaluator
    from src.zoo.dfine.gt_refinement import DFINEGTRefiner
    from src.zoo.dfine.matcher import HungarianMatcher
    from src.zoo.dfine.pred_refinement_criterion import PredRefinementCriterion
    from tools.analysis.candidate_feature_ablation import candidate_feature_ablation

    output, cache = Path(args.output_file), Path(args.prediction_cache)
    if output.resolve() == cache.resolve():
        raise ValueError('Summary and prediction cache must use different paths')
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
        raise ValueError('This diagnostic requires DFINEAllLayerStrip')
    if len(model.refiner.heads) != model.decoder.eval_idx + 1:
        raise ValueError('This diagnostic requires a strip head at every evaluated layer')
    if any(head.use_log_prior or head.query_fusion != 'channel_gate'
           for head in model.refiner.heads):
        raise ValueError('Expected G5: channel_gate and no explicit log-prior input')
    loader = cfg.val_dataloader
    postprocessor = cfg.postprocessor.to(args.device).eval()
    evaluators = {mode: CocoEvaluator(loader.dataset.coco, ['bbox']) for mode in MODES}
    for evaluator in evaluators.values():
        if list(evaluator.coco_eval['bbox'].params.maxDets) != [1, 10, 100]:
            raise ValueError('Expected COCO maxDets=[1,10,100]')
    costs = {'cost_class': 0, 'cost_bbox': 5, 'cost_giou': 2}
    matcher = PredRefinementCriterion(HungarianMatcher(
        costs, use_focal_loss=True
    )).to(args.device).eval()
    meters = {mode: LayerEffectMetrics() for mode in MODES}
    records, images_seen, gt_seen, fixed_pairs = [], 0, 0, 0
    input_sizes = set()
    started = time.monotonic()
    for batch_index, (images, original_targets) in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        images = images.to(args.device)
        predictions = {}
        batch_seed = args.seed + batch_index * 1009
        for mode in MODES:
            with candidate_feature_ablation(model, mode, seed=batch_seed):
                predictions[mode] = model(images)
            validate_outputs(predictions[mode], mode)
        if batch_index == 0:
            restored = model(images)
            for key in ('pred_boxes', 'pred_logits'):
                torch.testing.assert_close(predictions['normal'][key], restored[key],
                                           atol=0, rtol=0)
            print('First batch: normal and restored ordinary inference are exactly equal.',
                  flush=True)
        targets, ignored = DFINEGTRefiner._prepare_targets(
            original_targets, images.shape[-2:], images.device, 'xyxy_absolute',
            predictions['normal']['pred_logits'].shape[-1],
        )
        for original, target, count in zip(original_targets, targets, ignored):
            if count or len(original['area']) != len(target['boxes']):
                raise ValueError('Canonical GT filtering changed annotation alignment')
            target['area_original'] = original['area'].detach().cpu()
        indices = update_fixed_geometry(predictions, targets, images.shape[-2:], matcher, meters)
        original_sizes = torch.stack([target['orig_size'] for target in original_targets]).to(args.device)
        for mode, prediction in predictions.items():
            detections = postprocessor(prediction, original_sizes)
            evaluators[mode].update({
                int(target['image_id'].item()): detection
                for target, detection in zip(original_targets, detections)
            })
        for b, (original, target) in enumerate(zip(original_targets, targets)):
            records.append({
                'image_id': int(original['image_id'].item()),
                'input_size_hw': list(images.shape[-2:]),
                'orig_size_wh': original['orig_size'].tolist(),
                'batch_seed': batch_seed,
                'target': {key: value.detach().cpu() for key, value in target.items()},
                'normal_geometry_matches': tuple(value.detach().cpu() for value in indices[b]),
                'predictions': {mode: {key: prediction[key][b].detach().float().cpu()
                                       for key in ('pred_boxes', 'pred_logits')}
                                for mode, prediction in predictions.items()},
            })
        images_seen += len(images)
        gt_seen += sum(len(target['boxes']) for target in targets)
        fixed_pairs += sum(len(query_ids) for query_ids, _ in indices)
        input_sizes.add(tuple(images.shape[-2:]))
        if batch_index % 25 == 0 or images_seen == len(loader.dataset):
            print(f'Feature ablation: {images_seen}/{len(loader.dataset)} images; '
                  f'{time.monotonic() - started:.1f}s', flush=True)
    if not images_seen:
        raise ValueError('No validation images processed')
    results = {}
    for mode, evaluator in evaluators.items():
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        print(f'COCO results: {mode}', flush=True)
        evaluator.summarize()
        results[mode] = {
            'coco_percent': dict(zip(METRIC_NAMES,
                                    [float(value) * 100 for value in evaluator.coco_eval['bbox'].stats])),
            'fixed_geometry': meters[mode].compute(),
        }
    normal_metrics = results['normal']['coco_percent']
    for result in results.values():
        result['delta_coco_pp_vs_normal'] = {
            name: value - normal_metrics[name] for name, value in result['coco_percent'].items()
        }
    full_validation = images_seen == len(loader.dataset)
    reference_check = None
    if args.expected_ap is not None and full_validation:
        difference = normal_metrics['AP'] - args.expected_ap
        reference_check = {'expected_ap': args.expected_ap, 'difference_pp': difference,
                           'tolerance_pp': args.ap_tolerance,
                           'within_tolerance': abs(difference) <= args.ap_tolerance}
    summary = {
        'model': {**metadata, 'config': str(Path(args.config).resolve())},
        'dataset': cfg.yaml_cfg['val_dataloader']['dataset'],
        'images': images_seen, 'valid_gt': gt_seen, 'fixed_pairs': fixed_pairs,
        'full_validation': full_validation, 'input_sizes_hw': sorted(input_sizes),
        'inference_uses_gt': False, 'inference_precision': 'float32',
        'intervention': 'Every strip head: edge_fusion output, before query channel modulation; original distribution residual untouched.',
        'noise': {'distribution': 'independent standard normal N(0,1)', 'base_seed': args.seed,
                  'batch_seed': 'base_seed + batch_index * 1009',
                  'layer_seed': 'batch_seed + layer_index * 1000003',
                  'batch_size': args.batch_size,
                  'reproducibility': 'Fixed batch size, validation order and device; not batch-size invariant.'},
        'max_dets': [1, 10, 100], 'matching_costs': costs,
        'matching_policy': 'Normal final unfiltered query boxes matched once; all modes reuse identical query/GT indices.',
        'geometry_before_after': 'before=normal; after=the indicated mode',
        'clip_boxes': False, 'score_threshold': None,
        'first_batch_restore_exact': True, 'normal_ap_reference_check': reference_check,
        'interpretation': 'Frozen-checkpoint inference sensitivity, including distribution shift and downstream propagation; not a training ablation or proof of superiority over G0.',
        'results': results, 'prediction_cache': str(cache.resolve()),
        'elapsed_seconds': time.monotonic() - started,
    }
    torch.save({'schema_version': 1, 'model': summary['model'], 'records': records}, cache)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    for mode, result in results.items():
        geometry = result['fixed_geometry']['groups']['all']
        print(f'{mode}: AP={result["coco_percent"]["AP"]:.4f}; '
              f'AP75={result["coco_percent"]["AP75"]:.4f}; '
              f'edge MAE={geometry["after"]["mean_edge_mae_px"]}', flush=True)
    print(f'Saved {output}; cache {cache}', flush=True)
    if reference_check is not None and not reference_check['within_tolerance']:
        raise ValueError(f'Normal AP differs from the saved reference: {reference_check}')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--prediction-cache', required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-batches', type=int)
    parser.add_argument('--expected-ap', type=float)
    parser.add_argument('--ap-tolerance', type=float, default=0.02)
    args = parser.parse_args()
    if min(args.batch_size, args.threads) < 1 or args.workers < 0:
        parser.error('batch size/threads must be positive; workers nonnegative')
    if args.max_batches is not None and args.max_batches < 1:
        parser.error('max-batches must be positive')
    if args.seed < 0 or args.ap_tolerance < 0:
        parser.error('seed and ap-tolerance must be nonnegative')
    run(args)


if __name__ == '__main__':
    main()
