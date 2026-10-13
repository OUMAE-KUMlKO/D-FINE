"""Evaluate alpha=0/0.5 while reusing the saved alpha=1 predictions."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.analysis.diagnose_candidate_features import METRIC_NAMES, validate_outputs
from tools.analysis.evaluate_detector_geometry import load_ema_model
from tools.analysis.layer_strip_effect_metrics import LayerEffectMetrics


SCALES = {'alpha_0': 0., 'alpha_0.5': .5}


def reference_batch(records, originals, targets, image_size, device):
    """Align cached normal outputs and matches by ID, checking GT identity."""
    selected = []
    for original, target in zip(originals, targets):
        image_id = int(original['image_id'].item())
        if image_id not in records:
            raise ValueError(f'Image {image_id} missing from normal cache')
        record = records[image_id]
        if (tuple(record['input_size_hw']) != tuple(image_size)
                or record['orig_size_wh'] != original['orig_size'].tolist()):
            raise ValueError(f'Image {image_id} size differs from normal cache')
        for key in ('boxes', 'labels', 'area_original'):
            if not torch.equal(record['target'][key], target[key].detach().cpu()):
                raise ValueError(f'Image {image_id} GT {key} differs from normal cache')
        selected.append(record)
    outputs = {key: torch.stack([record['predictions']['normal'][key]
                                 for record in selected]).to(device)
               for key in ('pred_boxes', 'pred_logits')}
    indices = [record['normal_geometry_matches'] for record in selected]
    return outputs, indices


@torch.inference_mode()
def run(args):
    from src.core import YAMLConfig
    from src.data import CocoEvaluator
    from src.zoo.dfine.gt_refinement import DFINEGTRefiner
    from tools.analysis.strip_correction_scale import strip_correction_scale

    output, cache_path = Path(args.output_file), Path(args.prediction_cache)
    if output.resolve() == cache_path.resolve():
        raise ValueError('Summary and cache must have different paths')
    for path in (output, cache_path):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    reference = json.loads(Path(args.baseline_summary).read_text())
    saved = torch.load(args.baseline_cache, map_location='cpu', weights_only=False)
    if not reference['full_validation'] or reference['inference_precision'] != 'float32':
        raise ValueError('Reference must be a complete FP32 validation')
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    cfg = YAMLConfig(args.config, HGNetv2={'pretrained': False},
                     val_dataloader={'total_batch_size': args.batch_size,
                                     'num_workers': args.workers})
    model, metadata = load_ema_model(cfg, args.checkpoint, args.device)
    for source in (reference['model'], saved['model']):
        for key in ('checkpoint', 'checkpoint_last_epoch', 'weights', 'model_class'):
            if source[key] != metadata[key]:
                raise ValueError(f'Normal reference model mismatch: {key}')
    if (type(model).__name__ != 'DFINEAllLayerStrip'
            or len(model.refiner.heads) != model.decoder.eval_idx + 1):
        raise ValueError('Expected a strip head at every evaluated decoder layer')
    records = {record['image_id']: record for record in saved['records']}
    loader = cfg.val_dataloader
    if not len(records) == len(saved['records']) == reference['images'] == len(loader.dataset):
        raise ValueError('Normal cache must cover each validation image exactly once')
    postprocessor = cfg.postprocessor.to(args.device).eval()
    evaluators = {mode: CocoEvaluator(loader.dataset.coco, ['bbox']) for mode in SCALES}
    if reference['max_dets'] != [1, 10, 100] or any(
        list(evaluator.coco_eval['bbox'].params.maxDets) != reference['max_dets']
        for evaluator in evaluators.values()
    ):
        raise ValueError('Expected matching COCO maxDets=[1,10,100]')
    meters = {mode: LayerEffectMetrics() for mode in SCALES}
    saved_predictions, seen = [], set()
    gt_count = pair_count = 0
    started = time.monotonic()
    for batch_index, (images, originals) in enumerate(loader):
        images = images.to(args.device)
        targets, ignored = DFINEGTRefiner._prepare_targets(
            originals, images.shape[-2:], images.device, 'xyxy_absolute',
            cfg.yaml_cfg['num_classes'],
        )
        for original, target, count in zip(originals, targets, ignored):
            if count or len(original['area']) != len(target['boxes']):
                raise ValueError('Canonical GT filtering changed annotation alignment')
            target['area_original'] = original['area'].detach().cpu()
        normal, indices = reference_batch(records, originals, targets, images.shape[-2:], args.device)
        predictions = {}
        original_sizes = torch.stack([target['orig_size'] for target in originals]).to(args.device)
        for mode, alpha in SCALES.items():
            with strip_correction_scale(model, alpha):
                prediction = model(images)
            validate_outputs(prediction, mode)
            meters[mode].update(normal['pred_boxes'], prediction['pred_boxes'],
                                targets, indices, images.shape[-2:])
            detections = postprocessor(prediction, original_sizes)
            evaluators[mode].update({int(target['image_id'].item()): detection
                                     for target, detection in zip(originals, detections)})
            predictions[mode] = prediction
        if batch_index == 0:
            with strip_correction_scale(model, 1.):
                restored = model(images)
            for key in ('pred_boxes', 'pred_logits'):
                torch.testing.assert_close(restored[key], normal[key], atol=1e-6, rtol=1e-6)
            print('First batch: restored alpha=1 agrees with the saved normal reference.', flush=True)
        for b, original in enumerate(originals):
            image_id = int(original['image_id'].item())
            if image_id in seen:
                raise ValueError(f'Duplicate validation image: {image_id}')
            seen.add(image_id)
            saved_predictions.append({
                'image_id': image_id,
                'predictions': {mode: {key: prediction[key][b].detach().float().cpu()
                                       for key in ('pred_boxes', 'pred_logits')}
                                for mode, prediction in predictions.items()},
            })
        gt_count += sum(len(target['boxes']) for target in targets)
        pair_count += sum(len(query_ids) for query_ids, _ in indices)
        if batch_index % 25 == 0 or len(seen) == len(loader.dataset):
            print(f'Correction scale: {len(seen)}/{len(loader.dataset)} images; '
                  f'{time.monotonic() - started:.1f}s', flush=True)
    if seen != set(records) or pair_count != reference['fixed_pairs'] or gt_count != reference['valid_gt']:
        raise ValueError('Validation coverage differs from the normal reference')
    results = {'alpha_1': {key: reference['results']['normal'][key]
                           for key in ('coco_percent', 'fixed_geometry')}}
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
    for result in results.values():
        result['delta_coco_pp_vs_alpha1'] = {
            name: value - results['alpha_1']['coco_percent'][name]
            for name, value in result['coco_percent'].items()
        }
    summary = {
        'model': {**metadata, 'config': str(Path(args.config).resolve())},
        'dataset': cfg.yaml_cfg['val_dataloader']['dataset'],
        'images': len(seen), 'valid_gt': gt_count, 'fixed_pairs': pair_count,
        'full_validation': True, 'inference_precision': 'float32', 'inference_uses_gt': False,
        'input_sizes_hw': reference['input_sizes_hw'], 'batch_size': args.batch_size,
        'intervention': 'All strip heads: scale score_head output delta before masking and addition to original logits; downstream states propagate normally.',
        'new_inference_scales': SCALES, 'reused_alpha': 1.,
        'baseline_summary': str(Path(args.baseline_summary).resolve()),
        'baseline_cache': str(Path(args.baseline_cache).resolve()),
        'first_batch_reference_close': True, 'reference_check_atol': 1e-6, 'reference_check_rtol': 1e-6,
        'max_dets': reference['max_dets'], 'matching_costs': reference['matching_costs'],
        'matching_policy': 'Reuse saved normal final geometry query/GT assignment for every image; no rematching.',
        'geometry_before_after': 'before=cached alpha=1; after=indicated scale',
        'interpretation': 'Frozen jointly trained G5; alpha=0 is not the independently trained G0. Uniform inference scaling does not prove or rule out target-dependent gating.',
        'results': results, 'prediction_cache': str(cache_path.resolve()),
        'elapsed_seconds': time.monotonic() - started,
    }
    torch.save({'schema_version': 1, 'model': summary['model'],
                'baseline_cache': summary['baseline_cache'], 'records': saved_predictions}, cache_path)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    for mode, result in results.items():
        print(f'{mode}: AP={result["coco_percent"]["AP"]:.4f}; '
              f'AP75={result["coco_percent"]["AP75"]:.4f}; '
              f'edge MAE={result["fixed_geometry"]["groups"]["all"]["after"]["mean_edge_mae_px"]}',
              flush=True)
    print(f'Saved {output}; cache {cache_path}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'baseline-summary', 'baseline-cache',
                 'output-file', 'prediction-cache'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if min(args.batch_size, args.threads) < 1 or args.workers < 0:
        parser.error('batch size/threads must be positive; workers nonnegative')
    run(args)


if __name__ == '__main__':
    main()
