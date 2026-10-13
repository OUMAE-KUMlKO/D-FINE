"""Cache an unchanged detector EMA for comparison with existing validation predictions."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.analysis.diagnose_candidate_features import METRIC_NAMES, validate_outputs
from tools.analysis.evaluate_detector_geometry import load_ema_model


def check_record(reference, target, original, input_size):
    if reference['orig_size_wh'] != original['orig_size'].tolist():
        raise ValueError('Original image size differs from reference')
    if reference['input_size_hw'] != list(input_size):
        raise ValueError('Input image size differs from reference')
    for key in ('boxes', 'labels', 'area_original'):
        if not torch.equal(reference['target'][key], target[key].cpu()):
            raise ValueError(f'Validation GT differs from reference: {key}')


@torch.inference_mode()
def run(args):
    from src.core import YAMLConfig
    from src.data import CocoEvaluator
    from src.zoo.dfine.gt_refinement import DFINEGTRefiner

    output, cache_path = Path(args.output_file), Path(args.prediction_cache)
    if output.resolve() == cache_path.resolve():
        raise ValueError('Output and cache paths must differ')
    for path in (output, cache_path):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    reference_summary = json.loads(Path(args.reference_summary).read_text())
    reference_cache = torch.load(args.reference_cache, map_location='cpu', weights_only=False)
    if (reference_cache['schema_version'] != 1 or not reference_summary['full_validation']
            or reference_summary['inference_precision'] != 'float32'
            or reference_cache['model'] != reference_summary['model']):
        raise ValueError('Expected a full FP32 cache matching reference metadata')
    reference = {record['image_id']: record for record in reference_cache['records']}
    if len(reference) != len(reference_cache['records']):
        raise ValueError('Duplicate reference image IDs')
    del reference_cache
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    cfg = YAMLConfig(args.config, HGNetv2={'pretrained': False},
                     val_dataloader={'total_batch_size': args.batch_size,
                                     'num_workers': args.workers})
    model, metadata = load_ema_model(cfg, args.checkpoint, args.device)
    loader = cfg.val_dataloader
    if set(reference) != set(loader.dataset.coco.getImgIds()):
        raise ValueError('Reference must contain exactly the current validation images')
    postprocessor = cfg.postprocessor.to(args.device).eval()
    evaluator = CocoEvaluator(loader.dataset.coco, ['bbox'])
    if list(evaluator.coco_eval['bbox'].params.maxDets) != [1, 10, 100]:
        raise ValueError('Expected maxDets=[1,10,100]')
    records, seen = [], set()
    started = time.monotonic()
    for images, original_targets in loader:
        images = images.to(args.device)
        prediction = model(images)
        validate_outputs(prediction, 'normal')
        targets, ignored = DFINEGTRefiner._prepare_targets(
            original_targets, images.shape[-2:], images.device, 'xyxy_absolute',
            prediction['pred_logits'].shape[-1])
        original_sizes = torch.stack([t['orig_size'] for t in original_targets]).to(args.device)
        detections = postprocessor(prediction, original_sizes)
        for b, (original, target, count, detection) in enumerate(
                zip(original_targets, targets, ignored, detections)):
            image_id = int(original['image_id'].item())
            if image_id in seen:
                raise ValueError('Duplicate image in validation loader')
            seen.add(image_id)
            if count or len(original['area']) != len(target['boxes']):
                raise ValueError('GT filtering changed annotation alignment')
            target['area_original'] = original['area'].detach().cpu()
            check_record(reference[image_id], target, original, images.shape[-2:])
            evaluator.update({image_id: detection})
            records.append({
                'image_id': image_id, 'input_size_hw': list(images.shape[-2:]),
                'orig_size_wh': original['orig_size'].tolist(),
                'target': {key: value.detach().cpu() for key, value in target.items()},
                'predictions': {'normal': {key: prediction[key][b].detach().float().cpu()
                                           for key in ('pred_boxes', 'pred_logits')}},
            })
        if len(records) % 100 == 0 or len(records) == len(loader.dataset):
            print(f'Cache: {len(records)}/{len(loader.dataset)}; '
                  f'{time.monotonic() - started:.1f}s', flush=True)
    if seen != set(reference):
        raise ValueError('Validation incomplete')
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    metrics = dict(zip(METRIC_NAMES, [float(x) * 100
                       for x in evaluator.coco_eval['bbox'].stats]))
    difference = metrics['AP'] - args.expected_ap
    summary = {
        'model': {**metadata, 'config': str(Path(args.config).resolve())},
        'dataset': cfg.yaml_cfg['val_dataloader']['dataset'],
        'images': len(records), 'valid_gt': sum(len(r['target']['boxes']) for r in records),
        'full_validation': True, 'inference_precision': 'float32', 'inference_uses_gt': False,
        'input_sizes_hw': sorted({tuple(r['input_size_hw']) for r in records}),
        'max_dets': [1, 10, 100], 'results': {'normal': {'coco_percent': metrics}},
        'prediction_cache': str(cache_path.resolve()),
        'reference_cache': str(Path(args.reference_cache).resolve()),
        'all_image_sizes_and_gt_equal': True,
        'ap_reference_check': {'expected_ap': args.expected_ap, 'difference_pp': difference,
                               'tolerance_pp': .02, 'within_tolerance': abs(difference) <= .02},
        'elapsed_seconds': time.monotonic() - started,
    }
    torch.save({'schema_version': 1, 'model': summary['model'], 'records': records}, cache_path)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    if abs(difference) > .02:
        raise ValueError(f'AP differs from saved reference by {difference} pp')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'reference-summary', 'reference-cache',
                 'output-file', 'prediction-cache'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--expected-ap', type=float, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    run(parser.parse_args())
