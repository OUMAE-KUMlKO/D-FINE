"""Compare error composition of two full-validation normal prediction caches."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_cache(summary_path, cache_path):
    summary = json.loads(Path(summary_path).read_text())
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    if (cache['schema_version'] != 1 or cache['model'] != summary['model']
            or not summary['full_validation'] or summary['inference_precision'] != 'float32'
            or summary['max_dets'] != [1, 10, 100]):
        raise ValueError('Expected full FP32 normal predictions with matching metadata')
    records = {r['image_id']: r for r in cache['records']}
    if len(records) != len(cache['records']) or len(records) != summary['images']:
        raise ValueError('Duplicate or missing prediction records')
    return summary, records


def recall_precision_decomposition(analyses):
    """Exact AP identity using the 101-point ceiling of fixed retained TPs."""
    grid = np.linspace(0, 1, 101)
    result = {}
    for name, analysis in analyses.items():
        ceilings = []
        categories = {}
        for category_id, thresholds in analysis['iou_metrics']['categories'].items():
            values = [0. if metrics['AR100'] == 0 else float(np.mean(grid <= metrics['AR100']))
                      for metrics in thresholds.values() if metrics['AR100'] is not None]
            if values:
                categories[category_id] = float(np.mean(values)) * 100
                ceilings.extend(values)
        ceiling = float(np.mean(ceilings)) * 100
        ap = analysis['original']['AP'] * 100
        result[name] = {'ap_percent': ap, 'recall_ceiling_percent': ceiling,
                        'fp_precision_loss_pp': ceiling - ap,
                        'category_recall_ceiling_percent': categories}
    for output_key, metric in (('delta_ap_pp', 'ap_percent'),
                               ('delta_recall_ceiling_pp', 'recall_ceiling_percent'),
                               ('delta_fp_precision_loss_pp', 'fp_precision_loss_pp')):
        result[output_key] = result['g5'][metric] - result['g0'][metric]
    result['definition'] = (
        'For each category and IoU, retain the existing TP matches and delete all FPs. '
        'Ceiling is the fraction of the original 101 recall levels <= AR100 '
        '(zero if AR100=0); exclude no-GT categories. AP=ceiling-FP precision loss. '
        'Fixed predictions/maxDets; no new matching or recovered detections. '
        'This is a numerical identity, not a causal separation of model abilities.'
    )
    return result


@torch.inference_mode()
def postprocess(records):
    from src.zoo.dfine.postprocessor import DFINEPostProcessor

    processor = DFINEPostProcessor(num_classes=10, use_focal_loss=True,
                                   num_top_queries=300, remap_mscoco_category=False).eval()
    rows = []
    for image_id, record in records.items():
        outputs = {key: value.unsqueeze(0)
                   for key, value in record['predictions']['normal'].items()}
        if outputs['pred_logits'].shape[-1] != 10:
            raise ValueError('Expected 10 VisDrone classes')
        detection = processor(outputs, torch.tensor([record['orig_size_wh']]))[0]
        boxes = detection['boxes'].clone()
        boxes[:, 2:] -= boxes[:, :2]
        rows.extend({'image_id': int(image_id), 'category_id': int(label),
                     'bbox': box, 'score': float(score)}
                    for label, box, score in zip(detection['labels'].tolist(), boxes.tolist(),
                                                detection['scores'].tolist()))
    return rows


def run(args):
    from faster_coco_eval import COCO
    from tools.analysis.detection_error_metrics import analyze_predictions

    output = Path(args.output_file)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(args.threads)
    summaries, records = {}, {}
    for name in ('g0', 'g5'):
        summaries[name], records[name] = load_cache(getattr(args, name + '_summary'),
                                                    getattr(args, name + '_cache'))
    coco = COCO(args.ann_file)
    if set(records['g0']) != set(records['g5']) or set(records['g0']) != set(coco.getImgIds()):
        raise ValueError('Both caches must cover exactly the annotated validation images')
    for image_id, g0 in records['g0'].items():
        g5 = records['g5'][image_id]
        for key in ('orig_size_wh', 'input_size_hw'):
            if g0[key] != g5[key]:
                raise ValueError(f'Mismatched {key} at image {image_id}')
        for key in ('boxes', 'labels', 'area_original'):
            if not torch.equal(g0['target'][key], g5['target'][key]):
                raise ValueError(f'Mismatched GT {key} at image {image_id}')
        actual = coco.imgs[image_id]
        if g0['orig_size_wh'] != [actual['width'], actual['height']]:
            raise ValueError('Annotation image size differs from cached original size')
    started = time.monotonic()
    analyses, reference_checks = {}, {}
    for name in ('g0', 'g5'):
        print(f'Analyzing {name}: {len(records[name])} images', flush=True)
        analyses[name] = analyze_predictions(coco, postprocess(records[name]))
        ap = analyses[name]['metrics']['size_groups']['all']['AP'] * 100
        legacy_ap = analyses[name]['legacy_metrics']['size_groups']['all']['AP'] * 100
        expected = summaries[name]['results']['normal']['coco_percent']['AP']
        difference = legacy_ap - expected
        reference_checks[name] = {'expected_ap_original_gt_ids': expected,
                                  'measured_ap_original_gt_ids': legacy_ap,
                                  'difference_pp': difference, 'tolerance_pp': .02,
                                  'analysis_ap_positive_gt_ids': ap,
                                  'annotation_id_fix_delta_pp': ap - legacy_ap}
        if abs(difference) > .02:
            raise ValueError(f'{name}: cached AP differs by {difference} pp')
        print(f'{name}: AP={ap:.6f}; original IDs AP={legacy_ap:.6f}; '
              f'{time.monotonic() - started:.1f}s', flush=True)
    categories = {}
    for cat_id in analyses['g0']['metrics']['categories']:
        before = analyses['g0']['metrics']['categories'][cat_id]['all']
        after = analyses['g5']['metrics']['categories'][cat_id]['all']
        categories[cat_id] = {'name': coco.cats[int(cat_id)]['name'],
                              'g0_percent': {k: v * 100 if v is not None else None
                                             for k, v in before.items()},
                              'g5_percent': {k: v * 100 if v is not None else None
                                             for k, v in after.items()},
                              'delta_pp': {k: (after[k] - v) * 100 if v is not None else None
                                            for k, v in before.items()}}
    valid = [c for c in categories.values() if c['delta_pp']['AP'] is not None]
    for category in valid:
        category['total_ap_contribution_pp'] = category['delta_pp']['AP'] / len(valid)
    result = {
        'images': len(records['g0']), 'ann_file': str(Path(args.ann_file).resolve()),
        'models': {k: v['model'] for k, v in summaries.items()},
        'caches': {k: str(Path(getattr(args, k + '_cache')).resolve()) for k in summaries},
        'normal_ap_reference_checks': reference_checks,
        'all_image_sizes_and_gt_equal': True,
        'units': 'metrics AP/AR in [0,1]; *_pp and *_percent in percentage points/percent',
        'analyses': analyses, 'category_comparison': categories,
        'recall_precision_decomposition': recall_precision_decomposition(analyses),
        'elapsed_seconds': time.monotonic() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(f'Saved {output}', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('g0-summary', 'g0-cache', 'g5-summary', 'g5-cache', 'ann-file', 'output-file'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--threads', type=int, default=4)
    run(parser.parse_args())
