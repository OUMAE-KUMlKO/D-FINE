"""Evaluate saved detector predictions on canonical-GT image-density subsets."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BINS = ('le30', '31to60', '61to100', 'gt100')


def density_bin(count):
    return 'le30' if count <= 30 else '31to60' if count <= 60 else '61to100' if count <= 100 else 'gt100'


@torch.inference_mode()
def postprocess_records(records, model):
    from src.zoo.dfine.postprocessor import DFINEPostProcessor

    processor = DFINEPostProcessor(num_classes=10, use_focal_loss=True,
                                   num_top_queries=300, remap_mscoco_category=False)
    predictions = []
    for record in records:
        outputs = {key: record[model][key].unsqueeze(0)
                   for key in ('pred_boxes', 'pred_logits')}
        # Cache uses H,W; the repository postprocessor multiplies xyxy by W,H,W,H.
        height, width = record['orig_size']
        result = processor(outputs, torch.tensor([[width, height]]))[0]
        boxes = result['boxes'].clone()
        boxes[:, 2:] -= boxes[:, :2]
        predictions.extend(dict(image_id=int(record['image_id']), category_id=int(label),
                                bbox=box, score=float(score))
                           for label, box, score in zip(result['labels'].tolist(), boxes.tolist(),
                                                        result['scores'].tolist()))
    return predictions


def extract_metrics(evaluator):
    precision, recall = evaluator.eval['precision'], evaluator.eval['recall']
    params = evaluator.params
    maxdet = list(params.maxDets).index(100)
    i50 = int(np.flatnonzero(np.isclose(params.iouThrs, .5))[0])
    i75 = int(np.flatnonzero(np.isclose(params.iouThrs, .75))[0])

    def mean(values):
        valid = values[values > -1]
        return float(valid.mean()) if valid.size else None

    def group(category=slice(None)):
        values = {}
        for area, name in enumerate(params.areaRngLbl):
            p = precision[:, :, category, area, maxdet]
            r = recall[:, category, area, maxdet]
            values[name] = dict(AP=mean(p), AP50=mean(p[i50]), AP75=mean(p[i75]),
                                AR100=mean(r))
        return values

    return {'size_groups': group(),
            'categories': {str(cat): group(index) for index, cat in enumerate(params.catIds)}}


def check_training_ap(cache, model, measured):
    metadata = cache[model]
    log = Path(metadata['checkpoint']).parent / 'log.txt'
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    last = rows[-1]
    epoch = metadata.get('checkpoint_last_epoch')
    if epoch is not None and epoch != last['epoch']:
        raise ValueError(f"{model}: checkpoint epoch {epoch} differs from last log epoch {last['epoch']}")
    reference = last['test_coco_eval_bbox'][0]
    difference = (measured - reference) * 100
    if not np.isfinite(difference) or abs(difference) > .02:
        raise ValueError(f'{model}: cached AP differs from training by {difference:.6f} pp (>0.02)')
    return dict(log=str(log), epoch=last['epoch'], training_AP=reference,
                cached_AP=measured, difference_pp=difference, tolerance_pp=.02)


def run(args):
    from faster_coco_eval import COCO, COCOeval_faster

    started = time.monotonic()
    output = Path(args.output_file)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(args.threads)
    cache = torch.load(args.prediction_cache, map_location='cpu', weights_only=False)
    if cache['schema_version'] != 1:
        raise ValueError('Unsupported prediction-cache schema')
    records = cache['records']
    coco = COCO(str(cache['dataset']['ann_file']))
    ids = [int(r['image_id']) for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(coco.getImgIds()):
        raise ValueError('Prediction cache must contain each validation image exactly once')
    subsets = {'all': records, **{name: [r for r in records if density_bin(len(r['target']['boxes'])) == name]
                                  for name in BINS}}
    summary = {
        'prediction_cache': str(Path(args.prediction_cache).resolve()),
        'dataset': cache['dataset'], 'baseline': cache['baseline'], 'model': cache['model'],
        'units': 'AP and AR in [0,1]; validation differences in percentage points',
        'density_definition': 'canonical non-crowd GT per image: <=30,31-60,61-100,>100',
        'size_definition': 'COCO original-image annotation area; original crowd/ignore handling preserved',
        'max_detections': 100,
        'max_detections_interpretation': 'COCO per-image per-category limit, not total GT per image',
        'categories': coco.loadCats(coco.getCatIds()),
        'subsets': {name: dict(image_count=len(items), canonical_gt_count=sum(len(r['target']['boxes']) for r in items))
                    for name, items in subsets.items()},
        'models': {}, 'training_validation': {},
    }
    for model in ('baseline', 'model'):
        predictions = postprocess_records(records, model)
        detections = coco.loadRes(predictions)
        summary['models'][model] = {}
        for name, items in subsets.items():
            if not items:
                summary['models'][model][name] = None
                continue
            evaluator = COCOeval_faster(coco, detections, 'bbox')
            evaluator.params.imgIds = [int(r['image_id']) for r in items]
            evaluator.params.maxDets = [1, 10, 100]
            evaluator.evaluate()
            evaluator.accumulate()
            summary['models'][model][name] = extract_metrics(evaluator)
            if name == 'all':
                summary['training_validation'][model] = check_training_ap(
                    cache, model, summary['models'][model][name]['size_groups']['all']['AP'])
            print(f"{model}/{name}: {len(items)} images, {time.monotonic()-started:.1f}s", flush=True)
    summary['elapsed_seconds'] = time.monotonic() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction-cache', required=True)
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--threads', type=int, default=4)
    run(parser.parse_args())
