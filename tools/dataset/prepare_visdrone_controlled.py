#!/usr/bin/env python3
"""Prepare shared VisDrone annotations for LOCAL COCO bbox evaluation.

Native classes 1..10 map to COCO 0..9 (0 remains pedestrian). Native
ignored regions (0) and others (11) become iscrowd=1 boxes in all ten
classes. Unscored valid-class objects become crowd boxes in their own
class. CocoDetection excludes crowd targets during training; COCO bbox
matching suppresses detections overlapping crowds under its own IoA rules.
Nonpositive-width/height boxes are removed from training and evaluation alike.
This is deliberately NOT the official VisDrone evaluation protocol.
"""
import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path

CATEGORY_NAMES = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
                  'tricycle', 'awning-tricycle', 'bus', 'motor')
PROTOCOL = 'visdrone-local-coco-crowd-v1'


def prepare_split(root, split):
    root = Path(root)
    source_path = root / f'instances_UAV{split}_v1.json'
    data = json.loads(source_path.read_text())
    categories = sorted(data['categories'], key=lambda item: item['id'])
    if [(c['id'], c['name']) for c in categories] != list(enumerate(CATEGORY_NAMES)):
        raise ValueError('Expected categories 0..9, with 0=pedestrian')
    images = {image['id']: image for image in data['images']}
    if len(images) != len(data['images']):
        raise ValueError('Duplicate image IDs')
    records = defaultdict(deque)
    ids = set()
    for annotation in data['annotations']:
        if annotation['id'] in ids:
            raise ValueError('Duplicate source annotation ID')
        ids.add(annotation['id'])
        if annotation['image_id'] not in images or annotation['category_id'] not in range(10):
            raise ValueError('Invalid source image/category reference')
        records[(annotation['image_id'], annotation['category_id'], tuple(annotation['bbox']))].append(annotation)
    next_id = max(ids, default=-1) + 1
    additions = []
    native_counts = Counter()
    annotation_dir = root / f'VisDrone2019-DET-{split}' / 'annotations'
    expected_files = {Path(image['file_name']).with_suffix('.txt').name for image in images.values()}
    if {path.name for path in annotation_dir.glob('*.txt')} != expected_files:
        raise ValueError('Native TXT coverage differs from source image list')
    for image in data['images']:
        image_path = root / f'VisDrone2019-DET-{split}' / 'images' / image['file_name']
        if not image_path.is_file():
            raise ValueError(f'Missing image: {image_path}')
        native_path = annotation_dir / Path(image['file_name']).with_suffix('.txt').name
        for line_number, line in enumerate(native_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            values = [int(value) for value in line.strip().rstrip(',').split(',')]
            if len(values) != 8:
                raise ValueError(f'Expected 8 fields: {native_path}:{line_number}')
            x, y, width, height, score, category, _, _ = values
            if category not in range(12) or score not in (0, 1):
                raise ValueError(f'Invalid native annotation: {native_path}:{line_number}')
            native_counts[f'category_{category}_score_{score}'] += 1
            box = [x, y, width, height]
            if 1 <= category <= 10:
                key = (image['id'], category - 1, tuple(box))
                if not records[key]:
                    raise ValueError(f'Native object missing from source JSON: {native_path}:{line_number}')
                annotation = records[key].popleft()
                annotation['iscrowd'] = int(score == 0)
                annotation['ignore'] = int(score == 0)
            else:
                for category_id in range(10):
                    additions.append({'id': next_id, 'image_id': image['id'],
                                      'category_id': category_id, 'bbox': box,
                                      'area': width * height, 'iscrowd': 1, 'ignore': 1})
                    next_id += 1
    if any(records.values()):
        raise ValueError('Source JSON contains objects absent from native TXT')
    source_count = len(data['annotations'])
    data['annotations'].extend(additions)
    before_filter = len(data['annotations'])
    data['annotations'] = [a for a in data['annotations'] if a['bbox'][2] > 0 and a['bbox'][3] > 0]
    removed_degenerate = before_filter - len(data['annotations'])
    data['info'] = {'description': PROTOCOL, 'evaluation': 'Local COCO bbox, maxDets=100; not official VisDrone',
                    'ignore_policy': 'Native category0/11 -> crowd for every class; valid score0 -> same-class crowd'}
    crowd_count = sum(annotation['iscrowd'] for annotation in data['annotations'])
    return data, {'images': len(images), 'source_annotations': source_count,
                  'positive_annotations': len(data['annotations']) - crowd_count,
                  'crowd_annotations': crowd_count, 'added_crowd_annotations': sum(a['bbox'][2] > 0 and a['bbox'][3] > 0 for a in additions),
                  'removed_degenerate_annotations': removed_degenerate,
                  'missing_images': 0, 'native_txt_files': len(expected_files),
                  'native_counts': dict(sorted(native_counts.items())),
                  'source_json': str(source_path), 'native_annotations': str(annotation_dir)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/zzy/datasets/visdrone'))
    parser.add_argument('--output', type=Path, default=Path('output/visdrone_controlled'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'protocol': PROTOCOL, 'maxDets': 100, 'splits': {}}
    for split in ('train', 'val'):
        data, summary = prepare_split(args.root, split)
        path = args.output / f'instances_{split}.json'
        path.write_text(json.dumps(data, separators=(',', ':')))
        summary['prepared_json'] = str(path)
        manifest['splits'][split] = summary
        print(f'{split}: {summary["images"]} images, {summary["positive_annotations"]} positives, '
              f'{summary["crowd_annotations"]} crowd boxes, 0 missing images', flush=True)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
