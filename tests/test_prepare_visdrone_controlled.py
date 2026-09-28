import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'tools/dataset/prepare_visdrone_controlled.py'


class PrepareVisDroneTests(unittest.TestCase):
    def test_native_ignore_protocol_preserves_pedestrian_and_ids(self):
        self.assertTrue(SCRIPT.exists(), 'conversion utility must exist')
        spec = importlib.util.spec_from_file_location('prepare_visdrone', SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            split = root / 'VisDrone2019-DET-val'
            (split / 'images').mkdir(parents=True)
            (split / 'annotations').mkdir()
            (split / 'images/a.jpg').touch()
            (split / 'annotations/a.txt').write_text(
                '1,2,3,4,1,1,0,0\n5,6,7,8,0,4,0,0\n'
                '10,20,30,40,0,0,0,0\n11,21,31,41,1,11,0,0\n12,22,3,0,1,4,0,0\n')
            source = {'images': [{'id': 7, 'file_name': 'a.jpg', 'width': 100, 'height': 100}],
                      'categories': [{'id': i, 'name': name} for i, name in enumerate(module.CATEGORY_NAMES)],
                      'annotations': [
                          {'id': 12, 'image_id': 7, 'category_id': 0, 'bbox': [1, 2, 3, 4], 'area': 12, 'iscrowd': 0},
                          {'id': 13, 'image_id': 7, 'category_id': 3, 'bbox': [5, 6, 7, 8], 'area': 56, 'iscrowd': 0},
                          {'id': 14, 'image_id': 7, 'category_id': 3, 'bbox': [12, 22, 3, 0], 'area': 0, 'iscrowd': 0}]}
            (root / 'instances_UAVval_v1.json').write_text(json.dumps(source))
            converted, summary = module.prepare_split(root, 'val')
            self.assertEqual(converted['images'], source['images'])
            self.assertEqual(converted['annotations'][0]['id'], 12)
            self.assertEqual(converted['annotations'][0]['category_id'], 0)
            self.assertEqual(converted['annotations'][0]['iscrowd'], 0)
            self.assertEqual(converted['annotations'][1]['id'], 13)
            self.assertEqual(converted['annotations'][1]['iscrowd'], 1)
            self.assertEqual(len(converted['annotations']), 22)
            self.assertEqual(summary['positive_annotations'], 1)
            self.assertEqual(summary['crowd_annotations'], 21)
            self.assertEqual(summary['missing_images'], 0)
            self.assertEqual(summary['removed_degenerate_annotations'], 1)
            self.assertEqual({a['category_id'] for a in converted['annotations'][2:12]}, set(range(10)))
            (split / 'images/a.jpg').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing image'):
                module.prepare_split(root, 'val')


if __name__ == '__main__':
    unittest.main()
