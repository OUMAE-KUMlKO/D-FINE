"""The oracle preflight must detect silent changes to its frozen baseline."""

from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from tools.analysis import check_visdrone_gt_oracle as preflight
from tools.analysis.check_visdrone_gt_oracle import (
    assert_frozen_baseline,
    assert_initial_identity,
    snapshot_baseline,
)


class VisDroneGTOraclePreflightTests(unittest.TestCase):
    def test_cli_model_defaults_and_candidate_selection(self):
        arguments = ['preflight', '--config', 'config.yml', '--baseline-checkpoint',
                     'baseline.pth', '--output-file', 'summary.json']
        for extra, expected in (
            ([], 'DFINEGTRefiner'),
            (['--model', 'DFINEGTCandidateRefiner'], 'DFINEGTCandidateRefiner'),
        ):
            with self.subTest(model=expected), patch('sys.argv', arguments + extra):
                with patch.object(preflight, 'run') as run:
                    preflight.main()
                self.assertEqual(run.call_args.args[0].model, expected)

    def test_selected_model_config_key_and_exact_type(self):
        from src.core import YAMLConfig
        from src.core.yaml_utils import load_config as load_yaml
        from src.zoo.dfine import DFINEGTCandidateRefiner, DFINEGTRefiner
        from test_gt_candidate_refinement import make_model

        old_threads = torch.get_num_threads()
        try:
            for selected in (DFINEGTRefiner, DFINEGTCandidateRefiner):
                for actual in (DFINEGTRefiner, DFINEGTCandidateRefiner):
                    with self.subTest(selected=selected.__name__, actual=actual.__name__):
                        model = make_model(model_class=actual)
                        loaded = []

                        def load_config(path, **overrides):
                            # Keep the loader's mutable default dictionary out of this test.
                            with patch('src.core.yaml_config.load_config',
                                       side_effect=lambda path: load_yaml(path, {})):
                                config = YAMLConfig(path, **overrides)
                            loaded.append(config)
                            config._model = model
                            config._criterion = nn.Identity()
                            config._val_dataloader = []
                            return config

                        with tempfile.TemporaryDirectory() as folder:
                            config_path = Path(folder) / 'config.yml'
                            config_path.write_text(f'model: {actual.__name__}\n')
                            args = Namespace(
                                config=str(config_path), model=selected.__name__,
                                baseline_checkpoint='baseline.pth',
                                output_file=str(Path(folder) / 'summary.json'),
                                device='cpu', batch_size=1, workers=0, threads=1,
                            )
                            with patch('src.core.YAMLConfig', side_effect=load_config), \
                                    patch('builtins.print'):
                                if actual is selected:
                                    # An accepted model reaches the deliberately empty loader.
                                    with self.assertRaises(StopIteration):
                                        preflight.run(args)
                                else:
                                    with self.assertRaisesRegex(ValueError, selected.__name__):
                                        preflight.run(args)
                            self.assertEqual(
                                loaded[0].yaml_cfg[selected.__name__]['baseline_checkpoint'],
                                args.baseline_checkpoint,
                            )
                            other = (DFINEGTRefiner if selected is DFINEGTCandidateRefiner
                                     else DFINEGTCandidateRefiner)
                            self.assertNotIn(other.__name__, loaded[0].yaml_cfg)
        finally:
            torch.set_num_threads(old_threads)

    def test_frozen_parameter_and_buffer_changes_are_detected(self):
        model = nn.Module()
        model.backbone = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        model.encoder = nn.Identity()
        model.decoder = nn.Identity()
        model.matcher = nn.Identity()
        model.refiner = nn.Linear(2, 2)
        model.backbone.requires_grad_(False)
        for name in ('backbone', 'encoder', 'decoder', 'matcher'):
            getattr(model, name).eval()
        before = snapshot_baseline(model)
        assert_frozen_baseline(model, before)
        with torch.no_grad():
            model.backbone[0].weight.add_(1)
        with self.assertRaisesRegex(AssertionError, 'backbone.0.weight'):
            assert_frozen_baseline(model, before)
        with torch.no_grad():
            model.backbone[0].weight.copy_(before['backbone.0.weight'])
            model.backbone[1].running_mean.add_(1)
        with self.assertRaisesRegex(AssertionError, 'backbone.1.running_mean'):
            assert_frozen_baseline(model, before)

    def test_initial_identity_detects_changed_boxes_and_scores(self):
        native = {
            'pred_boxes': torch.tensor([[[.5, .5, .2, .2]]]),
            'pred_logits': torch.tensor([[[.1, .2]]]),
        }
        output = {name: value.clone() for name, value in native.items()}
        output['baseline_boxes'] = native['pred_boxes'].clone()
        assert_initial_identity(output, native)
        output['pred_boxes'][0, 0, 0] += .01
        with self.assertRaisesRegex(AssertionError, 'zero initialization'):
            assert_initial_identity(output, native)
        output['pred_boxes'] = native['pred_boxes'].clone()
        output['pred_logits'][0, 0, 0] += .01
        with self.assertRaisesRegex(AssertionError, 'classification'):
            assert_initial_identity(output, native)


if __name__ == '__main__':
    unittest.main()
