"""Preserve the failing batch and stop all ranks before loss collectives."""

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing

from src.solver.prediction_safety_diagnostics import guarded_prediction_forward
from src.zoo.dfine.prediction_safety import PredictionSafetyError


class FailingModel(torch.nn.Module):
    def __init__(self, fail=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.fail = fail

    def forward(self, samples, targets=None):
        if self.fail:
            raise PredictionSafetyError('invalid prior', {
                'layer_index': 2, 'stage': 'prior',
                'prior_boxes': samples * self.weight,
            })
        return {'pred_boxes': samples * self.weight}


def distributed_failure_worker(rank, store, output):
    torch.set_num_threads(1)
    distributed.init_process_group('gloo', init_method=f'file://{store}',
                                   rank=rank, world_size=2, timeout=timedelta(seconds=15))
    try:
        try:
            guarded_prediction_forward(
                FailingModel(fail=rank == 1), torch.ones(1, 4),
                [{'image_id': torch.tensor([rank])}], output,
                epoch=92, step=103, use_amp=False,
            )
        except PredictionSafetyError:
            Path(output, f'stopped_rank{rank}').touch()
        else:
            raise AssertionError('Every rank must stop')
    finally:
        distributed.destroy_process_group()


class PredictionSafetyDiagnosticsTests(unittest.TestCase):
    def test_training_engine_records_counts_without_changing_loss_api(self):
        from src.solver.det_engine import train_one_epoch
        from src.zoo.dfine.dfine_criterion import DFINECriterion
        from test_all_layer_strip import make_model, matcher, normalized_targets

        torch.set_num_threads(1)
        torch.manual_seed(41)
        model = make_model(adaptive_correction_gate=True)
        criterion = DFINECriterion(
            matcher(), {'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2,
                        'loss_fgl': 0.15, 'loss_ddf': 1.5},
            ['vfl', 'boxes', 'local'], num_classes=2, reg_max=4,
        )
        batch = (torch.randn(1, 3, 16, 24), normalized_targets())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        stats = train_one_epoch(model, criterion, [batch], optimizer,
                                torch.device('cpu'), epoch=0, use_wandb=False,
                                max_norm=0.1, num_visualization_sample_batch=0)
        self.assertTrue(torch.isfinite(torch.tensor(stats['loss'])))
        for layer in range(6):
            self.assertGreater(stats[f'prediction_safety_layer_{layer}_checked_queries'], 0)
            self.assertEqual(stats[f'prediction_safety_layer_{layer}_fallback_queries'], 0)

    def test_success_keeps_graph_and_creates_no_files(self):
        with tempfile.TemporaryDirectory() as directory:
            model = FailingModel(fail=False)
            samples = torch.ones(1, 4)
            output = guarded_prediction_forward(
                model, samples, [], directory, epoch=0, step=0, use_amp=False,
            )
            torch.testing.assert_close(output['pred_boxes'], samples * 2, atol=0, rtol=0)
            output['pred_boxes'].sum().backward()
            self.assertEqual(model.weight.grad.item(), 4)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_failure_saves_cpu_input_diagnostic_model_and_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            samples = torch.ones(1, 4, requires_grad=True)
            targets = [{'image_id': torch.tensor([17]), 'labels': torch.tensor([1])}]
            with self.assertRaisesRegex(PredictionSafetyError, 'invalid prior'):
                guarded_prediction_forward(
                    FailingModel(), samples, targets, directory,
                    epoch=92, step=103, use_amp=False,
                )
            files = list(Path(directory).glob('prediction_failures/*.pth'))
            self.assertEqual(len(files), 1)
            state = torch.load(files[0], map_location='cpu', weights_only=False)
            self.assertEqual((state['epoch'], state['step'], state['rank']), (92, 103, 0))
            self.assertEqual(state['diagnostic']['layer_index'], 2)
            self.assertFalse(state['samples'].requires_grad)
            self.assertEqual(state['targets'][0]['image_id'].item(), 17)
            self.assertEqual(state['model']['weight'].item(), 2)
            self.assertIn('torch_cpu', state['rng_before_forward'])
            self.assertFalse(state['full_training_checkpoint'])

    def test_non_main_failure_stops_both_ranks_and_is_saved_by_failing_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            multiprocessing.spawn(distributed_failure_worker,
                                  args=(str(Path(directory, 'store')), directory),
                                  nprocs=2, join=True)
            self.assertTrue(Path(directory, 'stopped_rank0').exists())
            self.assertTrue(Path(directory, 'stopped_rank1').exists())
            files = list(Path(directory).glob('prediction_failures/*.pth'))
            self.assertEqual(len(files), 1)
            state = torch.load(files[0], map_location='cpu', weights_only=False)
            self.assertEqual(state['rank'], 1)


if __name__ == '__main__':
    unittest.main()
