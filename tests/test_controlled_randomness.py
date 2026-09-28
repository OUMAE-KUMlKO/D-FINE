import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch.utils.data import TensorDataset

from src.core import YAMLConfig
from src.data import DataLoader
from src.misc import dist_utils


class ControlledRandomnessTests(unittest.TestCase):
    def config(self, directory):
        path = Path(directory) / 'config.yml'
        path.write_text('model: test_model\ncontrolled_seed: 0\n'
                        'train_dataloader: {total_batch_size: 2, shuffle: true}\n')
        return YAMLConfig(str(path))

    def test_model_initialization_is_repeatable_and_leaves_training_rng_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config(directory)
            torch.manual_seed(42)
            before = torch.get_rng_state().clone()
            with mock.patch('src.core.yaml_config.create', side_effect=lambda *a, **k: torch.nn.Linear(3, 2)):
                first = cfg.model
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                torch.rand(137)
                second = self.config(directory).model
            self.assertTrue(torch.equal(first.weight, second.weight))

    def test_loader_order_does_not_depend_on_model_random_draws(self):
        def build(*args, **kwargs):
            return DataLoader(TensorDataset(torch.arange(12)), shuffle=True, **kwargs)

        with tempfile.TemporaryDirectory() as directory, \
             mock.patch('src.core.yaml_config.create', side_effect=build):
            torch.manual_seed(42)
            first = self.config(directory).train_dataloader
            order1 = torch.cat([batch[0] for batch in first])
            torch.rand(137)
            second = self.config(directory).train_dataloader
            order2 = torch.cat([batch[0] for batch in second])
            self.assertTrue(torch.equal(order1, order2))

    def test_distributed_wrapper_preserves_independent_worker_rng(self):
        generator = torch.Generator().manual_seed(27)
        loader = DataLoader(TensorDataset(torch.arange(12)), batch_size=2,
                            generator=generator, num_workers=0)
        with mock.patch.object(dist_utils, 'is_dist_available_and_initialized', return_value=True), \
             mock.patch('torch.distributed.get_world_size', return_value=2), \
             mock.patch('torch.distributed.get_rank', return_value=0):
            wrapped = dist_utils.warp_loader(loader, shuffle=True)
        self.assertIs(wrapped.generator, generator)


if __name__ == '__main__':
    unittest.main()
