import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch.utils.data import TensorDataset

from src.core import YAMLConfig
from src.core.yaml_utils import load_config
from src.data import DataLoader
from src.misc import dist_utils


class ControlledRandomnessTests(unittest.TestCase):
    def config(self, directory, seed=0):
        path = Path(directory) / 'config.yml'
        seed_config = '' if seed is None else f'controlled_seed: {seed}\n'
        path.write_text('model: test_model\n' + seed_config +
                        'train_dataloader: {total_batch_size: 2, shuffle: true}\n')
        # Each independent run starts with a fresh config dictionary.
        with mock.patch('src.core.yaml_config.load_config', side_effect=lambda p: load_config(p, {})):
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
        self.assertEqual(wrapped.sampler.seed, 0)

    def test_model_seeds_pair_common_parameters_across_variants(self):
        weights = []
        with tempfile.TemporaryDirectory() as directory:
            for seed in (0, 1, 2):
                variants = []
                for extra_head in (False, True):
                    def build(*args, **kwargs):
                        model = torch.nn.Module()
                        model.common = torch.nn.Linear(3, 2)
                        if extra_head:
                            model.head = torch.nn.Linear(2, 2)
                        return model

                    before = torch.get_rng_state().clone()
                    with mock.patch('src.core.yaml_config.create', side_effect=build):
                        variants.append(self.config(directory, seed).model.common.weight)
                    self.assertTrue(torch.equal(before, torch.get_rng_state()))
                self.assertTrue(torch.equal(*variants))
                weights.append(variants[0])
        self.assertFalse(torch.equal(weights[0], weights[1]))
        self.assertFalse(torch.equal(weights[1], weights[2]))

    def test_distributed_sampling_uses_controlled_seed_without_rank_offset(self):
        def build(*args, **kwargs):
            return DataLoader(TensorDataset(torch.arange(40)), shuffle=True, **kwargs)

        orders = {}
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch('src.core.yaml_config.create', side_effect=build), \
             mock.patch.object(dist_utils, 'is_dist_available_and_initialized', return_value=True), \
             mock.patch('torch.distributed.get_world_size', return_value=2):
            for seed in (0, 1, 2):
                ranks = []
                for rank in (0, 1):
                    with mock.patch('torch.distributed.get_rank', return_value=rank):
                        first = self.config(directory, seed).train_dataloader
                        wrapped = dist_utils.warp_loader(first, shuffle=True)
                        self.assertEqual(wrapped.sampler.seed, seed)
                        self.assertEqual(wrapped.generator.initial_seed(), seed + rank)
                        wrapped.sampler.set_epoch(7)
                        order = list(wrapped.sampler)
                        torch.rand(137)  # Extra model draws must not affect the paired data.
                        paired = dist_utils.warp_loader(
                            self.config(directory, seed).train_dataloader, shuffle=True)
                        paired.sampler.set_epoch(7)
                        self.assertEqual(order, list(paired.sampler))
                        ranks.append(order)
                self.assertEqual(set(ranks[0]) | set(ranks[1]), set(range(40)))
                self.assertFalse(set(ranks[0]) & set(ranks[1]))
                orders[seed] = ranks
        self.assertNotEqual(orders[0], orders[1])
        self.assertNotEqual(orders[1], orders[2])

    def test_uncontrolled_model_and_loader_keep_global_rng_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            before = torch.get_rng_state().clone()
            with mock.patch('src.core.yaml_config.create', side_effect=lambda *a, **k: torch.nn.Linear(3, 2)):
                self.config(directory, None).model
            self.assertFalse(torch.equal(before, torch.get_rng_state()))
            with mock.patch('src.core.yaml_config.create', side_effect=lambda *a, **k:
                            DataLoader(TensorDataset(torch.arange(12)), **k)):
                loader = self.config(directory, None).train_dataloader
            self.assertIsNone(loader.generator)
            self.assertFalse(hasattr(loader, 'controlled_sampler_seed'))


if __name__ == '__main__':
    unittest.main()
