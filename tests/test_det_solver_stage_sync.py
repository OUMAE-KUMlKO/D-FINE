import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from src.solver.det_solver import DetSolver
import src.solver.det_solver as det_solver_module


class DetSolverStageSyncTests(unittest.TestCase):
    @staticmethod
    def make_solver(events):
        solver = object.__new__(DetSolver)
        solver.cfg = SimpleNamespace(
            epochs=1, clip_max_norm=0.0, print_freq=1,
            yaml_cfg={}, checkpoint_freq=100,
        )
        solver.use_wandb = False
        solver.last_epoch = -1
        solver.output_dir = Path("/tmp/det_solver_stage_sync_test")
        solver.ema = None
        solver.model = solver.criterion = solver.postprocessor = None
        solver.optimizer = solver.device = solver.scaler = None
        solver.lr_warmup_scheduler = solver.lr_scheduler = None
        solver.writer = solver.evaluator = solver.val_dataloader = None
        solver.train_dataloader = SimpleNamespace(
            collate_fn=SimpleNamespace(stop_epoch=0),
            sampler=SimpleNamespace(set_epoch=lambda epoch: None),
            set_epoch=lambda epoch: None,
        )
        solver.load_resume_state = lambda path: events.append(("load", path))
        return solver

    def test_distributed_stage_transition_barriers_before_loading(self):
        events = []
        solver = self.make_solver(events)
        with mock.patch.object(DetSolver, "train"), \
             mock.patch.object(det_solver_module, "stats", return_value=(0, "")), \
             mock.patch.object(
                 det_solver_module.dist_utils,
                 "is_dist_available_and_initialized", return_value=True,
             ), \
             mock.patch.object(
                 det_solver_module.torch.distributed, "barrier",
                 side_effect=lambda: events.append(("barrier", None)),
             ), \
             mock.patch.object(
                 det_solver_module, "train_one_epoch", side_effect=RuntimeError("stop")
             ):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                solver.fit()

        self.assertEqual(events[0][0], "barrier")
        self.assertEqual(events[1], (
            "load", str(solver.output_dir / "best_stg1.pth")
        ))

    def test_single_process_stage_transition_loads_without_barrier(self):
        events = []
        solver = self.make_solver(events)
        with mock.patch.object(DetSolver, "train"), \
             mock.patch.object(det_solver_module, "stats", return_value=(0, "")), \
             mock.patch.object(
                 det_solver_module.dist_utils,
                 "is_dist_available_and_initialized", return_value=False,
             ), \
             mock.patch.object(
                 det_solver_module.torch.distributed, "barrier",
                 side_effect=AssertionError("uninitialized barrier"),
             ), \
             mock.patch.object(
                 det_solver_module, "train_one_epoch", side_effect=RuntimeError("stop")
             ):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                solver.fit()

        self.assertEqual(events, [(
            "load", str(solver.output_dir / "best_stg1.pth")
        )])

    def run_stage_rollbacks(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        solver = self.make_solver([])
        del solver.load_resume_state  # Exercise actual checkpoint loading, including epoch restore.
        solver.output_dir = Path(directory.name)
        solver.cfg.epochs = 5
        solver.cfg.checkpoint_freq = 1
        solver.train_dataloader.collate_fn.stop_epoch = 2
        solver.train_dataloader.collate_fn.ema_restart_decay = 0.99
        solver.model = torch.nn.Linear(1, 1, bias=False)
        solver.ema = torch.nn.Module()
        solver.ema.module = torch.nn.Linear(1, 1, bias=False)
        solver.ema.decay = 0.999
        solver.optimizer = torch.optim.SGD(solver.model.parameters(), lr=0.1, momentum=0.9)
        solver.lr_scheduler = torch.optim.lr_scheduler.StepLR(solver.optimizer, step_size=1)
        evaluated_weights = {}

        def train_epoch(model, criterion, loader, optimizer, device, epoch, **kwargs):
            optimizer.zero_grad()
            model(torch.ones(1, 1)).sum().backward()
            optimizer.step()
            with torch.no_grad():
                model.weight.fill_(epoch + 1)
                solver.ema.module.weight.fill_(epoch + 1.5)
            return {"loss": 1.0}

        def evaluate_epoch(module, criterion, postprocessor, loader, evaluator,
                           device, epoch, *args, **kwargs):
            evaluated_weights[epoch] = module.weight.detach().clone()
            # Best stage1 at epoch 0; rollback at stage transition, epoch 2 and final epoch 4.
            return {"coco_eval_bbox": [[0.9, 0.8, 0.7, 1.0, 0.6][epoch]]}, None

        with mock.patch.object(DetSolver, "train"), \
             mock.patch.object(det_solver_module, "stats", return_value=(1, "")), \
             mock.patch.object(det_solver_module.dist_utils,
                               "is_dist_available_and_initialized", return_value=False), \
             mock.patch.object(det_solver_module.dist_utils,
                               "is_main_process", return_value=True), \
             mock.patch.object(det_solver_module, "train_one_epoch", side_effect=train_epoch), \
             mock.patch.object(det_solver_module, "evaluate", side_effect=evaluate_epoch):
            solver.fit()
        return solver, evaluated_weights

    def test_best_stage2_checkpoint_uses_outer_epoch_after_rollbacks(self):
        solver, _ = self.run_stage_rollbacks()
        checkpoint = torch.load(solver.output_dir / "best_stg2.pth", weights_only=False)
        self.assertEqual(checkpoint["last_epoch"], 3)
        self.assertEqual(checkpoint["model"]["weight"].item(), 4.0)

    def test_last_checkpoint_matches_final_evaluation_before_rollback(self):
        solver, evaluated_weights = self.run_stage_rollbacks()
        checkpoint = torch.load(solver.output_dir / "last.pth", weights_only=False)
        self.assertEqual(checkpoint["last_epoch"], 4)
        self.assertEqual(checkpoint["model"]["weight"].item(), 5.0)
        torch.testing.assert_close(checkpoint["ema"]["module.weight"], evaluated_weights[4])

    def test_final_rollback_preserves_epoch_and_existing_training_state_policy(self):
        solver, _ = self.run_stage_rollbacks()
        self.assertEqual(solver.last_epoch, 4)
        # Rollback still restores model, EMA, optimizer and scheduler from stage1.
        self.assertEqual(solver.model.weight.item(), 1.0)
        self.assertEqual(solver.ema.module.weight.item(), 1.5)
        self.assertAlmostEqual(solver.optimizer.param_groups[0]["lr"], 0.01)
        self.assertEqual(solver.lr_scheduler.last_epoch, 1)


if __name__ == "__main__":
    unittest.main()
