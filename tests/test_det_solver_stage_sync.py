import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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


if __name__ == "__main__":
    unittest.main()
