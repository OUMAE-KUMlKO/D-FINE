import copy
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from src.optim.ema import ModelEMA
from src.optim.warmup import LinearWarmup
from src.solver.det_solver import DetSolver
import src.solver.det_solver as det_solver_module


BASE_LRS = [1.25e-5, 2.5e-4, 2.5e-4]


class ResumeScheduleTests(unittest.TestCase):
    def make_solver(self, last_epoch=79, scheduler_epoch=78, decayed=False):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        checkpoint_path = Path(directory.name) / "last.pth"

        def components():
            model = torch.nn.ParameterList([
                torch.nn.Parameter(torch.ones(1)) for _ in BASE_LRS
            ])
            optimizer = torch.optim.AdamW([
                {"params": [parameter], "lr": lr}
                for parameter, lr in zip(model, BASE_LRS)
            ])
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[100], gamma=0.1
            )
            warmup = LinearWarmup(scheduler, warmup_duration=500, last_step=16159)
            # Construct real scaler state without requiring CUDA allocations.
            with mock.patch("torch.cuda.is_available", return_value=True):
                scaler = torch.cuda.amp.GradScaler()
            return model, optimizer, scheduler, warmup, ModelEMA(model), scaler

        model, optimizer, scheduler, warmup, ema, scaler = components()
        sum(parameter.square().sum() for parameter in model).backward()
        optimizer.step()
        scaler_state = scaler.state_dict()
        scaler_state.update(scale=4096.0, _growth_tracker=23)
        scaler.load_state_dict(scaler_state)
        ema.update(model)
        scheduler.last_epoch = scheduler_epoch
        # The original epoch79 checkpoint uses a different schedule and is two
        # epochs behind the next actual training epoch.
        scheduler.milestones = Counter([100] if decayed else [500])
        scheduler.gamma = 0.1 if decayed else 0.2
        for group, base_lr in zip(optimizer.param_groups, BASE_LRS):
            group["lr"] = base_lr * (0.1 if decayed else 1.0)
        scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
        checkpoint = {
            "last_epoch": last_epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "lr_warmup_scheduler": warmup.state_dict(),
            "ema": ema.state_dict(),
            "scaler": scaler.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)

        solver = object.__new__(DetSolver)
        (solver.model, solver.optimizer, solver.lr_scheduler,
         solver.lr_warmup_scheduler, solver.ema, solver.scaler) = components()
        solver.cfg = SimpleNamespace(
            resume=str(checkpoint_path), epochs=last_epoch + 1, clip_max_norm=0.0,
            print_freq=1, checkpoint_freq=1,
            yaml_cfg={
                "resume_lr_schedule": True,
                "stage2_rollback": False,
                "lr_scheduler": {"type": "MultiStepLR", "milestones": [100], "gamma": 0.1},
            },
        )
        solver.last_epoch = -1
        solver.use_wandb = False
        solver.output_dir = Path(directory.name) / "output"
        solver.output_dir.mkdir()
        solver.device = "cpu"
        solver.criterion = solver.postprocessor = solver.writer = solver.evaluator = None
        solver.val_dataloader = None
        solver.train_dataloader = SimpleNamespace(
            collate_fn=SimpleNamespace(stop_epoch=72), set_epoch=lambda epoch: None,
        )
        return solver, checkpoint

    def fit(self, solver, train_epoch=None):
        def evaluate_epoch(module, criterion, postprocessor, loader, evaluator,
                           device, epoch, *args, **kwargs):
            return {"coco_eval_bbox": [0.8 + epoch * 0.0001]}, None

        with mock.patch.object(
            DetSolver, "train", side_effect=lambda: solver.load_resume_state(solver.cfg.resume)
        ), mock.patch.object(det_solver_module, "stats", return_value=(3, "")), \
             mock.patch.object(det_solver_module.dist_utils,
                               "is_dist_available_and_initialized", return_value=False), \
             mock.patch.object(det_solver_module, "evaluate",
                               side_effect=evaluate_epoch), \
             mock.patch.object(det_solver_module, "train_one_epoch", side_effect=train_epoch):
            solver.fit()

    def test_epoch79_resume_uses_base_lrs_until_epoch100(self):
        solver, _ = self.make_solver()
        solver.cfg.epochs = 101
        observed_lrs = {}

        def train_epoch(model, criterion, loader, optimizer, device, epoch, **kwargs):
            observed_lrs[epoch] = [group["lr"] for group in optimizer.param_groups]
            optimizer.zero_grad()
            sum(parameter.square().sum() for parameter in model).backward()
            optimizer.step()
            return {"loss": 1.0}

        self.fit(solver, train_epoch)
        self.assertEqual(list(observed_lrs), list(range(80, 101)))
        for epoch in range(80, 100):
            self.assertEqual(observed_lrs[epoch], BASE_LRS)
        self.assertEqual(observed_lrs[100], [lr * 0.1 for lr in BASE_LRS])
        self.assertEqual(solver.lr_scheduler.last_epoch, 101)
        self.assertEqual(solver.lr_scheduler.milestones, Counter([100]))
        self.assertEqual(solver.lr_scheduler.gamma, 0.1)

    def test_epoch99_and_100_resume_do_not_decay_twice(self):
        for epoch in (99, 100):
            for decayed in (False, True):
                with self.subTest(epoch=epoch, decayed=decayed):
                    solver, _ = self.make_solver(
                        last_epoch=epoch, scheduler_epoch=epoch + 1, decayed=decayed
                    )
                    self.fit(solver)
                    self.assertEqual(solver.lr_scheduler.last_epoch, epoch + 1)
                    self.assertEqual(solver.lr_scheduler.get_last_lr(), [lr * 0.1 for lr in BASE_LRS])
                    self.assertEqual(
                        [group["lr"] for group in solver.optimizer.param_groups],
                        [lr * 0.1 for lr in BASE_LRS],
                    )

    def test_resume_preserves_optimizer_ema_scaler_and_warmup(self):
        solver, checkpoint = self.make_solver()
        scheduler = solver.lr_scheduler
        self.fit(solver)
        self.assertEqual(solver.lr_scheduler.last_epoch, 80)
        self.assertIs(solver.lr_scheduler, scheduler)
        self.assertIs(solver.lr_warmup_scheduler.lr_scheduler, scheduler)
        torch.testing.assert_close(solver.optimizer.state_dict(), checkpoint["optimizer"])
        torch.testing.assert_close(solver.model.state_dict(), checkpoint["model"])
        torch.testing.assert_close(solver.ema.state_dict(), checkpoint["ema"])
        self.assertEqual(solver.scaler.state_dict(), checkpoint["scaler"])
        self.assertEqual(solver.lr_warmup_scheduler.state_dict(), checkpoint["lr_warmup_scheduler"])

    def test_default_resume_retains_checkpoint_schedule(self):
        solver, checkpoint = self.make_solver()
        del solver.cfg.yaml_cfg["resume_lr_schedule"]
        self.fit(solver)
        self.assertEqual(solver.lr_scheduler.state_dict(), checkpoint["lr_scheduler"])
        torch.testing.assert_close(solver.optimizer.state_dict(), checkpoint["optimizer"])

    def test_schedule_override_requires_resume(self):
        solver, _ = self.make_solver()
        solver.cfg.resume = None
        solver.cfg.epochs = 0
        with mock.patch.object(solver, "train"), \
             mock.patch.object(det_solver_module, "stats", return_value=(3, "")):
            with self.assertRaisesRegex(ValueError, "resume"):
                solver.fit()

    def test_schedule_override_requires_completed_warmup(self):
        solver, checkpoint = self.make_solver()
        checkpoint = copy.deepcopy(checkpoint)
        checkpoint["lr_warmup_scheduler"]["last_step"] = 10
        torch.save(checkpoint, solver.cfg.resume)
        with self.assertRaisesRegex(ValueError, "warmup"):
            self.fit(solver)

    def test_schedule_override_requires_multistep_scheduler(self):
        solver, _ = self.make_solver()
        solver.cfg.yaml_cfg["lr_scheduler"] = {"type": "StepLR", "step_size": 10}
        with self.assertRaisesRegex(ValueError, "MultiStepLR"):
            self.fit(solver)


if __name__ == "__main__":
    unittest.main()
