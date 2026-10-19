import unittest

import torch

from src.solver.det_engine import train_one_epoch


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, samples, targets=None):
        return {
            "pred_boxes": torch.ones(1, 1, 4, device=samples.device) * self.weight,
            "value": self.weight.square(),
        }


class ToyCriterion(torch.nn.Module):
    def forward(self, outputs, targets, **kwargs):
        return {"loss_toy": outputs["value"]}


class FakeGradScaler:
    def __init__(self, skipped):
        self.skipped = iter(skipped)
        self.scale_value = 8.0
        self.current_skip = False

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        self.current_skip = next(self.skipped)
        if not self.current_skip:
            optimizer.step()

    def update(self):
        if self.current_skip:
            self.scale_value /= 2

    def get_scale(self):
        return self.scale_value


class NoScaleAccessScaler(FakeGradScaler):
    def get_scale(self):
        raise AssertionError("get_scale must remain opt-in")


class OptimizerUpdateLoggingTests(unittest.TestCase):
    def test_opt_in_counts_successful_and_amp_skipped_steps(self):
        model = ToyModel()
        stats = train_one_epoch(
            model=model,
            criterion=ToyCriterion(),
            data_loader=[(torch.ones(1), [{}]), (torch.ones(1), [{}])],
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            device=torch.device("cpu"),
            epoch=0,
            use_wandb=False,
            scaler=FakeGradScaler([False, True]),
            log_optimizer_updates=True,
            print_freq=10,
        )
        self.assertEqual(stats["optimizer_updates"], 1)
        self.assertEqual(stats["optimizer_skipped_steps"], 1)

    def test_opt_in_counts_cpu_optimizer_steps(self):
        model = ToyModel()
        stats = train_one_epoch(
            model=model,
            criterion=ToyCriterion(),
            data_loader=[(torch.ones(1), [{}]), (torch.ones(1), [{}])],
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            device=torch.device("cpu"),
            epoch=0,
            use_wandb=False,
            log_optimizer_updates=True,
            print_freq=10,
        )
        self.assertEqual(stats["optimizer_updates"], 2)
        self.assertEqual(stats["optimizer_skipped_steps"], 0)

    def test_optimizer_counts_are_absent_by_default(self):
        model = ToyModel()
        stats = train_one_epoch(
            model=model,
            criterion=ToyCriterion(),
            data_loader=[(torch.ones(1), [{}])],
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            device=torch.device("cpu"),
            epoch=0,
            use_wandb=False,
            print_freq=10,
        )
        self.assertNotIn("optimizer_updates", stats)
        self.assertNotIn("optimizer_skipped_steps", stats)

    def test_default_amp_path_does_not_read_scaler_scale(self):
        model = ToyModel()
        stats = train_one_epoch(
            model=model,
            criterion=ToyCriterion(),
            data_loader=[(torch.ones(1), [{}])],
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            device=torch.device("cpu"),
            epoch=0,
            use_wandb=False,
            scaler=NoScaleAccessScaler([False]),
            print_freq=10,
        )
        self.assertNotIn("optimizer_updates", stats)
        self.assertNotIn("optimizer_skipped_steps", stats)


if __name__ == "__main__":
    unittest.main()
