"""Stage tracing must describe the original rollback without changing it."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from src.optim import ModelEMA
from src.solver.det_solver import DetSolver


class DDFTrainingTraceTests(unittest.TestCase):
    def test_trace_records_restored_epoch_optimizer_and_ema(self):
        with tempfile.TemporaryDirectory() as directory:
            solver = object.__new__(DetSolver)
            solver.cfg = SimpleNamespace(yaml_cfg={
                "DFINECriterion": {"log_ddf_diagnostics": True},
            })
            solver.output_dir = Path(directory)
            solver.last_epoch = 65
            solver.model = torch.nn.Linear(2, 1)
            solver.optimizer = torch.optim.AdamW(solver.model.parameters(), lr=0.01)
            solver.model(torch.ones(1, 2)).sum().backward()
            solver.optimizer.step()
            solver.ema = ModelEMA(solver.model)
            solver.ema.update(solver.model)
            saved = solver.state_dict()
            torch.save(saved, solver.output_dir / "best_stg1.pth")
            solver.last_epoch = 71
            solver.optimizer.param_groups[0]["lr"] = 0.5
            solver.ema.updates = 999

            event = solver._load_stage_checkpoint(72, "stage_transition")

            self.assertEqual(solver.last_epoch, 65)
            self.assertEqual(solver.optimizer.param_groups[0]["lr"], 0.01)
            self.assertEqual(solver.ema.updates, saved["ema"]["updates"])
            self.assertEqual(event["outer_epoch"], 72)
            self.assertEqual(event["internal_epoch_before"], 71)
            self.assertEqual(event["checkpoint_epoch"], 65)
            self.assertTrue(event["optimizer_restored"])
            self.assertTrue(event["ema_restored"])
            self.assertEqual(event["reason"], "stage_transition")
            recorded = json.loads((solver.output_dir / "stage_reload_events.jsonl").read_text())
            self.assertEqual(recorded, event)


if __name__ == "__main__":
    unittest.main()
