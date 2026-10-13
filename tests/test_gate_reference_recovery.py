"""Recovery submission for the interrupted E1-E3 gate/reference matrix."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import torch


class GateReferenceRecoveryTests(unittest.TestCase):
    def test_resume_mode_chains_e1_resume_then_e2_e3_fresh(self):
        from tools import submit_visdrone_gate_reference_recovery as module
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(module.subprocess, "run", side_effect=[
                    subprocess.CompletedProcess([], 0, f"{job}\n", "")
                    for job in (701, 702, 703)]) as run:
                jobs = module.submit_chain(root / "source", root, "resume")
            self.assertEqual([job["experiment"] for job in jobs], ["E1", "E2", "E3"])
            self.assertEqual([job["mode"] for job in jobs], ["resume", "fresh", "fresh"])
            self.assertEqual([job["afterok"] for job in jobs], [None, "701", "702"])
            first = run.call_args_list[0].args[0]
            self.assertIn("run_resume_visdrone_gate_reference.slurm", first[-7])
            self.assertIn(str(module.RESUME_CHECKPOINT), first)
            self.assertIn(str(module.HISTORICAL_BEST), first)
            for call in run.call_args_list[1:]:
                command = call.args[0]
                self.assertIn("run_train_visdrone_gate_reference.slurm", command[-6])
                self.assertIn("--dependency=afterok:", " ".join(command))
            manifest = json.loads((root / "jobs.json").read_text())
            self.assertEqual(manifest["status"], "submitted")
            self.assertFalse(manifest["automatic_retry"])
            self.assertEqual(manifest["jobs"], jobs)

    def test_fresh_mode_chains_all_three_fresh_jobs(self):
        from tools import submit_visdrone_gate_reference_recovery as module
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(module.subprocess, "run", side_effect=[
                    subprocess.CompletedProcess([], 0, f"{job}\n", "")
                    for job in (801, 802, 803)]) as run:
                jobs = module.submit_chain(root / "source", root, "fresh")
            self.assertEqual([job["mode"] for job in jobs], ["fresh"] * 3)
            self.assertTrue(all(
                "run_train_visdrone_gate_reference.slurm" in call.args[0][-6]
                for call in run.call_args_list
            ))
            self.assertEqual([job["afterok"] for job in jobs], [None, "801", "802"])

    def test_resume_checkpoint_requires_epoch91_and_complete_training_state(self):
        from tools import submit_visdrone_gate_reference_recovery as module
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pth"
            state = {key: {} for key in module.REQUIRED_RESUME_KEYS}
            state["last_epoch"] = 91
            torch.save(state, path)
            summary = module.validate_checkpoint(path, expected_epoch=91)
            self.assertEqual(summary["last_epoch"], 91)
            self.assertEqual(summary["required_keys"], list(module.REQUIRED_RESUME_KEYS))

            del state["scaler"]
            torch.save(state, path)
            with self.assertRaisesRegex(RuntimeError, "scaler"):
                module.validate_checkpoint(path, expected_epoch=91)
            state["scaler"] = {}
            state["last_epoch"] = 90
            torch.save(state, path)
            with self.assertRaisesRegex(RuntimeError, "epoch91"):
                module.validate_checkpoint(path, expected_epoch=91)

    def test_resume_slurm_archives_true_best_and_labels_segment_selection(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "slurm_files/run_resume_visdrone_gate_reference.slurm").read_text()
        self.assertIn("archive/pre_resume_best_stg2.pth", text)
        self.assertIn("best_overall_stg2.pth", text)
        self.assertIn("last_epoch'] != 91", text)
        self.assertIn("last_epoch'] != 89", text)
        for key in ("model", "optimizer", "ema", "scaler",
                    "lr_scheduler", "lr_warmup_scheduler"):
            self.assertIn(key, text)
        self.assertIn("resume_segment_selection", text)
        self.assertIn("historical_true_best", text)
        self.assertIn("selected_checkpoint", text)
        self.assertIn("test_coco_eval_bbox", text)
        self.assertIn("no automatic retry", text)
        self.assertNotIn("exec bash slurm_files/run_resume_dfine_controlled.slurm", text)

    def test_submission_failure_is_recorded_immediately_without_retry(self):
        from tools import submit_visdrone_gate_reference_recovery as module
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(module.subprocess, "run", side_effect=[
                    subprocess.CompletedProcess([], 0, "901\n", ""),
                    subprocess.CalledProcessError(1, "sbatch")]) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    module.submit_chain(root / "source", root, "resume")
            self.assertEqual(run.call_count, 2)
            manifest = json.loads((root / "jobs.json").read_text())
            self.assertEqual(manifest["status"], "submission_failed")
            self.assertEqual([job["experiment"] for job in manifest["jobs"]], ["E1"])
            self.assertEqual(manifest["failed_experiment"], "E2")
            self.assertFalse(manifest["automatic_retry"])
            with self.assertRaises(FileExistsError):
                module.submit_chain(root / "source", root, "resume")


if __name__ == "__main__":
    unittest.main()
