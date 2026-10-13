"""The approved matrix reuses E0 and trains E1-E3 with the old 80+40 protocol."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from src.core.yaml_utils import load_config


GROUPS = ('g5_adaptive_band_gate', 'g5_adaptive_band_pre_reference',
          'g5_adaptive_band_gate_pre_reference')


class GateReferenceSubmissionTests(unittest.TestCase):
    def test_three_success_chained_jobs_exclude_e0_and_use_distinct_two_stage_outputs(self):
        from tools import submit_visdrone_gate_reference as module
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(module.subprocess, 'run', side_effect=[
                    subprocess.CompletedProcess([], 0, f'{n}\n', '') for n in (501, 502, 503)]) as run:
                jobs = module.submit_chain(root / 'source', root)
            self.assertEqual([j['experiment'] for j in jobs], ['E1', 'E2', 'E3'])
            self.assertEqual([j['group'] for j in jobs], list(GROUPS))
            self.assertEqual([j['afterok'] for j in jobs], [None, '501', '502'])
            self.assertEqual(len({j['output_dir'] for j in jobs}), 3)
            self.assertEqual(json.loads((root / 'jobs.json').read_text()), jobs)
            for i, (job, call) in enumerate(zip(jobs, run.call_args_list)):
                command = call.args[0]
                self.assertIn('run_train_visdrone_gate_reference.slurm', command[-6])
                self.assertEqual(command[-1], '0')
                self.assertTrue(job['config80'].endswith(f"visdrone_{job['group']}.yml"))
                self.assertTrue(job['config120'].endswith(f"visdrone_{job['group']}_continue120.yml"))
                self.assertIn('--kill-on-invalid-dep=yes', command)
                if i:
                    self.assertIn(f'--dependency=afterok:{500 + i}', command)
            with self.assertRaises(FileExistsError):
                module.submit_chain(root / 'source', root)
            self.assertEqual(run.call_count, 3)

    def test_configs_only_change_approved_model_switches_and_output_names(self):
        root = Path(__file__).resolve().parents[1] / 'configs/controlled'
        for group, gate, reference in zip(GROUPS, (True, False, True),
                                          ('corrected', 'pre_correction', 'pre_correction')):
            for suffix in ('', '_continue120'):
                with self.subTest(group=group, suffix=suffix):
                    base = load_config(str(root / f'visdrone_g5_adaptive_band{suffix}.yml'), {})
                    modified = load_config(str(root / f'visdrone_{group}{suffix}.yml'), {})
                    expected_model = {**base['DFINEAllLayerStrip'],
                                      'adaptive_correction_gate': gate,
                                      'next_reference_source': reference}
                    self.assertEqual(modified['DFINEAllLayerStrip'], expected_model)
                    for key in set(base) | set(modified):
                        if key not in ('DFINEAllLayerStrip', 'exp_name', 'output_dir', '__include__'):
                            self.assertEqual(base.get(key), modified.get(key), key)
                    self.assertEqual(modified['train_dataloader']['total_batch_size'], 32)
                    self.assertEqual(modified['controlled_seed'], 0)

    def test_submission_failure_preserves_manifest_without_retries(self):
        from tools import submit_visdrone_gate_reference as module
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(module.subprocess, 'run', side_effect=[
                    subprocess.CompletedProcess([], 0, '501\n', ''),
                    subprocess.CalledProcessError(1, 'sbatch')]) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    module.submit_chain(root / 'source', root)
            self.assertEqual(run.call_count, 2)
            jobs = json.loads((root / 'jobs.json').read_text())
            self.assertEqual([j['experiment'] for j in jobs], ['E1'])


if __name__ == '__main__':
    unittest.main()
