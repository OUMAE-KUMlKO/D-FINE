"""Submit exactly the three approved resume jobs, without replacing old runs."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.core.yaml_utils import load_config


class ContinuationSubmissionTests(unittest.TestCase):
    def test_three_jobs_resume_distinct_checkpoints_in_success_dependency_order(self):
        from tools import submit_visdrone_continuation as module
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / 'continued'
            run_root.mkdir()
            with mock.patch.object(module.subprocess, 'run', side_effect=[
                subprocess.CompletedProcess([], 0, f'{job}\n', '') for job in (401, 402, 403)
            ]) as run:
                jobs = module.submit_chain(root, run_root / 'source', run_root)
            self.assertEqual([j['group'] for j in jobs],
                             ['g0', 'g5_adaptive_band', 'g5_adaptive_band_first2'])
            self.assertEqual([j['afterok'] for j in jobs], [None, '401', '402'])
            self.assertEqual(len({j['resume'] for j in jobs}), 3)
            self.assertEqual(len({j['output_dir'] for j in jobs}), 3)
            self.assertEqual(json.loads((run_root / 'jobs.json').read_text()), jobs)
            for i, (job, call) in enumerate(zip(jobs, run.call_args_list)):
                command = call.args[0]
                self.assertEqual(job['seed'], 0)
                self.assertTrue(job['resume'].endswith('/last.pth'))
                self.assertTrue(job['config'].endswith(f"visdrone_{job['group']}_continue120.yml"))
                self.assertIn(str(run_root / 'results'), job['output_dir'])
                self.assertIn(job['resume'], command)
                self.assertIn('--kill-on-invalid-dep=yes', command)
                self.assertEqual(command[-1], '0')
                if i:
                    self.assertIn(f'--dependency=afterok:{400 + i}', command)
                else:
                    self.assertFalse(any(c.startswith('--dependency=') for c in command))

    def test_failed_submission_keeps_manifest_for_jobs_already_created(self):
        from tools import submit_visdrone_continuation as module
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(module.subprocess, 'run', side_effect=[
                subprocess.CompletedProcess([], 0, '401\n', ''),
                subprocess.CalledProcessError(1, 'sbatch'),
            ]):
                with self.assertRaises(subprocess.CalledProcessError):
                    module.submit_chain(root, root / 'source', root)
            jobs = json.loads((root / 'jobs.json').read_text())
            self.assertEqual([j['job_id'] for j in jobs], ['401'])

    def test_configs_change_training_duration_and_resume_policy_only(self):
        project = Path(__file__).resolve().parents[1]
        for group in ('g0', 'g5_adaptive_band', 'g5_adaptive_band_first2'):
            with self.subTest(group=group):
                folder = project / 'configs/controlled'
                base = load_config(str(folder / f'visdrone_{group}.yml'), {})
                continued = load_config(str(folder / f'visdrone_{group}_continue120.yml'), {})
                changed = {key for key in set(base) | set(continued)
                           if base.get(key) != continued.get(key)}
                self.assertEqual(changed, {'epochs', 'resume_lr_schedule', 'stage2_rollback',
                                           'lr_scheduler', 'output_dir', 'exp_name', '__include__'})
                self.assertEqual(continued['epochs'], 120)
                self.assertTrue(continued['resume_lr_schedule'])
                self.assertFalse(continued['stage2_rollback'])
                self.assertEqual(continued['lr_scheduler'],
                                 {'type': 'MultiStepLR', 'milestones': [100], 'gamma': 0.1})
                self.assertEqual(continued['train_dataloader']['total_batch_size'], 32)
                self.assertEqual(continued['train_dataloader']['collate_fn']['stop_epoch'], 72)
                self.assertEqual(continued['seed'], 0)


if __name__ == '__main__':
    unittest.main()
