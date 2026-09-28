import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / 'tools/submit_controlled_experiments.py'


class ControlledSubmissionTests(unittest.TestCase):
    def load(self):
        self.assertTrue(SCRIPT.is_file(), 'controlled submission script is missing')
        spec = importlib.util.spec_from_file_location('controlled_submit', SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_all_twelve_jobs_form_one_success_dependency_chain(self):
        module = self.load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commands = []

            def sbatch(command, **kwargs):
                commands.append(command)
                return subprocess.CompletedProcess(command, 0, f'{100 + len(commands)};cluster\n', '')

            with mock.patch.object(module.subprocess, 'run', side_effect=sbatch):
                jobs = module.submit_chain(root / 'source', root)
            self.assertEqual(len(jobs), 12)
            self.assertEqual([(j['dataset'], j['group']) for j in jobs[:4]],
                             [('uavswarm', 'g0'), ('uavswarm', 'g5'),
                              ('visdrone', 'g0'), ('visdrone', 'g5')])
            self.assertEqual(len({(j['dataset'], j['group']) for j in jobs}), 12)
            self.assertFalse(any(x.startswith('--dependency=') for x in commands[0]))
            for i, command in enumerate(commands[1:], 1):
                self.assertIn(f'--dependency=afterok:{100 + i}', command)
                self.assertIn('--kill-on-invalid-dep=yes', command)
            saved = json.loads((root / 'jobs.json').read_text())
            self.assertEqual(saved, jobs)

    def test_submission_failure_keeps_record_of_already_submitted_jobs(self):
        module = self.load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(module.subprocess, 'run', side_effect=[
                subprocess.CompletedProcess([], 0, '101\n', ''),
                subprocess.CalledProcessError(1, ['sbatch'], stderr='queue error'),
            ]) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    module.submit_chain(root / 'source', root)
            self.assertEqual(run.call_count, 2)
            saved = json.loads((root / 'jobs.json').read_text())
            self.assertEqual([job['job_id'] for job in saved], ['101'])


if __name__ == '__main__':
    unittest.main()
