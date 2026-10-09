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

    def test_uav_main_comparisons_use_three_real_seeds_in_one_chain(self):
        module = self.load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commands = []

            def sbatch(command, **kwargs):
                commands.append(command)
                return subprocess.CompletedProcess(command, 0, f'{201 + len(commands)}\n', '')

            with mock.patch.object(module.subprocess, 'run', side_effect=sbatch):
                jobs = module.submit_chain(root / 'source', root,
                                           datasets=['uavswarm'], groups=['g0', 'g5'],
                                           seeds=[1, 2, 3])
            self.assertEqual([(j['dataset'], j['group'], j['seed']) for j in jobs],
                             [('uavswarm', g, s) for s in (1, 2, 3) for g in ('g0', 'g5')])
            self.assertEqual(len({j['output_dir'] for j in jobs}), 6)
            self.assertFalse(any(x.startswith('--dependency=') for x in commands[0]))
            for i, (job, command) in enumerate(zip(jobs, commands)):
                self.assertEqual(command[-1], str(job['seed']))
                self.assertIn(f'seed{job["seed"]}', job['output_dir'])
                self.assertIn(f'_s{job["seed"]}', job['stdout'])
                if i:
                    self.assertIn(f'--dependency=afterok:{jobs[i-1]["job_id"]}', command)

    def test_uav_only_snapshot_does_not_require_visdrone_annotations(self):
        module = self.load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / 'project'
            (project / 'src').mkdir(parents=True)
            files = ['weight/hgnetv2/PPHGNetV2_B4_stage1.pth', 'train.py',
                     'configs/controlled/uavswarm_g0.yml', 'configs/controlled/uavswarm_g5.yml',
                     'slurm_files/run_train_dfine_controlled.slurm',
                     'tools/submit_controlled_experiments.py',
                     'tools/dataset/prepare_visdrone_controlled.py',
                     'docs/research/uavswarm_visdrone_controlled_experiments_plan.md',
                     'docs/research/controlled_experiments_execution.md']
            for name in files:
                path = project / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('test fixture\n')
            with mock.patch.object(module.subprocess, 'run',
                                   return_value=subprocess.CompletedProcess([], 0, 'test\n', '')):
                snapshot = module.freeze_source(project, root / 'batch',
                                                datasets=['uavswarm'], groups=['g0', 'g5'])
            self.assertTrue((snapshot / 'configs/controlled/uavswarm_g5.yml').is_file())
            self.assertTrue((snapshot / 'weight/hgnetv2/PPHGNetV2_B4_stage1.pth').is_file())
            self.assertFalse((snapshot / 'output/visdrone_controlled').exists())

    def test_adaptive_band_selection_submits_only_requested_visdrone_job(self):
        module = self.load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(module.subprocess, 'run',
                                   return_value=subprocess.CompletedProcess([], 0, '301\n', '')) as run:
                jobs = module.submit_chain(root / 'source', root,
                                           datasets=['visdrone'], groups=['g5_adaptive_band'],
                                           seeds=[0])
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]['config'], 'configs/controlled/visdrone_g5_adaptive_band.yml')
            self.assertEqual(jobs[0]['group'], 'g5_adaptive_band')
            self.assertEqual(jobs[0]['seed'], 0)
            self.assertNotIn('--resume', run.call_args.args[0])
            self.assertIsNone(jobs[0]['afterok'])
        self.assertEqual(len(list(module.experiment_order())), 12)


if __name__ == '__main__':
    unittest.main()
