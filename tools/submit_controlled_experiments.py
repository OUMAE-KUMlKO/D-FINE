"""Freeze this experiment batch and submit its twelve Slurm jobs in sequence."""

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


def experiment_order():
    yield from [('uavswarm', 'g0'), ('uavswarm', 'g5'),
                ('visdrone', 'g0'), ('visdrone', 'g5')]
    for dataset in ('uavswarm', 'visdrone'):
        for group in ('g1', 'g2', 'g3', 'g4'):
            yield dataset, group


def freeze_source(project, run_root):
    required = [project / 'weight/hgnetv2/PPHGNetV2_B4_stage1.pth',
                project / 'output/visdrone_controlled/manifest.json']
    required += [project / f'output/visdrone_controlled/instances_{split}.json'
                 for split in ('train', 'val')]
    required += [project / f'configs/controlled/{dataset}_{group}.yml'
                 for dataset, group in experiment_order()]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    # A fresh batch directory prevents duplicate submissions and output reuse.
    run_root.mkdir(parents=True, exist_ok=False)
    snapshot = run_root / 'source'
    snapshot.mkdir()
    for name in ('src', 'configs'):
        shutil.copytree(project / name, snapshot / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(project / 'train.py', snapshot / 'train.py')
    shutil.copytree(project / 'output/visdrone_controlled',
                    snapshot / 'output/visdrone_controlled')
    (snapshot / 'weight/hgnetv2').mkdir(parents=True)
    shutil.copy2(required[0], snapshot / 'weight/hgnetv2' / required[0].name)
    (snapshot / 'slurm_files').mkdir()
    shutil.copy2(project / 'slurm_files/run_train_dfine_controlled.slurm',
                 snapshot / 'slurm_files/run_train_dfine_controlled.slurm')
    for source in ('tools/submit_controlled_experiments.py',
                   'tools/dataset/prepare_visdrone_controlled.py',
                   'docs/research/uavswarm_visdrone_controlled_experiments_plan.md',
                   'docs/research/controlled_experiments_execution.md'):
        destination = snapshot / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / source, destination)
    for name, command in (
        ('git_head.txt', ['git', 'rev-parse', 'HEAD']),
        ('git_status.txt', ['git', 'status', '--short']),
        ('working_changes.patch', ['git', 'diff', 'HEAD', '--', 'src', 'configs', 'tools', 'slurm_files']),
    ):
        result = subprocess.run(command, cwd=project, check=True, capture_output=True, text=True)
        (run_root / name).write_text(result.stdout)
    preflight = project / 'output/controlled_preflight'
    if (preflight / 'memory.json').is_file():
        shutil.copytree(preflight, run_root / 'preflight')
    return snapshot


def submit_chain(snapshot, run_root):
    logs = run_root / 'logs'
    logs.mkdir(exist_ok=True)
    jobs = []
    previous = None
    for dataset, group in experiment_order():
        name = f'controlled_{dataset}_{group}_s0'
        config = f'configs/controlled/{dataset}_{group}.yml'
        output = run_root / 'results' / dataset / group
        command = [
            'sbatch', '--parsable', f'--job-name={name}',
            f'--output={logs / (name + "_%j.out")}',
            f'--error={logs / (name + "_%j.err")}',
            f'--chdir={snapshot}', '--kill-on-invalid-dep=yes',
        ]
        if previous:
            command.append(f'--dependency=afterok:{previous}')
        command += [str(snapshot / 'slurm_files/run_train_dfine_controlled.slurm'),
                    str(snapshot), config, str(output)]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        job_id = result.stdout.strip().split(';')[0]
        if not job_id.isdecimal():
            raise RuntimeError(f'Unexpected sbatch response: {result.stdout!r}')
        jobs.append(dict(job_id=job_id, dataset=dataset, group=group, seed=0,
                         afterok=previous, config=config, output_dir=str(output),
                         stdout=str(logs / f'{name}_{job_id}.out'),
                         stderr=str(logs / f'{name}_{job_id}.err')))
        (run_root / 'jobs.json').write_text(json.dumps(jobs, indent=2) + '\n')
        print(f'{job_id}: {dataset}/{group}, afterok={previous or "none"}', flush=True)
        previous = job_id
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', action='store_true', help='freeze source and submit jobs')
    parser.add_argument('--run-root', type=Path, help='new directory for this entire batch')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    run_root = (args.run_root or project / 'output' /
                datetime.now().strftime('controlled_%Y%m%d_%H%M%S')).resolve()
    if not args.submit:
        print('Order: ' + ' -> '.join(f'{d}/{g}' for d, g in experiment_order()))
        print(f'Batch: {run_root}\nUse --submit to freeze source and submit.')
        return
    snapshot = freeze_source(project, run_root)
    submit_chain(snapshot, run_root)
    print(f'Job manifest: {run_root / "jobs.json"}')


if __name__ == '__main__':
    main()
