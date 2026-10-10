"""Resume the three approved VisDrone seed0 comparisons through epoch119."""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.submit_controlled_experiments import freeze_source


EXPERIMENTS = (
    ('g0', 'output/controlled_20260926_212304/results/visdrone/g0/last.pth'),
    ('g5_adaptive_band', 'output/controlled_visdrone_adaptive_band_20261006_seed0/'
     'results/visdrone/seed0/g5_adaptive_band/last.pth'),
    ('g5_adaptive_band_first2', 'output/visdrone_g5_adaptive_band_first2_20261009/'
     'results/visdrone/seed0/g5_adaptive_band_first2/last.pth'),
)


def submit_chain(project, snapshot, run_root):
    logs = run_root / 'logs'
    logs.mkdir(exist_ok=True)
    jobs, previous = [], None
    for group, checkpoint in EXPERIMENTS:
        name = f'continue120_visdrone_{group}_s0'
        config = f'configs/controlled/visdrone_{group}_continue120.yml'
        output = run_root / 'results' / 'visdrone' / 'seed0' / group
        resume = project / checkpoint
        command = [
            'sbatch', '--parsable', f'--job-name={name}',
            f'--output={logs / (name + "_%j.out")}',
            f'--error={logs / (name + "_%j.err")}',
            f'--chdir={snapshot}', '--kill-on-invalid-dep=yes',
        ]
        if previous:
            command.append(f'--dependency=afterok:{previous}')
        command += [str(snapshot / 'slurm_files/run_resume_dfine_controlled.slurm'),
                    str(snapshot), config, str(output), str(resume), '0']
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        job_id = result.stdout.strip().split(';')[0]
        if not job_id.isdecimal():
            raise RuntimeError(f'Unexpected sbatch response: {result.stdout!r}')
        jobs.append(dict(
            job_id=job_id, dataset='visdrone', group=group, seed=0,
            afterok=previous, config=config, output_dir=str(output), resume=str(resume),
            stdout=str(logs / f'{name}_{job_id}.out'),
            stderr=str(logs / f'{name}_{job_id}.err'),
        ))
        (run_root / 'jobs.json').write_text(json.dumps(jobs, indent=2) + '\n')
        print(f'{job_id}: {group}, resume={resume}, afterok={previous or "none"}', flush=True)
        previous = job_id
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--run-root', type=Path, help='new directory for the three continuations')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    run_root = (args.run_root or project / 'output' /
                datetime.now().strftime('visdrone_continue120_%Y%m%d_%H%M%S')).resolve()
    print('Order: ' + ' -> '.join(group for group, _ in EXPERIMENTS))
    print(f'Batch: {run_root}')
    if not args.submit:
        print('Use --submit to freeze source and submit three serial resume jobs.')
        return
    for group, checkpoint in EXPERIMENTS:
        for path in (project / checkpoint,
                     project / f'configs/controlled/visdrone_{group}_continue120.yml'):
            if not path.is_file():
                raise FileNotFoundError(path)
    snapshot = freeze_source(project, run_root, datasets=('visdrone',),
                             groups=tuple(group for group, _ in EXPERIMENTS))
    for relative in ('tools/submit_visdrone_continuation.py',
                     'slurm_files/run_resume_dfine_controlled.slurm',
                     'docs/research/visdrone_continue120_plan.md'):
        shutil.copy2(project / relative, snapshot / relative)
    (run_root / 'experiment_selection.json').write_text(json.dumps({
        'datasets': ['visdrone'], 'groups': [group for group, _ in EXPERIMENTS],
        'seeds': [0], 'start_epoch': 80, 'epochs': 120,
        'milestones': [100], 'gamma': 0.1, 'stage2_rollback': False,
        'resume': {group: str(project / checkpoint) for group, checkpoint in EXPERIMENTS},
    }, indent=2) + '\n')
    submit_chain(project, snapshot, run_root)
    print(f'Job manifest: {run_root / "jobs.json"}')


if __name__ == '__main__':
    main()
