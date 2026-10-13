"""Train only the approved E1-E3 matrix, each with the original 80+40 schedule."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.submit_controlled_experiments import freeze_source


EXPERIMENTS = (
    ('E1', 'g5_adaptive_band_gate'),
    ('E2', 'g5_adaptive_band_pre_reference'),
    ('E3', 'g5_adaptive_band_gate_pre_reference'),
)
BASELINE = 'output/visdrone_continue120_20261009/results/visdrone/seed0/g5_adaptive_band/best_stg2.pth'


def prepare(project, run_root):
    for _, group in EXPERIMENTS:
        for suffix in ('', '_continue120'):
            path = project / f'configs/controlled/visdrone_{group}{suffix}.yml'
            if not path.is_file():
                raise FileNotFoundError(path)
    if not (project / BASELINE).is_file():
        raise FileNotFoundError(project / BASELINE)
    # freeze_source copies all configs; its group list only checks prerequisites.
    snapshot = freeze_source(project, run_root, datasets=('visdrone',),
                             groups=('g5_adaptive_band',))
    for relative in ('tools/submit_visdrone_gate_reference.py',
                     'tools/preflight_visdrone_gate_reference.py',
                     'slurm_files/run_preflight_visdrone_gate_reference.slurm',
                     'slurm_files/run_resume_dfine_controlled.slurm',
                     'slurm_files/run_train_visdrone_gate_reference.slurm',
                     'docs/research/visdrone_g5_gate_reference_plan_20261010.md'):
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / relative, destination)
    selection = {
        'dataset': 'visdrone', 'seed': 0, 'batch_size': 32, 'gpus': 8,
        'experiments': dict(EXPERIMENTS), 'E0_reused_checkpoint': str(project / BASELINE),
        'E0_retrained': False, 'initialization': 'HGNetv2 backbone pretrained; new detector/strip heads',
        'epochs': {'stage1': [0, 79], 'stage2': [80, 119]},
        'stage2': {'resume': 'each group epoch80/last.pth; full training state',
                   'resume_lr_schedule': True, 'milestones': [100], 'gamma': .1,
                   'stage2_rollback': False},
        'LQE': 'unchanged, uses corrected distribution',
        'training_annotation_ids': 'unchanged, same historical evaluation and checkpoint-selection convention',
        'final_comparison': 'same best EMA weights reevaluated with positive annotation IDs',
        'failures': 'no automatic retry or recovery; later groups depend on success',
    }
    (run_root / 'experiment_selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    return snapshot


def submit_chain(snapshot, run_root):
    manifest = run_root / 'jobs.json'
    if manifest.exists():
        raise FileExistsError(manifest)
    logs = run_root / 'logs'
    logs.mkdir(exist_ok=True)
    jobs, previous = [], None
    for experiment, group in EXPERIMENTS:
        name = f'visdrone_{experiment.lower()}_gate_ref_s0'
        config80 = f'configs/controlled/visdrone_{group}.yml'
        config120 = f'configs/controlled/visdrone_{group}_continue120.yml'
        output = run_root / 'results' / 'visdrone' / 'seed0' / group
        command = ['sbatch', '--parsable', f'--job-name={name}',
                   f'--output={logs / (name + "_%j.out")}',
                   f'--error={logs / (name + "_%j.err")}',
                   f'--chdir={snapshot}', '--kill-on-invalid-dep=yes']
        if previous:
            command.append(f'--dependency=afterok:{previous}')
        command += [str(snapshot / 'slurm_files/run_train_visdrone_gate_reference.slurm'),
                    str(snapshot), config80, config120, str(output), '0']
        response = subprocess.run(command, check=True, capture_output=True, text=True)
        job_id = response.stdout.strip().split(';')[0]
        if not job_id.isdecimal():
            raise RuntimeError(f'Unexpected sbatch response: {response.stdout!r}')
        jobs.append({'job_id': job_id, 'experiment': experiment, 'group': group,
                     'seed': 0, 'afterok': previous, 'config80': config80,
                     'config120': config120, 'output_dir': str(output),
                     'stdout': str(logs / f'{name}_{job_id}.out'),
                     'stderr': str(logs / f'{name}_{job_id}.err')})
        manifest.write_text(json.dumps(jobs, indent=2) + '\n')
        print(f'{job_id}: {experiment}/{group}; afterok={previous or "none"}', flush=True)
        previous = job_id
    return jobs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--run-root', type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = (args.run_root or project / 'output' /
            datetime.now().strftime('visdrone_gate_reference_%Y%m%d_%H%M%S')).resolve()
    print('E0 reused; new training: ' + ' -> '.join(name for name, _ in EXPERIMENTS))
    print(f'Run root: {root}')
    if args.submit:
        submit_chain(prepare(project, root), root)
    else:
        print('Use --submit to freeze source and submit the three approved variants.')
