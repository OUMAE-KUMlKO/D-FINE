"""Two effective AMP updates for one approved G5 variant; no checkpoint saved."""

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import YAMLConfig


def run(args):
    output = Path(args.output_file)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    cfg = YAMLConfig(args.config, use_amp=True,
                     train_dataloader={'total_batch_size': 4, 'num_workers': 4})
    model = cfg.model.cuda().train()
    criterion = cfg.criterion.cuda().train()
    optimizer, scaler = cfg.optimizer, cfg.scaler
    warmup = cfg.lr_warmup_scheduler
    loader = cfg.train_dataloader
    loader.set_epoch(0)
    if len(model.refiner.heads) != 6:
        raise ValueError('Expected six strip heads')
    rows, successful_updates = [], 0
    torch.cuda.reset_peak_memory_stats()
    for step, (images, targets) in enumerate(loader):
        if step == 12:
            break
        images = images.cuda()
        targets = [{k: v.cuda() if isinstance(v, torch.Tensor) else v
                    for k, v in target.items()} for target in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda'):
            predictions = model(images, targets)
        with torch.autocast(device_type='cuda', enabled=False):
            losses = criterion(predictions, targets, epoch=0, step=step,
                               global_step=step, epoch_step=len(loader))
            loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise ValueError('Non-finite smoke loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        finite_gradients = all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                               for parameter in model.parameters())
        norm = (torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_max_norm)
                if finite_gradients else None)
        gate_grad = {name: float(parameter.grad.abs().sum())
                     for name, parameter in model.named_parameters()
                     if finite_gradients and 'correction_gate' in name and parameter.grad is not None}
        scale_before = scaler.get_scale()
        # AMP may legitimately skip initial updates while calibrating its scale.
        scaler.step(optimizer)
        scaler.update()
        warmup.step()
        successful_updates += int(finite_gradients)
        row = {'step': step, 'input_size': list(images.shape),
               'gt_counts': [len(t['labels']) for t in targets],
               'loss': float(loss), 'gradient_norm': None if norm is None else float(norm),
               'optimizer_step_taken': bool(finite_gradients),
               'scale_before': scale_before, 'scale_after': scaler.get_scale(),
               'gate_gradient_l1': gate_grad}
        rows.append(row)
        print(json.dumps(row), flush=True)
        if successful_updates == 2:
            break
    if successful_updates != 2:
        raise ValueError('Failed to obtain two finite AMP updates within 12 attempts')
    torch.cuda.synchronize()
    result = {'config': args.config, 'device': torch.cuda.get_device_name(),
              'batch_size_per_gpu': 4, 'steps': rows, 'successful_updates': successful_updates,
              'model_options': cfg.yaml_cfg['DFINEAllLayerStrip'],
              'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024 ** 3,
              'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024 ** 3,
              'passed': True, 'checkpoint_saved': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'PREFLIGHT PASSED: {args.config}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-file', required=True)
    run(parser.parse_args())
