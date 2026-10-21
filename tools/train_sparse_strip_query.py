#!/usr/bin/env python3
"""Fixed-rate, 24-epoch paired continuation of the stripped G5 detector."""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_CHECKPOINT = ROOT / 'output/visdrone_g5_ddf_off_20261019_103206/results/g5_ddf_off/best_stg2.pth'


def optimizer_groups(model):
    norms = (nn.LayerNorm, nn.GroupNorm, nn.modules.batchnorm._BatchNorm,
             nn.modules.instancenorm._InstanceNorm)
    norm_ids = {id(p) for m in model.modules() if isinstance(m, norms)
                for p in m.parameters(recurse=False)}
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith('decoder.'):
            lr = 2.5e-5
        elif name.startswith('sparse_refiner.'):
            lr = 1e-3
        else:
            raise ValueError(f'Unexpected trainable parameter: {name}')
        decay = 0.0 if id(parameter) in norm_ids else 1.25e-4
        groups.setdefault((lr, decay), []).append(parameter)
    return [dict(params=params, lr=lr, weight_decay=decay)
            for (lr, decay), params in groups.items()]


def require_finite(value, path='output'):
    if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f'Nonfinite {path}')
    elif isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, f'{path}.{key}')
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite(item, f'{path}[{index}]')


def guard_raw_losses(criterion):
    """Check each native loss before DFINECriterion's final nan_to_num."""
    original = criterion.get_loss
    def checked(*args, **kwargs):
        losses = original(*args, **kwargs)
        require_finite({key: value for key, value in losses.items()
                        if criterion.weight_dict.get(key, 0) != 0}, 'raw_loss')
        return losses
    criterion.get_loss = checked


def scaled_step(scaler, optimizer, ema, model):
    before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    stepped = scaler.get_scale() >= before
    if stepped:
        ema.update(model)
    return stepped


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    from src.core import YAMLConfig
    from src.misc import dist_utils
    from src.optim.ema import ModelEMA
    from src.solver.det_engine import evaluate
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT))
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--test-only', action='store_true')
    args = parser.parse_args()
    dist_utils.setup_distributed(seed=args.seed)
    rank = dist_utils.get_rank()
    device = torch.device('cuda', rank) if torch.cuda.is_available() else torch.device('cpu')
    output = Path(args.output_dir)
    if dist_utils.is_main_process():
        output.mkdir(parents=True, exist_ok=False)
    if dist_utils.is_dist_available_and_initialized():
        torch.distributed.barrier()
    cfg = YAMLConfig(args.config)
    if cfg.yaml_cfg['epochs'] != 24:
        raise ValueError('This protocol requires exactly 24 epochs')
    # Module initialization consumes neither paired loader nor DN/dropout RNG streams.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        raw_model = cfg.model
    provenance = raw_model.load_detector_checkpoint(args.checkpoint)
    raw_model.to(device)
    criterion = cfg.criterion.to(device)
    guard_raw_losses(criterion)
    postprocessor = cfg.postprocessor.to(device)
    train_loader = dist_utils.warp_loader(cfg.train_dataloader, shuffle=True)
    val_loader = dist_utils.warp_loader(cfg.val_dataloader, shuffle=False)
    for loader, offset in ((train_loader, 0), (val_loader, 100000)):
        loader.generator = torch.Generator().manual_seed(args.seed + offset + rank)
        if isinstance(loader.sampler, torch.utils.data.RandomSampler):
            loader.sampler.generator = loader.generator
        if isinstance(loader.sampler, torch.utils.data.DistributedSampler):
            loader.sampler.seed = args.seed + offset
    if len(train_loader.dataset) != 6471 or len(val_loader.dataset) != 548:
        raise ValueError('Expected the fixed 6471 train / 548 val split')
    optimizer = torch.optim.AdamW(optimizer_groups(raw_model), betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    ema = ModelEMA(raw_model, decay=0.999, warmups=0)
    model = dist_utils.warp_model(raw_model, find_unused_parameters=True)
    # Also check evaluation outputs without altering native evaluator behavior.
    def finite_output_hook(module, inputs, result):
        require_finite(result)
    raw_model.register_forward_hook(finite_output_hook)
    ema.module.register_forward_hook(finite_output_hook)
    if dist_utils.is_main_process():
        (output / 'effective_config.yml').write_text(yaml.safe_dump(cfg.yaml_cfg, sort_keys=False))
        write_json(output / 'provenance.json', dict(checkpoint=str(Path(args.checkpoint).resolve()),
                   checkpoint_load=provenance, seed=args.seed, world_size=dist_utils.get_world_size(),
                   model_init_seed=0, ema_decay=0.999, ema_warmups=0,
                   decoder_lr=2.5e-5, sparse_lr=1e-3, weight_decay=1.25e-4))
    evaluator = cfg.evaluator
    def validate(epoch):
        stats, _ = evaluate(ema.module, criterion, postprocessor, val_loader, evaluator,
                            device, epoch=epoch, use_wandb=False)
        return stats
    initial = validate(-1)
    baseline_ap = initial['coco_eval_bbox'][0]
    if dist_utils.is_main_process():
        write_json(output / 'initial_metrics.json', initial)
    if args.test_only:
        dist_utils.cleanup()
        return
    best_ap = float('-inf')
    consecutive_drop = 0
    total_skipped = 0
    for epoch in range(24):
        train_loader.set_epoch(epoch)
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        criterion.train()
        actual = skipped = 0
        loss_sums = {}
        diagnostic_sums = {}
        for step, (samples, targets) in enumerate(train_loader):
            samples = samples.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                outputs = model(samples, targets=targets)
            with torch.cuda.amp.autocast(enabled=False):
                losses = criterion(outputs, targets, epoch=epoch,
                                   step=step, global_step=epoch * len(train_loader) + step)
                require_finite(losses, 'weighted_loss')
                if any(float(value.detach()) != 0 for key, value in losses.items()
                       if key.startswith('loss_ddf')):
                    raise ValueError('DDF must contribute exactly zero')
                loss = sum(losses.values())
                require_finite(loss, 'total_loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.yaml_cfg['clip_max_norm'])
            stepped = scaled_step(scaler, optimizer, ema, model)
            actual += int(stepped)
            skipped += int(not stepped)
            for key, value in losses.items():
                loss_sums[key] = loss_sums.get(key, 0.0) + float(value.detach())
            for key, value in raw_model.last_diagnostics.items():
                diagnostic_sums[key] = diagnostic_sums.get(key, 0.0) + float(value)
        total_skipped += skipped
        stats = validate(epoch)
        ap = stats['coco_eval_bbox'][0]
        consecutive_drop = consecutive_drop + 1 if baseline_ap - ap > 0.005 else 0
        stop = not raw_model.enabled and consecutive_drop >= 3
        improved = ap > best_ap
        best_ap = max(best_ap, ap)
        averages = {**{f'loss/{k}': v / len(train_loader) for k, v in loss_sums.items()},
                    **{f'diagnostic/{k}': v / len(train_loader) for k, v in diagnostic_sums.items()}}
        averages = dist_utils.reduce_dict({k: torch.tensor(v, device=device) for k, v in averages.items()})
        record = dict(epoch=epoch, completed_epochs=epoch + 1, metrics=stats,
                      initial_ap=baseline_ap, best_ap=best_ap, actual_updates=actual,
                      skipped_updates=skipped, total_skipped_updates=total_skipped,
                      ema_updates=ema.updates, consecutive_control_drop=consecutive_drop,
                      **{k: float(v) for k, v in averages.items()})
        if dist_utils.is_main_process():
            state = dict(model=raw_model.state_dict(), ema=ema.state_dict(),
                         optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                         epoch=epoch, last_epoch=epoch, completed_epochs=epoch + 1,
                         optimizer_updates=ema.updates, skipped_updates=total_skipped,
                         args=vars(args), initial_metrics=initial, metrics=stats)
            torch.save(state, output / 'last.pth')
            if improved:
                torch.save(state, output / 'best.pth')
            with (output / 'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(record, allow_nan=False) + '\n')
            print(json.dumps(record, allow_nan=False))
            if stop:
                write_json(output / 'status.json', dict(status='stopped_control_degradation',
                           reason='AP dropped more than 0.5 points from initial for 3 consecutive epochs',
                           epoch=epoch, initial_ap=baseline_ap, current_ap=ap))
        if stop:
            break
    if dist_utils.is_main_process() and not stop:
        write_json(output / 'status.json', dict(status='completed', completed_epochs=24,
                   best_ap=best_ap, optimizer_updates=ema.updates))
    dist_utils.cleanup()


if __name__ == '__main__':
    main()
