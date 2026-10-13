"""Validate full-state resume with one EMA forward and two in-memory AMP updates."""

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.diagnose_visdrone_prediction_safety import capture_state, cpu_tree, guard_stats


STATE_KEYS = ('model', 'criterion', 'postprocessor', 'ema', 'scaler', 'optimizer',
              'lr_scheduler', 'lr_warmup_scheduler')


def restore_components(components, path, cfg):
    from src.misc import dist_utils
    from src.solver.det_solver import DetSolver

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    missing = [key for key in (*STATE_KEYS, 'last_epoch') if key not in checkpoint]
    if missing:
        raise ValueError(f'Missing full resume state: {missing}')
    for key in STATE_KEYS:
        if components[key] is None:
            raise ValueError(f'Configuration does not construct required component: {key}')
    for key in STATE_KEYS:
        dist_utils.de_parallel(components[key]).load_state_dict(checkpoint[key])
    last_epoch = int(checkpoint['last_epoch'])
    if cfg.yaml_cfg.get('resume_lr_schedule', False):
        # Use the training implementation; it retains optimizer/scheduler identity.
        context = SimpleNamespace(cfg=cfg, last_epoch=last_epoch, **components)
        DetSolver._resume_lr_schedule(context)
    return {'restored_keys': ['last_epoch', *STATE_KEYS], 'start_epoch': last_epoch + 1,
            'checkpoint_last_epoch': last_epoch}


def step_metadata(epoch, step, dataset_size, drop_last):
    # Native DistributedSampler pads to ceil(N/8), followed by local batch=4.
    samples_per_rank = (dataset_size + 7) // 8
    epoch_steps = samples_per_rank // 4 if drop_last else (samples_per_rank + 3) // 4
    if epoch_steps < 1:
        raise ValueError('Dataset cannot form one native batch')
    return {'epoch': epoch, 'step': step, 'global_step': epoch * epoch_steps + step,
            'epoch_step': epoch_steps}


def loader_overrides(world_size):
    if world_size not in (1, 8):
        raise ValueError('Preflight supports one GPU or the native eight-GPU recipe')
    settings = {'total_batch_size': 32} if world_size == 8 else {'total_batch_size': 4, 'num_workers': 4}
    return {'train_dataloader': dict(settings), 'val_dataloader': dict(settings)}


def execution_metadata(model, world_size, configured_sync_bn):
    synced = sum(isinstance(module, torch.nn.SyncBatchNorm) for module in model.modules())
    ordinary = sum(isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                   and not isinstance(module, torch.nn.SyncBatchNorm) for module in model.modules())
    return {'world_size': world_size, 'configured_sync_bn': bool(configured_sync_bn),
            'sync_bn': bool(synced and world_size > 1), 'sync_bn_modules': synced,
            'batch_norm_modules': ordinary,
            'ddp': isinstance(model, torch.nn.parallel.DistributedDataParallel)}


def require_all(condition, message):
    """Keep all ranks on the same side of the next loss/backward collective."""
    from src.misc import dist_utils

    failed = torch.tensor(int(not condition), device='cuda', dtype=torch.int32)
    if dist_utils.is_dist_available_and_initialized():
        torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
    if failed.item():
        raise ValueError(message if not condition else f'Another rank failed: {message}')


def run_rank(args, output, rank, world_size):
    from src.core import YAMLConfig
    from src.misc import dist_utils
    from src.solver.prediction_safety_diagnostics import guarded_prediction_forward

    result = {'passed': False, 'config': str(Path(args.config).resolve()),
              'checkpoint': str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
              'mode': 'resume' if args.checkpoint else 'fresh',
              'restored_keys': [], 'start_epoch': None, 'ema_forward_passed': None,
              'batch_size_per_gpu': 4, 'workers': 4, 'native_global_batch': 32,
              'native_world_size': 8, 'successful_updates': 0, 'steps': [],
              'rank': rank, 'world_size': world_size, 'sync_bn': None,
              'training_checkpoint_saved': False, 'source_checkpoint_modified': False}
    samples = originals = predictions = state_before = active_model = None
    phase = 'initialize'
    try:
        torch.set_num_threads(4)
        cfg = YAMLConfig(args.config, use_amp=True, resume=args.checkpoint,
                         HGNetv2={'pretrained': not bool(args.checkpoint)},
                         **loader_overrides(world_size))
        if not args.checkpoint and cfg.yaml_cfg.get('resume_lr_schedule', False):
            raise ValueError('Fresh preflight requires the 80-epoch configuration')
        model = dist_utils.warp_model(cfg.model.cuda(), sync_bn=cfg.sync_bn,
                                      find_unused_parameters=cfg.find_unused_parameters).train()
        result.update(execution_metadata(model, world_size, cfg.sync_bn))
        require_all(world_size == 1 or (result['ddp'] and
                    (not cfg.sync_bn or result['sync_bn'])), 'Native DDP/SyncBN was not constructed')
        # Construct the optimizer only after moving the same model to CUDA.
        components = {'model': model, 'criterion': cfg.criterion.cuda().train(),
                      'postprocessor': cfg.postprocessor.cuda().eval(),
                      'ema': cfg.ema.to('cuda'), 'scaler': cfg.scaler,
                      'optimizer': cfg.optimizer, 'lr_scheduler': cfg.lr_scheduler,
                      'lr_warmup_scheduler': cfg.lr_warmup_scheduler}
        if args.checkpoint:
            phase = 'restore'
            restore_error = None
            try:
                result.update(restore_components(components, args.checkpoint, cfg))
            except Exception as error:
                restore_error = error
            require_all(restore_error is None, f'Full state restore failed: {restore_error}')
        else:
            result['start_epoch'] = 0
        result['options'] = cfg.yaml_cfg['DFINEAllLayerStrip']
        optimizer, scaler = components['optimizer'], components['scaler']
        scheduler, warmup = components['lr_scheduler'], components['lr_warmup_scheduler']
        result['initial_lrs'] = [group['lr'] for group in optimizer.param_groups]
        result['scheduler_epoch'] = scheduler.last_epoch
        result['scheduler_milestones'] = dict(scheduler.milestones)
        result['warmup_last_step'] = warmup.last_step
        if args.checkpoint:
            phase = 'ema_eval_forward'
            active_model = components['ema'].module.eval()
            val_loader = dist_utils.warp_loader(cfg.val_dataloader,
                                               shuffle=cfg.val_dataloader.shuffle)
            samples, originals = next(iter(val_loader))
            state_before = capture_state(active_model)
            with torch.no_grad():
                predictions = guarded_prediction_forward(
                    active_model, samples.cuda(), None, output,
                    epoch=result['start_epoch'], step=-1, use_amp=True)
            boxes, logits = predictions['pred_boxes'], predictions['pred_logits']
            valid = (torch.isfinite(boxes).all() and torch.isfinite(logits).all()
                     and (boxes[..., 2:] > 0).all())
            require_all(bool(valid), 'Loaded EMA produced invalid validation predictions')
            result['ema_forward_passed'] = True
            result['ema_guard_stats'] = guard_stats(active_model)
            print('Loaded EMA: one validation batch passed.', flush=True)
        loader = dist_utils.warp_loader(cfg.train_dataloader, shuffle=cfg.train_dataloader.shuffle)
        epoch = result['start_epoch']
        loader.set_epoch(epoch)
        if world_size > 1:
            loader.sampler.set_epoch(epoch)
        result['workers'] = loader.num_workers
        result['actual_epoch_steps'] = len(loader)
        result['batch_size_per_gpu'] = loader.batch_size
        active_model = model
        criterion = components['criterion']
        for step, (samples, originals) in enumerate(loader):
            if step == 12:
                break
            phase = 'training_forward'
            predictions = None
            state_before = capture_state(dist_utils.de_parallel(model))
            images = samples.cuda()
            targets = [{key: value.cuda() if isinstance(value, torch.Tensor) else value
                        for key, value in target.items()} for target in originals]
            metas = step_metadata(epoch, step, len(loader.dataset), loader.drop_last)
            row = {**metas, 'image_ids': [int(t['image_id'].item()) for t in originals],
                   'input_shape': list(samples.shape), 'gt_counts': [len(t['labels']) for t in originals]}
            result['steps'].append(row)
            optimizer.zero_grad(set_to_none=True)
            predictions = guarded_prediction_forward(
                model, images, targets, output, epoch=epoch, step=step, use_amp=True)
            row['safety_counts'] = guard_stats(dist_utils.de_parallel(model))
            phase = 'criterion'
            with torch.autocast(device_type='cuda', enabled=False):
                losses = criterion(predictions, targets, **metas)
                loss = sum(losses.values())
            row['loss'] = float(loss) if torch.isfinite(loss) else None
            require_all(bool(torch.isfinite(loss)), 'Non-finite preflight loss')
            phase = 'backward'
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            bad_names = [name for name, parameter in model.named_parameters()
                         if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
            finite = not bad_names
            if world_size > 1:
                flags = torch.tensor([int(finite)], device='cuda', dtype=torch.int32)
                torch.distributed.all_reduce(flags)
                require_all(flags.item() in (0, world_size), 'Gradient finiteness differs across ranks')
            norm = None
            if finite and cfg.clip_max_norm > 0:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_max_norm)
                require_all(bool(torch.isfinite(norm)), 'Non-finite total gradient norm')
            row.update(gradient_norm=float(norm) if norm is not None else None,
                       nonfinite_gradient_examples=bad_names[:8],
                       scale_before=scaler.get_scale())
            row['gate_gradient_l1'] = {
                name: float(parameter.grad.abs().sum())
                for name, parameter in model.named_parameters()
                if finite and 'correction_gate' in name and parameter.grad is not None}
            phase = 'optimizer'
            scaler.step(optimizer)
            scaler.update()
            # Preserve native AMP skip / EMA / warmup ordering.
            components['ema'].update(model)
            warmup.step()
            row.update(optimizer_step_taken=finite, scale_after=scaler.get_scale())
            result['successful_updates'] += int(finite)
            print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
            if result['successful_updates'] == 2:
                break
        require_all(result['successful_updates'] == 2,
                    'Did not obtain two finite AMP updates within 12 batches')
        torch.cuda.synchronize()
        result['passed'] = True
    except Exception as error:
        result['error'] = {'phase': phase, 'type': type(error).__name__, 'message': str(error)}
        if phase == 'ema_eval_forward':
            result['ema_forward_passed'] = False
        if active_model is not None:
            result['failure_safety_counts'] = guard_stats(dist_utils.de_parallel(active_model))
        try:
            torch.save({'error': result['error'], 'diagnostic': cpu_tree(getattr(error, 'diagnostic', None)),
                        'samples': cpu_tree(samples), 'targets': cpu_tree(originals),
                        'state_before': state_before,
                        'predictions': {key: cpu_tree(predictions[key])
                                        for key in ('pred_boxes', 'pred_logits')}
                                       if predictions is not None else None}, output / 'failure_batch.pth')
            result['failure_artifact'] = str((output / 'failure_batch.pth').resolve())
        except Exception as save_error:
            result['artifact_write_error'] = str(save_error)
    finally:
        (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                       allow_nan=False) + '\n')
    return result


def run(args):
    from src.misc import dist_utils

    requested_world = int(os.environ.get('WORLD_SIZE', '1'))
    loader_overrides(requested_world)
    if requested_world > 1:
        dist_utils.setup_distributed(seed=0)
        if dist_utils.get_world_size() != requested_world:
            raise RuntimeError('torchrun requested distributed preflight but initialization failed')
    else:
        dist_utils.setup_seed(0)
    rank, world_size = dist_utils.get_rank(), dist_utils.get_world_size()
    root = Path(args.output_dir)
    directory_error = None
    if rank == 0:
        try:
            root.mkdir(parents=True, exist_ok=False)
        except Exception as error:
            directory_error = str(error)
    if world_size > 1:
        messages = [directory_error]
        torch.distributed.broadcast_object_list(messages, src=0)
        directory_error = messages[0]
    if directory_error:
        if world_size > 1:
            dist_utils.cleanup()
        raise RuntimeError(directory_error)
    output = root / f'rank_{rank}'
    output.mkdir(exist_ok=False)
    result = run_rank(args, output, rank, world_size)
    results = [None] * world_size
    if world_size > 1:
        torch.distributed.all_gather_object(results, result)
    else:
        results[0] = result
    overall = {**results[0], 'passed': all(item['passed'] for item in results),
               'rank_results': results,
               'rank_summary_paths': [str(root / f'rank_{index}' / 'summary.json')
                                      for index in range(world_size)]}
    if rank == 0:
        (root / 'summary.json').write_text(json.dumps(overall, ensure_ascii=False, indent=2,
                                                     allow_nan=False) + '\n')
        print(f'PREFLIGHT {"PASSED" if overall["passed"] else "FAILED"}: {root / "summary.json"}', flush=True)
    if world_size > 1:
        dist_utils.cleanup()
    if not overall['passed']:
        raise RuntimeError(f'Preflight failed; see {root / "summary.json"}')
    return overall


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', help='Full last.pth only; omit for a separate fresh preflight')
    parser.add_argument('--output-dir', required=True)
    run(parser.parse_args())
