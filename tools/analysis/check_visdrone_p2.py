"""Two disposable P2 updates on real images before launching VisDrone training."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _assert_frozen_partition(model):
    for name in ('backbone', 'encoder', 'decoder'):
        if any(module.training for module in getattr(model, name).modules()):
            raise AssertionError(f'Frozen module is in training mode: {name}')
    for name, parameter in model.named_parameters():
        if name.startswith('refiner.'):
            if not parameter.requires_grad:
                raise AssertionError(f'Refiner parameter is frozen: {name}')
        elif parameter.requires_grad or parameter.grad is not None:
            raise AssertionError(f'Frozen detector received gradients: {name}')


def check_batch(model, criterion, optimizer, images, targets, *, use_amp, clip_max_norm):
    """Check actual forward/backward behavior, including target-free inference."""
    from src.zoo.dfine.pred_refinement import DFINEPredRefiner

    if type(model) is not DFINEPredRefiner or model.refiner.sampling_mode != 'distribution':
        raise ValueError('This preflight requires distribution-guided DFINEPredRefiner')
    if model.requires_gt or model.oracle_evaluation:
        raise ValueError('P2 must perform prediction-only inference')
    device_type = images.device.type
    model.eval()
    _assert_frozen_partition(model)
    frozen = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith('refiner.')
    }
    expected = {id(p) for p in model.refiner.parameters()}
    optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
    if expected != optimized:
        raise AssertionError('Optimizer must contain exactly the refiner parameters')
    with torch.no_grad(), torch.autocast(device_type=device_type, enabled=use_amp):
        initial = model(images)
        native = model.decoder(model.encoder(model.backbone(images)))
    if not torch.equal(initial['pred_boxes'], initial['baseline_boxes']):
        raise AssertionError('Fresh zero-initialized P2 changed baseline boxes')
    if not torch.equal(initial['baseline_boxes'], native['pred_boxes'].float()):
        raise AssertionError('P2 baseline boxes differ from the native detector')
    if not torch.equal(initial['pred_logits'], native['pred_logits']):
        raise AssertionError('P2 classification differs from the native detector')
    if not initial['refinement_mask'].any():
        raise ValueError('No eligible prediction-guided queries in the selected batch')
    del native
    model.train()
    _assert_frozen_partition(model)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    result = {'successful_updates': 0, 'steps': []}
    for attempt in range(12):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device_type, enabled=use_amp):
            outputs = model(images, targets=targets)
        with torch.autocast(device_type=device_type, enabled=False):
            losses = criterion(outputs, targets)
            loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite P2 refinement loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        _assert_frozen_partition(model)
        gradients_finite = True
        for name, parameter in model.refiner.named_parameters():
            if parameter.grad is None:
                raise AssertionError(f'Missing P2 gradient: {name}')
            gradients_finite &= bool(torch.isfinite(parameter.grad).all())
        if not gradients_finite and not use_amp:
            raise ValueError('Nonfinite P2 gradients without AMP scaling')
        if clip_max_norm > 0:
            norm = torch.nn.utils.clip_grad_norm_(model.refiner.parameters(), clip_max_norm)
            if gradients_finite and not torch.isfinite(norm):
                raise ValueError('Nonfinite P2 gradient norm before clipping')
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()
        updated = scale_after >= scale_before
        if updated and not gradients_finite:
            raise AssertionError('Optimizer updated with nonfinite P2 gradients')
        result['successful_updates'] += int(updated)
        result['steps'].append({
            'attempt': attempt + 1, 'loss': loss.item(),
            'losses': {name: value.item() for name, value in losses.items()},
            'scale_before': scale_before, 'scale_after': scale_after,
            'effective_update': updated, 'refiner_gradients_finite': gradients_finite,
        })
        if result['successful_updates'] == 2:
            break
    if result['successful_updates'] != 2:
        raise RuntimeError('Did not obtain two finite AMP optimizer updates in 12 attempts')
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device_type, enabled=use_amp):
        final = model(images)
        with_gt = model(images, targets=targets)
    for key in ('pred_boxes', 'pred_logits', 'baseline_boxes', 'refinement_mask'):
        if not torch.equal(final[key], with_gt[key]):
            raise AssertionError(f'P2 prediction depends on GT: {key}')
    for key in ('pred_logits', 'baseline_boxes', 'refinement_mask'):
        if not torch.equal(final[key], initial[key]):
            raise AssertionError(f'Frozen P2 predictions changed: {key}')
    invalid = ~final['refinement_mask']
    if not torch.equal(final['pred_boxes'][invalid], final['baseline_boxes'][invalid]):
        raise AssertionError('P2 changed invalid queries')
    if not torch.isfinite(final['pred_boxes']).all():
        raise ValueError('Nonfinite P2 boxes after optimizer updates')
    box_change = (final['pred_boxes'] - final['baseline_boxes']).abs().max().item()
    if box_change <= 0:
        raise AssertionError('Two optimizer updates did not change P2 predictions')
    _assert_frozen_partition(model)
    for name, value in model.state_dict().items():
        if not name.startswith('refiner.') and not torch.equal(value.detach().cpu(), frozen[name]):
            raise AssertionError(f'Frozen baseline state changed: {name}')
    result.update({
        'with_without_gt_identical': True,
        'native_baseline_boxes_preserved': True,
        'native_baseline_classification_preserved': True,
        'classification_preserved_after_updates': True,
        'frozen_parameters_and_buffers_unchanged': True,
        'only_refiner_has_gradients': True,
        'invalid_queries_preserved': True,
        'refined_queries': int(final['refinement_mask'].sum()),
        'total_queries': final['refinement_mask'].numel(),
        'max_box_change_after_two_updates': box_change,
        'clip_max_norm': clip_max_norm,
    })
    return result


def run(args):
    from src.core import YAMLConfig
    from src.zoo.dfine.gt_refinement import DFINEGTRefiner

    output_file = Path(args.output_file)
    if output_file.exists():
        raise FileExistsError(f'Output already exists: {output_file}')
    device = torch.device(args.device)
    use_amp = device.type == 'cuda'
    started = time.monotonic()
    summary = {
        'status': 'running', 'phase': 'initialization',
        'config': str(Path(args.config).resolve()),
        'baseline_checkpoint': str(Path(args.baseline_checkpoint).resolve()),
        'device': str(device), 'batch_size': args.batch_size, 'seed': 0,
        'amp': use_amp, 'full_training': False, 'checkpoint_saved': False,
    }
    try:
        torch.set_num_threads(args.threads)
        torch.manual_seed(0)
        config = YAMLConfig(
            args.config,
            DFINEPredRefiner={'baseline_checkpoint': args.baseline_checkpoint},
            val_dataloader={'total_batch_size': args.batch_size, 'num_workers': args.workers},
        )
        model = config.model.to(device)
        criterion = config.criterion.to(device)
        summary['phase'] = 'load_validation_batch'
        images, raw_targets = next(iter(config.val_dataloader))
        images = images.to(device)
        if tuple(images.shape[-2:]) != (640, 640):
            raise ValueError(f'Expected fixed 640 input, got {tuple(images.shape[-2:])}')
        targets, _ = DFINEGTRefiner._prepare_targets(
            raw_targets, images.shape[-2:], device, 'xyxy_absolute', model.decoder.num_classes,
        )
        summary.update({
            'images': len(images), 'image_size': list(images.shape[-2:]),
            'image_ids': [int(torch.as_tensor(t['image_id']).item()) for t in raw_targets],
            'valid_gt_counts': [len(t['boxes']) for t in targets],
            'model_class': type(model).__name__, 'feature_level': model.feature_level,
            'target_format': 'cxcywh_normalized', 'sampling_mode': model.refiner.sampling_mode,
        })
        if use_amp:
            torch.cuda.reset_peak_memory_stats(device)
        summary['phase'] = 'two_updates_and_invariants'
        summary.update(check_batch(
            model, criterion, config.optimizer, images, targets,
            use_amp=use_amp, clip_max_norm=config.clip_max_norm,
        ))
        summary.update({'status': 'passed', 'phase': 'complete'})
    except Exception as error:
        summary.update({'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        summary['elapsed_seconds'] = time.monotonic() - started
        if use_amp and torch.cuda.is_initialized():
            summary['peak_cuda_memory_allocated_gib'] = torch.cuda.max_memory_allocated(device) / 2**30
            summary['peak_cuda_memory_reserved_gib'] = torch.cuda.max_memory_reserved(device) / 2**30
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--baseline-checkpoint', required=True)
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if min(args.batch_size, args.threads) < 1 or args.workers < 0:
        parser.error('batch size/threads must be positive; workers must be nonnegative')
    run(args)


if __name__ == '__main__':
    main()
