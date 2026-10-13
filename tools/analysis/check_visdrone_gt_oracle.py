"""Check one real GT-oracle batch and two disposable AMP optimizer updates."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _assert_frozen_partition(model):
    for name in ('backbone', 'encoder', 'decoder', 'matcher'):
        if any(module.training for module in getattr(model, name).modules()):
            raise AssertionError(f'Frozen module is in training mode: {name}')
    for name, parameter in model.named_parameters():
        if name.startswith('refiner.'):
            if not parameter.requires_grad:
                raise AssertionError(f'Refiner parameter is frozen: {name}')
        elif parameter.requires_grad or parameter.grad is not None:
            raise AssertionError(f'Frozen detector received gradients: {name}')


def snapshot_baseline(model):
    _assert_frozen_partition(model)
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith('refiner.')
    }


def assert_frozen_baseline(model, before):
    """Compare parameters and buffers, including BatchNorm running statistics."""
    _assert_frozen_partition(model)
    after = {
        name: value for name, value in model.state_dict().items()
        if not name.startswith('refiner.')
    }
    if after.keys() != before.keys():
        raise AssertionError('Frozen baseline state keys changed')
    for name, value in after.items():
        if not torch.equal(value.detach().cpu(), before[name]):
            raise AssertionError(f'Frozen baseline state changed: {name}')


def assert_initial_identity(outputs, native):
    if not torch.equal(outputs['pred_boxes'], outputs['baseline_boxes']):
        raise AssertionError('Refiner zero initialization changed baseline boxes')
    if not torch.equal(outputs['baseline_boxes'], native['pred_boxes'].float()):
        raise AssertionError('Oracle baseline boxes differ from the native detector')
    if (
        outputs['pred_logits'].dtype != native['pred_logits'].dtype
        or not torch.equal(outputs['pred_logits'], native['pred_logits'])
    ):
        raise AssertionError('Oracle classification differs from the native detector')


def run(args):
    from src.core import YAMLConfig
    from src.zoo.dfine.gt_candidate_refinement import DFINEGTCandidateRefiner
    from src.zoo.dfine.gt_refinement import DFINEGTRefiner

    model_class = {
        'DFINEGTRefiner': DFINEGTRefiner,
        'DFINEGTCandidateRefiner': DFINEGTCandidateRefiner,
    }[args.model]
    output_file = Path(args.output_file)
    if output_file.exists():
        raise FileExistsError(f'Output already exists: {output_file}')
    started = time.monotonic()
    device = torch.device(args.device)
    use_amp = device.type == 'cuda'
    summary = {
        'status': 'running', 'phase': 'initialization',
        'config': str(Path(args.config).resolve()),
        'requested_model': args.model,
        'baseline_checkpoint': str(Path(args.baseline_checkpoint).resolve()),
        'device': str(device), 'batch_size': args.batch_size, 'seed': 0,
        'amp': use_amp, 'full_training': False, 'checkpoint_saved': False,
        'successful_updates': 0, 'steps': [],
    }
    try:
        torch.set_num_threads(args.threads)
        torch.manual_seed(0)
        config = YAMLConfig(
            args.config,
            **{args.model: {'baseline_checkpoint': args.baseline_checkpoint}},
            val_dataloader={
                'total_batch_size': args.batch_size, 'num_workers': args.workers,
            },
        )
        model = config.model.to(device).eval()
        if type(model) is not model_class or model.feature_mode != 'real':
            raise ValueError(f'This preflight requires the real-feature {args.model}')
        criterion = config.criterion.to(device)
        frozen_before = snapshot_baseline(model)
        summary['phase'] = 'load_validation_batch'
        images, targets = next(iter(config.val_dataloader))
        images = images.to(device)
        targets = [
            {key: value.to(device) if torch.is_tensor(value) else value
             for key, value in target.items()}
            for target in targets
        ]
        if tuple(images.shape[-2:]) != (640, 640):
            raise ValueError(f'Expected fixed 640 input, got {tuple(images.shape[-2:])}')
        summary.update({
            'images': len(images), 'image_size': list(images.shape[-2:]),
            'image_ids': [int(torch.as_tensor(t['image_id']).item()) for t in targets],
            'raw_gt_counts': [len(t['boxes']) for t in targets],
            'model_class': type(model).__name__, 'target_format': 'xyxy_absolute',
            'feature_level': model.feature_level,
        })
        if use_amp:
            torch.cuda.reset_peak_memory_stats(device)
        summary['phase'] = 'initial_identity'
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
            initial = model(images, targets, target_format='xyxy_absolute')
            native = model.decoder(model.encoder(model.backbone(images)))
        assert_initial_identity(initial, native)
        summary.update({
            'zero_initialization_max_box_error': 0.0,
            'native_baseline_boxes_preserved': True,
            'native_baseline_classification_preserved': True,
            'refinement_counts': {
                key: int(value.item()) for key, value in initial['refinement_counts'].items()
            },
            'refinement_counts_per_image': {
                key: value.detach().cpu().tolist()
                for key, value in initial['refinement_counts_per_image'].items()
            },
        })
        if not initial['refinement_mask'].any():
            raise ValueError('The selected batch has no eligible GT-matched queries')
        del native
        model.train()
        _assert_frozen_partition(model)
        optimizer = config.optimizer
        expected_parameters = {id(p) for p in model.refiner.parameters()}
        optimizer_parameters = {id(p) for group in optimizer.param_groups for p in group['params']}
        if optimizer_parameters != expected_parameters:
            raise AssertionError('Optimizer must contain exactly the refiner parameters')
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        for attempt in range(12):
            summary['phase'] = 'training_forward'
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images, targets, target_format='xyxy_absolute')
            with torch.autocast(device_type=device.type, enabled=False):
                losses = criterion(outputs, outputs['refinement_targets'])
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite GT-refinement loss')
            summary['phase'] = 'training_backward'
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            _assert_frozen_partition(model)
            gradients_finite = True
            for name, parameter in model.refiner.named_parameters():
                if parameter.grad is None:
                    raise AssertionError(f'Missing refiner gradient: {name}')
                gradients_finite &= bool(torch.isfinite(parameter.grad).all())
            if not gradients_finite and not use_amp:
                raise ValueError('Nonfinite refiner gradients without AMP scaling')
            if config.clip_max_norm > 0:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_max_norm)
                if gradients_finite and not torch.isfinite(norm):
                    raise ValueError('Nonfinite gradient norm before clipping')
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            updated = scale_after >= scale_before
            if updated and not gradients_finite:
                raise AssertionError('Optimizer updated with nonfinite refiner gradients')
            summary['successful_updates'] += int(updated)
            summary['steps'].append({
                'attempt': attempt + 1, 'loss': loss.item(),
                'losses': {key: value.item() for key, value in losses.items()},
                'scale_before': scale_before, 'scale_after': scale_after,
                'effective_update': updated, 'refiner_gradients_finite': gradients_finite,
            })
            if summary['successful_updates'] == 2:
                break
        if summary['successful_updates'] != 2:
            raise RuntimeError('Did not obtain two finite AMP optimizer updates in 12 attempts')
        summary['phase'] = 'post_update_invariants'
        model.eval()
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
            final = model(images, targets, target_format='xyxy_absolute')
        for key in ('pred_logits', 'baseline_boxes', 'refinement_mask'):
            if not torch.equal(final[key], initial[key]):
                raise AssertionError(f'Frozen predictions or eligible matches changed: {key}')
        unchanged = ~final['refinement_mask']
        if not torch.equal(final['pred_boxes'][unchanged], final['baseline_boxes'][unchanged]):
            raise AssertionError('Unmatched or invalid-sampling queries changed')
        if not torch.isfinite(final['pred_boxes']).all():
            raise ValueError('Nonfinite refined boxes after optimizer updates')
        box_change = (final['pred_boxes'] - final['baseline_boxes']).abs().max().item()
        if box_change <= 0:
            raise AssertionError('Two optimizer updates did not change any matched boxes')
        assert_frozen_baseline(model, frozen_before)
        summary.update({
            'status': 'passed', 'phase': 'complete',
            'classification_preserved_after_updates': True,
            'unmatched_and_invalid_queries_preserved': True,
            'frozen_parameters_and_buffers_unchanged': True,
            'only_refiner_has_gradients': True,
            'max_box_change_after_two_updates': box_change,
            'clip_max_norm': config.clip_max_norm,
        })
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
    parser.add_argument('--model', choices=('DFINEGTRefiner', 'DFINEGTCandidateRefiner'),
                        default='DFINEGTRefiner')
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
