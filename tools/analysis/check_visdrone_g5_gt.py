"""Check two disposable G5 GT-guided updates from fresh or EMA initialization."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_strict_ema(model, checkpoint):
    state = torch.load(checkpoint, map_location='cpu')
    if 'ema' not in state or 'module' not in state['ema']:
        raise ValueError('G5 preflight requires a complete EMA checkpoint')
    weights = state['ema']['module']
    model.load_state_dict(weights, strict=True)
    return {
        'source': 'ema.module', 'epoch': int(state.get('last_epoch', state.get('epoch', -1))),
        'keys_loaded': len(weights), 'missing_keys': [], 'unexpected_keys': [],
    }


def assert_optimizer_partition(model, optimizer):
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    if set(actual) != expected or len(actual) != len(set(actual)):
        raise AssertionError('Optimizer must contain exactly the current trainable model parameters once')


def parameter_group(name):
    if name.startswith('refiner.heads.'):
        return f'refiner.{name.split(".")[2]}'
    return name.split('.')[0]


def gradient_groups(model):
    groups = {name: {'parameters': 0, 'with_grad': 0, 'finite': True, 'nonzero': False}
              for name in ('backbone', 'encoder', 'decoder', *(f'refiner.{i}' for i in range(6)))}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                raise AssertionError(f'Frozen parameter has a gradient: {name}')
            continue
        group = groups[parameter_group(name)]
        group['parameters'] += 1
        if parameter.grad is not None:
            group['with_grad'] += 1
            finite = bool(torch.isfinite(parameter.grad).all())
            group['finite'] &= finite
            group['nonzero'] |= finite and bool(parameter.grad.count_nonzero())
    if any(not group['parameters'] for group in groups.values()):
        raise AssertionError('G5 must jointly train the detector and all six strip heads')
    return groups


def assert_predictions_finite(outputs):
    for name in ('pred_boxes', 'pred_logits'):
        if not torch.isfinite(outputs[name]).all():
            raise ValueError(f'Nonfinite {name}')
    if (outputs['pred_boxes'][..., 2:] <= 0).any():
        raise ValueError('Nonpositive predicted box dimensions')


def check_batch(
    model, criterion, optimizer, images, raw_targets, *, use_amp, clip_max_norm,
    fresh_init=False, train_images=None, train_targets=None, ema=None,
):
    from src.zoo.dfine.all_layer_strip import DFINEAllLayerStrip
    from src.zoo.dfine.gt_refinement import DFINEGTRefiner

    assert_optimizer_partition(model, optimizer)
    device = images.device
    targets, ignored = DFINEGTRefiner._prepare_targets(
        raw_targets, images.shape[-2:], device, 'xyxy_absolute', model.decoder.num_classes)
    if (train_images is None) != (train_targets is None):
        raise ValueError('Supply both training images and normalized training targets')
    if train_images is None:
        train_images, train_targets = images, targets
    if not any(len(target['boxes']) for target in targets):
        raise ValueError('The preflight batch must have eligible GT boxes')
    if fresh_init:
        for head in model.refiner.heads:
            if head.score_head[-1].weight.count_nonzero() or head.score_head[-1].bias.count_nonzero():
                raise AssertionError('Fresh G5 score output layers must start at zero')
    model.eval()
    empty = [{'boxes': torch.empty(0, 4, device=device),
              'labels': torch.empty(0, dtype=torch.long, device=device)}]
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
        normal = DFINEAllLayerStrip.forward(model, images[:1])
        unguided = model(images[:1], empty)
        guided = model(images[:1], raw_targets[:1])
    for key in ('pred_boxes', 'pred_logits'):
        if not torch.equal(normal[key], unguided[key]):
            raise AssertionError(f'Empty GT changed the normal G5 path: {key}')
    assert_predictions_finite(guided)
    guidance_change = (guided['pred_boxes'] - normal['pred_boxes']).abs().max().item()
    if fresh_init:
        for key in ('pred_boxes', 'pred_logits'):
            if not torch.equal(guided[key], normal[key]):
                raise AssertionError(f'Fresh zero-score GT guidance changed {key}')
    elif guidance_change <= 0:
        raise AssertionError('GT guidance did not change the trained G5 predictions')
    summary = {
        'empty_gt_matches_normal': True, 'gt_guidance_max_box_change': guidance_change,
        'initial_gt_guidance_stats': dict(model.last_gt_guidance_stats),
        'ignored_gt': ignored, 'successful_updates': 0, 'steps': [],
        'optimizer_exact_trainable_partition': True,
        'fresh_zero_score_identity': bool(fresh_init),
        'training_image_size': list(train_images.shape[-2:]),
        'training_gt_counts': [len(target['boxes']) for target in train_targets],
    }
    before = {name: p.detach().cpu().clone() for name, p in model.named_parameters()
              if p.requires_grad}
    del normal, unguided, guided
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for attempt in range(12):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(train_images, train_targets)
        assert_predictions_finite(outputs)
        with torch.autocast(device_type=device.type, enabled=False):
            losses = criterion(outputs, train_targets)
            loss = sum(losses.values())
        if not torch.isfinite(loss) or not all(torch.isfinite(value) for value in losses.values()):
            raise ValueError('Nonfinite G5 oracle loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        groups = gradient_groups(model)
        finite = all(group['finite'] for group in groups.values())
        if finite and any(group['with_grad'] != group['parameters'] for group in groups.values()):
            missing = [name for name, parameter in model.named_parameters()
                       if parameter.requires_grad and parameter.grad is None]
            raise AssertionError(f'Trainable parameters unused with find_unused_parameters=False: {missing}')
        if finite:
            missing = [name for name, group in groups.items() if not group['nonzero']
                       and not (fresh_init and summary['successful_updates'] == 0
                                and name.startswith('refiner.'))]
            if missing:
                raise AssertionError(f'Missing nonzero gradients in groups: {missing}')
        if not finite and not use_amp:
            raise ValueError('Nonfinite G5 gradients without AMP scaling')
        grad_norm = None
        if finite and clip_max_norm > 0:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite finite-gradient norm')
            grad_norm = float(norm)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()
        updated = scale_after >= scale_before
        if updated and not finite:
            raise AssertionError('Optimizer updated despite nonfinite gradients')
        summary['successful_updates'] += int(updated)
        if updated and ema is not None:
            ema.update(model)
        summary['steps'].append({
            'attempt': attempt + 1, 'loss': float(loss),
            'losses': {name: float(value) for name, value in losses.items()},
            'scale_before': scale_before, 'scale_after': scale_after,
            'effective_update': updated, 'gradients_finite': finite,
            'gradient_groups': groups, 'gradient_norm': grad_norm,
        })
        del outputs, losses, loss
        if summary['successful_updates'] == 2:
            break
    if summary['successful_updates'] != 2:
        raise RuntimeError('Did not obtain two finite AMP optimizer updates in 12 attempts')
    updated_groups = {name: False for name in groups}
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise ValueError(f'Nonfinite updated parameter: {name}')
        if parameter.requires_grad:
            group = parameter_group(name)
            if not updated_groups[group]:
                updated_groups[group] = not torch.equal(parameter.detach().cpu(), before[name])
    if not all(updated_groups.values()):
        raise AssertionError(f'Optimizer did not change every trainable group: {updated_groups}')
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
        final = model(images, raw_targets)
        final_normal = DFINEAllLayerStrip.forward(model, images)
    assert_predictions_finite(final)
    post_guidance_change = (final['pred_boxes'] - final_normal['pred_boxes']).abs().max().item()
    if post_guidance_change <= 0:
        raise AssertionError('GT guidance does not affect predictions after two updates')
    summary.update({
        'updated_groups': updated_groups, 'final_predictions_finite': True,
        'final_gt_guidance_stats': dict(model.last_gt_guidance_stats),
        'clip_max_norm': clip_max_norm,
        'post_update_gt_guidance_max_box_change': post_guidance_change,
        'ema_updates': 0, 'ema_gt_eval_passed': False, 'ema_checked': False,
    })
    if ema is not None:
        if not ema.module.requires_gt or ema.module.supports_gt_refinement_metrics:
            raise AssertionError('EMA deepcopy lost GT oracle attributes')
        if ema.module.training or any(parameter.requires_grad for parameter in ema.module.parameters()):
            raise AssertionError('EMA must remain an evaluation-only copy')
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
            ema_outputs = ema.module(images, raw_targets)
        assert_predictions_finite(ema_outputs)
        summary.update({'ema_updates': ema.updates, 'ema_gt_eval_passed': True, 'ema_checked': True})
    return summary


def run(args):
    from src.core import YAMLConfig
    from src.zoo.dfine.gt_all_layer_strip import DFINEGTAllLayerStrip

    output_file = Path(args.output_file)
    if output_file.exists():
        raise FileExistsError(f'Output already exists: {output_file}')
    started = time.monotonic()
    device = torch.device(args.device)
    use_amp = device.type == 'cuda'
    summary = {
        'status': 'running', 'phase': 'initialization',
        'config': str(Path(args.config).resolve()),
        'checkpoint': str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        'initialization': 'imagenet_fresh' if args.fresh_init else 'g5_ema_checkpoint',
        'fresh_init': args.fresh_init, 'seed': 0,
        'device': str(device), 'amp': use_amp, 'batch_size': args.batch_size,
        'full_training': False, 'checkpoint_saved': False,
    }
    try:
        torch.set_num_threads(args.threads)
        torch.manual_seed(0)
        loader_overrides = {'total_batch_size': args.batch_size, 'num_workers': args.workers}
        config = YAMLConfig(args.config, val_dataloader=loader_overrides,
                            **({'train_dataloader': loader_overrides} if args.fresh_init else {}))
        if args.fresh_init and config.yaml_cfg['HGNetv2'].get('pretrained') is not True:
            raise ValueError('Fresh preflight requires the configured ImageNet backbone initialization')
        model = config.model
        if type(model) is not DFINEGTAllLayerStrip or model.num_refine_layers != 6:
            raise ValueError('This preflight requires a six-layer DFINEGTAllLayerStrip')
        summary['checkpoint_load'] = (
            {'source': 'HGNetv2 ImageNet via config; fresh decoder and strip heads', 'epoch': None}
            if args.fresh_init else load_strict_ema(model, args.checkpoint))
        model.to(device)
        criterion = config.criterion.to(device)
        ema = config.ema
        summary['ema_enabled'] = ema is not None
        summary['configured_sync_bn'] = bool(config.sync_bn)
        summary['single_gpu_sync_bn_conversion'] = False
        summary['phase'] = 'load_validation_batch'
        images, targets = next(iter(config.val_dataloader))
        images = images.to(device)
        targets = [{key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in target.items()} for target in targets]
        if tuple(images.shape[-2:]) != (640, 640):
            raise ValueError(f'Expected fixed 640 input, got {tuple(images.shape[-2:])}')
        summary.update({
            'image_size': list(images.shape[-2:]), 'images': len(images),
            'image_ids': [int(t['image_id']) for t in targets],
            'raw_gt_counts': [len(t['boxes']) for t in targets],
            'model_class': type(model).__name__, 'feature_level': model.feature_level,
            'train_target_format': 'cxcywh_normalized', 'eval_target_format': 'xyxy_absolute',
        })
        train_images = train_targets = None
        if args.fresh_init:
            summary['phase'] = 'load_augmented_training_batch'
            train_loader = config.train_dataloader
            train_loader.set_epoch(0)
            train_images, train_targets = next(iter(train_loader))
            summary['training_source_image_size'] = list(train_images.shape[-2:])
            train_images = train_images.to(device)
            train_targets = [{key: value.to(device) if torch.is_tensor(value) else value
                              for key, value in target.items()} for target in train_targets]
            summary['training_image_ids'] = [int(t['image_id']) for t in train_targets]
            if args.train_size < 1 or args.train_size % 32:
                raise ValueError('Training preflight size must be a positive multiple of 32')
            if tuple(train_images.shape[-2:]) != (args.train_size, args.train_size):
                train_images = torch.nn.functional.interpolate(
                    train_images, size=(args.train_size, args.train_size),
                    mode='bilinear', align_corners=False)
            summary['training_image_size'] = list(train_images.shape[-2:])
            summary['training_batch_from_augmented_loader'] = True
        if use_amp:
            torch.cuda.reset_peak_memory_stats(device)
        summary['phase'] = 'two_disposable_updates'
        summary.update(check_batch(model, criterion, config.optimizer, images, targets,
                                   use_amp=use_amp, clip_max_norm=config.clip_max_norm,
                                   fresh_init=args.fresh_init, train_images=train_images,
                                   train_targets=train_targets, ema=ema))
        summary.update({'status': 'passed', 'phase': 'complete'})
    except Exception as error:
        summary.update({'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        summary['elapsed_seconds'] = time.monotonic() - started
        if use_amp and torch.cuda.is_initialized():
            summary['peak_cuda_memory_bytes'] = torch.cuda.max_memory_allocated(device)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument('--checkpoint')
    initialization.add_argument('--fresh-init', action='store_true')
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--train-size', type=int, default=800,
                        help='Fresh-mode training memory check at the original multiscale maximum')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
