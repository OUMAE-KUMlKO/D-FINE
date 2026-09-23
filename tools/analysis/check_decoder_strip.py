"""Check one real batch and two disposable decoder-strip optimizer steps."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def run(args):
    from src.core import YAMLConfig
    from src.zoo.dfine.gt_refinement import DFINEGTRefiner

    output_file = Path(args.output_file)
    if output_file.exists():
        raise FileExistsError(f'Output already exists: {output_file}')
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    device = torch.device(args.device)
    config = YAMLConfig(
        args.config,
        DFINEDecoderStripRefiner={'baseline_checkpoint': args.baseline_checkpoint},
        val_dataloader={'total_batch_size': args.batch_size, 'num_workers': args.workers},
    )
    model = config.model.to(device).eval()
    criterion = config.criterion.to(device)
    images, targets = next(iter(config.val_dataloader))
    images = images.to(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
        initial = model(images)
    max_initial_error = (initial['pred_boxes'] - initial['baseline_boxes']).abs().max().item()
    torch.testing.assert_close(initial['pred_boxes'], initial['baseline_boxes'], atol=0, rtol=0)
    canonical, ignored = DFINEGTRefiner._prepare_targets(
        targets, images.shape[-2:], device, 'xyxy_absolute', initial['pred_logits'].shape[-1],
    )
    model.train()
    optimizer = config.optimizer
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    step_losses = []
    projection_grad_norms = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            outputs = model(images)
        with torch.autocast(device_type=device.type, enabled=False):
            loss = sum(criterion(outputs, canonical).values())
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite decoder strip loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        for name, parameter in model.named_parameters():
            if name.startswith('refiner.'):
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise ValueError(f'Missing or nonfinite adapter gradient: {name}')
            elif parameter.requires_grad or parameter.grad is not None:
                raise ValueError(f'Frozen detector received gradients: {name}')
        grad_norm = model.refiner.query_projection.weight.grad.norm().item()
        if grad_norm == 0:
            raise ValueError('Box loss did not reach the query adapter through the frozen decoder')
        projection_grad_norms.append(grad_norm)
        scaler.step(optimizer)
        scaler.update()
        step_losses.append(loss.item())
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
        final = model(images)
        changed_gt = model(images, targets=[{'unused_gt': object()} for _ in targets])
    for name in ('pred_boxes', 'pred_logits', 'refinement_mask'):
        torch.testing.assert_close(final[name], changed_gt[name], atol=0, rtol=0)
    torch.testing.assert_close(final['pred_logits'], initial['pred_logits'], atol=0, rtol=0)
    box_change = (final['pred_boxes'] - final['baseline_boxes']).abs().max().item()
    if box_change == 0:
        raise ValueError('Two optimizer steps did not change any decoded boxes')
    summary = {
        'config': str(Path(args.config).resolve()),
        'baseline_checkpoint': str(Path(args.baseline_checkpoint).resolve()),
        'images': len(images), 'batch_size': args.batch_size,
        'image_size': list(images.shape[-2:]), 'ignored_targets': sum(ignored),
        'refine_layers': list(model.refine_layers),
        'candidate_count_per_edge': model.decoder.reg_max + 1,
        'query_chunk_size': model.refiner.query_chunk_size,
        'zero_initialization_max_box_error': max_initial_error,
        'gt_independent_forward': True, 'baseline_classification_preserved': True,
        'frozen_detector_has_no_gradients': True,
        'disposable_train_step_losses': step_losses,
        'query_projection_gradient_norms': projection_grad_norms,
        'max_box_change_after_two_steps': box_change,
        'full_training': False, 'checkpoint_saved': False,
        'elapsed_seconds': time.monotonic() - started,
        'peak_cuda_memory_allocated_gib': (
            torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else None
        ),
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/dfine/pred_refinement/dfine_hgnetv2_l_uavswarm_decoder_strip.yml')
    parser.add_argument('--baseline-checkpoint', required=True)
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if min(args.batch_size, args.threads) < 1 or args.workers < 0:
        parser.error('batch size/threads must be positive; workers must be nonnegative')
    run(args)


if __name__ == '__main__':
    main()
