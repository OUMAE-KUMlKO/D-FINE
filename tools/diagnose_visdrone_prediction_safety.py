"""Bounded train-mode forwards from saved model weights; never optimize or resume."""

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_model_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    raw = checkpoint['model']
    weights = {}
    for name, value in raw.items():
        key = name.removeprefix('module.')
        # The legacy NaN dump contains both DDP and plain names for the same tensor.
        if key not in weights or not name.startswith('module.'):
            weights[key] = value
    model.load_state_dict(weights, strict=True)
    return {'checkpoint': str(Path(checkpoint_path).resolve()), 'weights': 'model',
            'checkpoint_last_epoch': checkpoint.get('last_epoch'),
            'duplicate_keys_removed': len(raw) - len(weights)}


def capture_state(model):
    return {'buffers': {name: value.detach().cpu().clone()
                        for name, value in model.named_buffers()},
            'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


@torch.no_grad()
def restore_state(model, state):
    for name, value in model.named_buffers():
        value.copy_(state['buffers'][name].to(value.device))
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all(state['cuda_rng'])


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def guard_stats(model):
    stats = {}
    for prefix, module in (('decoder', model.decoder.decoder), ('head', model.refiner)):
        for name, value in getattr(module, 'last_prediction_safety_stats', {}).items():
            stats[f'{prefix}.{name}'] = value.item() if isinstance(value, torch.Tensor) else value
    return stats


@torch.no_grad()
def forward_probe(model, images, targets, amp):
    from src.zoo.dfine.prediction_safety import PredictionSafetyError

    started = time.monotonic()
    diagnostic = None
    result = {'precision': 'amp_fp16' if amp else 'fp32', 'status': 'valid'}
    try:
        with torch.autocast(device_type=images.device.type, enabled=amp):
            output = model(images, targets=targets)
        boxes, logits = output['pred_boxes'], output['pred_logits']
        if (not torch.isfinite(boxes).all() or not torch.isfinite(logits).all()
                or not (boxes[..., 2:] > 0).all()):
            result['status'] = 'invalid_final_output'
            diagnostic = {'stage': 'final_output', 'boxes': boxes.detach().cpu(),
                          'logits': logits.detach().cpu()}
    except PredictionSafetyError as error:
        diagnostic = cpu_tree(error.diagnostic)
        result.update(status='prediction_safety_error', message=str(error),
                      stage=diagnostic.get('stage'), layer_index=diagnostic.get('layer_index'))
    if images.is_cuda:
        torch.cuda.synchronize(images.device)
    result['seconds'] = time.monotonic() - started
    result['guard_stats'] = guard_stats(model)
    if result['status'] == 'valid' and any(
        value > 0 for name, value in result['guard_stats'].items()
        if 'fallback' in name
    ):
        result['status'] = 'guard_fallback'
        diagnostic = {'stage': 'guard_fallback', 'guard_stats': result['guard_stats']}
    return result, diagnostic


def run(args):
    from src.core import YAMLConfig

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Full model weights are loaded strictly; do not reload/download the backbone.
    cfg = YAMLConfig(args.config, HGNetv2={'pretrained': False},
                     controlled_seed=args.seed,
                     train_dataloader={'total_batch_size': 4, 'num_workers': 4})
    model = cfg.model
    metadata = load_model_weights(model, args.checkpoint)
    model = model.cuda().train()
    loader = cfg.train_dataloader
    loader.set_epoch(args.epoch)
    metadata.update(config=str(Path(args.config).resolve()),
                    code_root=str(Path(__file__).resolve().parents[1]))
    rows = []
    started = time.monotonic()
    stop_reason = 'max_batches'
    iterator = iter(loader)
    with (output / 'batches.jsonl').open('x') as stream:
        for step in range(args.max_batches):
            if time.monotonic() - started >= args.max_seconds:
                stop_reason = 'time_budget'
                break
            try:
                samples, originals = next(iterator)
            except StopIteration:
                stop_reason = 'loader_exhausted'
                break
            if time.monotonic() - started >= args.max_seconds:
                stop_reason = 'time_budget'
                break
            images = samples.cuda()
            targets = [{key: value.cuda() if isinstance(value, torch.Tensor) else value
                        for key, value in target.items()} for target in originals]
            state_before = capture_state(model)
            row = {'batch_index': step, 'image_ids': [int(t['image_id'].item()) for t in originals],
                   'input_shape': list(samples.shape),
                   'gt_counts': [len(t['labels']) for t in originals]}
            row['amp'], diagnostic = forward_probe(model, images, targets, amp=True)
            if diagnostic is not None:
                stop_reason = 'first_safety_event'
                artifact = output / 'event_batch.pth'
                torch.save({'samples': cpu_tree(samples), 'targets': cpu_tree(originals),
                            'state_before': state_before, 'model': metadata,
                            'epoch': args.epoch, 'batch': row,
                            'diagnostic': diagnostic}, artifact)
                row['event_batch'] = str(artifact.resolve())
                if args.fp32_replay and time.monotonic() - started < args.max_seconds:
                    state_after_amp = capture_state(model)
                    try:
                        restore_state(model, state_before)
                        row['fp32'], replay_diagnostic = forward_probe(model, images, targets, amp=False)
                        if replay_diagnostic is not None:
                            torch.save(replay_diagnostic, output / 'fp32_diagnostic.pth')
                    finally:
                        restore_state(model, state_after_amp)
                elif args.fp32_replay:
                    row['fp32_replay_skipped'] = 'time_budget'
            rows.append(row)
            line = json.dumps(row, ensure_ascii=False, allow_nan=False)
            stream.write(line + '\n')
            stream.flush()
            print(line, flush=True)
            if diagnostic is not None:
                break
    summary = {'model': metadata, 'epoch': args.epoch, 'seed': args.seed,
               'batch_size': 4, 'workers': 4, 'train_mode': True, 'denoising_targets': True,
               'backward_or_optimizer_updates': False, 'source_checkpoint_modified': False,
               'max_batches': args.max_batches, 'max_seconds': args.max_seconds,
               'time_budget_policy': 'Checked between batches/replay; an in-flight forward is not interrupted.',
               'elapsed_seconds': time.monotonic() - started, 'stop_reason': stop_reason,
               'batches': rows,
               'limitations': 'Original failing inputs/RNG were not saved; this bounded single-GPU scan is not exact reproduction of the 8-GPU failure. Train-mode buffers can change in memory; FP32 replay restores pre-batch buffers/RNG and then the AMP state.'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                                   allow_nan=False) + '\n')
    print(f'Saved {output / "summary.json"}; stop={stop_reason}', flush=True)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--max-batches', type=int, default=16)
    parser.add_argument('--max-seconds', type=float, default=180.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=92)
    parser.add_argument('--fp32-replay', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.max_batches <= 16:
        parser.error('max-batches must be between 1 and 16')
    if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 180:
        parser.error('max-seconds must be finite and between 0 (exclusive) and 180')
    if args.seed < 0 or args.epoch < 0:
        parser.error('seed and epoch must be nonnegative')
    return args


if __name__ == '__main__':
    run(parse_args())
