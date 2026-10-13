"""Capture invalid G5 predictions before entering distributed loss computation."""

from pathlib import Path
import sys

import torch
import torch.distributed as distributed

from ..misc import dist_utils
from ..zoo.dfine.all_layer_strip import DFINEAllLayerStrip
from ..zoo.dfine.prediction_safety import PredictionSafetyError


def prediction_safety_enabled(model):
    return isinstance(dist_utils.de_parallel(model), DFINEAllLayerStrip)


def prediction_safety_stats(model):
    core = dist_utils.de_parallel(model)
    decoder = getattr(getattr(core, 'decoder', None), 'decoder', None)
    refiner = getattr(core, 'refiner', None)
    return {
        **getattr(decoder, 'last_prediction_safety_stats', {}),
        **getattr(refiner, 'last_prediction_safety_stats', {}),
    }


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(item) for item in value)
    return value


def guarded_prediction_forward(model, samples, targets, output_dir, *, epoch, step, use_amp):
    rng = {'torch_cpu': torch.get_rng_state()}
    if samples.is_cuda:
        rng['torch_cuda'] = torch.cuda.get_rng_state(samples.device)
    error, outputs = None, None
    try:
        with torch.autocast(device_type=samples.device.type, enabled=use_amp):
            outputs = model(samples, targets=targets)
    except PredictionSafetyError as caught:
        error = caught

    failed = torch.tensor(int(error is not None), device=samples.device, dtype=torch.int32)
    if dist_utils.is_dist_available_and_initialized():
        # All ranks reach this boundary before criterion's own collectives.
        distributed.all_reduce(failed, op=distributed.ReduceOp.MAX)
    if not failed.item():
        return outputs

    if error is None:
        raise PredictionSafetyError('Invalid prediction on another rank; all ranks stopped',
                                    {'stage': 'peer_failure', 'epoch': epoch, 'step': step})

    rank = dist_utils.get_rank()
    path = Path(output_dir or '.') / 'prediction_failures' / f'epoch{epoch}_step{step}_rank{rank}.pth'
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            'epoch': epoch, 'step': step, 'rank': rank, 'use_amp': use_amp,
            'samples': _cpu(samples), 'targets': _cpu(targets),
            'model': _cpu(dist_utils.de_parallel(model).state_dict()),
            'diagnostic': _cpu(error.diagnostic),
            'prediction_safety_stats': prediction_safety_stats(model),
            'rng_before_forward': rng,
            'model_buffers': 'captured after failed forward',
            'full_training_checkpoint': False,
        }
        with path.open('xb') as stream:
            torch.save(state, stream)
    except Exception as save_error:
        sys.stderr.write(f'Could not save prediction failure to {path}: {save_error}\n')
        raise error from save_error
    raise PredictionSafetyError(f'{error}; diagnostic saved to {path}', error.diagnostic) from error
