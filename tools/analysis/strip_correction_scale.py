"""Temporarily scale strip correction deltas during inference."""

import math
from contextlib import contextmanager
from numbers import Real


@contextmanager
def strip_correction_scale(model, alpha):
    """Scale every strip score head's output without changing its inputs."""
    if not isinstance(alpha, Real) or not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    if model.training:
        raise ValueError("strip correction scaling is eval-only")
    if getattr(model, "requires_gt", False):
        raise ValueError("strip correction scaling rejects requires_gt models")

    alpha = float(alpha)
    handles = []

    def scale_output(module, args, output):
        return output * alpha

    try:
        if alpha != 1.0:
            for head in model.refiner.heads:
                handles.append(head.score_head.register_forward_hook(scale_output))
        yield model
    finally:
        for handle in handles:
            handle.remove()
