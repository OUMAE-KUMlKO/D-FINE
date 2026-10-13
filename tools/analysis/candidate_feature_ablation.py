"""Temporarily replace encoded candidate features during inference."""

from contextlib import contextmanager

import torch


@contextmanager
def candidate_feature_ablation(model, mode, seed=0):
    """Ablate every strip head's encoded visual candidates for one eval scope."""
    if mode not in {"normal", "zero", "noise"}:
        raise ValueError("mode must be 'normal', 'zero', or 'noise'")
    if model.training:
        raise ValueError("candidate feature ablation is eval-only")
    if getattr(model, "requires_gt", False):
        raise ValueError("candidate feature ablation rejects requires_gt models")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    heads = model.refiner.heads
    generators = [None] * len(heads)
    handles = []

    def replacement(layer_index):
        def hook(module, args, output):
            if mode == "zero":
                return torch.zeros_like(output)
            generator = generators[layer_index]
            if generator is None:
                generator = torch.Generator(device=output.device)
                generator.manual_seed(seed + 1000003 * layer_index)
                generators[layer_index] = generator
            return torch.randn(
                output.shape, dtype=output.dtype, device=output.device,
                generator=generator,
            )
        return hook

    try:
        if mode != "normal":
            for layer_index, head in enumerate(heads):
                handles.append(head.edge_fusion.register_forward_hook(replacement(layer_index)))
        yield model
    finally:
        for handle in handles:
            handle.remove()
