"""Observe local strip corrections without changing production inference."""

import torch


@torch.inference_mode()
def trace_all_layer_strip(model, images):
    if model.training:
        raise ValueError('Layer tracing requires an eval-mode model')
    if getattr(model, 'requires_gt', False):
        raise ValueError('Layer tracing requires GT-free inference')
    layers = {}
    handles = []

    def observe(index, head, args, result):
        if index in layers:
            raise RuntimeError(f'Strip layer {index} ran more than once')
        before = args[2].detach().float().clone()
        height, width = args[3]
        if head.bandwidth_mode == 'box_relative':
            half_widths = (before[..., 2:] * before.new_tensor([width, height])
                           * head.bandwidth_ratio).clamp(head.min_offset_pixels, head.offset_pixels)
        else:
            half_widths = before.new_full((*before.shape[:2], 2), head.offset_pixels)
        layers[index] = {
            'boxes_before': before,
            'boxes_after': result['pred_boxes'].detach().float().clone(),
            'valid': result['refinement_mask'].detach().clone(),
            'half_widths_px': half_widths,
        }

    try:
        for index, head in enumerate(model.refiner.heads):
            handles.append(head.register_forward_hook(
                lambda module, args, result, i=index: observe(i, module, args, result)
            ))
        outputs = model(images)
    finally:
        for handle in handles:
            handle.remove()
    expected = list(range(model.decoder.eval_idx + 1))
    if list(layers) != expected:
        raise RuntimeError(f'Expected layers {expected}, got {list(layers)}')
    torch.testing.assert_close(layers[expected[-1]]['boxes_after'], outputs['pred_boxes'],
                               atol=1e-6, rtol=1e-6)
    return {'outputs': outputs, 'layers': layers}
