"""Trace GT-free query feedback and its local effect on decoder box outputs.

This module records tensors only; matching, metrics, and final baseline-box
fallback belong to the diagnostic caller.
"""

import torch

from src.zoo.dfine.dfine_utils import distance2bbox, weighting_function


@torch.inference_mode()
def trace_decoder_feedback(model, features, image_size, active_layers=None):
    """Return raw final predictions and each configured layer's local strip effect.

    ``logits_before/after`` are cumulative boundary logits, not classification
    scores. Both use the same current-layer query and previous accumulated
    logits, differing only by this layer's strip delta. The caller applies the
    wrapper's final-box validity check and baseline fallback.

    Layer indices are zero-based. All ``model.refine_layers`` are recorded;
    ``active_layers=None`` enables them all and ``()`` disables feedback.
    ``refinement_mask`` is the union of active-layer sampling validity, before
    the caller checks final boxes. ``pred_logits`` contains raw decoder scores.
    """
    target_layers = set(model.refine_layers)
    active_layers = target_layers if active_layers is None else set(active_layers)
    if not active_layers <= target_layers:
        raise ValueError("active_layers must be a subset of model.refine_layers")

    decoder = model.decoder
    previous_query = None
    injected = None
    records = {}

    def capture_first_query(module, args, output):
        nonlocal previous_query
        previous_query = output.detach()

    with torch.autocast(device_type=features[0].device.type, enabled=False):
        features = [feature.detach().float() for feature in features]
        support = weighting_function(decoder.reg_max, decoder.up, decoder.reg_scale)

        def decode(logits, distribution):
            return distance2bbox(
                distribution["ref_points"], decoder.integral(logits, support),
                distribution["reg_scale"],
            )

        def trace_layer(index, query, previous_boxes, distribution):
            nonlocal previous_query, injected
            if index not in target_layers:
                previous_query = query.detach()
                return query

            head = decoder.dec_bbox_head[index]
            logits_before = head(query + previous_query) + distribution["pred_corners"]
            boxes_before = decode(logits_before, distribution)
            if index in active_layers:
                delta, valid = model.refiner(
                    features[model.feature_level], query, previous_boxes,
                    image_size, distribution,
                )
                delta = delta.to(query.dtype)
                refined_query = query + delta
                logits_after = head(refined_query + previous_query) + distribution["pred_corners"]
                boxes_after = decode(logits_after, distribution)
            else:
                delta = torch.zeros_like(query)
                valid = torch.zeros(query.shape[:2], dtype=torch.bool, device=query.device)
                refined_query = query
                logits_after, boxes_after = logits_before, boxes_before
            injected = valid if injected is None else injected | valid
            records[index] = {
                "boxes_before": boxes_before,
                "boxes_after": boxes_after,
                "logits_before": logits_before,
                "logits_after": logits_after,
                "query_before": query,
                "query_delta": delta,
                "valid": valid,
            }
            previous_query = refined_query.detach()
            return refined_query

        hook = decoder.decoder.layers[0].register_forward_hook(capture_first_query)
        try:
            outputs = decoder(features, query_refiner=trace_layer)
        finally:
            hook.remove()

    if injected is None:
        injected = torch.zeros(
            outputs["pred_boxes"].shape[:2], dtype=torch.bool,
            device=outputs["pred_boxes"].device,
        )
    return {
        "pred_boxes": outputs["pred_boxes"],
        "pred_logits": outputs["pred_logits"],
        "refinement_mask": injected,
        "layers": records,
    }
