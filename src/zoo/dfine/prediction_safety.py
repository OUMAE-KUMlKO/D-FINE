"""Numerical guards for decoder distribution refinement."""

import torch


def _detach_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _detach_to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_detach_to_cpu(item) for item in value)
    return value


class PredictionSafetyError(RuntimeError):
    """Stop prediction when the decoder's uncorrected prior is unusable."""

    def __init__(self, message, diagnostic):
        super().__init__(message)
        self.diagnostic = _detach_to_cpu(diagnostic)


def _raise_invalid_prior(
    layer_index, stage, invalid, prior_logits, ref_points, boxes=None, dn_count=0,
):
    query_indices = torch.nonzero(invalid, as_tuple=False)
    query_indices[:, 1] += dn_count
    raise PredictionSafetyError(
        f"Invalid decoder prior at layer {layer_index} during {stage}",
        {
            "layer_index": int(layer_index),
            "stage": stage,
            "dn_count": int(dn_count),
            "query_indices": query_indices,
            "prior_logits": prior_logits,
            "corrected_logits": None,
            "boxes": boxes,
            "ref_points": ref_points,
        },
    )


def validate_prior_inputs(layer_index, prior_logits, ref_points, dn_count=0):
    """Reject invalid prior tensors before distribution decoding."""
    logits_finite = torch.isfinite(prior_logits[:, dn_count:]).flatten(2).all(-1)
    if not bool(logits_finite.all()):
        _raise_invalid_prior(
            layer_index, "prior_logits", ~logits_finite,
            prior_logits, ref_points, dn_count=dn_count,
        )
    references_finite = torch.isfinite(ref_points[:, dn_count:]).flatten(2).all(-1)
    if not bool(references_finite.all()):
        _raise_invalid_prior(
            layer_index, "prior_ref_points", ~references_finite,
            prior_logits, ref_points, dn_count=dn_count,
        )


def validate_prior_boxes(
    layer_index, prior_logits, prior_boxes, ref_points, dn_count=0,
):
    """Reject nonfinite priors; count finite nonpositive geometry separately.

    Native D-FINE can decode finite nonpositive sizes before G5 refinement.
    Preserve that distribution and the existing downstream box conversion;
    its geometry alone must not stop training or prevent a valid correction.
    """
    matching_boxes = prior_boxes[:, dn_count:]
    boxes_finite = torch.isfinite(matching_boxes).all(-1)
    if not bool(boxes_finite.all()):
        _raise_invalid_prior(
            layer_index, "prior_boxes", ~boxes_finite,
            prior_logits, ref_points, prior_boxes, dn_count,
        )
    return int((~(matching_boxes[..., 2:] > 0).all(-1)).sum().item())


def select_safe_correction(
    layer_index, prior_logits, prior_boxes, corrected_logits, ref_points, decode,
    dn_count=0,
):
    """Select corrected distributions query-wise, decoding only finite inputs."""
    if corrected_logits.shape != prior_logits.shape:
        raise PredictionSafetyError(
            f"Corrected logits shape mismatch at layer {layer_index}",
            {
                "layer_index": int(layer_index),
                "stage": "corrected_shape",
                "dn_count": int(dn_count),
                "query_indices": torch.empty((0, 2), dtype=torch.long),
                "prior_logits": prior_logits,
                "corrected_logits": corrected_logits,
                "boxes": prior_boxes,
                "ref_points": ref_points,
            },
        )

    # The refiner may compute in FP32, but downstream decoding uses the prior's
    # actual dtype. Validate after this conversion so overflow cannot escape.
    corrected_logits = corrected_logits.to(
        device=prior_logits.device, dtype=prior_logits.dtype,
    )
    matching_corrected = corrected_logits[:, dn_count:]
    matching_prior = prior_logits[:, dn_count:]
    logits_valid = torch.isfinite(matching_corrected).flatten(2).all(-1)
    safe_matching = torch.where(
        logits_valid[..., None], matching_corrected, matching_prior,
    )
    decode_logits = torch.cat((corrected_logits[:, :dn_count], safe_matching), dim=1)
    candidate_boxes = decode(decode_logits)
    matching_boxes = candidate_boxes[:, dn_count:]
    boxes_valid = (
        torch.isfinite(matching_boxes).all(-1)
        & (matching_boxes[..., 2:] > 0).all(-1)
    )
    corrected_valid = logits_valid & boxes_valid
    if bool(corrected_valid.all()):
        return corrected_logits, candidate_boxes, {
            "checked_queries": int(corrected_valid.numel()),
            "fallback_queries": 0,
        }
    selected_matching = torch.where(
        corrected_valid[..., None], matching_corrected, matching_prior,
    )
    selected_logits = torch.cat(
        (corrected_logits[:, :dn_count], selected_matching), dim=1,
    )
    # Re-decode the selected finite distributions. This excludes an invalid
    # candidate decode from the output graph instead of masking it afterwards.
    selected_boxes = decode(selected_logits)
    return selected_logits, selected_boxes, {
        "checked_queries": int(corrected_valid.numel()),
        "fallback_queries": int((~corrected_valid).sum().item()),
    }
