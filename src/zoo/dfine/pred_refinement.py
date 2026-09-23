"""Frozen D-FINE with prediction-guided strip refinement for every query."""

from collections.abc import Mapping
from pathlib import Path

import torch

from ...core import register
from .dfine import DFINE
from .pred_strip_head import PredictionStripHead


@register()
class DFINEPredRefiner(DFINE):
    """Train only the strip head; targets are consumed separately by the loss."""

    __inject__ = ["backbone", "encoder", "decoder"]
    requires_gt = False
    oracle_evaluation = False

    def __init__(
        self,
        backbone,
        encoder,
        decoder,
        feature_level=0,
        hidden_dim=64,
        num_points=5,
        kernel_size=3,
        offset_pixels=1.0,
        max_log_scale=2.0,
        sampling_mode="box",
        baseline_checkpoint=None,
    ):
        super().__init__(backbone, encoder, decoder)
        if feature_level < 0:
            raise ValueError("feature_level must be nonnegative")
        self.feature_level = feature_level
        self.refiner = PredictionStripHead(
            in_channels=encoder.hidden_dim,
            query_dim=decoder.hidden_dim,
            hidden_dim=hidden_dim,
            num_points=num_points,
            kernel_size=kernel_size,
            offset_pixels=offset_pixels,
            max_log_scale=max_log_scale,
            sampling_mode=sampling_mode,
        )
        self._baseline_loaded = False
        for module in (self.backbone, self.encoder, self.decoder):
            module.requires_grad_(False)
            module.eval()
        if baseline_checkpoint is not None:
            self.load_baseline_checkpoint(baseline_checkpoint)

    def train(self, mode=True):
        super().train(mode)
        for module in (self.backbone, self.encoder, self.decoder):
            module.eval()
        return self

    def _validate_baseline_state(self, state_dict):
        expected = {
            key: value for key, value in self.state_dict().items()
            if key.startswith(("backbone.", "encoder.", "decoder."))
        }
        missing = [key for key in expected if key not in state_dict]
        wrong_shape = [
            key for key, value in expected.items()
            if key in state_dict and state_dict[key].shape != value.shape
        ]
        if missing or wrong_shape:
            raise ValueError(
                "Prediction refinement requires a complete baseline with matching architecture "
                f"and classes; missing={missing[:8]}, shape_mismatch={wrong_shape[:8]}"
            )
        return expected

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        self._validate_baseline_state(state_dict)
        result = super().load_state_dict(state_dict, strict=strict, **kwargs)
        self._baseline_loaded = True
        return result

    def load_baseline_checkpoint(self, path):
        """Load detector weights only, leaving the new head freshly initialized."""
        checkpoint = torch.load(Path(path).expanduser(), map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Expected a D-FINE checkpoint or a tensor state dict")
        if isinstance(checkpoint.get("ema"), Mapping) and "module" in checkpoint["ema"]:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint.get("model", checkpoint)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        expected = self._validate_baseline_state(state)
        self.load_state_dict({key: state[key] for key in expected}, strict=False)

    def forward(self, images, targets=None):
        # targets is intentionally unused: train/eval share the same prediction path.
        if not self._baseline_loaded:
            raise RuntimeError(
                "Load a trained D-FINE baseline using baseline_checkpoint, --tuning, "
                "or a complete prediction-refiner --resume checkpoint before running this model"
            )
        if images.ndim != 4 or images.shape[0] == 0:
            raise ValueError("Expected a nonempty BCHW image batch")
        with torch.no_grad():
            features = self.encoder(self.backbone(images))
            if self.feature_level >= len(features):
                raise ValueError("feature_level is outside the Encoder output list")
            baseline = self.decoder(features, return_query=True)
            boxes = baseline["pred_boxes"].detach().float()
            logits = baseline["pred_logits"].detach()
            queries = baseline["pred_queries"].detach()

        batch_size, num_queries = boxes.shape[:2]
        batch_ids = torch.arange(batch_size, device=boxes.device).repeat_interleave(num_queries)
        refined, valid = self.refiner(
            features[self.feature_level].detach(),
            queries.reshape(-1, queries.shape[-1]),
            boxes.reshape(-1, 4),
            batch_ids,
            images.shape[-2:],
        )
        valid = valid.reshape(batch_size, num_queries) & torch.isfinite(logits).all(-1)
        refined = torch.where(valid[..., None], refined.reshape_as(boxes), boxes)
        return {
            "pred_logits": logits,
            "pred_boxes": refined,
            "baseline_boxes": boxes,
            "refinement_mask": valid,
            "refinement_zero_loss": self.refiner.zero_loss(),
        }
