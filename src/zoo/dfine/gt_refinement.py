"""Frozen D-FINE with GT-only strip sampling for stage-one oracle experiments."""

from collections.abc import Mapping
from pathlib import Path

import torch

from ...core import register
from .box_ops import box_xyxy_to_cxcywh
from .dfine import DFINE
from .gt_strip_head import GTStripHead


@register()
class DFINEGTRefiner(DFINE):
    """GT is required in both training and evaluation; this is not a deployment model.

    Baseline state names are unchanged, so a matching D-FINE checkpoint can be
    loaded with the existing --tuning path. Only refiner.* is trainable.
    """

    __inject__ = ["backbone", "encoder", "decoder", "matcher"]
    requires_gt = True
    oracle_evaluation = True

    def __init__(
        self,
        backbone,
        encoder,
        decoder,
        matcher,
        feature_level=0,
        hidden_dim=64,
        num_points=5,
        kernel_size=3,
        offset_pixels=1.0,
        max_log_scale=2.0,
        feature_mode="real",
        random_seed=0,
        baseline_checkpoint=None,
        train_target_format="cxcywh_normalized",
        eval_target_format="xyxy_absolute",
    ):
        super().__init__(backbone, encoder, decoder)
        if feature_mode not in ("real", "constant", "random", "baseline"):
            raise ValueError("feature_mode must be real, constant, random, or baseline")
        if feature_level < 0:
            raise ValueError("feature_level must be nonnegative")
        self.matcher = matcher
        self.feature_level = feature_level
        self.feature_mode = feature_mode
        self.train_target_format = train_target_format
        self.eval_target_format = eval_target_format
        self.refiner = GTStripHead(
            in_channels=encoder.hidden_dim,
            query_dim=decoder.hidden_dim,
            hidden_dim=hidden_dim,
            num_points=num_points,
            kernel_size=kernel_size,
            offset_pixels=offset_pixels,
            max_log_scale=max_log_scale,
            feature_mode="real" if feature_mode == "baseline" else feature_mode,
            random_seed=random_seed,
        )
        self._baseline_loaded = False
        for module in (self.backbone, self.encoder, self.decoder, self.matcher):
            module.requires_grad_(False)
            module.eval()
        if baseline_checkpoint is not None:
            self.load_baseline_checkpoint(baseline_checkpoint)

    def train(self, mode=True):
        super().train(mode)
        for module in (self.backbone, self.encoder, self.decoder, self.matcher):
            module.eval()
        return self

    def _validate_baseline_state(self, state_dict):
        expected = {
            key: value
            for key, value in self.state_dict().items()
            if key.startswith(("backbone.", "encoder.", "decoder."))
        }
        missing = [key for key in expected if key not in state_dict]
        wrong_shape = [
            key
            for key, value in expected.items()
            if key in state_dict and state_dict[key].shape != value.shape
        ]
        if missing or wrong_shape:
            raise ValueError(
                "Stage one requires a complete baseline with matching architecture and classes; "
                f"missing={missing[:8]}, shape_mismatch={wrong_shape[:8]}"
            )
        return expected

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        self._validate_baseline_state(state_dict)
        result = super().load_state_dict(state_dict, strict=strict, **kwargs)
        self._baseline_loaded = True
        return result

    def load_baseline_checkpoint(self, path):
        """Load baseline weights only; use normal resume for a trained refiner."""
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

    @staticmethod
    def _prepare_targets(targets, image_size, device, target_format, num_classes):
        formats = (
            "cxcywh_normalized", "xyxy_absolute", "xyxy_normalized", "cxcywh_absolute"
        )
        if target_format not in formats:
            raise ValueError(f"Unsupported target_format {target_format!r}; choose {formats}")
        height, width = image_size
        normalized = []
        ignored = []
        for target in targets:
            boxes = torch.as_tensor(target["boxes"], device=device).float().reshape(-1, 4)
            boxes = boxes.as_subclass(torch.Tensor).detach()
            labels = torch.as_tensor(target["labels"], device=device, dtype=torch.long).reshape(-1)
            if len(boxes) != len(labels):
                raise ValueError("Each target must have the same number of boxes and labels")
            if target_format.endswith("_absolute"):
                boxes = boxes / boxes.new_tensor([width, height, width, height])
            if target_format.startswith("xyxy_"):
                boxes = box_xyxy_to_cxcywh(boxes)
            keep = torch.isfinite(boxes).all(-1) & (boxes[:, 2:] > 0).all(-1)
            for flag in ("iscrowd", "ignore"):
                if flag in target:
                    ignored_flags = torch.as_tensor(target[flag], device=device).reshape(-1)
                    if len(ignored_flags) != len(boxes):
                        raise ValueError(f"{flag} must have one value per target box")
                    keep &= ~ignored_flags.bool()
            ignored.append(int((~keep).sum().item()))
            boxes, labels = boxes[keep], labels[keep]
            if ((labels < 0) | (labels >= num_classes)).any():
                raise ValueError("GT labels must use the baseline's contiguous class indices")
            item = {"boxes": boxes, "labels": labels}
            if "image_id" in target:
                item["image_id"] = target["image_id"]
            normalized.append(item)
        return normalized, ignored

    @torch.no_grad()
    def _match_baseline(self, boxes, logits, targets):
        """Match once, excluding invalid predictions without changing returned detections."""
        indices = []
        valid_boxes = torch.isfinite(boxes).all(-1) & (boxes[..., 2:] > 0).all(-1)
        valid_predictions = valid_boxes & torch.isfinite(logits).all(-1)
        for batch_index, target in enumerate(targets):
            query_ids = torch.where(valid_predictions[batch_index])[0]
            if not len(query_ids) or not len(target["boxes"]):
                empty = torch.empty(0, dtype=torch.long, device=boxes.device)
                indices.append((empty, empty))
                continue
            matches = self.matcher(
                {
                    "pred_boxes": boxes[batch_index, query_ids].float().unsqueeze(0),
                    "pred_logits": logits[batch_index, query_ids].float().unsqueeze(0),
                },
                [target],
            )["indices"][0]
            src, dst = (value.to(boxes.device) for value in matches)
            indices.append((query_ids[src], dst))
        return indices, (~valid_boxes).sum(1)

    def forward(self, images, targets=None, target_format=None):
        if targets is None:
            raise ValueError("DFINEGTRefiner requires GT targets, including during oracle evaluation")
        if not self._baseline_loaded:
            raise RuntimeError(
                "Load a trained D-FINE baseline using baseline_checkpoint, --tuning, "
                "or a complete stage-one --resume checkpoint before running this model"
            )
        if images.ndim != 4 or images.shape[0] == 0 or len(targets) != images.shape[0]:
            raise ValueError("Expected a nonempty BCHW batch and one target dictionary per image")
        if self.training and self.feature_mode == "baseline":
            raise RuntimeError("The G0 baseline control is evaluation-only")
        image_size = images.shape[-2:]
        with torch.no_grad():
            features = self.encoder(self.backbone(images))
            if self.feature_level >= len(features):
                raise ValueError("feature_level is outside the Encoder output list")
            baseline = self.decoder(features, return_query=True)
            boxes = baseline["pred_boxes"].detach().float()
            logits = baseline["pred_logits"].detach()
            queries = baseline["pred_queries"].detach()
            target_format = target_format or (
                self.train_target_format if self.training else self.eval_target_format
            )
            canonical_targets, ignored = self._prepare_targets(
                targets, image_size, images.device, target_format, logits.shape[-1]
            )
            indices, invalid_baseline = self._match_baseline(boxes, logits, canonical_targets)

        batch_ids = torch.cat([
            torch.full_like(src, batch_index)
            for batch_index, (src, _) in enumerate(indices)
        ])
        query_ids = torch.cat([src for src, _ in indices])
        gt_boxes = torch.cat([
            target["boxes"][dst]
            for target, (_, dst) in zip(canonical_targets, indices)
        ])
        matched = len(query_ids)
        refined = boxes.clone()
        refined_mask = torch.zeros(boxes.shape[:2], device=boxes.device, dtype=torch.bool)
        if self.feature_mode == "baseline":
            valid = torch.zeros(matched, device=boxes.device, dtype=torch.bool)
        else:
            image_ids = (
                [target["image_id"] for target in targets]
                if all("image_id" in target for target in targets) else None
            )
            new_boxes, valid = self.refiner(
                features[self.feature_level].detach(),
                queries[batch_ids, query_ids],
                boxes[batch_ids, query_ids],
                gt_boxes,
                batch_ids,
                image_size,
                image_ids=image_ids,
            )
            refined[batch_ids, query_ids] = new_boxes
            refined_mask[batch_ids[valid], query_ids[valid]] = True

        refinement_indices = []
        offset = 0
        for src, dst in indices:
            accepted = valid[offset:offset + len(src)]
            refinement_indices.append((src[accepted], dst[accepted]))
            offset += len(src)
        # Preserve each image's counters so evaluation can discard sampler
        # padding even when real and padded images share a batch.
        matched_per_image = boxes.new_tensor([len(src) for src, _ in indices], dtype=torch.long)
        refined_per_image = refined_mask.sum(1)
        total_gt_per_image = boxes.new_tensor(
            [len(target["boxes"]) for target in canonical_targets], dtype=torch.long
        )
        counts_per_image = {
            "matched": matched_per_image,
            "refined": refined_per_image,
            "ignored_gt": boxes.new_tensor(ignored, dtype=torch.long),
            "total_gt": total_gt_per_image,
            "unmatched_gt": total_gt_per_image - matched_per_image,
            "unmatched_queries": boxes.shape[1] - matched_per_image,
            "invalid_gt_samples": (
                torch.zeros_like(matched_per_image) if self.feature_mode == "baseline"
                else matched_per_image - refined_per_image
            ),
            "invalid_baseline_boxes": invalid_baseline,
        }
        return {
            "pred_logits": logits,
            "pred_boxes": refined,
            "baseline_boxes": boxes,
            "matched_indices": indices,
            "refinement_indices": refinement_indices,
            "refinement_targets": canonical_targets,
            "refinement_mask": refined_mask,
            "refinement_zero_loss": self.refiner.zero_loss(),
            "refinement_counts_per_image": counts_per_image,
            "refinement_counts": {
                key: value.sum() for key, value in counts_per_image.items()
            },
        }

    def deploy(self):
        raise RuntimeError("GT-guided stage-one diagnostics cannot be exported for GT-free deployment")
