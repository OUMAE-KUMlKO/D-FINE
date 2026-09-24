"""Prediction-guided strip sampling with the existing box residual head."""

import torch
import torch.nn.functional as F

from .dfine_utils import weighting_function
from .gt_strip_head import GTStripHead


def encode_distribution_candidates(
    projected, grid, point_valid, batch_indices, strip_encoder, edge_fusion,
    hidden_dim, num_points, query_chunk_size,
):
    """Encode [N, 4, K] strip candidates without replicating feature maps."""
    n, _, k = grid.shape[:3]
    normal_points = grid.shape[-2]
    edge_features = projected.new_zeros((n, 4, k, hidden_dim))
    for image_index in range(len(projected)):
        image_rows = torch.where(batch_indices == image_index)[0]
        for start in range(0, len(image_rows), query_chunk_size):
            rows = image_rows[start:start + query_chunk_size]
            if not len(rows):
                continue
            sampled = F.grid_sample(
                projected[image_index:image_index + 1],
                grid[rows].reshape(1, -1, normal_points, 2), mode="bilinear",
                padding_mode="border", align_corners=False,
            ).reshape(hidden_dim, len(rows), 4, k, num_points, normal_points)
            sampled = sampled.permute(1, 2, 3, 0, 4, 5)
            sampled = sampled * point_valid[rows, :, :, None].to(sampled.dtype)
            encoded = strip_encoder(sampled.reshape(
                len(rows) * 4 * k, hidden_dim, num_points, normal_points
            ))
            mask = point_valid[rows].reshape(
                len(rows) * 4 * k, 1, num_points, normal_points
            ).to(encoded.dtype)
            pooled = (encoded * mask).sum(2) / mask.sum(2).clamp_min(1)
            fused = edge_fusion(
                pooled.transpose(1, 2).reshape(len(rows) * 4 * k, -1)
            ).reshape(len(rows), 4, k, hidden_dim)
            edge_features = edge_features.index_copy(0, rows, fused)
    return edge_features


class PredictionStripHead(GTStripHead):
    """Use predicted boxes or full edge distributions to guide strip sampling."""

    def __init__(self, in_channels, query_dim, hidden_dim=64, num_points=5,
                 kernel_size=3, offset_pixels=1.0, max_log_scale=2.0,
                 sampling_mode="box", query_chunk_size=32):
        if sampling_mode not in {"box", "distribution"}:
            raise ValueError("sampling_mode must be 'box' or 'distribution'.")
        if not isinstance(query_chunk_size, int) or query_chunk_size < 1:
            raise ValueError("query_chunk_size must be a positive integer.")
        super().__init__(
            in_channels=in_channels, query_dim=query_dim, hidden_dim=hidden_dim,
            num_points=num_points, kernel_size=kernel_size,
            offset_pixels=offset_pixels, max_log_scale=max_log_scale,
            feature_mode="real",
        )
        self.sampling_mode = sampling_mode
        self.query_chunk_size = query_chunk_size

    @staticmethod
    def _masked_candidate_weights(logits, candidate_valid):
        logits = logits.detach().float()
        finite = torch.isfinite(logits).all(dim=-1, keepdim=True)
        safe_logits = torch.where(finite, logits, torch.zeros_like(logits))
        probabilities = safe_logits.softmax(dim=-1)
        probabilities = torch.where(finite, probabilities, torch.zeros_like(probabilities))
        usable = candidate_valid & finite
        has_candidate = usable.any(dim=-1, keepdim=True)
        masked_logits = safe_logits.masked_fill(~usable, float("-inf"))
        masked_logits = torch.where(has_candidate, masked_logits, torch.zeros_like(masked_logits))
        weights = masked_logits.softmax(dim=-1) * usable.to(masked_logits.dtype)
        excluded = (probabilities * ~candidate_valid).sum(dim=-1)
        return weights, probabilities, excluded

    @staticmethod
    def _distribution_geometry(distribution, n, device):
        required = {"pred_corners", "ref_points", "up", "reg_scale"}
        if distribution is None or required - set(distribution):
            raise ValueError(f"distribution must contain {sorted(required)}")
        logits, refs = distribution["pred_corners"], distribution["ref_points"]
        if logits.ndim != 2 or logits.shape[0] != n or logits.shape[1] % 4:
            raise ValueError("pred_corners must have shape [N, 4 * K].")
        if refs.shape != (n, 4):
            raise ValueError("ref_points must have shape [N, 4].")
        k = logits.shape[1] // 4
        if k < 3:
            raise ValueError("The distribution must contain at least three bins.")
        logits = logits.detach().to(device=device, dtype=torch.float32).reshape(n, 4, k)
        refs = refs.detach().to(device=device, dtype=torch.float32)
        up = torch.as_tensor(distribution["up"], device=device).detach().float().reshape(-1)
        reg_scale = torch.as_tensor(distribution["reg_scale"], device=device).detach().float().reshape(-1)
        scalar_valid = (
            up.numel() > 0 and reg_scale.numel() == 1
            and bool(torch.isfinite(up).all()) and bool(torch.isfinite(reg_scale).all())
            and float(reg_scale[0].abs()) > 0
        )
        distribution_valid = (
            torch.isfinite(logits).flatten(1).all(dim=1)
            & torch.isfinite(refs).all(dim=-1) & (refs[:, 2:] > 0).all(dim=-1)
        )
        if scalar_valid:
            support = weighting_function(k - 1, up, reg_scale).detach().float()
            scale = reg_scale[0].abs()
        else:
            distribution_valid &= False
            support, scale = logits.new_zeros(k), logits.new_tensor(1.0)
        x, y, w, h = refs.unbind(-1)
        positions = torch.stack((
            x[:, None] - w[:, None] / 2 - support * w[:, None] / scale,
            y[:, None] - h[:, None] / 2 - support * h[:, None] / scale,
            x[:, None] + w[:, None] / 2 + support * w[:, None] / scale,
            y[:, None] + h[:, None] / 2 + support * h[:, None] / scale,
        ), dim=1)
        positions = torch.where(
            distribution_valid[:, None, None], positions, torch.zeros_like(positions)
        ).detach()
        safe_logits = torch.where(
            distribution_valid[:, None, None], logits, torch.zeros_like(logits)
        )
        probabilities = safe_logits.softmax(-1)
        probabilities = torch.where(
            distribution_valid[:, None, None], probabilities, torch.zeros_like(probabilities)
        )
        return positions, probabilities, distribution_valid

    def _candidate_grid(self, positions, pred_boxes, distribution_valid, image_size):
        image_h, image_w = image_size
        if image_h <= 0 or image_w <= 0:
            raise ValueError("image_size must contain positive (height, width).")
        boxes = pred_boxes.detach().float()
        box_valid = torch.isfinite(boxes).all(-1) & (boxes[:, 2:] > 0).all(-1)
        safe = torch.where(box_valid[:, None], boxes, torch.zeros_like(boxes))
        x, y, w, h = safe.unbind(-1)
        left, top, right, bottom = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        t = torch.linspace(0, 1, self.num_points, device=boxes.device)
        along_x, along_y = left[:, None] + w[:, None] * t, top[:, None] + h[:, None] * t
        normal_points = getattr(self, "normal_points", 3)
        offsets = torch.linspace(1, -1, normal_points, device=boxes.device, dtype=boxes.dtype)
        dx, dy = offsets * self.offset_pixels / image_w, offsets * self.offset_pixels / image_h
        n, _, k = positions.shape
        span = (n, k, self.num_points, normal_points)

        def vertical(edge, delta):
            return torch.stack((
                (edge[:, :, None, None] + delta).expand(span),
                along_y[:, None, :, None].expand(span),
            ), -1)

        def horizontal(edge, delta):
            return torch.stack((
                along_x[:, None, :, None].expand(span),
                (edge[:, :, None, None] + delta).expand(span),
            ), -1)

        coords = torch.stack((
            vertical(positions[:, 0], dx), horizontal(positions[:, 1], dy),
            vertical(positions[:, 2], -dx), horizontal(positions[:, 3], -dy),
        ), dim=1)
        point_valid = (
            torch.isfinite(coords).all(-1) & (coords >= 0).all(-1) & (coords <= 1).all(-1)
            & box_valid[:, None, None, None, None]
            & distribution_valid[:, None, None, None, None]
        )
        coords = torch.where(point_valid[..., None], coords, torch.zeros_like(coords))
        candidate_valid = point_valid.any(dim=3).all(dim=-1)
        return coords * 2 - 1, point_valid, candidate_valid

    def _distribution_forward(self, feature_map, queries, pred_boxes, batch_indices,
                              image_size, distribution, return_diagnostics,
                              candidate_indices=None):
        n = len(pred_boxes)
        if queries.shape != (n, self.query_dim) or pred_boxes.shape != (n, 4) or batch_indices.shape != (n,):
            raise ValueError("queries, boxes, and batch_indices must describe the same N queries.")
        if batch_indices.dtype != torch.long:
            raise ValueError("batch_indices must have dtype torch.long.")
        if any(value.device != feature_map.device for value in (queries, pred_boxes, batch_indices)):
            raise ValueError("All tensor inputs must share a device.")
        if len(batch_indices) and ((batch_indices < 0).any() or (batch_indices >= len(feature_map)).any()):
            raise ValueError("batch_indices contains an image index outside the batch.")
        with torch.autocast(device_type=feature_map.device.type, enabled=False):
            positions, _, distribution_valid = self._distribution_geometry(distribution, n, feature_map.device)
            logits = distribution["pred_corners"].detach().float().reshape(
                n, 4, positions.shape[-1]
            )
            grid, point_valid, candidate_valid = self._candidate_grid(
                positions, pred_boxes, distribution_valid, image_size
            )
            weights, probabilities, excluded = self._masked_candidate_weights(logits, candidate_valid)
            selected_edges = None
            if candidate_indices is not None:
                candidate_indices = candidate_indices.to(
                    device=feature_map.device, dtype=torch.long
                )
                if candidate_indices.shape != (n, 4):
                    raise ValueError("candidate_indices must have shape [N, 4].")
                in_range = (candidate_indices >= 0) & (candidate_indices < positions.shape[-1])
                safe_indices = candidate_indices.clamp(0, positions.shape[-1] - 1)
                selected_edges = in_range & candidate_valid.gather(
                    -1, safe_indices[..., None]
                ).squeeze(-1)
                weights = torch.zeros_like(weights).scatter(
                    -1, safe_indices[..., None], selected_edges[..., None].to(weights.dtype)
                )
            k = positions.shape[-1]
            if not n:
                result = pred_boxes.float() + self.zero_loss()
                valid = torch.empty(0, dtype=torch.bool, device=feature_map.device)
                if not return_diagnostics:
                    return result, valid
                return result, valid, {
                    "candidate_positions": positions, "candidate_valid": candidate_valid,
                    "candidate_probabilities": probabilities,
                    "excluded_probability_mass": excluded,
                }
            projected = self.projection(feature_map.float())
            edge_features = encode_distribution_candidates(
                projected, grid, point_valid, batch_indices,
                self.strip_encoder, self.edge_fusion, self.hidden_dim,
                self.num_points, self.query_chunk_size,
            )
            edge_features = (edge_features * weights[..., None]).sum(dim=2)
            original = pred_boxes.float()
            valid = (
                distribution_valid & candidate_valid.any(-1).all(-1)
                & torch.isfinite(original).all(-1) & (original[:, 2:] > 0).all(-1)
                & torch.isfinite(queries).all(-1)
            )
            if selected_edges is not None:
                valid &= selected_edges.all(-1)
            safe_boxes = torch.where(valid[:, None], original, torch.zeros_like(original))
            safe_queries = torch.where(valid[:, None], queries.float(), torch.zeros_like(queries, dtype=torch.float32))
            inputs = torch.cat((safe_queries, self.box_embedding(safe_boxes), edge_features.reshape(n, -1)), -1)
            deltas = self.refiner(inputs)
            centers = safe_boxes[:, :2] + safe_boxes[:, 2:] * deltas[:, :2]
            sizes = safe_boxes[:, 2:] * deltas[:, 2:].clamp(
                -self.max_log_scale, self.max_log_scale
            ).exp()
            refined = torch.cat((centers, sizes), -1)
            result = torch.where(valid[:, None], refined, original) + self.zero_loss()
            if not return_diagnostics:
                return result, valid
            return result, valid, {
                "candidate_positions": positions, "candidate_valid": candidate_valid,
                "candidate_probabilities": probabilities,
                "excluded_probability_mass": excluded,
            }

    def forward(self, feature_map, queries, pred_boxes, batch_indices, image_size,
                distribution=None, return_diagnostics=False):
        if self.sampling_mode == "distribution":
            return self._distribution_forward(
                feature_map, queries, pred_boxes, batch_indices, image_size,
                distribution, return_diagnostics,
            )
        result = super().forward(
            feature_map, queries, pred_boxes, pred_boxes.detach(), batch_indices, image_size
        )
        return (result[0], result[1], {}) if return_diagnostics else result
