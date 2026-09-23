"""Distribution-guided strip adapter for decoder query updates."""

import math

import torch
import torch.nn as nn

from .pred_strip_head import PredictionStripHead, encode_distribution_candidates


class DistributionStripQueryAdapter(nn.Module):
    """Fuse distribution-strip evidence into a residual decoder query update."""

    def __init__(
        self, in_channels, query_dim, hidden_dim=64, num_points=5,
        kernel_size=3, offset_pixels=1.0, query_chunk_size=32,
    ):
        super().__init__()
        if min(in_channels, query_dim, hidden_dim) < 1:
            raise ValueError("Feature dimensions must be positive.")
        if num_points < 2:
            raise ValueError("num_points must be at least 2.")
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("kernel_size must be a positive odd integer.")
        if not math.isfinite(offset_pixels) or offset_pixels <= 0:
            raise ValueError("offset_pixels must be finite and positive.")
        if not isinstance(query_chunk_size, int) or query_chunk_size < 1:
            raise ValueError("query_chunk_size must be a positive integer.")
        self.in_channels = in_channels
        self.query_dim = query_dim
        self.hidden_dim = hidden_dim
        self.num_points = num_points
        self.offset_pixels = float(offset_pixels)
        self.query_chunk_size = query_chunk_size
        self.projection = nn.Conv2d(in_channels, hidden_dim, 1)
        self.strip_encoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, (kernel_size, 1),
                      padding=(kernel_size // 2, 0), groups=hidden_dim),
            nn.ReLU(), nn.Conv2d(hidden_dim, hidden_dim, 1), nn.ReLU(),
        )
        self.edge_fusion = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.ReLU()
        )
        self.query_projection = nn.Linear(4 * hidden_dim, query_dim)
        nn.init.zeros_(self.query_projection.weight)
        nn.init.zeros_(self.query_projection.bias)

    def zero_loss(self):
        return sum(parameter.reshape(-1)[0] * 0.0 for parameter in self.parameters())

    def forward(self, feature_map, queries, pred_boxes, image_size, distribution):
        if feature_map.ndim != 4 or feature_map.shape[1] != self.in_channels:
            raise ValueError("feature_map must have shape [B, in_channels, H, W].")
        if queries.ndim != 3 or queries.shape[-1] != self.query_dim:
            raise ValueError("queries must have shape [B, Q, query_dim].")
        if pred_boxes.shape != (*queries.shape[:2], 4):
            raise ValueError("pred_boxes must have shape [B, Q, 4].")
        batch_size, query_count = queries.shape[:2]
        if len(feature_map) != batch_size:
            raise ValueError("Features, queries, and boxes must share batch size.")
        required = {"pred_corners", "ref_points", "up", "reg_scale"}
        if distribution is None or required - set(distribution):
            raise ValueError(f"distribution must contain {sorted(required)}")
        corners, refs = distribution["pred_corners"], distribution["ref_points"]
        if corners.ndim != 3 or corners.shape[:2] != (batch_size, query_count):
            raise ValueError("pred_corners must have shape [B, Q, 4 * K].")
        if refs.shape != (batch_size, query_count, 4):
            raise ValueError("ref_points must have shape [B, Q, 4].")
        flat_boxes = pred_boxes.reshape(-1, 4)
        flat_queries = queries.reshape(-1, self.query_dim)
        flat_distribution = {
            "pred_corners": corners.reshape(-1, corners.shape[-1]),
            "ref_points": refs.reshape(-1, 4),
            "up": distribution["up"], "reg_scale": distribution["reg_scale"],
        }
        n = len(flat_boxes)
        with torch.autocast(device_type=feature_map.device.type, enabled=False):
            positions, _, distribution_valid = PredictionStripHead._distribution_geometry(
                flat_distribution, n, feature_map.device
            )
            grid, point_valid, candidate_valid = PredictionStripHead._candidate_grid(
                self, positions, flat_boxes, distribution_valid, image_size
            )
            logits = corners.detach().float().reshape(n, 4, positions.shape[-1])
            weights, _, _ = PredictionStripHead._masked_candidate_weights(
                logits, candidate_valid
            )
            valid = (
                distribution_valid & candidate_valid.any(-1).all(-1)
                & torch.isfinite(flat_boxes).all(-1)
                & (flat_boxes[:, 2:] > 0).all(-1)
                & torch.isfinite(flat_queries).all(-1)
            )
            if not n:
                delta = flat_queries.new_zeros((0, self.query_dim), dtype=torch.float32)
                return (
                    (delta + self.zero_loss()).reshape(
                        batch_size, query_count, self.query_dim
                    ),
                    valid.reshape(batch_size, query_count),
                )
            batch_indices = torch.arange(
                batch_size, device=feature_map.device
            ).repeat_interleave(query_count)
            candidate_features = encode_distribution_candidates(
                self.projection(feature_map.float()), grid, point_valid, batch_indices,
                self.strip_encoder, self.edge_fusion, self.hidden_dim,
                self.num_points, self.query_chunk_size,
            )
            edge_features = (candidate_features * weights[..., None]).sum(dim=2)
            safe_features = torch.where(
                valid[:, None, None], edge_features, torch.zeros_like(edge_features)
            )
            delta = self.query_projection(safe_features.reshape(n, -1))
            delta = torch.where(valid[:, None], delta, torch.zeros_like(delta))
            delta = delta + self.zero_loss()
            return delta.reshape(batch_size, query_count, -1), valid.reshape(
                batch_size, query_count
            )
