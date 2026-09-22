"""GT-only boundary evidence for the first-stage localization oracle.

This module deliberately accepts no boundary distributions. GT coordinates enter
only the sampler; the regression network receives query, original box geometry,
and visual strip features. Coordinates are normalized to the network input.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GTStripHead(nn.Module):
    """Read complete GT edges and regress a residual to an existing box.

    Edges are ordered left, top, right, bottom. Each strip has ordered columns
    inside, on the boundary, outside. A query is eligible only when its original
    box, GT, and query are finite, both boxes have positive width/height, and
    every one of the 12 edge/column combinations has at least one image-valid
    sample. Ineligible queries retain their original boxes in every mode.

    The module runs its small trainable path in float32 under AMP. Keep its
    parameters in float32, as in normal autocast training.
    """

    def __init__(
        self,
        in_channels,
        query_dim,
        hidden_dim=64,
        num_points=5,
        kernel_size=3,
        offset_pixels=1.0,
        max_log_scale=2.0,
        feature_mode="real",
        random_seed=0,
    ):
        super().__init__()
        if min(in_channels, query_dim, hidden_dim) < 1:
            raise ValueError("Feature dimensions must be positive.")
        if num_points < 2:
            raise ValueError("num_points must be at least 2 to cover a GT edge.")
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("kernel_size must be a positive odd integer.")
        if not math.isfinite(offset_pixels) or offset_pixels <= 0:
            raise ValueError("offset_pixels must be finite and positive.")
        if not math.isfinite(max_log_scale) or not 0 < max_log_scale <= 20:
            raise ValueError("max_log_scale must be in (0, 20].")
        if feature_mode not in {"real", "constant", "random"}:
            raise ValueError("feature_mode must be real, constant, or random.")

        self.in_channels = in_channels
        self.query_dim = query_dim
        self.hidden_dim = hidden_dim
        self.num_points = num_points
        self.offset_pixels = float(offset_pixels)
        self.max_log_scale = float(max_log_scale)
        self.feature_mode = feature_mode
        self.random_seed = int(random_seed)

        self.projection = nn.Conv2d(in_channels, hidden_dim, 1)
        self.strip_encoder = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                (kernel_size, 1),
                padding=(kernel_size // 2, 0),
                groups=hidden_dim,
            ),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.ReLU(),
        )
        self.edge_fusion = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.ReLU()
        )
        self.box_embedding = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU())
        self.refiner = nn.Sequential(
            nn.Linear(query_dim + 5 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )
        nn.init.zeros_(self.refiner[-1].weight)
        nn.init.zeros_(self.refiner[-1].bias)

    def zero_loss(self):
        """Connect all trainable parameters even for empty or invalid batches.

        The caller may add this scalar to its loss when there are no eligible
        matched queries. This also keeps the constant-content control safe for
        DDP, where projection parameters intentionally have zero gradients.
        """
        return sum(parameter.reshape(-1)[0] * 0.0 for parameter in self.parameters())

    def _sampling_grid(self, gt_boxes, image_size):
        """Build [N, 4, S, 3, 2] grids solely from GT and input dimensions."""
        image_h, image_w = image_size
        if image_h <= 0 or image_w <= 0:
            raise ValueError("image_size must contain positive (height, width).")
        gt = gt_boxes.detach().float()
        gt_valid = torch.isfinite(gt).all(-1) & (gt[:, 2:] > 0).all(-1)
        safe_gt = torch.where(gt_valid[:, None], gt, torch.zeros_like(gt))
        x, y, w, h = safe_gt.unbind(-1)
        left, top, right, bottom = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        t = torch.linspace(0, 1, self.num_points, device=gt.device)
        along_x = left[:, None] + w[:, None] * t
        along_y = top[:, None] + h[:, None] * t
        # Positive offset points inward for left/top and outward for right/bottom.
        offsets = gt.new_tensor([1.0, 0.0, -1.0])
        dx = offsets * (self.offset_pixels / image_w)
        dy = offsets * (self.offset_pixels / image_h)
        span_shape = (len(gt), self.num_points, 3)
        vertical_y = along_y[:, :, None].expand(span_shape)
        horizontal_x = along_x[:, :, None].expand(span_shape)

        def vertical(edge, delta):
            edge_x = (edge[:, None, None] + delta).expand(span_shape)
            return torch.stack((edge_x, vertical_y), -1)

        def horizontal(edge, delta):
            edge_y = (edge[:, None, None] + delta).expand(span_shape)
            return torch.stack((horizontal_x, edge_y), -1)

        coords = torch.stack(
            (vertical(left, dx), horizontal(top, dy),
             vertical(right, -dx), horizontal(bottom, -dy)),
            dim=1,
        )
        point_valid = (
            torch.isfinite(coords).all(-1)
            & (coords >= 0).all(-1)
            & (coords <= 1).all(-1)
            & gt_valid[:, None, None, None]
        )
        # Mask unsafe coordinates before grid_sample; mask sampled content too.
        coords = torch.where(point_valid[..., None], coords, torch.zeros_like(coords))
        grid = coords * 2 - 1
        valid_gt = gt_valid & point_valid.any(dim=2).flatten(1).all(dim=1)
        return grid, point_valid, valid_gt

    def _random_feature_map(self, feature_map, image_ids):
        # One spatial field per image, shared by all GTs and all four edges.
        if self.training:
            return torch.randn(feature_map.shape, device=feature_map.device, dtype=torch.float32)
        if image_ids is None:
            raise ValueError("Random oracle evaluation requires stable image_ids.")
        ids = torch.as_tensor(image_ids).detach().cpu().reshape(-1).tolist()
        if len(ids) != feature_map.shape[0]:
            raise ValueError("image_ids must contain one stable ID per image.")
        fields = []
        for image_id in ids:
            generator = torch.Generator(device=feature_map.device)
            seed = (self.random_seed + (int(image_id) + 1) * 0x9E3779B1) % (2**63 - 1)
            generator.manual_seed(seed)
            fields.append(torch.randn(
                feature_map.shape[1:], generator=generator,
                device=feature_map.device, dtype=torch.float32,
            ))
        return torch.stack(fields)

    def sample_strips(self, feature_map, gt_boxes, batch_indices, image_size, image_ids=None):
        """Return strips [N, 4, Cb, S, 3], point mask, and GT eligibility.

        With ``align_corners=False``, normalized input positions map to feature
        pixels using x_feature = x_normalized * W_feature - 0.5. The point mask
        describes the network image extent [0, 1], independently of feature
        content and the original predictions. Border padding avoids attenuating
        valid samples on the image edge; points outside the image are masked.
        """
        if feature_map.ndim != 4 or feature_map.shape[1] != self.in_channels:
            raise ValueError("feature_map must have shape [B, in_channels, H, W].")
        if gt_boxes.ndim != 2 or gt_boxes.shape[-1] != 4:
            raise ValueError("gt_boxes must have shape [N, 4].")
        if batch_indices.shape != (len(gt_boxes),):
            raise ValueError("batch_indices must have shape [N].")
        if batch_indices.dtype != torch.long:
            raise ValueError("batch_indices must have dtype torch.long.")
        if gt_boxes.device != feature_map.device or batch_indices.device != feature_map.device:
            raise ValueError("Features, boxes, and batch indices must share a device.")
        if len(batch_indices) and (
            (batch_indices < 0).any() or (batch_indices >= len(feature_map)).any()
        ):
            raise ValueError("batch_indices contains an image index outside the batch.")

        with torch.autocast(device_type=feature_map.device.type, enabled=False):
            grid, point_valid, valid_gt = self._sampling_grid(gt_boxes, image_size)
            shape = (len(gt_boxes), 4, self.hidden_dim, self.num_points, 3)
            strips = feature_map.new_zeros(shape, dtype=torch.float32)
            if not len(gt_boxes):
                return strips, point_valid, valid_gt
            source = feature_map.float()
            if self.feature_mode == "random":
                source = self._random_feature_map(source, image_ids)
            projected = self.projection(source)

            # Do not replicate the entire feature map once per matched query.
            for image_index in range(len(feature_map)):
                rows = torch.where(batch_indices == image_index)[0]
                if not len(rows):
                    continue
                image_grid = grid[rows].reshape(1, -1, 3, 2)
                sampled = F.grid_sample(
                    projected[image_index:image_index + 1], image_grid,
                    mode="bilinear", padding_mode="border", align_corners=False,
                )
                sampled = sampled.reshape(self.hidden_dim, len(rows), 4, self.num_points, 3)
                sampled = sampled.permute(1, 2, 0, 3, 4)
                strips = strips.index_copy(0, rows, sampled)
            if self.feature_mode == "constant":
                # Retain the same sampling/masks but erase all sampled content.
                strips = strips * 0.0
            strips = strips * point_valid[:, :, None].to(strips.dtype)
            return strips, point_valid, valid_gt

    def forward(
        self, feature_map, queries, pred_boxes, gt_boxes, batch_indices,
        image_size, image_ids=None,
    ):
        """Return (refined boxes [N, 4], eligible queries [N])."""
        n = len(gt_boxes)
        if queries.shape != (n, self.query_dim) or pred_boxes.shape != (n, 4):
            raise ValueError("queries and pred_boxes must describe the same N matched queries.")
        if queries.device != feature_map.device or pred_boxes.device != feature_map.device:
            raise ValueError("All inputs must share a device.")
        with torch.autocast(device_type=feature_map.device.type, enabled=False):
            strips, point_valid, valid_gt = self.sample_strips(
                feature_map, gt_boxes, batch_indices, image_size, image_ids
            )
            original = pred_boxes.float()
            valid = (
                valid_gt
                & torch.isfinite(original).all(-1)
                & (original[:, 2:] > 0).all(-1)
                & torch.isfinite(queries).all(-1)
            )
            if not n:
                return original + self.zero_loss(), valid

            encoded = self.strip_encoder(strips.reshape(n * 4, self.hidden_dim, self.num_points, 3))
            mask = point_valid.reshape(n * 4, 1, self.num_points, 3).to(encoded.dtype)
            pooled = (encoded * mask).sum(2) / mask.sum(2).clamp_min(1)
            # Flatten columns first, retaining the order inside / edge / outside.
            edge_features = self.edge_fusion(pooled.transpose(1, 2).reshape(n * 4, -1))
            safe_boxes = torch.where(valid[:, None], original, torch.zeros_like(original))
            safe_queries = torch.where(valid[:, None], queries.float(), torch.zeros_like(queries, dtype=torch.float32))
            inputs = torch.cat((safe_queries, self.box_embedding(safe_boxes), edge_features.reshape(n, -1)), -1)
            deltas = self.refiner(inputs)
            centers = safe_boxes[:, :2] + safe_boxes[:, 2:] * deltas[:, :2]
            sizes = safe_boxes[:, 2:] * deltas[:, 2:].clamp(-self.max_log_scale, self.max_log_scale).exp()
            refined = torch.cat((centers, sizes), -1)
            # Baseline clipping/filtering is deliberately left to its postprocessor.
            return torch.where(valid[:, None], refined, original) + self.zero_loss(), valid
