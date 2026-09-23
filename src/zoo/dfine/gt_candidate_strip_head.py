"""GT selection among fixed prediction-distribution strip candidates."""

import torch

from .pred_strip_head import PredictionStripHead


class GTCandidateStripHead(PredictionStripHead):
    """Use GT only to select the nearest valid candidate for each edge."""

    def __init__(
        self, in_channels, query_dim, hidden_dim=64, num_points=5,
        kernel_size=3, offset_pixels=1.0, max_log_scale=2.0,
        feature_mode="real", random_seed=0,
    ):
        if feature_mode != "real":
            raise ValueError("GTCandidateStripHead supports only feature_mode='real'.")
        super().__init__(
            in_channels=in_channels, query_dim=query_dim, hidden_dim=hidden_dim,
            num_points=num_points, kernel_size=kernel_size,
            offset_pixels=offset_pixels, max_log_scale=max_log_scale,
            sampling_mode="distribution", query_chunk_size=32,
        )
        self.random_seed = int(random_seed)

    @staticmethod
    def _select_candidate_indices(positions, candidate_valid, gt_boxes):
        gt = gt_boxes.detach().to(device=positions.device, dtype=torch.float32)
        gt_valid = torch.isfinite(gt).all(-1) & (gt[:, 2:] > 0).all(-1)
        safe_gt = torch.where(gt_valid[:, None], gt, torch.zeros_like(gt))
        x, y, w, h = safe_gt.unbind(-1)
        gt_edges = torch.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), -1)
        errors = (positions - gt_edges[..., None]).abs().masked_fill(
            ~candidate_valid, float("inf")
        )
        selected = errors.argmin(dim=-1)
        has_candidate = candidate_valid.any(dim=-1) & gt_valid[:, None]
        return torch.where(has_candidate, selected, torch.full_like(selected, -1))

    def forward(
        self, feature_map, queries, pred_boxes, gt_boxes, batch_indices,
        image_size, image_ids=None, distribution=None,
    ):
        n = len(pred_boxes)
        if gt_boxes.shape != (n, 4):
            raise ValueError("gt_boxes must have shape [N, 4].")
        if gt_boxes.device != feature_map.device:
            raise ValueError("GT boxes and features must share a device.")
        positions, _, distribution_valid = self._distribution_geometry(
            distribution, n, feature_map.device
        )
        _, _, candidate_valid = self._candidate_grid(
            positions, pred_boxes, distribution_valid, image_size
        )
        candidate_indices = self._select_candidate_indices(
            positions, candidate_valid, gt_boxes
        )
        return self._distribution_forward(
            feature_map, queries, pred_boxes, batch_indices, image_size,
            distribution, False, candidate_indices=candidate_indices,
        )
