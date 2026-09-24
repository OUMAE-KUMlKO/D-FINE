"""Calibrate full D-FINE boundary distributions using per-candidate strips."""

import math

import torch
import torch.nn as nn

from .dfine_decoder import Integral
from .dfine_utils import distance2bbox, weighting_function
from .pred_strip_head import PredictionStripHead, encode_distribution_candidates


class StripProbabilityHead(nn.Module):
    """Predict shared scalar logit residuals without removing original bins."""

    def __init__(
        self, in_channels, query_dim, hidden_dim=64, num_points=5,
        kernel_size=3, offset_pixels=1.0, query_chunk_size=32,
        detach_inputs=True, normal_points=3, normal_kernel_size=1,
    ):
        super().__init__()
        if min(in_channels, query_dim, hidden_dim) < 1:
            raise ValueError("Feature dimensions must be positive.")
        if num_points < 2:
            raise ValueError("num_points must be at least 2.")
        if type(normal_points) is not int or normal_points < 3 or normal_points % 2 != 1:
            raise ValueError("normal_points must be an odd integer of at least 3.")
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("kernel_size must be a positive odd integer.")
        if type(normal_kernel_size) is not int or normal_kernel_size < 1 or normal_kernel_size % 2 != 1:
            raise ValueError("normal_kernel_size must be a positive odd integer.")
        if not math.isfinite(offset_pixels) or offset_pixels <= 0:
            raise ValueError("offset_pixels must be finite and positive.")
        if not isinstance(query_chunk_size, int) or query_chunk_size < 1:
            raise ValueError("query_chunk_size must be a positive integer.")
        if not isinstance(detach_inputs, bool):
            raise ValueError("detach_inputs must be a boolean.")
        self.in_channels = in_channels
        self.query_dim = query_dim
        self.hidden_dim = hidden_dim
        self.num_points = num_points
        self.normal_points = normal_points
        self.normal_kernel_size = normal_kernel_size
        # Half-width of the boundary band, measured in network input pixels.
        self.offset_pixels = float(offset_pixels)
        self.query_chunk_size = query_chunk_size
        self.detach_inputs = detach_inputs
        self.projection = nn.Conv2d(in_channels, hidden_dim, 1)
        self.strip_encoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, (kernel_size, normal_kernel_size),
                      padding=(kernel_size // 2, normal_kernel_size // 2), groups=hidden_dim),
            nn.ReLU(), nn.Conv2d(hidden_dim, hidden_dim, 1), nn.ReLU(),
        )
        self.edge_fusion = nn.Sequential(nn.Linear(normal_points * hidden_dim, hidden_dim), nn.ReLU())
        self.query_projection = nn.Linear(query_dim, hidden_dim)
        self.edge_embedding = nn.Embedding(4, 8)
        self.score_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 8 + 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)

    def zero_loss(self):
        return sum(parameter.reshape(-1)[0] * 0.0 for parameter in self.parameters())

    def forward(self, feature_map, queries, pred_boxes, image_size, distribution):
        if feature_map.ndim != 4 or feature_map.shape[1] != self.in_channels:
            raise ValueError("feature_map must have shape [B, in_channels, H, W].")
        if queries.ndim != 3 or queries.shape[-1] != self.query_dim:
            raise ValueError("queries must have shape [B, Q, query_dim].")
        batch_size, query_count = queries.shape[:2]
        if len(feature_map) != batch_size or pred_boxes.shape != (batch_size, query_count, 4):
            raise ValueError("Features, queries, and boxes must share batch/query dimensions.")
        if any(value.device != feature_map.device for value in (queries, pred_boxes)):
            raise ValueError("Features, queries, and boxes must share a device.")
        required = {"pred_corners", "ref_points", "up", "reg_scale"}
        if distribution is None or required - set(distribution):
            raise ValueError(f"distribution must contain {sorted(required)}")
        corners, refs = distribution["pred_corners"], distribution["ref_points"]
        if corners.ndim != 3 or corners.shape[:2] != (batch_size, query_count):
            raise ValueError("pred_corners must have shape [B, Q, 4 * K].")
        if refs.shape != (batch_size, query_count, 4):
            raise ValueError("ref_points must have shape [B, Q, 4].")
        n = batch_size * query_count
        flat_distribution = {
            "pred_corners": corners.reshape(n, corners.shape[-1]),
            "ref_points": refs.reshape(n, 4),
            "up": distribution["up"], "reg_scale": distribution["reg_scale"],
        }
        with torch.autocast(device_type=feature_map.device.type, enabled=False):
            original = pred_boxes.detach().float().reshape(n, 4)
            query_input = queries.detach() if self.detach_inputs else queries
            source_input = feature_map.detach() if self.detach_inputs else feature_map
            corner_input = corners.detach() if self.detach_inputs else corners
            query = query_input.float().reshape(n, self.query_dim)
            source = source_input.float()
            positions, _, distribution_valid = PredictionStripHead._distribution_geometry(
                flat_distribution, n, feature_map.device
            )
            grid, point_valid, candidate_valid = PredictionStripHead._candidate_grid(
                self, positions, original, distribution_valid, image_size
            )
            k = positions.shape[-1]
            logits = corner_input.to(
                device=feature_map.device, dtype=torch.float32
            ).reshape(n, 4, k)
            logits_finite = torch.isfinite(logits).flatten(1).all(-1)
            prior = torch.where(logits_finite[:, None, None], logits, torch.zeros_like(logits))
            feature_finite = torch.isfinite(source).flatten(1).all(-1)
            input_valid = (
                distribution_valid & candidate_valid.any(-1).all(-1)
                & torch.isfinite(original).all(-1) & (original[:, 2:] > 0).all(-1)
                & torch.isfinite(query).all(-1)
                & feature_finite.repeat_interleave(query_count)
            )
            zero = self.zero_loss()
            if not n:
                return {
                    "pred_boxes": original.reshape(batch_size, query_count, 4) + zero,
                    "refinement_mask": input_valid.reshape(batch_size, query_count),
                    "input_valid": input_valid.reshape(batch_size, query_count),
                    "calibrated_corners": prior.reshape(batch_size, query_count, 4, k) + zero,
                    "candidate_positions": positions.reshape(batch_size, query_count, 4, k),
                    "candidate_valid": candidate_valid.reshape(batch_size, query_count, 4, k),
                }

            # Sanitize frozen inputs before trainable layers, including queries
            # that will ultimately be masked out of the residual and loss.
            source = torch.where(torch.isfinite(source), source, torch.zeros_like(source))
            safe_query = torch.where(input_valid[:, None], query, torch.zeros_like(query))
            safe_boxes = torch.where(
                input_valid[:, None], original, original.new_tensor([0, 0, 1, 1])
            )
            batch_indices = torch.arange(batch_size, device=feature_map.device).repeat_interleave(query_count)
            candidate_features = encode_distribution_candidates(
                self.projection(source), grid, point_valid, batch_indices,
                self.strip_encoder, self.edge_fusion, self.hidden_dim,
                self.num_points, self.query_chunk_size,
            )
            query_features = self.query_projection(safe_query)[:, None, None].expand(n, 4, k, -1)
            edge_features = self.edge_embedding(torch.arange(4, device=feature_map.device))
            edge_features = edge_features[None, :, None].expand(n, 4, k, -1)
            edges = torch.cat((
                safe_boxes[:, :2] - safe_boxes[:, 2:] / 2,
                safe_boxes[:, :2] + safe_boxes[:, 2:] / 2,
            ), -1)
            normal_scale = safe_boxes[:, 2:].repeat(1, 2)
            offsets = ((positions - edges[..., None]) / normal_scale[..., None]).nan_to_num(
                nan=0.0, posinf=0.0, neginf=0.0
            )
            log_prior = prior.log_softmax(-1).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            inputs = torch.cat((
                candidate_features, query_features, edge_features,
                offsets[..., None], log_prior[..., None],
            ), -1)
            delta = self.score_head(inputs).squeeze(-1)
            correction_valid = candidate_valid & input_valid[:, None, None]
            calibrated = prior + torch.where(correction_valid, delta, torch.zeros_like(delta)) + zero

            # Invalid rows use finite decode inputs. Their original boxes are
            # restored below; valid rows always decode, including at zero init.
            safe_refs = refs.detach().to(device=feature_map.device, dtype=torch.float32).reshape(n, 4)
            safe_refs = torch.where(input_valid[:, None], safe_refs, torch.zeros_like(safe_refs))
            if bool(input_valid.any()):
                up = torch.as_tensor(distribution["up"], device=feature_map.device).detach().float().reshape(-1)
                scale = torch.as_tensor(distribution["reg_scale"], device=feature_map.device).detach().float().reshape(-1)
            else:
                up, scale = original.new_tensor([0.5]), original.new_tensor([4.0])
            support = weighting_function(k - 1, up, scale)
            distances = Integral(k - 1)(calibrated.reshape(n, 4 * k), support)
            decoded = distance2bbox(safe_refs, distances, scale)
            valid = input_valid & torch.isfinite(decoded).all(-1) & (decoded[:, 2:] > 0).all(-1)
            refined = torch.where(valid[:, None], decoded, original) + zero
            return {
                "pred_boxes": refined.reshape(batch_size, query_count, 4),
                "refinement_mask": valid.reshape(batch_size, query_count),
                "input_valid": input_valid.reshape(batch_size, query_count),
                "calibrated_corners": calibrated.reshape(batch_size, query_count, 4, k),
                "candidate_positions": positions.reshape(batch_size, query_count, 4, k),
                "candidate_valid": candidate_valid.reshape(batch_size, query_count, 4, k),
            }
