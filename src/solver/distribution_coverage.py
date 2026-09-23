"""Read-only coverage diagnostics for fixed baseline matches and decoded candidates."""

import math

import torch


class DistributionCoverageMetrics:
    """Accumulate exact quantiles in input pixels without changing predictions.

    Candidate positions must already be decoded normalized edge coordinates in
    left/top/right/bottom order. Probability inputs are the original softmax,
    before masking invalid candidates. Targets are normalized cxcywh boxes.
    """

    EDGE_NAMES = ("left", "top", "right", "bottom")

    def __init__(self, offset_pixels=1.0):
        self.offset_pixels = float(offset_pixels)
        if not math.isfinite(self.offset_pixels) or self.offset_pixels < 0:
            raise ValueError("offset_pixels must be finite and nonnegative")
        self.matched_pairs = 0
        self.valid_metric_pairs = 0
        self._candidate_distances = [[] for _ in self.EDGE_NAMES]
        self._candidate_relative_distances = [[] for _ in self.EDGE_NAMES]
        self._baseline_distances = [[] for _ in self.EDGE_NAMES]
        self._candidate_inside = [0] * 4
        self._range_contains_gt = [0] * 4
        self._baseline_inside = [0] * 4
        self._candidate_all_inside = 0
        self._all_ranges_contain_gt = 0
        self._baseline_all_inside = 0
        self._tangent_sum = 0.0
        self._no_candidates = [0] * 4
        self._mass_sums = {key: [0.0] * 4 for key in ("original", "normalized", "excluded")}
        self._mass_counts = {key: [0] * 4 for key in self._mass_sums}

    @staticmethod
    def _xyxy(boxes):
        return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                          boxes[:, :2] + boxes[:, 2:] / 2), dim=-1)

    @staticmethod
    def _select(tensor, indices, dtype=torch.float64):
        return tensor.detach()[indices.to(tensor.device)].to(device="cpu", dtype=dtype)

    @staticmethod
    def _mean(total, count):
        value = total / count if count else None
        return value if value is not None and math.isfinite(value) else None

    def _add_mass(self, key, values, valid):
        for edge in range(4):
            selected = values[:, edge][valid[:, edge]]
            self._mass_sums[key][edge] += float(selected.sum())
            self._mass_counts[key][edge] += selected.numel()

    @torch.no_grad()
    def update(self, sampling_diagnostics, baseline_boxes, targets, indices, input_size):
        height, width = input_size
        if not all(math.isfinite(float(size)) and size > 0 for size in (height, width)):
            raise ValueError("input_size must contain positive finite height and width")
        scale = torch.tensor((width, height, width, height), dtype=torch.float64)
        for batch, (source, destination) in enumerate(indices):
            self.matched_pairs += source.numel()
            if source.numel() == 0:
                continue
            original = self._select(baseline_boxes[batch], source)
            target = self._select(targets[batch]["boxes"], destination)
            original_edges, target_edges = self._xyxy(original), self._xyxy(target)
            baseline_distance = (original_edges - target_edges).abs() * scale
            valid_pairs = (
                torch.isfinite(original).all(dim=-1)
                & torch.isfinite(target).all(dim=-1)
                & (original[:, 2:] > 0).all(dim=-1)
                & (target[:, 2:] > 0).all(dim=-1)
                & torch.isfinite(baseline_distance).all(dim=-1)
            )
            self.valid_metric_pairs += int(valid_pairs.sum())
            if not valid_pairs.any():
                continue
            original_edges = original_edges[valid_pairs]
            target_edges = target_edges[valid_pairs]
            target_size = target[valid_pairs, 2:]
            baseline_distance = baseline_distance[valid_pairs]
            positions = self._select(
                sampling_diagnostics["candidate_positions"][batch], source)[valid_pairs]
            candidate_valid = self._select(
                sampling_diagnostics["candidate_valid"][batch], source,
                dtype=torch.bool)[valid_pairs]
            probabilities = self._select(
                sampling_diagnostics["candidate_probabilities"][batch], source)[valid_pairs]
            excluded = self._select(
                sampling_diagnostics["excluded_probability_mass"][batch], source)[valid_pairs]

            signed_distance = (positions - target_edges.unsqueeze(-1)) * scale[None, :, None]
            distance = signed_distance.abs()
            candidate_valid = candidate_valid & torch.isfinite(distance)
            nearest = distance.masked_fill(~candidate_valid, float("inf")).amin(dim=-1)
            has_candidate = candidate_valid.any(dim=-1)
            target_edge_scale = target_size.repeat(1, 2) * scale
            relative_nearest = nearest / target_edge_scale
            minimum = signed_distance.masked_fill(~candidate_valid, float("inf")).amin(dim=-1)
            maximum = signed_distance.masked_fill(~candidate_valid, -float("inf")).amax(dim=-1)
            range_contains_gt = has_candidate & (minimum <= 1e-6) & (maximum >= -1e-6)
            # Match the single-strip diagnostic's tolerance at the band boundary.
            near_gt = candidate_valid & (distance <= self.offset_pixels + 1e-6)
            candidate_inside = near_gt.any(dim=-1)
            baseline_inside = baseline_distance <= self.offset_pixels + 1e-6
            self._candidate_all_inside += int(candidate_inside.all(dim=-1).sum())
            self._all_ranges_contain_gt += int(range_contains_gt.all(dim=-1).sum())
            self._baseline_all_inside += int(baseline_inside.all(dim=-1).sum())
            tangent_overlap = (
                torch.minimum(original_edges[:, 2:], target_edges[:, 2:])
                - torch.maximum(original_edges[:, :2], target_edges[:, :2])
            ).clamp_min(0)
            self._tangent_sum += float((tangent_overlap / target_size).mean(dim=-1).sum())
            for edge in range(4):
                self._candidate_distances[edge].extend(nearest[:, edge][has_candidate[:, edge]].tolist())
                finite_relative = has_candidate[:, edge] & torch.isfinite(relative_nearest[:, edge])
                self._candidate_relative_distances[edge].extend(relative_nearest[:, edge][finite_relative].tolist())
                self._baseline_distances[edge].extend(baseline_distance[:, edge].tolist())
                self._candidate_inside[edge] += int(candidate_inside[:, edge].sum())
                self._range_contains_gt[edge] += int(range_contains_gt[:, edge].sum())
                self._baseline_inside[edge] += int(baseline_inside[:, edge].sum())
                self._no_candidates[edge] += int((~has_candidate[:, edge]).sum())

            valid_probability = (
                torch.isfinite(probabilities).all(dim=-1)
                & (probabilities >= 0).all(dim=-1)
                & (probabilities.sum(dim=-1) > 0)
            )
            nearby_mass = probabilities.masked_fill(~near_gt, 0).sum(dim=-1)
            retained_mass = probabilities.masked_fill(~candidate_valid, 0).sum(dim=-1)
            self._add_mass("original", nearby_mass, valid_probability & torch.isfinite(nearby_mass))
            normalized = nearby_mass / retained_mass.clamp_min(torch.finfo(torch.float64).tiny)
            self._add_mass("normalized", normalized,
                           valid_probability & (retained_mass > 0) & torch.isfinite(normalized))
            self._add_mass("excluded", excluded, torch.isfinite(excluded))

    @staticmethod
    def _distance_summary(values):
        if not values:
            return {"count": 0, "mean": None, "p50": None, "p90": None, "p95": None}
        tensor = torch.tensor(values, dtype=torch.float64)
        quantiles = torch.quantile(tensor, tensor.new_tensor((0.5, 0.9, 0.95)))
        return {
            "count": len(values),
            "mean": DistributionCoverageMetrics._mean(float(tensor.sum()), len(values)),
            "p50": float(quantiles[0]), "p90": float(quantiles[1]), "p95": float(quantiles[2]),
        }

    def compute(self):
        """Return JSON-safe statistics; uncovered edges count as coverage failures.

        Distance quantiles exclude edges with no valid candidates. Coverage uses
        every valid matched pair, including those with no valid candidates.
        Candidate min/max range coverage does not imply a nearby candidate.
        Relative distances divide L/R by GT width and T/B by GT height.
        Normalized near-GT mass is conditional on positive retained probability;
        raw near-GT mass includes zero when all candidates are masked out.
        """
        def edge_values(values):
            return dict(zip(self.EDGE_NAMES, values))

        def fractions(counts):
            return edge_values([self._mean(count, self.valid_metric_pairs) for count in counts])

        return {
            "coordinate_system": "network_input_pixels",
            "pairing": "fixed_baseline_hungarian",
            "coverage_note": "Normal-coordinate coverage only; not feature receptive field or tangent coverage.",
            "offset_pixels": self.offset_pixels,
            "matched_pairs": self.matched_pairs,
            "valid_metric_pairs": self.valid_metric_pairs,
            "invalid_metric_pairs": self.matched_pairs - self.valid_metric_pairs,
            "candidate_coverage": {
                "nearest_distance_px": edge_values([self._distance_summary(x) for x in self._candidate_distances]),
                "nearest_distance_gt_fraction": edge_values([self._distance_summary(x) for x in self._candidate_relative_distances]),
                "inside_band_fraction": fractions(self._candidate_inside),
                "all_edges_inside_band_fraction": self._mean(self._candidate_all_inside, self.valid_metric_pairs),
                "range_contains_gt_fraction": fractions(self._range_contains_gt),
                "all_edges_range_contains_gt_fraction": self._mean(self._all_ranges_contain_gt, self.valid_metric_pairs),
                "no_valid_candidate_count": edge_values(self._no_candidates),
            },
            "baseline_coverage": {
                "edge_distance_px": edge_values([self._distance_summary(x) for x in self._baseline_distances]),
                "inside_band_fraction": fractions(self._baseline_inside),
                "all_edges_inside_band_fraction": self._mean(self._baseline_all_inside, self.valid_metric_pairs),
                "mean_tangent_coverage": self._mean(self._tangent_sum, self.valid_metric_pairs),
            },
            "probability_mass": {
                "near_gt_original_mean": edge_values([
                    self._mean(s, n) for s, n in zip(self._mass_sums["original"], self._mass_counts["original"])]),
                "near_gt_retained_normalized_mean": edge_values([
                    self._mean(s, n) for s, n in zip(self._mass_sums["normalized"], self._mass_counts["normalized"])]),
                "excluded_mean": edge_values([
                    self._mean(s, n) for s, n in zip(self._mass_sums["excluded"], self._mass_counts["excluded"])]),
                "sample_counts": {key: edge_values(values) for key, values in self._mass_counts.items()},
            },
        }
