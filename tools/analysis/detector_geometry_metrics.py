"""Compare independently matched detectors in network-input pixel coordinates."""

import torch

from src.solver.gt_refinement_metrics import GTRefinementMetrics


class DetectorGeometryComparison:
    """Accumulate own matches and compare both detectors on shared GT identities.

    ``targets`` must be the same canonical target list for both matchers. Each
    index tuple is (query_ids, target_ids), with unique target_ids per image.
    Predictions are normalized cxcywh; input_size is (height, width). No score
    threshold or clipping is applied. Invalid geometry uses the existing
    GTRefinementMetrics filter, independently for own and common pairs.
    """

    GROUPS = ("all", "small", "medium", "large")
    GEOMETRY = ("edge_mae_px_ltrb", "mean_edge_mae_px", "mean_iou")

    def __init__(self):
        self._baseline = GTRefinementMetrics()
        self._model = GTRefinementMetrics()
        self._common = GTRefinementMetrics()
        self.counts = dict(total_gt=0, baseline_matched=0, model_matched=0,
                           common_gt=0, baseline_only=0, model_only=0, neither=0)

    @staticmethod
    def _accumulate(accumulator, before, after, target, target_ids, input_size):
        query_ids = torch.arange(len(before), device=before.device)
        accumulator.update({
            "baseline_boxes": before[None], "pred_boxes": after[None],
            "matched_indices": [(query_ids, target_ids)],
            "refinement_targets": [target],
            # This adapter uses only geometry; independent detectors are not refinements.
            "refinement_mask": torch.zeros((1, len(before)), dtype=torch.bool,
                                            device=before.device),
        }, input_size)

    @torch.no_grad()
    def update(self, baseline_outputs, model_outputs, targets,
               baseline_indices, model_indices, input_size):
        baseline = baseline_outputs["pred_boxes"]
        model = model_outputs["pred_boxes"]
        for image, target in enumerate(targets):
            baseline_queries, baseline_targets = baseline_indices[image]
            model_queries, model_targets = model_indices[image]
            baseline_map = dict(zip(baseline_targets.tolist(), baseline_queries.tolist()))
            model_map = dict(zip(model_targets.tolist(), model_queries.tolist()))
            baseline_ids, model_ids = set(baseline_map), set(model_map)
            common_ids = sorted(baseline_ids & model_ids)
            self.counts["total_gt"] += len(target["boxes"])
            self.counts["baseline_matched"] += len(baseline_ids)
            self.counts["model_matched"] += len(model_ids)
            self.counts["common_gt"] += len(common_ids)
            self.counts["baseline_only"] += len(baseline_ids - model_ids)
            self.counts["model_only"] += len(model_ids - baseline_ids)
            self.counts["neither"] += len(target["boxes"]) - len(baseline_ids | model_ids)

            baseline_boxes = baseline[image, baseline_queries]
            model_boxes = model[image, model_queries]
            self._accumulate(self._baseline, baseline_boxes, baseline_boxes,
                             target, baseline_targets, input_size)
            self._accumulate(self._model, model_boxes, model_boxes,
                             target, model_targets, input_size)
            common_targets = torch.tensor(common_ids, dtype=torch.long,
                                          device=target["boxes"].device)
            baseline_common = torch.tensor([baseline_map[i] for i in common_ids],
                                           dtype=torch.long, device=baseline.device)
            model_common = torch.tensor([model_map[i] for i in common_ids],
                                        dtype=torch.long, device=model.device)
            self._accumulate(self._common, baseline[image, baseline_common],
                             model[image, model_common], target, common_targets, input_size)

    def compute(self):
        def geometry(values):
            return {key: values[key] for key in self.GEOMETRY}

        def own(accumulator):
            result = accumulator.compute()
            return {
                "pairing": "own_hungarian_matches",
                "invalid_metric_pairs": result["counts"]["invalid_metric_pairs"],
                "groups": {name: {"pairs": result["groups"][name]["pairs"],
                                  **geometry(result["groups"][name]["baseline"])}
                           for name in self.GROUPS},
            }

        common = self._common.compute()
        return {
            "coordinate_system": common["coordinate_system"],
            "size_grouping": common["size_grouping"],
            "pairing": "independent_hungarian_matches_aligned_by_canonical_gt_id",
            "counts": dict(self.counts),
            "models": {"baseline": own(self._baseline), "model": own(self._model)},
            "common_gt_comparison": {
                "pairing": "intersection_of_matched_gt_ids_per_image",
                "invalid_metric_pairs": common["counts"]["invalid_metric_pairs"],
                "groups": {
                    name: {"pairs": common["groups"][name]["pairs"],
                           "baseline": geometry(common["groups"][name]["baseline"]),
                           "model": geometry(common["groups"][name]["refined"]),
                           "mean_iou_change": common["groups"][name]["mean_iou_change"]}
                    for name in self.GROUPS
                },
            },
        }
