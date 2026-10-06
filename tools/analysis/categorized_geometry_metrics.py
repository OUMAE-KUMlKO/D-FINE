"""Separate detector geometry diagnostics by canonical GT category."""

import torch

from tools.analysis.detector_geometry_metrics import DetectorGeometryComparison


class CategorizedGeometryComparison:
    """Accumulate overall geometry and the same metrics for each GT label.

    Each category retains the independently matched query identities, while GT
    indices are remapped to that category's target list. Predictions from other
    categories therefore cannot inflate its unmatched-GT counts.
    """

    def __init__(self):
        self.overall = DetectorGeometryComparison()
        self.categories = {}

    @staticmethod
    def _category_indices(indices, target_remap):
        queries, targets = indices
        remapped = target_remap.to(targets.device)[targets]
        selected = remapped >= 0
        return queries[selected.to(queries.device)], remapped[selected]

    @torch.no_grad()
    def update(self, baseline_outputs, model_outputs, targets,
               baseline_indices, model_indices, input_size):
        self.overall.update(baseline_outputs, model_outputs, targets,
                            baseline_indices, model_indices, input_size)
        for image, target in enumerate(targets):
            labels = target["labels"]
            for category in labels.unique().tolist():
                selected = labels == category
                target_remap = torch.full_like(labels, -1, dtype=torch.long)
                target_remap[selected] = torch.arange(
                    int(selected.sum()), device=labels.device,
                )
                category_target = {
                    "boxes": target["boxes"][selected],
                    "labels": labels[selected],
                }
                if category not in self.categories:
                    self.categories[category] = DetectorGeometryComparison()
                self.categories[category].update(
                    {"pred_boxes": baseline_outputs["pred_boxes"][image:image + 1]},
                    {"pred_boxes": model_outputs["pred_boxes"][image:image + 1]},
                    [category_target],
                    [self._category_indices(baseline_indices[image], target_remap)],
                    [self._category_indices(model_indices[image], target_remap)],
                    input_size,
                )

    def compute(self):
        return {
            "overall": self.overall.compute(),
            "by_category": {
                str(category): self.categories[category].compute()
                for category in sorted(self.categories)
            },
        }
