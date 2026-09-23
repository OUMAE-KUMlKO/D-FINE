"""Frozen D-FINE oracle selecting strip locations from predicted candidates."""

from ...core import register
from .gt_candidate_strip_head import GTCandidateStripHead
from .gt_refinement import DFINEGTRefiner


@register()
class DFINEGTCandidateRefiner(DFINEGTRefiner):
    """Use GT only to match queries and choose among their original candidates."""

    __inject__ = ["backbone", "encoder", "decoder", "matcher"]
    requires_gt = True
    oracle_evaluation = True
    strip_head_cls = GTCandidateStripHead

    def _decode_baseline(self, features):
        return self.decoder(features, return_query=True, return_distribution=True)

    def _refiner_kwargs(self, baseline, batch_ids, query_ids):
        distribution = {
            name: baseline[name][batch_ids, query_ids].detach()
            for name in ("pred_corners", "ref_points")
        }
        distribution.update({
            name: baseline[name].detach() for name in ("up", "reg_scale")
        })
        return {"distribution": distribution}
