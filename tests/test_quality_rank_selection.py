import unittest
import torch
from src.zoo.dfine.boundary_query_selection import (
    BoundaryQuerySelector, quality_targets, quality_pairs, relative_quality_loss,
    relation_targets, gate_relations, selection_losses)
from tools.probe_selection_target import FrozenSelectionProbe, frozen_digest, oracle_selection
from test_all_layer_strip import make_model
from test_gt_refinement import normalized_targets


class QualityRankTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def test_quality_requires_class_match_and_localization(self):
        boxes=torch.tensor([[.5,.5,.2,.2],[.5,.5,.2,.2],[.5,.5,.15,.15],[.1,.1,.1,.1]])
        logits=torch.tensor([[5.,0.],[0.,5.],[5.,0.],[5.,0.]])
        quality,owner,amb=quality_targets(boxes,logits,{'boxes':boxes[:1],'labels':torch.tensor([0])})
        torch.testing.assert_close(quality,torch.tensor([1.,0.,.5625,0.]),atol=1e-6,rtol=1e-6)
        self.assertEqual(owner.tolist(),[0,0,0,-1])
        self.assertFalse(amb.any())

    def test_ranking_pushes_correct_high_iou_above_wrong_and_poor_candidates(self):
        quality=torch.tensor([.9,0.,.5,.8,.8])
        owner=torch.tensor([0,0,0,1,1])
        scores=torch.tensor([0.,2.,1.,0.,0.],requires_grad=True)
        better,worse=quality_pairs(quality,owner,torch.ones(5,dtype=torch.bool))
        self.assertEqual(better.tolist(),[0,0]);self.assertEqual(worse.tolist(),[1,2])
        relative_quality_loss(scores,quality,owner,torch.ones(5,dtype=torch.bool)).backward()
        self.assertLess(scores.grad[0],0)
        self.assertTrue((scores.grad[1:3]>0).all())
        self.assertTrue((scores.grad[3:]==0).all())

    def test_background_edges_are_negative_ambiguity_ignored(self):
        owner=torch.tensor([-1,-1,0,0,1])
        ambiguous=torch.tensor([False,False,False,False,True])
        neighbors=torch.tensor([[1],[2],[3],[4],[3]])
        positive,valid=relation_targets(owner,ambiguous,neighbors,torch.ones(5,1,dtype=torch.bool))
        self.assertEqual(positive[:3,0].tolist(),[False,False,True])
        self.assertEqual(valid[:,0].tolist(),[True,True,True,False,False])

    def test_predicted_background_gate_suppresses_edges_without_gate_gradients(self):
        probability=torch.ones(1,2,1,requires_grad=True)
        existence=torch.tensor([[10.,-10.]],requires_grad=True)
        gated=gate_relations(probability,existence,torch.tensor([[[1],[0]]]))
        self.assertLess(gated.max().item(),.0001)
        gated.sum().backward()
        self.assertIsNone(existence.grad)
        self.assertIsNotNone(probability.grad)

    def test_empty_gt_and_no_quality_pairs_are_finite(self):
        selector=BoundaryQuerySelector(8,2,12,3,supervision='quality_rank')
        aux=selector(torch.randn(1,20,8),torch.randn(1,20,2),torch.randn(1,20,4),
                     torch.nn.Linear(8,4),torch.randn(1,8,8,8),(64,64),3)[-1]
        losses=selection_losses(aux,[{'boxes':torch.empty(0,4),'labels':torch.empty(0,dtype=torch.long)}])
        self.assertEqual(float(losses['loss_query_rank']),0.)
        sum(losses.values()).backward()
        for name,p in selector.named_parameters():
            self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)

    def test_native_initialization_and_frozen_detector_survive_updates(self):
        detector=make_model().eval();image=torch.randn(1,3,16,24)
        with torch.no_grad():base=detector(image)
        detector.query_selector=BoundaryQuerySelector(8,2,20,3,supervision='quality_rank')
        with torch.no_grad():
            out=detector(image); changed=detector(image,[{'ignored':True}])
        for k in ('pred_boxes','pred_logits'):
            torch.testing.assert_close(base[k],out[k],atol=1e-6,rtol=1e-6)
            torch.testing.assert_close(out[k],changed[k],atol=0,rtol=0)
        probe=FrozenSelectionProbe(detector);before=frozen_digest(detector)
        optimizer=torch.optim.Adam(detector.query_selector.parameters(),lr=.001)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            sum(selection_losses(probe(image),normalized_targets()).values()).backward();optimizer.step()
        self.assertEqual(before,frozen_digest(detector))
        self.assertGreater(detector.query_selector.projection.weight.grad.abs().sum().item(),0.)

    def test_quality_oracle_uses_class_correct_quality_in_native_score_fusion(self):
        memory=torch.arange(12,dtype=torch.float).reshape(1,3,4)
        logits=torch.tensor([[[0.,3.],[2.,0.],[1.,0.]]])
        boxes=torch.tensor([[[.5,.5,.2,.2],[.5,.5,.2,.2],[.1,.1,.1,.1]]])
        bbox=torch.nn.Linear(4,4);torch.nn.init.zeros_(bbox.weight);torch.nn.init.zeros_(bbox.bias)
        aux={'boxes':boxes,'quality_logits':torch.zeros(1,3),'candidate_logits':logits}
        result=oracle_selection((memory,logits,torch.logit(boxes),bbox),aux,
                                [{'boxes':boxes[0,:1],'labels':torch.tensor([0])}],1)
        torch.testing.assert_close(result[0],memory[:,1:2])

if __name__=='__main__':unittest.main()
