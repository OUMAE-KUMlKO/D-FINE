import unittest
import torch
from test_all_layer_strip import make_model
from test_gt_refinement import normalized_targets
from src.zoo.dfine.boundary_query_selection import BoundaryQuerySelector, selection_losses
from tools.probe_selection_target import FrozenSelectionProbe, frozen_digest, oracle_selection


class FrozenProbeTests(unittest.TestCase):
    def test_only_selector_learns_and_detector_buffers_stay_fixed(self):
        torch.set_num_threads(1)
        detector = make_model()
        detector.query_selector = BoundaryQuerySelector(8, 2, 20, 3)
        probe = FrozenSelectionProbe(detector)
        before = frozen_digest(detector)
        self.assertFalse(detector.training)
        self.assertTrue(detector.query_selector.training)
        calls=[]
        hook=detector.decoder.decoder.register_forward_hook(lambda *args: calls.append(True))
        optimizer=torch.optim.Adam(probe.parameters(),lr=.001)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            aux=probe(torch.randn(1,3,16,24))
            sum(selection_losses(aux,normalized_targets()).values()).backward()
            optimizer.step()
        hook.remove()
        self.assertEqual(calls,[])
        self.assertEqual(before,frozen_digest(detector))
        for name,p in detector.named_parameters():
            self.assertEqual(p.requires_grad,name.startswith('query_selector.'))
            if not name.startswith('query_selector.'):
                self.assertIsNone(p.grad)
        self.assertGreater(detector.query_selector.projection.weight.grad.abs().sum().item(),0)
        probe.eval(); self.assertFalse(detector.query_selector.training)
        self.assertEqual(probe(torch.randn(1,3,16,24))['pred_boxes'].shape,(1,3,4))

    def test_oracle_prioritizes_gt_representative_without_changing_candidates(self):
        memory=torch.arange(12,dtype=torch.float).reshape(1,3,4)
        logits=torch.tensor([[[3.],[2.],[1.]]])
        boxes=torch.tensor([[[.1,.1,.1,.1],[.2,.2,.1,.1],[.8,.8,.1,.1]]])
        anchors=torch.logit(boxes)
        bbox=torch.nn.Linear(4,4)
        torch.nn.init.zeros_(bbox.weight);torch.nn.init.zeros_(bbox.bias)
        result=oracle_selection((memory,logits,anchors,bbox),{'boxes':boxes},[{'boxes':boxes[0,2:]}],1)
        torch.testing.assert_close(result[0],memory[:,2:])
        torch.testing.assert_close(result[2],anchors[:,2:])

if __name__=='__main__':
    unittest.main()
