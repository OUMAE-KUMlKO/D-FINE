import unittest
import torch
from src.zoo.dfine.boundary_query_selection import (
    BoundaryQuerySelector, association, local_neighbors, selection_losses)
from test_all_layer_strip import make_model, matcher
from test_gt_refinement import normalized_targets
from src.zoo.dfine.dfine_criterion import DFINECriterion


class SelectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        torch.set_num_threads(1)

    def test_zero_init_native_topk_and_unchanged_content_boxes(self):
        head = BoundaryQuerySelector(8, 2, 20, 3)
        mem, logits = torch.randn(2, 40, 8), torch.randn(2, 40, 2)
        anchors = torch.randn(2, 40, 4)
        bbox = torch.nn.Linear(8, 4)
        out = head(mem, logits, anchors, bbox, torch.randn(2, 8, 8, 8), (64,64), 5)
        idx = logits.max(-1).values.topk(5,dim=1).indices
        for i in range(2):
            torch.testing.assert_close(out[0][i], mem[i,idx[i]],atol=0,rtol=0)
            torch.testing.assert_close(out[2][i], (bbox(mem)+anchors)[i,idx[i]],atol=1e-6,rtol=1e-6)
        self.assertFalse(any('content' in n or 'correction' in n for n,_ in head.named_parameters()))

    def test_neighbors_local_exclude_self_and_sparse_validity(self):
        boxes=torch.tensor([[[.1,.1,.1,.1],[.11,.1,.1,.1],[.9,.9,.1,.1]]])
        ids,valid=local_neighbors(boxes,(640,640),2)
        self.assertEqual(ids[0,0][valid[0,0]].tolist(),[1])
        self.assertFalse(valid[0,2].any())

    def test_hybrid_ambiguity_and_hungarian(self):
        gt=torch.tensor([[.5,.5,.2,.2],[.5,.5,.2,.2]])
        boxes=torch.cat((gt,torch.tensor([[.1,.1,.02,.02]])))
        owner,amb,pos,groups=association(boxes,gt)
        self.assertEqual(amb.tolist(),[True,True,False])
        self.assertEqual(pos.sum().item(),2)
        self.assertEqual(sorted(groups[pos].tolist()),[0,1])
        self.assertEqual(owner[-1].item(),-1)

    def test_empty_gt_and_invalid_edges_have_finite_gradients(self):
        head=BoundaryQuerySelector(8,2,12,3)
        out=head(torch.randn(1,20,8),torch.randn(1,20,2),torch.randn(1,20,4),torch.nn.Linear(8,4),torch.randn(1,8,8,8),(64,64),3)
        losses=selection_losses(out[-1],[{'boxes':torch.empty(0,4)}])
        sum(losses.values()).backward()
        for n,p in head.named_parameters():
            self.assertIsNotNone(p.grad,n)
            self.assertTrue(torch.isfinite(p.grad).all(),n)

    def test_full_model_zero_equivalence_and_gt_independence(self):
        model=make_model().eval()
        image=torch.randn(1,3,16,24)
        with torch.no_grad():
            base=model(image)
            model.query_selector=BoundaryQuerySelector(8,2,20,3)
            selected=model(image)
            changed=model(image,[{'bad':'ignored'}])
        for k in ('pred_boxes','pred_logits'):
            torch.testing.assert_close(base[k],selected[k],atol=1e-6,rtol=1e-6)
            torch.testing.assert_close(selected[k],changed[k],atol=0,rtol=0)

    def test_training_aux_indices_and_unselected_gradients(self):
        model=make_model().train()
        model.query_selector=BoundaryQuerySelector(8,2,20,3)
        outputs=model(torch.randn(1,3,16,24),normalized_targets())
        aux=outputs['query_aux_outputs']; aux['existence'].retain_grad()
        self.assertEqual(outputs['enc_aux_outputs'][0]['pred_logits'].shape[1],3)
        criterion=DFINECriterion(matcher(),{'loss_vfl':1,'loss_bbox':5,'loss_giou':2,'loss_fgl':.15,'loss_ddf':0},['vfl','boxes','local'],num_classes=2,reg_max=4)
        sum(criterion(outputs,normalized_targets()).values()).backward()
        mask=torch.ones_like(aux['existence'],dtype=torch.bool).scatter(1,aux['selected'],False)
        self.assertTrue((aux['existence'].grad[mask].abs()>0).all())
        self.assertGreater(model.query_selector.representative[-1].weight.grad.abs().sum().item(),0)

if __name__=='__main__':
    unittest.main()
