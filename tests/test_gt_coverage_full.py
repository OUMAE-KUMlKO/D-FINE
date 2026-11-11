import unittest
import torch
from test_all_layer_strip import make_model, matcher
from test_gt_refinement import normalized_targets
from src.zoo.dfine.dfine_criterion import DFINECriterion
from tools.train_gt_coverage_full import GTCoverageFull


class FullGTSelectionTests(unittest.TestCase):
    def test_backbone_candidate_heads_decoder_and_refiner_receive_gradients(self):
        torch.manual_seed(3); torch.set_num_threads(1)
        model = GTCoverageFull(make_model(), candidates=20, budget=2).train()
        output = model(torch.randn(1,3,16,24),normalized_targets())
        criterion = DFINECriterion(matcher(),dict(loss_vfl=1,loss_bbox=5,loss_giou=2,
            loss_fgl=.15,loss_ddf=0),['vfl','boxes','local'],num_classes=2,reg_max=4)
        self.assertTrue(output['enc_aux_outputs'][0]['pred_logits'].requires_grad)
        self.assertTrue(output['enc_aux_outputs'][0]['pred_boxes'].requires_grad)
        self.assertNotIn('query_aux_outputs',output)
        self.assertIn('dn_outputs',output)
        sum(criterion(output,normalized_targets()).values()).backward()
        for prefix in ('backbone.','encoder.','decoder.enc_score_head.','decoder.enc_bbox_head.','decoder.decoder.','refiner.'):
            grad=sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None)
            self.assertGreater(grad,0,prefix)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA AMP regression check')
    def test_amp_oracle_does_not_detach_regression_head_weights(self):
        model=GTCoverageFull(make_model(),candidates=20,budget=2).cuda().train()
        targets=[{k:v.cuda() for k,v in t.items()} for t in normalized_targets()]
        with torch.autocast('cuda'):
            output=model(torch.randn(1,3,16,24,device='cuda'),targets)
            loss=output['enc_aux_outputs'][0]['pred_boxes'].sum()
        loss.backward()
        grad=model.decoder.enc_bbox_head.layers[-1].weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(grad.abs().sum().item(),0)

    def test_eval_requires_gt_converts_pixel_boxes_and_recomputes_choices(self):
        model=GTCoverageFull(make_model(),candidates=20,budget=2).eval()
        images=torch.randn(1,3,16,24)
        with self.assertRaises(ValueError):model(images)
        targets=[dict(boxes=torch.tensor([[1.,2.,10.,12.]]),labels=torch.tensor([0]),image_id=torch.tensor([7]))]
        with torch.no_grad():
            first=model(images,targets)
            second=model(images,[dict(boxes=torch.empty(0,4),labels=torch.empty(0,dtype=torch.long),image_id=torch.tensor([7]))])
        self.assertEqual(first['gt_selection_records'][0]['gt'],1)
        self.assertEqual(second['gt_selection_records'][0]['gt'],0)
        self.assertEqual(second['gt_selection_records'][0]['replaced'],0)
        self.assertEqual(first['pred_boxes'].shape,(1,3,4))


if __name__=='__main__':unittest.main()
