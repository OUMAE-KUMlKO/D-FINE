"""Focused sparse selection, sampling, and trainable writeback checks."""
import unittest
import torch
from src.zoo.dfine.sparse_strip_query import SparseStripQuery


class SparseStripQueryTests(unittest.TestCase):
    def inputs(self):
        q = torch.randn(1, 2, 16, requires_grad=True)
        boxes = torch.tensor([[[.5,.5,.4,.4],[.45,.55,.3,.3]]], requires_grad=True)
        logits = torch.randn(1,2,132, requires_grad=True)
        refs = boxes.detach().clone().requires_grad_()
        return q, boxes, dict(pred_corners=logits, ref_points=refs, up=torch.tensor([.5]), reg_scale=torch.tensor([4.]))

    def test_selection_ties_mass_duplicates_and_all_mode(self):
        model = SparseStripQuery(in_channels=8, query_dim=16)
        x = torch.tensor([[4., 0., 2., 2., 8.]])
        p = torch.ones_like(x)/5
        ids, valid, mass = model.select_candidates(x, p, torch.ones_like(x,dtype=torch.bool))
        self.assertEqual(ids[0, valid[0]].tolist(), [1,2,0,4])
        self.assertTrue(torch.allclose(mass.sum(-1), torch.ones(1)))
        self.assertAlmostEqual(mass[0,1].item(), .4)
        model.selection_mode = 'all'
        ids, valid, mass = model.select_candidates(x,p,torch.ones_like(x,dtype=torch.bool))
        self.assertEqual(valid.sum().item(),5)
        self.assertTrue(torch.allclose(mass.sum(-1),torch.ones(1)))

    def test_identity_detachment_gradients_and_cap(self):
        torch.manual_seed(4)
        model = SparseStripQuery(in_channels=8,query_dim=16)
        feature = torch.randn(1,8,16,16,requires_grad=True)
        q, boxes, distribution = self.inputs()
        out, diagnostics = model(1,model.project_features(feature),q,boxes,(128,128),distribution)
        self.assertTrue(torch.equal(out,q))
        self.assertTrue(all(v.ndim == 0 and not v.requires_grad for v in diagnostics.values()))
        optimizer = torch.optim.Adam(model.parameters(),lr=.01)
        target = torch.randn_like(q)
        for _ in range(3):
            optimizer.zero_grad()
            out,_ = model(1,model.project_features(feature),q,boxes,(128,128),distribution)
            (out-target).square().mean().backward()
            optimizer.step()
        for parameter in (model.strip_encoder[0].weight,model.readers[0].query_projection.weight,model.readers[0].key_projection.weight):
            self.assertGreater(parameter.grad.abs().sum().item(),0)
        self.assertIsNone(boxes.grad)
        self.assertIsNone(distribution['pred_corners'].grad)
        self.assertIsNone(distribution['ref_points'].grad)
        self.assertGreater(q.grad.abs().sum().item(),0)
        with torch.no_grad():
            model.readers[0].output.bias.fill_(100)
        out,_ = model(1,model.project_features(feature),q,boxes,(128,128),distribution)
        self.assertTrue(torch.all((out-q).square().mean(-1).sqrt() <= .10001*q.square().mean(-1).sqrt()))

    def test_invalid_edges_and_nonfinite(self):
        model=SparseStripQuery(in_channels=8,query_dim=16)
        q,boxes,d = self.inputs()
        with torch.no_grad():
            model.readers[0].output.bias.fill_(1)
            boxes[...,2]=0
        out,_=model(1,model.project_features(torch.randn(1,8,16,16)),q,boxes,(128,128),d)
        self.assertTrue(torch.equal(out,q))
        d['pred_corners']=torch.full_like(d['pred_corners'],float('nan'))
        with self.assertRaisesRegex(FloatingPointError,'finite'):
            model(1,model.project_features(torch.randn(1,8,16,16)),q,boxes,(128,128),d)

    def test_geometry_reference_support_and_single_entropy(self):
        model=SparseStripQuery(in_channels=8,query_dim=16,selection_mode='all')
        q, boxes, d=self.inputs()
        d['up']=torch.tensor([0.])
        grid,points,prior,mask,entropy,position,excluded,coords=model._geometry(boxes,d,(128,128))
        self.assertEqual(mask.shape[-1],33)
        self.assertTrue(mask.all())
        self.assertTrue(torch.allclose(prior.sum(-1),torch.ones_like(entropy)))
        expected=boxes.detach()[...,0]-boxes.detach()[...,2]/2
        self.assertTrue(torch.allclose(grid[:,:,0,:,0,0],expected[...,None].expand(-1,-1,33)))
        self.assertTrue(torch.allclose(grid[:,:,0,:,0,1],(boxes.detach()[...,1]-boxes.detach()[...,3]/2)[...,None].expand(-1,-1,33)))
        self.assertTrue(torch.equal(position,torch.zeros_like(position)))
        x=torch.tensor([[0.,0.,0.]])
        valid=torch.tensor([[False,True,False]])
        ids,selected,mass=model.select_candidates(x,valid.float(),valid)
        self.assertEqual(selected.sum().item(),1)
        self.assertEqual(mass.sum().item(),1)

    def test_center_sampling_pixel_ramp_and_point_mask(self):
        model=SparseStripQuery(in_channels=1,query_dim=16,hidden_dim=1)
        with torch.no_grad():
            model.strip_encoder[0].weight.zero_()
            model.strip_encoder[0].weight[0,0,1]=1
            model.strip_encoder[0].bias.zero_()
            model.strip_encoder[1].weight.fill_(1)
            model.strip_encoder[1].bias.zero_()
        model.strip_encoder[2]=torch.nn.Identity()
        # Pixel centers of width-eight image are (i+.5)/8.
        feature=torch.arange(8.).reshape(1,1,1,8).expand(1,1,8,8)
        grid=torch.tensor([[[[[[.3125,.1875],[.3125,.3125],[.3125,.4375],[.3125,.5625],[.3125,.6875]]]]]])
        mask=torch.ones(grid.shape[:-1],dtype=torch.bool)
        self.assertTrue(torch.allclose(model._sample(feature,grid,mask),torch.tensor([[[[[2.]]]]])))
        mask[...,0]=False
        self.assertTrue(torch.allclose(model._sample(feature,grid,mask),torch.tensor([[[[[2.]]]]])))

    def test_endpoint_priority_spacing_fill_and_prior_only(self):
        model=SparseStripQuery(in_channels=8,query_dim=16,max_strips=9)
        x=torch.arange(33.).reshape(1,-1)*2
        p=torch.ones_like(x)/33
        ids,valid,mass=model.select_candidates(x,p,torch.ones_like(x,dtype=torch.bool))
        self.assertEqual(valid.sum().item(),9)
        self.assertIn(32,ids[0].tolist())
        self.assertTrue(((x.gather(-1,ids)[...,1:]-x.gather(-1,ids)[...,:-1])>=2).all())
        model=SparseStripQuery(in_channels=8,query_dim=16,prior_only=True)
        self.assertFalse(model.readers[0].query_projection.weight.requires_grad)
        q,boxes,d=self.inputs()
        out,diag=model(1,model.project_features(torch.randn(1,8,16,16)),q,boxes,(128,128),d)
        self.assertEqual(diag['attention_change'].item(),0)

    def test_zero_query_gradients_and_half_projected_features(self):
        model=SparseStripQuery(in_channels=8,query_dim=16)
        q,boxes,d=self.inputs()
        q=torch.zeros_like(q,requires_grad=True)
        projected=model.project_features(torch.randn(1,8,16,16)).half()
        out,_=model(1,projected,q,boxes,(128,128),d)
        out.sum().backward()
        self.assertTrue(torch.isfinite(q.grad).all())

    def test_valid_prior_normalizes_despite_excluded_logit_dominating(self):
        model=SparseStripQuery(in_channels=8,query_dim=16)
        q,boxes,d=self.inputs()
        logits=torch.full((1,2,4,33),-1000.)
        logits[...,32]=0.  # Outward support endpoints fall outside the image.
        d['pred_corners']=logits.flatten(-2)
        d['up']=torch.tensor([4.])
        _,_,prior,mask,_,_,excluded,_=model._geometry(boxes,d,(128,128))
        self.assertTrue(mask.any(-1).all())
        self.assertTrue(torch.allclose(prior.sum(-1),torch.ones_like(excluded)))

    def test_per_forward_cache_preserves_dynamic_geometry_and_gradients(self):
        model=SparseStripQuery(in_channels=8,query_dim=16)
        with torch.no_grad():
            model.readers[0].output.weight.normal_(std=.1)
        q,boxes,d=self.inputs()
        feature=torch.randn(1,8,16,16)
        for reference_shift in (0., .1):
            current=dict(d,ref_points=d['ref_points'].detach()+reference_shift)
            cache={}
            cached=dict(current,_strip_geometry_cache=cache)
            for shift in (0.,.05):
                dynamic_boxes=boxes.detach()+shift
                dynamic_logits=d['pred_corners'].detach()+torch.randn_like(d['pred_corners'])*shift*10
                current['pred_corners']=dynamic_logits
                cached['pred_corners']=dynamic_logits
                first=model(1,model.project_features(feature),q,dynamic_boxes,(128,128),cached)[0]
                self.assertIn('coordinates',cache)
                second=model(1,model.project_features(feature),q,dynamic_boxes,(128,128),current)[0]
                self.assertTrue(torch.equal(first,second))
                first_grad=torch.autograd.grad(first.sum(),model.projection.weight)[0]
                second_grad=torch.autograd.grad(second.sum(),model.projection.weight)[0]
                self.assertTrue(torch.equal(first_grad,second_grad))

    def test_large_finite_reduction_handles_nan_and_both_infinities(self):
        # Batch-two query shape that triggered the compiler's loop-carried bool bug.
        values=torch.ones(2,300,256)
        self.assertTrue(SparseStripQuery._finite_flag(values))
        for invalid in (float('nan'),float('inf'),-float('inf')):
            values[-1,-1,-1]=invalid
            self.assertFalse(SparseStripQuery._finite_flag(values))

    def test_deferred_validation_raises_on_finalization(self):
        model=SparseStripQuery(in_channels=8,query_dim=16)
        q,boxes,d=self.inputs()
        d['pred_corners']=torch.full_like(d['pred_corners'],float('nan'))
        cache={'_defer_validation':True}
        d['_strip_geometry_cache']=cache
        model(1,model.project_features(torch.randn(1,8,16,16)),q,boxes,(128,128),d)
        with self.assertRaisesRegex(FloatingPointError,'logits'):
            model.validate_geometry_cache(cache)

    def test_decoder_support_reused_without_recomputing(self):
        from unittest.mock import patch
        from src.zoo.dfine.dfine_utils import weighting_function
        model=SparseStripQuery(in_channels=8,query_dim=16)
        q,boxes,d=self.inputs()
        expected=model._geometry(boxes,d,(128,128))
        d['project']=weighting_function(32,d['up'],d['reg_scale'])
        with patch('src.zoo.dfine.sparse_strip_query.weighting_function',side_effect=AssertionError('must reuse decoder support')):
            actual=model._geometry(boxes,d,(128,128))
        for a,b in zip(actual,expected):
            self.assertTrue(torch.equal(a,b))

    def test_pointwise_linear_matches_conv_output_and_gradients(self):
        from unittest.mock import patch
        model=SparseStripQuery(in_channels=3,query_dim=16,hidden_dim=3)
        feature=torch.randn(1,3,8,8,requires_grad=True)
        grid=torch.rand(1,2,4,3,5,2)
        mask=torch.rand(1,2,4,3,5)>.2
        sampled=torch.nn.functional.grid_sample(feature,2*grid.reshape(1,24,5,2)-1,padding_mode='border',align_corners=False)
        sampled=sampled.permute(0,2,1,3).reshape(24,3,5)
        flatmask=mask.reshape(24,1,5)
        expected=((model.strip_encoder(sampled*flatmask)*flatmask).sum(-1)/flatmask.sum(-1).clamp_min(1)).reshape(1,2,4,3,3)
        expected_grad=torch.autograd.grad(expected.square().sum(),(feature,model.strip_encoder[1].weight),retain_graph=True)
        with patch.object(model.strip_encoder[1],'forward',side_effect=AssertionError('PW Conv1d must use equivalent linear')):
            actual=model._sample(feature,grid,mask)
        actual_grad=torch.autograd.grad(actual.square().sum(),(feature,model.strip_encoder[1].weight))
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        for a,b in zip(actual_grad,expected_grad):
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)

    def test_compiled_results_own_output_storage(self):
        from unittest.mock import patch
        from src.zoo.dfine.sparse_strip_query import _compiled_eval
        model=SparseStripQuery()
        query=torch.ones(1,2,256)
        value=torch.tensor(1.)
        coordinates=torch.ones(1,2,4,33)
        flag=torch.tensor(True)
        cache={}
        def fake_compiled(*args):
            cache['coordinates']=coordinates
            cache['_pending_checks']=[('queries',flag)]
            return query,{'gate':value}
        with patch('src.zoo.dfine.sparse_strip_query._COMPILED_EVAL',fake_compiled):
            actual,diag=_compiled_eval(model,{'_strip_geometry_cache':cache})
        query.zero_()
        value.zero_()
        coordinates.zero_()
        flag.zero_()
        self.assertTrue((actual==1).all())
        self.assertEqual(diag['gate'].item(),1.)
        self.assertTrue((cache['coordinates']==1).all())
        self.assertTrue(cache['_pending_checks'][0][1])

    def test_parameter_budget(self):
        self.assertLess(sum(p.numel() for p in SparseStripQuery().parameters()),300000)

if __name__ == '__main__':
    unittest.main()
