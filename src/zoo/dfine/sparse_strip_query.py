"""Detached distribution-guided center strips with bounded query writeback."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .dfine_utils import weighting_function


class _Reader(nn.Module):
    def __init__(self, query_dim, hidden_dim, prior_only):
        super().__init__()
        self.norm = nn.LayerNorm(query_dim)
        self.query_projection = nn.Linear(query_dim, hidden_dim)
        self.edge_embedding = nn.Parameter(torch.zeros(4, hidden_dim))
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.position_projection = nn.Linear(2, hidden_dim)
        self.output = nn.Linear(4 * hidden_dim, query_dim)
        self.gate = nn.Linear(query_dim + 4, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(.05 / .95))
        if prior_only:
            for module in (self.query_projection, self.key_projection, self.position_projection):
                module.requires_grad_(False)
            self.edge_embedding.requires_grad_(False)


class SparseStripQuery(nn.Module):
    def __init__(self, in_channels=256, query_dim=256, hidden_dim=32,
                 num_layers=6, max_strips=9, min_separation_px=2., num_points=5,
                 feature_stride=8, selection_mode='sparse', prior_only=False,
                 compile_eval=False):
        super().__init__()
        if selection_mode not in ('sparse', 'all'):
            raise ValueError('selection_mode must be sparse or all')
        if max_strips < 1 or num_points < 1 or min_separation_px < 0 or feature_stride <= 0:
            raise ValueError('Invalid strip selection settings')
        self.max_strips = max_strips
        self.min_separation_px = min_separation_px
        self.num_points = num_points
        self.feature_stride = feature_stride
        self.selection_mode = selection_mode
        self.prior_only = prior_only
        self.compile_eval = compile_eval
        self.projection = nn.Conv2d(in_channels, hidden_dim, 1)
        self.strip_encoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, 1), nn.GELU())
        self.readers = nn.ModuleList([_Reader(query_dim, hidden_dim, prior_only)
                                      for _ in range(num_layers - 1)])

    def project_features(self, feature):
        return self.projection(feature)

    @staticmethod
    def _finite_flag(tensor):
        # Torch 2.1/Triton can promote masked bool reductions to int32 inside
        # a reduction loop. Explicit integer accumulation keeps its type stable.
        return torch.isfinite(tensor).to(torch.int32).sum(dtype=torch.int32) == tensor.numel()

    @staticmethod
    def _finite(name, tensor):
        if not SparseStripQuery._finite_flag(tensor):
            raise FloatingPointError(f'SparseStripQuery: {name} must be finite')

    def select_candidates(self, positions, probability, valid):
        """Vectorized rows; stable original-bin tie breaks, then spatial ordering."""
        k = positions.shape[-1]
        if self.selection_mode == 'all':
            indices = positions.masked_fill(~valid, float('inf')).argsort(dim=-1, stable=True)
            mask = valid.gather(-1, indices)
            return indices, mask, probability.gather(-1, indices) * mask
        slots = min(self.max_strips, k)
        chosen = torch.zeros_like(valid)
        indices, masks = [], []
        distance = torch.full_like(positions, float('inf'))
        for step in range(slots + 2):
            eligible = valid & ~chosen & (distance >= self.min_separation_px) & (chosen.sum(-1, keepdim=True) < slots)
            if step == 0:
                score = probability.masked_fill(~eligible, -float('inf'))
            elif step == 1:
                # Endpoints are tested as endpoints, not replaced by the next bin.
                endpoint = positions.masked_fill(~valid, float('inf')).argmin(-1)
                eligible = eligible & (torch.arange(k, device=positions.device) == endpoint[..., None])
                score = probability.masked_fill(~eligible, -float('inf'))
            elif step == 2:
                endpoint = positions.masked_fill(~valid, -float('inf')).argmax(-1)
                eligible = eligible & (torch.arange(k, device=positions.device) == endpoint[..., None])
                score = probability.masked_fill(~eligible, -float('inf'))
            else:
                score = (probability * distance).masked_fill(~eligible, -float('inf'))
            index = score.argmax(-1)
            active = eligible.any(-1)
            indices.append(index)
            masks.append(active)
            chosen.scatter_(-1, index[..., None], chosen.gather(-1, index[..., None]) | active[..., None])
            new_distance = (positions - positions.gather(-1,index[...,None])).abs()
            distance = torch.minimum(distance, new_distance.masked_fill(~active[...,None],float('inf')))
        indices, mask = torch.stack(indices,-1), torch.stack(masks,-1)
        compact = (~mask).int().argsort(dim=-1, stable=True)[..., :slots]
        indices, mask = indices.gather(-1, compact), mask.gather(-1, compact)
        selected = positions.gather(-1,indices)
        # Group ties use the smallest original selected bin, independently of spatial order.
        distances = (positions[..., :, None] - selected[...,None,:]).abs().masked_fill(~mask[...,None,:],float('inf'))
        nearest = distances == distances.min(-1,keepdim=True).values
        tie_ids = indices[...,None,:].expand_as(distances).masked_fill(~nearest, k)
        assignment = tie_ids.argmin(-1)
        mass = torch.zeros_like(selected).scatter_add(-1,assignment,probability*valid)
        order = selected.masked_fill(~mask,float('inf')).argsort(dim=-1,stable=True)
        return indices.gather(-1,order), mask.gather(-1,order), mass.gather(-1,order)*mask.gather(-1,order)

    @staticmethod
    def _validate_checks(checks, cache):
        # One host synchronization on the normal path; identify failures lazily.
        if checks and not (torch.stack([check for _, check in checks]).to(torch.int32).sum(dtype=torch.int32) == len(checks)):
            for name, check in checks:
                if not check:
                    cache.clear()
                    raise FloatingPointError(f'SparseStripQuery: {name} must be finite and valid')

    @staticmethod
    def validate_geometry_cache(cache):
        """Finish a wrapper-owned forward before exposing any model outputs."""
        SparseStripQuery._validate_checks(cache.pop('_pending_checks', []), cache)

    def _geometry(self, boxes, distribution, image_size, validation_inputs=()):
        boxes = boxes.detach().float()
        logits = distribution['pred_corners'].detach().float()
        cache = distribution.get('_strip_geometry_cache')
        if cache is None:
            cache = {}
        checks = [('boxes', self._finite_flag(boxes)),
                  ('logits', self._finite_flag(logits))]
        for name, tensor in validation_inputs:
            if name == 'projected features':
                if 'projected_finite' not in cache:
                    cache['projected_finite'] = self._finite_flag(tensor)
                checks.append((name, cache['projected_finite']))
            else:
                checks.append((name, self._finite_flag(tensor)))
        if logits.shape[-1] % 4 or logits.shape[-1] < 20 or (logits.shape[-1] // 4) % 2 != 1:
            raise ValueError("pred_corners must contain four odd bin counts of at least five")
        k = logits.shape[-1] // 4
        if 'coordinates' not in cache:
            if len(image_size) != 2 or any(not math.isfinite(float(size)) or size <= 0 for size in image_size):
                raise ValueError("image_size must contain two finite positive dimensions")
            refs = distribution['ref_points'].detach().float()
            up = torch.as_tensor(distribution['up'], device=boxes.device, dtype=torch.float32).detach().reshape(-1)
            scale = torch.as_tensor(distribution['reg_scale'], device=boxes.device, dtype=torch.float32).detach().reshape(-1)
            if up.numel() != 1 or scale.numel() != 1:
                raise FloatingPointError('SparseStripQuery requires finite scalar up and nonzero reg_scale')
            support = (distribution['project'].detach().float() if 'project' in distribution
                       else weighting_function(k - 1, up, scale).float())
            axes = torch.tensor([0, 1, 0, 1], device=boxes.device)
            signs = boxes.new_tensor([-1, -1, 1, 1])
            coordinates = refs[..., axes, None] + signs[:, None] * refs[..., axes + 2, None] * (.5 + support / scale.abs())
            for name, tensor in [('reference points', refs), ('up', up), ('reg_scale', scale),
                                 ('support', support), ('coordinates', coordinates)]:
                checks.append((name, self._finite_flag(tensor)))
            checks.append(('nonzero reg_scale', (scale != 0).all()))
            cache['coordinates'] = coordinates
            cache['axes'] = axes
            cache['signs'] = signs
            cache['t'] = torch.linspace(-.5, .5, self.num_points, device=boxes.device)
            cache['pixels'] = boxes.new_tensor([image_size[1], image_size[0], image_size[1], image_size[0]])
        if cache.get('_defer_validation', False):
            cache['_pending_checks'] = cache.get('_pending_checks', []) + checks
        else:
            self._validate_checks(checks, cache)
        coordinates, axes, signs = cache['coordinates'], cache['axes'], cache['signs']
        previous=boxes[...,axes]+signs*boxes[...,axes+2]/2
        along_axes=1-axes
        t=cache['t']
        along=boxes[...,along_axes,None]+boxes[...,along_axes+2,None]*t
        point_valid=(along>=0)&(along<=1)
        valid=(coordinates>=0)&(coordinates<=1)&point_valid.any(-1)[...,None]&(boxes[...,2:]>0).all(-1)[...,None,None]
        logits=logits.reshape(*boxes.shape[:2],4,k)
        probs=logits.softmax(-1)
        excluded=(probs*~valid).sum(-1)
        conditional_logits=logits.masked_fill(~valid,-float("inf"))
        conditional_logits=torch.where(valid.any(-1,keepdim=True),conditional_logits,torch.zeros_like(logits))
        conditional=conditional_logits.softmax(-1)*valid
        count=valid.sum(-1)
        entropy=-(conditional*conditional.clamp_min(1e-30).log()).sum(-1)/count.clamp_min(2).float().log()
        pixels=cache['pixels']
        ids,mask,mass=self.select_candidates(coordinates*pixels[:,None],conditional,valid)
        selected=coordinates.gather(-1,ids)
        normal=selected[...,None].expand(*selected.shape,self.num_points)
        along=along[...,None,:].expand_as(normal)
        vertical=(axes==0)[...,None,None]
        grid=torch.stack((torch.where(vertical,normal,along),torch.where(vertical,along,normal)),-1)
        point_mask=point_valid[...,None,:]&mask[...,None]
        displacement=(selected-previous[...,None])*pixels[:,None]
        position=torch.stack((displacement/(boxes[...,axes+2]*pixels).clamp_min(1)[...,None],displacement/self.feature_stride),-1)
        return grid, point_mask, mass, mask, entropy, position, excluded, selected*pixels[:,None]

    def _sample(self, projected, grid, point_mask):
        b,n,e,s,p,_=grid.shape
        sampled=F.grid_sample(projected.float(),(2*grid-1).reshape(b,n*e*s,p,2),mode='bilinear',padding_mode='border',align_corners=False)
        sampled=sampled.permute(0,2,1,3).reshape(b*n*e*s,projected.shape[1],p)
        mask=point_mask.reshape(-1,1,p)
        encoded = self.strip_encoder[0](sampled * mask)
        pointwise = self.strip_encoder[1]
        encoded = F.linear(encoded.transpose(1, 2), pointwise.weight[:, :, 0],
                           pointwise.bias).transpose(1, 2)
        encoded = self.strip_encoder[2](encoded) * mask
        pooled=encoded.sum(-1)/mask.sum(-1).clamp_min(1)
        return pooled.reshape(b,n,e,s,-1)

    def forward(self, layer_index, projected, queries, boxes, image_size, distribution):
        if self.compile_eval and queries.is_cuda and not self.training and not torch.is_grad_enabled():
            return _compiled_eval(self, layer_index, projected, queries, boxes, image_size, distribution)
        return self._forward_impl(layer_index, projected, queries, boxes, image_size, distribution)

    def _forward_impl(self, layer_index, projected, queries, boxes, image_size, distribution):
        if not 1 <= layer_index <= len(self.readers):
            raise ValueError('SparseStripQuery layer_index must identify a later decoder layer')
        with torch.autocast(device_type=queries.device.type,enabled=False):
            grid,point_mask,prior,mask,entropy,position,excluded,coordinates=self._geometry(
                boxes,distribution,image_size,validation_inputs=(('queries', queries), ('projected features', projected)))
            visual=self._sample(projected,grid,point_mask)
            reader=self.readers[layer_index-1]
            normalized=reader.norm(queries.float())
            if self.prior_only:
                residual=torch.zeros_like(prior)
            else:
                q=reader.query_projection(normalized)[...,None,:]+reader.edge_embedding
                keys=reader.key_projection(visual)+reader.position_projection(position)
                score=(q[...,None,:]*keys).sum(-1)/math.sqrt(keys.shape[-1])
                residual=2*torch.tanh(score/2)
            scores=(prior.clamp_min(1e-8).log()+residual).masked_fill(~mask,-1e30)
            attention=scores.softmax(-1)*mask
            attention=attention/attention.sum(-1,keepdim=True).clamp_min(1e-30)
            if self.prior_only:
                attention=prior
            pooled=(visual*attention[...,None]).sum(-2).flatten(-2)
            gate=reader.gate(torch.cat((normalized,entropy),-1)).sigmoid()
            delta=gate*reader.output(pooled)
            delta=delta*mask.any(-1).all(-1)[...,None]
            rms=(torch.linalg.vector_norm(queries.float(),dim=-1,keepdim=True)/math.sqrt(queries.shape[-1])).clamp_min(1e-6)
            delta_rms=torch.linalg.vector_norm(delta,dim=-1,keepdim=True)/math.sqrt(delta.shape[-1])
            factor=(.1*rms/delta_rms.clamp_min(1e-12)).clamp_max(1)
            delta=delta*factor
            adjacent=mask[...,1:]&mask[...,:-1]
            def masked_mean(value,valid):
                return (value*valid).sum()/valid.sum().clamp_min(1)
            diagnostics={
                'selected_count':mask.sum(-1).float().mean(),
                'selected_spacing_px':masked_mean((coordinates[...,1:]-coordinates[...,:-1]).abs(),adjacent),
                'excluded_mass':excluded.mean(),
                'adjacent_visual_cosine':masked_mean(F.cosine_similarity(visual[...,1:,:],visual[...,:-1,:],dim=-1),adjacent),
                'attention_change':(attention-prior).abs().sum(-1).mean(),
                'score_saturation':masked_mean((residual.abs()>1.9).float(),mask),
                'gate':gate.mean(),
                'relative_rms':(delta.square().mean(-1,keepdim=True).sqrt()/rms).mean(),
                'cap_hit':(factor<1).float().mean(),
            }
            return queries+delta.to(queries.dtype),{k:v.detach() for k,v in diagnostics.items()}


_COMPILED_EVAL = None


def _compiled_eval(module, *args):
    # Compile the unbound method: deepcopy/EMA must use the supplied module's
    # parameters, never a callable bound to the live training model.
    global _COMPILED_EVAL
    if _COMPILED_EVAL is None:
        from torch import _dynamo
        # Five independent readers and validation batches 1/2/4 exceed the
        # PyTorch 2.1 default of eight shape/layer specializations.
        _dynamo.config.cache_size_limit = max(_dynamo.config.cache_size_limit, 64)
        _COMPILED_EVAL = torch.compile(SparseStripQuery._forward_impl,
                                       fullgraph=False, dynamic=False, mode="reduce-overhead")
    cache = args[-1].get('_strip_geometry_cache') if args and isinstance(args[-1], dict) else None
    previous = dict(cache) if cache is not None else {}
    updated, diagnostics = _COMPILED_EVAL(module, *args)
    # CUDA graph outputs may be reused by later invocations. Own every tensor
    # retained by the decoder or its forward-local cache outside the graph.
    updated = updated.clone()
    if diagnostics:
        values = torch.stack(list(diagnostics.values())).clone().unbind()
        diagnostics = dict(zip(diagnostics, values))
    if cache is not None:
        for key, value in list(cache.items()):
            if isinstance(value, torch.Tensor) and value is not previous.get(key):
                cache[key] = value.clone()
        checks = cache.get('_pending_checks', [])
        if checks:
            flags = torch.stack([value for _, value in checks]).clone().unbind()
            cache['_pending_checks'] = [(name, flag) for (name, _), flag in zip(checks, flags)]
    return updated, diagnostics
