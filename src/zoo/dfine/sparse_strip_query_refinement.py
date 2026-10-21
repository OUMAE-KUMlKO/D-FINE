"""Joint decoder refinement with sparse center strips and native detection heads."""
from pathlib import Path

import torch

from ...core import register
from .dfine import DFINE
from .sparse_strip_query import SparseStripQuery


@register()
class DFINESparseStripQuery(DFINE):
    __inject__ = ['backbone', 'encoder', 'decoder']
    requires_gt = False
    oracle_evaluation = False

    def __init__(self, backbone, encoder, decoder, enabled=True, freeze_visual=True,
                 feature_level=0, hidden_dim=32, max_strips=9,
                 min_separation_px=2.0, num_points=5, feature_stride=8,
                 selection_mode='sparse', prior_only=False, compile_eval=False):
        super().__init__(backbone, encoder, decoder)
        if feature_level < 0:
            raise ValueError('feature_level must be nonnegative')
        self.enabled = enabled
        self.freeze_visual = freeze_visual
        self.feature_level = feature_level
        self.sparse_refiner = SparseStripQuery(
            in_channels=encoder.hidden_dim, query_dim=decoder.hidden_dim,
            hidden_dim=hidden_dim, num_layers=decoder.num_layers,
            max_strips=max_strips, min_separation_px=min_separation_px,
            num_points=num_points, feature_stride=feature_stride,
            selection_mode=selection_mode, prior_only=prior_only,
            compile_eval=compile_eval,
        ) if enabled else None
        self.last_diagnostics = {}
        if freeze_visual:
            for module in (self.backbone, self.encoder):
                module.requires_grad_(False)
                module.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_visual:
            self.backbone.eval()
            self.encoder.eval()
        return self

    def load_detector_checkpoint(self, path):
        """Load common detector tensors exactly; report discarded old G5 heads."""
        checkpoint = torch.load(Path(path), map_location='cpu')
        if 'ema' in checkpoint and checkpoint['ema'] is not None:
            state = checkpoint['ema']['module']
            source = 'ema'
        else:
            state = checkpoint.get('model', checkpoint)
            source = 'model'
        expected = {k:v for k,v in self.state_dict().items()
                    if not k.startswith('sparse_refiner.')}
        ignored = sorted(k for k in state if k.startswith('refiner.'))
        supplied = {k:v for k,v in state.items() if k not in ignored}
        missing = sorted(expected.keys() - supplied.keys())
        unexpected = sorted(supplied.keys() - expected.keys())
        mismatched = [k for k in expected.keys() & supplied.keys()
                      if supplied[k].shape != expected[k].shape]
        if missing or unexpected or mismatched:
            raise ValueError(f'Checkpoint mismatch: missing={missing}, unexpected={unexpected}, '
                             f'shape_mismatch={mismatched}')
        result = self.load_state_dict(supplied, strict=False)
        assert not result.unexpected_keys
        assert all(k.startswith('sparse_refiner.') for k in result.missing_keys)
        return {'checkpoint':str(Path(path).resolve()), 'source':source,
                'loaded_tensors':len(supplied), 'ignored_keys':ignored,
                'new_keys':result.missing_keys}

    def forward(self, images, targets=None, return_distribution=False, return_query=False):
        if self.freeze_visual:
            with torch.no_grad():
                features = self.encoder(self.backbone(images))
        else:
            features = self.encoder(self.backbone(images))
        self.last_diagnostics = {}
        if not self.enabled:
            return self.decoder(features, targets, return_query=return_query,
                                return_distribution=return_distribution)
        if not 0 <= self.feature_level < len(features):
            raise ValueError('feature_level is outside the encoder output list')
        projected = self.sparse_refiner.project_features(features[self.feature_level])
        layer_stats = []
        geometry_cache = {'_defer_validation': True}

        def refine(layer_index, queries, boxes, distribution):
            dn_count = distribution['num_denoising']
            normal_distribution = {
                **distribution,
                'pred_corners':distribution['pred_corners'][:, dn_count:],
                'ref_points':distribution['ref_points'][:, dn_count:],
                '_strip_geometry_cache':geometry_cache,
            }
            updated, diagnostics = self.sparse_refiner(
                layer_index, projected, queries[:, dn_count:], boxes[:, dn_count:],
                images.shape[-2:], normal_distribution)
            layer_stats.append(diagnostics)
            return torch.cat((queries[:, :dn_count], updated), dim=1)

        output = self.decoder(features, targets, return_query=return_query,
                              return_distribution=return_distribution,
                              query_refiner=refine, query_refiner_trainable=True)
        self.sparse_refiner.validate_geometry_cache(geometry_cache)
        if layer_stats:
            self.last_diagnostics = {
                key:torch.stack([stats[key] for stats in layer_stats]).mean().detach()
                for key in layer_stats[0]
            }
        return output
