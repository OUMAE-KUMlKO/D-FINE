#!/usr/bin/env python3
"""G5 全量训练 80 轮：仅加载主干预训练，训练和验证均使用 GT 覆盖筛选。"""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core import YAMLConfig
from src.misc import dist_utils
from src.solver import TASKS
from src.zoo.dfine.boundary_query_selection import gather
from src.zoo.dfine.box_ops import box_xyxy_to_cxcywh
from tools.gt_query_selection import candidate_pool, gt_candidate_selection
from tools.train_boundary_selection import Float32Criterion


class GTCoverageFull(nn.Module):
    requires_gt = True
    oracle_evaluation = True
    gt_selection_only = True

    def __init__(self, detector, candidates=1200, budget=40):
        super().__init__()
        assert detector.query_selector is None
        assert detector.decoder.query_select_method == 'default'
        assert not detector.decoder.learn_query_content
        # 保留原模型参数名、优化器分组和默认可训练范围。
        self.backbone, self.encoder = detector.backbone, detector.encoder
        self.decoder, self.refiner = detector.decoder, detector.refiner
        self.feature_level = detector.feature_level
        self.checkpoint_strips = detector.checkpoint_strips
        self.candidates, self.budget = candidates, budget
        self.last_records = []

    def forward(self, images, targets=None):
        if targets is None:
            raise ValueError('GT query selection requires targets in training and validation')
        normalized = targets
        if not self.training:
            h, w = images.shape[-2:]
            norm = images.new_tensor([w, h, w, h])
            normalized = [{**t, 'boxes': box_xyxy_to_cxcywh(t['boxes'].as_subclass(torch.Tensor)/norm)} for t in targets]
        features = self.encoder(self.backbone(images))

        def select(memory, logits, anchors, bbox_head, count):
            # GT 只决定离散索引；选中张量在下方重新计算，使编码器辅助损失
            # 能训练候选分类头、回归头和特征。全量训练每次重算，不缓存选择。
            # 此处禁用 AMP，避免 no_grad 内缓存的回归头权重被有梯度调用复用。
            with torch.autocast(memory.device.type, enabled=False):
                _, chosen, records = gt_candidate_selection(
                    memory.float(), logits.float(), anchors.float(), bbox_head, count,
                    normalized, self.candidates, self.budget, cached=None)
            with torch.no_grad():
                pool, _ = candidate_pool(logits.float(), count, min(self.candidates, logits.shape[1]))
                indices = pool.gather(1, chosen)
            selected_memory = gather(memory, indices)
            self.last_records = records
            return (selected_memory, gather(logits, indices),
                    bbox_head(selected_memory) + gather(anchors, indices), None)

        def refine(layer_index, queries, boxes, distribution):
            return self.refiner(layer_index, features[self.feature_level], queries, boxes,
                images.shape[-2:], distribution, self.checkpoint_strips)

        outputs = self.decoder(features, normalized, candidate_selector=select, distribution_refiner=refine)
        if not self.training:
            outputs['gt_selection_records'] = self.last_records
        return outputs


def build_config(path):
    cfg = YAMLConfig(path)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.yaml_cfg.get('seed', 0))
        detector = cfg.model
    cfg._model = GTCoverageFull(detector, cfg.yaml_cfg['gt_selection_candidates'], cfg.yaml_cfg['gt_selection_budget'])
    cfg._criterion = Float32Criterion(cfg.criterion)
    return cfg


def preflight(args):
    dist_utils.setup_seed(0)
    cfg = build_config(args.config)
    cfg.yaml_cfg['train_dataloader']['total_batch_size'] = 4
    cfg.yaml_cfg['train_dataloader']['num_workers'] = 0
    cfg.yaml_cfg['val_dataloader']['total_batch_size'] = 4
    cfg.yaml_cfg['val_dataloader']['num_workers'] = 0
    images, targets = next(iter(cfg.train_dataloader))
    images = images.cuda()
    targets = [{k:v.cuda() if isinstance(v, torch.Tensor) else v for k,v in t.items()} for t in targets]
    model = cfg.model.cuda().train()
    criterion = cfg.criterion.cuda()
    optimizer = cfg.optimizer
    scaler = torch.cuda.amp.GradScaler(init_scale=1024)
    prefixes = ('backbone.', 'encoder.', 'decoder.enc_score_head.', 'decoder.enc_bbox_head.', 'decoder.decoder.', 'refiner.')
    records = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            outputs = model(images, targets)
        assert 'query_aux_outputs' not in outputs
        assert outputs['enc_aux_outputs'][0]['pred_logits'].requires_grad
        assert outputs['enc_aux_outputs'][0]['pred_boxes'].requires_grad
        with torch.autocast('cuda', enabled=False):
            losses = criterion(outputs, targets)
            assert all(torch.isfinite(v) for v in losses.values())
            assert all(float(v)==0 for k,v in losses.items() if k.startswith('loss_ddf'))
            loss = sum(losses.values())
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        grads = {prefix:sum(float(p.grad.abs().sum()) for n,p in model.named_parameters()
            if n.startswith(prefix) and p.grad is not None) for prefix in prefixes}
        assert all(v>0 for v in grads.values()), grads
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
        scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
        assert scaler.get_scale() >= scale
        records.append(dict(step=step,loss=float(loss),grad_norm=float(norm),gradient_sums=grads,
                            replacements=[r['replaced'] for r in model.last_records]))
    model.eval()
    val_images, val_targets = next(iter(cfg.val_dataloader))
    val_images = val_images.cuda()
    val_targets = [{k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in t.items()} for t in val_targets]
    with torch.no_grad():
        output = model(val_images, val_targets)
    assert len(output['gt_selection_records']) == len(val_images)
    assert all(torch.isfinite(output[k]).all() for k in ('pred_boxes','pred_logits'))
    from src.solver.det_engine import evaluate
    stats, _ = evaluate(model, criterion, cfg.postprocessor.cuda(), [(val_images,val_targets)],
        cfg.evaluator, torch.device('cuda'), epoch=0, use_wandb=False)
    assert stats['oracle_evaluation'] and stats['oracle_query_coverage']['images'] == len(val_images)
    Path(args.output).write_text(json.dumps(dict(status='passed',updates=records,
        validation_gt_forward=True, validation_gt_evaluator=stats['oracle_query_coverage'],
        peak_memory_mib=torch.cuda.max_memory_allocated()/2**20),indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.preflight:
        preflight(args); return
    dist_utils.setup_distributed(seed=0)
    assert dist_utils.get_world_size() == 6
    cfg = build_config(args.config)
    if args.output:
        cfg.output_dir = args.output
        cfg.yaml_cfg["output_dir"] = args.output
    assert cfg.epochs == 80
    assert not cfg.resume and not cfg.tuning
    if dist_utils.is_main_process():
        import yaml
        out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
        (out/'effective_config.yml').write_text(yaml.safe_dump(cfg.yaml_cfg,sort_keys=False))
    TASKS[cfg.yaml_cfg['task']](cfg).fit()
    if dist_utils.is_main_process():
        (Path(cfg.output_dir)/'status.json').write_text(json.dumps(dict(status='completed',epochs=80,oracle_evaluation=True))+'\n')
    dist_utils.cleanup()


if __name__ == '__main__':
    main()
