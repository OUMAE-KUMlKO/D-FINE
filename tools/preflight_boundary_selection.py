#!/usr/bin/env python3
"""Real-image equivalence and three AMP optimizer updates before the full run."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.train_boundary_selection import build_config
from src.misc import dist_utils


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    dist_utils.setup_seed(0)
    cfg=build_config(args.config)
    cfg.yaml_cfg['train_dataloader']['total_batch_size']=4
    cfg.yaml_cfg['train_dataloader']['num_workers']=0
    loader=cfg.train_dataloader
    assert len(loader.dataset)==6471
    assert len(cfg.val_dataloader.dataset)==548
    images,targets=next(iter(loader))
    images=images.cuda()
    targets=[{k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in t.items()} for t in targets]
    model=cfg.model.cuda().eval()
    selector=model.query_selector
    with torch.no_grad():
        # Eval cached anchors require the configured 640 x 640 resolution.
        probe=torch.nn.functional.interpolate(images[:1],size=(640,640))
        actual=model(probe)
        altered=model(probe,[{'ignored':True}])
        model.query_selector=None
        baseline=model(probe)
        model.query_selector=selector
        differences={k:float((actual[k]-baseline[k]).abs().max()) for k in ('pred_logits','pred_boxes')}
        for k in differences:
            torch.testing.assert_close(actual[k],baseline[k],atol=2e-5,rtol=1e-5)
            torch.testing.assert_close(actual[k],altered[k],atol=0,rtol=0)
    del actual,altered,baseline,probe
    model.train()
    criterion=cfg.criterion.cuda()
    optimizer=cfg.optimizer
    scaler=torch.cuda.amp.GradScaler(init_scale=1024)
    records=[]
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            outputs=model(images,targets)
        aux=outputs['query_aux_outputs']
        aux['existence'].retain_grad()
        with torch.autocast('cuda',enabled=False):
            losses=criterion(outputs,targets)
            assert all(torch.isfinite(v) for v in losses.values())
            loss=sum(losses.values())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients={name:float(p.grad.abs().sum()) for name,p in selector.named_parameters() if p.grad is not None}
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in selector.parameters())
        assert gradients['existence.weight']>0 and gradients['representative.2.weight']>0
        if step>0:
            assert gradients['projection.weight']>0
        mask=torch.ones_like(aux['existence'],dtype=torch.bool).scatter(1,aux['selected'],False)
        assert (aux['existence'].grad[mask].abs()>0).all()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.1)
        before=scaler.get_scale()
        scaler.step(optimizer); scaler.update()
        assert scaler.get_scale()>=before,'AMP optimizer update was skipped'
        record=dict(step=step,loss=float(loss),grad_norm=float(norm),gradients=gradients,
                    unselected_gradients=int(mask.sum()),amp_scale=scaler.get_scale(),
                    query_losses={k:float(v) for k,v in losses.items() if k.startswith('loss_query')})
        records.append(record)
        print(json.dumps(record),flush=True)
        del outputs,losses,loss,aux
    report=dict(status='passed',image_shape=list(images.shape),zero_init_max_abs=differences,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,updates=records)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    main()
