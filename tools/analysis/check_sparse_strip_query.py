#!/usr/bin/env python3
"""Real-data single-GPU functional and cost gate; does not train formal runs."""
import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.train_sparse_strip_query import (
    DEFAULT_CHECKPOINT, guard_raw_losses, optimizer_groups, require_finite,
)


def move_batch(batch, device):
    images, targets = batch
    return images.to(device), [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                                for k, v in target.items()} for target in targets]


def same_outputs(left, right, atol=1e-6):
    return {key: bool(torch.allclose(left[key], right[key],
                                    atol=1e-6 if key == 'pred_boxes' else atol, rtol=1e-5))
            for key in ('pred_logits', 'pred_boxes', 'pred_corners', 'pred_queries')}


def output_errors(left, right):
    errors = {}
    for key in ('pred_logits', 'pred_boxes', 'pred_corners', 'pred_queries'):
        difference = (left[key].float() - right[key].float()).abs()
        errors[key] = dict(max_abs=float(difference.max()), mean_abs=float(difference.mean()))
        if key == 'pred_boxes':
            errors[key].update(max_abs_pixels_640=float(difference.max()) * 640,
                               mean_abs_pixels_640=float(difference.mean()) * 640)
    return errors


def main():
    from src.core import YAMLConfig
    from src.misc import dist_utils
    from src.solver.det_engine import evaluate

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT))
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default=str(ROOT / 'configs/dfine/sparse_strip_query/sparse.yml'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--full-eval', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    report = dict(status='running', source_dir=str(ROOT), checkpoint=str(Path(args.checkpoint).resolve()),
                  scope='single-GPU batch4 functional/cost preflight; formal protocol remains 8GPU/globalbatch32',
                  comparisons='stripped common G5 detector, not complete historical G5',
                  thresholds=dict(inference_ratio=1.20, training_ratio=1.30, added_parameters=300000),
                  checks={})
    report['comparison_errors'] = {}
    report['post_training_tolerance'] = dict(
        atol=1e-4, rtol=1e-5, boxes_atol=1e-6, max_box_error_pixels_640=1e-3,
        max_coco_metric_difference_points=0.001,
        interpretation='FP32 compiled equivalence bound, not bitwise identity; '
                       'cross-validated using all 12 COCO metrics on 548 post-update validation images.',
        evidence='Job 17670 after 6 updates: logits 4.25e-5, queries 1.91e-5, corners 1e-5, '
                 'boxes 7.45e-8 normalized (4.77e-5 pixels); eager repeated queries 4.77e-6. '
                 'Original atol=1e-5 outcomes remain reported. Initial zero-writeback and GT atol=1e-6 unchanged.')
    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    try:
        device = torch.device(args.device)
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('GPU preflight requires CUDA')
        device = torch.device('cuda', 0 if device.index is None else device.index)
        torch.cuda.set_device(device)
        report['device'] = torch.cuda.get_device_name(device)
        dist_utils.setup_seed(0)
        cfg = YAMLConfig(args.config)
        cfg.yaml_cfg['DFINESparseStripQuery']['enabled'] = True
        for name in ('train_dataloader', 'val_dataloader'):
            cfg.yaml_cfg[name]['total_batch_size'] = 4
            cfg.yaml_cfg[name]['num_workers'] = 2
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            model = cfg.model
        report['checkpoint_load'] = model.load_detector_checkpoint(args.checkpoint)
        report['checks']['ema_checkpoint'] = report['checkpoint_load']['source'] == 'ema'
        model.to(device)
        initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        visual_before = {k: v for k, v in initial_state.items()
                         if k.startswith(('backbone.', 'encoder.'))}
        report['added_parameters'] = sum(p.numel() for p in model.sparse_refiner.parameters())
        train_loader, val_loader = cfg.train_dataloader, cfg.val_dataloader
        train_loader.generator = torch.Generator().manual_seed(0)
        if hasattr(train_loader.sampler, 'generator'):
            train_loader.sampler.generator = train_loader.generator
        val_loader.generator = torch.Generator().manual_seed(100000)
        report['checks']['dataset_counts'] = len(train_loader.dataset) == 6471 and len(val_loader.dataset) == 548
        val_images, val_targets = move_batch(next(iter(val_loader)), device)
        train_images, train_targets = move_batch(next(iter(train_loader)), device)
        report['checks']['input_shapes'] = (list(val_images.shape) == [4, 3, 640, 640]
                                           and list(train_images.shape) == [4, 3, 640, 640])
        criterion = cfg.criterion.to(device)
        guard_raw_losses(criterion)
        report['checks']['ddf_disabled'] = criterion.weight_dict['loss_ddf'] == 0
        layer_sums, layer_counts = {}, {}
        collect_diagnostics = False
        def capture(module, inputs, result):
            if not collect_diagnostics:
                return
            layer = str(inputs[0])
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
            sums = layer_sums.setdefault(layer, {})
            for key, value in result[1].items():
                sums[key] = sums.get(key, 0.) + float(value)
        handle = model.sparse_refiner.register_forward_hook(capture)
        model.eval()
        with torch.no_grad():
            images, targets = val_images[:2], val_targets[:2]
            model.enabled = True
            collect_diagnostics = True
            sparse = model(images, return_distribution=True, return_query=True)
            collect_diagnostics = False
            require_finite(sparse)
            with_gt = model(images, targets=targets, return_distribution=True, return_query=True)
            permuted = model(images, targets=targets[::-1], return_distribution=True, return_query=True)
            model.enabled = False
            control = model(images, return_distribution=True, return_query=True)
        report['zero_init_equivalence'] = same_outputs(sparse, control)
        report['gt_independence'] = dict(original=same_outputs(sparse, with_gt),
                                         permuted=same_outputs(sparse, permuted))
        report['comparison_errors'].update(
            zero_init=output_errors(sparse, control),
            gt_original=output_errors(sparse, with_gt),
            gt_permuted=output_errors(sparse, permuted))
        report['checks']['zero_init_equivalence'] = all(report['zero_init_equivalence'].values())
        report['checks']['gt_independence'] = all(all(v.values()) for v in report['gt_independence'].values())
        del sparse, control, with_gt, permuted
        if args.full_eval:
            report['initial_metrics'], _ = evaluate(model, criterion, cfg.postprocessor.to(device),
                val_loader, cfg.evaluator, device, epoch=-1, use_wandb=False)
            report['evaluation'] = dict(images=len(val_loader.dataset), max_dets=cfg.evaluator.coco_eval['bbox'].params.maxDets,
                                        annotation=cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'])
            report['checks']['full_validation'] = report['evaluation']['max_dets'][-1] == 100
        else:
            report['checks']['full_validation'] = False
        save()

        def restore(enabled):
            model.load_state_dict(initial_state, strict=True)
            model.enabled = enabled
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        def inference_cost(enabled):
            restore(enabled)
            model.eval()
            durations = []
            with torch.no_grad():
                for i in range(7):
                    if i == 2:
                        torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    prediction = model(val_images[:1])
                    torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - start
                    if i >= 2:
                        durations.append(elapsed)
                    del prediction
            return dict(median_seconds=statistics.median(durations), seconds=durations,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        batch_size=1, precision='FP32', warmup=2)

        def grad_norms():
            groups = dict(output=[], strip_encoder=[], query_projection=[], key_projection=[])
            for name, parameter in model.sparse_refiner.named_parameters():
                for key in groups:
                    if (key == 'output' and '.output.' in name) or (key != 'output' and key in name):
                        if parameter.grad is not None:
                            groups[key].append(parameter.grad.detach().float().norm())
            return {key: float(torch.stack(values).norm()) if values else 0.
                    for key, values in groups.items()}

        def training_cost(enabled):
            nonlocal collect_diagnostics
            restore(enabled)
            model.train()
            criterion.train()
            optimizer = torch.optim.AdamW(optimizer_groups(model), betas=(.9, .999))
            scaler = torch.cuda.amp.GradScaler()
            durations, skips, successes, gradient_checks = [], 0, 0, {}
            torch.cuda.reset_peak_memory_stats(device)
            for attempt in range(12):
                torch.manual_seed(attempt)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                with torch.cuda.amp.autocast():
                    prediction = model(train_images, targets=train_targets)
                require_finite(prediction)
                with torch.cuda.amp.autocast(enabled=False):
                    losses = criterion(prediction, train_targets, epoch=0, step=attempt, global_step=attempt)
                    require_finite(losses, 'weighted_losses')
                    if any(float(v.detach()) != 0 for k, v in losses.items() if k.startswith('loss_ddf')):
                        raise ValueError('DDF contributes nonzero weighted loss')
                    loss = sum(losses.values())
                    require_finite(loss, 'total_loss')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradients = grad_norms() if enabled else {}
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.yaml_cfg['clip_max_norm'])
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize(device)
                duration = time.perf_counter() - start
                if scaler.get_scale() < scale_before:
                    skips += 1
                else:
                    successes += 1
                    if successes in (1, 3):
                        gradient_checks[str(successes)] = gradients
                    if successes > 1:
                        durations.append(duration)
                del prediction, losses, loss
                if len(durations) >= 5:
                    break
            if len(durations) < 5:
                raise RuntimeError(f'Only {successes} successful AMP steps in12 attempts; skipped={skips}')
            frozen_same = all(torch.equal(value, model.state_dict()[key].detach().cpu())
                              for key, value in visual_before.items())
            if enabled:
                report['gradient_norms'] = gradient_checks
                report['checks']['first_output_gradient'] = gradient_checks['1']['output'] > 0
                report['checks']['third_step_encoder_query_key_gradients'] = all(
                    gradient_checks['3'][key] > 0 for key in ('strip_encoder', 'query_projection', 'key_projection'))
                report['checks']['frozen_visual_unchanged'] = frozen_same
                collect_diagnostics = True
                with torch.no_grad(), torch.cuda.amp.autocast():
                    model(train_images, targets=train_targets)
                collect_diagnostics = False
                model.eval()
                with torch.no_grad():
                    compiled = model(val_images[:1], return_distribution=True, return_query=True)
                    model.sparse_refiner.compile_eval = False
                    eager = model(val_images[:1], return_distribution=True, return_query=True)
                    model.sparse_refiner.compile_eval = cfg.yaml_cfg['DFINESparseStripQuery'].get('compile_eval', False)
                report['compiled_eager_after_updates_strict_atol_1e_minus5'] = same_outputs(compiled, eager, atol=1e-5)
                report['compiled_eager_after_updates'] = same_outputs(compiled, eager, atol=1e-4)
                report['comparison_errors']['compiled_eager_after_updates'] = output_errors(compiled, eager)
                report['checks']['compiled_eager_after_updates'] = all(report['compiled_eager_after_updates'].values())
                report['checks']['post_update_box_pixel_bound'] = (
                    report['comparison_errors']['compiled_eager_after_updates']['pred_boxes']['max_abs_pixels_640'] <= 1e-3)
            else:
                report['checks']['control_frozen_visual_unchanged'] = frozen_same
            return dict(median_seconds=statistics.median(durations), seconds=durations,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        batch_size=4, precision='AMP', actual_updates=successes, skipped_updates=skips,
                        warmup_successful_steps=1)

        report['inference_cost'] = dict(control=inference_cost(False), sparse=inference_cost(True))
        report['training_cost'] = dict(control=training_cost(False), sparse=training_cost(True))
        handle.remove()
        report['per_layer_diagnostics'] = {
            layer: {key: value / layer_counts[layer] for key, value in sums.items()}
            for layer, sums in layer_sums.items()}
        report['cost_ratios'] = {
            label: report[f'{label}_cost']['sparse']['median_seconds'] / report[f'{label}_cost']['control']['median_seconds']
            for label in ('inference', 'training')}
        cost_pass = (report['cost_ratios']['inference'] <= 1.20
                     and report['cost_ratios']['training'] <= 1.30
                     and report['added_parameters'] <= 300000)
        report['cost_pass'] = cost_pass
        # evaluate() starts by calling coco_evaluator.cleanup(), so the cached
        # evaluator can be reused without carrying results across these passes.
        report['checks']['post_update_full_validation_equivalence'] = False
        if args.full_eval:
            model.eval()
            model.sparse_refiner.compile_eval = True
            compiled_metrics, _ = evaluate(
                model, criterion, cfg.postprocessor.to(device), val_loader,
                cfg.evaluator, device, epoch=-1, use_wandb=False)
            model.sparse_refiner.compile_eval = False
            eager_metrics, _ = evaluate(
                model, criterion, cfg.postprocessor.to(device), val_loader,
                cfg.evaluator, device, epoch=-1, use_wandb=False)
            model.sparse_refiner.compile_eval = True
            compiled_coco = compiled_metrics['coco_eval_bbox']
            eager_coco = eager_metrics['coco_eval_bbox']
            differences = [abs(a - b) * 100 for a, b in zip(compiled_coco, eager_coco)]
            report['post_update_full_validation'] = dict(
                compiled=compiled_metrics, eager=eager_metrics, images=len(val_loader.dataset),
                difference_points=differences, max_abs_difference_points=max(differences),
                threshold_points=0.001, evaluation_order=['compiled', 'eager'],
                evaluator_reset='native evaluate calls coco_evaluator.cleanup() before each pass')
            report['checks']['post_update_full_validation_equivalence'] = (
                len(val_loader.dataset) == 548 and len(compiled_coco) == len(eager_coco) == 12
                and all(value <= 0.001 for value in differences))
            save()
        # All timings and peak-memory measurements are finalized before this
        # EMA-like copy is allocated or its compiled graph is exercised.
        model.eval()
        copied_model = copy.deepcopy(model).eval()
        copied_model.sparse_refiner.compile_eval = True
        original_parameters = dict(model.named_parameters())
        original_sparse_state = {key: value.detach().clone()
                                 for key, value in model.sparse_refiner.state_dict().items()}
        with torch.no_grad():
            original_before = model(val_images[:1], return_distribution=True, return_query=True)
            copied_before = copied_model(val_images[:1], return_distribution=True, return_query=True)
            independent_storage = all(
                parameter.data_ptr() != original_parameters[name].data_ptr()
                for name, parameter in copied_model.named_parameters())
            for reader in copied_model.sparse_refiner.readers:
                reader.output.bias.add_(torch.linspace(
                    -.5, .5, reader.output.bias.numel(),
                    device=reader.output.bias.device, dtype=reader.output.bias.dtype))
            copied_compiled = copied_model(val_images[:1], return_distribution=True, return_query=True)
            copied_model.sparse_refiner.compile_eval = False
            copied_eager = copied_model(val_images[:1], return_distribution=True, return_query=True)
            copied_eager_repeat = copied_model(val_images[:1], return_distribution=True, return_query=True)
            original_after = model(val_images[:1], return_distribution=True, return_query=True)
        report['comparison_errors'].update(
            deepcopy_initial=output_errors(original_before, copied_before),
            deepcopy_modified_compiled_eager=output_errors(copied_compiled, copied_eager),
            original_before_after_copy_mutation=output_errors(original_before, original_after),
            deepcopy_queries_mutation=output_errors(copied_before, copied_compiled),
            eager_self_repeat=output_errors(copied_eager, copied_eager_repeat))
        copy_checks = dict(
            independent_parameter_storage=independent_storage,
            initial_equivalence=all(same_outputs(original_before, copied_before, atol=1e-4).values()),
            modified_copy_compiled_eager=all(same_outputs(copied_compiled, copied_eager, atol=1e-4).values()),
            original_unchanged=all(same_outputs(original_before, original_after, atol=1e-4).values()),
            box_pixel_bound=all(
                report['comparison_errors'][key]['pred_boxes']['max_abs_pixels_640'] <= 1e-3
                for key in ('deepcopy_initial', 'deepcopy_modified_compiled_eager',
                            'original_before_after_copy_mutation')),
            original_sparse_state_exactly_unchanged=all(
                torch.equal(value, model.sparse_refiner.state_dict()[key])
                for key, value in original_sparse_state.items()),
            copied_queries_changed=not torch.allclose(
                copied_before['pred_queries'], copied_compiled['pred_queries'], atol=1e-4, rtol=1e-5))
        report['compiled_deepcopy_strict_atol_1e_minus5'] = dict(
            initial_equivalence=same_outputs(original_before, copied_before, atol=1e-5),
            modified_copy_compiled_eager=same_outputs(copied_compiled, copied_eager, atol=1e-5),
            original_unchanged=same_outputs(original_before, original_after, atol=1e-5))
        report['compiled_deepcopy_independence'] = copy_checks
        report['checks']['compiled_deepcopy_independence'] = all(copy_checks.values())
        del copied_model, original_before, copied_before, copied_compiled, copied_eager, original_after
        del copied_eager_repeat, original_sparse_state
        functional_pass = all(report['checks'].values())
        report['status'] = 'passed' if functional_pass and cost_pass else ('cost_failed' if functional_pass else 'failed')
        save()
        print(json.dumps(dict(status=report['status'], output=str(output),
                              cost_ratios=report['cost_ratios']), indent=2))
        return 0 if report['status'] == 'passed' else 1
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
