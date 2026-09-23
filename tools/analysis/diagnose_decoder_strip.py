"""Compare best/last decoder-strip checkpoints on a fixed validation subset.

This entry point loads checkpoints and data, obtains GT-free layer traces, and
passes every comparison to the same final-B0 Hungarian pairs and IoU groups.
Layer keys are zero-based decoder indices; geometry uses network-input pixels.
The output contains subset geometry statistics, not COCO detection metrics.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.analysis.decoder_strip_metrics import FixedPairComparison
from tools.analysis.decoder_strip_trace import trace_decoder_feedback


def final_boxes(trace, baseline):
    """Apply the production wrapper's final validity and fallback rules."""
    boxes, original = trace['pred_boxes'].float(), baseline['pred_boxes'].float()
    valid = (
        trace['refinement_mask'] & torch.isfinite(boxes).all(-1) & (boxes[..., 2:] > 0).all(-1)
        & torch.isfinite(original).all(-1) & (original[..., 2:] > 0).all(-1)
        & torch.isfinite(baseline['pred_logits']).all(-1)
    )
    return torch.where(valid[..., None], boxes, original), valid


def matched_values(tensor, indices):
    return torch.cat([tensor[b, src] for b, (src, _) in enumerate(indices)])


def record_signals(storage, layer, indices):
    """Summarize query and boundary-logit changes on the fixed B0 pairs."""
    query = matched_values(layer['query_before'].float(), indices)
    delta = matched_values(layer['query_delta'].float(), indices)
    if not len(query):
        return
    before = matched_values(layer['logits_before'].float(), indices).reshape(len(query), 4, -1)
    after = matched_values(layer['logits_after'].float(), indices).reshape_as(before)
    values = {
        'query_delta_l2': delta.norm(dim=-1),
        'query_relative_delta_l2': delta.norm(dim=-1) / query.norm(dim=-1).clamp_min(1e-12),
        'mean_edge_probability_total_variation': (
            (after.softmax(-1) - before.softmax(-1)).abs().sum(-1).mean(-1) / 2
        ),
        'mean_abs_logit_change': (after - before).abs().mean((1, 2)),
        'valid': matched_values(layer['valid'], indices).float(),
    }
    for name, value in values.items():
        storage.setdefault(name, []).append(value.detach().cpu())


def summarize_signals(storage):
    summary = {}
    for name, chunks in storage.items():
        values = torch.cat(chunks).double()
        summary[name] = {
            'mean': values.mean().item(), 'p90': values.quantile(0.9).item(),
            'max': values.max().item(), 'pairs': len(values),
        }
    return summary


def matched_iou(boxes, targets, indices):
    from src.solver.gt_refinement_metrics import GTRefinementMetrics

    prediction = matched_values(boxes, indices).double()
    truth = torch.cat([target['boxes'][dst] for target, (_, dst) in zip(targets, indices)]).double()
    return GTRefinementMetrics._paired_iou(
        GTRefinementMetrics._xyxy(prediction), GTRefinementMetrics._xyxy(truth),
    )


@torch.inference_mode()
def run(args):
    from src.core import YAMLConfig
    from src.zoo.dfine.gt_refinement import DFINEGTRefiner

    output = Path(args.output_file)
    if output.exists():
        raise FileExistsError(f'Diagnostic already exists: {output}')
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    device = torch.device(args.device)
    config = YAMLConfig(
        args.config,
        DFINEDecoderStripRefiner={'baseline_checkpoint': None},
        val_dataloader={'total_batch_size': args.batch_size, 'num_workers': args.workers},
    )
    if config.yaml_cfg['model'] != 'DFINEDecoderStripRefiner':
        raise ValueError('Expected the decoder strip model')
    best = torch.load(args.best_checkpoint, map_location='cpu')
    last = torch.load(args.last_checkpoint, map_location='cpu')
    detector_keys = {k for k in best['model'] if not k.startswith('refiner.')}
    if detector_keys != {k for k in last['model'] if not k.startswith('refiner.')}:
        raise ValueError('Checkpoint detector keys differ')
    if any(not torch.equal(best['model'][k], last['model'][k]) for k in detector_keys):
        raise ValueError('Best and last checkpoints do not share the same frozen detector')
    model = config.model
    model.load_state_dict(best['model'])
    adapter_states = {
        name: {k.removeprefix('refiner.'): value.to(device) for k, value in ckpt['model'].items()
               if k.startswith('refiner.')}
        for name, ckpt in [('best', best), ('last', last)]
    }
    checkpoint_info = {
        'best': {'path': str(Path(args.best_checkpoint).resolve()), 'epoch': best['last_epoch']},
        'last': {'path': str(Path(args.last_checkpoint).resolve()), 'epoch': last['last_epoch']},
    }
    del best, last
    model.to(device).eval()
    criterion = config.criterion.to(device).eval()
    layers = model.refine_layers
    if len(layers) != 2:
        raise ValueError('This bounded diagnostic compares exactly two configured refinement layers')
    first, second = layers
    modes = {'both': layers, 'early_only': (first,), 'late_only': (second,)}
    base_loader = config.val_dataloader
    count = min(args.samples, len(base_loader.dataset))
    sample_indices = torch.linspace(0, len(base_loader.dataset) - 1, count).round().long().tolist()
    loader = DataLoader(
        Subset(base_loader.dataset, sample_indices), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, collate_fn=base_loader.collate_fn,
    )
    categories = {
        'final_vs_baseline': tuple(modes),
        'local_injection': tuple(str(i) for i in layers),
        'same_layer_vs_baseline': tuple(str(i) for i in layers),
        'toggle_effects': ('add_last_given_first', 'add_first_given_last'),
    }
    metrics = {
        checkpoint: {category: {name: FixedPairComparison() for name in names}
                     for category, names in categories.items()}
        for checkpoint in adapter_states
    }
    signals = {checkpoint: {str(i): {} for i in layers} for checkpoint in adapter_states}
    retention = {checkpoint: {
        'matched_pairs': 0, 'early_stage_improved': 0,
        'early_improved_but_early_only_final_nonpositive': 0,
        'early_improved_but_both_final_nonpositive': 0,
    } for checkpoint in adapter_states}
    image_ids, ignored_gt = [], 0
    equivalence = {checkpoint: None for checkpoint in adapter_states}
    started = time.monotonic()
    for batch_index, (images, targets) in enumerate(loader):
        images = images.to(device)
        image_size = images.shape[-2:]
        features = model.encoder(model.backbone(images))
        baseline = trace_decoder_feedback(model, features, image_size, active_layers=())
        reference = baseline['pred_boxes'].float()
        canonical, ignored = DFINEGTRefiner._prepare_targets(
            targets, image_size, device, 'xyxy_absolute', baseline['pred_logits'].shape[-1],
        )
        # Match once per batch; checkpoint and layer switches reuse these pairs.
        indices = criterion._match_baseline(reference, baseline['pred_logits'], canonical)
        image_ids.extend(int(torch.as_tensor(target['image_id']).item()) for target in targets)
        ignored_gt += sum(ignored)
        reference_iou = matched_iou(reference, canonical, indices)
        for checkpoint, state in adapter_states.items():
            model.refiner.load_state_dict(state)
            traces, final = {}, {}
            for mode, active in modes.items():
                traces[mode] = trace_decoder_feedback(model, features, image_size, active_layers=active)
                final[mode], mask = final_boxes(traces[mode], baseline)
                metrics[checkpoint]['final_vs_baseline'][mode].update(
                    reference, reference, final[mode], canonical, indices, image_size,
                )
                if batch_index == 0 and mode == 'both':
                    production = model(images)
                    torch.testing.assert_close(final[mode], production['pred_boxes'], atol=1e-6, rtol=1e-5)
                    torch.testing.assert_close(mask, production['refinement_mask'], atol=0, rtol=0)
                    torch.testing.assert_close(baseline['pred_logits'], production['pred_logits'], atol=0, rtol=0)
                    equivalence[checkpoint] = (final[mode] - production['pred_boxes']).abs().max().item()
            full = traces['both']
            for i in layers:
                layer = full['layers'][i]
                metrics[checkpoint]['local_injection'][str(i)].update(
                    reference, layer['boxes_before'], layer['boxes_after'], canonical, indices, image_size,
                )
                metrics[checkpoint]['same_layer_vs_baseline'][str(i)].update(
                    reference, baseline['layers'][i]['boxes_after'], layer['boxes_after'],
                    canonical, indices, image_size,
                )
                record_signals(signals[checkpoint][str(i)], layer, indices)
            for name, mode in [('add_last_given_first', 'early_only'), ('add_first_given_last', 'late_only')]:
                metrics[checkpoint]['toggle_effects'][name].update(
                    reference, final[mode], final['both'], canonical, indices, image_size,
                )
            early_gain = (
                matched_iou(full['layers'][first]['boxes_after'], canonical, indices)
                - matched_iou(baseline['layers'][first]['boxes_after'], canonical, indices)
            )
            early_helped = early_gain > 1e-6
            stat = retention[checkpoint]
            stat['matched_pairs'] += len(early_gain)
            stat['early_stage_improved'] += int(early_helped.sum())
            for mode in ('early_only', 'both'):
                nonpositive = matched_iou(final[mode], canonical, indices) - reference_iou <= 1e-6
                stat[f'early_improved_but_{mode}_final_nonpositive'] += int((early_helped & nonpositive).sum())
        print(f'Decoder strip diagnostic: {len(image_ids)}/{count} images', flush=True)
    summary = {
        'config': str(Path(args.config).resolve()), 'checkpoints': checkpoint_info,
        'detector_tensors_identical_between_checkpoints': True,
        'images': len(image_ids), 'dataset_images': len(base_loader.dataset),
        'dataset_indices': sample_indices, 'image_ids': image_ids, 'ignored_gt': ignored_gt,
        'sampling': 'deterministic evenly spaced validation dataset indices',
        'evaluation_dtype': 'float32', 'inference_uses_gt': False,
        'refine_layers_zero_based': list(layers), 'full_validation': False,
        'production_equivalence_max_box_error': equivalence,
        'notes': [
            'All stages, toggles and checkpoints use the same final B0 Hungarian matches and B0 IoU groups.',
            'Local before/after holds the incoming history fixed; same-layer comparisons use the unadapted decoder at that depth.',
            'Layer switches reuse jointly trained adapter weights. They are diagnostic counterfactuals, not independently trained ablations.',
            'This subset geometry analysis does not compute COCO AP or establish whole-validation performance.',
            'GT is used only for fixed matching and metrics, never for sampling, query updates or toggles.',
        ],
        'results': {
            checkpoint: {
                **{category: {name: metric.compute() for name, metric in group.items()}
                   for category, group in metric_categories.items()},
                'injection_signals': {name: summarize_signals(values) for name, values in signals[checkpoint].items()},
                'early_gain_retention': retention[checkpoint],
            }
            for checkpoint, metric_categories in metrics.items()
        },
        'elapsed_seconds': time.monotonic() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(f'Saved diagnostic: {output}', flush=True)


def main():
    root = '/cluster/home/zzy/D-FINE/output/uavswarm_strip_beststg2_20260923/decoder_strip_last2'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/dfine/pred_refinement/dfine_hgnetv2_l_uavswarm_decoder_strip.yml')
    parser.add_argument('--best-checkpoint', default=f'{root}/best_stg1.pth')
    parser.add_argument('--last-checkpoint', default=f'{root}/last.pth')
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--samples', type=int, default=96)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if min(args.samples, args.batch_size, args.threads) < 1 or args.workers < 0:
        parser.error('samples/batch size/threads must be positive and workers nonnegative')
    run(args)


if __name__ == '__main__':
    main()
