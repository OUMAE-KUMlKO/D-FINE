"""Focused numerical checks for fixed-pair oracle geometry diagnostics."""

import contextlib
import importlib.util
import io
import multiprocessing
from datetime import timedelta
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler


MODULE_PATH = Path(__file__).resolve().parents[1] / "src/solver/gt_refinement_metrics.py"
SPEC = importlib.util.spec_from_file_location("gt_refinement_metrics", MODULE_PATH)
METRICS_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS_MODULE)
GTRefinementMetrics = METRICS_MODULE.GTRefinementMetrics


def make_outputs(before, after, targets, mask=None):
    before = torch.as_tensor(before, dtype=torch.float64).reshape(1, -1, 4)
    after = torch.as_tensor(after, dtype=torch.float64).reshape_as(before)
    targets = torch.as_tensor(targets, dtype=torch.float64).reshape(-1, 4)
    count = len(targets)
    if mask is None:
        mask = torch.ones((1, count), dtype=torch.bool)
    else:
        mask = torch.as_tensor(mask, dtype=torch.bool).reshape(1, -1)
    indices = torch.arange(count)
    return {
        "baseline_boxes": before,
        "pred_boxes": after,
        "refinement_targets": [{"boxes": targets}],
        "matched_indices": [(indices, indices.clone())],
        "refinement_mask": mask,
        "refinement_counts": {
            "matched": torch.tensor(count), "refined": mask.sum(),
            "total_gt": torch.tensor(count),
        },
    }


def make_image_batch(image_ids):
    """One fixed match per image with nonuniform counters and one improvement."""
    count = len(image_ids)
    target = torch.tensor([0.25, 0.25, 0.125, 0.125], dtype=torch.float64)
    before = target.repeat(count, 3, 1)
    after = before.clone()
    mask = torch.zeros(count, 3, dtype=torch.bool)
    targets = []
    for batch_index, image_id in enumerate(image_ids):
        if image_id == 0:
            before[batch_index, 0, :2] = 0.75  # Disjoint baseline, perfect refinement.
        if image_id % 2:
            before[batch_index, 2, 2] = 0.0  # Invalid unmatched query.
            after[batch_index, 2, 2] = 0.0
        mask[batch_index, 0] = image_id % 3 != 2
        targets.append({"boxes": target.repeat(image_id + 1, 1)})
    ids = torch.tensor(image_ids, dtype=torch.long)
    per_image_counts = {
        "matched": torch.ones(count, dtype=torch.long),
        "refined": mask.sum(dim=1),
        "ignored_gt": ids + 1,
        "total_gt": ids + 1,
        "unmatched_gt": ids,
        "unmatched_queries": torch.full((count,), 2, dtype=torch.long),
        "invalid_gt_samples": (~mask[:, 0]).long(),
        "invalid_baseline_boxes": ids % 2,
    }
    return {
        "baseline_boxes": before,
        "pred_boxes": after,
        "refinement_targets": targets,
        "matched_indices": [(torch.tensor([0]), torch.tensor([0])) for _ in image_ids],
        "refinement_mask": mask,
        "refinement_counts_per_image": per_image_counts,
        "refinement_counts": {key: value.sum() for key, value in per_image_counts.items()},
    }


def accumulate_sampler(sampler, batch_size):
    metrics = GTRefinementMetrics(sampler=sampler)
    image_ids = list(sampler)
    for offset in range(0, len(image_ids), batch_size):
        metrics.update(make_image_batch(image_ids[offset:offset + batch_size]), (100, 200))
    return metrics


def gloo_padding_worker(rank, rendezvous, output_dir):
    """Exercise the real collective without a model, optimizer, or training."""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        sampler = DistributedSampler(range(3), num_replicas=2, rank=rank, shuffle=False)
        metrics = accumulate_sampler(sampler, batch_size=2)
        metrics.synchronize_between_processes()
        torch.save(metrics.values, Path(output_dir) / f"rank_{rank}.pt")
    finally:
        dist.destroy_process_group()


class GTRefinementMetricsTests(unittest.TestCase):
    def test_known_pixel_errors_and_iou(self):
        outputs = make_outputs(
            [[0.45, 0.5, 0.2, 0.4]],
            [[0.5, 0.5, 0.2, 0.4]],
            [[0.5, 0.5, 0.2, 0.4]],
        )
        metrics = GTRefinementMetrics()
        metrics.update(outputs, input_size=(100, 200))
        summary = metrics.compute()
        group = summary["groups"]["all"]
        torch.testing.assert_close(
            torch.tensor(group["baseline"]["edge_mae_px_ltrb"]),
            torch.tensor([10.0, 0.0, 10.0, 0.0]),
        )
        self.assertAlmostEqual(group["baseline"]["mean_edge_mae_px"], 5.0)
        self.assertAlmostEqual(group["baseline"]["center_mae_px_xy"][0], 10.0)
        self.assertEqual(group["baseline"]["size_mae_px_wh"], [0.0, 0.0])
        self.assertAlmostEqual(group["baseline"]["mean_iou"], 0.6)
        self.assertAlmostEqual(group["refined"]["mean_iou"], 1.0)
        self.assertAlmostEqual(group["mean_iou_change"], 0.4)
        self.assertEqual(group["refined"]["mean_edge_mae_px"], 0.0)
        self.assertEqual(group["iou_improved"], 1)
        self.assertEqual(summary["groups"]["medium"]["pairs"], 1)
        self.assertEqual(summary["groups"]["baseline_iou_0_50_to_0_75"]["pairs"], 1)
        self.assertEqual(summary["coordinate_system"], "network_input_pixels")

    def test_groups_include_improvement_degradation_and_unchanged_pairs(self):
        target = [[0.5, 0.5, 0.1, 0.1], [0.5, 0.5, 0.25, 0.25], [0.5, 0.5, 0.5, 0.5]]
        before = [[0.7, 0.5, 0.1, 0.1], target[1], target[2]]
        after = [target[0], [0.7, 0.5, 0.25, 0.25], target[2]]
        outputs = make_outputs(before, after, target, mask=[True, True, False])
        metrics = GTRefinementMetrics()
        metrics.update(outputs, input_size=(200, 200))
        summary = metrics.compute()
        self.assertEqual(summary["counts"]["unrefined_matched"], 1)
        group = summary["groups"]["all"]
        self.assertEqual(group["pairs"], 3)
        self.assertEqual(group["refined_pairs"], 2)
        self.assertEqual([group[f"iou_{key}"] for key in ("improved", "degraded", "stable")], [1, 1, 1])
        for name in ("small", "medium", "large"):
            self.assertEqual(summary["groups"][name]["pairs"], 1)
        self.assertEqual(summary["groups"]["baseline_iou_lt_0_50"]["pairs"], 1)
        self.assertEqual(summary["groups"]["baseline_iou_ge_0_75"]["pairs"], 2)

    def test_iou_group_boundaries(self):
        before = [[0.5, 0.5, width, 1.0] for width in (0.5, 0.75, 1.0)]
        target = [[0.5, 0.5, 1.0, 1.0]] * 3
        outputs = make_outputs(before, before, target, mask=[False] * 3)
        metrics = GTRefinementMetrics()
        metrics.update(outputs, input_size=(100, 100))
        groups = metrics.compute()["groups"]
        self.assertEqual(groups["baseline_iou_lt_0_50"]["pairs"], 0)
        self.assertEqual(groups["baseline_iou_0_50_to_0_75"]["pairs"], 1)
        self.assertEqual(groups["baseline_iou_ge_0_75"]["pairs"], 2)

    def test_matching_uses_canonical_gt_indices_without_rematching(self):
        target = [[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]
        before = [target[1], [0.9, 0.9, 0.1, 0.1], target[0]]
        outputs = make_outputs(before, before, target, mask=[True, False, True])
        outputs["matched_indices"] = [(torch.tensor([2, 0]), torch.tensor([0, 1]))]
        metrics = GTRefinementMetrics()
        metrics.update(outputs, input_size=(200, 200))
        group = metrics.compute()["groups"]["all"]
        self.assertEqual(group["pairs"], 2)
        self.assertEqual(group["baseline"]["mean_iou"], 1.0)
        self.assertEqual(group["iou_stable"], 2)

    def test_invalid_pairs_are_counted_and_not_averaged(self):
        valid = [0.5, 0.5, 0.2, 0.2]
        before = [[float("nan"), 0.5, 0.2, 0.2], [0.5, 0.5, 0.0, 0.2], valid]
        after = [valid, valid, valid]
        outputs = make_outputs(before, after, [valid] * 3, mask=[False, False, True])
        outputs["refinement_counts"]["invalid_baseline_boxes"] = torch.tensor(2)
        metrics = GTRefinementMetrics()
        metrics.update(outputs, input_size=(100, 100))
        summary = metrics.compute()
        self.assertEqual(summary["counts"]["invalid_metric_pairs"], 2)
        self.assertEqual(summary["counts"]["invalid_baseline_boxes"], 2)
        self.assertEqual(summary["groups"]["all"]["pairs"], 1)
        self.assertEqual(summary["groups"]["all"]["baseline"]["mean_iou"], 1.0)

    def test_empty_matches_and_accumulation(self):
        metrics = GTRefinementMetrics()
        metrics.update(make_outputs([], [], []), input_size=(100, 200))
        empty = metrics.compute()["groups"]["all"]
        self.assertEqual(empty["pairs"], 0)
        self.assertIsNone(empty["baseline"]["mean_iou"])
        valid = [[0.5, 0.5, 0.2, 0.2]]
        outputs = make_outputs(valid, valid, valid)
        metrics.update(outputs, input_size=(100, 200))
        metrics.update(outputs, input_size=(100, 200))
        self.assertEqual(metrics.compute()["counts"]["matched"], 2)
        self.assertEqual(metrics.compute()["groups"]["all"]["pairs"], 2)

    def test_distributed_reduction_sums_counts_and_preserves_means(self):
        valid = [[0.5, 0.5, 0.2, 0.2]]
        metrics = GTRefinementMetrics()
        metrics.update(make_outputs(valid, valid, valid), input_size=(100, 100))
        with mock.patch.object(METRICS_MODULE.dist, "is_available", return_value=True), \
             mock.patch.object(METRICS_MODULE.dist, "is_initialized", return_value=True), \
             mock.patch.object(METRICS_MODULE.dist, "all_reduce", side_effect=lambda value, op: value.mul_(2)) as reduce:
            metrics.synchronize_between_processes()
        reduce.assert_called_once()
        summary = metrics.compute()
        self.assertEqual(summary["counts"]["matched"], 2)
        self.assertEqual(summary["groups"]["all"]["pairs"], 2)
        self.assertEqual(summary["groups"]["all"]["baseline"]["mean_iou"], 1.0)


class DistributedPaddingMetricsTests(unittest.TestCase):
    @staticmethod
    def reference(image_ids):
        metrics = GTRefinementMetrics()
        metrics.update(make_image_batch(list(image_ids)), (100, 200))
        return metrics

    def test_three_images_two_ranks_match_single_process_for_both_batch_sizes(self):
        expected = self.reference(range(3))
        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size):
                ranks = [
                    accumulate_sampler(
                        DistributedSampler(range(3), num_replicas=2, rank=rank, shuffle=False),
                        batch_size,
                    )
                    for rank in range(2)
                ]
                combined = GTRefinementMetrics()
                combined.values = torch.stack([rank.values for rank in ranks]).sum(dim=0)
                torch.testing.assert_close(combined.values, expected.values, rtol=0, atol=1e-12)
                summary = combined.compute()
                self.assertEqual(summary["counts"]["matched"], 3)
                self.assertEqual(summary["counts"]["ignored_gt"], 6)
                self.assertEqual(summary["counts"]["unmatched_gt"], 3)
                self.assertEqual(summary["counts"]["invalid_gt_samples"], 1)
                self.assertAlmostEqual(summary["groups"]["all"]["mean_iou_change"], 1 / 3)

    def test_more_ranks_than_images_can_contribute_zero_without_skipping_forward(self):
        ranks = [
            accumulate_sampler(
                DistributedSampler(range(1), num_replicas=4, rank=rank, shuffle=False), 1
            )
            for rank in range(4)
        ]
        for rank in ranks[1:]:
            self.assertEqual(torch.count_nonzero(rank.values).item(), 0)
        torch.testing.assert_close(
            torch.stack([rank.values for rank in ranks]).sum(dim=0),
            self.reference(range(1)).values,
        )

    def test_shuffled_and_divisible_datasets_match_single_process(self):
        for image_count, world_size, shuffle in ((5, 3, True), (4, 2, False), (4, 2, True)):
            with self.subTest(image_count=image_count, world_size=world_size, shuffle=shuffle):
                ranks = []
                for rank in range(world_size):
                    sampler = DistributedSampler(
                        range(image_count), num_replicas=world_size, rank=rank, shuffle=shuffle,
                    )
                    sampler.set_epoch(7)
                    ranks.append(accumulate_sampler(sampler, batch_size=2))
                torch.testing.assert_close(
                    torch.stack([rank.values for rank in ranks]).sum(dim=0),
                    self.reference(range(image_count)).values,
                )

    def test_drop_last_sampler_keeps_all_yielded_samples_with_aggregate_counts(self):
        ranks = []
        retained_ids = []
        for rank in range(2):
            sampler = DistributedSampler(
                range(5), num_replicas=2, rank=rank, shuffle=True, drop_last=True,
            )
            ids = list(sampler)
            retained_ids.extend(ids)
            outputs = make_image_batch(ids)
            del outputs["refinement_counts_per_image"]
            metrics = GTRefinementMetrics(sampler=sampler)
            metrics.update(outputs, (100, 200))
            ranks.append(metrics)
        torch.testing.assert_close(
            torch.stack([rank.values for rank in ranks]).sum(dim=0),
            self.reference(retained_ids).values,
        )

    def test_partial_batch_rejects_missing_per_image_counts(self):
        sampler = DistributedSampler(range(3), num_replicas=2, rank=1, shuffle=False)
        outputs = make_image_batch(list(sampler))
        del outputs["refinement_counts_per_image"]
        with self.assertRaises(ValueError):
            GTRefinementMetrics(sampler=sampler).update(outputs, (100, 200))

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "CPU Gloo is unavailable")
    def test_real_two_process_gloo_reduce_matches_single_process(self):
        expected = self.reference(range(3)).values
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory(prefix="gt_metrics_gloo_") as temp_dir:
            rendezvous = str(Path(temp_dir) / "rendezvous")
            processes = [
                context.Process(target=gloo_padding_worker, args=(rank, rendezvous, temp_dir))
                for rank in range(2)
            ]
            try:
                for process in processes:
                    process.start()
                deadline = time.monotonic() + 45
                for process in processes:
                    process.join(timeout=max(0, deadline - time.monotonic()))
                self.assertFalse(any(process.is_alive() for process in processes), "Gloo timed out")
                self.assertEqual([process.exitcode for process in processes], [0, 0])
                for rank in range(2):
                    actual = torch.load(Path(temp_dir) / f"rank_{rank}.pt", map_location="cpu")
                    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    if process.pid is not None:
                        process.join(timeout=5)


class OracleEvaluationIntegrationTests(unittest.TestCase):
    def test_engine_passes_gt_only_for_oracle_and_keeps_dataset_coordinates(self):
        from src.solver import det_engine

        class Model(torch.nn.Module):
            def __init__(self, oracle):
                super().__init__()
                self.requires_gt = oracle
                self.received_targets = None

            def forward(self, samples, targets=None):
                self.received_targets = targets
                boxes = [[0.25, 0.6, 0.2, 0.4]]
                return make_outputs(boxes, boxes, boxes)

        class Postprocessor:
            remap_mscoco_category = False

            def __call__(self, outputs, orig_sizes):
                return [{"boxes": torch.tensor([[30., 40., 70., 80.]]),
                         "labels": torch.tensor([0]), "scores": torch.tensor([0.9])}]

        target = {"boxes": torch.tensor([[30., 40., 70., 80.]]),
                  "labels": torch.tensor([0]), "image_id": torch.tensor([7]),
                  "orig_size": torch.tensor([100, 200])}
        original_boxes = target["boxes"].clone()

        class Loader(list):
            pass

        loader = Loader([(torch.zeros(1, 3, 100, 200), [target])])
        loader.sampler = DistributedSampler(range(1), num_replicas=2, rank=0, shuffle=False)
        for oracle in (False, True):
            with self.subTest(oracle=oracle):
                model = Model(oracle)
                evaluator = mock.Mock()
                evaluator.iou_types = ["bbox"]
                evaluator.coco_eval = {"bbox": SimpleNamespace(stats=torch.zeros(12))}
                with mock.patch.object(det_engine, "Validator") as validator, \
                     mock.patch.object(det_engine, "GTRefinementMetrics", wraps=GTRefinementMetrics) as accumulator, \
                     contextlib.redirect_stdout(io.StringIO()):
                    validator.return_value.compute_metrics.return_value = {}
                    stats, _ = det_engine.evaluate(
                        model, torch.nn.Identity(), Postprocessor(),
                        loader, evaluator,
                        torch.device("cpu"), epoch=0, use_wandb=False,
                    )
                self.assertEqual(len(stats["coco_eval_bbox"]), 12)
                torch.testing.assert_close(target["boxes"], original_boxes)
                if oracle:
                    self.assertIs(accumulator.call_args.kwargs["sampler"], loader.sampler)
                    torch.testing.assert_close(model.received_targets[0]["boxes"], original_boxes)
                    self.assertTrue(stats["oracle_evaluation"])
                    self.assertEqual(stats["oracle_geometry"]["groups"]["all"]["pairs"], 1)
                else:
                    accumulator.assert_not_called()
                    self.assertIsNone(model.received_targets)
                    self.assertNotIn("oracle_evaluation", stats)
                    self.assertNotIn("oracle_geometry", stats)

    def test_oracle_profiler_counts_parameters_without_deploying_or_running_forward(self):
        from src.misc import profiler_utils

        class OracleModel(torch.nn.Module):
            requires_gt = True

            def __init__(self):
                super().__init__()
                self.frozen = torch.nn.Parameter(torch.zeros(10), requires_grad=False)
                self.refiner = torch.nn.Parameter(torch.zeros(3))

            def deploy(self):
                raise AssertionError("A GT-dependent oracle cannot be deployed")

            def forward(self, images):
                raise AssertionError("Profiling must not invent GT for an oracle model")

        cfg = SimpleNamespace(
            model=OracleModel(),
            train_dataloader=SimpleNamespace(collate_fn=SimpleNamespace(base_size=100)),
        )
        with mock.patch.object(profiler_utils, "calculate_flops") as calculate:
            parameters, description = profiler_utils.stats(cfg)
        calculate.assert_not_called()
        self.assertEqual(parameters, 13)
        description = " ".join(description)
        self.assertIn("Trainable Params:3", description)
        self.assertIn("real images and GT are required", description)


if __name__ == "__main__":
    unittest.main()
