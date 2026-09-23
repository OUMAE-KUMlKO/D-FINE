import json
import unittest

import torch

from src.solver.distribution_coverage import DistributionCoverageMetrics


torch.set_num_threads(1)


class DistributionCoverageMetricsTest(unittest.TestCase):
    def test_nonuniform_candidates_respect_matches_mask_and_pixel_axes(self):
        positions = torch.tensor([[
            [[0.0] * 3] * 4,
            [[0.38, 0.4025, 0.45], [float('nan'), 0.305, 0.33],
             [0.595, 0.605, 0.7], [0.695, 0.7, float('inf')]],
        ]], dtype=torch.float64)
        valid = torch.tensor([[
            [[True] * 3] * 4,
            [[True, True, True], [False, True, True],
             [True, False, False], [False, False, False]],
        ]])
        probabilities = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
        sampling = {
            'candidate_positions': positions,
            'candidate_valid': valid,
            'candidate_probabilities': probabilities.expand_as(positions),
            'excluded_probability_mass': torch.tensor(
                [[[0.0] * 4, [0.0, 0.2, 0.8, 1.0]]], dtype=torch.float64),
        }
        baseline = torch.tensor(
            [[[0.1, 0.1, 0.1, 0.1], [0.51, 0.52, 0.2, 0.4]]],
            dtype=torch.float64,
        )
        targets = [{'boxes': torch.tensor(
            [[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.2, 0.4]],
            dtype=torch.float64,
        )}]
        metrics = DistributionCoverageMetrics(offset_pixels=1.0)
        metrics.update(sampling, baseline, targets,
                       [(torch.tensor([1]), torch.tensor([1]))], (100, 200))

        report = metrics.compute()
        self.assertEqual(report['matched_pairs'], 1)
        self.assertEqual(report['valid_metric_pairs'], 1)
        coverage = report['candidate_coverage']
        for edge, distance in [('left', 0.5), ('top', 0.5), ('right', 1.0)]:
            self.assertAlmostEqual(coverage['nearest_distance_px'][edge]['mean'], distance)
            self.assertEqual(coverage['inside_band_fraction'][edge], 1.0)
        self.assertIsNone(coverage['nearest_distance_px']['bottom']['mean'])
        self.assertEqual(coverage['no_valid_candidate_count']['bottom'], 1)
        self.assertEqual(coverage['inside_band_fraction']['bottom'], 0.0)
        self.assertEqual(coverage['all_edges_inside_band_fraction'], 0.0)
        self.assertEqual(coverage['range_contains_gt_fraction']['left'], 1.0)
        self.assertEqual(coverage['range_contains_gt_fraction']['top'], 0.0)
        self.assertEqual(coverage['range_contains_gt_fraction']['bottom'], 0.0)
        self.assertAlmostEqual(coverage['nearest_distance_gt_fraction']['left']['mean'], 0.0125)
        self.assertAlmostEqual(coverage['nearest_distance_gt_fraction']['right']['mean'], 0.025)
        self.assertAlmostEqual(report['baseline_coverage']['mean_tangent_coverage'], 0.95)
        for edge in ('left', 'top', 'right', 'bottom'):
            self.assertAlmostEqual(report['baseline_coverage']['edge_distance_px'][edge]['mean'], 2.0)
            self.assertEqual(report['baseline_coverage']['inside_band_fraction'][edge], 0.0)
        mass = report['probability_mass']
        for edge, raw, normalized in [('left', 0.3, 0.3), ('top', 0.3, 0.375), ('right', 0.2, 1.0)]:
            self.assertAlmostEqual(mass['near_gt_original_mean'][edge], raw)
            self.assertAlmostEqual(mass['near_gt_retained_normalized_mean'][edge], normalized)
        self.assertEqual(mass['near_gt_original_mean']['bottom'], 0.0)
        self.assertIsNone(mass['near_gt_retained_normalized_mean']['bottom'])
        self.assertAlmostEqual(mass['excluded_mean']['right'], 0.8)
        json.dumps(report, allow_nan=False)

    def test_quantiles_use_every_matched_sample_across_updates(self):
        metrics = DistributionCoverageMetrics()
        for errors in ([1.0, 3.0], [5.0]):
            count = len(errors)
            boxes = torch.tensor([[0.5, 0.5, 0.2, 0.4]], dtype=torch.float64).repeat(count, 1)
            edges = torch.tensor([0.4, 0.3, 0.6, 0.7], dtype=torch.float64)
            offsets = torch.tensor(errors, dtype=torch.float64) / 100
            positions = (edges[None, :] + offsets[:, None])[None, :, :, None]
            metrics.update({
                'candidate_positions': positions,
                'candidate_valid': torch.ones_like(positions, dtype=torch.bool),
                'candidate_probabilities': torch.ones_like(positions),
                'excluded_probability_mass': torch.zeros(1, count, 4),
            }, boxes[None], [{'boxes': boxes}],
                [(torch.arange(count), torch.arange(count))], (100, 100))

        report = metrics.compute()
        self.assertEqual(report['matched_pairs'], 3)
        for edge in ('left', 'top', 'right', 'bottom'):
            stats = report['candidate_coverage']['nearest_distance_px'][edge]
            self.assertEqual(stats['count'], 3)
            for key, expected in [('mean', 3.0), ('p50', 3.0), ('p90', 4.6), ('p95', 4.8)]:
                self.assertAlmostEqual(stats[key], expected)
            self.assertAlmostEqual(report['candidate_coverage']['inside_band_fraction'][edge], 1 / 3)
            self.assertEqual(report['baseline_coverage']['inside_band_fraction'][edge], 1.0)
        self.assertAlmostEqual(report['candidate_coverage']['all_edges_inside_band_fraction'], 1 / 3)

    def test_range_coverage_does_not_imply_nearby_candidate(self):
        boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]], dtype=torch.float64)
        edges = torch.tensor([0.4, 0.4, 0.6, 0.6], dtype=torch.float64)
        # Both axes have candidates +/- 4 input pixels from GT, but GT widths
        # and heights are 40 and 20 pixels on this non-square input.
        offsets = torch.tensor([4 / 200, 4 / 100, 4 / 200, 4 / 100], dtype=torch.float64)
        positions = torch.stack((edges - offsets, edges + offsets), dim=-1)[None, None]
        metrics = DistributionCoverageMetrics()
        metrics.update({
            'candidate_positions': positions,
            'candidate_valid': torch.ones_like(positions, dtype=torch.bool),
            'candidate_probabilities': torch.full_like(positions, 0.5),
            'excluded_probability_mass': torch.zeros(1, 1, 4),
        }, boxes, [{'boxes': boxes[0]}], [(torch.tensor([0]), torch.tensor([0]))], (100, 200))

        coverage = metrics.compute()['candidate_coverage']
        self.assertEqual(coverage['all_edges_range_contains_gt_fraction'], 1.0)
        self.assertEqual(coverage['all_edges_inside_band_fraction'], 0.0)
        for edge, fraction in [('left', 0.1), ('top', 0.2), ('right', 0.1), ('bottom', 0.2)]:
            self.assertEqual(coverage['range_contains_gt_fraction'][edge], 1.0)
            self.assertAlmostEqual(coverage['nearest_distance_px'][edge]['mean'], 4.0)
            self.assertAlmostEqual(coverage['nearest_distance_gt_fraction'][edge]['mean'], fraction)
            self.assertAlmostEqual(coverage['nearest_distance_gt_fraction'][edge]['p95'], fraction)

    def test_empty_matches_return_null_statistics_and_finite_json(self):
        metrics = DistributionCoverageMetrics()
        empty = torch.empty(1, 0, 4, 3)
        metrics.update({
            'candidate_positions': empty,
            'candidate_valid': empty.bool(),
            'candidate_probabilities': empty,
            'excluded_probability_mass': torch.empty(1, 0, 4),
        }, torch.empty(1, 0, 4), [{'boxes': torch.empty(0, 4)}],
            [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))],
            (100, 200))

        report = metrics.compute()
        self.assertEqual(report['matched_pairs'], 0)
        for coverage_key, distance_key in [('candidate_coverage', 'nearest_distance_px'),
                                           ('baseline_coverage', 'edge_distance_px')]:
            self.assertIsNone(report[coverage_key]['all_edges_inside_band_fraction'])
            self.assertIsNone(report[coverage_key]['inside_band_fraction']['left'])
            self.assertIsNone(report[coverage_key][distance_key]['left']['mean'])
            self.assertIsNone(report[coverage_key][distance_key]['left']['p95'])
        self.assertIsNone(report['probability_mass']['near_gt_original_mean']['left'])
        self.assertIsNone(report['probability_mass']['excluded_mean']['left'])
        self.assertIsNone(report['candidate_coverage']['range_contains_gt_fraction']['left'])
        self.assertIsNone(report['candidate_coverage']['nearest_distance_gt_fraction']['left']['mean'])
        self.assertIsNone(report['baseline_coverage']['mean_tangent_coverage'])
        json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
