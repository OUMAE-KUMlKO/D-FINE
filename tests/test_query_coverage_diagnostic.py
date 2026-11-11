import unittest
import numpy as np
import torch
from types import SimpleNamespace
from tools.diagnose_query_coverage import exchange_plans, cardinality_hits, detection_hits


class CoverageDiagnosticTests(unittest.TestCase):
    def test_replaces_duplicate_with_missing_instance(self):
        q = np.array([[.9, 0], [.85, 0], [0, .65]])
        plans, trace = exchange_plans(q, 2, (1, 2))
        np.testing.assert_array_equal(plans[1], [0, 2])
        np.testing.assert_array_equal(plans[2], plans[1])
        self.assertAlmostEqual(trace[0]['gain'], .65)

    def test_swap_accounts_for_lost_coverage(self):
        q = np.array([[.9, 0], [0, .8], [.95, 0]])
        plans, trace = exchange_plans(q, 2, (1,))
        np.testing.assert_array_equal(plans[1], [2, 1])
        self.assertAlmostEqual(trace[0]['gain'], .05)

    def test_greedy_gains_match_bruteforce_and_budget(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            q = rng.random((12, 5))
            plans, trace = exchange_plans(q, 4, (1, 2, 3))
            selected = np.arange(4)
            for step in trace:
                old = q[selected].max(0).sum()
                all_gains = []
                for slot in np.flatnonzero(selected < 4):
                    for added in set(range(4, len(q))) - set(selected):
                        other = selected.copy(); other[slot] = added
                        all_gains.append(q[other].max(0).sum() - old)
                self.assertAlmostEqual(step['gain'], max(all_gains))
                selected[selected == step['removed']] = step['added']
            for budget, pick in plans.items():
                self.assertEqual(len(set(pick)), 4)
                self.assertLessEqual(int((pick >= 4).sum()), budget)
                self.assertGreaterEqual(q[pick].max(0).sum()+1e-9, q[:4].max(0).sum())

    def test_one_query_does_not_count_as_two_instances(self):
        q = np.array([[.8, .7], [0, 0]])
        self.assertEqual(int(cardinality_hits(q, .5).sum()), 1)

    def test_no_gt_preserves_native(self):
        plans, trace = exchange_plans(np.zeros((5, 0)), 2, (1, 3))
        np.testing.assert_array_equal(plans[3], [0, 1])
        self.assertEqual(trace, [])


    def test_final_detection_matching_ignores_crowd_and_duplicate(self):
        anns = [dict(id=1, image_id=0, category_id=0, bbox=[0, 0, 10, 10], area=100, iscrowd=0),
                dict(id=2, image_id=0, category_id=0, bbox=[20, 20, 10, 10], area=100, iscrowd=1)]
        coco = SimpleNamespace(imgToAnns={0: anns}, getCatIds=lambda: [0])
        prediction = dict(boxes=torch.tensor([[0., 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30]]),
                          labels=torch.tensor([0, 0, 0]), scores=torch.tensor([.9, .8, .7]))
        self.assertEqual(detection_hits(coco, 0, prediction, [1], 100), {'50': [0], '75': [0]})


if __name__ == '__main__':
    unittest.main()
