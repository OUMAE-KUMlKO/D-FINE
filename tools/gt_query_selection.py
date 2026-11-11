"""GT 覆盖筛选公共逻辑：供冻结诊断、12 轮微调和 80 轮全量训练共用。

该模块需要真实标注，是研究诊断工具；可部署筛选器仍位于 boundary_query_selection.py。
整理仅移动原有实现，保留候选池顺序、贪心交换、停止条件和返回值语义。
"""
import numpy as np
import torch

from src.zoo.dfine.boundary_query_selection import gather
from src.zoo.dfine.box_ops import box_cxcywh_to_xyxy, box_iou


def candidate_pool(logits, count, candidates):
    """保留原生 TopK 及其并列排序，再补齐候选池。"""
    scores = logits.max(-1).values
    original = scores.topk(count, dim=1).indices
    remaining = scores.clone().scatter(1, original, float('-inf'))
    pool = torch.cat((original, remaining.topk(candidates - count, dim=1).indices), 1)
    return pool, scores.gather(1, pool)


def exchange_plans(quality, count, budgets=(10, 20, 40)):
    """逐步选取使 sum_g max_i q_ig 增益最大的单次交换。

    从原生 TopK 开始，只移除原生候选、加入尚未入选的候选，保持其余槽位。
    每一步同时计算加入收益和移除损失；收益不为正时提前停止。
    这是受限贪心诊断，不保证全局最优，也不是 AP 的数学上界。
    """
    q = np.asarray(quality, dtype=np.float64)
    selected = np.arange(count)
    available = np.ones(len(q), dtype=bool)
    available[:count] = False
    snapshots, trace = {}, []
    for step in range(1, max(budgets) + 1):
        drops = np.flatnonzero(selected < count)
        adds = np.flatnonzero(available)
        if not q.shape[1] or not len(drops) or not len(adds):
            break
        current = q[selected]
        winner = current.argmax(0)
        best = current[winner, np.arange(q.shape[1])]
        second = np.partition(current, -2, axis=0)[-2] if count > 1 else np.zeros_like(best)
        gain = np.maximum(q[adds] - best, 0).sum(1)
        # Loss for removing the current best of a GT, after accounting for
        # recovery by the added candidate. Tied bests have zero removal loss.
        losses = np.zeros((count, len(adds)), dtype=np.float64)
        np.add.at(losses, winner, np.maximum(best - np.maximum(second, q[adds]), 0).T)
        delta = gain[None] - losses[drops]
        maximum = delta.max()
        if maximum <= 1e-9:
            break
        # For identical gains, preserve the higher-ranked native query.
        flat = np.argmax(delta[::-1])
        row, column = np.unravel_index(flat, delta.shape)
        slot, added = drops[::-1][row], adds[column]
        removed = int(selected[slot])
        selected[slot] = added
        available[added] = False
        trace.append(dict(step=step, removed=removed, added=int(added), gain=float(maximum)))
        if step in budgets:
            snapshots[step] = selected.copy()
    for budget in budgets:
        if budget not in snapshots:
            snapshots[budget] = selected.copy()
    return snapshots, trace


@torch.no_grad()
def gt_candidate_selection(memory, logits, anchors, bbox_head, count, targets,
                           candidates=1200, budget=40, cached=None):
    """按类别正确的 IoU 覆盖 GT，返回无梯度候选张量、池内索引和统计。

    targets 的 boxes 必须是归一化 cxcywh。GT 只参与索引选择，不写入框或特征。
    cached 仅供候选生成器冻结、验证变换固定的旧微调实验使用；全量训练必须为 None。
    全量训练调用方需在 FP32 下计算选择，并在 no_grad 外重算选中张量，
    避免 AMP 缓存无梯度权重，同时保证候选分类/回归头收到辅助损失梯度。
    """
    pool, _ = candidate_pool(logits, count, min(candidates, logits.shape[1]))
    candidate_memory = gather(memory, pool)
    candidate_logits = gather(logits, pool)
    boxes = (bbox_head(candidate_memory) + gather(anchors, pool)).sigmoid()
    picks, records = [], []
    for i, target in enumerate(targets):
        image_id = int(target['image_id']) if 'image_id' in target else i
        if cached is not None and image_id in cached:
            pick, record = cached[image_id]
        else:
            gt = target['boxes'].float()
            iou = box_iou(box_cxcywh_to_xyxy(boxes[i].float()), box_cxcywh_to_xyxy(gt))[0]
            quality = (iou * (candidate_logits[i].argmax(-1)[:, None] == target['labels'][None])).cpu().numpy()
            objective = (quality >= .5).astype(float) + .01 / max(1, len(gt)) * quality
            plans, _ = exchange_plans(objective, count, (budget,))
            pick = plans[budget]
            record = dict(image_id=image_id, gt=len(gt), replaced=int((pick >= count).sum()),
                native_covered=int((quality[:count].max(0) >= .5).sum()),
                selected_covered=int((quality[pick].max(0) >= .5).sum()),
                pool_covered=int((quality.max(0) >= .5).sum()))
            if cached is not None:
                cached[image_id] = (pick, record)
        picks.append(torch.as_tensor(pick, device=memory.device))
        records.append(record)
    chosen = torch.stack(picks)
    selected = pool.gather(1, chosen)
    selected_memory = gather(memory, selected)
    return (selected_memory, gather(logits, selected),
            bbox_head(selected_memory) + gather(anchors, selected), None), chosen, records
