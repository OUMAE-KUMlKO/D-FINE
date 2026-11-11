"""Boundary-evidence candidate selection. No query or reference-box correction."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from .box_ops import box_cxcywh_to_xyxy, box_iou


def gather(values, indices):
    return values.gather(1, indices[..., None].expand(*indices.shape, values.shape[-1]))


@torch.no_grad()
def local_neighbors(boxes, image_size, k=8, cell_size=32):
    """Nine cells, at most K+1 candidates/cell; never allocate an M x M array."""
    b, m, _ = boxes.shape
    h, w = image_size
    nx, ny = math.ceil(w / cell_size), math.ceil(h / cell_size)
    xy = boxes[..., :2] * boxes.new_tensor([w, h])
    cells = (xy / cell_size).long()
    cells[..., 0].clamp_(0, nx - 1)
    cells[..., 1].clamp_(0, ny - 1)
    keys = cells[..., 1] * nx + cells[..., 0]
    order = keys.argsort(dim=1, stable=True)
    sorted_keys = keys.gather(1, order)
    offsets = torch.tensor([(x, y) for y in (-1, 0, 1) for x in (-1, 0, 1)], device=boxes.device)
    near = cells[:, :, None] + offsets
    valid_cell = ((near >= 0) & (near < near.new_tensor([nx, ny]))).all(-1)
    near_keys = near[..., 1] * nx + near[..., 0]
    start = torch.searchsorted(sorted_keys.contiguous(), near_keys.reshape(b, -1).contiguous()).reshape(b, m, 9)
    slots = start[..., None] + torch.arange(k + 1, device=boxes.device)
    slot_keys = sorted_keys.gather(1, slots.clamp(max=m - 1).reshape(b, -1)).reshape_as(slots)
    ids = order.gather(1, slots.clamp(max=m - 1).reshape(b, -1)).reshape(b, m, -1)
    valid = (valid_cell[..., None] & (slots < m) & (slot_keys == near_keys[..., None])).reshape(b, m, -1)
    valid &= ids != torch.arange(m, device=boxes.device)[None, :, None]
    neighbor_xy = gather(xy, ids.reshape(b, -1)).reshape(b, m, -1, 2)
    distance = (neighbor_xy - xy[:, :, None]).square().sum(-1).masked_fill(~valid, float('inf'))
    distance, chosen = distance.topk(k, dim=-1, largest=False)
    return ids.gather(2, chosen), distance.isfinite()


class BoundaryQuerySelector(nn.Module):
    def __init__(self, channels=256, num_classes=10, candidates=1200, k=8,
                 supervision="representative"):
        super().__init__()
        if supervision not in ('representative', 'quality_rank'):
            raise ValueError('Unknown selection supervision: ' + supervision)
        self.supervision = supervision
        self.candidates, self.k = candidates, k
        self.projection = nn.Conv2d(channels, 32, 1)
        self.boundary_encoder = nn.Sequential(nn.Linear(4 * 3 * 32, 128), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(channels + num_classes + 4 + 128, 128), nn.ReLU())
        self.relation = nn.Sequential(nn.Linear(128 * 2 + 4, 64), nn.ReLU(), nn.Linear(64, 1))
        self.existence = nn.Linear(128, 1)
        self.representative = nn.Sequential(nn.Linear(128 + 3, 64), nn.ReLU(), nn.Linear(64, 1))
        for head in (self.relation[-1], self.existence, self.representative[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def boundary_features(self, feature, boxes, image_size):
        b, m, _ = boxes.shape
        h, w = image_size
        corners = box_cxcywh_to_xyxy(boxes)
        x0, y0, x1, y1 = corners.unbind(-1)
        t = boxes.new_tensor([0., .5, 1.])
        xs = x0[..., None] + (x1 - x0)[..., None] * t
        ys = y0[..., None] + (y1 - y0)[..., None] * t
        edges = torch.stack((torch.stack((x0[..., None].expand_as(ys), ys), -1),
                             torch.stack((xs, y0[..., None].expand_as(xs)), -1),
                             torch.stack((x1[..., None].expand_as(ys), ys), -1),
                             torch.stack((xs, y1[..., None].expand_as(xs)), -1)), 2)
        normals = boxes.new_tensor([[4 / w, 0], [0, 4 / h], [4 / w, 0], [0, 4 / h]])
        points = edges[..., None, :] + normals[None, None, :, None, None, :] * boxes.new_tensor([-1., 0., 1.])[None, None, None, None, :, None]
        valid = ((points >= 0) & (points <= 1)).all(-1)
        projected = self.projection(feature)
        sampled = F.grid_sample(projected, (points * 2 - 1).reshape(b, m, 36, 2).to(projected.dtype), align_corners=False)
        sampled = sampled.reshape(b, 32, m, 4, 3, 3).permute(0, 2, 3, 4, 5, 1)
        pooled = (sampled * valid[..., None]).sum(3) / valid.sum(3).clamp(min=1)[..., None]
        return self.boundary_encoder(pooled.flatten(2))

    def forward(self, memory, logits, anchors, bbox_head, feature, image_size, num_queries):
        scores = logits.max(-1).values
        m = min(self.candidates, scores.shape[1])
        # Explicitly preserve the native Top300, including its tie behavior.
        original = scores.topk(num_queries, dim=1).indices
        rest = scores.detach().clone().scatter(1, original, float('-inf'))
        pool = torch.cat((original, rest.topk(m - num_queries, dim=1).indices), 1)
        candidate_memory, candidate_logits = gather(memory, pool), gather(logits, pool)
        unact = bbox_head(candidate_memory) + gather(anchors, pool)
        boxes = unact.detach().sigmoid()
        z = self.boundary_features(feature, boxes, image_size)
        state = self.fusion(torch.cat((candidate_memory, candidate_logits, boxes, z), -1))
        neighbors, valid = local_neighbors(boxes, image_size, self.k)
        b = boxes.shape[0]
        other_state = gather(state, neighbors.flatten(1)).reshape(b, m, self.k, -1)
        other_boxes = gather(boxes, neighbors.flatten(1)).reshape(b, m, self.k, 4)
        relation = self.relation(torch.cat((state[:, :, None].expand_as(other_state), other_state,
                                            other_boxes - boxes[:, :, None]), -1)).squeeze(-1)
        candidate_scores = scores.gather(1, pool)
        other_scores = candidate_scores.gather(1, neighbors.flatten(1)).reshape(b, m, self.k)
        existence = self.existence(state).squeeze(-1)
        prob = relation.sigmoid() * valid
        if self.supervision == 'quality_rank':
            prob = gate_relations(prob, existence, neighbors)
        count = valid.sum(-1).clamp(min=1)
        summary = torch.stack((prob.sum(-1) / count,
                               (prob * (other_scores > candidate_scores[..., None])).sum(-1) / count,
                               (prob * (other_scores - candidate_scores[..., None]).clamp(min=0)).sum(-1) / count), -1)
        representative = self.representative(torch.cat((state, summary), -1)).squeeze(-1)
        # Global indexing guarantees identical native ordering when delta == 0.
        ranked = torch.full_like(scores, float('-inf')).scatter(1, pool, candidate_scores + .5 * representative)
        selected = ranked.topk(num_queries, dim=1).indices
        inverse = torch.zeros_like(scores, dtype=torch.long).scatter(1, pool, torch.arange(m, device=pool.device)[None].expand(b, -1))
        chosen = inverse.gather(1, selected)
        aux = dict(boxes=boxes, existence=existence,
                   representative=representative, relation=relation,
                   neighbors=neighbors, neighbor_valid=valid, selected=chosen)
        if self.supervision == 'quality_rank':
            aux.update(quality_logits=representative, candidate_logits=candidate_logits.detach(),
                       ranking_scores=candidate_scores.detach() + .5 * representative)
        return gather(candidate_memory, chosen), gather(candidate_logits, chosen), gather(unact, chosen), aux


@torch.no_grad()
def association(boxes, gt):
    m, g = len(boxes), len(gt)
    owner = torch.full((m,), -1, device=boxes.device, dtype=torch.long)
    ambiguous = torch.zeros(m, device=boxes.device, dtype=torch.bool)
    representative = torch.zeros_like(ambiguous)
    representative_owner = owner.clone()
    if not g:
        return owner, ambiguous, representative, representative_owner
    iou, _ = box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(gt))
    inside = ((boxes[:, None, :2] - gt[None, :, :2]).abs() <= gt[None, :, 2:] / 2).all(-1)
    ratio = boxes[:, 2:].prod(-1)[:, None] / gt[:, 2:].prod(-1).clamp(min=1e-8)[None]
    valid = (iou >= .3) | ((iou >= .05) & inside & (ratio >= .25) & (ratio <= 4))
    quality = iou.masked_fill(~valid, -1)
    best, owners = quality.max(-1)
    owner[best >= 0] = owners[best >= 0]
    if g > 1:
        two = quality.topk(2, dim=1).values
        ambiguous = (two[:, 1] >= 0) & (two[:, 0] - two[:, 1] < .05)
    scale = gt[:, [2, 3, 2, 3]].clamp(min=1e-8)
    cost = 1 - iou + ((boxes[:, None] - gt[None]).abs() / scale[None]).mean(-1)
    # A forbidden edge costs more than all valid edges combined: max cardinality first.
    penalty = (cost[valid].max().item() + 1) * (min(m, g) + 1) if valid.any() else 1.
    rows, cols = linear_sum_assignment(cost.masked_fill(~valid, penalty).cpu().numpy())
    rows, cols = torch.as_tensor(rows, device=boxes.device), torch.as_tensor(cols, device=boxes.device)
    keep = valid[rows, cols]
    representative[rows[keep]] = True
    representative_owner[rows[keep]] = cols[keep]
    return owner, ambiguous, representative, representative_owner


def balanced_bce(logits, positive, valid, group):
    """Balance labels, then GT groups within each label (background is one group)."""
    loss = F.binary_cross_entropy_with_logits(logits.float(), positive.float(), reduction='none')
    terms = []
    for label in (True, False):
        mask = valid & (positive == label)
        if mask.any():
            ids, inv = group[mask].unique(return_inverse=True)
            sums = loss.new_zeros(len(ids)).scatter_add_(0, inv, loss[mask])
            counts = loss.new_zeros(len(ids)).scatter_add_(0, inv, torch.ones_like(loss[mask]))
            terms.append((sums / counts).mean())
    return torch.stack(terms).mean() if terms else logits.sum() * 0


def selection_losses(aux, targets):
    if 'quality_logits' in aux:
        return quality_selection_losses(aux, targets)
    totals = {key: [] for key in ('existence', 'representative', 'relation')}
    for i, target in enumerate(targets):
        owner, ambiguous, positive, rep_owner = association(aux['boxes'][i].float(), target['boxes'].float())
        assigned = owner >= 0
        valid = torch.ones_like(assigned)
        totals['existence'].append(balanced_bce(aux['existence'][i], assigned, valid, owner))
        groups = torch.where(positive, rep_owner, owner)
        totals['representative'].append(balanced_bce(aux['representative'][i], positive, ~ambiguous | positive, groups))
        neighbor = aux['neighbors'][i]
        clear = assigned & ~ambiguous
        edge_valid = aux['neighbor_valid'][i] & clear[:, None] & clear[neighbor]
        totals['relation'].append(balanced_bce(aux['relation'][i], owner[:, None] == owner[neighbor], edge_valid,
                                              owner[:, None].expand_as(neighbor)))
    return {'loss_query_' + key: torch.stack(values).mean() * (.25 if key == 'relation' else 1.)
            for key, values in totals.items()}


def gate_relations(probability, existence, neighbors):
    """Predicted gates only. Detach gates so ranking cannot evade existence BCE."""
    foreground = existence.detach().sigmoid()
    other = foreground.gather(1, neighbors.flatten(1)).reshape_as(neighbors)
    return probability * foreground[..., None] * other


@torch.no_grad()
def quality_targets(boxes, logits, target):
    """Class-correct coarse IoU; ambiguous ownership is ignored for quality/rank."""
    gt = target['boxes'].float()
    owner, ambiguous, _, _ = association(boxes.float(), gt)
    quality = boxes.new_zeros(len(boxes), dtype=torch.float32)
    assigned = owner >= 0
    if len(gt):
        iou, _ = box_iou(box_cxcywh_to_xyxy(boxes.float()), box_cxcywh_to_xyxy(gt))
        rows = torch.arange(len(boxes), device=boxes.device)[assigned]
        correct = logits.argmax(-1)[assigned] == target['labels'][owner[assigned]]
        quality[assigned] = iou[rows, owner[assigned]] * correct
    return quality, owner, ambiguous


@torch.no_grad()
def quality_pairs(quality, owner, valid, min_gap=.05):
    """Each GT's best candidate versus lower-quality candidates; O(M) pairs.

    Equal-quality candidates are not forced apart. Ties for best use pool order,
    which is native score order. GTs without a class-correct candidate add no pair.
    """
    rows = torch.arange(len(quality), device=quality.device)
    eligible = valid & (owner >= 0)
    if not eligible.any():
        return rows[:0], rows[:0]
    groups = int(owner[eligible].max()) + 1
    best_quality = quality.new_full((groups,), -1)
    best_quality.scatter_reduce_(0, owner[eligible], quality[eligible], reduce='amax', include_self=True)
    safe_owner = owner.clamp(min=0, max=groups-1)
    is_best = eligible & (quality == best_quality[safe_owner])
    best_index = rows.new_full((groups,), len(quality))
    best_index.scatter_reduce_(0, owner[is_best], rows[is_best], reduce='amin', include_self=True)
    inferior = eligible & (best_quality[safe_owner] > 0) & (best_quality[safe_owner] - quality >= min_gap)
    return best_index[owner[inferior]], rows[inferior]


def group_mean(values, group, weights=None):
    if not values.numel():
        return values.sum() * 0
    ids, inverse = group.unique(return_inverse=True)
    weights = torch.ones_like(values) if weights is None else weights
    sums = values.new_zeros(len(ids)).scatter_add_(0, inverse, values * weights)
    counts = values.new_zeros(len(ids)).scatter_add_(0, inverse, weights)
    return (sums / counts.clamp(min=1e-8)).mean()


def quality_loss(logits, quality, owner, valid):
    losses = F.binary_cross_entropy_with_logits(logits.float(), quality, reduction='none')
    terms = []
    for positive in (True, False):
        mask = valid & ((quality > 0) == positive)
        if mask.any():
            terms.append(group_mean(losses[mask], owner[mask]))
    return torch.stack(terms).mean() if terms else logits.sum() * 0


def relative_quality_loss(scores, quality, owner, valid):
    better, worse = quality_pairs(quality, owner, valid)
    if not len(better):
        return scores.sum() * 0
    gap = quality[better] - quality[worse]
    # Train the same class_logit + 0.5 * residual score used by inference.
    loss = F.softplus(gap - (scores[better].float() - scores[worse].float()))
    return group_mean(loss, owner[worse], weights=gap)


def relation_targets(owner, ambiguous, neighbors, neighbor_valid):
    known = ~ambiguous  # Includes definite background, unlike the original loss.
    valid = neighbor_valid & known[:, None] & known[neighbors]
    positive = (owner[:, None] >= 0) & (owner[:, None] == owner[neighbors])
    return positive, valid


def quality_selection_losses(aux, targets):
    totals = {key: [] for key in ('existence', 'quality', 'rank', 'relation')}
    for i, target in enumerate(targets):
        quality, owner, ambiguous = quality_targets(aux['boxes'][i], aux['candidate_logits'][i], target)
        assigned = owner >= 0
        totals['existence'].append(balanced_bce(aux['existence'][i], assigned,
                                               torch.ones_like(assigned), owner))
        totals['quality'].append(quality_loss(aux['quality_logits'][i], quality, owner, ~ambiguous))
        totals['rank'].append(relative_quality_loss(aux['ranking_scores'][i], quality, owner, ~ambiguous))
        neighbor = aux['neighbors'][i]
        positive, valid = relation_targets(owner, ambiguous, neighbor, aux['neighbor_valid'][i])
        totals['relation'].append(balanced_bce(aux['relation'][i], positive, valid,
                                              owner[:, None].expand_as(neighbor)))
    weights = dict(existence=1., quality=1., rank=.5, relation=.25)
    return {'loss_query_' + key: torch.stack(values).mean() * weights[key] for key, values in totals.items()}
