import torch
from torch import Tensor
from torch_geometric.utils import to_dense_batch


def rank_candidates(x: Tensor, batch_ids: Tensor) -> tuple[Tensor, Tensor]:
    dense_x, x_mask = to_dense_batch(x=x, batch=batch_ids, fill_value=-1e08)
    _, perm = dense_x.sort(dim=-1, descending=True)
    # todo: you cant recover original values without the extra indexing ops
    return perm, x_mask.sum(dim=-1)


def average_precision(ranked_suggestions: list[int], relevant_items: set[int]) -> float:
    ap = 0
    for item in relevant_items:
        to_reach = set(ranked_suggestions[:ranked_suggestions.index(item) + 1])
        ap += len(to_reach & relevant_items)/(len(to_reach) + 1e-08)
    return ap/(len(relevant_items) + 1e-08)


def rprecision(ranked_suggestions: list[int], relevant_items: set[int]) -> float:
    return len(set(ranked_suggestions[:len(set(relevant_items))]) & relevant_items) / (len(relevant_items) + 1e-08)


def evaluate_rankings(predictions: Tensor, batch_ids: Tensor, truths: Tensor) -> list[tuple[float, float]]:
    if len(batch_ids) == 0:
        return []

    ranked_suggestions = rank_candidates(predictions, batch_ids)[0].cpu().tolist()
    dense_truths, _ = to_dense_batch(truths, batch_ids, fill_value=0)
    relevant_items = [{i for i, value in enumerate(row) if value} for row in dense_truths.cpu().bool().tolist()]
    return [(average_precision(rs, ri), rprecision(rs, ri))
            for rs, ri in zip(ranked_suggestions, relevant_items)]
