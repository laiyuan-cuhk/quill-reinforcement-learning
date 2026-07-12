"""Reinforcement-learning objective for premise selection.

Premise selection is framed here as a *contextual bandit* (a one-step Markov
decision process):

* **State**  - a hole together with the set of lemmas that are in scope for it.
* **Action** - picking one of the in-scope lemmas as a premise.
* **Policy** - ``pi(a | s) = softmax(scores)`` over the candidate lemmas of a
  hole, where ``scores`` are produced by the neural encoder.
* **Reward** - ``1`` if the sampled lemma is an actual premise of the hole,
  otherwise ``0``.

The policy is trained with the REINFORCE policy-gradient estimator

    grad J = E_{a ~ pi} [ (R(a) - b) * grad log pi(a | s) ]

where ``b`` is a variance-reducing baseline (the average sampled reward per
hole) and an entropy bonus is added to encourage exploration.  This replaces
the previous supervised ``infoNCE`` contrastive objective while reusing the
exact same network as the policy.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Categorical
from torch_geometric.utils import to_dense_batch

from typing import TypedDict


# Value used to mask out padded (non-existent) candidates so that the policy
# assigns them a vanishing probability.
_NEG_INF = -1e9


class RLCfg(TypedDict):
    num_samples:    int      # actions sampled per hole for the MC estimate
    entropy_coef:   float    # weight of the entropy-regularisation bonus
    use_baseline:   bool     # subtract a per-hole reward baseline
    reward_correct: float    # reward for selecting a true premise
    reward_wrong:   float    # reward for selecting a non-premise


DEFAULT_RL_CFG: RLCfg = {
    'num_samples': 8,
    'entropy_coef': 0.01,
    'use_baseline': True,
    'reward_correct': 1.0,
    'reward_wrong': 0.0,
}


def reinforce_loss(
        scores: Tensor,
        targets: Tensor,
        edge_index: Tensor,
        num_samples: int = 8,
        entropy_coef: float = 0.01,
        use_baseline: bool = True,
        reward_correct: float = 1.0,
        reward_wrong: float = 0.0) -> Tensor:
    """Compute the per-hole REINFORCE loss.

    :param scores: ``(num_edges,)`` policy logits, one per (lemma, hole) pair.
    :param targets: ``(num_edges,)`` mask that is truthy where the lemma is a
        genuine premise of the hole.
    :param edge_index: ``(2, num_edges)`` tensor whose second row holds the
        hole id of every edge; used to group candidates by hole.
    :returns: ``(num_holes,)`` tensor with the loss of each hole (episode).
    """
    if scores.numel() == 0:
        return scores

    hole_ids = edge_index[1]

    # Group the flat per-edge tensors into a dense ``(num_holes, max_cands)``
    # layout, padding shorter candidate sets.
    dense_scores, candidate_mask = to_dense_batch(scores, hole_ids, fill_value=_NEG_INF)
    reward_table = torch.where(targets.bool(),
                               scores.new_tensor(reward_correct),
                               scores.new_tensor(reward_wrong))
    dense_rewards, _ = to_dense_batch(reward_table, hole_ids, fill_value=0.0)

    # Padded candidates must never be selectable by the policy.
    dense_scores = dense_scores.masked_fill(~candidate_mask, _NEG_INF)

    policy = Categorical(logits=dense_scores)

    # Monte-Carlo roll-outs: sample ``num_samples`` actions per hole.
    actions = policy.sample((num_samples,))                       # (S, H)
    log_probs = policy.log_prob(actions)                          # (S, H)
    rewards = torch.gather(
        dense_rewards.unsqueeze(0).expand(num_samples, -1, -1),
        dim=2,
        index=actions.unsqueeze(-1)).squeeze(-1)                  # (S, H)

    if use_baseline and num_samples > 1:
        baseline = rewards.mean(dim=0, keepdim=True)
    else:
        baseline = torch.zeros_like(rewards)
    advantage = (rewards - baseline).detach()

    policy_gradient = -(advantage * log_probs).mean(dim=0)        # (H,)
    entropy = policy.entropy()                                    # (H,)
    return policy_gradient - entropy_coef * entropy


def expected_reward(scores: Tensor, targets: Tensor, edge_index: Tensor) -> Tensor:
    """Return the analytic expected reward ``sum_a pi(a) * R(a)`` per hole.

    Handy as a monitoring signal: it is the probability the policy assigns to
    picking a correct premise in a single draw.
    """
    if scores.numel() == 0:
        return scores
    hole_ids = edge_index[1]
    dense_scores, candidate_mask = to_dense_batch(scores, hole_ids, fill_value=_NEG_INF)
    dense_targets, _ = to_dense_batch(targets.float(), hole_ids, fill_value=0.0)
    dense_scores = dense_scores.masked_fill(~candidate_mask, _NEG_INF)
    probs = torch.softmax(dense_scores, dim=-1)
    return (probs * dense_targets).sum(dim=-1)
