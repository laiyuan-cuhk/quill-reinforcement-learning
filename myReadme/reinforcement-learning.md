# Reinforcement Learning in this project

The source code implements a reinforcement-learning-style training loop for premise selection in the Agda proof assistant setting. The core logic is in [src/quill/nn/training.py](../src/quill/nn/training.py), with the model architecture in [src/quill/nn/model.py](../src/quill/nn/model.py) and the training entry point in [scripts/train.py](../scripts/train.py).

## What the model is learning

The task is not a classic environment-driven RL problem with long trajectories. Instead, it is closer to a contextual bandit or policy-gradient setup:

- Each input example contains a set of candidate premises for a hole.
- The model produces a score for each candidate premise.
- The training procedure samples one action (one candidate) from a probability distribution over those candidates.
- The reward is sparse and binary: it is 1 if the sampled candidate is a correct premise and 0 otherwise.

In other words, the model learns to assign higher probability to good premises and lower probability to bad ones.

## Model structure

The neural model has two heads:

- a policy head, which produces scores used to form a categorical policy over candidate premises;
- a value head, which predicts a scalar value for the current example.

The policy head is used for action selection, while the value head provides a baseline used to reduce variance in the policy update.

## Formula used

The main training objective is implemented in `Trainer.compute_loss` and combines three terms:

1. Policy loss
2. Value loss
3. Entropy regularization

The code computes:

- logits from the model predictions,
- a softmax distribution over the candidate premises,
- a sampled action from that distribution,
- a reward from whether the sampled action is correct.

The mathematical form is:

$$
L = L_{policy} + \lambda_v L_{value} - \lambda_H H(\pi)
$$

where:

- $L_{policy} = -\mathbb{E}[A_t \log \pi(a_t \mid s_t)]$
- $A_t = r_t - V(s_t)$ is the advantage
- $L_{value} = \text{MSE}(V(s_t), r_t)$
- $H(\pi)$ is the entropy of the policy distribution

In the implementation:

- `advantages = rewards - values`
- `policy_loss = -(advantages.detach() * log_probs).mean()`
- `value_loss = mse_loss(values, rewards)`
- `loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy`

This is essentially REINFORCE-style policy gradient with a learned value baseline and an entropy bonus.

## Strategy used

The training strategy is as follows:

1. Encode the current batch of scopes and holes with the transformer-based encoder.
2. Produce policy scores for each possible premise-to-hole match.
3. Convert those scores into a probability distribution over candidates.
4. Sample an action from that distribution.
5. Compare the sampled action against the ground-truth premises.
6. Compute a reward and update the policy and value heads.
7. Use entropy regularization to avoid collapsing the policy too aggressively.

### Practical details from the implementation

- The policy is a categorical distribution over candidate premises.
- Invalid positions are masked with a very large negative value so they receive negligible probability.
- The reward is 1 for a correct premise and 0 otherwise.
- The value head predicts the expected reward for each example.
- The optimizer is AdamW with a learning-rate schedule.

## Training loop

The script in [scripts/train.py](../scripts/train.py) runs the loop as:

- load the tokenized data,
- build training and development batches,
- run training epochs,
- compute training and development loss,
- report mAP and R-Precision,
- save the best checkpoint based on development mAP.

## Hyperparameters used

The configuration in [data/config.json](../data/config.json) sets the training regime to:

- `num_epochs`: 1000
- `batch_size_s`: 1
- `batch_size_h`: 1
- `max_lr`: 0.0005
- `min_lr`: 1e-06
- `entropy_coef`: 0.01 (default in the training code)
- `value_loss_coef`: 0.5 (default in the training code)
- `allow_self_loops`: false

## Evaluation metrics

The project evaluates rankings using:

- mean Average Precision (mAP)
- R-Precision

These are computed in [src/quill/nn/utils/ranking.py](../src/quill/nn/utils/ranking.py).

## Summary

This project uses a compact RL-style training method for premise selection. The core idea is to train a policy that prefers correct premises, use a value network as a baseline, and regularize with entropy so the policy remains exploratory. The implementation is best understood as REINFORCE with a critic, applied to a structured ranking problem rather than a full sequential control task.
