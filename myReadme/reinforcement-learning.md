# Reinforcement Learning in this project

The reinforcement-learning style training is implemented mainly in the training module.

## Where it appears

- [src/quill/nn/training.py](src/quill/nn/training.py)
  - This is the core RL part.
  - It computes a policy loss, a value loss, and entropy regularization.
  - It uses rewards from correct premise selections and builds advantages from them.

- [src/quill/nn/model.py](src/quill/nn/model.py)
  - The model includes a policy head and a value head, which are used by the training objective.

- [scripts/train.py](scripts/train.py)
  - This is the training entry point.
  - It creates the trainer, runs training epochs and batches, and updates the optimizer.

## Main idea

The model scores candidate premises, then training uses those scores to optimize:

1. a policy objective that favors correct premises,
2. a value objective that predicts the reward,
3. an entropy term to keep the policy from becoming too deterministic.

## Key functions

- `Trainer.compute_loss`
- `Trainer.train_epoch`
- `Trainer.train_batch`
- `Trainer.eval_batch`
