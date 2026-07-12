from __future__ import annotations

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


from typing import TypedDict, Iterator, IO, Any
from dataclasses import dataclass

from .model import ModelCfg, Model
from .batching import Batch
from .rl import RLCfg, DEFAULT_RL_CFG, reinforce_loss, expected_reward
from .utils.ranking import evaluate_rankings, rank_candidates, to_dense_batch


class TrainCfg(TypedDict):
    model_config:       ModelCfg
    rl_config:          RLCfg
    num_epochs:         int
    warmup_epochs:      int
    warmdown_epochs:    int
    batch_size_s:       int
    batch_size_h:       int
    max_tokens:         int
    backprop_every:     int
    max_lr:             float
    min_lr:             float
    train_files:        list[str]
    dev_files:          list[str]
    test_files:         list[str]
    allow_self_loops:   bool
    half_precision:     bool


@dataclass
class TrainStats:
    loss:               tuple[float, ...] = ()
    reward:             tuple[float, ...] = ()
    ap:                 tuple[float, ...] = ()
    rp:                 tuple[float, ...] = ()

    def __add__(self, other: TrainStats) -> TrainStats:
        return TrainStats(loss=self.loss + other.loss,
                          reward=self.reward + other.reward,
                          ap=self.ap + other.ap,
                          rp=self.rp + other.rp)


class Trainer(Model):
    def __init__(self, config: ModelCfg, rl_config: RLCfg | None = None):
        super().__init__(config)
        self.rl_config: RLCfg = rl_config if rl_config is not None else dict(DEFAULT_RL_CFG)

    def compute_loss(self, batch: Batch) -> tuple[Tensor, Tensor]:
        predictions = self.get_predictions(batch)
        loss = reinforce_loss(
            scores=predictions,
            targets=batch.premises.bool(),
            edge_index=batch.edge_index,
            num_samples=self.rl_config['num_samples'],
            entropy_coef=self.rl_config['entropy_coef'],
            use_baseline=self.rl_config['use_baseline'],
            reward_correct=self.rl_config['reward_correct'],
            reward_wrong=self.rl_config['reward_wrong'])
        return predictions, loss

    def to_stats(self, batch: Batch, predictions: Tensor, loss: Tensor) -> TrainStats:
        zipped = tuple(zip(*evaluate_rankings(predictions, batch.edge_index[1], batch.premises)))
        ap, rp = zipped if zipped else ((), ())
        reward = expected_reward(predictions, batch.premises.bool(), batch.edge_index)
        return TrainStats(loss=tuple(loss.tolist()),
                          reward=tuple(reward.tolist()),
                          ap=ap,
                          rp=rp)

    def save_checkpoint(self,
                        path: str,
                        optimizer: Optimizer,
                        scheduler: LRScheduler,
                        epoch: int,
                        best_ap: float) -> None:
        torch.save({'model': self.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_ap': best_ap}, path)

    def load_checkpoint(self,
                        path: str,
                        optimizer: Optimizer,
                        scheduler: LRScheduler,
                        map_location: str) -> tuple[int, float]:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        self.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['epoch'] + 1, checkpoint['best_ap']

    def train_epoch(self,
                    epoch: Iterator[Batch],
                    optimizer: Optimizer,
                    scheduler: LRScheduler,
                    backprop_every: int) -> TrainStats:
        self.train()

        epoch_stats = TrainStats()
        for i, batch in enumerate(epoch):
            epoch_stats += self.train_batch(
                batch=batch,
                optimizer=optimizer,
                scheduler=scheduler,
                backprop=(i + 1) % backprop_every == 0)
        return epoch_stats

    def train_batch(self,
                    batch: Batch,
                    optimizer: Optimizer,
                    scheduler: LRScheduler,
                    backprop: bool) -> TrainStats:
        predictions, loss = self.compute_loss(batch)
        loss.mean().backward()

        if backprop:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        return self.to_stats(batch, predictions, loss)

    def eval_batch(self, batch: Batch) -> TrainStats:
        predictions, loss = self.compute_loss(batch)
        return self.to_stats(batch, predictions, loss)

    def eval_epoch(self, epoch: Iterator[Batch]) -> TrainStats:
        self.eval()
        epoch_stats = TrainStats()

        with torch.no_grad():
            for i, batch in enumerate(epoch):
                epoch_stats += self.eval_batch(batch)
        return epoch_stats

    def infer_epoch(self, epoch: Iterator[Batch]) -> tuple[list[list[int]], list[set[int]]]:
        ps = []
        ts = []
        with torch.no_grad():
            for batch in epoch:
                predictions = rank_candidates(self.get_predictions(batch), batch.edge_index[1])[0].cpu().tolist()
                truths = to_dense_batch(batch.premises, batch.edge_index[1], fill_value=0)[0].cpu().bool().tolist()
                ps.extend(predictions)
                ts.extend([{i for i, x in enumerate(xs) if x} for xs in truths])
            return ps, ts

    def get_scores(self, epoch: Iterator[Batch]) -> tuple[list[list[float]], list[list[bool]]]:
        ps = []
        ts = []
        with torch.no_grad():
            for batch in epoch:
                predictions = to_dense_batch(self.get_predictions(batch), batch.edge_index[1], fill_value=-1e08)[0].cpu().tolist()
                truths = to_dense_batch(batch.premises, batch.edge_index[1], fill_value=0)[0].cpu().bool().tolist()
                ps.extend(predictions)
                ts.extend(truths)
            return ps, ts


class Logger:
    def __init__(self, stdout: IO[str], log: str):
        self.stdout = stdout
        self.log = log

    def write(self, obj: Any) -> None:
        with open(self.log, 'a') as f:
            f.write(f'{obj}')
        self.stdout.write(f'{obj}')

    def flush(self):
        self.stdout.flush()
