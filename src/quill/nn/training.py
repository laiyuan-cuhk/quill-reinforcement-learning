from __future__ import annotations

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


from typing import TypedDict, Iterator, IO, Any
from dataclasses import dataclass

from .model import ModelCfg, Model
from .batching import Batch
from .utils.ranking import evaluate_rankings, rank_candidates, to_dense_batch


class TrainCfg(TypedDict):
    model_config:       ModelCfg
    num_epochs:         int
    warmup_epochs:      int
    warmdown_epochs:    int
    batch_size_s:       int
    batch_size_h:       int
    max_tokens:         int
    backprop_every:     int
    max_lr:             float
    min_lr:             float
    entropy_coef:       float
    value_loss_coef:    float
    train_files:        list[str]
    dev_files:          list[str]
    test_files:         list[str]
    allow_self_loops:   bool
    half_precision:     bool


@dataclass
class TrainStats:
    loss:               tuple[float, ...] = ()
    ap:                 tuple[float, ...] = ()
    rp:                 tuple[float, ...] = ()

    def __add__(self, other: TrainStats) -> TrainStats:
        return TrainStats(loss=self.loss + other.loss, ap=self.ap + other.ap, rp=self.rp + other.rp)


class Trainer(Model):
    def compute_loss(self, batch: Batch, entropy_coef: float, value_loss_coef: float) -> tuple[Tensor, Tensor]:
        predictions = self.get_predictions(batch)
        scope_reprs, hole_reprs = self.encode(batch)
        values = self.value(hole_reprs)

        dense_preds, mask = to_dense_batch(predictions, batch.edge_index[1], fill_value=-1e08)
        dense_truths, _ = to_dense_batch(batch.premises, batch.edge_index[1], fill_value=0)

        logits = dense_preds.masked_fill(~mask, -1e08)
        probs = torch.softmax(logits, dim=-1)
        distribution = torch.distributions.Categorical(probs)
        actions = distribution.sample()
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy().mean()

        rewards = dense_truths[torch.arange(mask.size(0)), actions].float()
        advantages = rewards - values
        value_loss = torch.nn.functional.mse_loss(values, rewards)
        policy_loss = -(advantages.detach() * log_probs).mean()
        loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
        return predictions, loss

    def to_stats(self, batch: Batch, predictions: Tensor, loss: Tensor) -> TrainStats:
        zipped = tuple(zip(*evaluate_rankings(predictions, batch.edge_index[1], batch.premises)))
        ap, rp = zipped if zipped else ((), ())

        loss_value = loss.detach().cpu()
        if loss_value.ndim == 0:
            loss_values = (float(loss_value.item()),)
        else:
            loss_values = tuple(float(x) for x in loss_value.tolist())

        return TrainStats(loss=loss_values,
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
                    backprop_every: int,
                    entropy_coef: float,
                    value_loss_coef: float) -> TrainStats:
        self.train()

        epoch_stats = TrainStats()
        for i, batch in enumerate(epoch):
            epoch_stats += self.train_batch(
                batch=batch,
                optimizer=optimizer,
                scheduler=scheduler,
                backprop=(i + 1) % backprop_every == 0,
                entropy_coef=entropy_coef,
                value_loss_coef=value_loss_coef)
        return epoch_stats

    def train_batch(self,
                    batch: Batch,
                    optimizer: Optimizer,
                    scheduler: LRScheduler,
                    backprop: bool,
                    entropy_coef: float,
                    value_loss_coef: float) -> TrainStats:
        predictions, loss = self.compute_loss(batch, entropy_coef=entropy_coef, value_loss_coef=value_loss_coef)
        loss.mean().backward()

        if backprop:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        return self.to_stats(batch, predictions, loss)

    def eval_batch(self, batch: Batch, entropy_coef: float, value_loss_coef: float) -> TrainStats:
        predictions, loss = self.compute_loss(batch, entropy_coef=entropy_coef, value_loss_coef=value_loss_coef)
        return self.to_stats(batch, predictions, loss)

    def eval_epoch(self, epoch: Iterator[Batch], entropy_coef: float, value_loss_coef: float) -> TrainStats:
        self.eval()
        epoch_stats = TrainStats()

        with torch.no_grad():
            for i, batch in enumerate(epoch):
                epoch_stats += self.eval_batch(batch, entropy_coef=entropy_coef, value_loss_coef=value_loss_coef)
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
