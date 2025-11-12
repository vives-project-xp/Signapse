import os
from typing import Optional, Dict

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

from .model_utils import MODEL_DIR


class EarlyStopping:
    """
    Early stopping utility.

    Args:
        patience: epochs to wait without improvement before stopping
        min_delta: minimum change to qualify as an improvement
        mode: 'min' (lower is better) or 'max' (higher is better)
        restore_best_weights: if True, restores model weights from the epoch with the best metric
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True,
    ) -> None:
        assert mode in {"min", "max"}
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best: Optional[float] = None
        self.num_bad_epochs: int = 0
        self.should_stop: bool = False
        self._best_state: Optional[Dict[str, torch.Tensor]] = None

    def _is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return (self.best - value) > self.min_delta
        else:  # max
            return (value - self.best) > self.min_delta

    def step(self, metric_value: float, model: Optional[torch.nn.Module] = None) -> bool:
        """Update early stopping state with a new metric value.
        Returns True if training should stop.
        """
        if self._is_improvement(metric_value):
            self.best = metric_value
            self.num_bad_epochs = 0
            if self.restore_best_weights and model is not None:
                # Keep a copy of best weights
                self._best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.should_stop = True
                if self.restore_best_weights and model is not None and self._best_state is not None:
                    model.load_state_dict(self._best_state)
        return self.should_stop


class ModelCheckpoint:
    """
    Save model checkpoints during training.

    Args:
        filepath: filename for checkpoint (will be placed inside MODEL_DIR if relative)
        monitor: metric name (for bookkeeping)
        mode: 'min' or 'max' for comparison if save_best_only
        save_best_only: if True, only save when monitored metric improves
    """

    def __init__(
        self,
        filepath: str = "best.ckpt",
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
    ) -> None:
        assert mode in {"min", "max"}
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best: Optional[float] = None

        # Resolve path
        if not os.path.isabs(filepath):
            if not os.path.exists(MODEL_DIR):
                os.makedirs(MODEL_DIR, exist_ok=True)
            filepath = os.path.join(MODEL_DIR, filepath)
        self.filepath = filepath

    def _is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best
        else:
            return value > self.best

    def save(self, epoch: int, model: torch.nn.Module, optimizer: Optimizer, metric_value: float) -> bool:
        """Save a checkpoint. Returns True if a file was written."""
        should_write = True
        if self.save_best_only:
            should_write = self._is_improvement(metric_value)
        if should_write:
            self.best = metric_value
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "monitor": self.monitor,
                    "metric": metric_value,
                    "mode": self.mode,
                },
                self.filepath,
            )
            return True
        return False


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: Optional[str] = None,
    **kwargs,
):
    """
    Factory for common learning rate schedulers.

    Args:
        optimizer: Optimizer to schedule
        scheduler_type: one of None, 'plateau', 'step', 'cosine'
        kwargs: params specific to the scheduler
    """
    if scheduler_type is None or scheduler_type == "none":
        return None
    st = scheduler_type.lower()
    if st == "plateau":
        factor = float(kwargs.get("factor", 0.5))
        patience = int(kwargs.get("patience", 3))
        min_lr = float(kwargs.get("min_lr", 1e-6))
        return ReduceLROnPlateau(optimizer, factor=factor, patience=patience, min_lr=min_lr)
    if st == "step":
        step_size = int(kwargs.get("step_size", 10))
        gamma = float(kwargs.get("gamma", 0.1))
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    if st == "cosine":
        T_max = int(kwargs.get("T_max", 10))
        eta_min = float(kwargs.get("eta_min", 0.0))
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    raise ValueError(f"Unknown scheduler_type: {scheduler_type}")
