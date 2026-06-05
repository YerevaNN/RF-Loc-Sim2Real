from typing import List, Optional

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class LinearWarmupCosineAnnealingLR(LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by cosine decay.

    The schedule is parameterized in epochs by default. When ``interval='step'``,
    epoch counts are converted to steps using ``steps_per_epoch``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: Optional[int],
        warmup_start_lr: float,
        warmup_end_lr: float,
        eta_min: float,
        last_epoch: int,
        decay_epochs: Optional[int] = None,
        interval: str = 'epoch',
        steps_per_epoch: int = None,
    ) -> None:
        if decay_epochs is None:
            if max_epochs is None:
                raise ValueError("Either decay_epochs or max_epochs must be provided")
            decay_epochs = max_epochs - warmup_epochs

        if decay_epochs <= 0:
            raise ValueError("decay_epochs must be positive")
        if max_epochs is not None and warmup_epochs + decay_epochs > max_epochs:
            raise ValueError("warmup_epochs + decay_epochs must be less than or equal to max_epochs")

        self.interval = interval
        if self.interval == 'step':
            if steps_per_epoch is None:
                raise ValueError("steps_per_epoch must be provided for step-based scheduling")
            self.warmup_steps = warmup_epochs * steps_per_epoch
            self.decay_steps = decay_epochs * steps_per_epoch
            self.max_steps = (max_epochs if max_epochs is not None else warmup_epochs + decay_epochs) * steps_per_epoch
        else:
            self.warmup_steps = warmup_epochs
            self.decay_steps = decay_epochs
            self.max_steps = max_epochs if max_epochs is not None else warmup_epochs + decay_epochs

        self.warmup_start_lr = warmup_start_lr
        self.warmup_end_lr = warmup_end_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def _compute_lr(self, current_step: int) -> float:
        if self.warmup_steps > 0 and current_step < self.warmup_steps:
            return (
                self.warmup_start_lr
                + (self.warmup_end_lr - self.warmup_start_lr)
                * current_step / max(1, self.warmup_steps - 1)
            )

        decay_start_step = self.warmup_steps
        decay_end_step = decay_start_step + self.decay_steps
        if current_step >= decay_end_step:
            return self.eta_min

        decay_progress = (current_step - decay_start_step) / max(1, self.decay_steps - 1)
        return self.eta_min + 0.5 * (self.warmup_end_lr - self.eta_min) * (
            1 + math.cos(math.pi * decay_progress)
        )

    def get_lr(self) -> List[float]:
        current_step = self.last_epoch
        return [self._compute_lr(current_step) for _ in self.base_lrs]

    def get_closed_form_lr(self) -> List[float]:
        return [self._compute_lr(self.last_epoch) for _ in self.base_lrs]


class WarmupStableDecayLR(LRScheduler):
    """
    Learning rate scheduler that warms up, holds stable, and then decays linearly.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_epochs (int): The number of epochs for the learning rate to warm up.
        stable_epochs (int): The number of epochs to keep the learning rate stable after warmup.
        max_epochs (int): The total number of epochs.
        warmup_start_lr (float): The starting learning rate for the warmup phase.
        stable_lr (float): The learning rate for the stable phase.
        eta_min (float): The minimum learning rate at the end of the decay phase.
        last_epoch (int): The index of the last epoch. Default: -1.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        decay_epochs: int,
        max_epochs: int,
        warmup_start_lr: float,
        stable_lr: float,
        eta_min: float,
        last_epoch: int,
        interval: str = 'epoch',
        steps_per_epoch: int = None,
    ) -> None:
        assert warmup_epochs + decay_epochs <= max_epochs, "Warmup epochs + decay epochs must be less than or equal to max epochs"

        self.interval = interval
        if self.interval == 'step':
            if steps_per_epoch is None:
                raise ValueError("steps_per_epoch must be provided for step-based scheduling")
            # Convert all epoch-based parameters to step-based
            self.warmup_steps = warmup_epochs * steps_per_epoch
            self.decay_steps = decay_epochs * steps_per_epoch
            self.max_steps = max_epochs * steps_per_epoch
        else:
            # Keep epoch-based parameters
            self.warmup_steps = warmup_epochs
            self.decay_steps = decay_epochs
            self.max_steps = max_epochs
        
        self.warmup_start_lr = warmup_start_lr
        self.stable_lr = stable_lr
        self.decay_end_lr = eta_min

        super().__init__(optimizer, last_epoch)
        
        print(f"warmup_steps: {self.warmup_steps}, decay_steps: {self.decay_steps}, max_steps: {self.max_steps}, stable_lr: {self.stable_lr}")
    
    def get_lr(self) -> List[float]:
        current_step = self.last_epoch
        if current_step <= self.warmup_steps and self.warmup_steps > 0:
            # Linear warmup phase
            return [
                self.warmup_start_lr
                + (self.stable_lr - self.warmup_start_lr)
                * current_step / self.warmup_steps
                for _ in self.base_lrs
            ]
        
        decay_start_step = self.max_steps - self.decay_steps
        if current_step < decay_start_step:
            # Stable phase
            return [self.stable_lr for _ in self.base_lrs]
                
        # Linear decay phase
        return [
            self.stable_lr
            - (self.stable_lr - self.decay_end_lr)
            * (current_step - decay_start_step + 1) / self.decay_steps
            for _ in self.base_lrs
        ]
