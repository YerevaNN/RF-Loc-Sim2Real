from typing import List

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class LinearWarmupCosineAnnealingLR(LRScheduler):
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float,
        warmup_end_lr: float,
        eta_min: float,
        last_epoch: int,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.warmup_end_lr = warmup_end_lr
        self.eta_min = eta_min
        self.cosine_start_lr = warmup_end_lr
        self.cosine_end_lr = eta_min
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if cur_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr
                + (self.warmup_end_lr - self.warmup_start_lr)
                * (cur_epoch / (self.warmup_epochs - 1))
                for _ in self.base_lrs
            ]
        else:
            return [
                self.eta_min + 0.5 * (self.warmup_end_lr - self.eta_min) * (1 + math.cos(
                    math.pi * (cur_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                ))
                for _ in self.base_lrs
            ]
    
    def get_closed_form_lr(self) -> List[float]:
        if cur_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr
                + cur_epoch
                * (self.warmup_end_lr - self.warmup_start_lr)
                / max(1, self.warmup_epochs - 1)
                for _ in self.base_lrs
            ]
        return [
            self.eta_min + 0.5 * (self.warmup_end_lr - self.eta_min) * (1 + math.cos(
                math.pi * (cur_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            ))
            for _ in self.base_lrs
        ]


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
        assert warmup_epochs + decay_epochs < max_epochs, "Warmup epochs + decay epochs must be less than or equal to max epochs"

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
        
        print(f"warmup_steps: {self.warmup_steps}, decay_steps: {self.decay_steps}, max_steps: {self.max_steps}")
    
    def get_lr(self) -> List[float]:
        current_step = self.last_epoch
        if current_step <= self.warmup_steps:
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

