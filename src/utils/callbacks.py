from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from src.utils.schedulers import WarmupStableDecayLR


class DecayPhaseCheckpoint(Callback):
    def __init__(self, dirpath: str, filename: str = "pre_decay_checkpoint"):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.saved = False

    def _save_checkpoint(self, trainer: pl.Trainer, current_unit: int, unit_name: str):
        """Helper function to save a checkpoint."""
        if self.saved:
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.dirpath / f"{self.filename}_at_{unit_name}_{current_unit}.ckpt"
        trainer.save_checkpoint(checkpoint_path)
        self.saved = True
        
    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx
    ) -> None:
        scheduler_config = trainer.lr_scheduler_configs[0]
        scheduler = scheduler_config.scheduler

        if self.saved or not isinstance(scheduler, WarmupStableDecayLR) or scheduler.interval != 'step':
            return
        
        current_step = scheduler.last_epoch
        decay_start_step = scheduler.max_steps - scheduler.decay_steps
        
        if current_step == decay_start_step - 1:
            self._save_checkpoint(trainer, current_step, "step")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        scheduler_config = trainer.lr_scheduler_configs[0]
        scheduler = scheduler_config.scheduler
        
        if self.saved or not isinstance(scheduler, WarmupStableDecayLR) or scheduler.interval != 'epoch':
            return
        
        current_epoch = trainer.current_epoch
        decay_start_epoch = scheduler.max_steps - scheduler.decay_steps
        
        if current_epoch == decay_start_epoch - 1:
            self._save_checkpoint(trainer, current_epoch, "epoch") 