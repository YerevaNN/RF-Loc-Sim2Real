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
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.saved:
            return
        
        # Assuming one scheduler
        scheduler_config = trainer.lr_scheduler_configs[0]
        scheduler = scheduler_config.scheduler
        if not isinstance(scheduler, WarmupStableDecayLR):
            return
        
        current_epoch = trainer.current_epoch
        decay_start_epoch = scheduler.max_epochs - scheduler.decay_epochs
        
        if current_epoch == decay_start_epoch - 1:
            self.dirpath.mkdir(parents=True, exist_ok=True)
            checkpoint_path = self.dirpath / f"{self.filename}_epoch_{current_epoch}.ckpt"
            trainer.save_checkpoint(checkpoint_path)
            self.saved = True 