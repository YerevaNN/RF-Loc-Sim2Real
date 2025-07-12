import logging
from collections import defaultdict
from typing import Any

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.flop_counter import FlopCounterMode

from src.utils import CompileParams

log = logging.getLogger(__name__)


class AlgorithmBase(pl.LightningModule):
    
    def __init__(
        self,
        compiled: CompileParams,
        optimizer_conf: DictConfig = None,
        scheduler_conf: DictConfig = None,
        network: nn.Module = None,
        network_conf: DictConfig = None,
        gpu: int = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        
        self.compile = compiled
        self.optimizer_conf = optimizer_conf
        self.scheduler_conf = scheduler_conf
        
        if network is None:
            self.network_field: nn.Module = hydra.utils.instantiate(
                OmegaConf.create(network_conf)
            )
        else:
            self.network_field: nn.Module = network
        
        self.gpu = gpu
        if self.gpu is not None:
            self.network_field.cuda(gpu)
        
        self.training_step_outputs = defaultdict(list)
        self.validation_step_outputs = defaultdict(lambda: defaultdict(list))
        self.test_step_outputs = defaultdict(lambda: defaultdict(list))
        
        self.num_flop = None
        self.first_step = True
        
        self.flop_counter = FlopCounterMode(display=False, depth=1)
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
    
    @property
    def network(self) -> nn.Module:
        return self.network_field
    
    def forward(self, *args, **kwargs):
        outputs = self.network_field(*args, **kwargs)
        return outputs
    
    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            OmegaConf.create(self.optimizer_conf),
            params=filter(lambda p: p.requires_grad, self.parameters()),
        )
        
        ret_opt = {"optimizer": optimizer}
        if self.scheduler_conf is not None:
            scheduler_conf = OmegaConf.create(self.scheduler_conf)
            # Get monitor if exists, else None
            monitor = scheduler_conf.get("monitor", None)
            if "monitor" in scheduler_conf:
                del scheduler_conf.monitor
            
            scheduler: LRScheduler = hydra.utils.instantiate(
                scheduler_conf, optimizer=optimizer
            )
            sch_opt = {"scheduler": scheduler}
            
            # noinspection PyUnboundLocalVariable
            if monitor:
                # noinspection PyUnboundLocalVariable
                sch_opt["monitor"] = monitor
            
            ret_opt.update({"lr_scheduler": sch_opt})
        
        return ret_opt
    
    def pred(self, batch):
        raise NotImplementedError
    
    def step(self, batch, *args, **kwargs):
        raise NotImplementedError
    
    def step_base(self, batch, split_name):
        if self.first_step:
            with self.flop_counter:
                self.start.record()
                output = self.step(batch, split_name)
                self.end.record()
                torch.cuda.synchronize()
            
            if self.num_flop is None:
                self.num_flop = self.flop_counter.get_total_flops()
            
            if not self.compile.disable:
                log.info("Compiling the model.")
                self.network_field = torch.compile(
                    self.network_field,
                    fullgraph=self.compile.fullgraph,
                    dynamic=self.compile.dynamic,
                    backend=self.compile.backend,
                    mode=self.compile.mode,
                    options=self.compile.options,
                    disable=self.compile.disable,
                )
            
            self.first_step = False
        else:
            self.start.record()
            output = self.step(batch, split_name)
            self.end.record()
            torch.cuda.synchronize()
        
        # In test step we get 0 FLOP, so we use the previous known value
        
        flops = self.num_flop / (self.start.elapsed_time(self.end) / 1000)
        output["flops"] = flops
        progress_bar_dict = dict(flops=flops)
        
        progress_bar_dict["loss"] = output["loss"].item()
        self.trainer.progress_bar_metrics.update(progress_bar_dict)
        return output
    
    @staticmethod
    def convert_to_numpy(output_dict: dict[str, Any]):
        for key in output_dict:
            if isinstance(output_dict[key], torch.Tensor):
                output_dict[key] = output_dict[key].detach().cpu()
        
        return output_dict
    
    def on_train_batch_end(self, outputs: dict[str, Any], batch: Any, batch_idx: int) -> None:
        outputs = AlgorithmBase.convert_to_numpy(outputs)
        for key, value in outputs.items():
            self.training_step_outputs[key].append(value)
    
    def on_validation_batch_end(
        self, outputs: dict[str, Any], batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        outputs = AlgorithmBase.convert_to_numpy(outputs)
        for key, value in outputs.items():
            self.validation_step_outputs[dataloader_idx][key].append(value)
    
    def on_test_batch_end(
        self, outputs: dict[str, Any], batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        outputs = AlgorithmBase.convert_to_numpy(outputs)
        for key, value in outputs.items():
            self.test_step_outputs[dataloader_idx][key].append(value)
    
    def training_step(self, batch, *args, **kwargs):
        output = self.step_base(batch, split_name="train")
        return output
    
    def validation_step(self, batch, *args, **kwargs):
        output = self.step_base(batch, split_name="val")
        return output
    
    def test_step(self, batch, *args, **kwargs):
        output = self.step_base(batch, split_name="test")
        return output
    
    def epoch_end(self, outputs: dict[str, list], split_name):
        epoch_metrics = self.calculate_epoch_metrics(outputs)
        epoch_metrics = {f"{split_name}_{k}": v for k, v in epoch_metrics.items()}
        for checkpoint in self.trainer.checkpoint_callbacks:
            # noinspection PyUnresolvedReferences
            if checkpoint.monitor in epoch_metrics:
                # noinspection PyUnresolvedReferences
                epoch_metrics[checkpoint.monitor] = torch.Tensor(
                    epoch_metrics[checkpoint.monitor]
                )
        
        self.trainer.callback_metrics.update(epoch_metrics)
        if self.logger:
            self.logger.log_metrics(epoch_metrics, self.trainer.current_epoch)
        else:
            log.info(f"""\n{epoch_metrics}\n""")
    
    # def on_train_epoch_start(self) -> None:
    #     current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
    #     log.info(f"Epoch {self.trainer.current_epoch + 1}: Learning rate is {current_lr}")

    def on_train_epoch_end(self) -> None:
        outputs = self.training_step_outputs
        num_dataloaders = 1
        if isinstance(self.trainer.val_dataloaders, (list, tuple)):
            num_dataloaders = max(len(self.trainer.val_dataloaders), 1)
        for i in range(num_dataloaders):
            self.epoch_end(outputs, split_name=f"train_{i}")
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self) -> None:
        for validation_num in self.validation_step_outputs:
            outputs = self.validation_step_outputs[validation_num]
            self.epoch_end(outputs, split_name=f"val_{validation_num}")
        
        self.validation_step_outputs.clear()
    
    def on_test_epoch_end(self) -> None:
        for test_num in self.test_step_outputs:
            outputs = self.test_step_outputs[test_num]
            self.epoch_end(outputs, split_name=f"test_{test_num}")
        
        self.test_step_outputs.clear()
    
    def calculate_epoch_metrics(self, outputs: dict[str, list]) -> dict:
        epoch_metrics_sep = {}
        
        # add all output values to combined_group_metrics
        for metric_name, metric_values in outputs.items():
            epoch_metrics_sep[metric_name] = torch.tensor(
                sum(metric_values) / len(metric_values)
            )
        
        epoch_metrics_shared = {
            "learning_rate": torch.tensor(
                self.trainer.optimizers[0].param_groups[0]["lr"]
            )
        }
        
        if self.logger:
            # noinspection PyTypeChecker
            self.logger.log_metrics(epoch_metrics_shared, self.trainer.current_epoch)
        else:
            log.info(f"""\n{epoch_metrics_shared}\n""")
        
        return epoch_metrics_sep
