import logging
import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf

from src.algorithms.algorithm_base import AlgorithmBase
from src.utils import CompileParams, dice_loss

log = logging.getLogger(__name__)


def dann_lambda(p: float) -> float:
    """
    DANN lambda schedule from the original paper.
    p: progress in [0, 1] over training
    Returns lambda that ramps from ~0 to 1
    """
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


class RomeDANN(AlgorithmBase):
    """
    Domain Adversarial Neural Network (DANN) algorithm for Rome dataset.
    
    Trains a model with:
    - Task loss on labeled source data
    - Domain adversarial loss on both source and target data via GRL
    
    The network should be ViTPlusPlusDANN or similar that returns (task_logits, domain_logits).
    """
    
    def __init__(
        self,
        compiled: CompileParams,
        use_ce: bool,
        use_dice: bool,
        error_tolerance: list,
        domain_loss_weight: float = 1.0,
        optimizer_conf: DictConfig = None,
        scheduler_conf: DictConfig = None,
        network: nn.Module = None,
        network_conf: DictConfig = None,
        gpu: int = None,
        supervised: bool = False,
        *args, **kwargs
    ):
        super().__init__(
            compiled=compiled,
            optimizer_conf=optimizer_conf,
            scheduler_conf=scheduler_conf,
            network=network,
            network_conf=network_conf,
            gpu=gpu
        )
        # Enable manual optimization to support dual optimizers
        self.automatic_optimization = False
        
        assert use_ce or use_dice, "Loss function is not specified."
        self.use_ce = use_ce
        self.use_dice = use_dice
        self.mse = nn.MSELoss(reduction='none')
        self.error_tolerance = error_tolerance
        self.domain_loss_weight = domain_loss_weight
        self.supervised = supervised
    
    # ----- Dual optimizers (task vs domain) -----
    def configure_optimizers(self):
        """
        Create two optimizers:
          - Task optimizer: backbone + task head parameters (feature extractor + predictor)
          - Domain optimizer: domain discriminator parameters only
        Both are instantiated from the same optimizer config by default.
        If a scheduler config is provided, attach ONE scheduler to the task optimizer
        using the same dictionary structure as in AlgorithmBase (interval/monitor handling).
        """
        # Instantiate a base optimizer config from Hydra
        opt_conf = OmegaConf.create(self.optimizer_conf)
        # Collect parameter groups from the network
        task_params = list(self.network_field.task_parameters())
        domain_params = list(self.network_field.domain_parameters())
        
        opt_task = hydra.utils.instantiate(opt_conf, params=filter(lambda p: p.requires_grad, task_params))
        opt_domain = hydra.utils.instantiate(opt_conf, params=filter(lambda p: p.requires_grad, domain_params))
        
        # Default return: two optimizers, optionally with schedulers attached to BOTH
        if self.scheduler_conf is None:
            return [opt_task, opt_domain]

        # Build scheduler dicts for each optimizer (mirrors AlgorithmBase)
        def build_scheduler_for(optimizer):
            _conf = OmegaConf.create(self.scheduler_conf)
            _monitor = _conf.get("monitor", None)
            _interval = _conf.get("interval", "epoch")
            if "monitor" in _conf:
                del _conf.monitor
            _extra = {"optimizer": optimizer}
            if _interval == 'step':
                if not self.trainer.max_epochs or self.trainer.max_epochs == -1:
                    raise ValueError("max_epochs must be set in trainer for step-based scheduling.")
                steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
                _extra['steps_per_epoch'] = steps_per_epoch
            _scheduler = hydra.utils.instantiate(_conf, **_extra)
            _sch_opt = {"scheduler": _scheduler, "interval": _interval}
            if _monitor:
                _sch_opt["monitor"] = _monitor
            return _sch_opt

        sch_task = build_scheduler_for(opt_task)
        sch_domain = build_scheduler_for(opt_domain)

        return [
            {"optimizer": opt_task, "lr_scheduler": sch_task},
            {"optimizer": opt_domain, "lr_scheduler": sch_domain},
        ]
        # return (
        #     [opt_task, opt_domain],
        #     [
        #         {"scheduler": sch_task, "interval": "step", "name": "lr_task"},
        #         {"scheduler": sch_domain, "interval": "step", "name": "lr_domain"},
        #     ]
        # )
    
    def update_grl_lambda(self):
        """Update GRL lambda based on training progress"""
        if self.trainer.max_steps > 0:
            p = self.trainer.global_step / self.trainer.max_steps
        else:
            # Fallback to epoch-based if max_steps not available
            p = self.trainer.current_epoch / max(self.trainer.max_epochs, 1)
        
        lambd = dann_lambda(p) * self.domain_loss_weight
        self.network_field.set_lambda(lambd)
        return lambd
    
    def step(self, batch, split_name):
        """
        Unified step function that handles both training (paired batches) and 
        validation/test (single batches).
        """
        if split_name == "train":
            return self.dann_training_step(batch)
        else:
            # Validation and test use standard single-domain evaluation
            return self.standard_eval_step(batch)
    
    def dann_training_step(self, batch):
        """
        Training step for DANN with paired source/target batches.
        
        Args:
            batch: Tuple of (source_batch, target_batch)
        
        Returns:
            Dictionary with task_loss, domain_loss, total_loss, and accuracy metrics
        """
        
        # Grab optimizers for manual optimization
        opt_task, opt_domain = self.optimizers()
        sched_task, sched_domain = self.lr_schedulers()
        
        source_batch, target_batch = batch
        
        # Unpack source batch
        (source_input_image, source_sequence, source_sequence_lengths,
         source_supervision_image, source_image_size, source_ue_loc_y_x) = source_batch
        
        # Unpack target batch (we don't use target labels for task loss)
        (target_input_image, target_sequence, target_sequence_lengths,
         target_supervision_image, target_image_size, target_ue_loc_y_x) = target_batch
        
        # Update GRL lambda based on training progress
        lambd = self.update_grl_lambda()
        
        # Forward pass on source domain
        source_task_logits, source_domain_logits = self.network_field(
            source_input_image, source_sequence
        )
        
        # Forward pass on target domain
        target_task_logits, target_domain_logits = self.network_field(
            target_input_image, target_sequence
        )
        
        # ===== TASK LOSS (only on source data) =====
        task_metrics = self.get_task_metrics(
            source_task_logits, source_supervision_image,
            source_image_size, source_ue_loc_y_x
        )
        task_loss = task_metrics['loss']
        
        # ===== TARGET TASK LOSS (only on target data) =====
        target_task_metrics = self.get_task_metrics(
            target_task_logits, target_supervision_image,
            target_image_size, target_ue_loc_y_x
        )
        target_task_loss = target_task_metrics['loss']
        
        # ===== DOMAIN LOSS (on both source and target) =====
        # Domain labels: 0 for source, 1 for target
        source_domain_labels = torch.zeros(source_domain_logits.size(0), device=source_domain_logits.device)
        target_domain_labels = torch.ones(target_domain_logits.size(0), device=target_domain_logits.device)
        
        # Combine and compute BCE
        domain_logits = torch.cat([source_domain_logits, target_domain_logits], dim=0)
        domain_labels = torch.cat([source_domain_labels, target_domain_labels], dim=0)
        domain_loss = F.binary_cross_entropy_with_logits(domain_logits, domain_labels)
        
        # ===== MANUAL OPTIMIZATION =====
        opt_task.zero_grad(set_to_none=True)
        opt_domain.zero_grad(set_to_none=True)
        
        # Backprop task first (retain graph for domain), GRL flips gradients w.r.t. features
        self.manual_backward(task_loss, retain_graph=True)
        if self.supervised:
            self.manual_backward(target_task_loss, retain_graph=True)
        self.manual_backward(domain_loss)
        
        # Optional grad clipping could be added here
        opt_task.step()
        opt_domain.step()
        
        # Update schedulers
        sched_task.step()
        sched_domain.step()
        
        # # ===== TOTAL LOSS (for logging/return only) =====
        # total_loss = task_loss + domain_loss
        
        # ===== METRICS =====
        # Domain classification accuracy (for monitoring)
        domain_preds = (torch.sigmoid(domain_logits) > 0.5).float()
        domain_acc = (domain_preds == domain_labels).float().mean()
        
        metrics = {
            # "loss": total_loss,  # Main loss for optimization
            "loss": task_loss.detach(),
            "target_loss": target_task_loss.detach(),
            "domain_loss": domain_loss.detach(),
            "domain_acc": domain_acc.detach(),
            "grl_lambda": lambd,
            **{f"{k}": v for k, v in task_metrics.items() if k != 'loss'},
            **{f"target_{k}": v for k, v in target_task_metrics.items() if k != 'loss'}
        }
        
        return metrics
    
    def standard_eval_step(self, batch):
        """
        Standard evaluation step for validation/test (single domain at a time).
        Only computes task metrics, no domain adversarial loss.
        """
        input_image, sequence, sequence_lengths, supervision_image, image_size, ue_loc_y_x = batch
        
        # Forward pass - network returns (task_logits, domain_logits) but we only use task
        task_logits, _ = self.network_field(input_image, sequence)
        
        return self.get_task_metrics(task_logits, supervision_image, image_size, ue_loc_y_x)
    
    def get_task_metrics(
        self, pred_image: torch.Tensor, supervision_image: torch.Tensor,
        image_size: torch.Tensor, ue_loc_y_x: torch.Tensor
    ):
        """
        Compute task-specific metrics (localization accuracy and loss).
        Same as in RomeTransformerUnet but adapted for DANN.
        """
        # Argmax localization
        max_ind_pred = pred_image.flatten(1).argmax(dim=-1)
        ue_location_pred_y_x = torch.stack(
            [max_ind_pred // max(pred_image[0][0].shape), max_ind_pred % max(pred_image[0][0].shape)], dim=1
        )
        
        # MSE in meters
        mses_meters = self.mse(
            ue_location_pred_y_x.to(torch.float32), ue_loc_y_x.to(torch.float32)
        ).sum(dim=1).sqrt() * image_size / max(pred_image[0][0].shape)
        
        # Accuracy at different thresholds
        accuracies = {f"acc_{p}": (mses_meters < p).sum() / len(mses_meters) for p in self.error_tolerance}
        mse_meters = mses_meters.mean()
        
        # Task loss (CE and/or Dice)
        pred_image_sigmoid = torch.sigmoid(pred_image)
        
        loss = 0
        if self.use_ce:
            loss += F.binary_cross_entropy_with_logits(pred_image, supervision_image)
        if self.use_dice:
            loss += dice_loss(pred_image_sigmoid[:, 0], supervision_image[:, 0], multiclass=False)
        
        metrics = {
            "loss": loss,
            **{acc: acc_val.to('cpu').detach() for acc, acc_val in accuracies.items()},
            'mse_meters': mse_meters.to('cpu').detach(),
        }
        
        return metrics
    
    def pred(self, batch):
        """
        Prediction method for inference (not used during training).
        Same as RomeTransformerUnet.
        """
        input_image, sequence, supervision_image, image_size, ue_loc_y_x, map_center, ue_initial_lat_lon = batch
        
        # Forward pass - only use task logits
        pred_image, _ = self.network_field(
            torch.Tensor([input_image]).cuda(self.gpu),
            torch.Tensor([sequence]).cuda(self.gpu)
        )
        out = F.sigmoid(pred_image).detach().cpu().numpy()[0, 0]
        lat, lon = map_center[0], map_center[1]
        original_img_size = image_size
        original_lat, original_lon = ue_initial_lat_lon[0], ue_initial_lat_lon[1]
        
        return dict(
            out=out,
            center_lat=lat,
            center_lon=lon,
            original_img_size=original_img_size,
            original_ue_lat=original_lat,
            original_ue_lon=original_lon,
            ue_loc_y_x=ue_loc_y_x
        )
        
    # Override on_validation_epoch_end
    def on_validation_epoch_end(self) -> None:
        names = getattr(self.trainer.datamodule, 'val_dataloader_names', None)
        for validation_num in self.validation_step_outputs:
            outputs = self.validation_step_outputs[validation_num]
            if names and validation_num < len(names):
                split_name = f"val_{names[validation_num]}"
            else:
                split_name = f"val_{validation_num}"
            self.epoch_end(outputs, split_name=split_name)
        
        self.validation_step_outputs.clear()
    
    # Override on_test_epoch_end
    def on_test_epoch_end(self) -> None:
        names = getattr(self.trainer.datamodule, 'test_dataloader_names', None)
        for test_num in self.test_step_outputs:
            outputs = self.test_step_outputs[test_num]
            if names and test_num < len(names):
                split_name = f"test_{names[test_num]}"
            else:
                split_name = f"test_{test_num}"
            self.epoch_end(outputs, split_name=split_name)
        
        self.test_step_outputs.clear()

