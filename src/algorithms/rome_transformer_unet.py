import io
import logging
from typing import Any, IO, Optional, Union

import torch
import torch.nn as nn
# noinspection PyProtectedMember
from lightning_fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from omegaconf import DictConfig
from typing_extensions import Self

from src.algorithms.algorithm_base import AlgorithmBase
from src.utils import CompileParams, dice_loss

log = logging.getLogger(__name__)


class RomeTransformerUnet(AlgorithmBase):
    
    def __init__(
        self,
        compiled: CompileParams,
        use_ce: bool,
        use_dice: bool,
        error_tolerance: list,
        optimizer_conf: DictConfig = None,
        scheduler_conf: DictConfig = None,
        network: nn.Module = None,
        network_conf: DictConfig = None,
        gpu: int = None,
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
        
        assert use_ce or use_dice, "Loss function is not specified."
        self.use_ce = use_ce
        self.use_dice = use_dice
        self.mse = nn.MSELoss(reduction='none')
        self.error_tolerance = error_tolerance
    
    def pred(self, batch):
        input_image, sequence, supervision_image, image_size, ue_loc_y_x, map_center, ue_initial_lat_lon = batch
        # noinspection PyTypeChecker
        pred_image = self.network_field(
            torch.Tensor([input_image]).cuda(self.gpu),
            torch.Tensor([sequence]).cuda(self.gpu)
        )
        out = nn.functional.sigmoid(pred_image).detach().cpu().numpy()[0, 0]
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
    
    def step(self, batch, *args, **kwargs):
        input_image, sequence, sequence_lengths, supervision_image, image_size, ue_loc_y_x = batch
        pred_image = self.network_field(input_image, sequence)
        return self.get_metrics(pred_image, supervision_image, image_size, ue_loc_y_x)
    
    def get_metrics(
        self, pred_image: torch.Tensor, supervision_image: torch.Tensor, image_size: torch.Tensor,
        ue_loc_y_x: torch.Tensor
    ):
        max_ind_pred = pred_image.flatten(1).argmax(dim=-1)
        ue_location_pred = torch.stack(
            [max_ind_pred % max(pred_image[0][0].shape), max_ind_pred // max(pred_image[0][0].shape)], dim=1
        )  # .cuda()
        max_ind = supervision_image.flatten(1).argmax(dim=-1)
        
        mses_meters = self.mse(
            ue_location_pred.to(torch.float32), ue_loc_y_x.to(torch.float32)
        ).sum(dim=1).sqrt() * image_size / max(pred_image[0][0].shape)
        
        # noinspection PyUnresolvedReferences
        accuracies = {f"acc_{p}": (mses_meters < p).sum() / len(mses_meters) for p in self.error_tolerance}
        mse_meters = mses_meters.mean()
        
        pred_image_sigmoid = torch.sigmoid(pred_image)
        
        loss = 0
        if self.use_ce:
            loss += nn.functional.binary_cross_entropy_with_logits(pred_image, supervision_image)
        if self.use_dice:
            loss += dice_loss(pred_image_sigmoid[:, 0], supervision_image[:, 0], multiclass=False)
        
        metrics = {
            "loss": loss,
            **{acc: acc_val.to('cpu').detach() for acc, acc_val in accuracies.items()},
            'mse_meters': mse_meters.to('cpu').detach(),
        }
        
        return metrics
    
    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[_PATH, IO],
        map_location: _MAP_LOCATION_TYPE = None,
        hparams_file: Optional[_PATH] = None,
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Self:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        state_dict = {k.replace("_network", "network_field"): v for k, v in checkpoint["state_dict"].items()}
        
        if state_dict.keys() != checkpoint["state_dict"].keys():
            checkpoint["state_dict"] = state_dict
        
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        
        return super().load_from_checkpoint(
            buffer,
            map_location=map_location,
            hparams_file=hparams_file,
            strict=strict,
            **kwargs
        )
