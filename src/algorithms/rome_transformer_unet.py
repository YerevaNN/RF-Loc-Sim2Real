import io
import logging
from typing import Any, IO, Optional, Union

import torch
import torch.nn as nn
# noinspection PyProtectedMember
from lightning_fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from omegaconf import DictConfig, OmegaConf
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
        use_soft_argmax: bool = False,
        coord_loss_weight: float = 1.0,
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
        self.use_soft_argmax = use_soft_argmax
        self.coord_loss_weight = coord_loss_weight
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
    
    @staticmethod
    def _soft_argmax(logits):
        """Differentiable soft-argmax: spatial expectation of softmax probabilities."""
        B, _, H, W = logits.shape
        probs = torch.softmax(logits.view(B, -1), dim=-1).view(B, 1, H, W)

        device = logits.device
        coords_y = torch.arange(H, device=device, dtype=logits.dtype)
        coords_x = torch.arange(W, device=device, dtype=logits.dtype)
        grid_y, grid_x = torch.meshgrid(coords_y, coords_x, indexing='ij')

        pred_y = (probs[:, 0] * grid_y).sum(dim=(1, 2))
        pred_x = (probs[:, 0] * grid_x).sum(dim=(1, 2))
        return torch.stack([pred_y, pred_x], dim=1)

    def get_metrics(
        self, pred_image: torch.Tensor, supervision_image: torch.Tensor, image_size: torch.Tensor,
        ue_loc_y_x: torch.Tensor
    ):
        H, W = pred_image.shape[-2], pred_image.shape[-1]

        if self.use_soft_argmax:
            ue_location_pred_y_x = self._soft_argmax(pred_image)
        else:
            max_ind_pred = pred_image.flatten(1).argmax(dim=-1)
            ue_location_pred_y_x = torch.stack(
                [max_ind_pred // W, max_ind_pred % W], dim=1
            )

        mses_meters = self.mse(
            ue_location_pred_y_x.to(torch.float32), ue_loc_y_x.to(torch.float32)
        ).sum(dim=1).sqrt() * image_size / max(H, W)

        # noinspection PyUnresolvedReferences
        accuracies = {f"acc_{p}": (mses_meters < p).sum() / len(mses_meters) for p in self.error_tolerance}
        mse_meters = mses_meters.mean()

        pred_image_sigmoid = torch.sigmoid(pred_image)

        loss = 0
        if self.use_ce:
            loss += nn.functional.binary_cross_entropy_with_logits(pred_image, supervision_image)
        if self.use_dice:
            loss += dice_loss(pred_image_sigmoid[:, 0], supervision_image[:, 0], multiclass=False)
        if self.use_soft_argmax:
            loss += self.coord_loss_weight * nn.functional.l1_loss(
                ue_location_pred_y_x, ue_loc_y_x.to(ue_location_pred_y_x.dtype)
            )

        metrics = {
            "loss": loss,
            **{acc: acc_val.to('cpu').detach() for acc, acc_val in accuracies.items()},
            'mse_meters': mse_meters.to('cpu').detach(),
        }

        return metrics

    @staticmethod
    def _get_expected_mlp_input_dim(network_conf: Optional[Union[str, DictConfig]]) -> Optional[int]:
        if network_conf is None:
            return None

        network_conf = OmegaConf.create(network_conf)
        mlp_input_dim = network_conf.get("mlp_input_dim")
        return None if mlp_input_dim is None else int(mlp_input_dim)

    @staticmethod
    def _pad_mlp_input_weights(state_dict: dict[str, torch.Tensor], expected_input_dim: Optional[int]) -> bool:
        if expected_input_dim is None:
            return False

        patched = False
        for key, weight in list(state_dict.items()):
            if not key.endswith("vit_pp.mlp.0.weight"):
                continue
            if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
                continue

            checkpoint_input_dim = weight.shape[1]
            if checkpoint_input_dim == expected_input_dim:
                continue
            if checkpoint_input_dim > expected_input_dim:
                log.warning(
                    "Checkpoint parameter %s has input dim %s, but the configured model expects %s.",
                    key,
                    checkpoint_input_dim,
                    expected_input_dim,
                )
                continue

            padding = weight.new_zeros(weight.shape[0], expected_input_dim - checkpoint_input_dim)
            state_dict[key] = torch.cat([weight, padding], dim=1)
            log.info(
                "Padded checkpoint parameter %s from input dim %s to %s with zero weights.",
                key,
                checkpoint_input_dim,
                expected_input_dim,
            )
            patched = True

        return patched
    
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
        patched = cls._pad_mlp_input_weights(
            state_dict,
            cls._get_expected_mlp_input_dim(kwargs.get("network_conf")),
        )
        
        if patched or state_dict.keys() != checkpoint["state_dict"].keys():
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
