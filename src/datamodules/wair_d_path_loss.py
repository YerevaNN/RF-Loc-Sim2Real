import numpy as np
import torch

from src.datamodules.datasets import WAIRDDatasetPathLoss
from src.datamodules.wair_d_base import DatamoduleBase


class PathLossDatamodule(DatamoduleBase):
    
    def __init__(
        self, batch_size: int, num_workers: int, drop_last: bool,
        multi_gpu: bool = False, *args, **kwargs
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.multi_gpu = multi_gpu
        super().__init__(
            batch_size=batch_size, num_workers=num_workers, drop_last=drop_last, multi_gpu=multi_gpu,
            *args, **kwargs
        )
    
    @staticmethod
    def collate_fn(
        batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        map_resized, base_stations_data, ue_loc_img, orig_image_size, ue_loc_y_x = zip(*batch)
        
        map_resized_batch = torch.stack([torch.tensor(m) for m in map_resized])
        # map_resized_batch = map_resized_batch.unsqueeze(1)
        # map_resized_batch = map_resized_batch.expand(-1, 3, -1, -1)
        ue_loc_img_batch = torch.stack([torch.tensor(u) for u in ue_loc_img])
        # ue_loc_img_batch = ue_loc_img_batch.unsqueeze(1)
        
        # noinspection DuplicatedCode
        max_base_stations = max([b.shape[0] for b in base_stations_data])
        
        base_stations_data_padded = [
            np.pad(
                b, ((0, max_base_stations - b.shape[0]), (0, 0)),
                mode='constant', constant_values=0
            ) for b in base_stations_data
        ]
        base_stations_data_batch = torch.stack([torch.tensor(b) for b in base_stations_data_padded])
        base_station_lengths = torch.tensor([len(b) for b in base_stations_data])
        
        orig_image_size_batch = torch.tensor(orig_image_size)
        ue_loc_y_x = torch.tensor(ue_loc_y_x)
        
        return (
            map_resized_batch,
            base_stations_data_batch,
            base_station_lengths,
            ue_loc_img_batch,
            orig_image_size_batch,
            ue_loc_y_x
        )
    
    def prepare_data(self) -> None:
        self.train_set_field = WAIRDDatasetPathLoss(
            split="train",
            *self.args, **self.kwargs,
        )
        self.val_set_field = WAIRDDatasetPathLoss(
            split="val",
            *self.args, **self.kwargs,
        )
        self.test_set_field = WAIRDDatasetPathLoss(
            split="test",
            *self.args, **self.kwargs,
        )
