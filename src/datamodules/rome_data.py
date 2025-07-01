from typing import Union

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, DistributedSampler

from src.datamodules.datasets import RomeDataset
from src.datamodules.wair_d_base import DatamoduleBase


class RomeDatamodule(DatamoduleBase):
    
    def __init__(
        self, batch_size: int, num_workers: int, drop_last: bool, train_dataset_json_paths: list[str],
        hard_val_json_paths: list[str], hard_test_json_paths: list[str],
        medium_val_json_paths: list[str], medium_test_json_paths: list[str],
        easy_val_json_paths: list[str], easy_test_json_paths: list[str], dataset_main_paths: list[str],
        hard_train_ratio: float, dataset_types: list[str],
        multi_gpu: bool = False, *args, **kwargs
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        
        self.train_json_paths = train_dataset_json_paths
        
        self.hard_val_json_paths = hard_val_json_paths
        self.hard_test_json_paths = hard_test_json_paths
        
        self.medium_val_json_paths = medium_val_json_paths
        self.medium_test_json_paths = medium_test_json_paths
        
        self.easy_val_json_paths = easy_val_json_paths
        self.easy_test_json_paths = easy_test_json_paths
        
        self.hard_train_ratio = hard_train_ratio
        self.dataset_types = dataset_types
        
        self.dataset_main_paths = dataset_main_paths
        self.multi_gpu = multi_gpu
        
        self.train_set_field = None
        self.hard_val_set_field = None
        self.hard_test_set_field = None
        self.medium_val_set_field = None
        self.medium_test_set_field = None
        self.easy_val_set_field = None
        self.easy_test_set_field = None
        self.dataloader_val_hard = None
        self.dataloader_val_medium = None
        self.dataloader_val_easy = None
        self.dataloader_test_hard = None
        self.dataloader_test_medium = None
        self.dataloader_test_easy = None
        super().__init__(
            batch_size=batch_size, num_workers=num_workers, drop_last=drop_last, multi_gpu=multi_gpu,
            *args, **kwargs
        )
    
    @staticmethod
    def collate_fn(
        batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]
    ) -> Union[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor
        ]
    ]:
        has_additional_data = len(batch[0]) > 5
        if not has_additional_data:
            # batch for default case
            map_resized, base_stations_data, ue_loc_img, orig_image_size, ue_loc_y_x = zip(*batch)
        else:
            # batch when additional data is present
            map_resized, base_stations_data, ue_loc_img, orig_image_size, ue_loc_y_x, map_center, ue_initial_lat_lon = zip(
                *batch
            )
        
        map_resized_batch = torch.stack([torch.tensor(m) for m in map_resized])
        # map_resized_batch = map_resized_batch.unsqueeze(1)
        # map_resized_batch = map_resized_batch.expand(-1, 3, -1, -1)
        ue_loc_img_batch = torch.stack([torch.tensor(u) for u in ue_loc_img])
        ue_loc_img_batch = ue_loc_img_batch.unsqueeze(1)
        
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
        
        if has_additional_data:
            # noinspection PyUnboundLocalVariable
            map_center_batch = torch.stack([torch.tensor(m) for m in map_center])
            # noinspection PyUnboundLocalVariable
            ue_initial_lat_lon_batch = torch.stack([torch.tensor(u) for u in ue_initial_lat_lon])
            return (
                map_resized_batch,
                base_stations_data_batch,
                base_station_lengths,
                ue_loc_img_batch,
                orig_image_size_batch,
                ue_loc_y_x,
                map_center_batch,
                ue_initial_lat_lon_batch
            )
        else:
            return (
                map_resized_batch,
                base_stations_data_batch,
                base_station_lengths,
                ue_loc_img_batch,
                orig_image_size_batch,
                ue_loc_y_x
            )
    
    def prepare_data(self) -> None:
        scaler = StandardScaler()
        self.train_set_field = RomeDataset(
            split="train",
            dataset_json_paths=self.train_json_paths,
            dataset_main_paths=self.dataset_main_paths,
            hard_campaign_ratio=self.hard_train_ratio,
            scaler=scaler, dataset_types=self.dataset_types,
            *self.args, **self.kwargs,
        )
        self.hard_val_set_field = RomeDataset(
            split="hard_val",
            dataset_json_paths=self.hard_val_json_paths,
            dataset_main_paths=self.dataset_main_paths,
            hard_campaign_ratio=1.0,
            scaler=scaler, dataset_types=["dataSet"] * len(self.dataset_types),
            *self.args, **self.kwargs,
        )
        self.hard_test_set_field = RomeDataset(
            split="hard_test",
            dataset_json_paths=self.hard_test_json_paths,
            dataset_main_paths=self.dataset_main_paths,
            hard_campaign_ratio=1.0,
            scaler=scaler, dataset_types=["dataSet"] * len(self.dataset_types),
            *self.args, **self.kwargs,
        )
        
        self.medium_val_set_field = RomeDataset(
            split="medium_val",
            dataset_json_paths=self.medium_val_json_paths,
            dataset_main_paths=self.dataset_main_paths,
            hard_campaign_ratio=1.0,
            scaler=scaler, dataset_types=["dataSet"] * len(self.dataset_types),
            *self.args, **self.kwargs,
        )
        self.medium_test_set_field = RomeDataset(
            split="medium_test",
            dataset_json_paths=self.medium_test_json_paths,
            dataset_main_paths=self.dataset_main_paths,
            hard_campaign_ratio=1.0,
            scaler=scaler, dataset_types=["dataSet"] * len(self.dataset_types),
            *self.args, **self.kwargs,
        )
        
        self.easy_val_set_field = RomeDataset(
            split="easy_val",
            dataset_json_paths=self.easy_val_json_paths,
            dataset_main_paths=self.dataset_main_paths,
            hard_campaign_ratio=1.0,
            scaler=scaler, dataset_types=["dataSet"] * len(self.dataset_types),
            *self.args, **self.kwargs,
        )
        self.easy_test_set_field = RomeDataset(
            split="easy_test",
            dataset_json_paths=self.easy_test_json_paths,
            dataset_main_paths=self.dataset_main_paths,
            hard_campaign_ratio=1.0,
            scaler=scaler, dataset_types=["dataSet"] * len(self.dataset_types),
            *self.args, **self.kwargs,
        )
    
    def train_dataloader(self) -> DataLoader:
        sampler = DistributedSampler(self.train_set_field, drop_last=True) if self.multi_gpu else None
        return DataLoader(
            self.train_set_field, batch_size=self.batch_size, num_workers=self.num_workers,
            sampler=sampler, shuffle=None if self.multi_gpu else True, collate_fn=self.collate_fn,
            drop_last=True
        )
    
    def val_dataloader(self) -> list[DataLoader]:
        # noinspection DuplicatedCode
        self.dataloader_val_hard = DataLoader(
            self.hard_val_set_field, batch_size=self.batch_size, num_workers=self.num_workers,
            sampler=DistributedSampler(
                self.hard_val_set_field, shuffle=False, drop_last=self.drop_last
            ) if self.multi_gpu else None,
            collate_fn=self.collate_fn, drop_last=self.drop_last
        )
        self.dataloader_val_medium = DataLoader(
            self.medium_val_set_field, batch_size=self.batch_size, num_workers=self.num_workers,
            sampler=DistributedSampler(
                self.medium_val_set_field, shuffle=False, drop_last=self.drop_last
            ) if self.multi_gpu else None,
            collate_fn=self.collate_fn, drop_last=self.drop_last
        )
        self.dataloader_val_easy = DataLoader(
            self.easy_val_set_field, batch_size=self.batch_size, num_workers=self.num_workers,
            sampler=DistributedSampler(
                self.easy_val_set_field, shuffle=False, drop_last=self.drop_last
            ) if self.multi_gpu else None,
            collate_fn=self.collate_fn, drop_last=self.drop_last
        )
        return [self.dataloader_val_hard, self.dataloader_val_medium, self.dataloader_val_easy]
        # return self.dataloader_val_hard
    
    def test_dataloader(self) -> list[DataLoader]:
        # noinspection DuplicatedCode
        self.dataloader_test_hard = DataLoader(
            self.hard_test_set_field, batch_size=self.batch_size, num_workers=self.num_workers,
            sampler=DistributedSampler(
                self.hard_test_set_field, shuffle=False, drop_last=self.drop_last
            ) if self.multi_gpu else None,
            collate_fn=self.collate_fn, drop_last=self.drop_last
        )
        self.dataloader_test_medium = DataLoader(
            self.medium_test_set_field, batch_size=self.batch_size, num_workers=self.num_workers,
            sampler=DistributedSampler(
                self.medium_test_set_field, shuffle=False, drop_last=self.drop_last
            ) if self.multi_gpu else None,
            collate_fn=self.collate_fn, drop_last=self.drop_last
        )
        self.dataloader_test_easy = DataLoader(
            self.easy_test_set_field, batch_size=self.batch_size, num_workers=self.num_workers,
            sampler=DistributedSampler(
                self.easy_test_set_field, shuffle=False, drop_last=self.drop_last
            ) if self.multi_gpu else None,
            collate_fn=self.collate_fn, drop_last=self.drop_last
        )
        return [self.dataloader_test_hard, self.dataloader_test_medium, self.dataloader_test_easy]
        # return self.dataloader_test_hard
    
    @property
    def train_set(self):
        return self.train_set_field
    
    @property
    def test_set(self):
        return self.hard_test_set_field, self.medium_test_set_field, self.easy_test_set_field
    
    @property
    def val_set(self):
        return self.hard_val_set_field, self.medium_val_set_field, self.easy_val_set_field
