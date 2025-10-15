from typing import Union

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from src.datamodules.datasets import RomeDataset
from src.datamodules.wair_d_base import DatamoduleBase


class PairedDataset(Dataset):
    """
    Wraps two datasets (source and target) and returns paired samples.
    Useful for domain adaptation training where we need simultaneous batches from both domains.
    """
    
    def __init__(self, source_dataset: RomeDataset, target_dataset: RomeDataset):
        self.source = source_dataset
        self.target = target_dataset
        self.length = min(len(source_dataset), len(target_dataset))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Return tuple of (source_item, target_item)
        return self.source[idx], self.target[idx]


class RomeDANNDatamodule(DatamoduleBase):
    """
    Datamodule for DANN (Domain Adversarial Neural Network) training.
    
    Loads two separate domains (source and target) and provides:
    - Paired training batches from both domains
    - Separate validation loaders for both domains (6 total: 3 difficulties × 2 domains)
    - Separate test loaders for both domains (6 total: 3 difficulties × 2 domains)
    """
    
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        drop_last: bool,
        # Source domain paths
        source_dataset_main_paths: list[str],
        source_train_dataset_json_paths: list[str],
        source_hard_val_json_paths: list[str],
        source_hard_test_json_paths: list[str],
        source_medium_val_json_paths: list[str],
        source_medium_test_json_paths: list[str],
        source_easy_val_json_paths: list[str],
        source_easy_test_json_paths: list[str],
        # Target domain paths
        target_dataset_main_paths: list[str],
        target_train_dataset_json_paths: list[str],
        target_hard_val_json_paths: list[str],
        target_hard_test_json_paths: list[str],
        target_medium_val_json_paths: list[str],
        target_medium_test_json_paths: list[str],
        target_easy_val_json_paths: list[str],
        target_easy_test_json_paths: list[str],
        # Source domain filtering
        source_hard_campaign: str,
        source_hard_train_ratio: float,
        # Target domain filtering
        target_hard_campaign: str,
        target_hard_train_ratio: float,
        # Shared parameters
        dataset_types: list[str],
        multi_gpu: bool = False,
        *args,
        **kwargs
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        
        # Source domain paths
        self.source_dataset_main_paths = source_dataset_main_paths
        self.source_train_json_paths = source_train_dataset_json_paths
        self.source_hard_val_json_paths = source_hard_val_json_paths
        self.source_hard_test_json_paths = source_hard_test_json_paths
        self.source_medium_val_json_paths = source_medium_val_json_paths
        self.source_medium_test_json_paths = source_medium_test_json_paths
        self.source_easy_val_json_paths = source_easy_val_json_paths
        self.source_easy_test_json_paths = source_easy_test_json_paths
        
        # Target domain paths
        self.target_dataset_main_paths = target_dataset_main_paths
        self.target_train_json_paths = target_train_dataset_json_paths
        self.target_hard_val_json_paths = target_hard_val_json_paths
        self.target_hard_test_json_paths = target_hard_test_json_paths
        self.target_medium_val_json_paths = target_medium_val_json_paths
        self.target_medium_test_json_paths = target_medium_test_json_paths
        self.target_easy_val_json_paths = target_easy_val_json_paths
        self.target_easy_test_json_paths = target_easy_test_json_paths
        
        # Source domain filtering parameters
        self.source_hard_campaign = source_hard_campaign
        self.source_hard_train_ratio = source_hard_train_ratio
        
        # Target domain filtering parameters
        self.target_hard_campaign = target_hard_campaign
        self.target_hard_train_ratio = target_hard_train_ratio
        
        # Shared parameters
        self.dataset_types = dataset_types
        self.multi_gpu = multi_gpu
        
        # Source domain datasets
        self.source_train_set_field = None
        self.source_hard_val_set_field = None
        self.source_hard_test_set_field = None
        self.source_medium_val_set_field = None
        self.source_medium_test_set_field = None
        self.source_easy_val_set_field = None
        self.source_easy_test_set_field = None
        
        # Target domain datasets
        self.target_train_set_field = None
        self.target_hard_val_set_field = None
        self.target_hard_test_set_field = None
        self.target_medium_val_set_field = None
        self.target_medium_test_set_field = None
        self.target_easy_val_set_field = None
        self.target_easy_test_set_field = None
        
        # Dataloaders
        self.dataloader_val_source_hard = None
        self.dataloader_val_source_medium = None
        self.dataloader_val_source_easy = None
        self.dataloader_val_target_hard = None
        self.dataloader_val_target_medium = None
        self.dataloader_val_target_easy = None
        
        self.dataloader_test_source_hard = None
        self.dataloader_test_source_medium = None
        self.dataloader_test_source_easy = None
        self.dataloader_test_target_hard = None
        self.dataloader_test_target_medium = None
        self.dataloader_test_target_easy = None
        
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
        """Standard collate function from RomeDatamodule"""
        has_additional_data = len(batch[0]) > 5
        if not has_additional_data:
            map_resized, base_stations_data, ue_loc_img, orig_image_size, ue_loc_y_x = zip(*batch)
        else:
            map_resized, base_stations_data, ue_loc_img, orig_image_size, ue_loc_y_x, map_center, ue_initial_lat_lon = zip(
                *batch
            )
        
        map_resized_batch = torch.stack([torch.tensor(m) for m in map_resized])
        ue_loc_img_batch = torch.stack([torch.tensor(u) for u in ue_loc_img])
        ue_loc_img_batch = ue_loc_img_batch.unsqueeze(1)
        
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
            map_center_batch = torch.stack([torch.tensor(m) for m in map_center])
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
    
    @staticmethod
    def dann_collate_fn(batch):
        """
        Collate function for paired dataset.
        Unpacks (source_item, target_item) tuples and collates each domain separately.
        """
        source_items, target_items = zip(*batch)
        source_batch = RomeDANNDatamodule.collate_fn(list(source_items))
        target_batch = RomeDANNDatamodule.collate_fn(list(target_items))
        return (source_batch, target_batch)
    
    def prepare_data(self) -> None:
        # Use separate scalers for source and target to handle different distributions
        source_scaler = StandardScaler()
        target_scaler = StandardScaler()
        
        # SOURCE DOMAIN
        self.source_train_set_field = RomeDataset(
            split="train",
            dataset_json_paths=self.source_train_json_paths,
            dataset_main_paths=self.source_dataset_main_paths,
            hard_campaign=self.source_hard_campaign,
            hard_campaign_ratio=self.source_hard_train_ratio,
            scaler=source_scaler,
            dataset_types=self.dataset_types,
            *self.args, **self.kwargs,
        )
        self.source_hard_val_set_field = RomeDataset(
            split="hard_val",
            dataset_json_paths=self.source_hard_val_json_paths,
            dataset_main_paths=self.source_dataset_main_paths,
            hard_campaign=self.source_hard_campaign,
            hard_campaign_ratio=1.0,
            scaler=source_scaler,
            dataset_types=["dataSet"] * len(self.dataset_types),
            *self.args, **self.kwargs,
        )
        self.source_hard_test_set_field = RomeDataset(
            split="hard_test",
            dataset_json_paths=self.source_hard_test_json_paths,
            dataset_main_paths=self.source_dataset_main_paths,
            hard_campaign=self.source_hard_campaign,
            hard_campaign_ratio=1.0,
            scaler=source_scaler,
            dataset_types=["dataSet"] * len(self.dataset_types),
            *self.args, **self.kwargs,
        )
        self.source_medium_val_set_field = RomeDataset(
            split="medium_val",
            dataset_json_paths=self.source_medium_val_json_paths,
            dataset_main_paths=self.source_dataset_main_paths,
            hard_campaign=self.source_hard_campaign,
            hard_campaign_ratio=1.0,
            scaler=source_scaler,
            dataset_types=["dataSet"] * len(self.dataset_types),
            *self.args, **self.kwargs,
        )
        self.source_medium_test_set_field = RomeDataset(
            split="medium_test",
            dataset_json_paths=self.source_medium_test_json_paths,
            dataset_main_paths=self.source_dataset_main_paths,
            hard_campaign=self.source_hard_campaign,
            hard_campaign_ratio=1.0,
            scaler=source_scaler,
            dataset_types=["dataSet"] * len(self.dataset_types),
            *self.args, **self.kwargs,
        )
        self.source_easy_val_set_field = RomeDataset(
            split="easy_val",
            dataset_json_paths=self.source_easy_val_json_paths,
            dataset_main_paths=self.source_dataset_main_paths,
            hard_campaign=self.source_hard_campaign,
            hard_campaign_ratio=1.0,
            scaler=source_scaler,
            dataset_types=["dataSet"] * len(self.dataset_types),
            *self.args, **self.kwargs,
        )
        self.source_easy_test_set_field = RomeDataset(
            split="easy_test",
            dataset_json_paths=self.source_easy_test_json_paths,
            dataset_main_paths=self.source_dataset_main_paths,
            hard_campaign=self.source_hard_campaign,
            hard_campaign_ratio=1.0,
            scaler=source_scaler,
            dataset_types=["dataSet"] * len(self.dataset_types),
            *self.args, **self.kwargs,
        )
        
        # TARGET DOMAIN
        self.target_train_set_field = RomeDataset(
            split="train",
            dataset_json_paths=self.target_train_json_paths,
            dataset_main_paths=self.target_dataset_main_paths,
            hard_campaign=self.target_hard_campaign,
            hard_campaign_ratio=self.target_hard_train_ratio,
            scaler=target_scaler,
            dataset_types=self.dataset_types,
            *self.args, **self.kwargs,
        )
        self.target_hard_val_set_field = RomeDataset(
            split="hard_val",
            dataset_json_paths=self.target_hard_val_json_paths,
            dataset_main_paths=self.target_dataset_main_paths,
            hard_campaign=self.target_hard_campaign,
            hard_campaign_ratio=1.0,
            scaler=target_scaler,
            dataset_types=["dataSet"] * len(self.dataset_types),
            *self.args, **self.kwargs,
        )
        self.target_hard_test_set_field = RomeDataset(
            split="hard_test",
            dataset_json_paths=self.target_hard_test_json_paths,
            dataset_main_paths=self.target_dataset_main_paths,
            hard_campaign=self.target_hard_campaign,
            hard_campaign_ratio=1.0,
            scaler=target_scaler,
            dataset_types=["dataSet"] * len(self.dataset_types),
            *self.args, **self.kwargs,
        )
        self.target_medium_val_set_field = RomeDataset(
            split="medium_val",
            dataset_json_paths=self.target_medium_val_json_paths,
            dataset_main_paths=self.target_dataset_main_paths,
            hard_campaign=self.target_hard_campaign,
            hard_campaign_ratio=1.0,
            scaler=target_scaler,
            dataset_types=["dataSet"] * len(self.dataset_types),
            *self.args, **self.kwargs,
        )
        self.target_medium_test_set_field = RomeDataset(
            split="medium_test",
            dataset_json_paths=self.target_medium_test_json_paths,
            dataset_main_paths=self.target_dataset_main_paths,
            hard_campaign=self.target_hard_campaign,
            hard_campaign_ratio=1.0,
            scaler=target_scaler,
            dataset_types=["dataSet"] * len(self.dataset_types),
            *self.args, **self.kwargs,
        )
        self.target_easy_val_set_field = RomeDataset(
            split="easy_val",
            dataset_json_paths=self.target_easy_val_json_paths,
            dataset_main_paths=self.target_dataset_main_paths,
            hard_campaign=self.target_hard_campaign,
            hard_campaign_ratio=1.0,
            scaler=target_scaler,
            dataset_types=["dataSet"] * len(self.dataset_types),
            *self.args, **self.kwargs,
        )
        self.target_easy_test_set_field = RomeDataset(
            split="easy_test",
            dataset_json_paths=self.target_easy_test_json_paths,
            dataset_main_paths=self.target_dataset_main_paths,
            hard_campaign=self.target_hard_campaign,
            hard_campaign_ratio=1.0,
            scaler=target_scaler,
            dataset_types=["dataSet"] * len(self.dataset_types),
            *self.args, **self.kwargs,
        )
    
    def train_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader that yields paired batches: (source_batch, target_batch)
        """
        paired_dataset = PairedDataset(self.source_train_set_field, self.target_train_set_field)
        sampler = DistributedSampler(paired_dataset, drop_last=True) if self.multi_gpu else None
        return DataLoader(
            paired_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=None if self.multi_gpu else True,
            collate_fn=self.dann_collate_fn,
            drop_last=True
        )
    
    def val_dataloader(self) -> list[DataLoader]:
        """
        Returns 6 validation dataloaders in order:
        [source_hard, source_medium, source_easy, target_hard, target_medium, target_easy]
        """
        # Source validation loaders
        self.dataloader_val_source_hard = DataLoader(
            self.source_hard_val_set_field,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=DistributedSampler(
                self.source_hard_val_set_field, shuffle=False, drop_last=self.drop_last
            ) if self.multi_gpu else None,
            collate_fn=self.collate_fn,
            drop_last=self.drop_last
        )
        self.dataloader_val_source_medium = DataLoader(
            self.source_medium_val_set_field,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=DistributedSampler(
                self.source_medium_val_set_field, shuffle=False, drop_last=self.drop_last
            ) if self.multi_gpu else None,
            collate_fn=self.collate_fn,
            drop_last=self.drop_last
        )
        self.dataloader_val_source_easy = DataLoader(
            self.source_easy_val_set_field,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=DistributedSampler(
                self.source_easy_val_set_field, shuffle=False, drop_last=self.drop_last
            ) if self.multi_gpu else None,
            collate_fn=self.collate_fn,
            drop_last=self.drop_last
        )
        
        # Target validation loaders
        self.dataloader_val_target_hard = DataLoader(
            self.target_hard_val_set_field,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=DistributedSampler(
                self.target_hard_val_set_field, shuffle=False, drop_last=self.drop_last
            ) if self.multi_gpu else None,
            collate_fn=self.collate_fn,
            drop_last=self.drop_last
        )
        self.dataloader_val_target_medium = DataLoader(
            self.target_medium_val_set_field,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=DistributedSampler(
                self.target_medium_val_set_field, shuffle=False, drop_last=self.drop_last
            ) if self.multi_gpu else None,
            collate_fn=self.collate_fn,
            drop_last=self.drop_last
        )
        self.dataloader_val_target_easy = DataLoader(
            self.target_easy_val_set_field,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=DistributedSampler(
                self.target_easy_val_set_field, shuffle=False, drop_last=self.drop_last
            ) if self.multi_gpu else None,
            collate_fn=self.collate_fn,
            drop_last=self.drop_last
        )
        
        return [
            self.dataloader_val_source_hard,
            self.dataloader_val_source_medium,
            self.dataloader_val_source_easy,
            self.dataloader_val_target_hard,
            self.dataloader_val_target_medium,
            self.dataloader_val_target_easy
        ]
    
    def test_dataloader(self) -> list[DataLoader]:
        """
        Returns 6 test dataloaders in order:
        [source_hard, source_medium, source_easy, target_hard, target_medium, target_easy]
        """
        # Source test loaders
        self.dataloader_test_source_hard = DataLoader(
            self.source_hard_test_set_field,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=DistributedSampler(
                self.source_hard_test_set_field, shuffle=False, drop_last=self.drop_last
            ) if self.multi_gpu else None,
            collate_fn=self.collate_fn,
            drop_last=self.drop_last
        )
        self.dataloader_test_source_medium = DataLoader(
            self.source_medium_test_set_field,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=DistributedSampler(
                self.source_medium_test_set_field, shuffle=False, drop_last=self.drop_last
            ) if self.multi_gpu else None,
            collate_fn=self.collate_fn,
            drop_last=self.drop_last
        )
        self.dataloader_test_source_easy = DataLoader(
            self.source_easy_test_set_field,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=DistributedSampler(
                self.source_easy_test_set_field, shuffle=False, drop_last=self.drop_last
            ) if self.multi_gpu else None,
            collate_fn=self.collate_fn,
            drop_last=self.drop_last
        )
        
        # Target test loaders
        self.dataloader_test_target_hard = DataLoader(
            self.target_hard_test_set_field,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=DistributedSampler(
                self.target_hard_test_set_field, shuffle=False, drop_last=self.drop_last
            ) if self.multi_gpu else None,
            collate_fn=self.collate_fn,
            drop_last=self.drop_last
        )
        self.dataloader_test_target_medium = DataLoader(
            self.target_medium_test_set_field,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=DistributedSampler(
                self.target_medium_test_set_field, shuffle=False, drop_last=self.drop_last
            ) if self.multi_gpu else None,
            collate_fn=self.collate_fn,
            drop_last=self.drop_last
        )
        self.dataloader_test_target_easy = DataLoader(
            self.target_easy_test_set_field,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=DistributedSampler(
                self.target_easy_test_set_field, shuffle=False, drop_last=self.drop_last
            ) if self.multi_gpu else None,
            collate_fn=self.collate_fn,
            drop_last=self.drop_last
        )
        
        return [
            self.dataloader_test_source_hard,
            self.dataloader_test_source_medium,
            self.dataloader_test_source_easy,
            self.dataloader_test_target_hard,
            self.dataloader_test_target_medium,
            self.dataloader_test_target_easy
        ]
    
    @property
    def train_set(self):
        return self.source_train_set_field, self.target_train_set_field
    
    @property
    def test_set(self):
        return (
            self.source_hard_test_set_field, self.source_medium_test_set_field, self.source_easy_test_set_field,
            self.target_hard_test_set_field, self.target_medium_test_set_field, self.target_easy_test_set_field
        )
    
    @property
    def val_set(self):
        return (
            self.source_hard_val_set_field, self.source_medium_val_set_field, self.source_easy_val_set_field,
            self.target_hard_val_set_field, self.target_medium_val_set_field, self.target_easy_val_set_field
        )

