import json
import logging
import os
import random
from typing import Union

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from torch.utils.data import Dataset
from tqdm import tqdm

log = logging.getLogger(__name__)


class RomeDataset(Dataset):
    
    def __init__(
        self,
        split: str,
        dataset_json_paths: list[str],
        dataset_main_paths: list[str],
        min_num_bs: int,
        max_num_bs: int,
        hard_campaign: str,
        hard_campaign_ratio: float,
        dataset_types: list[str],
        map_size: int,
        features: tuple,
        feature_means: dict[str, float],
        feature_vars: dict[str, float],
        output_gaussian_sigma: float,
        decimal_places: int,
        eps_image: int,
        for_viz: bool,
        map_channels: str,
        crops_per_epoch: int,
        bs_power_dropout_rate: float = 0.0,
        *args, **kwargs
    ):
        super().__init__()
        
        self.split = split
        self.dataset_json_paths = dataset_json_paths
        self.dataset_main_paths = dataset_main_paths
        self.min_num_bs = min_num_bs
        self.max_num_bs = max_num_bs
        self.hard_campaign = hard_campaign
        self.hard_campaign_ratio = hard_campaign_ratio
        self.dataset_types = dataset_types
        
        self.features = features
        self.feature_means = feature_means
        self.feature_vars = feature_vars
        
        self.num_ues = 0
        self.decimal_places = decimal_places
        
        self.json_map_paths = self.get_json_map_paths()
        self.map_size = map_size
        self.output_gaussian_sigma = output_gaussian_sigma
        self.eps_image = eps_image
        self.for_viz = for_viz
        self.map_channels = map_channels
        self.crop_per_epoch = crops_per_epoch
        self.bs_power_dropout_rate = bs_power_dropout_rate
    
    def __getitem__(
        self, idx: int
    ) -> Union[
        tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]
    ]:
        while True:
            try:
                if self.split == "train":
                    idx = random.randint(0, len(self.json_map_paths) - 1)
                json_path, npz_path, dataset_idx = self.json_map_paths[idx]
                map_img = np.load(npz_path)["image"]
                
                if self.map_channels == "buildings":
                    map_img = map_img[:, :, 0]
                elif self.map_channels == "roads":
                    map_img = map_img[:, :, 1]
                elif self.map_channels == "both":
                    map_img = np.mean(map_img[:, :, :2], axis=-1)
                
                map_resized = resize(map_img, (self.map_size, self.map_size))
                
                with open(json_path, "r") as file:
                    map_info = json.load(file)
                orig_image_size = map_info["half_square_size_meters"] * 2 + 1
                assert orig_image_size == map_img.shape[0] == map_img.shape[1]
                
                map_center = np.array(map_info["center_coord"], dtype=np.float32)
                ue_initial_lat_lon = np.array(map_info["UE"]["lat_lon"], dtype=np.float32)
                ue_info = map_info["UE"]
                base_stations_info = map_info["BaseStations"]
                
                ue_orig_loc_y_x = np.array(ue_info["proj_map_pos"])
                ue_loc_y_x: np.ndarray = ue_orig_loc_y_x / orig_image_size * self.map_size
                
                map_resized = np.expand_dims(map_resized, axis=0).repeat(3, axis=0)
                map_resized[2] = np.full_like(map_resized[2], self.map_size / orig_image_size)
                
                if self.split == "train":
                    random.shuffle(base_stations_info)
                    num_bs = random.randint(self.min_num_bs, min(self.max_num_bs, len(base_stations_info)))
                else:
                    num_bs = min(self.max_num_bs, len(base_stations_info))
                
                base_stations_data = []
                curr_num_bs = 0
                
                for bs in base_stations_info:
                    if self.dataset_types[dataset_idx] == "dataSet" and bs["interpolated"]:
                        continue
                    bs_orig_loc_y_x = np.array(bs["proj_map_pos"])
                    bs_loc_y_x: np.ndarray = bs_orig_loc_y_x / orig_image_size * self.map_size
                    
                    bs_measurements = bs["measurements"]
                    bs_ue_data_values = []
                    for feature in self.features:
                        raw_value = bs_measurements.get(feature, 0.0)
                        if feature == "bs_power_dbm" and random.random() < self.bs_power_dropout_rate:
                            raw_value = 0.0
                        bs_ue_data_values.append(
                            (raw_value - self.feature_means[feature]) / self.feature_vars[feature]
                        )
                    bs_info_slice = (
                        slice(max(0, int(bs_loc_y_x[0]) - self.eps_image), int(bs_loc_y_x[0]) + self.eps_image),
                        slice(max(0, int(bs_loc_y_x[1]) - self.eps_image), int(bs_loc_y_x[1]) + self.eps_image)
                    )
                    
                    # Drawing base station location and a feature on the map in the form of a square
                    map_resized[0][bs_info_slice[0], bs_info_slice[1]] = bs_loc_y_x[0] / self.map_size
                    map_resized[1][bs_info_slice[0], bs_info_slice[1]] = bs_loc_y_x[1] / self.map_size
                    map_resized[2][bs_info_slice[0], bs_info_slice[1]] = bs_ue_data_values[0]
                    
                    # Saving base station info for sequence
                    base_stations_data.append(np.concatenate((bs_loc_y_x / self.map_size, bs_ue_data_values)))
                    
                    curr_num_bs += 1
                    if curr_num_bs == num_bs:
                        break
                
                base_stations_data = np.stack(base_stations_data)
                ue_loc_img = np.zeros_like(map_resized[0])
                ue_loc_img[tuple(ue_loc_y_x.astype(np.int16))] = 1.0
                ue_loc_img: np.ndarray = gaussian_filter(ue_loc_img, self.output_gaussian_sigma)
                ue_loc_img = ue_loc_img / ue_loc_img.max()
                
                result = (
                    map_resized.astype(np.float32),
                    base_stations_data.astype(np.float32),
                    ue_loc_img.astype(np.float32),
                    orig_image_size,
                    ue_loc_y_x.astype(np.float32)
                )
                if self.for_viz:
                    return result + (
                        map_center,
                        ue_initial_lat_lon
                    )
                return result
            except Exception as e:
                log.error(f"Error in __getitem__: {e}")
                idx = random.randint(0, len(self.json_map_paths) - 1)
                continue
    
    def __len__(self):
        if self.split == "train":
            return int(self.crop_per_epoch * self.num_ues)
        return len(self.json_map_paths)
    
    def get_json_map_paths(self) -> list[tuple[str, str, int]]:
        json_map_paths = []
        for dataset_idx, (json_path, main_path) in enumerate(zip(self.dataset_json_paths, self.dataset_main_paths)):
            dataset_type = self.dataset_types[dataset_idx]
            info_json_path = os.path.join(main_path, f"info_{dataset_type}.json")
            with open(info_json_path, "r") as file:
                info_json = json.load(file)
            
            log.info(f"Loading Json and PNG paths from {json_path}")
            
            with open(json_path, "r") as file:
                dataset_info: dict[str, dict[str, dict[str, int]]] = json.load(file)
                for campaign_id, ues in tqdm(dataset_info.items()):
                    self.num_ues += len(ues)
                    if campaign_id == self.hard_campaign:
                        ues: dict[str, dict[str, int]] = dict(
                            random.sample(list(ues.items()), int(len(ues) * self.hard_campaign_ratio))
                        )
                    for ueid, samples in ues.items():
                        for crop_id, _ in samples.items():
                            bs_count = info_json[campaign_id][ueid][crop_id]
                            if bs_count >= self.min_num_bs:
                                json_map_paths.append(
                                    (
                                        os.path.join(
                                            main_path, campaign_id, ueid, f"{crop_id}.json"
                                        ),
                                        os.path.join(
                                            main_path, campaign_id, ueid, f"{crop_id}.npz"
                                        ),
                                        dataset_idx
                                    )
                                )
                                if self.split != "train":
                                    break
                log.info(
                    f"Loading is Done. Number of UEs is {self.num_ues}. Number of samples is {len(json_map_paths)}"
                )
        
        return json_map_paths
