import logging
import os
import random
from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class WAIRDDatasetPathLoss(Dataset):
    
    def __init__(
        self, data_path: str, scenario: str, scenario2_path: str,
        split: str, min_num_bs: int,
        path_response_ghz: Literal['2.6GHz', '6GHz', '28GHz', '60GHz', '100GHz'],
        path_loss_mean: float, path_loss_std: float, img_size: int,
        eps_image: int, output_kernel_size: int, empty_features: int, *args, **kwargs
    ):
        super().__init__()
        
        if scenario != "scenario_1":
            raise NotImplementedError(f'There is only an implementation for scenario_1. Given {scenario}')
        
        if split != "train" and split != "val" and split != "test":
            raise NotImplementedError(f'There is only an implementation for train and test. Given {split}')
        
        self.scenario_path: str = os.path.join(data_path, scenario)
        self.scenario2_path = scenario2_path
        self.split: str = split
        self.path_response_ghz = path_response_ghz
        self.output_kernel_size: int = output_kernel_size
        self.img_size = img_size
        self.eps_image = eps_image
        self.path_loss_mean = path_loss_mean
        self.path_loss_std = path_loss_std
        
        self.min_num_bs = min_num_bs
        self.num_bss_per_env = 5
        self.num_ues_per_env = 30
        self.empty_features = empty_features
        
        self.environments: list[str] = self.prepare_environments()
    
    def __getitem__(self, idx: int):
        environment_idx: int = idx // self.num_ues_per_env
        ue_idx = idx % self.num_ues_per_env
        
        environment: str = self.environments[environment_idx]
        
        env_path = os.path.join(self.scenario_path, environment)
        metadata = dict(np.load(os.path.join(env_path, "metadata.npz")))
        
        bs_path_losses = []
        num_bss = random.choice(range(self.min_num_bs, self.num_bss_per_env + 1))
        bss = random.sample(range(self.num_bss_per_env), num_bss)
        
        for bs_idx in bss:
            pair_path = os.path.join(env_path, f"bs{bs_idx}_ue{ue_idx}")
            pair_data_path = os.path.join(pair_path, "data.npz")
            pair_data = np.load(pair_data_path, allow_pickle=True)
            path_response = pair_data["path_responses"][()][self.path_response_ghz]
            magnitude = np.sum(np.abs(path_response))
            path_loss = -10 * np.log10(magnitude ** 2)
            path_loss = (path_loss - self.path_loss_mean) / (self.path_loss_std ** 2)
            bs_location = pair_data["locations"][()]["bs"][:2][::-1] * self.img_size
            bs_path_losses.append((path_loss, bs_location))
        
        # noinspection PyUnboundLocalVariable
        input_img = list(np.load(os.path.join(pair_path, "input_img.npz")).values())[0].astype(np.float32)[0]
        assert input_img.shape[-1] == self.img_size
        
        map_img = -1 * np.expand_dims(input_img, axis=0).repeat(3, axis=0)
        # noinspection PyTypeChecker
        map_img[2] = np.full_like(map_img[2], self.img_size / (metadata["img_size"] / 2))
        
        for path_loss, bs_loc_y_x in bs_path_losses:
            # noinspection DuplicatedCode
            bs_info_slice = (
                slice(max(0, int(bs_loc_y_x[0]) - self.eps_image), int(bs_loc_y_x[0]) + self.eps_image),
                slice(max(0, int(bs_loc_y_x[1]) - self.eps_image), int(bs_loc_y_x[1]) + self.eps_image)
            )
            map_img[0][bs_info_slice[0], bs_info_slice[1]] = bs_loc_y_x[0] / self.img_size
            map_img[1][bs_info_slice[0], bs_info_slice[1]] = bs_loc_y_x[1] / self.img_size
            map_img[2][bs_info_slice[0], bs_info_slice[1]] = path_loss
        
        bs_path_losses = np.vstack(
            [
                np.concatenate((loc / self.img_size, [path_loss], [0.0] * self.empty_features))
                for path_loss, loc in bs_path_losses
            ]
        )
        
        # noinspection PyUnboundLocalVariable
        locations = pair_data["locations"].item()
        ue_location = locations['ue'].astype(np.float32)[:2].astype(np.float32) * max(input_img.shape)
        ue_location = ue_location[::-1]
        
        ue_loc_img = list(np.load(os.path.join(pair_path, "ue_loc_img.npz")).values())[0].astype(np.float32)
        ue_loc_img = resize(ue_loc_img, (1, 224, 224))
        output_kernel_size = self.output_kernel_size
        ue_loc_img = gaussian_filter(ue_loc_img, output_kernel_size)
        ue_loc_img_max = ue_loc_img.max()
        if ue_loc_img_max != 0:
            ue_loc_img /= ue_loc_img_max
        
        # noinspection PyTypeChecker
        return (
            map_img,
            bs_path_losses.astype(np.float32),
            ue_loc_img,
            metadata["img_size"] / 2,
            ue_location
        )
    
    def __len__(self) -> int:
        # return 88
        return len(self.environments) * self.num_ues_per_env
    
    # noinspection PyTypeChecker
    def prepare_environments(self) -> list[str]:
        environments = os.listdir(self.scenario_path)
        environments = sorted(filter(str.isnumeric, environments))
        scenario2_environments = set(filter(str.isnumeric, os.listdir(self.scenario2_path)))
        
        if self.split == "train":
            return sorted(set(environments[:900] + environments[1000: 9499]) - scenario2_environments)
        elif self.split == "val":
            return sorted(set(environments[9499:]) | scenario2_environments)
        else:
            return environments[900:1000]
