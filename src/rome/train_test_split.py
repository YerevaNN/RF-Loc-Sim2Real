import json
import logging
import math
import os
from collections import defaultdict
from copy import deepcopy

from geopy.distance import distance
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)


class DataSplit:
    
    def __init__(
        self,
        json_path: str,
        data_out_dir: str,
        train_only: bool,
        hard_test_nsew: str,
        hard_train: float,
        hard_test: float,
        medium_train: float,
        medium_test: float,
        easy_train: float,
        easy_test: float,
        hard_val_nsew: list,
        hard_margin_meters: float,
        drop_overlapping_bboxes: bool,
        overlap_precheck_half_square_size_meters: float,
        random_state: int = 42,
    ):
        with open(json_path, "r") as file:
            # info_json = {
            #     campaign_id: {
            #         point_id: {
            #             sample_id: some_number
            #         }
            #     }
            # }
            # Now this info_json is either info_dataSet or info_dataSet_interp with corresponding infos stored for interpolated and non-interpolated data
            self.info_json: dict[str, dict[str, dict[str, int]]] = json.load(file)
        
        self.data_out_dir = data_out_dir
        self.train_only = train_only
        self.hard_train = hard_train
        self.hard_test = hard_test
        self.medium_train = medium_train
        self.medium_test = medium_test
        self.easy_train = easy_train
        self.easy_test = easy_test
        
        self.hard_test_nsew = hard_test_nsew
        self.hard_val_nsew = hard_val_nsew
        self.hard_margin_meters = hard_margin_meters
        
        self.drop_overlapping_bboxes = drop_overlapping_bboxes
        self.overlap_precheck_buffer_meters = 2 * overlap_precheck_half_square_size_meters * math.sqrt(2)
        self.random_state = random_state
    
    def split_dict(self, data: dict, test_size: float = 0.3):
        assert 0.0 <= test_size <= 1.0
        keys = list(data.keys())
        # keys = [str(key) for key in keys]
        if test_size == 0.0:
            train_split = deepcopy(data)
            test_split = {}
        elif test_size == 1.0:
            test_split = deepcopy(data)
            train_split = {}
        else:
            if len(keys) > 1:
                train_keys, test_keys = train_test_split(
                    keys, test_size=test_size, random_state=self.random_state
                )
                # train_keys = [str(k) for k in train_keys]
                # test_keys = [str(k) for k in test_keys]
                train_split = {key: data[key] for key in train_keys}
                test_split = {key: data[key] for key in test_keys}
            else:
                # If there's only one key, we can't split it, so we return it as train and leave test empty
                train_split = data
                test_split = {}
        
        return train_split, test_split
    
    def apply_margin_to_nsew(self, nsew):
        """
        Apply margin to nsew coordinates by moving boundaries inward by margin_meters.
        
        Args:
            nsew: [north, south, east, west] coordinates in degrees
        
        Returns:
            [north, south, east, west] with margins applied
        """
        north, south, east, west = nsew
        
        # Convert meters to degrees (approximate)
        # 1 degree of latitude ≈ 111,000 meters
        # 1 degree of longitude ≈ 111,000 * cos(latitude) meters
        lat_center = (north + south) / 2
        meters_per_degree_lat = 111000
        meters_per_degree_lon = 111000 * math.cos(math.radians(lat_center))
        
        # Convert margin from meters to degrees
        margin_degrees_lat = self.hard_margin_meters / meters_per_degree_lat
        margin_degrees_lon = self.hard_margin_meters / meters_per_degree_lon
        
        # Apply margin inward
        north_margined = north - margin_degrees_lat
        south_margined = south + margin_degrees_lat
        east_margined = east - margin_degrees_lon
        west_margined = west + margin_degrees_lon
        
        return [north_margined, south_margined, east_margined, west_margined]

    @staticmethod
    def expand_nsew_by_meters(nsew, buffer_meters):
        north, south, east, west = nsew
        lat_center = (north + south) / 2
        meters_per_degree_lat = 111000
        meters_per_degree_lon = 111000 * math.cos(math.radians(lat_center))

        buffer_degrees_lat = buffer_meters / meters_per_degree_lat
        buffer_degrees_lon = buffer_meters / meters_per_degree_lon

        return [
            north + buffer_degrees_lat,
            south - buffer_degrees_lat,
            east + buffer_degrees_lon,
            west - buffer_degrees_lon,
        ]

    @staticmethod
    def point_in_nsew(lat, lon, nsew):
        north, south, east, west = nsew
        return south <= lat <= north and west <= lon <= east

    @staticmethod
    def bboxes_intersect(a_nsew, b_nsew) -> bool:
        a_north, a_south, a_east, a_west = a_nsew
        b_north, b_south, b_east, b_west = b_nsew
        return not (
            a_south > b_north
            or a_north < b_south
            or a_west > b_east
            or a_east < b_west
        )

    @staticmethod
    def parse_pid_lat_lon(pid: str):
        lat_str, lon_str = pid.split('_')
        lat_str = lat_str.replace('p', '.')
        lon_str = lon_str.replace('p', '.')
        return float(lat_str), float(lon_str)

    @staticmethod
    def crop_bbox_from_json(crop_json_path: str):
        with open(crop_json_path, "r") as file:
            crop_info = json.load(file)
        center_lat, center_lon = crop_info["center_coord"]
        half_square_size_meters = crop_info["half_square_size_meters"]
        north_point = distance(meters=float(half_square_size_meters)).destination((center_lat, center_lon), bearing=0)
        south_point = distance(meters=float(half_square_size_meters)).destination((center_lat, center_lon), bearing=180)
        east_point = distance(meters=float(half_square_size_meters)).destination((center_lat, center_lon), bearing=90)
        west_point = distance(meters=float(half_square_size_meters)).destination((center_lat, center_lon), bearing=270)
        return [float(north_point[0]), float(south_point[0]), float(east_point[1]), float(west_point[1])]

    def drop_train_overlaps(self, train_info: dict) -> dict:
        if not self.drop_overlapping_bboxes:
            return train_info

        expanded_test = self.expand_nsew_by_meters(self.hard_test_nsew, self.overlap_precheck_buffer_meters)
        expanded_val = self.expand_nsew_by_meters(self.hard_val_nsew, self.overlap_precheck_buffer_meters)

        removed_crops = 0
        removed_ues = 0

        for cid in list(train_info.keys()):
            for pid in list(train_info[cid].keys()):
                try:
                    lat, lon = self.parse_pid_lat_lon(pid)
                except ValueError:
                    log.info(f"Invalid lat/lon format in pid: {pid}. Skipping.")
                    continue

                if not (
                    self.point_in_nsew(lat, lon, expanded_test)
                    or self.point_in_nsew(lat, lon, expanded_val)
                ):
                    continue

                for crop_id in list(train_info[cid][pid].keys()):
                    crop_json_path = os.path.join(self.data_out_dir, str(cid), pid, f"{crop_id}.json")
                    if not os.path.exists(crop_json_path):
                        log.info(f"Missing crop json: {crop_json_path}. Skipping overlap check.")
                        continue
                    crop_nsew = self.crop_bbox_from_json(crop_json_path)
                    if self.bboxes_intersect(crop_nsew, self.hard_test_nsew) or self.bboxes_intersect(
                        crop_nsew, self.hard_val_nsew
                    ):
                        del train_info[cid][pid][crop_id]
                        removed_crops += 1

                if not train_info[cid][pid]:
                    del train_info[cid][pid]
                    removed_ues += 1

            if cid in train_info and not train_info[cid]:
                del train_info[cid]

        log.info(f"Removed {removed_crops} overlapping crops from train.json.")
        log.info(f"Removed {removed_ues} UEs with no remaining crops from train.json.")
        return train_info
    
    # def split_hard_test(self) -> tuple[dict, dict]:
    #     campaign_info = self.info_json[self.hard_test_nsew]
    #     train_info = deepcopy(self.info_json)
    #     train_info.pop(self.hard_test_nsew)
    #     hard_test_info = {self.hard_test_nsew: campaign_info}
    #
    #     return train_info, hard_test_info
    
    def split_hard(self, train_info: dict, nsew) -> tuple[dict, dict]:
        hard_val_info = {}
        
        # Apply margin to nsew coordinates
        nsew_with_margin = nsew  # self.apply_margin_to_nsew(nsew)
        north, south, east, west = nsew
        north_m, south_m, east_m, west_m = nsew_with_margin
        
        for cid in list(train_info.keys()):
            for pid in list(train_info[cid].keys()):
                samples = train_info[cid][pid]
                # Parse UE lat/lon from pid
                try:
                    lat_str, lon_str = pid.split('_')
                    lat_str = lat_str.replace('p', '.')
                    lon_str = lon_str.replace('p', '.')
                    lat = float(lat_str)
                    lon = float(lon_str)
                except ValueError:
                    log.info(f"Invalid lat/lon format in pid: {pid}. Skipping.")
                    continue
                
                if south_m <= lat <= north_m and west_m <= lon <= east_m:
                    
                    if cid not in hard_val_info:
                        hard_val_info[cid] = {}
                    hard_val_info[cid][pid] = samples
                
                if south <= lat <= north and west <= lon <= east:
                    del train_info[cid][pid]
                
                if cid in train_info and not train_info[cid]:
                    del train_info[cid]
                    log.info(f"All points in campaign {cid} have been moved to validation.")
        
        return train_info, hard_val_info
    
    def split_medium_test(self, train_info_: dict) -> tuple[dict, dict, dict]:
        train_info, test_info, val_info = {}, {}, {}
        for cid, points in train_info_.items():
            train_points, temp_points = self.split_dict(
                points,
                test_size=1 - self.medium_train,
            )
            test_points, val_points = self.split_dict(
                temp_points,
                test_size=self.medium_test,
            )
            train_info[cid] = train_points
            test_info[cid] = test_points
            val_info[cid] = val_points
        
        return train_info, test_info, val_info
    
    def split_easy_test(self, train_info_: dict[str, dict]) -> tuple[dict, dict, dict]:
        train_info, test_info, val_info = defaultdict(dict), defaultdict(dict), defaultdict(dict)
        for cid, points in train_info_.items():
            for pid, samples in points.items():
                train_samples, temp_samples = self.split_dict(
                    samples,
                    test_size=1 - self.easy_train,
                )
                test_samples, val_samples = self.split_dict(
                    temp_samples, test_size=self.easy_test
                )
                train_info[cid][pid] = train_samples
                test_info[cid][pid] = test_samples
                val_info[cid][pid] = val_samples
        
        return dict(train_info), dict(test_info), dict(val_info)
    
    # noinspection PyTypeChecker
    def save_train_test_val(self, out_dir: str):
        log.info("Starting data splitting...")
        if self.train_only:
            train_info = self.info_json
            hard_test_info, hard_val_info, medium_test_info, medium_val_info, easy_test_info, easy_val_info = ({},) * 6
        else:
            train_info, hard_test_info = self.split_hard(self.info_json, self.hard_test_nsew)
            train_info, hard_val_info = self.split_hard(train_info, self.hard_val_nsew)
            train_info, medium_test_info, medium_val_info = self.split_medium_test(train_info)
            train_info, easy_test_info, easy_val_info = self.split_easy_test(train_info)

        train_info = self.drop_train_overlaps(train_info)
        
        with open(os.path.join(out_dir, "train.json"), "w") as file:
            json.dump(train_info, file, indent=4)
        
        with open(os.path.join(out_dir, "hard_test.json"), "w") as file:
            json.dump(hard_test_info, file, indent=4)
        
        with open(os.path.join(out_dir, "hard_val.json"), "w") as file:
            json.dump(hard_val_info, file, indent=4)
        
        with open(os.path.join(out_dir, "medium_test.json"), "w") as file:
            json.dump(medium_test_info, file, indent=4)
        
        with open(os.path.join(out_dir, "medium_val.json"), "w") as file:
            json.dump(medium_val_info, file, indent=4)
        
        with open(os.path.join(out_dir, "easy_test.json"), "w") as file:
            json.dump(easy_test_info, file, indent=4)
        
        with open(os.path.join(out_dir, "easy_val.json"), "w") as file:
            json.dump(easy_val_info, file, indent=4)
        
        log.info(f"Hard Test Campaign ({self.hard_test_nsew}) has {sum(map(len, hard_test_info.values()))} points.")
        log.info(f"Medium Test has {sum(map(len, medium_test_info.values()))} points.")
        log.info(f"Hard Val has {sum(map(len, hard_val_info.values()))} points.")
        log.info(f"Medium Val has {sum(map(len, medium_val_info.values()))} points.")
        log.info(f"Train has {sum(map(len, train_info.values()))} points.")


def train_test_val(config: DictConfig) -> None:
    train_test_splitter = DataSplit(
        json_path=os.path.join(config["out_dir"], f"info_{config.get('dataset_type', 'dataSet')}.json"),
        data_out_dir=config["out_dir"],
        train_only=config["train_only"],
        hard_test_nsew=config["hard_test_nsew"],
        hard_train=config["hard_train"],
        hard_test=config["hard_test"],
        medium_train=config["medium_train"],
        medium_test=config["medium_test"],
        easy_train=config["easy_train"],
        easy_test=config["easy_test"],
        hard_val_nsew=config["hard_val_nsew"],
        hard_margin_meters=config["hard_margin_meters"],
        drop_overlapping_bboxes=config.get("drop_overlapping_bboxes", True),
        overlap_precheck_half_square_size_meters=config.get("overlap_precheck_half_square_size_meters", 0),
        random_state=config["seed"],
    )
    
    train_test_splitter.save_train_test_val("./")
