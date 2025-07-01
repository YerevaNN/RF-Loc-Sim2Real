import json
import os
from collections import defaultdict
from glob import glob

from omegaconf import DictConfig
from tqdm import tqdm


# TODO for each dataset type create a new json file with filtered BSs
class InfoJSONCreator:
    
    def __init__(self, main_path: str, dataset_type: str):
        self.main_path = main_path
        self.dataset_type = dataset_type
        self.decimal_places = 10
        self.info_dict = defaultdict(lambda: defaultdict(dict))
    
    @staticmethod
    def get_subpath_from_point(path, start=-3):
        parts = [part for part in path.split('/') if part]
        sub_path = '/'.join(parts[start:])
        
        return sub_path
    
    def get_num_bs(self, json_path):
        """
        Get number of base stations for a specific UE location
        Returns total BS count for dataSet_interp, only non interpolated BS count for dataSet
        """
        with open(json_path, "r") as file:
            map_info = json.load(file)
        base_stations_info = map_info["BaseStations"]
        if self.dataset_type == "dataSet":
            # exclude BS
            num_bs = sum(1 for bs in base_stations_info if not bs.get("interpolated", False))
        else:
            # include BS
            num_bs = len(base_stations_info)
        return num_bs
    
    def gen_point(self, json_path):
        """Generate point information for a specific JSON file"""
        campaign_id, point, sample = InfoJSONCreator.get_subpath_from_point(json_path, start=-3).split("/")
        sample = sample.split(".")[0]
        self.info_dict[campaign_id][point][sample] = self.get_num_bs(json_path)
    
    def gen_json(self):
        """Generate the final info JSON file"""
        all_json_files = sorted(glob(os.path.join(self.main_path, "**/*.json"), recursive=True))
        
        for json_path in tqdm(all_json_files):
            if os.path.basename(json_path) == "info.json":
                continue
            try:
                self.gen_point(json_path)
            except Exception as ex:
                print(f"Error in {json_path}: {ex}")
        
        with open(os.path.join(self.main_path, f"info_{self.dataset_type}.json"), "w") as file:
            # noinspection PyTypeChecker
            json.dump(dict(self.info_dict), file, indent=4)


def create_info_json(config: DictConfig) -> None:
    info_json_creator = InfoJSONCreator(config["main_path"], config["dataset_type"])
    info_json_creator.gen_json()
