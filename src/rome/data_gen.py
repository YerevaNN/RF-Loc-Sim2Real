import json
import logging
import os
import sys
import warnings
from random import choice

import geopandas
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
import psutil
from geopy.distance import distance
from omegaconf import DictConfig
from pyproj import CRS
from scipy.io import loadmat
from shapely.geometry import Point
from tqdm import tqdm

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyproj.transformer")
# warnings.filterwarnings("ignore", message="Conversion of an array with ndim > 0 to a scalar is deprecated")
warnings.filterwarnings("ignore")


# TODO only crop here. Radio stuff should be done in info json creator
# TODO or directly create the info json for each dataset type here
class DataGen:
    
    def __init__(
        self,
        oslo: bool,
        mat_file: str,
        cell_file: str,
        sionna_csv: str,
        out_dir: str,
        random_point_scale_factor: float,
        nsew: list[float],
        workers: list[int]
    ):
        self.random_point_scale_factor = random_point_scale_factor
        self.oslo = oslo
        if sionna_csv:
            log.info(f"Reading {sionna_csv}. {mat_file} and {cell_file} are ignored.")
            self.simulated = True
            
            self.info_df = pd.read_csv(sionna_csv)
            self.info_df.rename(
                columns={
                    "ue_lat": "latitude",
                    "ue_lon": "longitude",
                    "bs_lat": "cellLatitude",
                    "bs_lon": "cellLongitude",
                    "sim_rssi": "RSSI",
                    "sim_nsinr": "NSINR",
                    "sim_nrsrp": "NRSRP",
                    "sim_nrsrq": "NRSRQ",
                },
                inplace=True
            )
            self.info_df["eNodeB ID"] = -1
            self.info_df["eNodeBID"] = -1
            self.info_df["MNC"] = -1
            self.info_df["NPCI"] = -1
            self.info_df["interpolated"] = False
            self.info_df["ToA"] = -1
            self.info_df["campaignID"] = 0
        else:
            self.simulated = False
            mat = loadmat(mat_file)
            cell_df = pd.read_excel(cell_file)[[
                "Name", "eNodeID" if oslo else "eNodeBID", "Latitude", "Longitude", "PosErrorDirection",
                "PosErrorLambda1", "PosErrorLambda2", "MNC", "TowerID"
            ]]
            
            self.info_df = self.merge_cell_data(mat, cell_df)
            self.info_df = self.info_df[~self.info_df.isnull().any(axis=1)]
        
        self.ue_positions = np.unique(self.info_df[["latitude", "longitude"]].values, axis=0)
        self.out_dir = out_dir
        north, south, east, west = nsew
        log.info("Getting all the buildings")
        self.buildings = ox.features_from_bbox(bbox=(north, south, east, west), tags={"building": True})
        log.info("Getting all the roads")
        # noinspection PyPep8Naming
        G = ox.graph_from_bbox(
            bbox=(north, south, east, west), network_type="drive", retain_all=True, simplify=False
        )
        self.roads = ox.graph_to_gdfs(G, nodes=False)
        self.workers = workers
        
        total_campaigns = len(self.info_df["campaignID"].unique())
        for i, campaign_id in enumerate(self.info_df["campaignID"].unique()):
            log.info(f"Creating folders for campaign {i+1}/{total_campaigns}")
            campaign_id = int(campaign_id)
            for lat, lon in tqdm(self.info_df[
                self.info_df["campaignID"] == campaign_id
            ][["latitude", "longitude"]].drop_duplicates().values):
                os.makedirs(
                    os.path.join(self.out_dir, str(campaign_id), DataGen.create_folder_name(lat, lon)),
                    exist_ok=True
                )
    
    def mat_to_pd(self, mat: dict):
        # Collect all keys that match any of the keywords in mat_keywords
        keys = ["dataSet"]
        if not self.oslo:
            keys.append("dataSet_interp")
        
        info_keys = [
            "NPCI", "eNode ID" if self.oslo else "eNodeB ID", "RSSI", "NSINR", "NRSRP",
            "NRSRQ", "ToA", "operatorID", "campaignID"
        ]
        
        dct = {}
        item_id = 0
        for dataset_type in keys:
            is_interpolated = "interp" in dataset_type
            for idx in range(len(mat[dataset_type])):
                info_matrix = mat[dataset_type][idx, 2]
                latitude = mat[dataset_type][idx, 0].item()
                longitude = mat[dataset_type][idx, 1].item()
                for info_matrix_i in info_matrix:
                    info_dct = {
                        "latitude": latitude,
                        "longitude": longitude,
                        "interpolated": is_interpolated
                    }
                    for key_idx, key in enumerate(info_keys):
                        info_dct[key] = info_matrix_i[key_idx]
                    dct[item_id] = info_dct
                    item_id += 1
        
        df = pd.DataFrame.from_dict(dct, orient="index")
        df["operatorID"] = df["operatorID"].fillna(0).astype(int)
        df.rename(columns={"operatorID": "MNC"}, inplace=True)
        
        df = df.dropna()
        df = df.sort_values(by="interpolated")
        df = df.groupby(
            ["latitude", "longitude", "eNode ID" if self.oslo else "eNodeB ID", "MNC", "NPCI"]
        ).first().reset_index()
        
        return df
    
    def merge_cell_data(self, mat, cell_df):
        data_df = self.mat_to_pd(mat)
        cell_df["NPCI"] = cell_df["Name"].str.extract(r"NPCI:(\d+)")
        cell_df["NPCI"] = pd.to_numeric(cell_df["NPCI"])
        cell_df.rename(columns={"Latitude": "cellLatitude", "Longitude": "cellLongitude"}, inplace=True)
        
        merged_df = pd.merge(
            data_df,
            cell_df[["eNodeID" if self.oslo else "eNodeBID", "MNC", "NPCI", "cellLatitude", "cellLongitude"]],
            left_on=["eNode ID" if self.oslo else "eNodeB ID", "MNC", "NPCI"],
            right_on=["eNodeID" if self.oslo else "eNodeBID", "MNC", "NPCI"],
            how="left"
        )
        merged_df = merged_df[~merged_df["cellLongitude"].isna()]
        merged_df = merged_df[merged_df["cellLongitude"] != 0]
        
        return merged_df
    
    @staticmethod
    def get_cardinal_pos(lat: float, lon: float, dist: float):
        north_point = distance(meters=float(dist)).destination((lat, lon), bearing=0)
        south_point = distance(meters=float(dist)).destination((lat, lon), bearing=180)
        east_point = distance(meters=float(dist)).destination((lat, lon), bearing=90)
        west_point = distance(meters=float(dist)).destination((lat, lon), bearing=270)
        
        north = float(north_point[0])
        south = float(south_point[0])
        east = float(east_point[1])
        west = float(west_point[1])
        return north, south, east, west
    
    @staticmethod
    def random_point(random_ue_lat, random_ue_lon, half_square_size_meters, scale_factor):
        north, south, east, west = DataGen.get_cardinal_pos(
            random_ue_lat, random_ue_lon, (scale_factor * half_square_size_meters)
        )
        center_lat = np.random.rand() * (north - south) + south
        center_lon = np.random.rand() * (east - west) + west
        return center_lat, center_lon
    
    @staticmethod
    def create_custom_tm_crs(latitude, longitude):
        proj_params = {
            "proj": "tmerc",  # Transverse Mercator
            "lat_0": latitude,
            "lon_0": longitude,
            "k": 1,
            "x_0": 0,
            "y_0": 0,
            "datum": "WGS84",
            "units": "m"
        }
        
        custom_crs = CRS(proj_params)
        
        return custom_crs
    
    @staticmethod
    def create_folder_name(latitude, longitude):
        lat_str = "{:.6f}".format(latitude)
        lon_str = "{:.6f}".format(longitude)
        
        folder_name = f"{lat_str}_{lon_str}"
        folder_name = folder_name.replace(".", "p")
        
        return folder_name
    
    def get_corresponding_cells(self, lat, lon, south, north, west, east):
        return self.info_df[(
            (self.info_df["latitude"] == float(lat)) & (self.info_df["longitude"] == float(lon)) &
            (self.info_df["cellLatitude"] >= float(south)) & (self.info_df["cellLatitude"] <= float(north)) &
            (self.info_df["cellLongitude"] >= float(west)) & (self.info_df["cellLongitude"] <= float(east))
        )]
    
    def generate_plot(self, center_lat, center_lon, half_square_size, custom_crs):
        dpi = 128
        figsize = ((2 * half_square_size + 1) / dpi, (2 * half_square_size + 1) / dpi)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        north, south, east, west = self.get_cardinal_pos(float(center_lat), float(center_lon), float(half_square_size))
        bbox = (float(west), float(south), float(east), float(north))
        # noinspection PyPep8Naming
        buildings_gdf = geopandas.clip(self.buildings, bbox)
        roads_gdf = geopandas.clip(self.roads, bbox)
        buildings_gdf.to_crs(custom_crs).plot(ax=ax, color="red")
        roads_gdf.to_crs(custom_crs).plot(ax=ax, color="green")
        
        ax.set_xlim(-half_square_size, half_square_size)
        ax.set_ylim(-half_square_size, half_square_size)
        ax.axis("off")
        fig.tight_layout(pad=0)
        
        return fig, ax
    
    def project_ue_and_cells(self, custom_crs, half_square_size, ue_lat_lon, cells_df) -> dict:
        info_dict = {
            "UE": {},
            "BaseStations": []
        }
        
        # Project UE
        ue_point = Point(float(ue_lat_lon[1]), float(ue_lat_lon[0]))
        # noinspection PyUnresolvedReferences
        ue_projected = ox.projection.project_geometry(
            ue_point, crs="epsg:4326", to_crs=custom_crs
        )[0]
        info_dict["UE"]["lat_lon"] = tuple(map(float, ue_lat_lon))
        info_dict["UE"]["proj_map_pos"] = (
            float(half_square_size - ue_projected.y),
            float(half_square_size + ue_projected.x)
        )
        
        log.debug(f"cells_df columns: {cells_df.columns}")
        log.debug(f"cells_df head:\n{cells_df.head()}")
        
        for _, cell in cells_df.iterrows():
            cell_lat, cell_lon = float(cell["cellLatitude"]), float(cell["cellLongitude"])
            cell_point = Point(cell_lon, cell_lat)
            # noinspection PyUnresolvedReferences
            cell_projected = ox.projection.project_geometry(
                cell_point, crs="epsg:4326", to_crs=custom_crs
            )[0]
            
            measurements = cell[["RSSI", "NSINR", "NRSRP", "NRSRQ", "ToA"]].to_dict()
            interpolated_flag = cell["interpolated"]
            proj_y = float(half_square_size - cell_projected.y)
            proj_x = float(half_square_size + cell_projected.x)
            info_dict["BaseStations"].append(
                {
                    "npci": float(cell["NPCI"]),
                    "enode": float(cell["eNodeID" if self.oslo else "eNodeBID"]),
                    "mnc": float(cell["MNC"]),
                    "lat_lon": (cell_lat, cell_lon),
                    "proj_map_pos": (proj_y, proj_x),
                    "measurements": measurements,
                    "interpolated": interpolated_flag
                }
            )
        
        return info_dict
    
    def save_visualization(self, fig, info_dict, campaign_id, ue_lat, ue_lon, i):
        """Save the plot and info dictionary as PNG and JSON files in a unique directory."""
        output_dir = os.path.join(self.out_dir, str(campaign_id), DataGen.create_folder_name(ue_lat, ue_lon))
        
        npz_path = os.path.join(output_dir, f"{i}.npz")
        info_path = os.path.join(output_dir, f"{i}.json")
        
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        image_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        image_array = image_array.reshape(height, width, 4)[:, :, :3]
        rg_array = image_array[:, :, :2]
        
        np.savez_compressed(npz_path, image=rg_array)
        
        # Save the info dictionary
        with open(info_path, "w") as f:
            # noinspection PyTypeChecker
            json.dump(info_dict, f, indent=4)
        plt.close(fig)
    
    def gen_ran_ue_with_bs(self, args: tuple[int, int, bool, int]):
        i, half_square_size_meters, to_use_sample_per_ue, ue_j = args
        
        if to_use_sample_per_ue:
            random_ue_lat, random_ue_lon = self.ue_positions[ue_j]
        else:
            random_ue_lat, random_ue_lon = choice(self.ue_positions)
        center_lat, center_lon = DataGen.random_point(
            random_ue_lat, random_ue_lon, half_square_size_meters, self.random_point_scale_factor
        )
        north, south, east, west = DataGen.get_cardinal_pos(center_lat, center_lon, half_square_size_meters)
        
        assert south <= random_ue_lat <= north and west <= random_ue_lon <= east
        
        corresponding_cells: pd.DataFrame = self.get_corresponding_cells(
            random_ue_lat, random_ue_lon, south, north, west, east
        )
        if corresponding_cells.empty:
            return
        
        campaign_id = int(corresponding_cells["campaignID"].values[0])
        
        if self.simulated:
            id_columns = ["bs_idx"]
            other_columns = ["NPCI", "eNodeBID", "MNC", "cellLatitude", "cellLongitude", "interpolated"]
        else:
            id_columns = [
                "NPCI", "eNodeID" if self.oslo else "eNodeBID", "MNC", "cellLatitude", "cellLongitude", "interpolated"
            ]
            other_columns = []
        measurement_columns = ["RSSI", "NSINR", "NRSRP", "NRSRQ", "ToA"]
        unique_cells = corresponding_cells[id_columns + measurement_columns + other_columns].drop_duplicates(id_columns)
        
        custom_crs = DataGen.create_custom_tm_crs(center_lat, center_lon)
        try:
            fig, _ = self.generate_plot(
                center_lat, center_lon, half_square_size_meters, custom_crs
            )
        except Exception as ex:
            log.error(ex)
            self.gen_ran_ue_with_bs((i, half_square_size_meters))
            return
        
        info_dict = {
            "campaign_id": campaign_id,
            "center_coord": (center_lat, center_lon),
            "half_square_size_meters": half_square_size_meters
        }
        info_dict.update(
            self.project_ue_and_cells(
                custom_crs, half_square_size_meters, (random_ue_lat, random_ue_lon), unique_cells
            )
        )
        
        self.save_visualization(fig, info_dict, campaign_id, random_ue_lat, random_ue_lon, i)
    
    @staticmethod
    def generate_random_code(mean: float, std: float, min_value: float, max_value: float) -> int:
        while True:
            random_code = np.random.normal(mean, std)
            if min_value <= random_code <= max_value:
                return int(random_code)
    
    def save_cell_info(self):
        cell_info_path = os.path.join(self.out_dir, f"cell_info.csv")
        self.info_df.to_csv(cell_info_path, index=False)


def generate_data(config: DictConfig) -> None:
    assert config["num_points"] is None or config["sample_per_ue"] is None, "num_points and sample_per_ue should not be set at the same time"
    assert not (config["num_points"] is None and config["sample_per_ue"] is None), "num_points or sample_per_ue should be set to None"
    
    # noinspection PyUnresolvedReferences
    ox.settings.max_query_area_size = 2 ** 64
    log.info("Initializing data generation")
    logging.basicConfig(level=logging.INFO)
    # log.info("CPUs: " + config["workers"])
    parent = psutil.Process()
    # workers = parent.cpu_affinity()
    workers = [0]
    log.info(f"Process {parent.pid} uses {workers} CPUs")
    
    plt.style.use("dark_background")
    
    data = DataGen(
        oslo=config["name"].startswith("oslo"),
        mat_file=config["mat_file"],
        cell_file=config["cell_file"],
        sionna_csv=config["sionna_csv"],
        out_dir=config["out_dir"],
        random_point_scale_factor=config["random_point_scale_factor"],
        nsew=config["nsew"],
        workers=workers
    )
    log.info("Saving cell_info CSV")
    data.save_cell_info()
    
    log.info("Generating random sizes")
    
    if config["num_points"] is not None:
        num_points = config["num_points"]
    else:
        num_points = config["sample_per_ue"] * len(data.ue_positions)
    
    random_sizes = [
        DataGen.generate_random_code(
            mean=config["random_size_mean"], std=config["random_size_std"],
            min_value=config["random_size_min"], max_value=config["random_size_max"]
        )
        for _ in tqdm(range(num_points), total=num_points, file=sys.stdout)
    ]
    
    log.info("Generating data")
    
    to_use_sample_per_ue = config["sample_per_ue"] is not None
    for i in tqdm(range(num_points), total=num_points, file=sys.stdout):
        ue_j = i // config["sample_per_ue"] if to_use_sample_per_ue else -1
        data.gen_ran_ue_with_bs((i, random_sizes[i], to_use_sample_per_ue, ue_j))
