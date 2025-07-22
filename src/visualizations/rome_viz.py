import json
import os
from itertools import chain

import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
import streamlit as st
from geopandas import GeoDataFrame
from shapely.geometry import Point


class DatasetVisualizer:
    
    def __init__(self, path: str):
        """
        Initialize the DatasetVisualizer with the path to the dataset.

        :param path: Path to the directory containing dataset JSON files.
        """
        self.path = path
        self.datasets = ["hard_test", "hard_val", "medium_test", "medium_val", "train"]
        self.data = self.create_dataset_from_path(path)
        self.coordinates_data = {split: self.get_coordinates_list(data_info) for split, data_info in self.data.items()}
        self.boundaries_nsew = self.get_boundaries_nsew(self.coordinates_data)
    
    def create_dataset_from_path(self, path: str) -> dict[str, dict[str, dict[str, int]]]:
        """
        Load dataset JSON files from the specified path.

        :param path: Path to the directory containing dataset JSON files.
        :return: A dictionary containing dataset information.
        """
        data = {}
        for dataset in self.datasets:
            with open(os.path.join(path, f"{dataset}.json")) as f:
                data[dataset] = json.load(f)
        return data
    
    @staticmethod
    def get_coordinates_list(data_info: dict[str, dict[str, int]]) -> dict[str, list[tuple[str, str]]]:
        """
        Extract coordinates from dataset information.

        :param data_info: Dataset information for a specific data type (train, val, test).
        :return: A dictionary containing coordinates for each campaign.
        """
        coordinates = {}
        for campaign, campaign_info in data_info.items():
            current_campaign_coordinates = []
            for coordinate in campaign_info:
                coordinates_list = coordinate.split("_", 1)
                y = coordinates_list[0].replace("p", ".")
                x = coordinates_list[1].replace("p", ".")
                current_campaign_coordinates.append((y, x))
            coordinates[campaign] = current_campaign_coordinates
        return coordinates
    
    @staticmethod
    def get_boundaries_nsew(coordinates_data) -> tuple[float, float, float, float]:
        """
        Calculate central coordinates from all dataset coordinates.

        :param coordinates_data: Dictionary containing coordinates for all datasets.
        :return: Tuple containing central x and y coordinates.
        """
        coords = list(
            chain.from_iterable(
                [coord_info for campaign_info in coordinates_data.values() for coord_info in campaign_info.values()]
            )
        )
        xs, ys = zip(*coords)
        xs = list(map(float, xs))
        ys = list(map(float, ys))
        
        return max(xs), min(xs), max(ys), min(ys)
    
    # noinspection PyMethodParameters
    @st.cache_data()
    def get_edges(_self, _selected_campaigns, _datasets) -> GeoDataFrame:
        # noinspection PyPep8Naming
        G = ox.graph_from_bbox(bbox=_self.boundaries_nsew, network_type="drive", simplify=True)
        edges = ox.graph_to_gdfs(G, nodes=False)
        
        return edges
    
    def get_gdfs(self, selected_campaigns, datasets, edges):
        
        gdfs = {}
        for split in datasets:
            if split in self.coordinates_data:
                coords = [
                    Point(lon, lat)
                    for campaign in self.coordinates_data[split]
                    if campaign in selected_campaigns and campaign in self.coordinates_data[split]
                    for lat, lon in self.coordinates_data[split][campaign]
                ]
                if coords:
                    gdf = gpd.GeoDataFrame(geometry=coords, crs="EPSG:4326")
                    gdfs[split] = gdf.to_crs(edges.crs)
        
        return gdfs
    
    @staticmethod
    def plot_rome_dataset(ax, edges: GeoDataFrame, gdfs):
        edges.plot(ax=ax, linewidth=1, edgecolor="gray", zorder=1)
        marker_color_map = {
            "medium_test": ("o", "pink"),
            "medium_val": ("o", "lime"),
            "hard_test": ("o", "firebrick"),
            "hard_val": ("o", "darkgreen"),
            "train": ("o", "blue"),
        }
        
        for split, gdf in gdfs.items():
            gdf.plot(
                ax=ax,
                marker=marker_color_map[split][0],
                color=marker_color_map[split][1],
                markersize=3,
                label=" ".join(split.split("_")).title() + f" ({len(gdf)} UEs)",
                zorder=0,
            )
        
        ax.set_title("Street Map with UE Coordinates")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(bbox_to_anchor=(1, 1), borderaxespad=0)
    
    def __call__(self, *args, **kwargs):
        """
        Run the visualizer with Streamlit interface.
        """
        st.sidebar.header("Select Datasets and Campaigns")
        campaigns = sorted(set(chain.from_iterable(campaign.keys() for campaign in self.data.values())))
        selected_campaigns = st.sidebar.multiselect("Select Campaigns", campaigns, default=campaigns)
        selected_datasets = st.sidebar.multiselect("Select Datasets", self.datasets, default=self.datasets)
        edges = self.get_edges(selected_campaigns, selected_datasets)
        gdfs = self.get_gdfs(selected_campaigns, selected_datasets, edges=edges)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # noinspection PyTypeChecker
        DatasetVisualizer.plot_rome_dataset(ax=ax, edges=edges, gdfs=gdfs)
        st.pyplot(fig, clear_figure=False)
