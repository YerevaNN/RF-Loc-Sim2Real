import os

import streamlit as st
from omegaconf import DictConfig

from src.visualizations import DatasetVisualizer, RomeMLVisualizer


def rome_visualize(config: DictConfig) -> None:
    if config["name"] == "rome_visualize" or config["name"] == "oslo_visualize":
        st.title("Street Map with Coordinates Visualization")
        path = st.text_input("Enter the path to dataset:", config["splits_path"])
        if path:
            visualizer = DatasetVisualizer(path)
            visualizer()
    elif config["name"] == "rome_ml_visualize":
        st.set_page_config(layout="wide")
        st.title("Global Map Visualization of Predicted Coordinates")
        pred_dir = config.get("pred_dir", "")
        subdirs = config.get("subdirs", [])
        default_num_points = config.get("default_num_points", 100)
        st.write("Please enter the prediction directory and subdirectories.")
        if pred_dir and os.path.isdir(pred_dir):
            visualizer = RomeMLVisualizer(pred_dir, subdirs, default_num_points)
            visualizer()
        else:
            st.warning("Please provide a valid 'pred_dir' in your configuration.")
    else:
        st.error(f"Unknown visualization type: {config['name']}")
