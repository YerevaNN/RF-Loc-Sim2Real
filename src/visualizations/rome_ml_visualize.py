import os

import folium
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from geopy.distance import geodesic


class RomeMLVisualizer:
    
    def __init__(self, pred_dir, subdirectories, default_num_points=100):
        self.pred_dir = pred_dir
        self.subdirectories = subdirectories
        self.default_num_points = default_num_points
    
    @staticmethod
    def load_data(pred_dir, subdirectories, csv_filename=None):
        """
        Args:
        - pred_dir 
        - subdirectories: List of subdirectory names.
        Returns:
        - df with reconstructed coordinates.
        """
        
        if "knn" in pred_dir:
            if csv_filename:
                file_path = os.path.join(pred_dir, csv_filename)
                if not os.path.isfile(file_path):
                    st.error(f"CSV file '{csv_filename}' does not exist in '{pred_dir}'.")
                    return pd.DataFrame()
                # noinspection PyTypeChecker
                csv_df = pd.read_csv(file_path)
                csv_df['Center_Lat'] = np.nan
                csv_df['Center_Lon'] = np.nan
                csv_df['X'] = np.nan
                csv_df['Y'] = np.nan
                df = csv_df[[
                    'Filename',
                    'Center_Lat',
                    'Center_Lon',
                    'X',
                    'Y',
                    'Original_UE_Lat',
                    'Original_UE_Lon',
                    'Reconstructed_Lat',
                    'Reconstructed_Lon'
                ]]
                return df
            else:
                st.error("Please specify a CSV filename.")
                return pd.DataFrame()
        else:
            subdirectories = list(subdirectories)
            data = []
            for subdir in subdirectories:
                directory = os.path.join(pred_dir, str(subdir))
                if not os.path.isdir(directory):
                    st.warning(f"Subdirectory '{subdir}' does not exist in '{pred_dir}'. Skipping.")
                    continue
                for filename in os.listdir(directory):
                    if filename.endswith(".npz"):
                        file_path = os.path.join(directory, filename)
                        npz_data = np.load(file_path)
                        array = npz_data["out"]
                        argmax_index = np.argmax(array)
                        y, x = np.unravel_index(argmax_index, array.shape)
                        center_lat = npz_data["center_lat"].item()
                        center_lon = npz_data["center_lon"].item()
                        original_ue_lat = npz_data["original_ue_lat"].item()
                        original_ue_lon = npz_data["original_ue_lon"].item()
                        original_img_size = npz_data["original_img_size"].item()
                        pixel_size = original_img_size / array.shape[0]
                        x_distance = (x - array.shape[1] // 2) * pixel_size
                        y_distance = (array.shape[0] // 2 - y) * pixel_size
                        reconstructed_lat = geodesic(
                            meters=y_distance
                        ).destination((center_lat, center_lon), 0).latitude
                        reconstructed_lon = geodesic(
                            meters=x_distance
                        ).destination((center_lat, center_lon), 90).longitude
                        data.append(
                            [
                                filename,
                                center_lat,
                                center_lon,
                                x,
                                y,
                                original_ue_lat,
                                original_ue_lon,
                                reconstructed_lat,
                                reconstructed_lon
                            ]
                        )
            if data:
                df = pd.DataFrame(
                    data, columns=[
                        "Filename",
                        "Center_Lat",
                        "Center_Lon",
                        "X",
                        "Y",
                        "Original_UE_Lat",
                        "Original_UE_Lon",
                        "Reconstructed_Lat",
                        "Reconstructed_Lon"
                    ]
                )
                return df
            else:
                return pd.DataFrame()
    
    @staticmethod
    def visualize_map(sampled_df):
        if sampled_df.empty:
            st.error("No data to visualize.")
            return
        
        all_coords = (
            list(sampled_df[["Original_UE_Lat", "Original_UE_Lon"]].values) +
            list(sampled_df[["Reconstructed_Lat", "Reconstructed_Lon"]].values)
        )
        avg_lat = sum(lat for lat, _ in all_coords) / len(all_coords)
        avg_long = sum(lon for _, lon in all_coords) / len(all_coords)
        zoom_level = 18 if len(sampled_df) == 1 else 13
        rome_map = folium.Map(location=[avg_lat, avg_long], zoom_start=zoom_level, max_zoom=20)
        plotted_ue_coords = set()
        for _, row in sampled_df.iterrows():
            original_coords = (row["Original_UE_Lat"], row["Original_UE_Lon"])
            reconstructed_coords = (row["Reconstructed_Lat"], row["Reconstructed_Lon"])
            if original_coords not in plotted_ue_coords:
                folium.CircleMarker(
                    location=original_coords,
                    radius=6,
                    color="green",
                    fill=True,
                    fill_color="green",
                    fill_opacity=1,
                    tooltip=f"Original UE: {original_coords}"
                ).add_to(rome_map)
                plotted_ue_coords.add(original_coords)
            folium.CircleMarker(
                location=reconstructed_coords,
                radius=6,
                color="blue",
                fill=True,
                fill_color="blue",
                fill_opacity=1,
                tooltip=f"Reconstructed: {reconstructed_coords}"
            ).add_to(rome_map)
            
            distance_meters = geodesic(original_coords, reconstructed_coords).meters
            folium.PolyLine(
                [original_coords, reconstructed_coords],
                color="red",
                weight=2.5,
                opacity=1,
                tooltip=f"Distance: {distance_meters:.2f} meters"
            ).add_to(rome_map)
        
        # Render the Folium map to HTML
        # noinspection PyProtectedMember
        map_html = rome_map._repr_html_()
        # Display the map using st.components.v1.html
        components.html(map_html, height=1000)
    
    def __call__(self):
        # Allow the user to change pred_dir and subdirectories from UI
        self.pred_dir = st.text_input("Enter the prediction directory:", self.pred_dir)
        if "knn" in self.pred_dir:
            csv_files = [f for f in os.listdir(self.pred_dir) if f.endswith('.csv')]
            if not csv_files:
                st.error("No CSV files found in the specified directory.")
                return
            csv_filename = st.selectbox("Select CSV file to visualize:", csv_files)
            df = self.load_data(self.pred_dir, self.subdirectories, csv_filename)
        else:
            subdirs_input = st.text_input(
                "Enter subdirectories (comma-separated):", ",".join(map(str, self.subdirectories))
            )
            self.subdirectories = [s.strip() for s in subdirs_input.split(",") if s.strip()]
            df = self.load_data(self.pred_dir, self.subdirectories)
        if df.empty:
            st.error("No data found to visualize.")
            return
        filename_input = st.text_input("Enter Filename to visualize (e.g., '0.npz'):", value="")
        if filename_input.strip():
            sampled_df = df[df["Filename"] == filename_input.strip()]
            if sampled_df.empty:
                st.error(f"No data found for Filename '{filename_input.strip()}'.")
                return
        else:
            # Existing logic if no filename is provided
            num_points = st.slider(
                "Number of points to visualize:",
                min_value=1,
                max_value=len(df),
                value=min(self.default_num_points, len(df))
            )
            omit_repeated_ue = st.checkbox("Omit repeated UE locations", value=True)
            sampled_df = df.sample(n=min(num_points, len(df)))
            if omit_repeated_ue:
                sampled_df = sampled_df.drop_duplicates(subset=["Original_UE_Lat", "Original_UE_Lon"])
        
        self.visualize_map(sampled_df)
