import pandas as pd
import numpy as np
import osmnx as ox
import logging
import os
import math
from typing import Dict, Tuple, List, Optional
from shapely.geometry import Point, Polygon
from pyproj import Transformer
import xml.etree.ElementTree as ET
from xml.dom import minidom
from plyfile import PlyData, PlyElement

logger = logging.getLogger(__name__)


class DataPipeline:
    
    def __init__(self, rome_csv_path: str = None):
        if rome_csv_path is None:
            home_dir = os.path.expanduser("~")
            rome_csv_path = os.path.join(home_dir, "wair_d_iot", "jups", "bs_ue_rome.csv")
        
        self.rome_csv_path = rome_csv_path
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
        
        logger.info(f"DataPipeline initialized with Rome CSV: {rome_csv_path}")
    
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000
        return c * r
    
    def calculate_region_center_and_distance(self, bounds: Dict[str, Tuple[float, float]]) -> Tuple[Tuple[float, float], float]:
        center_lat = (bounds['lat_range'][0] + bounds['lat_range'][1]) / 2
        center_lon = (bounds['lon_range'][0] + bounds['lon_range'][1]) / 2
        
        corner_distances = [
            self.haversine_distance(center_lat, center_lon, bounds['lat_range'][0], bounds['lon_range'][0]),
            self.haversine_distance(center_lat, center_lon, bounds['lat_range'][0], bounds['lon_range'][1]),
            self.haversine_distance(center_lat, center_lon, bounds['lat_range'][1], bounds['lon_range'][0]),
            self.haversine_distance(center_lat, center_lon, bounds['lat_range'][1], bounds['lon_range'][1])
        ]
        distance = max(corner_distances)
        
        return (center_lat, center_lon), distance
    
    def extract_osm_buildings(self, bounds: Dict[str, Tuple[float, float]]) -> Tuple[List[Dict], Tuple[float, float]]:
        center_point, distance = self.calculate_region_center_and_distance(bounds)
        center_lat, center_lon = center_point
        
        logger.info(f"Extracting OSM data for center ({center_lat:.6f}, {center_lon:.6f}) with {distance:.2f}m radius")
        
        try:
            ox.settings.timeout = 300
            ox.settings.use_cache = True
            
            buildings = ox.features.features_from_point(
                center_point, {'building': True}, dist=distance)

            # Convert buildings to the correct CRS if needed
            if buildings.crs != 'EPSG:4326':
                buildings = buildings.to_crs('EPSG:4326')

            logger.info(f"Number of buildings extracted: {len(buildings)}")
            
        except Exception as e:
            logger.warning(f"Point-based OSM extraction failed: {e}")
            logger.info("Trying bounding box approach...")
            
            try:
                margin = 0.001
                north = bounds['lat_range'][1] + margin
                south = bounds['lat_range'][0] - margin
                east = bounds['lon_range'][1] + margin
                west = bounds['lon_range'][0] - margin
                
                buildings = ox.features.features_from_bbox(
                    north, south, east, west, {'building': True})
                logger.info(f"Using fallback bounding box, extracted {len(buildings)} buildings")
                
            except Exception as e2:
                logger.error(f"Both OSM extraction methods failed: {e2}")
                logger.info("Creating minimal scene with ground plane only")
                return [], center_point
        
        building_geometries = self._process_building_geometries(buildings, center_lon, center_lat)
        
        return building_geometries, center_point
    
    def _process_building_geometries(self, buildings, center_lon: float, center_lat: float) -> List[Dict]:
        building_geometries = []
        
        logger.info("Processing building geometries...")
        
        if len(buildings) == 0:
            logger.warning("No buildings to process")
            return building_geometries
        
        for idx, building in buildings.iterrows():
            try:
                geometry = building['geometry']
                
                if geometry is None or geometry.is_empty:
                    continue
                
                # Handle different geometry types
                if geometry.geom_type == 'Polygon':
                    polygons = [geometry]
                elif geometry.geom_type == 'MultiPolygon':
                    polygons = list(geometry.geoms)
                else:
                    logger.debug(f"Skipping non-polygon geometry: {geometry.geom_type}")
                    continue
                
                # Estimate building height
                height = self._estimate_building_height(building)
                
                # Process each polygon
                for poly_idx, polygon in enumerate(polygons):
                    if polygon.is_empty or not polygon.is_valid:
                        continue
                        
                    try:
                        exterior_coords = list(polygon.exterior.coords)
                        local_coords = [self._lonlat_to_local(lon, lat, center_lon, center_lat) 
                                      for lon, lat in exterior_coords]
                        
                        if len(local_coords) >= 3:
                            interior_rings = []
                            for interior in polygon.interiors:
                                interior_coords = list(interior.coords)
                                local_interior = [self._lonlat_to_local(lon, lat, center_lon, center_lat) 
                                                for lon, lat in interior_coords]
                                if len(local_interior) >= 3:
                                    interior_rings.append(local_interior)
                            
                            # Generate a unique ID for the building
                            building_id = f"building_{idx}_{poly_idx}"
                            safe_building_id = self._sanitize_building_id(building_id)
                            
                            building_geometries.append({
                                'id': safe_building_id,
                                'exterior': local_coords,
                                'interiors': interior_rings,
                                'height': height
                            })
                    except Exception as e:
                        logger.debug(f"Error processing polygon {poly_idx} of building {idx}: {e}")
                        continue
                        
            except Exception as e:
                logger.debug(f"Error processing building {idx}: {e}")
                continue
        
        logger.info(f"Processed {len(building_geometries)} building geometries")
        return building_geometries
    
    def _lonlat_to_local(self, lon: float, lat: float, center_lon: float, center_lat: float) -> Tuple[float, float]:
        try:
            x, y = self.transformer.transform(lon, lat)
            center_x, center_y = self.transformer.transform(center_lon, center_lat)
            return x - center_x, y - center_y
        except Exception as e:
            logger.warning(f"Coordinate transformation failed for ({lon}, {lat}): {e}")
            lat_m = (lat - center_lat) * 111000
            lon_m = (lon - center_lon) * 111000 * math.cos(math.radians(center_lat))
            return lon_m, lat_m
    
    def _estimate_building_height(self, building) -> float:
        try:
            if 'height' in building and pd.notna(building['height']):
                try:
                    height = float(str(building['height']).replace('m', '').strip())
                    if 5 <= height <= 200:
                        return height
                except (ValueError, TypeError):
                    pass
                    
            if 'building:levels' in building and pd.notna(building['building:levels']):
                try:
                    levels = float(str(building['building:levels']).strip())
                    if 1 <= levels <= 50:  # Reasonable level range
                        return levels * 3.0  # 3m per level
                except (ValueError, TypeError):
                    pass
                    
            building_type = building.get('building', '').lower() if 'building' in building else ''
            if building_type in ['house', 'residential']:
                return 8.0
            elif building_type in ['office', 'commercial']:
                return 15.0
            elif building_type in ['industrial', 'warehouse']:
                return 12.0
            elif building_type in ['apartment', 'apartments']:
                return 20.0
            
        except Exception as e:
            logger.debug(f"Error estimating building height: {e}")
        
        return 10.0
    
    def _sanitize_building_id(self, building_id: str) -> str:
        safe_id = str(building_id).replace("'", "").replace('"', "").replace("(", "").replace(")", "")
        safe_id = safe_id.replace(",", "_").replace(" ", "_").replace(":", "_").replace("/", "_")
        safe_id = safe_id.replace("\\", "_").replace("*", "_").replace("?", "_").replace("<", "_")
        safe_id = safe_id.replace(">", "_").replace("|", "_")
        return safe_id
    
    def filter_rome_csv(self, bounds: Dict[str, Tuple[float, float]], 
                       output_path: str = None) -> Tuple[pd.DataFrame, str]:
        logger.info(f"Loading Rome CSV from: {self.rome_csv_path}")
        
        if not os.path.exists(self.rome_csv_path):
            raise FileNotFoundError(f"Rome CSV file not found: {self.rome_csv_path}")
        
        try:
            df = pd.read_csv(self.rome_csv_path)
            logger.info(f"Loaded {len(df)} total records from Rome CSV")
        except Exception as e:
            logger.error(f"Failed to load Rome CSV: {e}")
            raise
        
        required_columns = ['BSLatitude', 'BSLongitude', 'UELatitude', 'UELongitude']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in Rome CSV: {missing_columns}")
        
        logger.info(f"Columns: {list(df.columns)}")
        
        if 'BS_location' not in df.columns:
            logger.info("Creating BS_location column from BSLatitude and BSLongitude")
            df['BS_location'] = df['BSLatitude'].round(6).astype(str) + '_' + df['BSLongitude'].round(6).astype(str)
        
        if 'UE_location' not in df.columns:
            logger.info("Creating UE_location column from UELatitude and UELongitude")
            df['UE_location'] = df['UELatitude'].round(6).astype(str) + '_' + df['UELongitude'].round(6).astype(str)
        
        logger.info(f"Original data bounds:")
        logger.info(f"  BS Latitude: {df['BSLatitude'].min():.6f} to {df['BSLatitude'].max():.6f}")
        logger.info(f"  BS Longitude: {df['BSLongitude'].min():.6f} to {df['BSLongitude'].max():.6f}")
        logger.info(f"  UE Latitude: {df['UELatitude'].min():.6f} to {df['UELatitude'].max():.6f}")
        logger.info(f"  UE Longitude: {df['UELongitude'].min():.6f} to {df['UELongitude'].max():.6f}")
        
        lat_min, lat_max = bounds['lat_range']
        lon_min, lon_max = bounds['lon_range']
        
        logger.info(f"Filtering with bounds:")
        logger.info(f"  Latitude: {lat_min:.6f} to {lat_max:.6f}")
        logger.info(f"  Longitude: {lon_min:.6f} to {lon_max:.6f}")
        
        # Filter both BS and UE coordinates to be within bounds
        mask = (
            (df['BSLatitude'] >= lat_min) & (df['BSLatitude'] <= lat_max) &
            (df['BSLongitude'] >= lon_min) & (df['BSLongitude'] <= lon_max) &
            (df['UELatitude'] >= lat_min) & (df['UELatitude'] <= lat_max) &
            (df['UELongitude'] >= lon_min) & (df['UELongitude'] <= lon_max)
        )
        
        bs_lat_mask = (df['BSLatitude'] >= lat_min) & (df['BSLatitude'] <= lat_max)
        bs_lon_mask = (df['BSLongitude'] >= lon_min) & (df['BSLongitude'] <= lon_max)
        ue_lat_mask = (df['UELatitude'] >= lat_min) & (df['UELatitude'] <= lat_max)
        ue_lon_mask = (df['UELongitude'] >= lon_min) & (df['UELongitude'] <= lon_max)
        
        logger.info(f"Filtering step by step:")
        logger.info(f"  Records with BS in lat range: {bs_lat_mask.sum()}")
        logger.info(f"  Records with BS in lon range: {bs_lon_mask.sum()}")
        logger.info(f"  Records with UE in lat range: {ue_lat_mask.sum()}")
        logger.info(f"  Records with UE in lon range: {ue_lon_mask.sum()}")
        logger.info(f"  Records with both BS and UE in bounds: {mask.sum()}")
        
        if mask.sum() == 0:
            logger.warning("No records found with both BS and UE in bounds. Trying BS-only filtering...")
            mask_bs_only = bs_lat_mask & bs_lon_mask
            logger.info(f"  Records with BS in bounds (ignoring UE): {mask_bs_only.sum()}")
            
            if mask_bs_only.sum() > 0:
                logger.info("Using BS-only filtering to get some data")
                mask = mask_bs_only
            else:
                logger.warning("No records found even with BS-only filtering. Using relaxed bounds...")
                lat_margin = (lat_max - lat_min) * 0.2
                lon_margin = (lon_max - lon_min) * 0.2
                
                relaxed_mask = (
                    (df['BSLatitude'] >= lat_min - lat_margin) & (df['BSLatitude'] <= lat_max + lat_margin) &
                    (df['BSLongitude'] >= lon_min - lon_margin) & (df['BSLongitude'] <= lon_max + lon_margin)
                )
                
                logger.info(f"  Records with relaxed BS bounds: {relaxed_mask.sum()}")
                if relaxed_mask.sum() > 0:
                    mask = relaxed_mask
        
        filtered_df = df[mask].copy()
        logger.info(f"Filtered to {len(filtered_df)} records within bounds")
        
        if len(filtered_df) == 0:
            logger.error("No data found after filtering. Check bounds and data coverage.")
            raise ValueError("No data found after filtering")
        
        unique_bs = filtered_df['BS_location'].unique()
        logger.info(f"Unique BS locations in filtered data: {len(unique_bs)}")
        for bs_loc in unique_bs[:5]:
            bs_data = filtered_df[filtered_df['BS_location'] == bs_loc]
            bs_lat = bs_data['BSLatitude'].iloc[0]
            bs_lon = bs_data['BSLongitude'].iloc[0]
            logger.info(f"  BS {bs_loc}: ({bs_lat:.6f}, {bs_lon:.6f}) - {len(bs_data)} records")
        
        if 'RSSI' in filtered_df.columns:
            nan_count = filtered_df['RSSI'].isna().sum()
            filtered_df['RSSI'] = filtered_df['RSSI'].fillna(-90.0)
            logger.info(f"Filled {nan_count} NaN RSSI values with -90.0")
        else:
            logger.warning("RSSI column not found in dataset")
        
        if output_path is None:
            lat_center = (lat_min + lat_max) / 2
            lon_center = (lon_min + lon_max) / 2
            output_path = f"rome_filtered_{lat_center:.6f}_{lon_center:.6f}.csv"
        
        try:
            filtered_df.to_csv(output_path, index=False)
            logger.info(f"Filtered CSV saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save filtered CSV: {e}")
            raise
        
        return filtered_df, output_path
    
    def generate_region_data(self, bounds: Dict[str, Tuple[float, float]], 
                           output_dir: str = "generated_region",
                           scene_name: str = None) -> Dict[str, str]:
        logger.info(f"Generating region data for bounds: {bounds}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if scene_name is None:
            center_lat = (bounds['lat_range'][0] + bounds['lat_range'][1]) / 2
            center_lon = (bounds['lon_range'][0] + bounds['lon_range'][1]) / 2
            scene_name = f"scene_{center_lat:.6f}_{center_lon:.6f}"
        
        csv_output_path = os.path.join(output_dir, f"{scene_name}_filtered.csv")
        filtered_df, csv_path = self.filter_rome_csv(bounds, csv_output_path)
        
        try:
            building_geometries, center_point = self.extract_osm_buildings(bounds)
        except Exception as e:
            logger.error(f"OSM building extraction failed: {e}")
            building_geometries, center_point = [], (
                (bounds['lat_range'][0] + bounds['lat_range'][1]) / 2,
                (bounds['lon_range'][0] + bounds['lon_range'][1]) / 2
            )
        
        from .scene_generator import SceneGenerator
        scene_gen = SceneGenerator(output_dir)
        
        try:
            xml_path, meshes_dir = scene_gen.generate_complete_scene(
                building_geometries, bounds, center_point, scene_name
            )
        except Exception as e:
            logger.error(f"Scene generation failed: {e}")
            raise
        
        result_paths = {
            'csv_path': csv_path,
            'xml_path': xml_path,
            'meshes_dir': meshes_dir,
            'output_dir': output_dir,
            'scene_name': scene_name,
            'center_point': center_point,
            'bounds': bounds,
            'num_buildings': len(building_geometries),
            'num_records': len(filtered_df)
        }
        
        logger.info(f"Region data generation completed:")
        logger.info(f"  - CSV: {csv_path} ({len(filtered_df)} records)")
        logger.info(f"  - XML: {xml_path}")
        logger.info(f"  - Meshes: {meshes_dir} ({len(building_geometries)} buildings)")
        
        return result_paths
