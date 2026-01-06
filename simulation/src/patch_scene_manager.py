import os
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np

from .data_pipeline import DataPipeline
from .scene_generator import SceneGenerator

logger = logging.getLogger(__name__)


class PatchSceneManager:
    
    def __init__(self, bounds: Dict, grid_size: Tuple[int, int] = (10, 10), 
                 cache_dir: str = "./scene_cache", rome_csv_path: str = None):
        self.bounds = bounds
        self.grid_size = grid_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_pipeline = DataPipeline(rome_csv_path=None)
        
        self.lat_min, self.lat_max = bounds['lat_range']
        self.lon_min, self.lon_max = bounds['lon_range']
        self.patch_height = (self.lat_max - self.lat_min) / grid_size[0]
        self.patch_width = (self.lon_max - self.lon_min) / grid_size[1]
        
        self.area_id = self._generate_area_id()
        self.area_cache_dir = self.cache_dir / self.area_id
        self.area_cache_dir.mkdir(exist_ok=True)
        
        logger.info(f"PatchSceneManager initialized:")
        logger.info(f"  Area: {self.lat_max-self.lat_min:.4f}° x {self.lon_max-self.lon_min:.4f}°")
        logger.info(f"  Grid: {grid_size[0]}x{grid_size[1]} = {grid_size[0]*grid_size[1]} patches")
        logger.info(f"  Patch size: {self.patch_height:.4f}° x {self.patch_width:.4f}°")
        logger.info(f"  Cache: {self.area_cache_dir}")
    
    def _generate_area_id(self) -> str:
        config_str = f"{self.lat_min}_{self.lat_max}_{self.lon_min}_{self.lon_max}_{self.grid_size}"
        return hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    def get_patch_bounds(self, patch_idx: int) -> Dict:
        row = patch_idx // self.grid_size[1]
        col = patch_idx % self.grid_size[1]
        
        lat_north = self.lat_max - row * self.patch_height
        lat_south = lat_north - self.patch_height
        lon_west = self.lon_min + col * self.patch_width
        lon_east = lon_west + self.patch_width
        
        return {
            'lat_range': (lat_south, lat_north),
            'lon_range': (lon_west, lon_east),
            'center_lat': (lat_south + lat_north) / 2,
            'center_lon': (lon_west + lon_east) / 2,
            'patch_idx': patch_idx,
            'row': row,
            'col': col
        }
    
    def get_patch_scene_path(self, patch_idx: int) -> Path:
        patch_dir = self.area_cache_dir / f"patch_{patch_idx:03d}"
        patch_dir.mkdir(exist_ok=True)
        return patch_dir / f"scene_patch_{patch_idx:03d}.xml"
    
    def is_patch_cached(self, patch_idx: int) -> bool:
        scene_path = self.get_patch_scene_path(patch_idx)
        meshes_dir = scene_path.parent / "meshes"
        return scene_path.exists() and meshes_dir.exists()
    
    def generate_patch_scene(self, patch_idx: int) -> Path:
        scene_path = self.get_patch_scene_path(patch_idx)
        
        if self.is_patch_cached(patch_idx):
            logger.info(f"Using cached scene for patch {patch_idx}: {scene_path}")
            return scene_path
        
        logger.info(f"Generating new scene for patch {patch_idx}")
        patch_bounds = self.get_patch_bounds(patch_idx)
        patch_dir = scene_path.parent
        
        try:
            building_geometries, center_point = self.data_pipeline.extract_osm_buildings(patch_bounds)
            logger.info(f"Extracted {len(building_geometries)} OSM buildings for patch {patch_idx}")
        except Exception as e:
            logger.error(f"OSM extraction failed for patch {patch_idx}: {e}")
            building_geometries = []
            center_point = (patch_bounds['center_lat'], patch_bounds['center_lon'])
        
        scene_gen = SceneGenerator(str(patch_dir))
        
        xml_path, meshes_dir = scene_gen.generate_complete_scene(
            building_geometries=building_geometries,
            bounds=patch_bounds,
            center_point=center_point,
            scene_name=f"scene_patch_{patch_idx:03d}",
            ground_margin_pct=0.1
        )
        
        info_path = patch_dir / "patch_info.json"
        patch_info = patch_bounds.copy()
        patch_info['num_buildings'] = len(building_geometries)
        patch_info['center_point'] = center_point
        with open(info_path, 'w') as f:
            json.dump(patch_info, f, indent=2)
        
        logger.info(f"Generated scene for patch {patch_idx}: {xml_path}")
        return Path(xml_path)
    
    def generate_random_bs_locations(self, patch_idx: int, num_bs: int = 15) -> List[Dict]:
        patch_bounds = self.get_patch_bounds(patch_idx)
        
        bs_locations = []
        for i in range(num_bs):
            lat = np.random.uniform(patch_bounds['lat_range'][0], patch_bounds['lat_range'][1])
            lon = np.random.uniform(patch_bounds['lon_range'][0], patch_bounds['lon_range'][1])
            
            altitude = np.random.uniform(20, 60)
            azimuth = np.random.uniform(0, 360)
            
            bs_locations.append({
                'bs_idx': i,
                'lat': lat,
                'lon': lon,
                'altitude': altitude,
                'azimuth': azimuth,
                'patch_idx': patch_idx
            })
        
        return bs_locations
