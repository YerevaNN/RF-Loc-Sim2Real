from __future__ import annotations
import logging
import pandas as pd
import random
import math
from pathlib import Path
from typing import Optional
import json

from .pretraining_data_generator import PretrainingDataGenerator
from .patch_scene_manager import PatchSceneManager
from .simulation import setup_scene_arrays, clear_scene, add_transmitters_receivers, run_simulation, calculate_metrics
from config.parameters import OPTIMIZED_PARAMS

log = logging.getLogger(__name__)


class PretrainingDataGeneratorPatches(PretrainingDataGenerator):
    
    def __init__(
        self,
        output_dir: str,
        n_fake: int = 500,
        batch_size: int = 256,
        max_rx: int = 1024,
        ue_radius_m: float = 1500.0,
        area_bounds: dict = None,
        grid_size: tuple = (10, 10),
        num_bs_per_patch: int = 15,
        scene_cache_dir: str = "./scene_cache",
        start_patch: int = 0,
    ):
        super().__init__(
            optimization_dir="./dummy",
            output_dir=output_dir,
            n_fake=n_fake,
            batch_size=batch_size,
            max_rx=max_rx,
            ue_radius_m=ue_radius_m
        )
        
        self.pairs_csv = self.output_dir / "pairs_patches.csv"
        self.dataset_csv = self.output_dir / "pretraining_dataset_patches.csv"
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        
        self.area_bounds = area_bounds
        self.grid_size = grid_size
        self.num_bs_per_patch = num_bs_per_patch
        self.start_patch = start_patch
        
        self.patch_manager = PatchSceneManager(
            bounds=area_bounds,
            grid_size=grid_size,
            cache_dir=scene_cache_dir,
        )
        
        log.info(f"Initialized patch-based generator:")
        log.info(f"  Grid: {grid_size[0]}x{grid_size[1]} = {grid_size[0]*grid_size[1]} patches")
        log.info(f"  BS per patch: {num_bs_per_patch}")
        log.info(f"  UE per patch: {n_fake}")
        log.info(f"  Total simulations: {grid_size[0]*grid_size[1]*num_bs_per_patch*n_fake:,}")
        if start_patch > 0:
            log.info(f"  Starting from patch: {start_patch}")
    
    def _save_checkpoint(self, patch_idx: int, total_saved: int, failed_count: int):
        checkpoint = {
            'last_completed_patch': patch_idx,
            'total_saved': total_saved,
            'failed_count': failed_count
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def _load_checkpoint(self) -> dict:
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {'last_completed_patch': -1, 'total_saved': 0, 'failed_count': 0}
    
    def run(self) -> Path:
        log.info("─── Patch-based pre-training data generation started ───")
        
        if not self.pairs_csv.exists():
            self._create_pairs_patches()
        
        self._simulate_pairs_patches()
        
        log.info("Dataset written to %s", self.dataset_csv)
        return self.dataset_csv
    
    def _create_pairs_patches(self) -> None:
        total_patches = self.grid_size[0] * self.grid_size[1]
        log.info(f"Creating pairs for {total_patches} patches, {self.num_bs_per_patch} BS per patch, {self.n_fake} UE per patch")
        
        rows = []
        for patch_idx in range(total_patches):
            patch_bounds = self.patch_manager.get_patch_bounds(patch_idx)
            lat_min, lat_max = patch_bounds['lat_range']
            lon_min, lon_max = patch_bounds['lon_range']
            
            bs_locations = self.patch_manager.generate_random_bs_locations(
                patch_idx, self.num_bs_per_patch
            )
            
            ue_positions = []
            for ue_idx in range(self.n_fake):
                ue_lat = random.uniform(lat_min, lat_max)
                ue_lon = random.uniform(lon_min, lon_max)
                ue_positions.append({
                    'ue_idx': ue_idx,
                    'ue_lat': ue_lat,
                    'ue_lon': ue_lon
                })
            
            for ue in ue_positions:
                for bs in bs_locations:
                    dlat = ue['ue_lat'] - bs["lat"]
                    dlon = ue['ue_lon'] - bs["lon"]
                    
                    distance_m = math.sqrt(
                        (dlat * 111_000)**2 + 
                        (dlon * 111_000 * math.cos(math.radians(bs["lat"])))**2
                    )
                    
                    bearing_rad = math.atan2(
                        dlon * 111_000 * math.cos(math.radians(bs["lat"])),
                        dlat * 111_000
                    )
                    
                    rows.append({
                        'patch_idx': patch_idx,
                        'bs_idx': bs["bs_idx"],
                        'ue_idx': ue['ue_idx'],
                        'bs_lat': bs["lat"],
                        'bs_lon': bs["lon"],
                        'bs_alt': bs["altitude"],
                        'bs_azimuth': bs["azimuth"],
                        'ue_lat': ue['ue_lat'],
                        'ue_lon': ue['ue_lon'],
                        'distance_m': distance_m,
                        'bearing_rad': bearing_rad,
                    })
        
        df = pd.DataFrame(rows)
        
        log.info(f"Created pairs:")
        log.info(f"  Total pairs: {len(df)}")
        log.info(f"  Patches: {df['patch_idx'].nunique()}")
        log.info(f"  BS per patch: {self.num_bs_per_patch}")
        log.info(f"  UE per patch: {self.n_fake}")
        log.info(f"  Expected: {total_patches} × {self.n_fake} × {self.num_bs_per_patch} = {total_patches * self.n_fake * self.num_bs_per_patch}")
        
        df.to_csv(self.pairs_csv, index=False)
        log.info(f"Patch pairs written to {self.pairs_csv} ({len(df)} rows)")
    
    def _simulate_pairs_patches(self) -> None:
        log.info("Stage 2 – running batched simulations (patch mode)")
        
        df_pairs = pd.read_csv(self.pairs_csv)
        log.info(f"Total pairs to process: {len(df_pairs)}")
        
        checkpoint = self._load_checkpoint()
        if self.start_patch > 0:
            checkpoint['last_completed_patch'] = self.start_patch - 1
        
        if checkpoint['last_completed_patch'] < 0:
            first_row = {
                'patch_idx': 0, 'bs_idx': 0, 'ue_idx': 0, 'bs_lat': 0.0, 'bs_lon': 0.0,
                'bs_alt': 0.0, 'bs_azimuth': 0.0, 'ue_lat': 0.0, 'ue_lon': 0.0,
                'distance_m': 0.0, 'bearing_rad': 0.0, 'sim_rssi': 0.0,
                'sim_nsinr': 0.0, 'sim_nrsrp': 0.0, 'sim_nrsrq': 0.0
            }
            pd.DataFrame([first_row]).head(0).to_csv(self.dataset_csv, index=False)
            log.info("Created new output CSV with headers")
        else:
            log.info(f"Resuming from patch {checkpoint['last_completed_patch'] + 1}")
        
        total_saved = checkpoint['total_saved']
        failed_count = checkpoint['failed_count']
        
        from sionna.rt import load_scene
        
        patch_groups = df_pairs.groupby('patch_idx')
        total_patches = len(patch_groups)
        
        for patch_idx, patch_data in patch_groups:
            if patch_idx <= checkpoint['last_completed_patch']:
                continue
                
            log.info(f"\n{'='*60}")
            log.info(f"Processing patch {patch_idx}/{total_patches-1}")
            
            patch_rows = []
            patch_failed = 0
            
            scene_xml = self.patch_manager.generate_patch_scene(patch_idx)
            
            scene = load_scene(str(scene_xml))
            setup_scene_arrays(scene, OPTIMIZED_PARAMS)
            
            clear_scene(scene)
            
            unique_bs = patch_data.drop_duplicates(subset=['bs_idx'])
            bs_map = {}
            bs_params = {}
            
            for _, bs_row in unique_bs.iterrows():
                bs_idx = int(bs_row.bs_idx)
                bs_lat = bs_row.bs_lat
                bs_lon = bs_row.bs_lon
                bs_alt = bs_row.bs_alt
                bs_azimuth = bs_row.bs_azimuth
                
                bs_key = f"{bs_lat:.6f}_{bs_lon:.6f}"
                bs_map[bs_key] = (
                    bs_idx,
                    f"tx_{patch_idx}_{bs_idx}",
                    bs_lat,
                    bs_lon,
                )
                
                bs_params[bs_idx] = {
                    "altitude": bs_alt,
                    "azimuth": bs_azimuth
                }
            
            log.info(f"  Added {len(bs_map)} BSs to scene")
            
            # Add all transmitters ONCE at the beginning
            add_transmitters_receivers(
                scene, bs_map, {},  # Empty rx_map for now
                OPTIMIZED_PARAMS,
                bs_params,
            )
            
            ue_groups = patch_data.groupby('ue_idx')
            ue_indices = list(ue_groups.groups.keys())
            
            from sionna.rt import Receiver
            
            for batch_start in range(0, len(ue_indices), self.batch_size):
                batch_ue_indices = ue_indices[batch_start:batch_start + self.batch_size]
                batch_num = batch_start // self.batch_size
                
                batch_rows = patch_data[patch_data['ue_idx'].isin(batch_ue_indices)]
                
                for rx_id in list(scene.receivers.keys()):
                    scene.remove(rx_id)
                
                for ue_idx in batch_ue_indices:
                    ue_row = batch_rows[batch_rows['ue_idx'] == ue_idx].iloc[0]
                    
                    from .simulation import lonlat_to_local
                    rx_pos = lonlat_to_local(ue_row.ue_lon, ue_row.ue_lat)
                    
                    rx = Receiver(
                        name=f"rx_{ue_idx}",
                        position=(float(rx_pos[0]), float(rx_pos[1]), 1.0),
                        orientation=[0.0, 0.0, 0.0]
                    )
                    scene.add(rx)
                
                log.info(f"  Batch {batch_num}: {len(scene.transmitters)} TX, {len(scene.receivers)} RX")
                
                paths = run_simulation(scene, OPTIMIZED_PARAMS)
                if paths is None:
                    log.warning(f"  Batch {batch_num} failed, skipping {len(batch_ue_indices)} UEs")
                    patch_failed += len(batch_rows)
                    continue
                
                metrics = calculate_metrics(paths)
                if metrics is None:
                    log.warning(f"  Metrics calculation failed for batch {batch_num}")
                    patch_failed += len(batch_rows)
                    continue
                
                rssi, nsinr, nrsrp, nrsrq = metrics
                
                log.debug(f"  Metrics shape: {rssi.shape}")
                
                batch_saved = 0
                for ue_local_idx, ue_idx in enumerate(batch_ue_indices):
                    ue_rows = batch_rows[batch_rows['ue_idx'] == ue_idx]
                    
                    for bs_local_idx, (_, row) in enumerate(ue_rows.iterrows()):
                        try:
                            if rssi.ndim == 2:
                                metric_rssi = float(rssi[ue_local_idx, bs_local_idx])
                                metric_nsinr = float(nsinr[ue_local_idx, bs_local_idx])
                                metric_nrsrp = float(nrsrp[ue_local_idx, bs_local_idx])
                                metric_nrsrq = float(nrsrq[ue_local_idx, bs_local_idx])
                            else:
                                idx = ue_local_idx * len(bs_map) + bs_local_idx
                                metric_rssi = float(rssi[idx])
                                metric_nsinr = float(nsinr[idx])
                                metric_nrsrp = float(nrsrp[idx])
                                metric_nrsrq = float(nrsrq[idx])
                            
                            patch_rows.append({
                                'patch_idx': int(row.patch_idx),
                                'bs_idx': int(row.bs_idx),
                                'ue_idx': int(row.ue_idx),
                                'bs_lat': float(row.bs_lat),
                                'bs_lon': float(row.bs_lon),
                                'bs_alt': float(row.bs_alt),
                                'bs_azimuth': float(row.bs_azimuth),
                                'ue_lat': float(row.ue_lat),
                                'ue_lon': float(row.ue_lon),
                                'distance_m': float(row.distance_m),
                                'bearing_rad': float(row.bearing_rad),
                                'sim_rssi': metric_rssi,
                                'sim_nsinr': metric_nsinr,
                                'sim_nrsrp': metric_nrsrp,
                                'sim_nrsrq': metric_nrsrq,
                            })
                            batch_saved += 1
                        except Exception as e:
                            log.error(f"    Error saving row for UE {ue_idx}, BS {bs_local_idx}: {e}")
                            patch_failed += 1
                
                log.info(f"  Batch {batch_num}: processed {len(batch_ue_indices)} UEs × {len(bs_map)} BSs, saved {batch_saved} pairs")
            
            if patch_rows:
                patch_df = pd.DataFrame(patch_rows)
                patch_df.to_csv(self.dataset_csv, mode='a', header=False, index=False)
                total_saved += len(patch_rows)
                failed_count += patch_failed
                log.info(f"Patch {patch_idx} complete: saved {len(patch_rows)} rows to CSV")
                
                self._save_checkpoint(patch_idx, total_saved, failed_count)
                
                del patch_rows
                del patch_df
            else:
                failed_count += patch_failed
                log.warning(f"Patch {patch_idx} complete: no data saved")
        
        log.info("\nPatch processing complete:")
        log.info(f"  Total saved: {total_saved}/{len(df_pairs)} ({100*total_saved/len(df_pairs):.1f}%)")
        log.info(f"  Failed: {failed_count}")
        log.info(f"  Output written to {self.dataset_csv}")


def _cli() -> None:
    import argparse
    
    p = argparse.ArgumentParser(description="Generate synthetic pre-training data using patch-based approach")
    p.add_argument("--out_dir", required=True, help="Output directory for CSVs")
    p.add_argument("--n_fake", type=int, default=500, help="Synthetic UE per patch")
    p.add_argument("--batch", type=int, default=256, help="UE receivers per simulation run")
    p.add_argument("--max_rx", type=int, default=1024, help="Safety cap")
    
    # Patch-specific arguments
    p.add_argument("--lat_min", type=float, default=41.7175, help="Minimum latitude")
    p.add_argument("--lat_max", type=float, default=41.9675, help="Maximum latitude")
    p.add_argument("--lon_min", type=float, default=12.3025, help="Minimum longitude")
    p.add_argument("--lon_max", type=float, default=12.6525, help="Maximum longitude")
    p.add_argument("--grid_rows", type=int, default=10, help="Grid rows")
    p.add_argument("--grid_cols", type=int, default=10, help="Grid columns")
    p.add_argument("--bs_per_patch", type=int, default=15, help="BS per patch")
    p.add_argument("--scene_cache", default="./scene_cache", help="Scene cache directory")
    p.add_argument("--start_patch", type=int, default=0, help="Start from specific patch (for resuming)")  # NEW
    
    args = p.parse_args()
    
    bounds = {
        'lat_range': (args.lat_min, args.lat_max),
        'lon_range': (args.lon_min, args.lon_max)
    }
    
    gen = PretrainingDataGeneratorPatches(
        output_dir=args.out_dir,
        n_fake=args.n_fake,
        batch_size=args.batch,
        max_rx=args.max_rx,
        area_bounds=bounds,
        grid_size=(args.grid_rows, args.grid_cols),
        num_bs_per_patch=args.bs_per_patch,
        scene_cache_dir=args.scene_cache,
        start_patch=args.start_patch,
    )
    
    gen.run()


if __name__ == "__main__":
    _cli()
