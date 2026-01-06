"""
Generate a synthetic pre-training dataset with FIXED UE positions.

Stage 1 – Generate N_FAKE unique UE positions around the city center,
then pair each UE with ALL base stations.

Stage 2 – Feed the (BS, UE) pairs to the existing Sionna simulation
pipeline in manageable batches and append the simulated
[RSSI, NSINR, NRSRP, NRSRQ] to a final CSV.
"""

from __future__ import annotations
import argparse, glob, json, logging, math, os, random
from pathlib import Path
import pandas as pd
import numpy as np
import re   

from src.simulation import lonlat_to_local, calculate_metrics, clear_scene, setup_scene_arrays, add_transmitters_receivers, run_simulation
from src.data_processing import create_tx_rx_maps
from config.parameters import OPTIMIZED_PARAMS, BS_OPTIMIZED_PARAMS

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

class PretrainingDataGenerator:
    def __init__(
        self,
        optimization_dir: str,
        output_dir: str,
        n_fake: int = 500,
        batch_size: int = 256,
        max_rx: int = 1024,
        ue_radius_m: float = 768.0,
    ):
        self.optimization_dir = Path(optimization_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.n_fake      = int(n_fake)
        self.batch_size  = int(batch_size)
        self.max_rx      = int(max_rx)
        self.ue_radius_m = float(ue_radius_m)

        # stage-1 / stage-2 product files
        self.pairs_csv   = self.output_dir / "pairs.csv"
        self.dataset_csv = self.output_dir / "pretraining_dataset.csv"

    def run(self) -> Path:
        log.info("─── Pre-training data generation started ───")
        if not self.pairs_csv.exists():
            self._stage1_create_pairs()
        self._stage2_simulate_pairs()
        log.info("Dataset written to %s", self.dataset_csv)
        return self.dataset_csv

    def _stage1_create_pairs(self) -> None:
        bs_entries = self._load_optimised_bs()
        num_bs = len(bs_entries)
        log.info("Creating %d unique UE positions to pair with %d BSs", self.n_fake, num_bs)

        # Find the bounding box of all BS positions to generate UEs within reasonable area
        bs_lats = [bs['lat'] for bs in bs_entries]
        bs_lons = [bs['lon'] for bs in bs_entries]
        
        min_lat, max_lat = min(bs_lats), max(bs_lats)
        min_lon, max_lon = min(bs_lons), max(bs_lons)
        
        # Expand the bounding box by some margin (in degrees)
        # Roughly 1 degree = 111km, so for 768m radius, add ~0.007 degrees
        # margin = 0.01  # ~1.1km margin
        # min_lat -= margin
        # max_lat += margin
        # min_lon -= margin
        # max_lon += margin
        #for bs_corretions_individual_opt
        # min_lat = 41.8584
        # max_lat = 41.8766
        # min_lon = 12.4536
        # max_lon = 12.4754
        #for trastevere_scene.xml
        min_lat = 41.8196
        max_lat = 41.8350
        min_lon = 12.4575
        max_lon = 12.4763


        log.info(f"Generating UEs within bounds: lat[{min_lat:.6f}, {max_lat:.6f}], lon[{min_lon:.6f}, {max_lon:.6f}]")

        # Generate N_FAKE random UE positions within the expanded bounding box
        ue_absolute_positions = []
        for i in range(self.n_fake):
            # Random lat/lon within bounds
            ue_lat = random.uniform(min_lat, max_lat)
            ue_lon = random.uniform(min_lon, max_lon)
            
            ue_absolute_positions.append({
                'ue_id': i,
                'ue_lat': ue_lat,
                'ue_lon': ue_lon,
            })
        
        log.info("Generated %d unique UE positions", len(ue_absolute_positions))
        
        # Now pair each UE with each BS
        rows = []
        for ue in ue_absolute_positions:
            for bs in bs_entries:
                # Calculate distance and bearing from THIS BS to THIS UE
                dlat_bs_ue = ue['ue_lat'] - bs["lat"]
                dlon_bs_ue = ue['ue_lon'] - bs["lon"]
                
                # Approximate distance calculation
                distance_m = np.sqrt(
                    (dlat_bs_ue * 111_000)**2 + 
                    (dlon_bs_ue * 111_000 * math.cos(math.radians(bs["lat"])))**2
                )
                
                # Bearing from BS to UE
                bearing_rad = math.atan2(
                    dlon_bs_ue * 111_000 * math.cos(math.radians(bs["lat"])),
                    dlat_bs_ue * 111_000
                )
                
                rows.append({
                    'bs_idx': bs["bs_idx"],
                    'ue_id': ue['ue_id'],
                    'bs_lat': bs["lat"],
                    'bs_lon': bs["lon"],
                    'bs_alt': bs["altitude"],
                    'bs_azimuth': bs["azimuth"],
                    'ue_lat': ue['ue_lat'],  # Same for all BSs with same ue_id
                    'ue_lon': ue['ue_lon'],  # Same for all BSs with same ue_id
                    'distance_m': distance_m,  # Different for each BS-UE pair
                    'bearing_rad': bearing_rad,  # Different for each BS-UE pair
                })

        df = pd.DataFrame(rows)
        
        # Verify the structure
        unique_ues = df['ue_id'].nunique()
        unique_bs = df['bs_idx'].nunique()
        total_pairs = len(df)
        log.info(f"Verification: {unique_ues} unique UEs × {unique_bs} BSs = {total_pairs} total pairs")
        
        # Additional verification: check that each UE has same lat/lon across all BSs
        ue_positions_check = df.groupby('ue_id')[['ue_lat', 'ue_lon']].agg(['min', 'max'])
        position_variance = ((ue_positions_check['ue_lat']['max'] - ue_positions_check['ue_lat']['min']).abs().max(),
                           (ue_positions_check['ue_lon']['max'] - ue_positions_check['ue_lon']['min']).abs().max())
        log.info(f"UE position consistency check - max variance: lat={position_variance[0]:.9f}, lon={position_variance[1]:.9f}")
        
        df.to_csv(self.pairs_csv, index=False)
        log.info("Stage 1 complete – pairs written to %s (%d rows)", self.pairs_csv, len(df))

    def _generate_ues_around_each_bs(self, bs_entries: list[dict]) -> list[dict]:
        """
        Alternative method: Generate UEs within ue_radius_m of each BS.
        This creates clusters of UEs around each base station.
        """
        ue_absolute_positions = []
        ue_id_counter = 0
        
        ues_per_bs = self.n_fake // len(bs_entries)  # Distribute UEs evenly among BSs
        remaining_ues = self.n_fake % len(bs_entries)
        
        for bs_idx, bs in enumerate(bs_entries):
            # Number of UEs for this BS
            num_ues_this_bs = ues_per_bs + (1 if bs_idx < remaining_ues else 0)
            
            for _ in range(num_ues_this_bs):
                # Generate random point within ue_radius_m of this BS
                # Use polar coordinates: random angle and random distance
                angle = random.uniform(0, 2 * math.pi)
                # Use sqrt for uniform distribution in circle
                distance_m = random.uniform(0, self.ue_radius_m) 
                
                # Convert to lat/lon offset
                # Approximate: 1 degree lat ≈ 111km, 1 degree lon ≈ 111km * cos(lat)
                dlat = distance_m * math.cos(angle) / 111_000
                dlon = distance_m * math.sin(angle) / (111_000 * math.cos(math.radians(bs['lat'])))
                
                ue_lat = bs['lat'] + dlat
                ue_lon = bs['lon'] + dlon
                
                ue_absolute_positions.append({
                    'ue_id': ue_id_counter,
                    'ue_lat': ue_lat,
                    'ue_lon': ue_lon,
                })
                ue_id_counter += 1
        
        log.info(f"Generated {len(ue_absolute_positions)} UEs clustered around {len(bs_entries)} BSs within {self.ue_radius_m}m radius")
        return ue_absolute_positions

    def _load_optimised_bs(self) -> list[dict]:
        """Read ONLY the corrected_*.json files and return a clean list of BS entries."""
        pattern = str(self.optimization_dir / "corrected_*_result.json")
        files   = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No corrected_*.json files in {self.optimization_dir}")

        entries: list[dict] = []
        for fp in files:
            try:
                data = json.loads(Path(fp).read_text())
            except Exception as e:
                log.warning("Cannot read %s – skipped (%s)", fp, e)
                continue

            best = (data.get("best_parameters") or
                    data.get("optimization_result", {}).get("best_parameters"))
            if best is None:
                log.debug("No best_parameters in %s – skipped", fp)
                continue

            raw_idx = data.get("bs_idx") or data.get("target_bs_idx")
            if raw_idx is None:
                m = re.search(r"bs_(\d+)", Path(fp).stem)
                raw_idx = m.group(1) if m else len(entries)
            bs_idx_int = int(raw_idx)

            entries.append({
                'bs_idx': bs_idx_int,
                'lat': float(best["latitude"]),
                'lon': float(best["longitude"]),
                'altitude': float(best.get("altitude", 30.0)),
                'azimuth': float(best.get("azimuth", 0.0)),
            })

        if not entries:
            raise RuntimeError("No usable optimisation JSONs found.")
        
        # Sort by bs_idx for consistent ordering
        entries.sort(key=lambda x: x['bs_idx'])
        log.info("Loaded %d corrected BS entries", len(entries))
        return entries

    def _stage2_simulate_pairs(self) -> None:
        """Run simulations maintaining UE ID consistency."""
        log.info("Stage 2 – running batched simulations")

        df_pairs = pd.read_csv(self.pairs_csv)
        log.info(f"Total pairs to process: {len(df_pairs)}")
        
        # Verify data structure
        num_unique_ues = df_pairs['ue_id'].nunique()
        num_unique_bs = df_pairs['bs_idx'].nunique()
        unique_bs_indices = sorted(df_pairs['bs_idx'].unique())
        log.info(f"Found {num_unique_ues} unique UEs and {num_unique_bs} unique BSs")
        log.info(f"BS indices: {unique_bs_indices[0]} to {unique_bs_indices[-1]} (with gaps)")
        

        first_row = {
            'bs_idx': 0, 'ue_id': 0, 'bs_lat': 0.0, 'bs_lon': 0.0, 'bs_alt': 0.0, 
            'bs_azimuth': 0.0, 'ue_lat': 0.0, 'ue_lon': 0.0, 'distance_m': 0.0, 
            'bearing_rad': 0.0, 'sim_rssi': 0.0, 'sim_nsinr': 0.0, 'sim_nrsrp': 0.0, 'sim_nrsrq': 0.0
        }
        pd.DataFrame([first_row]).head(0).to_csv(self.dataset_csv, index=False)
        log.info(f"Initialized output CSV: {self.dataset_csv}")
        
        total_saved = 0
        failed_count = 0

        # Load scene once
        scene_xml = "trastevere_scene.xml"
        from sionna.rt import load_scene
        scene = load_scene(scene_xml)
        setup_scene_arrays(scene, OPTIMIZED_PARAMS)

        # Process each BS separately
        grouped = df_pairs.groupby('bs_idx')
        
        for bs_idx, bs_group in grouped:
            bs_group = bs_group.reset_index(drop=True)
            log.info(f"\nProcessing BS {bs_idx} with {len(bs_group)} UEs")
            
            # Get BS info from first row
            first_row = bs_group.iloc[0]
            bs_lat = first_row.bs_lat
            bs_lon = first_row.bs_lon
            bs_alt = first_row.bs_alt
            bs_azimuth = first_row.bs_azimuth
            
            # Create TX map with only this single BS
            bs_map = {
                f"{bs_lat:.6f}_{bs_lon:.6f}": (
                    int(bs_idx),
                    f"tx_{bs_idx}",
                    bs_lat,
                    bs_lon,
                )
            }
            
            # BS parameters for this single BS
            bs_params = {
                int(bs_idx): {
                    "altitude": bs_alt,
                    "azimuth": bs_azimuth
                }
            }
            
            # Process this BS's UEs in batches
            bs_saved = 0
            for i_start in range(0, len(bs_group), self.batch_size):
                chunk = bs_group.iloc[i_start : i_start + self.batch_size]
                batch_num = i_start // self.batch_size
                
                # Build RX map for this batch - use actual UE IDs for tracking
                rx_map = {}
                ue_id_mapping = {}  # Map from rx_idx to actual ue_id
                for idx, r in enumerate(chunk.itertuples(index=False)):
                    rx_map[f"ue_{idx}"] = (idx, f"rx_{idx}", r.ue_lat, r.ue_lon)
                    ue_id_mapping[idx] = r.ue_id
                
                # Clear scene and add only this BS and its UEs
                clear_scene(scene)
                add_transmitters_receivers(
                    scene, bs_map, rx_map,
                    OPTIMIZED_PARAMS,
                    bs_params,
                )
                
                log.debug(f"  Batch {batch_num}: {len(scene.transmitters)} TX, {len(scene.receivers)} RX")
                
                # Run simulation
                paths = run_simulation(scene, OPTIMIZED_PARAMS)
                if paths is None:
                    log.warning(f"  Batch {batch_num} for BS {bs_idx} failed, skipping {len(chunk)} UEs")
                    failed_count += len(chunk)
                    continue
                
                # Calculate metrics
                metrics = calculate_metrics(paths)
                if metrics is None:
                    log.warning(f"  Metrics calculation failed for batch {batch_num}")
                    failed_count += len(chunk)
                    continue
                    
                rssi, nsinr, nrsrp, nrsrq = metrics

                # Handle different possible shapes
                if rssi.ndim == 2 and rssi.shape == (len(chunk), 1):
                    rssi = rssi[:, 0]
                    nsinr = nsinr[:, 0]
                    nrsrp = nrsrp[:, 0]
                    nrsrq = nrsrq[:, 0]
                elif rssi.ndim == 2 and rssi.shape == (1, len(chunk)):
                    rssi = rssi[0, :]
                    nsinr = nsinr[0, :]
                    nrsrp = nrsrp[0, :]
                    nrsrq = nrsrq[0, :]
                elif rssi.ndim != 1 or len(rssi) != len(chunk):
                    log.error(f"  Unexpected metrics shape: {rssi.shape}, chunk size: {len(chunk)}")
                    failed_count += len(chunk)
                    continue
                

                batch_rows = []
                batch_saved = 0
                for idx in range(len(chunk)):
                    row = chunk.iloc[idx]
                    try:
                        batch_rows.append({
                            'bs_idx': int(row.bs_idx),
                            'ue_id': int(row.ue_id),  # Actual UE ID from pairs.csv
                            'bs_lat': float(row.bs_lat),
                            'bs_lon': float(row.bs_lon),
                            'bs_alt': float(row.bs_alt),
                            'bs_azimuth': float(row.bs_azimuth),
                            'ue_lat': float(row.ue_lat),
                            'ue_lon': float(row.ue_lon),
                            'distance_m': float(row.distance_m),
                            'bearing_rad': float(row.bearing_rad),
                            'sim_rssi': float(rssi[idx]),
                            'sim_nsinr': float(nsinr[idx]),
                            'sim_nrsrp': float(nrsrp[idx]),
                            'sim_nrsrq': float(nrsrq[idx]),
                        })
                        batch_saved += 1
                    except Exception as e:
                        log.error(f"    Error saving row {idx}: {e}")
                        failed_count += 1
                
                # Write batch to CSV immediately
                if batch_rows:
                    batch_df = pd.DataFrame(batch_rows)
                    batch_df.to_csv(self.dataset_csv, mode='a', header=False, index=False)
                    total_saved += len(batch_rows)
                    del batch_rows, batch_df  # Free memory immediately
                
                log.info(f"  Batch {batch_num}: processed {len(chunk)} UEs, saved {batch_saved}")
                bs_saved += batch_saved
            
            log.info(f"BS {bs_idx} complete: saved {bs_saved}/{len(bs_group)} UEs")
        
        # Final summary
        log.info("\nSimulation complete:")
        log.info(f"  Total saved: {total_saved}/{len(df_pairs)} ({100*total_saved/len(df_pairs):.1f}%)")
        log.info(f"  Failed: {failed_count}")
        log.info(f"  Output written to {self.dataset_csv}")
        

        if total_saved > 0:
            try:
                df_final = pd.read_csv(self.dataset_csv)
                log.info(f"  Final verification: {len(df_final)} rows in CSV")
                log.info(f"  Unique UEs: {df_final['ue_id'].nunique()}")
                log.info(f"  Unique BSs: {df_final['bs_idx'].nunique()}")
            except Exception as e:
                log.warning(f"Could not verify final CSV: {e}")
        else:
            log.error("No data was successfully processed!")



def _cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--opt_dir", required=True, help="directory with corrected_*.json files")
    p.add_argument("--out_dir", required=True, help="where to store CSVs")
    p.add_argument("--n_fake", type=int, default=500, help="synthetic UE per BS")
    p.add_argument("--batch",  type=int, default=256, help="UE receivers per simulation run")
    p.add_argument("--max_rx", type=int, default=1024, help="safety cap (unused)")
    args = p.parse_args()

    gen = PretrainingDataGenerator(
        optimization_dir=args.opt_dir,
        output_dir=args.out_dir,
        n_fake=args.n_fake,
        batch_size=args.batch,
        max_rx=args.max_rx,
    )
    gen.run()

if __name__ == "__main__":
    _cli()