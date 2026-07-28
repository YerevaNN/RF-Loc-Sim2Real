from __future__ import annotations
import logging
import pandas as pd
import numpy as np
import random
import math
import os
from pathlib import Path
from typing import Optional
import json

from .pretraining_data_generator import PretrainingDataGenerator
from .patch_scene_manager import PatchSceneManager
from .simulation import (setup_scene_arrays, clear_scene, clear_receivers_only,
                         add_transmitters_receivers, add_receivers_only,
                         run_simulation, calculate_metrics,
                         lonlat_to_local, local_to_lonlat)
from . import bs_placement
from config.parameters import OPTIMIZED_PARAMS

_DEFAULT_PATTERNS_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs", "bs_patterns.json")

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
        end_patch: int = None,
        bs_variant: str = "constrained",
        ue_strategy: str = "street",
        ue_max_dist_m: float = 400.0,
        sidewalk_jitter_m: float = 4.0,
        ue_height_m: float = 1.5,
        road_cache_dir: str = "./road_cache",
        patterns_json: str = None,
        seed: int = 0,
        emit_viz: bool = True,
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
        self.end_patch = end_patch  # exclusive upper bound (None = all patches)

        # C upgrade: synthetic-BS variant + street-UE sampling + priors
        self.bs_variant = bs_variant
        self.ue_strategy = ue_strategy
        self.ue_max_dist_m = float(ue_max_dist_m)
        self.sidewalk_jitter_m = float(sidewalk_jitter_m)
        self.ue_height_m = float(ue_height_m)
        self.road_cache_dir = road_cache_dir
        self.emit_viz = emit_viz
        self.patterns = bs_placement.load_patterns(patterns_json or _DEFAULT_PATTERNS_JSON)
        self.rng = np.random.default_rng(seed)

        self.patch_manager = PatchSceneManager(
            bounds=area_bounds,
            grid_size=grid_size,
            cache_dir=scene_cache_dir,
        )

        log.info(f"Initialized patch-based generator:")
        log.info(f"  Grid: {grid_size[0]}x{grid_size[1]} = {grid_size[0]*grid_size[1]} patches")
        log.info(f"  BS per patch: {num_bs_per_patch} (variant={bs_variant})")
        log.info(f"  UE per patch: {n_fake} (strategy={ue_strategy}, height={ue_height_m}m)")
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
    
    def _load_patch_buildings(self, patch_idx: int):
        """Read the per-patch buildings.json sidecar (footprints + roof heights)."""
        bpath = self.patch_manager.get_patch_scene_path(patch_idx).parent / "buildings.json"
        if not bpath.exists():
            log.warning(f"  buildings.json missing for patch {patch_idx}: {bpath}")
            return None
        with open(bpath) as f:
            return json.load(f)

    def _sample_street_ues(self, buildings: dict, bs_list: list, n: int):
        """Sample n UEs on the road network near BSs (B'' street strategy),
        rejecting in-building points. Footprints (incl. courtyards) come from
        buildings.json; everything is computed in the patch local frame and
        mapped back to lon/lat."""
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        from .ue_sampling import fetch_road_network, sample_on_streets, sample_open_space

        clat = float(buildings["center_lat"])
        clon = float(buildings["center_lon"])
        polys = []
        for b in buildings["buildings"]:
            try:
                p = Polygon(b["exterior_local"], b.get("interiors_local") or None)
                if not p.is_valid:
                    p = p.buffer(0)
                if not p.is_empty and p.area > 0:
                    polys.append(p)
            except Exception:
                continue
        footprints = unary_union(polys) if polys else unary_union([])

        lat_min, lat_max = buildings["lat_range"]
        lon_min, lon_max = buildings["lon_range"]
        bounds = dict(north=lat_max, south=lat_min, east=lon_max, west=lon_min)
        roads = fetch_road_network(bounds, clon, clat, self.road_cache_dir, lonlat_to_local)
        bs_xy = [lonlat_to_local(b["lon"], b["lat"], center_lon=clon, center_lat=clat)
                 for b in bs_list]
        try:
            if roads is not None:
                xy = sample_on_streets(roads, footprints, bs_xy, n,
                                       max_dist_m=self.ue_max_dist_m,
                                       sidewalk_jitter_m=self.sidewalk_jitter_m,
                                       rng=self.rng)
            else:
                log.warning("  roads unavailable — open-space fallback")
                xy = sample_open_space(footprints, bs_xy, n,
                                       max_dist_m=self.ue_max_dist_m, rng=self.rng)
        except ValueError as e:
            log.warning(f"  street sampling failed ({e}) — open-space fallback")
            xy = sample_open_space(footprints, bs_xy, n,
                                   max_dist_m=self.ue_max_dist_m, rng=self.rng)
        ues = []
        for x, y in xy:
            lon, lat = local_to_lonlat(float(x), float(y), center_lon=clon, center_lat=clat)
            ues.append((float(lat), float(lon)))
        return ues

    def _create_pairs_patches(self) -> None:
        total_patches = self.grid_size[0] * self.grid_size[1]
        end = self.end_patch if self.end_patch is not None else total_patches
        end = min(end, total_patches)
        log.info(f"Stage 1 - pairs for patches [{self.start_patch}, {end}) "
                 f"of {total_patches}, {self.num_bs_per_patch} BS/patch "
                 f"({self.bs_variant}), {self.n_fake} UE/patch ({self.ue_strategy})")

        rows = []
        skipped = 0
        for patch_idx in range(self.start_patch, end):
            # Build (or reuse) the scene FIRST: rooftop BS placement and street
            # UE sampling both need the per-patch footprints/buildings.json that
            # scene generation emits. Scenes are cached, so Stage 2 reuses them.
            self.patch_manager.generate_patch_scene(patch_idx)
            buildings = self._load_patch_buildings(patch_idx)
            if not buildings or buildings.get("num_buildings", 0) == 0:
                log.warning(f"  patch {patch_idx}: no buildings - skipped")
                skipped += 1
                continue

            try:
                bs_list = bs_placement.place_bs(
                    self.bs_variant, buildings, self.patterns,
                    self.num_bs_per_patch, self.rng)
            except ValueError as e:
                log.warning(f"  patch {patch_idx}: BS placement failed ({e}) - skipped")
                skipped += 1
                continue

            if self.ue_strategy == "street":
                ue_positions = self._sample_street_ues(buildings, bs_list, self.n_fake)
            else:
                lat_min, lat_max = buildings["lat_range"]
                lon_min, lon_max = buildings["lon_range"]
                ue_positions = [(float(self.rng.uniform(lat_min, lat_max)),
                                 float(self.rng.uniform(lon_min, lon_max)))
                                for _ in range(self.n_fake)]
            if len(ue_positions) == 0:
                log.warning(f"  patch {patch_idx}: no UEs sampled - skipped")
                skipped += 1
                continue

            for ue_idx, (ue_lat, ue_lon) in enumerate(ue_positions):
                for bs in bs_list:
                    dlat = ue_lat - bs["lat"]
                    dlon = ue_lon - bs["lon"]
                    distance_m = math.sqrt(
                        (dlat * 111_000)**2 +
                        (dlon * 111_000 * math.cos(math.radians(bs["lat"])))**2)
                    bearing_rad = math.atan2(
                        dlon * 111_000 * math.cos(math.radians(bs["lat"])),
                        dlat * 111_000)
                    rows.append({
                        'patch_idx': patch_idx,
                        'bs_idx': bs["bs_idx"],
                        'ue_idx': ue_idx,
                        'bs_lat': bs["lat"],
                        'bs_lon': bs["lon"],
                        'bs_alt': bs["altitude"],
                        'bs_azimuth': bs["azimuth"],
                        'bs_power_dbm': bs["power_dbm"],
                        'ue_lat': ue_lat,
                        'ue_lon': ue_lon,
                        'distance_m': distance_m,
                        'bearing_rad': bearing_rad,
                        'bs_on_roof': bs["on_roof"],
                        'bs_variant': bs["variant"],
                    })

        df = pd.DataFrame(rows)
        n_patches = df['patch_idx'].nunique() if len(df) else 0
        log.info(f"Created pairs: {len(df)} rows across {n_patches} patches "
                 f"({skipped} patches skipped)")
        df.to_csv(self.pairs_csv, index=False)
        log.info(f"Patch pairs written to {self.pairs_csv} ({len(df)} rows)")

        # One confirmation image (this shard's first patch) right after Stage 1,
        # so scene/BS/UE placement can be eyeballed minutes into the job rather
        # than after the multi-hour sim.
        if self.emit_viz and len(df):
            try:
                self._viz_first_patch(int(df['patch_idx'].min()), df)
            except Exception as e:
                log.warning(f"  viz skipped: {e}")

    def _viz_first_patch(self, patch_idx: int, df: pd.DataFrame) -> None:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.patches import Polygon as _MplPoly

        buildings = self._load_patch_buildings(patch_idx)
        if not buildings:
            return
        clat, clon = buildings["center_lat"], buildings["center_lon"]
        sub = df[df["patch_idx"] == patch_idx]
        bs = sub.drop_duplicates("bs_idx")
        ue = sub.drop_duplicates("ue_idx")
        if len(ue) > 4000:
            ue = ue.sample(4000, random_state=0)

        fig, ax = plt.subplots(figsize=(11, 11))
        for b in buildings["buildings"]:
            ax.add_patch(_MplPoly(b["exterior_local"], closed=True,
                                  facecolor="0.80", edgecolor="0.55", lw=0.3, zorder=1))
            for ring in b.get("interiors_local", []):
                ax.add_patch(_MplPoly(ring, closed=True, facecolor="white",
                                      edgecolor="0.55", lw=0.3, zorder=2))
        uxy = [lonlat_to_local(r.ue_lon, r.ue_lat, center_lon=clon, center_lat=clat)
               for r in ue.itertuples()]
        if uxy:
            ax.scatter([p[0] for p in uxy], [p[1] for p in uxy], s=3,
                       c="tab:blue", alpha=0.5, zorder=3, label=f"UE (n={len(ue)})")
        for r in bs.itertuples():
            x, y = lonlat_to_local(r.bs_lon, r.bs_lat, center_lon=clon, center_lat=clat)
            on = bool(getattr(r, "bs_on_roof", False))
            ax.scatter([x], [y], s=40 + 3 * float(r.bs_alt), marker="^",
                       c=("tab:green" if on else "tab:red"),
                       edgecolors="black", linewidths=0.6, zorder=5)
        on_n = int(bs["bs_on_roof"].sum()) if "bs_on_roof" in bs else 0
        ax.set_title(f"Patch {patch_idx} | {bs['bs_variant'].iloc[0]} | "
                     f"{len(buildings['buildings'])} bldgs | {len(bs)} BS ({on_n} on-roof) | "
                     f"green=roof red=off blue=UE")
        ax.set_aspect("equal"); ax.set_xlabel("local x (m)"); ax.set_ylabel("local y (m)")
        ax.legend(loc="upper right")
        out = self.output_dir / f"patch_{patch_idx:03d}_viz.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Wrote placement viz -> {out}")

    @staticmethod
    def _flatten_metric(m, n: int):
        """Coerce a Sionna metric array to 1-D length n (single-TX batch), or
        None if the shape is unexpected. Mirrors the B'' shape handling."""
        if m.ndim == 2 and m.shape == (n, 1):
            return m[:, 0]
        if m.ndim == 2 and m.shape == (1, n):
            return m[0, :]
        if m.ndim == 1 and len(m) == n:
            return m
        return None

    def _simulate_pairs_patches(self) -> None:
        log.info("Stage 2 - batched simulations (patch mode, per-BS)")

        df_pairs = pd.read_csv(self.pairs_csv)
        log.info(f"Total pairs to process: {len(df_pairs)}")

        out_cols = ['patch_idx', 'bs_idx', 'ue_idx', 'bs_lat', 'bs_lon', 'bs_alt',
                    'bs_azimuth', 'bs_power_dbm', 'ue_lat', 'ue_lon', 'distance_m',
                    'bearing_rad', 'bs_on_roof', 'bs_variant',
                    'sim_rssi', 'sim_nsinr', 'sim_nrsrp', 'sim_nrsrq']

        checkpoint = self._load_checkpoint()
        # Seed the floor at start_patch-1 for a shard, but never roll BACK real
        # progress recorded in the checkpoint (so a crashed shard resumes where
        # it stopped instead of re-running — which would duplicate rows).
        if self.start_patch > 0:
            checkpoint['last_completed_patch'] = max(
                checkpoint['last_completed_patch'], self.start_patch - 1)

        # Write the header iff the output file doesn't exist yet (robust for a
        # fresh shard with start_patch>0; appends on resume).
        if not self.dataset_csv.exists():
            pd.DataFrame(columns=out_cols).to_csv(self.dataset_csv, index=False)
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

            # The scene was built in the patch's LOCAL frame (origin = patch
            # center). TX/RX must be transformed with the SAME center, else they
            # land in the wrong place relative to the meshes.
            pb = self.patch_manager.get_patch_bounds(patch_idx)
            center_lat = pb['center_lat']
            center_lon = pb['center_lon']

            scene_xml = self.patch_manager.generate_patch_scene(patch_idx)
            scene = load_scene(str(scene_xml))
            setup_scene_arrays(scene, OPTIMIZED_PARAMS)

            patch_rows = []
            patch_failed = 0

            # One BS at a time, so each BS's own power_dbm is applied in metrics.
            for bs_idx, bs_group in patch_data.groupby('bs_idx'):
                bs_group = bs_group.reset_index(drop=True)
                first = bs_group.iloc[0]
                bs_lat = first.bs_lat
                bs_lon = first.bs_lon
                bs_alt = first.bs_alt
                bs_azimuth = first.bs_azimuth
                bs_power = float(first.bs_power_dbm)

                bs_map = {
                    f"{bs_lat:.6f}_{bs_lon:.6f}": (int(bs_idx), f"tx_{patch_idx}_{bs_idx}",
                                                   bs_lat, bs_lon)
                }
                bs_params = {int(bs_idx): {"altitude": bs_alt, "azimuth": bs_azimuth,
                                           "power_dbm": bs_power}}

                clear_scene(scene)
                add_transmitters_receivers(
                    scene, bs_map, {}, OPTIMIZED_PARAMS, bs_params,
                    center_lon=center_lon, center_lat=center_lat,
                )

                n_batches = (len(bs_group) + self.batch_size - 1) // self.batch_size
                bs_floor = 0
                bs_seen = 0
                for i_start in range(0, len(bs_group), self.batch_size):
                    chunk = bs_group.iloc[i_start:i_start + self.batch_size]

                    rx_map = {}
                    for idx, r in enumerate(chunk.itertuples(index=False)):
                        rx_map[f"ue_{idx}"] = (idx, f"rx_{idx}", r.ue_lat, r.ue_lon)

                    clear_receivers_only(scene)
                    add_receivers_only(scene, rx_map, center_lon=center_lon,
                                       center_lat=center_lat, ue_height_m=self.ue_height_m)

                    paths = run_simulation(scene, OPTIMIZED_PARAMS)
                    if paths is None:
                        patch_failed += len(chunk)
                        continue

                    metrics = calculate_metrics(paths, tx_power_dbm=bs_power)
                    if metrics is None:
                        patch_failed += len(chunk)
                        continue

                    n = len(chunk)
                    flat = [self._flatten_metric(np.asarray(m), n) for m in metrics]
                    if any(f is None for f in flat):
                        log.error(f"    Unexpected metric shape for BS {bs_idx}, batch n={n}")
                        patch_failed += n
                        continue
                    rssi, nsinr, nrsrp, nrsrq = flat

                    # Lightweight progress so a 100k-UE patch isn't a black box:
                    # log every 25th batch with the running floor-rate (the
                    # per-batch log timestamps also reveal RT throughput).
                    bs_floor += int(((rssi <= -140.0) | np.isnan(rssi)).sum())
                    bs_seen += n
                    bnum = i_start // self.batch_size
                    _every = max(1, n_batches // 20)  # ~20 progress lines regardless of batch size
                    if (bnum % _every == 0) or (bnum == n_batches - 1):
                        log.info(f"    BS {bs_idx} batch {bnum+1}/{n_batches}: "
                                 f"{bs_seen} UEs done, floor={100.0*bs_floor/max(1,bs_seen):.1f}%")

                    for idx in range(n):
                        row = chunk.iloc[idx]
                        patch_rows.append({
                            'patch_idx': int(row.patch_idx),
                            'bs_idx': int(row.bs_idx),
                            'ue_idx': int(row.ue_idx),
                            'bs_lat': float(row.bs_lat),
                            'bs_lon': float(row.bs_lon),
                            'bs_alt': float(row.bs_alt),
                            'bs_azimuth': float(row.bs_azimuth),
                            'bs_power_dbm': float(row.bs_power_dbm),
                            'ue_lat': float(row.ue_lat),
                            'ue_lon': float(row.ue_lon),
                            'distance_m': float(row.distance_m),
                            'bearing_rad': float(row.bearing_rad),
                            'bs_on_roof': bool(row.bs_on_roof),
                            'bs_variant': str(row.bs_variant),
                            'sim_rssi': float(rssi[idx]),
                            'sim_nsinr': float(nsinr[idx]),
                            'sim_nrsrp': float(nrsrp[idx]),
                            'sim_nrsrq': float(nrsrq[idx]),
                        })

                # Per-BS floor-rate sanity (matches the B'' diagnostic).
                if len(bs_group):
                    sub = [r for r in patch_rows if r['bs_idx'] == int(bs_idx)]
                    if sub:
                        arr = np.asarray([r['sim_rssi'] for r in sub], dtype=float)
                        n_floor = int(((arr <= -140.0) | np.isnan(arr)).sum())
                        log.info(f"  BS {bs_idx}: {len(sub)} pairs, "
                                 f"floor(rssi<=-140)={n_floor}/{len(sub)} "
                                 f"({100.0*n_floor/max(1,len(sub)):.1f}%)")

            if patch_rows:
                pd.DataFrame(patch_rows)[out_cols].to_csv(
                    self.dataset_csv, mode='a', header=False, index=False)
                total_saved += len(patch_rows)
                failed_count += patch_failed
                log.info(f"Patch {patch_idx} complete: saved {len(patch_rows)} rows")
                self._save_checkpoint(patch_idx, total_saved, failed_count)
                del patch_rows
            else:
                failed_count += patch_failed
                self._save_checkpoint(patch_idx, total_saved, failed_count)
                log.warning(f"Patch {patch_idx} complete: no data saved")

        log.info("\nPatch processing complete:")
        denom = max(1, len(df_pairs))
        log.info(f"  Total saved: {total_saved}/{len(df_pairs)} ({100*total_saved/denom:.1f}%)")
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
    # C upgrade: synthetic-BS variant, street-UE sampling, priors
    p.add_argument("--bs_variant", choices=["constrained", "unconstrained"], default="constrained",
                   help="constrained=rooftop/capped height, unconstrained=free position")
    p.add_argument("--ue_strategy", choices=["street", "random"], default="street",
                   help="street=on-road near BS (B''), random=uniform in patch")
    p.add_argument("--ue_max_dist_m", type=float, default=400.0,
                   help="street strategy: sample road within this of a BS")
    p.add_argument("--sidewalk_jitter_m", type=float, default=4.0,
                   help="street strategy: lateral jitter off road centerline")
    p.add_argument("--ue_height_m", type=float, default=1.5, help="UE/receiver height (m)")
    p.add_argument("--road_cache_dir", default="./road_cache", help="OSM road-network cache dir")
    p.add_argument("--patterns_json", default=None,
                   help="BS priors JSON (default: configs/bs_patterns.json)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for placement/sampling")
    p.add_argument("--end_patch", type=int, default=None,
                   help="Exclusive upper patch bound for sharding (default: all). "
                        "Shard jobs must use distinct --out_dir.")
    p.add_argument("--no_viz", action="store_true",
                   help="Disable the post-Stage-1 placement confirmation PNG.")
    p.add_argument("--max_num_paths", type=int, default=None,
                   help="Override OPTIMIZED_PARAMS['max_num_paths_per_src'] (paths budget "
                        "per solver call, shared across the batch). Raise it when increasing "
                        "--batch so paths/receiver don't starve. Default: parameters.py (1e4).")

    args = p.parse_args()

    if args.max_num_paths is not None:
        OPTIMIZED_PARAMS['max_num_paths_per_src'] = int(args.max_num_paths)
        log.info(f"Overriding max_num_paths_per_src -> {int(args.max_num_paths)} "
                 f"(batch={args.batch}, ~{int(args.max_num_paths)//max(1,args.batch)} paths/receiver)")

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
        end_patch=args.end_patch,
        bs_variant=args.bs_variant,
        ue_strategy=args.ue_strategy,
        ue_max_dist_m=args.ue_max_dist_m,
        sidewalk_jitter_m=args.sidewalk_jitter_m,
        ue_height_m=args.ue_height_m,
        road_cache_dir=args.road_cache_dir,
        patterns_json=args.patterns_json,
        seed=args.seed,
        emit_viz=not args.no_viz,
    )

    gen.run()


if __name__ == "__main__":
    _cli()
