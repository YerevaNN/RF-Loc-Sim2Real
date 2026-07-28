"""
Minimal Sionna diagnostic for the B'' generation pipeline.

Goal: identify why ~90% of (UE, BS) rows in the existing
`/data/ararat/rome_data/rome_b_prime_20260206_csv/const/cluster_0/scene_0.csv`
have `sim_rssi = -140.0` (i.e. `path_gain` clamped to `epsilon`).

What this script does
---------------------
1. Loads a small, deterministic mix of (UE, BS) rows from the CSV: a few BSs,
   and for each BS some "floor" rows (sim_rssi == -140) and some
   "got-signal" rows (sim_rssi > -120), so we can compare both regimes.
2. Loads the cluster scene XML the CSV was generated against (default
   `cluster_0_scene.xml`) and reproduces the production setup using the
   existing `simulation/src/simulation.py` helpers.
3. For each BS, runs Sionna FOUR times on the same set of UEs with the
   exact production knobs from `config/parameters.OPTIMIZED_PARAMS`, then
   with three progressively-relaxed variants:
     A  production    : max_depth=3, specular=False, diffuse=True, refraction=True
     B  +specular     : production but specular_reflection=True
     C  +max_depth=5  : production but max_depth=5
     D  both relaxed  : max_depth=5, specular=True (most permissive)
   Prediction: B is the dominant fix. The cluster XMLs in
   `/data/ararat/rome_sionna/scene_data/new_xmls/` use only
   `<bsdf type="diffuse">` materials, but the Sionna PathSolver flag
   `specular_reflection` controls whether the solver computes specular
   ray paths regardless of BSDF roughness — turning it off removes the
   dominant urban NLoS contributor.
4. Prints, per BS-batch and per (UE, BS) pair:
   - scene mesh count, scene bbox (TX/RX inside?), local-coord positions
   - `paths.a` tuple shape, `paths.tau` shape, `paths.interactions` shape
   - `paths.valid` count of valid paths per pair
   - distribution of interaction types (0=LoS, 1=specular, 2=diffuse, 4=refraction)
   - sum |a|^2 over (rx_ant, tx_ant, paths) and over only-valid paths
   - reproduced rssi_dbm via the production formula
   - comparison with the value already in the CSV

No production code is modified. Run via SLURM through
`slurm_datagen/diag/diag_paths.sh`.
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# Make `from src.simulation import ...` work when invoked from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_ROOT = REPO_ROOT / "simulation"
sys.path.insert(0, str(SIM_ROOT))

from src.simulation import (  # noqa: E402
    lonlat_to_local,
    clear_scene,
    setup_scene_arrays,
    add_transmitters_receivers,
)
from config.parameters import OPTIMIZED_PARAMS  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger("diag_paths")


# ------------------------------------------------------------------- #
# Cluster center lookup.                                              #
# Each cluster's scene was built around its own bbox-midpoint center  #
# (per wair_d_iot/all_bs_optim/generate_clustered_scenes.py:259-260). #
# lonlat_to_local MUST be called with this center, or BS/UE positions #
# land shifted from where the scene meshes expect them.               #
# ------------------------------------------------------------------- #
def load_cluster_center(clustering_json: Path, cluster_id: int) -> Tuple[float, float]:
    with open(clustering_json) as f:
        d = json.load(f)
    for c in d["clusters"]:
        if int(c["cluster_id"]) == int(cluster_id):
            bbox = c["bbox"]
            return float(bbox["center_lon"]), float(bbox["center_lat"])
    raise KeyError(f"cluster_id {cluster_id} not found in {clustering_json}")


# ------------------------------------------------------------------- #
# Sample selection: a few BSs, mix of "floor" and "got-signal" UEs.    #
# ------------------------------------------------------------------- #
def select_pairs(csv_path: Path, bs_count: int, ue_per_bs: int, seed: int = 0) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    log.info(f"Loaded {len(df):,} rows from {csv_path}")
    log.info(f"  unique BSs: {df['bs_idx'].nunique()},  unique UEs: {df['ue_id'].nunique()}")
    log.info(f"  fraction sim_rssi==-140: {(df['sim_rssi']==-140.0).mean():.3f}")

    rng = np.random.default_rng(seed)
    floor_rate = df.groupby("bs_idx").apply(lambda g: (g["sim_rssi"] == -140.0).mean())
    # Take BSs spanning the floor-rate range so we cover both extremes.
    quantile_targets = np.linspace(0.05, 0.95, bs_count)
    bs_idx_pool = floor_rate.sort_values().index.values
    chosen_bs = [bs_idx_pool[int(q * (len(bs_idx_pool) - 1))] for q in quantile_targets]
    chosen_bs = list(dict.fromkeys(chosen_bs))[:bs_count]
    log.info(f"  selected BSs (varying floor rate): {chosen_bs}")
    log.info(f"  their floor rates: {[f'{floor_rate[b]:.2f}' for b in chosen_bs]}")

    picks = []
    half = max(1, ue_per_bs // 2)
    for bs in chosen_bs:
        sub = df[df["bs_idx"] == bs]
        floor = sub[sub["sim_rssi"] == -140.0]
        signal = sub[sub["sim_rssi"] > -120.0]
        n_floor = min(half, len(floor))
        n_signal = min(ue_per_bs - n_floor, len(signal))
        if n_floor:
            picks.append(floor.sample(n=n_floor, random_state=seed))
        if n_signal:
            picks.append(signal.sample(n=n_signal, random_state=seed))
    out = pd.concat(picks, ignore_index=True)
    log.info(f"  diagnostic sample: {len(out)} rows across {out['bs_idx'].nunique()} BSs")
    return out


# ------------------------------------------------------------------- #
# Scene inspection.                                                   #
# ------------------------------------------------------------------- #
def describe_scene(scene) -> None:
    n_obj = len(getattr(scene, "objects", {}) or {})
    log.info(f"  scene.objects count: {n_obj}")
    bbox = None
    try:
        bbox = scene.mi_scene.bbox()
        log.info(f"  scene mi_scene bbox: min={tuple(bbox.min)}, max={tuple(bbox.max)}")
    except Exception as e:
        log.info(f"  could not read scene bbox: {e}")
    return bbox


def position_inside(bbox, pos) -> bool:
    if bbox is None:
        return True
    try:
        mn, mx = bbox.min, bbox.max
        return all(mn[i] <= pos[i] <= mx[i] for i in range(3))
    except Exception:
        return True


# ------------------------------------------------------------------- #
# Paths inspection.                                                   #
# ------------------------------------------------------------------- #
def inspect_paths(paths, label: str, ue_pos_local, tx_pos_local, csv_rssi_per_ue) -> None:
    log.info(f"  --- inspecting paths ({label}) ---")
    if paths is None:
        log.warning("  paths is None")
        return

    # paths.a is a (real, imag) tuple of tensors with shape
    # [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths] when synthetic_array=False.
    a_real, a_imag = paths.a
    a_real_np = a_real.numpy()
    a_imag_np = a_imag.numpy()
    a_np = a_real_np + 1j * a_imag_np
    log.info(f"  paths.a shape (real): {a_real_np.shape}")
    log.info(f"  paths.a dtype:        {a_real_np.dtype}")
    if a_np.size == 0:
        log.warning(f"  *** ZERO-SIZED paths.a — Sionna returned NO paths "
                    f"for this batch under variant '{label}' ***")
    else:
        log.info(f"  paths.a |.|^2 global  "
                 f"min={np.abs(a_np).min()**2:.3e}  max={np.abs(a_np).max()**2:.3e}")

    # tau (delays) and interactions.
    try:
        tau_np = paths.tau.numpy()
        log.info(f"  paths.tau shape:      {tau_np.shape}")
        if tau_np.size > 0:
            log.info(f"  paths.tau range (us): "
                     f"[{np.nanmin(tau_np)*1e6:.3f}, {np.nanmax(tau_np)*1e6:.3f}]")
        else:
            log.info(f"  paths.tau is empty")
    except Exception as e:
        log.info(f"  paths.tau unavailable: {e}")

    interactions = None
    try:
        interactions = paths.interactions.numpy()
        # shape [max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
        log.info(f"  paths.interactions shape: {interactions.shape}")
    except Exception as e:
        log.info(f"  paths.interactions unavailable: {e}")

    valid = None
    try:
        valid = paths.valid.numpy()
        log.info(f"  paths.valid shape:    {valid.shape}    total valid={int(valid.sum())}/{valid.size}")
    except Exception as e:
        log.info(f"  paths.valid unavailable: {e}")

    # Per-(rx, tx) pair statistics.
    # Expecting axis layout (num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths).
    if a_np.ndim != 5:
        log.warning(f"  unexpected a_np.ndim={a_np.ndim}; expected 5. "
                    f"shape={a_np.shape}. Skipping per-pair breakdown.")
        return
    num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths = a_np.shape
    log.info(f"  per-pair breakdown: num_rx={num_rx}  num_rx_ant={num_rx_ant}  "
             f"num_tx={num_tx}  num_tx_ant={num_tx_ant}  num_paths={num_paths}")

    # Production reduction: sum |a|^2 over (rx_ant, tx_ant, paths) → shape [num_rx, num_tx].
    # Note: when num_paths == 0, np.sum over an axis of size 0 returns 0 — that is the
    # correct "no signal" value and exactly what triggers the -140 floor downstream.
    pg = np.sum(np.abs(a_np) ** 2, axis=(1, 3, 4))
    pg_db = 10.0 * np.log10(np.maximum(pg, 1e-30))
    rssi_db = np.maximum(43.0 + pg_db, -140.0)  # production tx_power_dbm default = 43

    # Same reduction but masking by paths.valid if available.
    pg_valid_db = pg_db.copy()
    if valid is not None and valid.shape[-1] == num_paths:
        # Broadcast valid over antenna dims to mask |a|^2 before sum.
        mask = valid.astype(bool)
        # Align shape: valid is [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths].
        if mask.shape == a_np.shape:
            a_masked = np.where(mask, a_np, 0.0)
        else:
            log.info(f"  valid shape {mask.shape} != a shape {a_np.shape}; skipping mask")
            a_masked = a_np
        pg_valid = np.sum(np.abs(a_masked) ** 2, axis=(1, 3, 4))
        pg_valid_db = 10.0 * np.log10(np.maximum(pg_valid, 1e-30))

    log.info(f"  per (rx,tx) summary:  pg_db min={pg_db.min():+7.2f}  max={pg_db.max():+7.2f}    "
             f"rssi_db min={rssi_db.min():+7.2f}  max={rssi_db.max():+7.2f}")

    # Per-row detail, with comparison to CSV's stored value.
    log.info(f"  per-row (UE_index, CSV_rssi, repro_rssi, pg_db, pg_valid_db, "
             f"n_valid_paths, dist_LoS/spec/diff/refr):")
    for rx_i in range(num_rx):
        for tx_i in range(num_tx):
            n_valid = (
                int(valid[rx_i, :, tx_i, :, :].sum())
                if valid is not None and valid.size > 0 else -1
            )
            # Interaction histogram for this (rx, tx) — sum across antenna dims.
            inter_summary = ""
            if interactions is not None and interactions.size > 0:
                # interactions shape [max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
                inter_pair = interactions[:, rx_i, :, tx_i, :, :]
                first_int = inter_pair[0]  # [num_rx_ant, num_tx_ant, num_paths]
                if valid is not None and valid.size > 0 and valid.shape[-1] == first_int.shape[-1]:
                    mask = valid[rx_i, :, tx_i, :, :].astype(bool)  # same shape
                    first_int = first_int[mask]
                else:
                    first_int = first_int.ravel()
                counts = {k: int((first_int == v).sum()) for k, v in [
                    ("LoS", 0), ("spec", 1), ("diff", 2), ("refr", 4),
                ]}
                inter_summary = f"  first-int counts={counts}"
            csv_v = csv_rssi_per_ue[rx_i] if rx_i < len(csv_rssi_per_ue) else float("nan")
            log.info(f"    rx={rx_i:2d} tx={tx_i:2d}  csv={csv_v:+7.2f}  "
                     f"repro={rssi_db[rx_i, tx_i]:+7.2f}  "
                     f"pg_db={pg_db[rx_i, tx_i]:+8.2f}  "
                     f"pg_valid_db={pg_valid_db[rx_i, tx_i]:+8.2f}  "
                     f"n_valid={n_valid}{inter_summary}")


# ------------------------------------------------------------------- #
# Main run loop: per BS, per variant.                                 #
# ------------------------------------------------------------------- #
SOLVER_VARIANTS = {
    "A_production": dict(),  # use OPTIMIZED_PARAMS as-is
    "B_specular_on": dict(specular_reflection=True),
    "C_depth5": dict(max_depth=5),
    "D_specular_depth5": dict(specular_reflection=True, max_depth=5),
}


def run_solver(scene, base_params: dict, overrides: dict):
    from sionna.rt.path_solvers import PathSolver
    p = dict(base_params)
    p.update(overrides)
    solver = PathSolver()
    args = dict(
        scene=scene,
        max_depth=p["max_depth"],
        max_num_paths_per_src=p["max_num_paths_per_src"],
        samples_per_src=p["samples_per_src"],
        synthetic_array=p["synthetic_array"],
        los=p["los"],
        specular_reflection=p["specular_reflection"],
        diffuse_reflection=p["diffuse_reflection"],
        refraction=p["refraction"],
    )
    log.info(f"  solver args: max_depth={args['max_depth']}, "
             f"specular={args['specular_reflection']}, "
             f"diffuse={args['diffuse_reflection']}, "
             f"refraction={args['refraction']}, "
             f"los={args['los']}, samples_per_src={args['samples_per_src']}")
    return solver(**args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="Path to a B'' CSV, e.g. .../cluster_0/scene_0.csv")
    ap.add_argument("--scene", required=True,
                    help="Path to the cluster scene XML, "
                         "e.g. /data/ararat/rome_sionna/scene_data/new_xmls/cluster_0_scene.xml")
    ap.add_argument("--clustering-json",
                    default="/data/ararat/rome_sionna/bs_data/bs_clustering.json",
                    help="Path to bs_clustering.json (provides per-cluster bbox center)")
    ap.add_argument("--cluster-id", type=int, default=None,
                    help="Cluster id whose bbox center to use as scene origin. "
                         "Defaults to auto-detect from CSV's cluster_id column.")
    ap.add_argument("--bs-count", type=int, default=4,
                    help="number of BSs to test (across the floor-rate range)")
    ap.add_argument("--ue-per-bs", type=int, default=6,
                    help="UEs per BS (half floor, half got-signal)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", default="A_production,B_specular_on,C_depth5,D_specular_depth5",
                    help="comma-separated variant keys to run")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in SOLVER_VARIANTS:
            raise SystemExit(f"unknown variant '{v}'; known: {list(SOLVER_VARIANTS)}")

    log.info("=" * 72)
    log.info("Sionna paths diagnostic")
    log.info(f"  csv:      {args.csv}")
    log.info(f"  scene:    {args.scene}")
    log.info(f"  bs_count: {args.bs_count}  ue_per_bs: {args.ue_per_bs}")
    log.info(f"  variants: {variants}")
    log.info(f"  OPTIMIZED_PARAMS: {OPTIMIZED_PARAMS}")
    log.info("=" * 72)

    pairs = select_pairs(Path(args.csv), args.bs_count, args.ue_per_bs, args.seed)
    pairs.to_csv("/tmp/diag_paths_sample.csv", index=False)
    log.info(f"  wrote selected pairs to /tmp/diag_paths_sample.csv")

    # Resolve which cluster center to use as the local-frame origin.
    if args.cluster_id is None:
        unique_cids = pairs["cluster_id"].unique() if "cluster_id" in pairs.columns else []
        if len(unique_cids) != 1:
            raise SystemExit(
                f"--cluster-id not given and CSV has cluster_ids={list(unique_cids)}; "
                f"pass --cluster-id explicitly."
            )
        cluster_id = int(unique_cids[0])
        log.info(f"  auto-detected cluster_id={cluster_id} from CSV")
    else:
        cluster_id = int(args.cluster_id)
    center_lon, center_lat = load_cluster_center(Path(args.clustering_json), cluster_id)
    log.info(f"  cluster {cluster_id} center: lat={center_lat:.6f}, lon={center_lon:.6f}")
    log.info(f"  (simulation.py default would have been: lat=41.869854, lon=12.462249 — "
             f"offset {(center_lat-41.8698541)*111e3:+.1f} m N, "
             f"{(center_lon-12.4622493)*111e3*np.cos(np.radians(center_lat)):+.1f} m E)")

    from sionna.rt import load_scene
    scene = load_scene(args.scene)
    log.info(f"Loaded scene from {args.scene}")
    setup_scene_arrays(scene, OPTIMIZED_PARAMS)
    bbox = describe_scene(scene)

    grouped = pairs.groupby("bs_idx")
    for bs_idx, bs_group in grouped:
        bs_group = bs_group.reset_index(drop=True)
        first = bs_group.iloc[0]
        log.info("-" * 72)
        log.info(f"BS {int(bs_idx)}: lat={first.bs_lat:.6f} lon={first.bs_lon:.6f} "
                 f"alt={first.bs_alt:.2f} az={first.bs_azimuth:.2f} "
                 f"power_dbm_csv={first.bs_power_dbm:.2f}")

        bs_map = {
            f"{first.bs_lat:.6f}_{first.bs_lon:.6f}": (
                int(bs_idx), f"tx_{int(bs_idx)}", float(first.bs_lat), float(first.bs_lon),
            )
        }
        bs_params = {int(bs_idx): {
            "altitude": float(first.bs_alt),
            "azimuth": float(first.bs_azimuth),
            # NOTE: production add_transmitters_receivers defaults power_dbm=43 when not set.
            # We mirror that exactly to reproduce production rssi.
        }}

        rx_map = {}
        for idx, r in enumerate(bs_group.itertuples(index=False)):
            rx_map[f"ue_{idx}"] = (idx, f"rx_{idx}", float(r.ue_lat), float(r.ue_lon))

        # Per-pair geometry: positions in local coords + distance + inside-bbox check.
        # Pass the cluster-specific center so positions land in the same local frame
        # the scene meshes were built in.
        tx_local = lonlat_to_local(
            float(first.bs_lon), float(first.bs_lat),
            center_lon=center_lon, center_lat=center_lat,
        )
        tx_pos3 = (tx_local[0], tx_local[1], float(first.bs_alt))
        in_bbox_tx = position_inside(bbox, tx_pos3)
        log.info(f"  TX local pos: x={tx_local[0]:+8.2f} y={tx_local[1]:+8.2f} z={first.bs_alt:.2f}  "
                 f"inside_bbox={in_bbox_tx}")
        ue_pos_local = []
        for idx, r in enumerate(bs_group.itertuples(index=False)):
            ue_local = lonlat_to_local(
                float(r.ue_lon), float(r.ue_lat),
                center_lon=center_lon, center_lat=center_lat,
            )
            ue_pos_local.append(ue_local)
            inside = position_inside(bbox, (ue_local[0], ue_local[1], 1.0))
            log.info(f"  RX[{idx}] (ue_id={int(r.ue_id):6d}) local x={ue_local[0]:+8.2f} "
                     f"y={ue_local[1]:+8.2f} z=1.00  d={r.distance_m:8.2f}m  "
                     f"csv_rssi={r.sim_rssi:+7.2f}  inside_bbox={inside}")

        csv_rssi_per_ue = bs_group["sim_rssi"].to_numpy()

        # Run each variant on the same scene state.
        for vkey in variants:
            log.info(f"")
            log.info(f"  ### variant {vkey} ###")
            clear_scene(scene)
            add_transmitters_receivers(
                scene, bs_map, rx_map, OPTIMIZED_PARAMS, bs_params,
                center_lon=center_lon, center_lat=center_lat,
            )
            try:
                paths = run_solver(scene, OPTIMIZED_PARAMS, SOLVER_VARIANTS[vkey])
            except Exception as e:
                log.error(f"  solver failed for variant {vkey}: {e}", exc_info=True)
                continue
            try:
                inspect_paths(paths, vkey, ue_pos_local, tx_local, csv_rssi_per_ue)
            except Exception as e:
                log.error(f"  inspect_paths failed for variant {vkey}: {e}", exc_info=True)
                continue

    log.info("=" * 72)
    log.info("Done.")


if __name__ == "__main__":
    main()
