#!/usr/bin/env python3
"""
Extract synthetic-BS placement priors from the B'' base-station optimization
results (constrained vs unconstrained), for Dataset C.

Background
----------
B'' optimized REAL base stations against measured RSSI. Two objective variants
were run per BS:
  * unconstrained  -> RMSE only            (dir: ..._rmse_only)
  * constrained    -> RMSE + P_xy + P_roof (dir: ..._penalized_New_penalty)
where P_xy penalises a BS that drifts off a building footprint and P_roof
penalises altitudes away from roof height. The net effect (verified): the
constrained runs sit lower (~roof height), the unconstrained runs float higher.

Dataset C synthesises BS from scratch, so it cannot re-run that optimization.
Instead we fit the EMPIRICAL marginal distributions of the optimized
parameters (altitude, power, azimuth) per variant and hand them to the C
placer (simulation/src/bs_placement.py) as priors, so synthetic BS reproduce
the real const/unconst statistics rather than arbitrary uniforms.

Output
------
A single JSON (default: configs/bs_patterns.json) holding, per variant, summary
stats + raw samples for altitude / power_dbm / azimuth (+ offset & penalty
diagnostics for documentation). Raw samples are kept so the placer can do
empirical resampling without re-reading 300+ files.

This is an OFFLINE, run-once analysis. It does not import sionna/tensorflow.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np

# Defaults: the on-disk B'' optimization result trees (see project memory).
DEFAULT_UNCONSTRAINED = (
    "/mnt/weka/amanukyan/wair_d_iot/"
    "optimization_results_filter_nan_only_rmse_only"
)
DEFAULT_CONSTRAINED = (
    "/mnt/weka/amanukyan/wair_d_iot/"
    "optimization_results_filter_nan_only_penalized_New_penalty"
)


def _finite(values: List[float]) -> np.ndarray:
    a = np.asarray([v for v in values if v is not None], dtype=float)
    return a[np.isfinite(a)]


def _summary(values: List[float]) -> Optional[Dict]:
    """Empirical summary + raw samples for a 1-D quantity (None if no data)."""
    a = _finite(values)
    if a.size == 0:
        return None
    pct = {f"p{p}": float(np.percentile(a, p)) for p in (5, 25, 50, 75, 95)}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        **pct,
        "samples": [float(v) for v in a],
    }


def _load_variant(results_dir: str) -> Dict[str, List[float]]:
    """Collect per-BS optimized quantities from a results tree."""
    cols = {k: [] for k in
            ("altitude", "power_dbm", "azimuth", "offset_m",
             "P_roof", "P_xy", "best_rmse")}
    files = sorted(glob.glob(os.path.join(results_dir, "bs_*", "optimization_results.json")))
    if not files:
        raise FileNotFoundError(f"no optimization_results.json under {results_dir}")
    used = 0
    for fp in files:
        try:
            with open(fp) as f:
                d = json.load(f)
        except Exception:
            continue
        bp = d.get("best_parameters") or {}
        if not bp:
            continue
        used += 1
        cols["altitude"].append(bp.get("altitude"))
        cols["power_dbm"].append(bp.get("power_dbm"))
        cols["azimuth"].append(bp.get("azimuth"))
        xo, yo = bp.get("x_offset"), bp.get("y_offset")
        cols["offset_m"].append(float(np.hypot(xo, yo)) if xo is not None and yo is not None else None)
        cols["P_roof"].append(d.get("P_roof_at_best"))
        cols["P_xy"].append(d.get("P_xy_at_best"))
        cols["best_rmse"].append(d.get("best_rmse"))
    print(f"  {results_dir}\n    files={len(files)} used={used}")
    return cols


def _variant_block(results_dir: str) -> Dict:
    cols = _load_variant(results_dir)
    block = {
        "source_dir": results_dir,
        # priors the placer samples from:
        "altitude": _summary(cols["altitude"]),
        "power_dbm": _summary(cols["power_dbm"]),
        "azimuth": _summary(cols["azimuth"]),
        # documentation / sanity only:
        "diagnostics": {
            "offset_m": _summary(cols["offset_m"]),
            "P_roof": _summary(cols["P_roof"]),
            "P_xy": _summary(cols["P_xy"]),
            "best_rmse": _summary(cols["best_rmse"]),
        },
    }
    return block


def _fmt(s: Optional[Dict]) -> str:
    if not s:
        return "n/a"
    return (f"n={s['n']:3d}  mean={s['mean']:6.1f}  std={s['std']:5.1f}  "
            f"[{s['min']:6.1f} .. p50={s['p50']:6.1f} .. {s['max']:6.1f}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unconstrained_dir", default=DEFAULT_UNCONSTRAINED)
    ap.add_argument("--constrained_dir", default=DEFAULT_CONSTRAINED)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "bs_patterns.json"))
    args = ap.parse_args()

    print("Extracting BS placement priors:")
    variants = {
        "unconstrained": _variant_block(args.unconstrained_dir),
        "constrained": _variant_block(args.constrained_dir),
    }

    out = {
        "description": "Empirical BS placement priors from B'' const/unconst "
                       "optimization results; consumed by Dataset C BS placer.",
        "variants": variants,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # Human-readable characterisation (the research artifact).
    print("\n" + "=" * 78)
    print("BS PLACEMENT PATTERN CHARACTERISATION  (const vs unconst)")
    print("=" * 78)
    for q in ("altitude", "power_dbm", "azimuth"):
        print(f"\n{q}:")
        print(f"  unconstrained : {_fmt(variants['unconstrained'][q])}")
        print(f"  constrained   : {_fmt(variants['constrained'][q])}")
    print("\nconstrained P_roof (rooftop-penalty activity):")
    print(f"  {_fmt(variants['constrained']['diagnostics']['P_roof'])}")
    print(f"\nWrote priors -> {args.out}")


if __name__ == "__main__":
    main()
