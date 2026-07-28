#!/usr/bin/env python3
"""
Visual sanity check for one Dataset C patch: overlay building footprints,
synthetic BS (colored by on_roof, sized by altitude), and street UEs, in the
patch local-meter frame. Saves a PNG.

This is the eyeball test for "are the scenes/placements correct" — you should
see BS markers sitting ON building footprints (constrained) and UE dots lying
along the street gaps BETWEEN buildings, never inside them.

Usage:
  python scripts/viz_patch.py --scene_cache /path/c_scene_cache \
      --pairs_csv /path/out/pairs_patches.csv --patch_idx 0 --out patch0.png

Needs only matplotlib + pandas + shapely (no Sionna). For a true 3-D check of
TX height on the roof, load the patch scene in Sionna and use scene.render(...).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

# import lonlat_to_local from the simulation package (simulation/ on path)
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simulation"))
from src.simulation import lonlat_to_local  # noqa: E402


def _find_buildings_json(scene_cache: str, patch_idx: int) -> str:
    hits = glob.glob(os.path.join(scene_cache, "*", f"patch_{patch_idx:03d}", "buildings.json"))
    if not hits:
        raise FileNotFoundError(
            f"no buildings.json for patch {patch_idx} under {scene_cache}")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene_cache", required=True)
    ap.add_argument("--pairs_csv", required=True)
    ap.add_argument("--patch_idx", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_ues", type=int, default=4000, help="cap UE dots plotted")
    args = ap.parse_args()

    bjpath = _find_buildings_json(args.scene_cache, args.patch_idx)
    bj = json.load(open(bjpath))
    clat, clon = bj["center_lat"], bj["center_lon"]

    df = pd.read_csv(args.pairs_csv)
    df = df[df["patch_idx"] == args.patch_idx]
    if df.empty:
        raise SystemExit(f"no rows for patch {args.patch_idx} in {args.pairs_csv}")
    bs = df.drop_duplicates("bs_idx")
    ue = df.drop_duplicates("ue_idx")
    if len(ue) > args.max_ues:
        ue = ue.sample(args.max_ues, random_state=0)

    fig, ax = plt.subplots(figsize=(11, 11))

    # building footprints (exterior; holes drawn as white)
    for b in bj["buildings"]:
        ax.add_patch(MplPolygon(b["exterior_local"], closed=True,
                                facecolor="0.80", edgecolor="0.55", linewidth=0.3, zorder=1))
        for ring in b.get("interiors_local", []):
            ax.add_patch(MplPolygon(ring, closed=True, facecolor="white",
                                    edgecolor="0.55", linewidth=0.3, zorder=2))

    # UEs (street)
    ue_xy = [lonlat_to_local(r.ue_lon, r.ue_lat, center_lon=clon, center_lat=clat)
             for r in ue.itertuples()]
    if ue_xy:
        ax.scatter([p[0] for p in ue_xy], [p[1] for p in ue_xy],
                   s=3, c="tab:blue", alpha=0.5, zorder=3, label=f"UE (n={len(ue)})")

    # BS, colored by on_roof, sized by altitude
    for r in bs.itertuples():
        x, y = lonlat_to_local(r.bs_lon, r.bs_lat, center_lon=clon, center_lat=clat)
        on_roof = bool(getattr(r, "bs_on_roof", False))
        ax.scatter([x], [y], s=40 + 3 * float(r.bs_alt),
                   marker="^", c=("tab:green" if on_roof else "tab:red"),
                   edgecolors="black", linewidths=0.6, zorder=5)
        ax.annotate(f"{r.bs_alt:.0f}m", (x, y), fontsize=7, zorder=6,
                    xytext=(3, 3), textcoords="offset points")

    on = int(bs["bs_on_roof"].sum()) if "bs_on_roof" in bs else 0
    variant = bs["bs_variant"].iloc[0] if "bs_variant" in bs else "?"
    ax.set_title(f"Patch {args.patch_idx} | variant={variant} | "
                 f"{len(bj['buildings'])} buildings | "
                 f"{len(bs)} BS ({on} on-roof) | {len(df)} pairs\n"
                 f"green=on-roof BS, red=off-roof, blue=UE  ({os.path.basename(bjpath)})")
    ax.set_xlabel("local x (m)"); ax.set_ylabel("local y (m)")
    ax.set_aspect("equal"); ax.legend(loc="upper right")

    out = args.out or f"patch_{args.patch_idx:03d}_viz.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}  ({len(bj['buildings'])} buildings, {len(bs)} BS, {len(ue)} UE)")


if __name__ == "__main__":
    main()
