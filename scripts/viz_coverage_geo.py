#!/usr/bin/env python3
"""
Geographic coverage check: plot the ACTUAL OSM building footprints (from each
cached scene's buildings.json) in lon/lat, with the patch grid overlaid and
built/empty cells shaded. This shows whether the empty patches are genuinely
building-free (coherent city outline + river/edges) rather than a detection bug.

No basemap dependency — the building outlines themselves are the map. (A street
tile basemap would need `contextily` + network.)

Usage:
  python scripts/viz_coverage_geo.py --scene_cache /mnt/weka/amanukyan/rome_c/scene_cache \
      --out c_coverage_geo.png
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
from pyproj import Transformer
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle

_FWD = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
_INV = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)


def _resolve_area(path):
    if glob.glob(os.path.join(path, "patch_*")):
        return path
    areas = [d for d in glob.glob(os.path.join(path, "*"))
             if glob.glob(os.path.join(d, "patch_*"))]
    return max(areas, key=lambda d: len(glob.glob(os.path.join(d, "patch_*"))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_cache", required=True)
    ap.add_argument("--out", default="c_coverage_geo.png")
    ap.add_argument("--max_buildings", type=int, default=120000, help="cap polygons plotted")
    args = ap.parse_args()
    area = _resolve_area(args.scene_cache)

    polys = []          # building footprints in lon/lat
    built = empty = 0
    fig, ax = plt.subplots(figsize=(13, 13))
    for f in sorted(glob.glob(os.path.join(area, "patch_*", "buildings.json"))):
        d = json.load(open(f))
        pid = int(os.path.basename(os.path.dirname(f)).split("_")[1])
        lat0, lat1 = d["lat_range"]; lon0, lon1 = d["lon_range"]
        n = d["num_buildings"]
        is_built = n > 0
        built += is_built; empty += (not is_built)
        # patch cell shading + label
        ax.add_patch(Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0,
                               facecolor=("#caffca" if is_built else "#f0d0d0"),
                               edgecolor="0.6", lw=0.5, alpha=0.35, zorder=1))
        ax.text((lon0 + lon1) / 2, (lat0 + lat1) / 2, f"{pid}\n{n}",
                ha="center", va="center", fontsize=9, fontweight="bold",
                color=("0.10" if is_built else "0.5"), zorder=4)
        # footprints local->lonlat (batched per patch); cap polygons but keep
        # drawing every grid cell so the empty south still renders.
        if len(polys) < args.max_buildings:
            cx, cy = _FWD.transform(d["center_lon"], d["center_lat"])
            for b in d.get("buildings", []):
                ext = np.asarray(b["exterior_local"], float)
                if len(ext) < 3:
                    continue
                lon, lat = _INV.transform(ext[:, 0] + cx, ext[:, 1] + cy)
                polys.append(np.column_stack([lon, lat]))

    ax.add_collection(PolyCollection(polys, facecolors="0.35", edgecolors="none", zorder=3))
    ax.autoscale_view()
    ax.set_aspect(1.0 / np.cos(np.radians(41.84)))  # lon/lat aspect at Rome
    ax.set_xlabel("longitude", fontsize=22); ax.set_ylabel("latitude", fontsize=22)
    ax.tick_params(axis="both", labelsize=18)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}  ({built} built / {empty} empty, {len(polys)} footprints)")


if __name__ == "__main__":
    main()
