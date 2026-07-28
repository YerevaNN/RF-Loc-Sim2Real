#!/usr/bin/env python3
"""
Side-by-side 2D view of ONE patch under both BS-placement variants:
constrained (rooftop) vs unconstrained (free). Shows building footprints +
synthetic BS (green=on-roof, red=off) + street UEs, in the patch local frame.

Placement is generated fresh with seed 0 (which reproduces the real run, since
each variant's shard processes this patch with a fresh RNG when it's the
shard's first patch) — so this works even before the unconstrained run executes.

Run from the repo with simulation/ on the path:
  PYTHONPATH=simulation python scripts/viz_patch_compare.py \
      --scene_cache /mnt/weka/amanukyan/rome_c/scene_cache \
      --road_cache /mnt/weka/amanukyan/wair_d_iot/road_cache \
      --patch_idx 30 --out patch30_compare.png
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from shapely.geometry import Polygon
from shapely.ops import unary_union

from src import bs_placement
from src.ue_sampling import fetch_road_network, sample_on_streets, sample_open_space
from src.simulation import lonlat_to_local


def _area(scene_cache, pid):
    for d in glob.glob(os.path.join(scene_cache, "*")):
        if os.path.exists(os.path.join(d, f"patch_{pid:03d}", "buildings.json")):
            return d
    if os.path.exists(os.path.join(scene_cache, f"patch_{pid:03d}", "buildings.json")):
        return scene_cache
    raise SystemExit(f"no buildings.json for patch {pid} under {scene_cache}")


def _footprints(buildings):
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
    return unary_union(polys) if polys else unary_union([])


def _sample_ues(buildings, bs_list, n, road_cache, rng):
    clat, clon = buildings["center_lat"], buildings["center_lon"]
    lat0, lat1 = buildings["lat_range"]; lon0, lon1 = buildings["lon_range"]
    foot = _footprints(buildings)
    bs_xy = [lonlat_to_local(b["lon"], b["lat"], center_lon=clon, center_lat=clat)
             for b in bs_list]
    tag = f"{lat0:.5f}_{lon0:.5f}_{lat1:.5f}_{lon1:.5f}"
    cached = os.path.exists(os.path.join(road_cache, f"roads_{tag}.pkl"))
    roads = fetch_road_network(dict(north=lat1, south=lat0, east=lon1, west=lon0),
                               clon, clat, road_cache, lonlat_to_local) if cached else None
    if roads is not None:
        return sample_on_streets(roads, foot, bs_xy, n, max_dist_m=250,
                                 sidewalk_jitter_m=4, rng=rng)
    return sample_open_space(foot, bs_xy, n, max_dist_m=250, rng=rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_cache", required=True)
    ap.add_argument("--road_cache", required=True)
    ap.add_argument("--patch_idx", type=int, default=30)
    ap.add_argument("--patterns_json", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "bs_patterns.json"))
    ap.add_argument("--n_ues", type=int, default=3000, help="UEs to draw (viz sample)")
    ap.add_argument("--bs_per_patch", type=int, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    area = _area(args.scene_cache, args.patch_idx)
    buildings = json.load(open(os.path.join(area, f"patch_{args.patch_idx:03d}", "buildings.json")))
    clat, clon = buildings["center_lat"], buildings["center_lon"]
    patterns = bs_placement.load_patterns(args.patterns_json)

    fig, axes = plt.subplots(1, 2, figsize=(22, 11), sharex=True, sharey=True)
    for ax, variant in zip(axes, ("constrained", "unconstrained")):
        rng = np.random.default_rng(0)
        bs = bs_placement.place_bs(variant, buildings, patterns, args.bs_per_patch, rng)
        ue_xy = _sample_ues(buildings, bs, args.n_ues, args.road_cache, rng)

        for b in buildings["buildings"]:
            ax.add_patch(MplPoly(b["exterior_local"], closed=True, facecolor="0.82",
                                 edgecolor="0.6", lw=0.2, zorder=1))
            for ring in b.get("interiors_local", []):
                ax.add_patch(MplPoly(ring, closed=True, facecolor="white",
                                     edgecolor="0.6", lw=0.2, zorder=2))
        if len(ue_xy):
            ax.scatter(ue_xy[:, 0], ue_xy[:, 1], s=4, c="tab:blue", alpha=0.45,
                       zorder=3, label=f"UE (n={len(ue_xy)})")
        on = 0
        for b in bs:
            x, y = lonlat_to_local(b["lon"], b["lat"], center_lon=clon, center_lat=clat)
            roof = bool(b["on_roof"]); on += roof
            ax.scatter([x], [y], s=80 + 4 * float(b["altitude"]), marker="^",
                       c=("tab:green" if roof else "tab:red"),
                       edgecolors="black", linewidths=0.8, zorder=5)
        ax.set_title(f"patch {args.patch_idx} — {variant}\n{len(bs)} BS "
                     f"({on} on-roof), {len(ue_xy)} UEs, {len(buildings['buildings'])} bldgs",
                     fontsize=26)
        ax.set_aspect("equal")
        ax.set_xlabel("local x (m)", fontsize=22); ax.set_ylabel("local y (m)", fontsize=22)
        ax.tick_params(axis="both", labelsize=20)
        ax.legend(loc="upper right", fontsize=22, markerscale=4)

    out = args.out or f"patch{args.patch_idx:03d}_compare.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
