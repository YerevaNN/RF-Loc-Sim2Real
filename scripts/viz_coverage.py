#!/usr/bin/env python3
"""
Dataset C patch-grid coverage map: which of the 144 patches actually have
buildings (and how many), vs the empty rural ones that get skipped.

Reads num_buildings from each cached scene's buildings.json (written by
patch_scene_manager for every patch, built or empty), and lays them out on the
grid (north up). Built patches are coloured by building count; empty ones grey.

Usage:
  python scripts/viz_coverage.py --scene_cache /mnt/weka/amanukyan/rome_c/scene_cache \
      --rows 12 --cols 12 --out c_coverage.png
(If --scene_cache is the parent dir, the area with the most patches is picked.)
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from matplotlib import pyplot as plt


def _resolve_area(path: str) -> str:
    if glob.glob(os.path.join(path, "patch_*")):
        return path
    areas = [d for d in glob.glob(os.path.join(path, "*"))
             if os.path.isdir(d) and glob.glob(os.path.join(d, "patch_*"))]
    if not areas:
        raise SystemExit(f"no patch_* dirs under {path}")
    return max(areas, key=lambda d: len(glob.glob(os.path.join(d, "patch_*"))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene_cache", required=True)
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--cols", type=int, default=12)
    ap.add_argument("--out", default="c_coverage.png")
    args = ap.parse_args()

    area = _resolve_area(args.scene_cache)
    grid = np.full((args.rows, args.cols), np.nan)   # NaN = scene not built yet
    nb = {}
    for f in glob.glob(os.path.join(area, "patch_*", "buildings.json")):
        pid = int(os.path.basename(os.path.dirname(f)).split("_")[1])
        try:
            n = int(json.load(open(f))["num_buildings"])
        except Exception:
            continue
        nb[pid] = n
        r, c = divmod(pid, args.cols)
        if r < args.rows and c < args.cols:
            grid[r, c] = n

    built = sum(1 for n in nb.values() if n > 0)
    total = args.rows * args.cols

    # 0 -> "bad" (grey); NaN (not built) -> white; >0 -> viridis by count
    masked = np.ma.masked_where(~(grid > 0), grid)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("0.85")
    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(masked, cmap=cmap, origin="upper")
    for r in range(args.rows):
        for c in range(args.cols):
            pid = r * args.cols + c
            n = nb.get(pid)
            label = f"{pid}\n{n}" if n is not None else f"{pid}\n·"
            built_cell = (n is not None and n > 0)
            ax.text(c, r, label, ha="center", va="center", fontsize=6,
                    color="white" if built_cell else "0.45")
    ax.set_title(f"Dataset C coverage (north↑, west←):  {built}/{total} patches built "
                 f"(grey = no buildings / skipped)\ncell = patch_idx / num_buildings")
    ax.set_xticks(range(args.cols)); ax.set_yticks(range(args.rows))
    ax.set_xlabel("grid col (W→E)"); ax.set_ylabel("grid row (N→S)")
    fig.colorbar(im, ax=ax, label="buildings in patch", fraction=0.046, pad=0.04)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}  ({built}/{total} built, {total-built} empty)")


if __name__ == "__main__":
    main()
