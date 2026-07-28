"""
Synthetic base-station placement for Dataset C.

Two variants, mirroring the B'' constrained-vs-unconstrained optimization:

  * constrained   — BS sits ON A ROOFTOP. XY is a point provably inside a
                    building footprint (courtyards excluded); Z is the building
                    roof height + a small mast offset, clipped to the empirical
                    constrained altitude band. This realises P_xy (on-building)
                    and P_roof (roof-height) as *construction* constraints.
  * unconstrained — BS position is free within the patch; altitude is drawn
                    from the empirical unconstrained altitude distribution.

Altitude / power / azimuth priors come from configs/bs_patterns.json, fitted
to the real B'' optimization results (see scripts/extract_bs_patterns.py).

Frame note: building footprints in buildings.json are in the scene's LOCAL
METER frame (origin = patch center). A rooftop BS is computed there and mapped
back to lon/lat with simulation.local_to_lonlat — the exact inverse of the
transform the simulator uses to place the transmitter, so there is no drift
between "where we found the roof" and "where Sionna puts the TX".

Mesh convention (scene_generator): building base at z=0, roof slab at
z=height. Hence altitude == height places the TX on the roof; we add a small
mast offset so it rests just above the slab.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

import numpy as np
from shapely.geometry import Polygon, Point

try:  # package context (normal import)
    from .simulation import local_to_lonlat
except ImportError:  # pragma: no cover - direct execution fallback
    from simulation import local_to_lonlat

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Priors
# --------------------------------------------------------------------------- #
def load_patterns(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def _resample(prior: Dict, rng: np.random.Generator) -> float:
    """Draw one value by empirical resampling of a prior's stored samples."""
    return float(rng.choice(prior["samples"]))


# --------------------------------------------------------------------------- #
# Footprint geometry
# --------------------------------------------------------------------------- #
def _make_polygon(exterior: List, interiors: Optional[List]) -> Optional[Polygon]:
    """Build a (cleaned) shapely Polygon with courtyard holes, or None."""
    try:
        poly = Polygon(exterior, holes=interiors or None)
        if not poly.is_valid:
            poly = poly.buffer(0)  # repair self-intersections
        if poly.is_empty or poly.geom_type != "Polygon" or poly.area <= 0.0:
            # buffer(0) on a degenerate ring can yield a MultiPolygon/empty
            if poly.geom_type == "MultiPolygon" and not poly.is_empty:
                poly = max(poly.geoms, key=lambda g: g.area)
            else:
                return None
        return poly if poly.area > 0.0 else None
    except Exception:
        return None


def _random_point_in(poly: Polygon, rng: np.random.Generator,
                     max_tries: int = 40) -> Point:
    """A point GUARANTEED inside the (solid) polygon. Rejection-sample the bbox
    for variety; fall back to representative_point() (always interior, even for
    concave shapes) so the on-roof guarantee never depends on luck."""
    minx, miny, maxx, maxy = poly.bounds
    for _ in range(max_tries):
        p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if poly.contains(p):
            return p
    return poly.representative_point()


# --------------------------------------------------------------------------- #
# Placement
# --------------------------------------------------------------------------- #
def place_constrained(buildings: Dict, patterns: Dict, n_bs: int,
                      rng: np.random.Generator, *,
                      # was a fixed 3.0 m; now a per-BS rooftop-mast prior so
                      # antenna height above the roof varies realistically.
                      mast_offset_range: tuple = (3.0, 6.0),
                      weight_by_area: bool = True) -> List[Dict]:
    """Place n_bs synthetic BS on rooftops within a patch.

    buildings: parsed buildings.json (center_lat/lon + buildings[] with
               exterior_local / interiors_local / height).
    """
    center_lat = float(buildings["center_lat"])
    center_lon = float(buildings["center_lon"])

    polys, heights, ids, areas = [], [], [], []
    for b in buildings.get("buildings", []):
        poly = _make_polygon(b.get("exterior_local"), b.get("interiors_local"))
        if poly is None:
            continue
        polys.append(poly)
        heights.append(float(b.get("height", 10.0)))
        ids.append(b.get("id"))
        areas.append(poly.area)
    if not polys:
        raise ValueError("no valid building footprints in patch — cannot place "
                         "constrained BS (empty/failed buildings.json?)")

    cpat = patterns["variants"]["constrained"]
    alt_floor = float(cpat["altitude"]["p5"])
    alt_cap = float(cpat["altitude"]["p95"])

    n_poly = len(polys)
    p = (np.asarray(areas) / float(np.sum(areas))) if weight_by_area else None
    replace = n_bs > n_poly  # can't host more rooftop BS than buildings w/o reuse
    chosen = rng.choice(n_poly, size=n_bs, replace=replace, p=p)

    out: List[Dict] = []
    n_capped_below_roof = 0
    for i, bi in enumerate(chosen):
        poly, h = polys[int(bi)], heights[int(bi)]
        pt = _random_point_in(poly, rng)
        # Hard guarantee: XY is inside the footprint (on the roof slab).
        assert poly.contains(pt) or poly.touches(pt), "BS XY escaped footprint"

        mast = float(rng.uniform(*mast_offset_range))
        altitude = float(np.clip(h + mast, alt_floor, alt_cap))
        on_roof = altitude >= h  # False only if the cap pulled it below a tall roof
        if not on_roof:
            n_capped_below_roof += 1

        lon, lat = local_to_lonlat(pt.x, pt.y, center_lon=center_lon, center_lat=center_lat)
        out.append({
            "bs_idx": i,
            "lat": float(lat),
            "lon": float(lon),
            "altitude": altitude,
            "azimuth": _resample(cpat["azimuth"], rng),
            "power_dbm": _resample(cpat["power_dbm"], rng),
            "variant": "constrained",
            "on_roof": bool(on_roof),
            "building_id": ids[int(bi)],
            "roof_height_m": float(h),
            "mast_offset_m": mast,
        })

    if n_capped_below_roof:
        logger.warning("  %d/%d constrained BS capped below their roof "
                       "(building taller than alt_cap=%.1fm)",
                       n_capped_below_roof, n_bs, alt_cap)
    return out


def place_unconstrained(buildings: Dict, patterns: Dict, n_bs: int,
                        rng: np.random.Generator) -> List[Dict]:
    """Place n_bs synthetic BS freely within the patch bbox, with altitude /
    power / azimuth drawn from the unconstrained empirical priors."""
    lat_min, lat_max = buildings["lat_range"]
    lon_min, lon_max = buildings["lon_range"]
    upat = patterns["variants"]["unconstrained"]

    out: List[Dict] = []
    for i in range(n_bs):
        out.append({
            "bs_idx": i,
            "lat": float(rng.uniform(lat_min, lat_max)),
            "lon": float(rng.uniform(lon_min, lon_max)),
            "altitude": _resample(upat["altitude"], rng),
            "azimuth": _resample(upat["azimuth"], rng),
            "power_dbm": _resample(upat["power_dbm"], rng),
            "variant": "unconstrained",
            "on_roof": False,
            "building_id": None,
            "roof_height_m": None,
        })
    return out


def place_bs(variant: str, buildings: Dict, patterns: Dict, n_bs: int,
             rng: np.random.Generator, **kwargs) -> List[Dict]:
    """Dispatch on variant ('constrained' | 'unconstrained')."""
    if variant == "constrained":
        return place_constrained(buildings, patterns, n_bs, rng, **kwargs)
    if variant == "unconstrained":
        return place_unconstrained(buildings, patterns, n_bs, rng)
    raise ValueError(f"unknown BS variant: {variant!r}")


def on_roof_fraction(bs_list: List[Dict]) -> float:
    if not bs_list:
        return 0.0
    return sum(b["on_roof"] for b in bs_list) / len(bs_list)
