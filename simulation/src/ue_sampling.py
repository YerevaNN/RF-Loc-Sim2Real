"""
Street-aware, BS-proximity-constrained UE sampling for the fixed-UE
pre-training data generator.

Why this exists
---------------
The legacy sampler in pretraining_data_generator_fixed_ues.py drops UEs as a
2-D Gaussian over the cluster bbox at a fixed z=1.0 m, with no awareness of
buildings or streets. Empirically (cluster_42) ~11% of those UEs land *inside*
building footprints and are ~98% floor (sim_rssi<=-140) — pure waste — and the
remaining outdoor UEs are spread far from BSs (median ~430 m), driving a high
floor rate.

This module samples UEs:
  1. ON STREETS    — along the OSM road network (LineStrings), so points sit in
                     the open street canyon rather than inside buildings.
  2. NEAR BSs      — only on road segments within `max_dist_m` of some BS.
  3. NOT IN-BUILDING — any jittered point that still lands in a building footprint
                     is rejected (safety net).

All geometry is handled in the scene's LOCAL METER frame (same frame the
building meshes live in). The caller converts back to lat/lon via
simulation.local_to_lonlat.

If the OSM road fetch fails (Overpass flaky), `fetch_road_network` returns None
and the caller can fall back to `sample_open_space` (uniform in BS-buffer minus
buildings) — street-ish without the network dependency.
"""
from __future__ import annotations

import glob
import logging
import pickle
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import requests
from plyfile import PlyData
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point
from shapely.ops import unary_union
from shapely.prepared import prep

logger = logging.getLogger(__name__)

OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

# Drivable / walkable road classes we want UEs on. Excludes footways inside
# parks etc. by default; tune as needed.
HIGHWAY_VALUES = (
    "motorway|trunk|primary|secondary|tertiary|unclassified|residential|"
    "living_street|service|pedestrian|road"
)


# --------------------------------------------------------------------------- #
# 1. Building footprints (local frame) from the cluster's PLY meshes.         #
# --------------------------------------------------------------------------- #
def load_building_footprints(mesh_dir: str):
    """Return a (shapely) MultiPolygon of building footprints in the local
    meter frame, built as the convex hull of each building_*.ply's xy vertices.

    Convex hull slightly over-covers concave buildings — conservative for
    rejection (we'd rather reject a near-wall point than place a UE inside)."""
    polys = []
    files = glob.glob(f"{mesh_dir}/building_*.ply")
    for ply in files:
        try:
            v = PlyData.read(ply)["vertex"]
            xy = np.column_stack([np.asarray(v["x"]), np.asarray(v["y"])])
            if len(xy) >= 3:
                hull = MultiPoint(xy).convex_hull
                if hull.geom_type == "Polygon":
                    polys.append(hull)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"  skip {Path(ply).name}: {e}")
    if not polys:
        logger.warning(f"  no building footprints loaded from {mesh_dir}")
        return unary_union([])
    footprints = unary_union(polys)
    logger.info(f"  loaded {len(polys)} building footprints from {mesh_dir} "
                f"(area={footprints.area/1e6:.3f} km^2)")
    return footprints


# --------------------------------------------------------------------------- #
# 2. OSM road network (local frame). Mirrors fetch_osm_buildings_direct().    #
# --------------------------------------------------------------------------- #
def fetch_road_network(
    bounds: Dict,
    center_lon: float,
    center_lat: float,
    cache_dir: Optional[str],
    lonlat_to_local: Callable[..., tuple],
):
    """Fetch OSM `highway` ways inside bounds and return them as a shapely
    MultiLineString in the LOCAL meter frame. Cached per-bbox as a .pkl.

    bounds: dict(north, south, east, west) in degrees.
    lonlat_to_local: simulation.lonlat_to_local (passed in to avoid a circular
        import and to guarantee the same transform the rest of the pipeline uses).

    Returns None if every Overpass server fails (caller should fall back)."""
    north, south = bounds["north"], bounds["south"]
    east, west = bounds["east"], bounds["west"]

    cache_path = None
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        tag = f"{south:.5f}_{west:.5f}_{north:.5f}_{east:.5f}"
        cache_path = Path(cache_dir) / f"roads_{tag}.pkl"
        if cache_path.exists():
            logger.info(f"  loading roads from cache: {cache_path.name}")
            with open(cache_path, "rb") as f:
                lines_lonlat = pickle.load(f)
            return _lines_to_local(lines_lonlat, center_lon, center_lat, lonlat_to_local)

    # NOTE: a `["highway"~"^(...)$"]` regex query is rejected (HTTP 406) by some
    # Overpass servers (e.g. overpass-api.de). Fetch all highway ways with the
    # plain tag filter (widely accepted, faster) and filter values in Python.
    query = f"""
    [out:json][timeout:180];
    (
      way["highway"]({south},{west},{north},{east});
    );
    out body;
    >;
    out skel qt;
    """

    for server_idx, server_url in enumerate(OVERPASS_SERVERS):
        for attempt in range(3):
            try:
                resp = requests.get(server_url, params={"data": query}, timeout=180)
                resp.raise_for_status()
                data = resp.json()
                nodes = {}
                ways = {}
                for el in data["elements"]:
                    if el["type"] == "node":
                        nodes[el["id"]] = (el["lon"], el["lat"])  # lon, lat
                    elif el["type"] == "way":
                        ways[el["id"]] = el
                allowed = set(HIGHWAY_VALUES.split("|"))
                lines_lonlat = []
                for way in ways.values():
                    tags = way.get("tags", {})
                    if tags.get("highway") in allowed:
                        coords = [nodes[nid] for nid in way["nodes"] if nid in nodes]
                        if len(coords) >= 2:
                            lines_lonlat.append(coords)
                logger.info(f"  fetched {len(lines_lonlat)} road ways from Overpass "
                            f"(server {server_idx+1})")
                if cache_path:
                    with open(cache_path, "wb") as f:
                        pickle.dump(lines_lonlat, f)
                return _lines_to_local(lines_lonlat, center_lon, center_lat, lonlat_to_local)
            except requests.exceptions.Timeout:
                logger.warning(f"  roads: timeout server {server_idx+1} attempt {attempt+1}/3")
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                continue
            except requests.exceptions.HTTPError as e:
                code = getattr(e.response, "status_code", None)
                if code in (504,):
                    if attempt < 2:
                        time.sleep(5 * (attempt + 1))
                    continue
                if code == 429:
                    time.sleep(60)
                    continue
                logger.error(f"  roads: HTTP {code} server {server_idx+1}: {e}")
                break
            except Exception as e:  # noqa: BLE001
                logger.error(f"  roads: error server {server_idx+1}: {e}")
                break
        if server_idx < len(OVERPASS_SERVERS) - 1:
            time.sleep(2)

    logger.warning("  roads: all Overpass servers failed — returning None (use fallback)")
    return None


def _lines_to_local(lines_lonlat, center_lon, center_lat, lonlat_to_local):
    segs = []
    for coords in lines_lonlat:
        xy = [lonlat_to_local(lon, lat, center_lon=center_lon, center_lat=center_lat)
              for lon, lat in coords]
        if len(xy) >= 2:
            segs.append(LineString([(float(x), float(y)) for x, y in xy]))
    return MultiLineString(segs) if segs else None


# --------------------------------------------------------------------------- #
# 3. Samplers (return ndarray[n,2] of local x,y).                             #
# --------------------------------------------------------------------------- #
def sample_on_streets(
    roads_local,
    buildings_local,
    bs_xy,
    n: int,
    *,
    max_dist_m: float = 400.0,
    sidewalk_jitter_m: float = 4.0,
    rng: Optional[np.random.Generator] = None,
    max_tries_factor: int = 50,
):
    """Sample n points along road segments within max_dist_m of some BS,
    with lateral sidewalk jitter, rejecting any that land in a building."""
    rng = rng or np.random.default_rng(0)

    # CLOSENESS: keep only roads within max_dist_m of any BS.
    near = unary_union([Point(float(x), float(y)).buffer(max_dist_m) for x, y in bs_xy])
    roads = roads_local.intersection(near)
    segs = [g for g in getattr(roads, "geoms", [roads])
            if g.geom_type == "LineString" and g.length > 0]
    if not segs:
        raise ValueError(f"no road within {max_dist_m} m of any BS — increase max_dist_m")
    w = np.array([s.length for s in segs], dtype=float)
    w /= w.sum()

    pb = prep(buildings_local) if not buildings_local.is_empty else None
    out: List[tuple] = []
    tries = 0
    cap = n * max_tries_factor
    while len(out) < n and tries < cap:
        tries += 1
        s = segs[rng.choice(len(segs), p=w)]
        d = rng.uniform(0, s.length)
        p = s.interpolate(d)
        q = s.interpolate(min(d + 0.1, s.length))
        ang = np.arctan2(q.y - p.y, q.x - p.x) + np.pi / 2  # perpendicular
        j = rng.normal(0, sidewalk_jitter_m)
        x, y = p.x + j * np.cos(ang), p.y + j * np.sin(ang)
        if pb is not None and pb.contains(Point(x, y)):
            continue  # rejected: inside a building
        out.append((x, y))
    if len(out) < n:
        logger.warning(f"  sample_on_streets: only {len(out)}/{n} after {tries} tries")
    return np.array(out)


def sample_open_space(
    buildings_local,
    bs_xy,
    n: int,
    *,
    max_dist_m: float = 400.0,
    rng: Optional[np.random.Generator] = None,
    max_tries_factor: int = 50,
):
    """Fallback when OSM roads are unavailable: uniform points inside the union
    of BS buffers, rejecting any inside a building. Street-ish (open outdoor
    space near BSs) without the network call."""
    rng = rng or np.random.default_rng(0)
    bs_xy = np.asarray(bs_xy, dtype=float)
    pb = prep(buildings_local) if not buildings_local.is_empty else None
    out: List[tuple] = []
    tries = 0
    cap = n * max_tries_factor
    while len(out) < n and tries < cap:
        tries += 1
        cx, cy = bs_xy[rng.integers(len(bs_xy))]
        r = max_dist_m * np.sqrt(rng.uniform())          # uniform in disk
        th = rng.uniform(0, 2 * np.pi)
        x, y = cx + r * np.cos(th), cy + r * np.sin(th)
        if pb is not None and pb.contains(Point(x, y)):
            continue
        out.append((x, y))
    if len(out) < n:
        logger.warning(f"  sample_open_space: only {len(out)}/{n} after {tries} tries")
    return np.array(out)
