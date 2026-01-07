import numpy as np
import logging
from typing import Dict, Tuple, Optional
from pyproj import Transformer

logger = logging.getLogger(__name__)

DEFAULT_CENTER_LON, DEFAULT_CENTER_LAT = 12.4622493, 41.8698541
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)


def lonlat_to_local(lon: float, lat: float, 
                    center_lon: float = None, center_lat: float = None) -> Tuple[float, float]:
    if center_lon is None:
        center_lon = DEFAULT_CENTER_LON
    if center_lat is None:
        center_lat = DEFAULT_CENTER_LAT
    
    x, y = transformer.transform(lon, lat)
    center_x, center_y = transformer.transform(center_lon, center_lat)
    return x - center_x, y - center_y


def clear_scene(scene) -> None:
    tx_ids = list(scene.transmitters.keys())
    rx_ids = list(scene.receivers.keys())
    for tx_id in tx_ids:
        scene.remove(tx_id)
    for rx_id in rx_ids:
        scene.remove(rx_id)


def clear_receivers_only(scene) -> None:
    rx_ids = list(scene.receivers.keys())
    for rx_id in rx_ids:
        scene.remove(rx_id)


def add_receivers_only(scene, rx_map: Dict, Receiver=None,
                       center_lon: float = None, center_lat: float = None) -> None:
    if Receiver is None:
        from sionna.rt import Receiver

    for loc, (rx_idx, rx_name, rx_lat, rx_lon) in rx_map.items():
        rx_pos = lonlat_to_local(rx_lon, rx_lat, center_lon=center_lon, center_lat=center_lat)
        rx = Receiver(
            name=rx_name,
            position=(float(rx_pos[0]), float(rx_pos[1]), 1.0),
            orientation=[0.0, 0.0, 0.0]
        )
        scene.add(rx)


def setup_scene_arrays(scene, params: Dict, PlanarArray=None) -> None:
    if PlanarArray is None:
        from sionna.rt import PlanarArray
    
    scene.tx_array = PlanarArray(
        num_rows=params["num_rows"],
        num_cols=params["num_cols"],
        vertical_spacing=0.7,
        horizontal_spacing=0.5,
        pattern=params['tx_pattern'],
        polarization=params['tx_polarization']
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern=params['rx_pattern'],
        polarization=params['rx_polarization']
    )


def add_transmitters_receivers(scene, tx_map: Dict, rx_map: Dict, 
                              params: Dict, bs_params: Dict, Transmitter=None, Receiver=None,
                              center_lon: float = None, center_lat: float = None) -> None:
    if Transmitter is None or Receiver is None:
        from sionna.rt import Transmitter, Receiver
    
    for loc, (tx_idx, tx_name, tx_lat, tx_lon) in tx_map.items():
        tx_pos = lonlat_to_local(tx_lon, tx_lat, center_lon=center_lon, center_lat=center_lat)

        bs_cfg = bs_params.get(tx_idx, {})
        altitude = bs_cfg.get("altitude", 40)
        azimuth = bs_cfg.get("azimuth", 0)
        power_dbm = bs_cfg.get("power_dbm", 43)

        orientation = [0.0, 0.0, float(np.radians(azimuth))]

        tx = Transmitter(
            name=tx_name,
            position=(float(tx_pos[0]), float(tx_pos[1]), float(altitude)),
            orientation=orientation,
            power_dbm=float(power_dbm)
        )
        scene.add(tx)
        logger.debug(f"  TX {tx_name} (BS {tx_idx}): pos=({tx_lat:.6f}, {tx_lon:.6f}), "
                     f"alt={altitude}m, az={azimuth:.1f}°, pwr={power_dbm:.1f}dBm")
    
    for loc, (rx_idx, rx_name, rx_lat, rx_lon) in rx_map.items():
        rx_pos = lonlat_to_local(rx_lon, rx_lat, center_lon=center_lon, center_lat=center_lat)
        rx = Receiver(
            name=rx_name, 
            position=(float(rx_pos[0]), float(rx_pos[1]), 1.0), 
            orientation=[0.0, 0.0, 0.0]
        )
        scene.add(rx)


_gpu_status_logged = False
_cached_solver = None

def run_simulation(scene, params: Dict, PathSolver=None) -> Optional[Tuple]:
    global _gpu_status_logged, _cached_solver

    try:
        logger.debug(f"Computing paths: {len(scene.transmitters)} TXs, {len(scene.receivers)} RXs")

        if not _gpu_status_logged:
            try:
                import sionna
                import mitsuba as mi
                using_gpu = getattr(sionna.config, 'gpu_enabled', False)
                variant = mi.variant()
                logger.info(f"  [GPU Config] Sionna={using_gpu}, Mitsuba={variant}")
                _gpu_status_logged = True
            except Exception as e:
                logger.debug(f"  Could not check GPU config: {e}")
                _gpu_status_logged = True

        if _cached_solver is None:
            if PathSolver is None:
                from sionna.rt.path_solvers import PathSolver
            _cached_solver = PathSolver()
            logger.info("  Created PathSolver (will be reused)")

        solver = _cached_solver
        
        solver_args = {
            "scene": scene,
            "max_depth": params["max_depth"],
            "max_num_paths_per_src": params["max_num_paths_per_src"],
            "samples_per_src": params["samples_per_src"],
            "synthetic_array": params["synthetic_array"],
            "los": params["los"],
            "specular_reflection": params["specular_reflection"],
            "diffuse_reflection": params["diffuse_reflection"],
            "refraction": params["refraction"]
        }
        
        logger.debug(f"  Calling PathSolver with args: {list(solver_args.keys())}")
        paths = solver(**solver_args)
        
        logger.debug(f"  Path computation successful!")
        return paths
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error computing paths: {e}")
        logger.error(f"  Error type: {type(e).__name__}")
        logger.error(f"  Error details: {error_msg}")
        
        if any(keyword in error_msg.lower() for keyword in ['cuda', 'gpu', 'drjit', 'seed']):
            logger.error("  This appears to be a GPU/CUDA related error.")
            logger.warning("  Attempting CPU fallback...")
            
            try:
                import mitsuba as mi
                current_variant = mi.variant()
                if 'cuda' in current_variant.lower():
                    logger.info("  Switching Mitsuba to CPU scalar mode")
                    mi.set_variant('scalar_rgb')
                    
                    solver = PathSolver()
                    paths = solver(**solver_args)
                    logger.info("  CPU fallback successful!")
                    return paths
                    
            except Exception as fallback_e:
                logger.error(f"  CPU fallback also failed: {fallback_e}")
            
        return None


def calculate_metrics(paths, num_re: int = 1008, num_rb: int = 84,
                     noise_power: float = -100.0, interference_power: float = -110.0,
                     tx_power_dbm: float = 43.0) -> Optional[Tuple]:
    logger.debug("Calculating metrics")
    try:
        a_real, a_imag = paths.a
        a_real_np = a_real.numpy()
        a_imag_np = a_imag.numpy()
        a_np = a_real_np + 1j * a_imag_np

        nd = a_np.ndim
        if nd == 5:
            path_gain = np.sum(np.abs(a_np)**2, axis=(1, 3, 4))
        elif nd == 3:
            path_gain = np.sum(np.abs(a_np)**2, axis=-1)
        else:
            path_gain = np.sum(np.abs(a_np)**2, axis=-1)

        epsilon = 1e-30
        path_gain = np.maximum(path_gain, epsilon)

        path_gain_db = 10 * np.log10(path_gain)

        rssi_dbm = tx_power_dbm + path_gain_db
        rssi_dbm = np.maximum(rssi_dbm, -140.0)

        rx_power_linear = 10 ** (rssi_dbm / 10.0)
        noise_linear = 10 ** (noise_power / 10.0)
        interference_linear = 10 ** (interference_power / 10.0)

        nsinr = rx_power_linear / (noise_linear + interference_linear)
        nsinr_db = 10 * np.log10(np.maximum(nsinr, epsilon))

        nrsrp_power = rx_power_linear / num_re
        nrsrp_dbm = 10 * np.log10(np.maximum(nrsrp_power, epsilon)) + 30

        nrsrq = (num_rb * nrsrp_power) / np.maximum(rx_power_linear, epsilon)
        nrsrq_db = 10 * np.log10(np.maximum(nrsrq, epsilon))

        logger.debug(f"  Metrics: TX_pwr={tx_power_dbm:.1f}dBm, PathGain=[{path_gain_db.min():.1f}, {path_gain_db.max():.1f}]dB, "
                     f"RSSI=[{rssi_dbm.min():.1f}, {rssi_dbm.max():.1f}]dBm")
        return rssi_dbm, nsinr_db, nrsrp_dbm, nrsrq_db

    except Exception as e:
        logger.error(f"Error in calculate_metrics: {e}", exc_info=True)
        return None
