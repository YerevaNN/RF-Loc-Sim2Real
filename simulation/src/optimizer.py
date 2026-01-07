import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sionna.rt import load_scene, Camera, Transmitter, Receiver, PlanarArray
from sionna.rt.path_solvers import PathSolver
from scipy.stats import spearmanr, norm
from scipy.optimize import minimize, differential_evolution
from pyproj import Transformer
from typing import Dict, Tuple, List, Optional
import warnings
import itertools
from tqdm import tqdm
import seaborn as sns
import time
import pickle

try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
    from skopt.acquisition import gaussian_ei
    SKOPT_AVAILABLE = True
    print("Using scikit-optimize for GP optimization")
except ImportError:
    SKOPT_AVAILABLE = False
    print("Warning: scikit-optimize not available, using sklearn GP")
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
CENTER_LON, CENTER_LAT = 12.4622493, 41.8698541


class BSParameterOptimizer:
    
    def __init__(self, csv_path: str, scene_path: str, base_params: Dict, 
                 target_bs_idx: int, search_bounds: Dict = None, correlation_type: str = 'filtered'):
        self.csv_path = csv_path
        self.scene_path = scene_path
        self.base_params = base_params
        self.target_bs_idx = target_bs_idx
        self.correlation_type = correlation_type
        
        try:
            from .data_processing import load_and_process_data, create_tx_rx_maps
            from .simulation import lonlat_to_local
        except ImportError:
            try:
                from data_processing import load_and_process_data, create_tx_rx_maps
                from simulation import lonlat_to_local
            except ImportError:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(current_dir)
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                from src.data_processing import load_and_process_data, create_tx_rx_maps
                from src.simulation import lonlat_to_local
        
        self.df = load_and_process_data(csv_path)
        self.tx_map, self.rx_map, self.rome_data = create_tx_rx_maps(self.df)
        
        self.original_location = self._get_original_bs_location()
        self.original_x, self.original_y = lonlat_to_local(
            self.original_location[1], self.original_location[0]
        )
        
        if search_bounds is None:
            self.search_bounds = {
                'x_offset': (-50.0, 50.0),
                'y_offset': (-50.0, 50.0),
                'altitude': (20.0, 100.0),
                'azimuth': (0.0, 360.0)
            }
        else:
            self.search_bounds = search_bounds
        
        self.X_train = []
        self.y_train = []
        self.iteration_count = 0
        
        self.scene = load_scene(scene_path)
        self.scene.frequency = base_params['frequency']
        self.scene._xml_path = scene_path
        
        logger.info(f"Initialized optimizer for BS {target_bs_idx}")
        logger.info(f"Correlation type: {correlation_type}")
        logger.info(f"Original location: {self.original_location}")
        logger.info(f"Search bounds: {self.search_bounds}")
    
    def _get_original_bs_location(self) -> Tuple[float, float]:
        for loc, (tx_idx, tx_name, tx_lat, tx_lon) in self.tx_map.items():
            if tx_idx == self.target_bs_idx:
                return (tx_lat, tx_lon)
        raise ValueError(f"BS {self.target_bs_idx} not found")
    
    def _convert_params_to_location(self, x_offset: float, y_offset: float) -> Tuple[float, float]:
        new_x = self.original_x + x_offset
        new_y = self.original_y + y_offset
        
        center_x, center_y = transformer.transform(CENTER_LON, CENTER_LAT)
        world_x = center_x + new_x
        world_y = center_y + new_y
        new_lon, new_lat = transformer.transform(world_x, world_y, direction='INVERSE')
        
        return new_lat, new_lon
    
    def objective_function(self, params: List[float], return_both_correlations: bool = False) -> float:
        try:
            from .simulation import clear_scene, setup_scene_arrays, add_transmitters_receivers
            from .simulation import run_simulation, calculate_metrics
            from ..config.parameters import BS_OPTIMIZED_PARAMS
        except ImportError:
            try:
                from simulation import clear_scene, setup_scene_arrays, add_transmitters_receivers
                from simulation import run_simulation, calculate_metrics
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from config.parameters import BS_OPTIMIZED_PARAMS
            except ImportError:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(current_dir)
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                from src.simulation import clear_scene, setup_scene_arrays, add_transmitters_receivers
                from src.simulation import run_simulation, calculate_metrics
                from config.parameters import BS_OPTIMIZED_PARAMS
        
        x_offset, y_offset, altitude, azimuth = params
        self.iteration_count += 1
        
        try:
            new_lat, new_lon = self._convert_params_to_location(x_offset, y_offset)
            
            logger.info(f"Iteration {self.iteration_count}: Testing x_offset={x_offset:.1f}m, "
                       f"y_offset={y_offset:.1f}m, altitude={altitude:.1f}m, azimuth={azimuth:.1f}°")
            logger.info(f"  New location: ({new_lat:.6f}, {new_lon:.6f})")
            
            modified_tx_map = self.tx_map.copy()
            for loc, (tx_idx, tx_name, tx_lat, tx_lon) in self.tx_map.items():
                if tx_idx == self.target_bs_idx:
                    modified_tx_map[loc] = (tx_idx, tx_name, new_lat, new_lon)
                    break
            
            bs_params = BS_OPTIMIZED_PARAMS.copy()
            bs_params[self.target_bs_idx] = {"altitude": altitude, "azimuth": azimuth}
            
            clear_scene(self.scene)
            setup_scene_arrays(self.scene, self.base_params)
            add_transmitters_receivers(self.scene, modified_tx_map, self.rx_map, 
                                     self.base_params, bs_params)
            
            paths = run_simulation(self.scene, self.base_params)
            if paths is None:
                logger.warning(f"  Simulation failed - returning default correlation")
                self.X_train.append([x_offset, y_offset, altitude, azimuth])
                self.y_train.append(1.0)
                return 1.0 if not return_both_correlations else (-1.0, -1.0)
            
            metrics = calculate_metrics(paths)
            if metrics is None:
                logger.warning(f"  Metrics calculation failed - returning default correlation")
                self.X_train.append([x_offset, y_offset, altitude, azimuth])
                self.y_train.append(1.0)
                return 1.0 if not return_both_correlations else (-1.0, -1.0)
            
            rssi_dbm, _, _, _ = metrics
            sionna_data = np.zeros_like(self.rome_data)
            sionna_data[:, :, 0] = rssi_dbm.transpose()
            
            # Calculate correlations for target BS
            real_rssi = self.rome_data[self.target_bs_idx, :, 0]
            sim_rssi = sionna_data[self.target_bs_idx, :, 0]
            
            logger.debug(f"  Real RSSI range: {real_rssi.min():.1f} to {real_rssi.max():.1f} dBm")
            logger.debug(f"  Sim RSSI range: {sim_rssi.min():.1f} to {sim_rssi.max():.1f} dBm")
            logger.debug(f"  Real RSSI samples with -90: {np.sum(real_rssi == -90)}/{len(real_rssi)}")
            logger.debug(f"  Non-zero sim values: {np.sum(sim_rssi != 0)}/{len(sim_rssi)}")
            
            corr_full, _ = spearmanr(real_rssi, sim_rssi)
            if np.isnan(corr_full):
                corr_full = -1.0
            
            valid_mask = real_rssi != -90
            if np.sum(valid_mask) > 1:
                real_filtered = real_rssi[valid_mask]
                sim_filtered = sim_rssi[valid_mask]
                
                logger.debug(f"  Filtered data: {len(real_filtered)} valid samples")
                logger.debug(f"  Filtered real range: {real_filtered.min():.1f} to {real_filtered.max():.1f}")
                logger.debug(f"  Filtered sim range: {sim_filtered.min():.1f} to {sim_filtered.max():.1f}")
                
                corr_filtered, _ = spearmanr(real_filtered, sim_filtered)
                if np.isnan(corr_filtered):
                    corr_filtered = -1.0
            else:
                corr_filtered = -1.0
            
            logger.info(f"  Results: full={corr_full:.3f}, filtered={corr_filtered:.3f}")
            
            target_corr = corr_full if self.correlation_type == 'full' else corr_filtered
            
            self.X_train.append([x_offset, y_offset, altitude, azimuth])
            self.y_train.append(-target_corr)
            
            if return_both_correlations:
                return corr_full, corr_filtered
            else:
                return -target_corr
                
        except Exception as e:
            logger.error(f"Error in objective function: {e}")
            if return_both_correlations:
                return -1.0, -1.0
            else:
                return 1.0
    
    def optimize_with_gp(self, n_calls: int = 100, n_initial_points: int = 20,
                        acq_func: str = 'EI', random_state: int = 42) -> Dict:
        if not SKOPT_AVAILABLE:
            return self._optimize_with_sklearn_gp(n_calls, n_initial_points, random_state)
        
        logger.info(f"Starting GP optimization with {n_calls} calls, {n_initial_points} initial points")
        
        dimensions = [
            Real(self.search_bounds['x_offset'][0], self.search_bounds['x_offset'][1], name='x_offset'),
            Real(self.search_bounds['y_offset'][0], self.search_bounds['y_offset'][1], name='y_offset'),
            Real(self.search_bounds['altitude'][0], self.search_bounds['altitude'][1], name='altitude'),
            Real(self.search_bounds['azimuth'][0], self.search_bounds['azimuth'][1], name='azimuth')
        ]
        
        start_time = time.time()
        
        # Run optimization
        result = gp_minimize(
            func=self.objective_function,
            dimensions=dimensions,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            acq_func=acq_func,
            random_state=random_state,
            verbose=True
        )
        
        end_time = time.time()
        
        best_x_offset, best_y_offset, best_altitude, best_azimuth = result.x
        best_lat, best_lon = self._convert_params_to_location(best_x_offset, best_y_offset)
        
        corr_full, corr_filtered = self.objective_function(result.x, return_both_correlations=True)
        
        optimization_results = {
            'target_bs_idx': self.target_bs_idx,
            'optimization_time': end_time - start_time,
            'n_iterations': len(result.func_vals),
            'original_location': self.original_location,
            'best_parameters': {
                'x_offset': best_x_offset,
                'y_offset': best_y_offset,
                'altitude': best_altitude,
                'azimuth': best_azimuth,
                'latitude': best_lat,
                'longitude': best_lon
            },
            'best_correlations': {
                'full': corr_full,
                'filtered': corr_filtered
            },
            'convergence_curve': result.func_vals,
            'all_evaluations': list(zip(self.X_train, self.y_train)),
            'search_bounds': self.search_bounds,
            'correlation_type': self.correlation_type
        }
        
        logger.info(f"GP Optimization completed in {end_time - start_time:.1f}s")
        logger.info(f"Best parameters: x_offset={best_x_offset:.1f}m, y_offset={best_y_offset:.1f}m, "
                   f"altitude={best_altitude:.1f}m, azimuth={best_azimuth:.1f}°")
        logger.info(f"Best correlations: full={corr_full:.3f}, filtered={corr_filtered:.3f}")
        
        return optimization_results
    
    def _optimize_with_sklearn_gp(self, n_calls: int, n_initial_points: int, 
                                 random_state: int) -> Dict:
        logger.info("Using sklearn GP optimization (fallback)")
        
        np.random.seed(random_state)
        
        bounds = np.array([
            [self.search_bounds['x_offset'][0], self.search_bounds['x_offset'][1]],
            [self.search_bounds['y_offset'][0], self.search_bounds['y_offset'][1]],
            [self.search_bounds['altitude'][0], self.search_bounds['altitude'][1]],
            [self.search_bounds['azimuth'][0], self.search_bounds['azimuth'][1]]
        ])
        
        for i in range(n_initial_points):
            x = np.random.uniform(bounds[:, 0], bounds[:, 1])
            self.objective_function(x)
        
        # Set up GP
        kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, 
                                     normalize_y=True, n_restarts_optimizer=5)
        
        best_x = None
        best_y = float('inf')
        
        for i in range(n_calls - n_initial_points):
            if len(self.X_train) < 2:
                continue
                
            X = np.array(self.X_train)
            y = np.array(self.y_train)
            gp.fit(X, y)
            
            next_x = self._optimize_acquisition(gp, bounds)
            next_y = self.objective_function(next_x)
            
            if next_y < best_y:
                best_y = next_y
                best_x = next_x
        
        if best_x is not None:
            best_lat, best_lon = self._convert_params_to_location(best_x[0], best_x[1])
            corr_full, corr_filtered = self.objective_function(best_x, return_both_correlations=True)
        else:
            best_x = [0, 0, 40, 0]
            best_lat, best_lon = self.original_location
            corr_full, corr_filtered = -1.0, -1.0
        
        return {
            'target_bs_idx': self.target_bs_idx,
            'optimization_time': 0,
            'n_iterations': len(self.X_train),
            'original_location': self.original_location,
            'best_parameters': {
                'x_offset': best_x[0],
                'y_offset': best_x[1], 
                'altitude': best_x[2],
                'azimuth': best_x[3],
                'latitude': best_lat,
                'longitude': best_lon
            },
            'best_correlations': {
                'full': corr_full,
                'filtered': corr_filtered
            },
            'convergence_curve': self.y_train,
            'all_evaluations': list(zip(self.X_train, self.y_train)),
            'search_bounds': self.search_bounds,
            'correlation_type': self.correlation_type
        }
    
    def _optimize_acquisition(self, gp, bounds, n_restarts: int = 25) -> np.ndarray:
        best_x = None
        best_acq = float('inf')
        
        for _ in range(n_restarts):
            x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])
            
            def neg_expected_improvement(x):
                x = x.reshape(1, -1)
                mu, sigma = gp.predict(x, return_std=True)
                if sigma == 0:
                    return 0
                # Expected improvement
                best_f = np.min(self.y_train)
                z = (best_f - mu) / sigma
                ei = (best_f - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei[0]
            
            res = minimize(neg_expected_improvement, x0, bounds=bounds, method='L-BFGS-B')
            
            if res.success and res.fun < best_acq:
                best_acq = res.fun
                best_x = res.x
        
        return best_x if best_x is not None else np.random.uniform(bounds[:, 0], bounds[:, 1])
