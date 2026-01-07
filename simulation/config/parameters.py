CENTER_LON, CENTER_LAT = 12.4622493, 41.8698541

OPTIMIZED_PARAMS = {
    "name": "optimized_simulation",
    "frequency": int(1.2e9),  # 1.2 GHz from paper
    "tx_height": 40,
    "max_depth": 3,
    "max_num_paths_per_src": int(1e4),
    "samples_per_src": int(1e6),
    "synthetic_array": False,
    "los": True,
    "specular_reflection": False,
    "diffuse_reflection": True,
    "refraction": True,
    "num_rows": 6,
    "num_cols": 6,
    "tx_pattern": "tr38901",
    "tx_polarization": "VH",
    "rx_pattern": "hw_dipole",
    "rx_polarization": "cross",
    "tx_orientation": [0, 0, 0],
    "rx_orientation": [0, 0, 0],
}

BS_OPTIMIZED_PARAMS = {
    0: {"altitude": 40, "azimuth": 120},
    1: {"altitude": 40, "azimuth": 240},
    2: {"altitude": 55, "azimuth": 0},
    3: {"altitude": 40, "azimuth": 90},
    4: {"altitude": 40, "azimuth": 210},
    5: {"altitude": 40, "azimuth": 0},
}

DEFAULT_SEARCH_BOUNDS = {
    'x_offset': (-50.0, 50.0),
    'y_offset': (-50.0, 50.0),
    'altitude': (20.0, 100.0),
    'azimuth': (0.0, 360.0)
}

EXPERIMENT_1_CONFIG = {
    'search_bounds': {
        'x_offset': (-30.0, 30.0),
        'y_offset': (-30.0, 30.0),
        'altitude': (30.0, 80.0),
        'azimuth': (0.0, 360.0)
    },
    'n_calls': 25,
    'n_initial_points': 8,
    'target_bs': 0,
}

EXPERIMENT_2_CONFIG = {
    'search_bounds': {
        'x_offset': (-60.0, 60.0),
        'y_offset': (-60.0, 60.0),
        'altitude': (30.0, 80.0),
        'azimuth': (0.0, 360.0)
    },
    'n_calls': 30,
    'n_initial_points': 6,
    'target_bs_list': [0, 1, 2, 3, 4, 5],
    'correlation_type': 'full'
}
