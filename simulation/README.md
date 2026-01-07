# Sionna RT Simulation Pipeline

Ray-tracing simulation framework for generating synthetic RF datasets.

## Directory Structure

```
simulation/
├── src/                    # Core simulation modules
│   ├── simulation.py       # Sionna RT simulation, coordinate transforms, metrics
│   ├── data_processing.py  # BS corrections, TX/RX mappings, Spearman correlations
│   ├── scene_generator.py  # PLY mesh & XML scene generation from OSM
│   ├── data_pipeline.py    # OSM extraction, Rome CSV filtering
│   ├── optimizer.py        # GP optimization for BS parameters (Dataset B')
│   ├── pretraining_data_generator.py        # Dataset B generator (real BS locations)
│   ├── pretraining_data_generator_patches.py # Dataset C generator (city-scale grid)
│   └── patch_scene_manager.py               # 10×10 grid patch management
├── config/
│   └── parameters.py       # Simulation parameters & BS configurations
└── scripts/
    ├── run_dataset_b.sh    # Generate Dataset B/B' (Trastevere/Ostiense)
    └── run_dataset_c.sh    # Generate Dataset C (city-scale 10×10 grid)
```

## Datasets

| Dataset | Description | Key Parameters |
|---------|-------------|----------------|
| **B** | Real BS locations, simulated signals (Trastevere, Ostiense) | 11-27 BSs per scene, 64k UEs |
| **B'** | Dataset B + GP-optimized BS parameters | Optimizes position, altitude, azimuth |
| **C** | Simulated BS + signals across city-scale grid | 10×10 patches, 10 BS/patch, ~5k UEs/patch |

## Key Parameters (from `config/parameters.py`)

```python
OPTIMIZED_PARAMS = {
    "frequency": 1.2e9,           # 1.2 GHz
    "max_depth": 3,               # Ray-tracing depth
    "samples_per_src": 1e6,       # Monte Carlo samples
    "max_num_paths_per_src": 1e4, # Max paths per source
    "diffuse_reflection": True,   
    "specular_reflection": False,
    "tx_pattern": "tr38901",      # antenna pattern
}
```

## Usage

### Dataset B/B' (Trastevere/Ostiense scenes)
```bash
sbatch scripts/run_dataset_b.sh
```

### Dataset C (City-scale grid)
```bash
sbatch scripts/run_dataset_c.sh
```

## Dependencies

- Sionna RT (v1.1.0+)
- TensorFlow
- osmnx, shapely, pyproj
- scikit-optimize (for GP optimization)
- plyfile

