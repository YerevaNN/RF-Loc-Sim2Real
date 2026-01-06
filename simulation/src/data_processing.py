import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

# BS corrections from original analysis
BS_CORRECTIONS = {
    "41.871766_12.461936": (41.87255347, 12.46204160),
    "41.868917_12.462432": (41.86982378, 12.46159222),
    "41.870385_12.46081":  (41.86982378, 12.46159222),
    "41.865419_12.465402": (41.86603287, 12.46527750),
    "41.869046_12.46551":  (41.86603287, 12.46527750),
    "41.871251_12.464301": (41.87255347, 12.46204160),
    "41.871046_12.471436": (41.87131243, 12.46805397),
    "41.873666_12.453575": (41.87290044, 12.45348311),
    "41.864758_12.469067": (41.86384904, 12.46890104),
    "41.866025_12.465944": (41.86603287, 12.46527750),
}


def load_and_process_data(csv_path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    df_corrected = correct_bs_coordinates(df, BS_CORRECTIONS)
    logger.info(f"Loaded {len(df_corrected)} records")
    return df_corrected


def correct_bs_coordinates(df: pd.DataFrame, 
                          mapping: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    df = df.copy()
    
    mapped = df["BS_location"].map(mapping)
    
    if mapped.isna().all():
        logger.info("No BS coordinate corrections needed - no matching locations found")
        return df
    
    mapped_list = []
    for val in mapped:
        if pd.notna(val) and val is not None:
            mapped_list.append(val)
        else:
            mapped_list.append((None, None))
    
    tmp = pd.DataFrame(mapped_list,
                      columns=["_lat_new", "_lon_new"],
                      index=df.index)
    
    df["BSLatitude"] = tmp["_lat_new"].combine_first(df["BSLatitude"])
    df["BSLongitude"] = tmp["_lon_new"].combine_first(df["BSLongitude"])
    
    mask = tmp["_lat_new"].notna()
    df.loc[mask, "BS_location"] = (
        df.loc[mask, "BSLatitude"].round(8).astype(str) + "_"
        + df.loc[mask, "BSLongitude"].round(8).astype(str)
    )
    return df


def create_tx_rx_maps(df: pd.DataFrame) -> Tuple[Dict, Dict, np.ndarray]:
    tx_map = {}
    rx_map = {}
    tx_counter = 0
    rx_counter = 0
    
    tx_coord_to_idx = {}
    
    for _, row in df.iterrows():
        tx_coord = (round(row['BSLatitude'], 6), round(row['BSLongitude'], 6))
        
        if tx_coord not in tx_coord_to_idx:
            tx_idx = tx_counter
            tx_name = f"tx_{tx_idx}"
            tx_coord_to_idx[tx_coord] = tx_idx
            tx_key = f"{tx_coord[0]:.6f}_{tx_coord[1]:.6f}"
            tx_map[tx_key] = (tx_idx, tx_name, tx_coord[0], tx_coord[1])
            tx_counter += 1
            
        if row['UE_location'] not in rx_map:
            rx_idx = rx_counter
            rx_name = f"rx_{rx_idx}"
            rx_map[row['UE_location']] = (rx_idx, rx_name, row['UELatitude'], row['UELongitude'])
            rx_counter += 1
    
    rome_data = np.full((len(tx_map), len(rx_map), 4), -90.0)
    
    for _, row in df.iterrows():
        tx_coord = (round(row['BSLatitude'], 6), round(row['BSLongitude'], 6))
        tx_idx = tx_coord_to_idx[tx_coord]
        
        rx_idx, _, _, _ = rx_map[row['UE_location']]
        
        rome_data[tx_idx, rx_idx, :] = [row["RSSI"], row["NSINR"], row["NRSRP"], row["NRSRQ"]]
    
    logger.info(f"Created maps: {len(tx_map)} TX, {len(rx_map)} RX")
    
    original_bs_count = df[['BSLatitude', 'BSLongitude']].drop_duplicates().shape[0]
    if len(tx_map) < original_bs_count:
        logger.info(f"Merged {original_bs_count - len(tx_map)} duplicate BS locations after corrections")
    
    return tx_map, rx_map, rome_data


def calculate_correlations_full_and_filtered(rome_data: np.ndarray, sionna_data: np.ndarray, 
                                           tx_map: Dict) -> Tuple[Dict, Dict]:
    correlations_full = {}
    correlations_filtered = {}
    
    for tx_key, (tx_idx, tx_name, tx_lat, tx_lon) in tx_map.items():
        real_rssi = rome_data[tx_idx, :, 0]
        sim_rssi = sionna_data[tx_idx, :, 0]
        
        # Full dataset (including -90 values)
        if len(real_rssi) > 1:
            corr_full, p_value_full = spearmanr(real_rssi, sim_rssi)
            correlations_full[tx_idx] = {
                'correlation': corr_full,
                'p_value': p_value_full,
                'n_samples': len(real_rssi),
                'tx_name': tx_name,
                'tx_key': tx_key,
                'tx_lat': tx_lat,
                'tx_lon': tx_lon,
                'dataset_type': 'full'
            }
        else:
            correlations_full[tx_idx] = {
                'correlation': np.nan,
                'p_value': np.nan,
                'n_samples': len(real_rssi),
                'tx_name': tx_name,
                'tx_key': tx_key,
                'tx_lat': tx_lat,
                'tx_lon': tx_lon,
                'dataset_type': 'full'
            }
        
        # Filtered dataset (excluding -90 values)
        valid_mask = real_rssi != -90
        if np.sum(valid_mask) > 1:
            corr_filtered, p_value_filtered = spearmanr(real_rssi[valid_mask], sim_rssi[valid_mask])
            correlations_filtered[tx_idx] = {
                'correlation': corr_filtered,
                'p_value': p_value_filtered,
                'n_samples': np.sum(valid_mask),
                'tx_name': tx_name,
                'tx_key': tx_key,
                'tx_lat': tx_lat,
                'tx_lon': tx_lon,
                'dataset_type': 'filtered'
            }
        else:
            correlations_filtered[tx_idx] = {
                'correlation': np.nan,
                'p_value': np.nan,
                'n_samples': np.sum(valid_mask),
                'tx_name': tx_name,
                'tx_key': tx_key,
                'tx_lat': tx_lat,
                'tx_lon': tx_lon,
                'dataset_type': 'filtered'
            }
        
        logger.info(f"BS {tx_idx} ({tx_name}) at ({tx_lat:.6f}, {tx_lon:.6f}):")
        logger.info(f"  Full dataset: Spearman = {correlations_full[tx_idx]['correlation']:.3f} (n={len(real_rssi)})")
        logger.info(f"  Filtered dataset: Spearman = {correlations_filtered[tx_idx]['correlation']:.3f} (n={np.sum(valid_mask)})")
    
    return correlations_full, correlations_filtered
