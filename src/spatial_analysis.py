"""Spatial analysis: density grids, proximity encounters, and risk hotspots."""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree

from .config import (
    PROCESSED_DIR, GRID_CELL_SIZE_DEG,
    PROXIMITY_THRESHOLD_KM, REGIONS,
)

EARTH_RADIUS_KM = 6371.0


def haversine_km(lon1, lat1, lon2, lat2):
    """Haversine distance in km (vectorized)."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def build_density_grid(
    df: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    cell_size: float = GRID_CELL_SIZE_DEG,
    weight_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D histogram of lat/lon points. Returns (grid, lon_edges, lat_edges)."""
    min_lon, min_lat, max_lon, max_lat = bounds

    lon_edges = np.arange(min_lon, max_lon + cell_size, cell_size)
    lat_edges = np.arange(min_lat, max_lat + cell_size, cell_size)

    weights = df[weight_col].values if weight_col and weight_col in df.columns else None

    grid, _, _ = np.histogram2d(
        df["longitude"].values,
        df["latitude"].values,
        bins=[lon_edges, lat_edges],
        weights=weights,
    )

    return grid, lon_edges, lat_edges


def compute_ship_density(ships_df: pd.DataFrame, region_key: str) -> tuple:
    bounds = REGIONS[region_key]["bounds"]
    return build_density_grid(ships_df, bounds)


def compute_whale_density(whales_df: pd.DataFrame, region_key: str) -> tuple:
    bounds = REGIONS[region_key]["bounds"]
    weight_col = "individualCount" if "individualCount" in whales_df.columns else None
    return build_density_grid(whales_df, bounds, weight_col=weight_col)


def find_overlap_hotspots(
    ship_grid: np.ndarray,
    whale_grid: np.ndarray,
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
    top_n: int = 50,
) -> pd.DataFrame:
    """Find cells where both ship and whale density are high. Log-normalized product scoring."""
    both_present = (ship_grid > 0) & (whale_grid > 0)
    if not both_present.any():
        return pd.DataFrame()

    ship_log = np.log1p(ship_grid.astype(float))
    whale_log = np.log1p(whale_grid.astype(float))

    ship_norm = ship_log / (ship_log.max() + 1e-10)
    whale_norm = whale_log / (whale_log.max() + 1e-10)

    ship_p25 = np.percentile(ship_grid[ship_grid > 0], 25)
    whale_p25 = np.percentile(whale_grid[whale_grid > 0], 25)
    meaningful = both_present & (ship_grid >= ship_p25) & (whale_grid >= whale_p25)

    risk = np.where(meaningful, ship_norm * whale_norm, 0.0)

    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2

    results = []
    for i in range(risk.shape[0]):
        for j in range(risk.shape[1]):
            if risk[i, j] > 0:
                results.append({
                    "longitude": lon_centers[i],
                    "latitude": lat_centers[j],
                    "risk_score": risk[i, j],
                    "ship_density": ship_grid[i, j],
                    "whale_density": whale_grid[i, j],
                })

    hotspots = pd.DataFrame(results)
    if hotspots.empty:
        return hotspots

    hotspots = hotspots.sort_values("risk_score", ascending=False).head(top_n).reset_index(drop=True)
    return hotspots


def find_close_encounters(
    ships_df: pd.DataFrame,
    whales_df: pd.DataFrame,
    threshold_km: float = PROXIMITY_THRESHOLD_KM,
) -> pd.DataFrame:
    """Find whale sightings within threshold_km of ship positions via KD-tree."""
    if ships_df.empty or whales_df.empty:
        return pd.DataFrame()

    ship_coords = np.radians(ships_df[["latitude", "longitude"]].values)
    whale_coords = np.radians(whales_df[["latitude", "longitude"]].values)

    threshold_rad = threshold_km / EARTH_RADIUS_KM

    ship_tree = cKDTree(ship_coords)

    print(f"  Querying {len(whale_coords):,} whale positions against {len(ship_coords):,} ship positions...")
    matches = ship_tree.query_ball_tree(cKDTree(whale_coords), threshold_rad)

    ship_lats = ships_df["latitude"].values
    ship_lons = ships_df["longitude"].values
    ship_names = ships_df["vessel_name"].values if "vessel_name" in ships_df.columns else np.full(len(ships_df), "Unknown")
    ship_types = ships_df["vessel_category"].values if "vessel_category" in ships_df.columns else np.full(len(ships_df), "Unknown")
    ship_speeds = ships_df["sog"].values if "sog" in ships_df.columns else np.full(len(ships_df), np.nan)
    ship_dates = ships_df["base_date_time"].astype(str).values if "base_date_time" in ships_df.columns else np.full(len(ships_df), "")

    whale_lats = whales_df["latitude"].values
    whale_lons = whales_df["longitude"].values
    whale_species = whales_df["species_common"].values if "species_common" in whales_df.columns else np.full(len(whales_df), "Unknown")
    whale_dates = whales_df["eventDate"].values if "eventDate" in whales_df.columns else np.full(len(whales_df), None)

    w_indices = []
    s_indices = []
    for s_idx, w_list in enumerate(matches):
        for w_idx in w_list:
            w_indices.append(w_idx)
            s_indices.append(s_idx)

    if not w_indices:
        return pd.DataFrame()

    w_arr = np.array(w_indices)
    s_arr = np.array(s_indices)

    distances = haversine_km(ship_lons[s_arr], ship_lats[s_arr], whale_lons[w_arr], whale_lats[w_arr])

    encounters_df = pd.DataFrame({
        "whale_lat": whale_lats[w_arr],
        "whale_lon": whale_lons[w_arr],
        "whale_species": whale_species[w_arr],
        "whale_date": whale_dates[w_arr],
        "ship_lat": ship_lats[s_arr],
        "ship_lon": ship_lons[s_arr],
        "ship_name": ship_names[s_arr],
        "ship_type": ship_types[s_arr],
        "ship_speed_knots": ship_speeds[s_arr],
        "ship_date": ship_dates[s_arr],
        "distance_km": distances,
    })

    encounters_df = encounters_df.sort_values("distance_km").reset_index(drop=True)
    return encounters_df


def compute_route_deviation_index(
    whales_df: pd.DataFrame,
    ships_df: pd.DataFrame,
    region_key: str,
    cell_size: float = GRID_CELL_SIZE_DEG,
) -> pd.DataFrame:
    """Estimate whale route deviation zones: where whale density drops under heavy traffic."""
    bounds = REGIONS[region_key]["bounds"]

    ship_grid, lon_edges, lat_edges = build_density_grid(ships_df, bounds, cell_size)

    ship_median = np.median(ship_grid[ship_grid > 0]) if ship_grid.any() else 0

    low_ship_cells = set()
    high_ship_cells = set()
    for i in range(ship_grid.shape[0]):
        for j in range(ship_grid.shape[1]):
            if ship_grid[i, j] <= ship_median:
                low_ship_cells.add((i, j))
            else:
                high_ship_cells.add((i, j))

    whale_grid, _, _ = build_density_grid(whales_df, bounds, cell_size)

    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2

    results = []
    for i in range(whale_grid.shape[0]):
        for j in range(whale_grid.shape[1]):
            if whale_grid[i, j] > 0 or ship_grid[i, j] > 0:
                results.append({
                    "longitude": lon_centers[i],
                    "latitude": lat_centers[j],
                    "whale_density": whale_grid[i, j],
                    "ship_density": ship_grid[i, j],
                    "high_ship_traffic": ship_grid[i, j] > ship_median,
                })

    return pd.DataFrame(results)


def _run_hotspots_and_encounters(
    ships_df: pd.DataFrame,
    whales_df: pd.DataFrame,
    region_key: str,
    label: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute hotspots + encounters for a ship subset."""
    prefix = f"  [{label}] " if label else "  "

    print(f"{prefix}Computing ship density grid...")
    ship_grid, lon_edges, lat_edges = compute_ship_density(ships_df, region_key)

    print(f"{prefix}Computing whale density grid...")
    whale_grid, _, _ = compute_whale_density(whales_df, region_key)

    min_shape = (
        min(ship_grid.shape[0], whale_grid.shape[0]),
        min(ship_grid.shape[1], whale_grid.shape[1]),
    )
    ship_grid_t = ship_grid[:min_shape[0], :min_shape[1]]
    whale_grid_t = whale_grid[:min_shape[0], :min_shape[1]]

    print(f"{prefix}Finding overlap hotspots...")
    hotspots = find_overlap_hotspots(ship_grid_t, whale_grid_t, lon_edges, lat_edges)
    print(f"{prefix}Found {len(hotspots)} risk hotspots")

    print(f"{prefix}Finding close encounters...")
    ENCOUNTER_SHIP_SAMPLE = 150_000
    if len(ships_df) > ENCOUNTER_SHIP_SAMPLE:
        ships_sample = ships_df.sample(n=ENCOUNTER_SHIP_SAMPLE, random_state=42)
        print(f"{prefix}(sampled {ENCOUNTER_SHIP_SAMPLE:,} of {len(ships_df):,} ships)")
    else:
        ships_sample = ships_df
    encounters = find_close_encounters(ships_sample, whales_df)
    print(f"{prefix}Found {len(encounters):,} close encounters")

    return hotspots, encounters


def run_spatial_analysis(
    ships_df: pd.DataFrame,
    whales_df: pd.DataFrame,
    region_key: str,
    save: bool = True,
) -> dict:
    """Run the full spatial analysis pipeline for a region."""
    print(f"\n=== Spatial Analysis: {REGIONS[region_key]['name']} ===")

    # All vessels
    print(f"\n--- All vessels ({len(ships_df):,}) ---")
    hotspots, encounters = _run_hotspots_and_encounters(
        ships_df, whales_df, region_key, label="all",
    )

    # Cargo + tanker only (hazardous large vessels)
    hazardous = ships_df[ships_df["vessel_category"].isin(["cargo", "tanker"])]
    print(f"\n--- Cargo + Tanker only ({len(hazardous):,}) ---")
    hotspots_ct, encounters_ct = _run_hotspots_and_encounters(
        hazardous, whales_df, region_key, label="cargo+tanker",
    )

    print("\nComputing route deviation index...")
    deviation = compute_route_deviation_index(whales_df, ships_df, region_key)

    if save:
        tag = f"_{region_key}"
        for name, df in [
            (f"hotspots{tag}", hotspots),
            (f"encounters{tag}", encounters),
            (f"hotspots_cargo_tanker{tag}", hotspots_ct),
            (f"encounters_cargo_tanker{tag}", encounters_ct),
            (f"deviation{tag}", deviation),
        ]:
            if not df.empty:
                df.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)

    return {
        "hotspots": hotspots,
        "encounters": encounters,
        "hotspots_cargo_tanker": hotspots_ct,
        "encounters_cargo_tanker": encounters_ct,
        "deviation": deviation,
    }
