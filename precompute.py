"""Pre-compute density grids, slow zones, and confidence layers for all regions."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull

from src.config import REGIONS, GRID_CELL_SIZE_DEG, PROCESSED_DIR
from src.spatial_analysis import build_density_grid, compute_whale_density, compute_ship_density

OUTPUT_DIR = PROCESSED_DIR / "map_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def precompute_region(region_key):
    ship_path = PROCESSED_DIR / f"ships_2024_monthly_{region_key}.parquet"
    whale_path = PROCESSED_DIR / f"whales_{region_key}.parquet"
    if not ship_path.exists() or not whale_path.exists():
        print(f"  Skipping {region_key}: missing data")
        return None

    ships = pd.read_parquet(ship_path)
    whales = pd.read_parquet(whale_path)
    bounds = REGIONS[region_key]["bounds"]
    print(f"  Ships: {len(ships):,}  Whales: {len(whales):,}")

    cell = GRID_CELL_SIZE_DEG

    # --- Whale density grid (smoothed, for route simulation) ---
    whale_grid, lon_edges, lat_edges = compute_whale_density(whales, region_key)
    whale_smooth = gaussian_filter(whale_grid.astype(float), sigma=2)
    wmax = whale_smooth.max()
    whale_norm = (whale_smooth / wmax) if wmax > 0 else whale_smooth

    lon_centers = ((lon_edges[:-1] + lon_edges[1:]) / 2).tolist()
    lat_centers = ((lat_edges[:-1] + lat_edges[1:]) / 2).tolist()

    # --- Ship density grid ---
    ship_grid, _, _ = compute_ship_density(ships, region_key)
    ship_smooth = gaussian_filter(ship_grid.astype(float), sigma=2)
    smax = ship_smooth.max()
    ship_norm = (ship_smooth / smax) if smax > 0 else ship_smooth

    # --- Monthly whale density (for seasonal route prediction) ---
    monthly_whale = {}
    if "month" in whales.columns:
        for m in range(1, 13):
            mdf = whales[whales["month"] == m]
            if len(mdf) < 10:
                continue
            mg, _, _ = build_density_grid(mdf, bounds, cell)
            ms = gaussian_filter(mg.astype(float), sigma=2)
            mmx = ms.max()
            monthly_whale[m] = (ms / mmx if mmx > 0 else ms).round(4).tolist()

    # --- Confidence grid (observation count per cell) ---
    whale_raw, _, _ = build_density_grid(whales, bounds, cell)
    ship_raw, _, _ = build_density_grid(ships, bounds, cell)
    min_shape = (min(whale_raw.shape[0], ship_raw.shape[0]),
                 min(whale_raw.shape[1], ship_raw.shape[1]))
    whale_count = whale_raw[:min_shape[0], :min_shape[1]]
    ship_count = ship_raw[:min_shape[0], :min_shape[1]]
    total_obs = whale_count + ship_count
    obs_max = total_obs.max()
    confidence = (np.log1p(total_obs) / np.log1p(obs_max)).round(4) if obs_max > 0 else total_obs

    # --- Slow-down zones (clusters of high-risk cells, ocean-only) ---
    min_shape2 = (min(ship_norm.shape[0], whale_norm.shape[0]),
                  min(ship_norm.shape[1], whale_norm.shape[1]))
    sn = ship_norm[:min_shape2[0], :min_shape2[1]]
    wn = whale_norm[:min_shape2[0], :min_shape2[1]]
    sr = ship_raw[:min_shape2[0], :min_shape2[1]]
    wr = whale_raw[:min_shape2[0], :min_shape2[1]]
    risk_product = sn * wn
    threshold = np.percentile(risk_product[risk_product > 0], 90) if (risk_product > 0).any() else 0

    high_risk_cells = []
    for i in range(risk_product.shape[0]):
        for j in range(risk_product.shape[1]):
            if (risk_product[i, j] >= threshold
                    and i < len(lon_centers) and j < len(lat_centers)
                    and sr[i, j] > 0 and wr[i, j] > 0):
                high_risk_cells.append((lon_centers[i], lat_centers[j]))

    slow_zones = _cluster_to_polygons(high_risk_cells, cluster_dist=0.5)

    result = {
        "region": region_key,
        "bounds": list(bounds),
        "lon_centers": [round(x, 4) for x in lon_centers],
        "lat_centers": [round(x, 4) for x in lat_centers],
        "whale_density": whale_norm.round(4).tolist(),
        "ship_density": ship_norm[:len(lon_centers), :len(lat_centers)].round(4).tolist(),
        "confidence": confidence.tolist(),
        "slow_zones": slow_zones,
        "monthly_whale": {str(k): v for k, v in monthly_whale.items()},
    }
    return result


def _cluster_to_polygons(points, cluster_dist=0.3):
    """Group nearby points into convex hull polygons."""
    if len(points) < 3:
        return []

    from scipy.cluster.hierarchy import fcluster, linkage
    pts = np.array(points)
    if len(pts) < 3:
        return []

    Z = linkage(pts, method="complete", metric="chebyshev")
    labels = fcluster(Z, t=cluster_dist, criterion="distance")

    polygons = []
    for lbl in set(labels):
        cluster_pts = pts[labels == lbl]
        if len(cluster_pts) < 3:
            continue
        try:
            hull = ConvexHull(cluster_pts)
            vertices = cluster_pts[hull.vertices]
            poly = [[round(float(v[1]), 4), round(float(v[0]), 4)] for v in vertices]
            poly.append(poly[0])
            polygons.append(poly)
        except Exception:
            pass
    return polygons


def main():
    all_data = {}
    for region_key in REGIONS:
        print(f"\n=== {region_key} ===")
        data = precompute_region(region_key)
        if data:
            all_data[region_key] = data

    out_path = OUTPUT_DIR / "precomputed.json"
    with open(out_path, "w") as f:
        json.dump(all_data, f)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved precomputed data to {out_path} ({size_mb:.1f} MB)")
    print(f"Regions: {list(all_data.keys())}")


if __name__ == "__main__":
    main()
