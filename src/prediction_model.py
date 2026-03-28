"""Encounter prediction: predict whale-ship risk along ship routes."""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path

from .config import PROCESSED_DIR, REGIONS, GRID_CELL_SIZE_DEG, ENCOUNTER_RISK_RADIUS_KM
from .spatial_analysis import (
    haversine_km, build_density_grid, EARTH_RADIUS_KM,
)


class EncounterPredictor:
    """Predicts encounter probability along a ship route using whale density surfaces."""

    def __init__(self, whales_df: pd.DataFrame, region_key: str):
        self.region_key = region_key
        self.bounds = REGIONS[region_key]["bounds"]
        self.whales_df = whales_df

        self._build_density_surface()
        self._build_monthly_surfaces()

    def _build_density_surface(self):
        grid, lon_edges, lat_edges = build_density_grid(
            self.whales_df, self.bounds, GRID_CELL_SIZE_DEG,
            weight_col="individualCount" if "individualCount" in self.whales_df.columns else None,
        )

        self.raw_grid = grid
        self.smoothed_grid = gaussian_filter(grid.astype(float), sigma=2)

        if self.smoothed_grid.max() > 0:
            self.norm_grid = self.smoothed_grid / self.smoothed_grid.max()
        else:
            self.norm_grid = self.smoothed_grid

        self.lon_edges = lon_edges
        self.lat_edges = lat_edges
        self.lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
        self.lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2

        if len(self.lon_centers) > 1 and len(self.lat_centers) > 1:
            self.interpolator = RegularGridInterpolator(
                (self.lon_centers, self.lat_centers),
                self.norm_grid,
                method="linear",
                bounds_error=False,
                fill_value=0.0,
            )
        else:
            self.interpolator = None

    def _build_monthly_surfaces(self):
        """Per-month density surfaces for seasonal weighting."""
        self.monthly_grids = {}
        if "month" not in self.whales_df.columns:
            return

        for m in range(1, 13):
            month_df = self.whales_df[self.whales_df["month"] == m]
            if len(month_df) < 10:
                continue
            grid, _, _ = build_density_grid(month_df, self.bounds, GRID_CELL_SIZE_DEG)
            smoothed = gaussian_filter(grid.astype(float), sigma=2)
            max_val = smoothed.max()
            self.monthly_grids[m] = smoothed / max_val if max_val > 0 else smoothed

    def predict_route_risk(
        self,
        route_df: pd.DataFrame,
        month: int | None = None,
    ) -> pd.DataFrame:
        """Predict encounter risk along a route (needs longitude, latitude columns)."""
        if self.interpolator is None:
            print("Warning: Insufficient data to build interpolator.")
            route_df = route_df.copy()
            route_df["whale_density"] = 0.0
            route_df["risk_score"] = 0.0
            return route_df

        points = np.column_stack([route_df["longitude"].values, route_df["latitude"].values])
        base_density = self.interpolator(points)

        if month and month in self.monthly_grids:
            monthly_interp = RegularGridInterpolator(
                (self.lon_centers, self.lat_centers),
                self.monthly_grids[month],
                method="linear",
                bounds_error=False,
                fill_value=0.0,
            )
            seasonal_density = monthly_interp(points)
            density = 0.5 * base_density + 0.5 * seasonal_density
        else:
            density = base_density

        result = route_df.copy()
        result["whale_density"] = density

        if "speed_knots" in result.columns:
            speed = result["speed_knots"].clip(lower=1)
            result["time_exposure"] = 1.0 / speed
            result["risk_score"] = density * result["time_exposure"]
        else:
            result["risk_score"] = density

        if result["risk_score"].max() > 0:
            result["risk_score"] = result["risk_score"] / result["risk_score"].max()

        result["risk_level"] = pd.cut(
            result["risk_score"],
            bins=[-0.01, 0.2, 0.5, 0.8, 1.01],
            labels=["low", "moderate", "high", "critical"],
        )

        return result

    def predict_encounter_zones(
        self, threshold: float = 0.3
    ) -> pd.DataFrame:
        """Grid cells above the risk threshold."""
        zones = []
        for i in range(self.norm_grid.shape[0]):
            for j in range(self.norm_grid.shape[1]):
                if self.norm_grid[i, j] >= threshold:
                    zones.append({
                        "longitude": self.lon_centers[i],
                        "latitude": self.lat_centers[j],
                        "whale_density_norm": self.norm_grid[i, j],
                    })
        return pd.DataFrame(zones)

    def get_density_surface(self) -> dict:
        """Density surface data for visualization."""
        return {
            "grid": self.norm_grid,
            "lon_centers": self.lon_centers,
            "lat_centers": self.lat_centers,
            "lon_edges": self.lon_edges,
            "lat_edges": self.lat_edges,
        }


def load_ship_route(csv_path: Path) -> pd.DataFrame:
    """Load a ship route CSV (needs longitude, latitude columns)."""
    df = pd.read_csv(csv_path)

    col_map = {}
    for col in df.columns:
        lower = col.lower().strip()
        if lower in ("lon", "lng", "long"):
            col_map[col] = "longitude"
        elif lower in ("lat",):
            col_map[col] = "latitude"
        elif lower in ("speed", "sog", "speed_knots"):
            col_map[col] = "speed_knots"

    df = df.rename(columns=col_map)

    if "longitude" not in df.columns or "latitude" not in df.columns:
        raise ValueError("Route CSV must have longitude and latitude columns")

    return df


def extract_ship_route_from_ais(
    ships_df: pd.DataFrame, mmsi: int | None = None, vessel_name: str | None = None
) -> pd.DataFrame:
    """Extract a single ship's route from AIS data by MMSI or vessel name."""
    if mmsi:
        route = ships_df[ships_df["mmsi"] == mmsi].copy()
    elif vessel_name:
        route = ships_df[ships_df["vessel_name"].str.contains(vessel_name, case=False, na=False)].copy()
    else:
        raise ValueError("Provide either mmsi or vessel_name")

    route = route.sort_values("base_date_time").reset_index(drop=True)

    result = route[["longitude", "latitude"]].copy()
    if "sog" in route.columns:
        result["speed_knots"] = route["sog"]
    if "base_date_time" in route.columns:
        result["timestamp"] = route["base_date_time"]
    if "vessel_name" in route.columns:
        result["vessel_name"] = route["vessel_name"]

    return result


def analyze_route(
    whales_df: pd.DataFrame,
    route_df: pd.DataFrame,
    region_key: str,
    month: int | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Full pipeline: build model and predict risk for a route."""
    predictor = EncounterPredictor(whales_df, region_key)
    result = predictor.predict_route_risk(route_df, month=month)

    high_risk = result[result["risk_level"].isin(["high", "critical"])]
    print(f"\nRoute analysis complete:")
    print(f"  Total waypoints: {len(result)}")
    print(f"  High/Critical risk points: {len(high_risk)}")

    if not high_risk.empty:
        print(f"  Highest risk location: ({high_risk.iloc[0]['latitude']:.4f}, {high_risk.iloc[0]['longitude']:.4f})")

    if save:
        out_path = PROCESSED_DIR / f"route_risk_{region_key}.parquet"
        result.to_parquet(out_path, index=False)
        print(f"  Saved to {out_path}")

    return result
