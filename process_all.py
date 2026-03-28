"""Load 12 monthly AIS CSVs, filter to west coast large vessels, save as parquet."""
import pandas as pd
from pathlib import Path
from src.config import LARGE_VESSEL_TYPE_RANGE, REGIONS, MIN_SHIP_SPEED_KNOTS, PROCESSED_DIR
from src.ship_data import AIS_DTYPES

REGION = "us_west_coast"
bounds = REGIONS[REGION]["bounds"]
min_lon, min_lat, max_lon, max_lat = bounds

data_dir = Path(__file__).parent
csv_files = sorted(data_dir.glob("ais-2024-*.csv"))
print(f"Found {len(csv_files)} AIS CSV files")

all_dfs = []
for f in csv_files:
    month_tag = int(f.stem.split("-")[2])
    print(f"  [{month_tag:02d}/12] Loading {f.name} ...", end="", flush=True)

    df = pd.read_csv(f, dtype=AIS_DTYPES, parse_dates=["base_date_time"], low_memory=False)
    n_raw = len(df)

    df = df[df["vessel_type"].isin(LARGE_VESSEL_TYPE_RANGE)]
    df = df[df["sog"] >= MIN_SHIP_SPEED_KNOTS]
    df = df[
        (df["longitude"] >= min_lon) & (df["longitude"] <= max_lon)
        & (df["latitude"] >= min_lat) & (df["latitude"] <= max_lat)
    ].copy()

    df["month"] = month_tag
    df["vessel_category"] = df["vessel_type"].map(
        lambda x: "cargo" if 70 <= x < 80 else "tanker" if 80 <= x < 90 else "passenger" if 60 <= x < 70 else "other"
    )

    print(f" {n_raw:>9,} raw -> {len(df):>7,} filtered")
    all_dfs.append(df)

combined = pd.concat(all_dfs, ignore_index=True)
print(f"\nTotal: {len(combined):,} ship records across 12 months")
print(f"\nPer-month breakdown:")
print(combined.groupby("month").size().to_string())
print(f"\nVessel categories:")
print(combined["vessel_category"].value_counts().to_string())

out_path = PROCESSED_DIR / f"ships_2024_monthly_{REGION}.parquet"
out_path.parent.mkdir(parents=True, exist_ok=True)
combined.to_parquet(out_path, index=False)
print(f"\nSaved to {out_path}")
