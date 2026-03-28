"""Process AIS CSVs for all regions in a single pass."""
import pandas as pd
from pathlib import Path
from src.config import LARGE_VESSEL_TYPE_RANGE, REGIONS, MIN_SHIP_SPEED_KNOTS, PROCESSED_DIR
from src.ship_data import AIS_DTYPES

data_dir = Path(__file__).parent
csv_files = sorted(data_dir.glob("ais-2024-*.csv"))
print(f"Found {len(csv_files)} AIS CSV files")

region_dfs = {r: [] for r in REGIONS}

for f in csv_files:
    month_tag = int(f.stem.split("-")[2])
    print(f"\n[{month_tag:02d}/12] Loading {f.name} ...", flush=True)

    df = pd.read_csv(f, dtype=AIS_DTYPES, parse_dates=["base_date_time"], low_memory=False)
    n_raw = len(df)

    df = df[df["vessel_type"].isin(LARGE_VESSEL_TYPE_RANGE)]
    df = df[df["sog"] >= MIN_SHIP_SPEED_KNOTS].copy()
    df["month"] = month_tag
    df["vessel_category"] = df["vessel_type"].map(
        lambda x: "cargo" if 70 <= x < 80 else "tanker" if 80 <= x < 90 else "passenger" if 60 <= x < 70 else "other"
    )

    for region_key, info in REGIONS.items():
        min_lon, min_lat, max_lon, max_lat = info["bounds"]
        rdf = df[
            (df["longitude"] >= min_lon) & (df["longitude"] <= max_lon)
            & (df["latitude"] >= min_lat) & (df["latitude"] <= max_lat)
        ]
        if not rdf.empty:
            region_dfs[region_key].append(rdf)
    print(f"  {n_raw:>9,} raw, filtered to large moving vessels", flush=True)

print("\n" + "=" * 60)
for region_key, dfs in region_dfs.items():
    if not dfs:
        print(f"  {region_key}: no data")
        continue
    combined = pd.concat(dfs, ignore_index=True)
    out_path = PROCESSED_DIR / f"ships_2024_monthly_{region_key}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    print(f"  {region_key}: {len(combined):,} records saved")

print("\nDone!")
