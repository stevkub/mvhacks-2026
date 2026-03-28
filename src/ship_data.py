"""AIS ship data: download, parse, filter by vessel type and region."""

import io
import csv
import zipfile
import requests
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import (
    AIS_BASE_URL, RAW_SHIPS_DIR, PROCESSED_DIR,
    LARGE_VESSEL_TYPE_RANGE, REGIONS, MIN_SHIP_SPEED_KNOTS,
)

AIS_COLUMNS = [
    "mmsi", "base_date_time", "longitude", "latitude", "sog", "cog",
    "heading", "vessel_name", "imo", "call_sign", "vessel_type",
    "status", "length", "width", "draft", "cargo", "transceiver",
]

AIS_DTYPES = {
    "mmsi": "int64",
    "longitude": "float64",
    "latitude": "float64",
    "sog": "float64",
    "cog": "float64",
    "heading": "float64",
    "vessel_type": "float64",
    "length": "float64",
    "width": "float64",
    "draft": "float64",
    "cargo": "float64",
}


def get_ais_url(d: date, year: int = 2024) -> str:
    return AIS_BASE_URL.format(year=year, month=d.month, day=d.day)


def date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def download_ais_day(d: date, year: int = 2024, cache: bool = True) -> Path | None:
    """Download a single day's AIS zip. Returns path or None."""
    url = get_ais_url(d, year)
    filename = f"AIS_{year}_{d.month:02d}_{d.day:02d}.zip"
    out_path = RAW_SHIPS_DIR / filename

    if cache and out_path.exists():
        print(f"  [cached] {filename}")
        return out_path

    print(f"  [downloading] {filename} ...")
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        print(f"  [done] {filename} ({out_path.stat().st_size / 1e6:.1f} MB)")
        return out_path
    except Exception as e:
        print(f"  [error] {filename}: {e}")
        return None


def download_ais_range(
    start: date, end: date, year: int = 2024, max_workers: int = 4
) -> list[Path]:
    """Download AIS data for a date range (parallel)."""
    print(f"Downloading AIS data: {start} to {end} ({(end - start).days + 1} days)")
    paths = []
    dates = list(date_range(start, end))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(download_ais_day, d, year): d for d in dates}
        for future in as_completed(futures):
            result = future.result()
            if result:
                paths.append(result)

    print(f"Downloaded {len(paths)}/{len(dates)} files")
    return sorted(paths)


def parse_ais_zip(
    zip_path: Path,
    region_key: str | None = None,
    large_vessels_only: bool = True,
    moving_only: bool = True,
) -> pd.DataFrame:
    """Parse AIS zip into DataFrame, optionally filtering by region/vessel type."""
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            return pd.DataFrame()

        dfs = []
        for csv_name in csv_names:
            with zf.open(csv_name) as f:
                df = pd.read_csv(
                    io.TextIOWrapper(f, encoding="utf-8"),
                    dtype=AIS_DTYPES,
                    parse_dates=["base_date_time"],
                    low_memory=False,
                )
                dfs.append(df)

        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

    if large_vessels_only:
        df = df[df["vessel_type"].isin(LARGE_VESSEL_TYPE_RANGE)].copy()

    if moving_only:
        df = df[df["sog"] >= MIN_SHIP_SPEED_KNOTS].copy()

    if region_key and region_key in REGIONS:
        bounds = REGIONS[region_key]["bounds"]
        min_lon, min_lat, max_lon, max_lat = bounds
        df = df[
            (df["longitude"] >= min_lon)
            & (df["longitude"] <= max_lon)
            & (df["latitude"] >= min_lat)
            & (df["latitude"] <= max_lat)
        ].copy()

    return df


def load_local_ais_csv(csv_path: Path, **filter_kwargs) -> pd.DataFrame:
    """Load a pre-extracted AIS CSV."""
    df = pd.read_csv(csv_path, dtype=AIS_DTYPES, parse_dates=["base_date_time"], low_memory=False)
    return _apply_filters(df, **filter_kwargs)


def load_local_ais_directory(
    directory: Path, pattern: str = "*.csv", **filter_kwargs
) -> pd.DataFrame:
    """Load all AIS CSVs from a directory."""
    csv_files = sorted(directory.glob(pattern))
    if not csv_files:
        print(f"No CSV files found in {directory}")
        return pd.DataFrame()

    print(f"Loading {len(csv_files)} CSV files from {directory}")
    dfs = []
    for i, f in enumerate(csv_files):
        print(f"  [{i+1}/{len(csv_files)}] {f.name}")
        dfs.append(load_local_ais_csv(f, **filter_kwargs))

    return pd.concat(dfs, ignore_index=True)


def _apply_filters(
    df: pd.DataFrame,
    region_key: str | None = None,
    large_vessels_only: bool = True,
    moving_only: bool = True,
) -> pd.DataFrame:
    if large_vessels_only:
        df = df[df["vessel_type"].isin(LARGE_VESSEL_TYPE_RANGE)].copy()
    if moving_only:
        df = df[df["sog"] >= MIN_SHIP_SPEED_KNOTS].copy()
    if region_key and region_key in REGIONS:
        bounds = REGIONS[region_key]["bounds"]
        min_lon, min_lat, max_lon, max_lat = bounds
        df = df[
            (df["longitude"] >= min_lon)
            & (df["longitude"] <= max_lon)
            & (df["latitude"] >= min_lat)
            & (df["latitude"] <= max_lat)
        ].copy()
    return df


def process_ais_data(
    start: date,
    end: date,
    year: int = 2024,
    region_key: str | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Full pipeline: download, parse, filter, and optionally save."""
    zip_paths = download_ais_range(start, end, year)

    dfs = []
    for zp in zip_paths:
        print(f"  Parsing {zp.name} ...")
        df = parse_ais_zip(zp, region_key=region_key)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        print("No ship data found for given parameters.")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    combined["vessel_category"] = combined["vessel_type"].map(
        lambda x: "cargo" if 70 <= x < 80 else "tanker" if 80 <= x < 90 else "passenger" if 60 <= x < 70 else "other"
    )

    if save:
        region_tag = f"_{region_key}" if region_key else ""
        out_name = f"ships_{year}_{start.isoformat()}_{end.isoformat()}{region_tag}.parquet"
        out_path = PROCESSED_DIR / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(out_path, index=False)
        print(f"Saved {len(combined):,} ship records to {out_path}")

    return combined
