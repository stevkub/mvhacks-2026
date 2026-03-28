"""Whale sighting data: download and parse from OBIS API."""

import time
import json
import requests
import pandas as pd
from pathlib import Path

from .config import (
    OBIS_API_URL, RAW_WHALES_DIR, PROCESSED_DIR,
    WHALE_SPECIES, REGIONS,
)

OBIS_FIELDS = [
    "scientificName", "decimalLatitude", "decimalLongitude",
    "eventDate", "date_year", "month", "day",
    "individualCount", "basisOfRecord", "datasetName",
    "vernacularName", "country", "bathymetry", "sst",
]

OBIS_PAGE_SIZE = 5000
OBIS_RATE_LIMIT_DELAY = 0.5


def _build_obis_params(
    species: str,
    region_key: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    size: int = OBIS_PAGE_SIZE,
    offset: int = 0,
) -> dict:
    params = {
        "scientificname": species,
        "size": size,
        "skip": offset,
    }
    if region_key and region_key in REGIONS:
        bounds = REGIONS[region_key]["bounds"]
        min_lon, min_lat, max_lon, max_lat = bounds
        wkt = f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))"
        params["geometry"] = wkt

    if start_year:
        params["startdate"] = f"{start_year}-01-01"
    if end_year:
        params["enddate"] = f"{end_year}-12-31"

    return params


def fetch_whale_occurrences(
    species: str,
    region_key: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    max_records: int = 50000,
) -> pd.DataFrame:
    """Paginated fetch of whale occurrences from OBIS."""
    common_name = WHALE_SPECIES.get(species, species)
    print(f"Fetching {common_name} ({species}) from OBIS...")

    all_records = []
    offset = 0

    while offset < max_records:
        params = _build_obis_params(species, region_key, start_year, end_year, OBIS_PAGE_SIZE, offset)
        try:
            resp = requests.get(OBIS_API_URL, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  Error at offset {offset}: {e}")
            break

        results = data.get("results", [])
        total = data.get("total", 0)
        print(f"  Fetched {offset + len(results)}/{total} records")

        all_records.extend(results)
        offset += len(results)

        if len(results) < OBIS_PAGE_SIZE or offset >= total:
            break

        time.sleep(OBIS_RATE_LIMIT_DELAY)

    if not all_records:
        return pd.DataFrame()

    df = pd.json_normalize(all_records)

    keep_cols = [c for c in OBIS_FIELDS if c in df.columns]
    df = df[keep_cols].copy()

    df = df.rename(columns={
        "decimalLatitude": "latitude",
        "decimalLongitude": "longitude",
    })

    df["species_common"] = common_name
    df["species_scientific"] = species

    df = df.dropna(subset=["latitude", "longitude"])

    return df


def fetch_all_whale_species(
    region_key: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    max_per_species: int = 50000,
    save_raw: bool = True,
) -> pd.DataFrame:
    """Fetch all configured whale species."""
    all_dfs = []

    for species in WHALE_SPECIES:
        df = fetch_whale_occurrences(species, region_key, start_year, end_year, max_per_species)
        if not df.empty:
            all_dfs.append(df)

            if save_raw:
                safe_name = species.replace(" ", "_").lower()
                region_tag = f"_{region_key}" if region_key else ""
                raw_path = RAW_WHALES_DIR / f"{safe_name}{region_tag}.csv"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(raw_path, index=False)
                print(f"  Saved {len(df)} raw records to {raw_path}")

    if not all_dfs:
        print("No whale data found.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


def process_whale_data(
    region_key: str | None = None,
    start_year: int | None = 2000,
    end_year: int | None = 2024,
    save: bool = True,
) -> pd.DataFrame:
    """Full pipeline: fetch all species, clean, and save."""
    combined = fetch_all_whale_species(region_key, start_year, end_year)

    if combined.empty:
        return combined

    if "date_year" in combined.columns:
        combined["date_year"] = pd.to_numeric(combined["date_year"], errors="coerce")
    if "month" in combined.columns:
        combined["month"] = pd.to_numeric(combined["month"], errors="coerce")

    if "individualCount" in combined.columns:
        combined["individualCount"] = pd.to_numeric(combined["individualCount"], errors="coerce").fillna(1)

    if save:
        region_tag = f"_{region_key}" if region_key else ""
        out_path = PROCESSED_DIR / f"whales{region_tag}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(out_path, index=False)
        print(f"Saved {len(combined):,} whale records to {out_path}")

    return combined


def load_whale_data(region_key: str | None = None) -> pd.DataFrame:
    """Load previously processed whale data."""
    region_tag = f"_{region_key}" if region_key else ""
    path = PROCESSED_DIR / f"whales{region_tag}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    print(f"No processed whale data found at {path}. Run process_whale_data() first.")
    return pd.DataFrame()
