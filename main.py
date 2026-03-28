"""Under the Sea — CLI for the data processing pipeline."""

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

from src.config import REGIONS, WHALE_SPECIES, PROCESSED_DIR
from src.ship_data import process_ais_data, load_local_ais_directory, load_local_ais_csv
from src.whale_data import process_whale_data, load_whale_data
from src.spatial_analysis import run_spatial_analysis
from src.prediction_model import (
    EncounterPredictor, analyze_route,
    load_ship_route, extract_ship_route_from_ais,
)


def cmd_download_ships(args):
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    ships_df = process_ais_data(start, end, year=args.year, region_key=args.region)
    print(f"\nShip data: {len(ships_df):,} records")
    if not ships_df.empty:
        print(ships_df[["vessel_name", "vessel_category", "sog", "latitude", "longitude"]].head(10))


def cmd_load_local_ships(args):
    path = Path(args.path)
    kwargs = {"region_key": args.region, "large_vessels_only": True, "moving_only": True}

    if path.is_dir():
        ships_df = load_local_ais_directory(path, **kwargs)
    elif path.suffix == ".csv":
        ships_df = load_local_ais_csv(path, **kwargs)
    else:
        print(f"Unsupported path: {path}")
        return

    print(f"\nLoaded {len(ships_df):,} ship records")
    if not ships_df.empty:
        tag = f"_{args.region}" if args.region else ""
        out = PROCESSED_DIR / f"ships_local{tag}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        ships_df["vessel_category"] = ships_df["vessel_type"].map(
            lambda x: "cargo" if 70 <= x < 80 else "tanker" if 80 <= x < 90 else "passenger" if 60 <= x < 70 else "other"
        )
        ships_df.to_parquet(out, index=False)
        print(f"Saved to {out}")
        print(ships_df[["vessel_name", "vessel_type", "sog", "latitude", "longitude"]].head(10))


def cmd_download_whales(args):
    whales_df = process_whale_data(
        region_key=args.region,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    print(f"\nWhale data: {len(whales_df):,} records")
    if not whales_df.empty:
        print("\nRecords per species:")
        print(whales_df["species_common"].value_counts())


def cmd_analyze(args):
    import pandas as pd

    region = args.region
    ship_files = sorted(PROCESSED_DIR.glob(f"ships*{region}*.parquet"))
    if not ship_files:
        print(f"No processed ship data for region '{region}'. Run download-ships or load-local-ships first.")
        return
    ships_df = pd.concat([pd.read_parquet(f) for f in ship_files], ignore_index=True)
    print(f"Loaded {len(ships_df):,} ship records")

    whales_df = load_whale_data(region)
    if whales_df.empty:
        print(f"No processed whale data for region '{region}'. Run download-whales first.")
        return
    print(f"Loaded {len(whales_df):,} whale records")

    results = run_spatial_analysis(ships_df, whales_df, region)

    print(f"\n--- Results Summary ---")
    print(f"Risk hotspots: {len(results['hotspots'])}")
    print(f"Close encounters: {len(results['encounters'])}")
    if not results["hotspots"].empty:
        print(f"\nTop 5 hotspots:")
        print(results["hotspots"].head())


def cmd_predict_route(args):
    import pandas as pd

    region = args.region
    whales_df = load_whale_data(region)
    if whales_df.empty:
        print(f"No whale data for region '{region}'. Run download-whales first.")
        return

    if args.route:
        route_df = load_ship_route(Path(args.route))
        print(f"Loaded route with {len(route_df)} waypoints")
    elif args.mmsi or args.vessel_name:
        ship_files = sorted(PROCESSED_DIR.glob(f"ships*{region}*.parquet"))
        if not ship_files:
            print("No ship data found. Download or load ship data first.")
            return
        ships_df = pd.concat([pd.read_parquet(f) for f in ship_files], ignore_index=True)
        route_df = extract_ship_route_from_ais(
            ships_df,
            mmsi=int(args.mmsi) if args.mmsi else None,
            vessel_name=args.vessel_name,
        )
        print(f"Extracted route with {len(route_df)} waypoints")
    else:
        print("Provide --route, --mmsi, or --vessel-name")
        return

    result = analyze_route(whales_df, route_df, region, month=args.month)

    print("\nRisk distribution:")
    print(result["risk_level"].value_counts())

    high_risk = result[result["risk_level"].isin(["high", "critical"])]
    if not high_risk.empty:
        print(f"\nHigh-risk waypoints:")
        print(high_risk[["latitude", "longitude", "risk_score", "risk_level", "whale_density"]].head(10))


def cmd_full_pipeline(args):
    print("=" * 60)
    print("FULL PIPELINE")
    print("=" * 60)

    print("\n[1/3] Downloading ship data...")
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    ships_df = process_ais_data(start, end, year=args.year, region_key=args.region)
    print(f"  -> {len(ships_df):,} ship records")

    print("\n[2/3] Downloading whale data...")
    whales_df = process_whale_data(region_key=args.region, start_year=args.start_year, end_year=args.end_year)
    print(f"  -> {len(whales_df):,} whale records")

    if ships_df.empty or whales_df.empty:
        print("\nInsufficient data for analysis.")
        return

    print("\n[3/3] Running spatial analysis...")
    results = run_spatial_analysis(ships_df, whales_df, args.region)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  Ship records: {len(ships_df):,}")
    print(f"  Whale records: {len(whales_df):,}")
    print(f"  Risk hotspots: {len(results['hotspots'])}")
    print(f"  Close encounters: {len(results['encounters'])}")
    print("=" * 60)


def cmd_list_regions(args):
    print("Available regions:")
    for key, info in REGIONS.items():
        print(f"  {key:20s} - {info['name']}")
        print(f"  {'':20s}   {info['description']}")
        print(f"  {'':20s}   Bounds: {info['bounds']}")
        print()


def cmd_list_species(args):
    print("Tracked whale species:")
    for sci, common in WHALE_SPECIES.items():
        print(f"  {common:30s} ({sci})")


def main():
    parser = argparse.ArgumentParser(
        description="Under the Sea - Whale-Ship Interaction Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # download-ships
    p = sub.add_parser("download-ships", help="Download AIS ship data for a date range")
    p.add_argument("--region", required=True, choices=REGIONS.keys())
    p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    p.add_argument("--year", type=int, default=2024)

    # load-local-ships
    p = sub.add_parser("load-local-ships", help="Load pre-downloaded AIS CSV files")
    p.add_argument("--path", required=True, help="Path to CSV file or directory of CSVs")
    p.add_argument("--region", default=None, choices=list(REGIONS.keys()) + [None])

    # download-whales
    p = sub.add_parser("download-whales", help="Download whale sighting data from OBIS")
    p.add_argument("--region", default=None, choices=list(REGIONS.keys()) + [None])
    p.add_argument("--start-year", type=int, default=2000)
    p.add_argument("--end-year", type=int, default=2024)

    # analyze
    p = sub.add_parser("analyze", help="Run spatial analysis on processed data")
    p.add_argument("--region", required=True, choices=REGIONS.keys())

    # predict-route
    p = sub.add_parser("predict-route", help="Predict encounter risk along a ship route")
    p.add_argument("--region", required=True, choices=REGIONS.keys())
    p.add_argument("--route", default=None, help="Path to route CSV (lon, lat columns)")
    p.add_argument("--mmsi", default=None, help="Extract route from AIS data by MMSI")
    p.add_argument("--vessel-name", default=None, help="Extract route by vessel name")
    p.add_argument("--month", type=int, default=None, help="Month for seasonal weighting (1-12)")

    # full-pipeline
    p = sub.add_parser("full-pipeline", help="Run entire pipeline: download -> analyze")
    p.add_argument("--region", required=True, choices=REGIONS.keys())
    p.add_argument("--start", required=True, help="Start date for ship data (YYYY-MM-DD)")
    p.add_argument("--end", required=True, help="End date for ship data (YYYY-MM-DD)")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--start-year", type=int, default=2000, help="Whale data start year")
    p.add_argument("--end-year", type=int, default=2024, help="Whale data end year")

    # list commands
    sub.add_parser("list-regions", help="Show available analysis regions")
    sub.add_parser("list-species", help="Show tracked whale species")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        "download-ships": cmd_download_ships,
        "load-local-ships": cmd_load_local_ships,
        "download-whales": cmd_download_whales,
        "analyze": cmd_analyze,
        "predict-route": cmd_predict_route,
        "full-pipeline": cmd_full_pipeline,
        "list-regions": cmd_list_regions,
        "list-species": cmd_list_species,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
