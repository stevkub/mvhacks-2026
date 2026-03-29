# Deep Route - Whale/Ship Collision Risk Analysis

Predicting and preventing whale-ship collisions using 7.7M ship records, 310K whale sightings, and a PyTorch neural network.

## Quick Start (View Results Only)

If you just want to see the outputs, open these files in a browser — no setup required:

- **`whale_ship_map.html`** — Interactive multi-layer risk map with route simulation
- **`nn_collision_report.html`** — Neural network prediction report with maps and charts

## Setup (Full Pipeline)

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/under-the-sea.git
cd under-the-sea
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. PyTorch is only needed for the neural network model — everything else runs without it.

### 3. Regenerate outputs from included processed data

The repo includes pre-processed data in `data/processed/` so you can regenerate the map and NN report without the raw AIS files:

```bash
# Regenerate the interactive map
python generate_map.py

# Regenerate the neural network report
python nn_collision_model.py
```

### 4. (Optional) Full pipeline from scratch

To rebuild everything from raw data, you need the AIS ship data files from NOAA.

#### Get AIS data

Download monthly AIS CSV files from [NOAA Marine Cadastre](https://marinecadastre.gov/ais/). You need one file per month named `ais-2024-MM-01.csv` (e.g., `ais-2024-01-01.csv` through `ais-2024-12-01.csv`), placed in the project root.

These are ~750 MB–1 GB each (~10.3 GB total) which is why they aren't included in the repo.

#### Download whale data from OBIS

```bash
# All regions (takes ~10-15 minutes, calls the OBIS API)
python main.py download-whales --region us_west_coast
python main.py download-whales --region us_east_coast
python main.py download-whales --region gulf_of_mexico
python main.py download-whales --region hawaii
python main.py download-whales --region alaska
```

#### Process AIS data

```bash
# Filters all 12 AIS CSVs to large vessels, clips to regions, saves as parquet
python process_all_regions.py
```

#### Run spatial analysis

```bash
python main.py analyze --region us_west_coast
python main.py analyze --region us_east_coast
python main.py analyze --region gulf_of_mexico
python main.py analyze --region hawaii
```

#### Precompute map grids

```bash
python precompute.py
```

#### Generate outputs

```bash
python generate_map.py
python nn_collision_model.py
```

## Project Structure

```
├── main.py                    # CLI entry point
├── process_all_regions.py     # Multi-region AIS processing
├── precompute.py              # Density grids, slow zones, confidence layers
├── generate_map.py            # Interactive map builder
├── nn_collision_model.py      # Standalone PyTorch collision risk model
├── src/
│   ├── config.py              # Regions, species, parameters
│   ├── ship_data.py           # AIS download/parse/filter
│   ├── whale_data.py          # OBIS API client
│   ├── spatial_analysis.py    # Density, hotspots, encounters
│   └── prediction_model.py    # Route risk prediction engine
├── data/
│   ├── raw/whales/            # Raw OBIS whale sighting CSVs
│   └── processed/             # Parquet outputs + precomputed JSON
├── whale_ship_map.html        # Generated interactive map
├── nn_collision_report.html   # Generated NN report
├── NN_WRITEUP.md              # Neural network technical writeup
├── PRESENTATION.md            # Hackathon presentation slides
└── requirements.txt
```

## Features

- **Interactive Map** — Multi-region heatmaps for ships and whales, toggleable risk hotspots, close encounter markers, slow-down zone overlays, and a data confidence layer
- **Route Simulation** — Draw a shipping route on the map to see per-segment risk scoring, a Chart.js risk profile, and economic impact estimates
- **Safer Rerouting** — Automatic pathfinder suggests a lower-risk alternative route with side-by-side cost comparison
- **Speed Slider** — "What-if" tool that recalculates fuel, time, and total cost as you drag between 6–22 knots
- **Slow-Down Zones** — Data-driven polygons marking where ships should reduce speed, generated from the overlap of real ship traffic and whale habitat
- **Neural Network** — PyTorch model trained on 18 spatial/temporal/biological features to predict collision risk for any ocean grid cell in any month

## Data Sources

| Source | Description |
|--------|-------------|
| [NOAA Marine Cadastre AIS](https://marinecadastre.gov/ais/) | 2024 ship positions (12 monthly snapshots, ~7.7M records after filtering) |
| [OBIS](https://obis.org/) | Whale sightings 2000–2024 for 7 major species (~310K records) |

## CLI Reference

```bash
python main.py list-regions              # Show available regions
python main.py list-species              # Show tracked whale species
python main.py download-whales --region us_west_coast
python main.py analyze --region us_west_coast
python main.py predict-route --region us_west_coast --route path/to/route.csv
python main.py predict-route --region us_west_coast --mmsi 367481310
```
