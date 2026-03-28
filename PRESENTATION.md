# Under the Sea — Whale-Ship Collision Risk Analysis

### MVHacks 2026 Presentation

---

## Slide 1: Title

# 🐋 Under the Sea

**Predicting and preventing whale-ship collisions with data science and neural networks**

*Built with Python, PyTorch, Folium, and 10+ GB of real maritime data*

---

## Slide 2: The Problem

### Every year, ships kill whales.

- Ship strikes are the **#1 human cause** of large whale deaths worldwide
- The endangered **North Atlantic Right Whale** has fewer than **350 individuals** left — ship strikes account for over 50% of known deaths
- Shipping routes overlap directly with whale migration corridors, breeding grounds, and feeding areas
- Current mitigation is limited: seasonal speed restrictions in a few known areas, but **most of the ocean has no protection**

### Why existing solutions fall short:
- Static speed zones don't adapt to seasonal whale migration
- Ship operators have **no real-time risk assessment** for their routes
- No tool lets you **simulate a route** and see the economic cost of going safer

---

## Slide 3: Our Solution

### A complete whale-ship collision risk platform

| Component | What it does |
|-----------|-------------|
| **Data Pipeline** | Ingests 10+ GB of AIS ship data + whale sightings from OBIS |
| **Spatial Analysis** | KD-tree proximity matching, density grids, hotspot detection |
| **Interactive Map** | Multi-layer Folium map with heatmaps, hotspots, encounters |
| **Route Simulator** | Draw a route → see risk per segment + economic impact |
| **Safer Rerouting** | Automatic pathfinder suggests lower-risk alternative |
| **Speed Slider** | "What-if" tool: drag to see cost of speed changes in real time |
| **Slow-Down Zones** | Data-driven recommended speed reduction areas |
| **Neural Network** | PyTorch model predicts collision risk from 18 features |

---

## Slide 4: Data Sources

### Ship Data — NOAA Marine Cadastre AIS

12 monthly AIS snapshots covering all of 2024.

**Raw CSV format** (each file 750 MB – 1 GB):
```
mmsi       base_date_time       longitude  latitude   sog   vessel_name       vessel_type  length
338075892  2024-01-01 00:00:03  -70.25298  43.65322   0.0   PILOT BOAT SP..   90           NaN
367669550  2024-01-01 00:00:04  -123.3857  46.20031  15.9   ALASKA CHALL..    30           30.0
636022875  2024-01-01 00:00:07  -124.6090  46.17595  15.9   MSC ATHOS         71           300.0
```

**12 files × ~850 MB each = ~10.3 GB of raw AIS data**

After filtering to large vessels (types 60-89) and clipping to 5 regions:

| Region | Ship Records | File Size |
|--------|-------------|-----------|
| US West Coast | 2,228,350 | 73 MB (parquet) |
| US East Coast | 2,867,402 | 93 MB |
| Gulf of Mexico | 2,625,399 | 86 MB |
| Hawaii | 54,386 | 1.9 MB |
| Alaska | 130 | 15 KB |
| **Total** | **7,775,667** | |

---

## Slide 5: Data Sources (continued)

### Whale Data — OBIS (Ocean Biodiversity Information System)

Fetched via paginated REST API for 7 major species across 5 regions, spanning 2000–2024.

**Raw whale sighting record:**
```
latitude   longitude   eventDate            month  species_common   individualCount  sst    bathymetry
35.500000  -123.500    2004-09-09T00:00:00  9.0    Blue Whale       6.0              14.32  3985
33.848900  -120.117    2019-07-23           NaN    Blue Whale       1.0              14.82  241
34.085500  -120.522    2021-10-12           NaN    Blue Whale       1.0              14.15  70
```

**Species tracked:**

| Species | West Coast | East Coast | Gulf | Hawaii |
|---------|-----------|-----------|------|--------|
| Humpback Whale | 50,000 | — | — | 50,000+ |
| Gray Whale | 25,000 | — | — | — |
| Blue Whale | 10,000 | — | — | — |
| Fin Whale | 3,930 | 11,000+ | — | — |
| N. Atlantic Right Whale | — | 8,000+ | — | — |
| Sperm Whale | 222 | 1,200+ | 6,000+ | — |
| Minke Whale | 253 | 3,500+ | — | — |
| **Total** | **89,405** | **39,671** | **10,110** | **130,127** |

**Grand total: 310,891 whale sighting records**

---

## Slide 6: Data Processing Pipeline

### How raw data becomes actionable insight

```
┌─────────────────────────────────────────────────────┐
│  12 × AIS CSV files (10.3 GB total)                 │
│  ├── Filter: vessel types 60-89 (large commercial)  │
│  ├── Filter: SOG ≥ 1 knot (moving vessels only)     │
│  ├── Clip: to 5 geographic bounding boxes            │
│  └── Tag: month, vessel_category (cargo/tanker/etc) │
│  → Saved as Parquet (compressed ~97%)                │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────┐
│  OBIS API (paginated, rate-limited)                  │
│  ├── 7 species × 5 regions × 5000-record pages      │
│  ├── Fields: lat, lon, date, SST, bathymetry, count │
│  └── Clean: coerce types, fill individualCount       │
│  → Saved as Parquet                                  │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────┐
│  Spatial Analysis (per region)                       │
│  ├── 2D density grids (0.1° cells, ~11 km)          │
│  ├── Hotspot detection (log-normalized product)      │
│  ├── KD-tree close encounters (<5 km)                │
│  ├── Cargo/tanker-specific analysis                  │
│  └── Route deviation index                           │
│  → 10,442,961 encounters found                      │
└─────────────────────────────────────────────────────┘
```

---

## Slide 7: Raw Processing Output

### Ship data processing (single-pass multi-region):
```
Found 12 AIS CSV files

[01/12] Loading ais-2024-01-01.csv ...
  5,261,442 raw, filtered to large moving vessels
[02/12] Loading ais-2024-02-01.csv ...
  5,450,897 raw, filtered to large moving vessels
  ...
[12/12] Loading ais-2024-12-01.csv ...
  5,723,918 raw, filtered to large moving vessels

  us_west_coast:    2,228,350 records saved
  us_east_coast:    2,867,402 records saved
  gulf_of_mexico:   2,625,399 records saved
  hawaii:              54,386 records saved
  alaska:                 130 records saved
```

### Encounter detection output:
```
=== Spatial Analysis: US West Coast ===
  [all] Computing ship density grid...
  [all] Finding close encounters...
  Querying 89,405 whale positions against 150,000 ship positions...
  [all] Found 4,635,944 close encounters
  [cargo+tanker] Found 2,401,838 close encounters
```

---

## Slide 8: Interactive Map — Overview

### Multi-layer Folium map with everything toggleable

**Layers available:**
- 🔶 Ship density heatmaps (per region, orange gradient)
- 🐋 Whale density heatmaps (per region, teal gradient)
- 🔴 Risk hotspots — all vessels (size = risk score)
- 🔴 Risk hotspots — cargo/tanker only
- ⚠️ Close encounter markers (color-coded by species)
- 🚨 Slow-down zone polygons (red overlays)
- 📊 Data confidence/coverage grid

**Base maps:** Esri Ocean, Satellite, CARTO Dark

**Dashboard:** Aggregate stats, per-region breakdown, legend, how-to-use guide

---

## Slide 9: Route Simulation Tool

### Draw a route → instant risk analysis

1. Use the draw tool to sketch a shipping route on the map
2. The system interpolates points every 2 km along the route
3. At each point, it samples the precomputed density grids:
   - Ship density at that location
   - Whale density at that location
   - Combined risk score = ship × whale density
4. The route is colored in real time:
   - 🟢 Green = low risk
   - 🟡 Yellow = moderate
   - 🔴 Red = high risk
5. A panel appears with:
   - Risk profile chart (Chart.js)
   - Total distance, estimated transit time
   - High-risk segment percentage
   - Economic impact estimate

---

## Slide 10: Safer Rerouting

### Automatic pathfinding for lower-risk alternatives

When a route is drawn, the system also computes a **safer alternative**:

- At each point along the original route, it tests lateral offsets (perpendicular to heading)
- Offsets tested: ±5 km, ±10 km, ±15 km, ±20 km
- Picks the offset that minimizes whale density while keeping the route practical
- The safer route is drawn as a **dashed green line**

**Comparison panel shows:**
```
┌──────────────────────────────────────────┐
│ Original Route      │ Safer Route        │
│ Avg risk: 0.42      │ Avg risk: 0.28     │
│ Max risk: 0.91      │ Max risk: 0.65     │
│ Fuel cost: $48,200  │ Fuel cost: $49,100 │
│ Time: 14.2 hr       │ Time: 14.8 hr      │
│                      │ Risk reduction: 33% │
│                      │ Extra cost: +$900   │
└──────────────────────────────────────────┘
```

---

## Slide 11: "What-If" Speed Slider

### Real-time economic impact of speed changes

A slider (6–22 knots) lets you instantly see how speed affects:

- **Fuel cost** — scales with speed³ (cubic relationship from marine engineering)
- **Transit time** — inversely proportional to speed
- **Crew cost** — proportional to time
- **Total voyage cost** — fuel + crew + port fees
- **Comparison** — original vs. safer route at any speed

Dragging from 14 knots down to 10 knots might show:
- Fuel savings: -45%
- Extra time: +8 hours
- Net cost change: -$12,000

---

## Slide 12: Slow-Down Zones

### Data-driven speed reduction recommendations

**How they're generated:**
1. Compute normalized ship density × whale density for each grid cell
2. Take the 90th percentile as the risk threshold
3. Filter: only cells with **both** ship observations **and** whale observations (prevents land zones)
4. Cluster nearby cells using hierarchical clustering (Chebyshev distance)
5. Generate convex hull polygons around each cluster

These aren't arbitrary — they're derived directly from the overlap of real ship traffic and real whale habitat.

---

## Slide 13: Neural Network Model

### PyTorch collision risk predictor

**Architecture:** Input(18) → 128 → 256 → 128 → 64 → 1

**18 input features per grid cell × month:**

| Category | Features |
|----------|----------|
| Spatial | lat (normalized), lon (normalized) |
| Temporal | month (sin/cos), season (sin/cos) |
| Traffic | ship count, avg SOG, std SOG, avg length, cargo/tanker fraction, unique vessels |
| Biological | whale count (habitat), whale monthly count (seasonal), species richness, avg individual count |
| Environmental | sea surface temperature, bathymetry |

**Target:** log-transformed encounter count (normalized)

**Training:** Adam optimizer, cosine annealing LR, 120 epochs, batch size 512

---

## Slide 14: Neural Network — Results

### The model captures patterns humans would miss

**Key findings from gradient-based feature importance:**
1. Ship count and whale count dominate (expected)
2. Seasonal features (month sin/cos) are surprisingly important — the model learned that risk peaks at different times in different regions
3. Bathymetry matters — continental shelf edges concentrate both ships and whales
4. Cargo/tanker fraction correlates with higher risk than passenger vessels

**Per-region top hotspots identified** (30 per region):
- West Coast: Monterey Bay, Santa Barbara Channel, San Francisco approach
- East Coast: Cape Cod, Chesapeake approach, Jacksonville calving grounds
- Gulf: Mississippi Delta, Tampa approach
- Hawaii: Maui Nui basin, Kauai channel

The model produces a self-contained HTML report with maps, charts, and tables.

---

## Slide 15: Challenges & Solutions

### 1. Scale: 10 GB of raw data
**Problem:** Each AIS CSV is 750 MB–1 GB. Loading all 12 at once would require 50+ GB RAM.
**Solution:** Single-pass processing — read each CSV once, filter and distribute to regions simultaneously, save as compressed Parquet (~97% compression).

### 2. Proximity matching: O(n²) doesn't work
**Problem:** Finding encounters between millions of ship and whale positions.
**Solution:** KD-tree spatial indexing with `scipy.spatial.cKDTree`. Converts to radians, uses `query_ball_tree` for batch matching. Reduced from hours to seconds.

### 3. Whale data sparsity
**Problem:** OBIS whale sightings are sparse per specific month — most grid cells showed 0 whales when filtered by month.
**Solution:** Dual-layer approach: `whale_count` from all-time data (habitat presence) + `whale_month_count` from month-filtered data (seasonal activity). The NN learns from both.

### 4. Slow zones appearing on land
**Problem:** Grid cells near the coast picked up coastal ship traffic even when centered inland.
**Solution:** Added filter requiring both ship AND whale observations in a cell — since whales only exist in water, this effectively filters out land.

### 5. Map performance
**Problem:** Rendering millions of data points on a web map was slow.
**Solution:** Aggressive sampling (10K points per heatmap), sparse confidence grid (every 3rd cell), client-side JS for route analysis instead of server round-trips.

---

## Slide 16: Technical Stack

| Tool | Purpose |
|------|---------|
| Python 3.12 | Core language |
| Pandas | Data manipulation and I/O |
| NumPy | Numerical computation |
| SciPy | KD-trees, clustering, gaussian smoothing, interpolation |
| PyTorch | Neural network training and inference |
| Folium | Interactive map generation |
| Chart.js | Client-side route risk charts |
| OBIS REST API | Whale sighting data |
| NOAA AIS | Ship position data |
| Parquet | Efficient columnar storage |

---

## Slide 17: Project Structure

```
mvhacks/
├── main.py                    # CLI orchestrator
├── process_all_regions.py     # Multi-region AIS processing
├── precompute.py              # Density grids, slow zones, confidence
├── generate_map.py            # Interactive map builder (~900 lines)
├── nn_collision_model.py      # Standalone neural network
├── NN_WRITEUP.md              # Detailed NN documentation
├── src/
│   ├── config.py              # Regions, species, parameters
│   ├── ship_data.py           # AIS download/parse/filter
│   ├── whale_data.py          # OBIS API client
│   ├── spatial_analysis.py    # Density, hotspots, encounters
│   └── prediction_model.py    # Route risk prediction
├── data/
│   ├── raw/                   # Downloaded source files
│   └── processed/             # Parquet outputs
├── whale_ship_map.html        # Generated interactive map
└── nn_collision_report.html   # Generated NN report
```

---

## Slide 18: Numbers at a Glance

| Metric | Value |
|--------|-------|
| Raw AIS data processed | **10.3 GB** (12 monthly files) |
| Ship position records | **7,775,667** |
| Whale sighting records | **310,891** |
| Close encounters detected | **10,442,961** |
| Regions analyzed | **5** (West Coast, East Coast, Gulf, Hawaii, Alaska) |
| Whale species tracked | **7** |
| NN features per sample | **18** |
| NN training samples | ~50,000+ grid cells × 12 months |
| Map layers | **15+** toggleable layers |
| Lines of Python | ~3,000 |

---

## Slide 19: Demo

### Live walkthrough

1. **Open the interactive map** — show multi-region heatmaps
2. **Toggle layers** — ship density vs. whale density vs. hotspots
3. **Draw a route** — show the risk analysis panel and Chart.js profile
4. **Drag the speed slider** — show real-time economic impact changes
5. **Show the safer reroute** — dashed green line with comparison stats
6. **Toggle slow zones** — recommended speed reduction areas
7. **Open the NN report** — prediction maps, feature importance, top hotspots

---

## Slide 20: Impact & Future Work

### What this enables:
- **Ship operators** can plan routes that avoid whale habitats
- **Regulators** can use data-driven evidence for speed restriction zones
- **Researchers** get a tool that combines AIS and biological data at scale
- **The public** can see exactly where the conflict between shipping and conservation is happening

### Future directions:
- Real-time AIS integration (live ship positions)
- Satellite-tagged whale movement data
- Weather and current overlays affecting whale behavior
- Mobile app for ship bridge officers
- Integration with IMO regulatory frameworks

---

## Slide 21: Thank You

# 🐋 Under the Sea

**Protecting whales through data-driven shipping intelligence**

*Questions?*
