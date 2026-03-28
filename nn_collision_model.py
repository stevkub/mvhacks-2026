"""Standalone PyTorch neural network for whale-ship collision risk prediction."""

import math
import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data" / "processed"
OUTPUT_HTML = Path(__file__).parent / "nn_collision_report.html"

REGIONS = {
    "us_west_coast": (-130, 30, -115, 50),
    "us_east_coast": (-82, 24, -65, 45),
    "gulf_of_mexico": (-98, 18, -80, 31),
    "hawaii": (-162, 17, -153, 24),
}

REGION_NAMES = {
    "us_west_coast": "US West Coast",
    "us_east_coast": "US East Coast",
    "gulf_of_mexico": "Gulf of Mexico",
    "hawaii": "Hawaii",
}

CELL_SIZE = 0.25  # degrees — coarser grid for NN cells to ensure enough data per cell
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Data loading ---

def load_region_data(region_key):
    ships_path = DATA_DIR / f"ships_2024_monthly_{region_key}.parquet"
    whales_path = DATA_DIR / f"whales_{region_key}.parquet"
    enc_path = DATA_DIR / f"encounters_{region_key}.parquet"

    ships = pd.read_parquet(ships_path) if ships_path.exists() else pd.DataFrame()
    whales = pd.read_parquet(whales_path) if whales_path.exists() else pd.DataFrame()
    encounters = pd.read_parquet(enc_path) if enc_path.exists() else pd.DataFrame()
    return ships, whales, encounters


# --- Feature engineering (grid cell × month) ---

def build_dataset():
    """Create training dataset from all regions: one sample per grid cell per month."""
    all_rows = []

    for rk, (min_lon, min_lat, max_lon, max_lat) in REGIONS.items():
        print(f"  Building features for {REGION_NAMES[rk]}...")
        ships, whales, encounters = load_region_data(rk)
        if ships.empty or whales.empty:
            continue

        lons = np.arange(min_lon, max_lon, CELL_SIZE)
        lats = np.arange(min_lat, max_lat, CELL_SIZE)

        # Pre-index encounter locations with KD-tree for fast lookup
        enc_coords = None
        if not encounters.empty and "whale_lat" in encounters.columns:
            enc_coords = encounters[["whale_lat", "whale_lon"]].values
            enc_tree = cKDTree(np.radians(enc_coords))

        # Full whale KD-tree (not month-filtered) — whale habitat is persistent
        whale_all_coords = whales[["latitude", "longitude"]].values if not whales.empty else np.empty((0, 2))
        whale_all_tree = cKDTree(np.radians(whale_all_coords)) if len(whale_all_coords) > 0 else None

        # Per-month whale KD-trees for seasonal signal
        whale_monthly_trees = {}
        for month in range(1, 13):
            wm = whales[whales["month"] == month] if "month" in whales.columns else pd.DataFrame()
            if not wm.empty and len(wm) >= 3:
                wmc = wm[["latitude", "longitude"]].values
                whale_monthly_trees[month] = (cKDTree(np.radians(wmc)), wm)

        radius_rad = (CELL_SIZE * 1.5) * math.pi / 180

        for month in range(1, 13):
            ships_m = ships[ships["month"] == month] if "month" in ships.columns else ships
            if ships_m.empty:
                continue

            ship_coords = ships_m[["latitude", "longitude"]].values
            ship_tree = cKDTree(np.radians(ship_coords))

            for lon in lons:
                for lat in lats:
                    center_rad = np.radians([[lat, lon]])

                    # Ship features within cell
                    s_idx = ship_tree.query_ball_point(center_rad[0], radius_rad)
                    ship_count = len(s_idx)
                    if ship_count == 0:
                        continue

                    s_local = ships_m.iloc[s_idx]
                    avg_sog = s_local["sog"].mean() if "sog" in s_local.columns else 0
                    std_sog = s_local["sog"].std() if "sog" in s_local.columns else 0
                    avg_length = s_local["length"].mean() if "length" in s_local.columns else 0
                    cargo_frac = (s_local["vessel_category"].isin(["cargo", "tanker"]).sum() / ship_count) if "vessel_category" in s_local.columns else 0
                    unique_vessels = s_local["mmsi"].nunique() if "mmsi" in s_local.columns else 0

                    # Whale habitat features from FULL dataset (location-based, not monthly)
                    whale_count = 0
                    n_species = 0
                    avg_sst = 0
                    avg_bathy = 0
                    avg_indiv = 0
                    if whale_all_tree is not None:
                        w_idx = whale_all_tree.query_ball_point(center_rad[0], radius_rad)
                        whale_count = len(w_idx)
                        if whale_count > 0:
                            w_local = whales.iloc[w_idx]
                            n_species = w_local["species_common"].nunique() if "species_common" in w_local.columns else 0
                            avg_sst = w_local["sst"].mean() if "sst" in w_local.columns else 0
                            avg_bathy = w_local["bathymetry"].mean() if "bathymetry" in w_local.columns else 0
                            avg_indiv = w_local["individualCount"].mean() if "individualCount" in w_local.columns else 0

                    # Monthly whale activity — seasonal signal from month-specific sightings
                    whale_month_count = 0
                    if month in whale_monthly_trees:
                        wm_tree, _ = whale_monthly_trees[month]
                        wm_idx = wm_tree.query_ball_point(center_rad[0], radius_rad)
                        whale_month_count = len(wm_idx)

                    # Encounter count (target proxy)
                    enc_count = 0
                    if enc_coords is not None:
                        e_idx = enc_tree.query_ball_point(center_rad[0], radius_rad)
                        enc_count = len(e_idx)

                    _safe = lambda v: 0 if (isinstance(v, float) and math.isnan(v)) else v
                    all_rows.append({
                        "region": rk,
                        "lat": lat,
                        "lon": lon,
                        "month": month,
                        "month_sin": math.sin(2 * math.pi * month / 12),
                        "month_cos": math.cos(2 * math.pi * month / 12),
                        "season_sin": math.sin(2 * math.pi * ((month % 12) // 3) / 4),
                        "season_cos": math.cos(2 * math.pi * ((month % 12) // 3) / 4),
                        "ship_count": ship_count,
                        "avg_sog": _safe(avg_sog),
                        "std_sog": _safe(std_sog),
                        "avg_ship_length": _safe(avg_length),
                        "cargo_tanker_frac": cargo_frac,
                        "unique_vessels": unique_vessels,
                        "whale_count": whale_count,
                        "whale_month_count": whale_month_count,
                        "n_species": n_species,
                        "avg_sst": _safe(avg_sst),
                        "avg_bathymetry": _safe(avg_bathy),
                        "avg_individual_count": _safe(avg_indiv),
                        "lat_norm": (lat - min_lat) / (max_lat - min_lat),
                        "lon_norm": (lon - min_lon) / (max_lon - min_lon),
                        "encounter_count": enc_count,
                    })

    df = pd.DataFrame(all_rows)
    print(f"  Dataset: {len(df):,} samples, {df['encounter_count'].gt(0).sum():,} with encounters")
    return df


# --- Neural network ---

FEATURE_COLS = [
    "lat_norm", "lon_norm",
    "month_sin", "month_cos", "season_sin", "season_cos",
    "ship_count", "avg_sog", "std_sog", "avg_ship_length",
    "cargo_tanker_frac", "unique_vessels",
    "whale_count", "whale_month_count", "n_species",
    "avg_sst", "avg_bathymetry", "avg_individual_count",
]

FEATURE_DISPLAY_NAMES = [
    "Latitude", "Longitude",
    "Month (sin)", "Month (cos)", "Season (sin)", "Season (cos)",
    "Ship Count", "Avg Speed (SOG)", "Speed Variance", "Avg Ship Length",
    "Cargo/Tanker Fraction", "Unique Vessels",
    "Whale Habitat Count", "Whale Monthly Activity", "Species Richness",
    "Sea Surface Temp", "Bathymetry", "Avg Individual Count",
]


class CollisionRiskNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def prepare_tensors(df):
    X = df[FEATURE_COLS].values.astype(np.float32)
    y_raw = df["encounter_count"].values.astype(np.float32)
    y = np.log1p(y_raw)  # log-transform target for stability
    y_max = y.max()
    if y_max > 0:
        y = y / y_max  # normalize to [0, 1]

    # Normalize features (z-score)
    means = X.mean(axis=0)
    stds = X.std(axis=0) + 1e-8
    X = (X - means) / stds

    return X, y, means, stds, y_max


def train_model(X, y, epochs=120, lr=1e-3, batch_size=512):
    n = len(X)
    perm = np.random.RandomState(42).permutation(n)
    split = int(n * 0.8)
    train_idx, val_idx = perm[:split], perm[split:]

    X_train = torch.tensor(X[train_idx], device=DEVICE)
    y_train = torch.tensor(y[train_idx], device=DEVICE).unsqueeze(1)
    X_val = torch.tensor(X[val_idx], device=DEVICE)
    y_val = torch.tensor(y[val_idx], device=DEVICE).unsqueeze(1)

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = CollisionRiskNet(X.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "val_mae": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_dl:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_idx)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
            val_mae = (val_pred - y_val).abs().mean().item()

        scheduler.step()
        history["train_loss"].append(round(epoch_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_mae"].append(round(val_mae, 6))

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  train_loss={epoch_loss:.5f}  val_loss={val_loss:.5f}  val_mae={val_mae:.5f}")

    return model, history


# --- Feature importance (gradient-based) ---

def compute_feature_importance(model, X):
    model.eval()
    X_t = torch.tensor(X, device=DEVICE, requires_grad=True)
    pred = model(X_t)
    pred.sum().backward()
    importance = X_t.grad.abs().mean(dim=0).cpu().numpy()
    importance = importance / (importance.sum() + 1e-10)
    return importance


# --- Prediction grid (every cell × month) ---

def predict_grid(model, df, means, stds):
    """Run predictions on every cell in the dataset, plus generate monthly aggregates."""
    model.eval()
    X = df[FEATURE_COLS].values.astype(np.float32)
    X_norm = (X - means) / stds

    with torch.no_grad():
        preds = model(torch.tensor(X_norm, device=DEVICE)).cpu().numpy().flatten()

    df = df.copy()
    df["predicted_risk"] = preds
    df["predicted_risk"] = df["predicted_risk"].clip(0, 1)
    return df


# --- Monthly pattern analysis ---

def monthly_analysis(pred_df):
    """Aggregate predicted risk by month and region."""
    monthly = pred_df.groupby(["region", "month"]).agg(
        mean_risk=("predicted_risk", "mean"),
        max_risk=("predicted_risk", "max"),
        mean_enc=("encounter_count", "mean"),
        total_enc=("encounter_count", "sum"),
        n_cells=("predicted_risk", "count"),
    ).reset_index()
    return monthly


# --- Top risk hotspots ---

def top_hotspots(pred_df, n_per_region=30):
    results = {}
    for rk in pred_df["region"].unique():
        rdf = pred_df[(pred_df["region"] == rk) & (pred_df["whale_count"] > 0)]
        top = rdf.nlargest(n_per_region, "predicted_risk")
        results[rk] = top[["region", "lat", "lon", "month", "predicted_risk",
                           "encounter_count", "ship_count", "whale_count",
                           "whale_month_count", "n_species"]].to_dict("records")
    return results


# --- HTML report ---

def generate_report(history, importance, monthly, hotspots, pred_df, model_params):
    monthly_by_region = defaultdict(lambda: {"months": [], "mean_risk": [], "max_risk": [], "total_enc": []})
    for _, row in monthly.iterrows():
        rk = row["region"]
        monthly_by_region[rk]["months"].append(int(row["month"]))
        monthly_by_region[rk]["mean_risk"].append(round(float(row["mean_risk"]), 4))
        monthly_by_region[rk]["max_risk"].append(round(float(row["max_risk"]), 4))
        monthly_by_region[rk]["total_enc"].append(int(row["total_enc"]))

    # Spatial predictions — take the max risk across months for each cell
    spatial = pred_df.groupby(["region", "lat", "lon"]).agg(
        peak_risk=("predicted_risk", "max"),
        peak_month=("predicted_risk", "idxmax"),
        whale_total=("whale_count", "max"),
    ).reset_index()
    spatial["peak_month"] = pred_df.loc[spatial["peak_month"].values, "month"].values
    spatial = spatial[spatial["whale_total"] > 0]

    spatial_data = {}
    for rk in REGIONS:
        rdf = spatial[spatial["region"] == rk]
        if rdf.empty:
            continue
        rdf_top = rdf.nlargest(500, "peak_risk")
        spatial_data[rk] = {
            "lats": rdf_top["lat"].round(3).tolist(),
            "lons": rdf_top["lon"].round(3).tolist(),
            "risks": rdf_top["peak_risk"].round(4).tolist(),
            "months": rdf_top["peak_month"].astype(int).tolist(),
        }

    feat_importance = [
        {"name": FEATURE_DISPLAY_NAMES[i], "value": round(float(importance[i]), 4)}
        for i in range(len(importance))
    ]
    feat_importance.sort(key=lambda x: -x["value"])

    region_names_json = json.dumps(REGION_NAMES)
    history_json = json.dumps(history)
    monthly_json = json.dumps(dict(monthly_by_region))
    feat_json = json.dumps(feat_importance)
    hotspots_json = json.dumps(hotspots)
    spatial_json = json.dumps(spatial_data)
    model_json = json.dumps(model_params)

    month_names = json.dumps(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Collision Risk Model — Under the Sea</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Inter',system-ui,sans-serif; background:#0a0a0a; color:#e8e0d4; line-height:1.55; }}
  .wrap {{ max-width:1100px; margin:0 auto; padding:28px 20px; }}
  .title {{ font-size:21px; font-weight:600; color:#f5f0e8; margin-bottom:3px; letter-spacing:-0.3px; }}
  .subtitle {{ font-size:12.5px; color:#807a6f; margin-bottom:22px; font-weight:400; }}
  .row {{ display:grid; gap:14px; margin-bottom:14px; }}
  .row-2 {{ grid-template-columns:1fr 1fr; }}
  .card {{ background:#1a1a1a; border-radius:10px; padding:16px; }}
  .card-title {{ font-size:12px; font-weight:600; color:#a09a8e; margin-bottom:10px; text-transform:uppercase; letter-spacing:0.6px; }}
  .stats {{ display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-bottom:16px; }}
  .stat {{ background:#141414; border-radius:8px; padding:10px; text-align:center; }}
  .stat .num {{ font-size:21px; font-weight:700; }}
  .stat .lbl {{ font-size:9px; color:#807a6f; margin-top:2px; font-weight:500; }}
  .c1 {{ color:#7cacf0; }} .c2 {{ color:#6ec9a0; }} .c3 {{ color:#e0c068; }} .c4 {{ color:#e88080; }}
  canvas {{ max-height:240px; }}
  table {{ width:100%; border-collapse:collapse; font-size:12px; }}
  th {{ text-align:left; padding:5px 8px; color:#807a6f; border-bottom:1px solid #2a2a2a; font-weight:600; font-size:11px; }}
  td {{ padding:5px 8px; border-bottom:1px solid #1a1a1a; color:#d4cec2; }}
  tr:hover td {{ background:#1f1f1f; }}
  .pill {{ display:inline-block; padding:2px 8px; border-radius:10px; font-size:10px; font-weight:600; }}
  .pill-r {{ background:#3d1515; color:#f0a0a0; }}
  .pill-a {{ background:#3d2e10; color:#f0d888; }}
  .pill-g {{ background:#153320; color:#90e0b0; }}
  #pred-map {{ height:340px; border-radius:8px; margin-top:8px; }}
  .tabs {{ display:flex; gap:5px; margin-bottom:8px; flex-wrap:wrap; }}
  .tab {{ padding:5px 12px; border-radius:5px; cursor:pointer; font-size:11px;
      background:#141414; border:1px solid #2a2a2a; color:#a09a8e; transition:all 0.15s; font-weight:500; }}
  .tab:hover {{ border-color:#7cacf0; color:#7cacf0; }}
  .tab.active {{ background:#2a4a80; border-color:#2a4a80; color:#f5f0e8; }}
  @media (max-width:800px) {{ .row-2 {{ grid-template-columns:1fr; }} .stats {{ grid-template-columns:1fr 1fr; }} }}
</style>
</head><body>

<div class="wrap">

<div class="title">Collision Risk Prediction Model</div>
<div class="subtitle">PyTorch neural network &middot; 18 features &middot; 4 US maritime regions &middot; NOAA AIS 2024 + OBIS whale sightings</div>

<div class="stats" id="top-stats"></div>

<div class="row row-2">
    <div class="card">
        <div class="card-title">Training Loss</div>
        <canvas id="lossChart"></canvas>
    </div>
    <div class="card">
        <div class="card-title">Feature Importance</div>
        <canvas id="featChart"></canvas>
    </div>
</div>

<div class="card" style="margin-bottom:14px;">
    <div class="card-title">Monthly Risk Profile</div>
    <div class="tabs" id="monthly-tabs"></div>
    <canvas id="monthlyChart"></canvas>
</div>

<div class="card" style="margin-bottom:14px;">
    <div class="card-title">Predicted Risk Map</div>
    <div class="tabs" id="map-tabs"></div>
    <div id="pred-map"></div>
</div>

<div class="card" style="margin-bottom:14px;">
    <div class="card-title">Top 30 Predicted Hotspots by Region</div>
    <div class="tabs" id="hotspot-tabs"></div>
    <table>
        <thead><tr><th>#</th><th>Lat</th><th>Lon</th><th>Month</th><th>Risk</th><th>Encounters</th><th>Ships</th><th>Whales</th><th>Monthly</th><th>Species</th></tr></thead>
        <tbody id="hotspot-table"></tbody>
    </table>
</div>

<div class="card">
    <div class="card-title">Seasonal Patterns (All Regions)</div>
    <canvas id="seasonChart" height="80"></canvas>
</div>

</div>

<div style="text-align:center;padding:24px;color:#4a4540;font-size:10px;">
    Under the Sea &middot; Collision Risk Neural Network &middot; NOAA AIS 2024 &amp; OBIS
</div>

<script>
const HISTORY = {history_json};
const MONTHLY = {monthly_json};
const FEATURES = {feat_json};
const HOTSPOTS = {hotspots_json};
const SPATIAL = {spatial_json};
const MODEL_INFO = {model_json};
const REGION_NAMES = {region_names_json};
const MONTH_NAMES = {month_names};
const REGION_COLORS = {{
    "us_west_coast": "#7cacf0",
    "us_east_coast": "#6ec9a0",
    "gulf_of_mexico": "#e0c068",
    "hawaii": "#e08888"
}};

// -- Top stats --
(function() {{
    let el = document.getElementById('top-stats');
    el.innerHTML = `
        <div class="stat"><div class="num c1">${{MODEL_INFO.total_samples.toLocaleString()}}</div><div class="lbl">Training Samples</div></div>
        <div class="stat"><div class="num c2">${{MODEL_INFO.n_features}}</div><div class="lbl">Input Features</div></div>
        <div class="stat"><div class="num c3">${{MODEL_INFO.total_params.toLocaleString()}}</div><div class="lbl">Parameters</div></div>
        <div class="stat"><div class="num c4">${{MODEL_INFO.final_val_loss}}</div><div class="lbl">Val Loss (MSE)</div></div>
    `;
}})();

// -- Loss chart --
new Chart(document.getElementById('lossChart'), {{
    type: 'line',
    data: {{
        labels: HISTORY.train_loss.map((_,i) => i+1),
        datasets: [
            {{ label: 'Train', data: HISTORY.train_loss, borderColor: '#60a5fa', backgroundColor: 'rgba(96,165,250,0.08)', fill: true, tension: 0.3, pointRadius: 0, borderWidth: 1.5 }},
            {{ label: 'Validation', data: HISTORY.val_loss, borderColor: '#f87171', backgroundColor: 'rgba(248,113,113,0.08)', fill: true, tension: 0.3, pointRadius: 0, borderWidth: 1.5 }},
        ]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ labels: {{ color: '#a09a8e', font: {{ size: 10 }} }} }} }},
        scales: {{
            x: {{ title: {{ display: true, text: 'Epoch', color: '#807a6f', font: {{ size: 10 }} }}, ticks: {{ color: '#605a4f', maxTicksLimit: 8, font: {{ size: 9 }} }}, grid: {{ color: '#1f1f1f' }} }},
            y: {{ title: {{ display: true, text: 'MSE Loss', color: '#807a6f', font: {{ size: 10 }} }}, ticks: {{ color: '#605a4f', font: {{ size: 9 }} }}, grid: {{ color: '#1f1f1f' }} }}
        }}
    }}
}});

// -- Feature importance --
new Chart(document.getElementById('featChart'), {{
    type: 'bar',
    data: {{
        labels: FEATURES.map(f => f.name),
        datasets: [{{ data: FEATURES.map(f => (f.value * 100).toFixed(1)),
            backgroundColor: '#5a8ac0', borderRadius: 3 }}]
    }},
    options: {{
        indexAxis: 'y', responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
            x: {{ title: {{ display: true, text: 'Importance %', color: '#807a6f', font: {{ size: 10 }} }}, ticks: {{ color: '#605a4f', font: {{ size: 9 }} }}, grid: {{ color: '#1f1f1f' }} }},
            y: {{ ticks: {{ color: '#a09a8e', font: {{ size: 9 }} }}, grid: {{ display: false }} }}
        }}
    }}
}});

// -- Monthly chart --
let monthlyChart = null;
function showMonthly(rk) {{
    document.querySelectorAll('#monthly-tabs .tab').forEach(t => t.classList.remove('active'));
    document.querySelector(`#monthly-tabs .tab[data-rk="${{rk}}"]`).classList.add('active');
    let d = MONTHLY[rk];
    if (!d) return;
    if (monthlyChart) monthlyChart.destroy();
    monthlyChart = new Chart(document.getElementById('monthlyChart'), {{
        type: 'bar',
        data: {{
            labels: d.months.map(m => MONTH_NAMES[m-1]),
            datasets: [
                {{ label: 'Mean Risk', data: d.mean_risk, backgroundColor: 'rgba(90,138,192,0.55)', borderRadius: 3, yAxisID: 'y' }},
                {{ label: 'Max Risk', data: d.max_risk, type: 'line', borderColor: '#e08080', pointRadius: 2, borderWidth: 1.5, yAxisID: 'y', fill: false }},
            ]
        }},
        options: {{
            responsive: true,
            plugins: {{ legend: {{ labels: {{ color: '#a09a8e', font: {{ size: 10 }} }} }} }},
            scales: {{
                x: {{ ticks: {{ color: '#a09a8e', font: {{ size: 9 }} }}, grid: {{ color: '#1f1f1f' }} }},
                y: {{ title: {{ display: true, text: 'Risk', color: '#807a6f', font: {{ size: 10 }} }}, ticks: {{ color: '#605a4f', font: {{ size: 9 }} }}, grid: {{ color: '#1f1f1f' }}, min: 0 }},
            }}
        }}
    }});
}}
(function() {{
    let html = '';
    for (let rk in MONTHLY) {{
        let active = html === '' ? ' active' : '';
        html += `<div class="tab${{active}}" data-rk="${{rk}}" onclick="showMonthly('${{rk}}')">${{REGION_NAMES[rk] || rk}}</div>`;
    }}
    document.getElementById('monthly-tabs').innerHTML = html;
    let first = Object.keys(MONTHLY)[0];
    if (first) showMonthly(first);
}})();

// -- Prediction map --
let predMap = L.map('pred-map', {{ zoomControl: true }}).setView([35, -120], 4);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
    attribution: '&copy; CARTO', maxZoom: 18
}}).addTo(predMap);

let mapLayers = {{}};
function showMapRegion(rk) {{
    document.querySelectorAll('#map-tabs .tab').forEach(t => t.classList.remove('active'));
    document.querySelector(`#map-tabs .tab[data-rk="${{rk}}"]`).classList.add('active');
    for (let k in mapLayers) {{ predMap.removeLayer(mapLayers[k]); }}
    let d = SPATIAL[rk];
    if (!d) return;
    let layer = L.layerGroup();
    let maxR = Math.max(...d.risks);
    for (let i = 0; i < d.lats.length; i++) {{
        let norm = d.risks[i] / (maxR + 1e-6);
        let r = Math.round(255 * Math.min(1, norm * 2));
        let g = Math.round(255 * Math.max(0, 1 - norm * 2));
        let color = `rgb(${{r}},${{g}},60)`;
        L.circleMarker([d.lats[i], d.lons[i]], {{
            radius: 3 + norm * 6, color: color, fill: true, fillColor: color,
            fillOpacity: 0.6 + norm * 0.3, weight: 0
        }}).bindPopup(
            `<b>Risk: ${{(d.risks[i]*100).toFixed(1)}}%</b><br>Peak month: ${{MONTH_NAMES[d.months[i]-1]}}<br>Lat: ${{d.lats[i]}}, Lon: ${{d.lons[i]}}`
        ).addTo(layer);
    }}
    layer.addTo(predMap);
    mapLayers[rk] = layer;
    let bounds = SPATIAL[rk];
    if (d.lats.length > 0) {{
        predMap.fitBounds([[Math.min(...d.lats)-1, Math.min(...d.lons)-1],
                           [Math.max(...d.lats)+1, Math.max(...d.lons)+1]]);
    }}
}}
(function() {{
    let html = '';
    for (let rk in SPATIAL) {{
        let active = html === '' ? ' active' : '';
        html += `<div class="tab${{active}}" data-rk="${{rk}}" onclick="showMapRegion('${{rk}}')">${{REGION_NAMES[rk] || rk}}</div>`;
    }}
    document.getElementById('map-tabs').innerHTML = html;
    let first = Object.keys(SPATIAL)[0];
    if (first) showMapRegion(first);
}})();

// -- Hotspot table (per-region tabs) --
function showHotspots(rk) {{
    document.querySelectorAll('#hotspot-tabs .tab').forEach(t => t.classList.remove('active'));
    let active = document.querySelector(`#hotspot-tabs .tab[data-rk="${{rk}}"]`);
    if (active) active.classList.add('active');
    let list = HOTSPOTS[rk] || [];
    let html = '';
    list.forEach((h, i) => {{
        let risk = (h.predicted_risk * 100).toFixed(1);
        let badge = h.predicted_risk > 0.6 ? 'pill-r' : h.predicted_risk > 0.3 ? 'pill-a' : 'pill-g';
        html += `<tr>
            <td>${{i+1}}</td>
            <td>${{h.lat.toFixed(2)}}</td><td>${{h.lon.toFixed(2)}}</td>
            <td>${{MONTH_NAMES[h.month-1]}}</td>
            <td><span class="pill ${{badge}}">${{risk}}%</span></td>
            <td>${{h.encounter_count.toLocaleString()}}</td>
            <td>${{h.ship_count.toLocaleString()}}</td>
            <td>${{h.whale_count.toLocaleString()}}</td>
            <td>${{h.whale_month_count.toLocaleString()}}</td>
            <td>${{h.n_species}}</td>
        </tr>`;
    }});
    document.getElementById('hotspot-table').innerHTML = html;
}}
(function() {{
    let html = '';
    for (let rk in HOTSPOTS) {{
        let active = html === '' ? ' active' : '';
        html += `<div class="tab${{active}}" data-rk="${{rk}}" onclick="showHotspots('${{rk}}')">${{REGION_NAMES[rk] || rk}}</div>`;
    }}
    document.getElementById('hotspot-tabs').innerHTML = html;
    let first = Object.keys(HOTSPOTS)[0];
    if (first) showHotspots(first);
}})();

// -- Season chart (all regions overlaid) --
(function() {{
    let datasets = [];
    for (let rk in MONTHLY) {{
        let d = MONTHLY[rk];
        datasets.push({{
            label: REGION_NAMES[rk] || rk,
            data: d.mean_risk,
            borderColor: REGION_COLORS[rk] || '#60a5fa',
            backgroundColor: (REGION_COLORS[rk] || '#60a5fa') + '22',
            fill: true, tension: 0.4, pointRadius: 3, borderWidth: 2,
        }});
    }}
    new Chart(document.getElementById('seasonChart'), {{
        type: 'line',
        data: {{ labels: MONTH_NAMES, datasets: datasets }},
        options: {{
            responsive: true,
            plugins: {{ legend: {{ labels: {{ color: '#a09a8e', font: {{ size: 10 }} }} }} }},
            scales: {{
                x: {{ ticks: {{ color: '#a09a8e', font: {{ size: 9 }} }}, grid: {{ color: '#1f1f1f' }} }},
                y: {{ title: {{ display: true, text: 'Mean Risk', color: '#807a6f', font: {{ size: 10 }} }}, ticks: {{ color: '#605a4f', font: {{ size: 9 }} }}, grid: {{ color: '#1f1f1f' }}, min: 0 }}
            }}
        }}
    }});
}})();
</script>
</body></html>"""
    return html


# --- Main ---

def main():
    print("=" * 60)
    print("  Neural Network Collision Risk Model")
    print("=" * 60)

    print("\n[1/5] Building dataset...")
    df = build_dataset()
    if df.empty:
        print("ERROR: No data available. Run the analysis pipeline first.")
        return

    print("\n[2/5] Preparing tensors and normalizing...")
    X, y, means, stds, y_max = prepare_tensors(df)
    print(f"  Features: {X.shape[1]}  Samples: {X.shape[0]:,}  Device: {DEVICE}")

    print(f"\n[3/5] Training neural network ({DEVICE})...")
    model, history = train_model(X, y, epochs=120, lr=1e-3, batch_size=512)

    print("\n[4/5] Computing feature importance and predictions...")
    importance = compute_feature_importance(model, X)
    pred_df = predict_grid(model, df, means, stds)
    monthly = monthly_analysis(pred_df)
    hotspots = top_hotspots(pred_df, n_per_region=30)

    total_params = sum(p.numel() for p in model.parameters())
    model_params = {
        "total_samples": int(len(df)),
        "n_features": int(X.shape[1]),
        "total_params": int(total_params),
        "epochs": 120,
        "lr": 0.001,
        "batch_size": 512,
        "device": str(DEVICE),
        "final_val_loss": round(history["val_loss"][-1], 6),
        "final_val_mae": round(history["val_mae"][-1], 6),
    }

    print("\n[5/5] Generating HTML report...")
    html = generate_report(history, importance, monthly, hotspots, pred_df, model_params)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    size_mb = OUTPUT_HTML.stat().st_size / 1e6
    print(f"\n  Report saved to {OUTPUT_HTML} ({size_mb:.1f} MB)")
    print("  Done!")


if __name__ == "__main__":
    main()
