"""Interactive whale-ship risk map generator."""

import json
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster, Draw
from branca.element import Element
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "processed"
MAP_DATA_DIR = DATA_DIR / "map_data"
OUTPUT = Path(__file__).parent / "whale_ship_map.html"

ACTIVE_REGIONS = ["us_west_coast", "us_east_coast", "gulf_of_mexico", "hawaii"]

REGION_VIEWS = {
    "us_west_coast": {"center": [38.5, -123.0], "zoom": 5},
    "us_east_coast": {"center": [36.0, -74.0], "zoom": 5},
    "gulf_of_mexico": {"center": [25.0, -90.0], "zoom": 5},
    "hawaii": {"center": [20.5, -157.0], "zoom": 7},
}

REGION_NAMES = {
    "us_west_coast": "US West Coast",
    "us_east_coast": "US East Coast",
    "gulf_of_mexico": "Gulf of Mexico",
    "hawaii": "Hawaii",
}

SPECIES_COLORS = {
    "Blue Whale": "#1565c0",
    "Humpback Whale": "#7b1fa2",
    "Gray Whale": "#5d4037",
    "Fin Whale": "#00838f",
    "Sperm Whale": "#e64a19",
    "Minke Whale": "#2e7d32",
    "N Atlantic Right Whale": "#c62828",
    "North Atlantic Right Whale": "#c62828",
}

SHIP_GRADIENT = {0.1: "#ffe082", 0.3: "#ffb300", 0.5: "#ff8f00", 0.7: "#e65100", 1.0: "#bf360c"}
WHALE_GRADIENT = {0.1: "#80deea", 0.3: "#26c6da", 0.5: "#00acc1", 0.7: "#00838f", 1.0: "#004d40"}


def load_region(region_key):
    """Load all data for a region, returning None for missing files."""
    result = {}
    for name, fname in [
        ("ships", f"ships_2024_monthly_{region_key}.parquet"),
        ("whales", f"whales_{region_key}.parquet"),
        ("hotspots", f"hotspots_{region_key}.parquet"),
        ("encounters", f"encounters_{region_key}.parquet"),
        ("hotspots_ct", f"hotspots_cargo_tanker_{region_key}.parquet"),
        ("encounters_ct", f"encounters_cargo_tanker_{region_key}.parquet"),
    ]:
        p = DATA_DIR / fname
        result[name] = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    return result


def make_map():
    print("Loading precomputed grid data...")
    with open(MAP_DATA_DIR / "precomputed.json") as f:
        precomputed = json.load(f)

    m = folium.Map(location=[33, -110], zoom_start=4, tiles=None, control_scale=True)

    # --- Tiles ---
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Ocean", name="Ocean Basemap",
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Satellite", name="Satellite",
    ).add_to(m)
    folium.TileLayer(
        tiles="https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="CARTO", name="Dark",
    ).add_to(m)

    all_stats = {}

    for region_key in ACTIVE_REGIONS:
        data = load_region(region_key)
        ships = data["ships"]
        whales = data["whales"]
        hotspots = data["hotspots"]
        encounters = data["encounters"]
        hotspots_ct = data["hotspots_ct"]
        encounters_ct = data["encounters_ct"]
        rname = REGION_NAMES.get(region_key, region_key)

        if ships.empty and whales.empty:
            continue

        print(f"  {rname}: {len(ships):,} ships, {len(whales):,} whales")

        all_stats[region_key] = {
            "ships": len(ships),
            "whales": len(whales),
            "hotspots": len(hotspots),
            "encounters": len(encounters),
            "unique_vessels": int(ships["mmsi"].nunique()) if not ships.empty else 0,
        }

        # Ship heatmap
        if not ships.empty:
            sg = folium.FeatureGroup(name=f"\u2693 Ships - {rname}", show=(region_key == "us_west_coast"))
            s_sample = ships.sample(n=min(10_000, len(ships)), random_state=42)
            HeatMap(
                s_sample[["latitude", "longitude"]].values.tolist(),
                radius=7, blur=5, max_zoom=14, min_opacity=0.4, gradient=SHIP_GRADIENT,
            ).add_to(sg)
            sg.add_to(m)

        # Whale heatmap
        if not whales.empty:
            wg = folium.FeatureGroup(name=f"\U0001F40B Whales - {rname}", show=False)
            w_sample = whales.sample(n=min(10_000, len(whales)), random_state=42)
            HeatMap(
                w_sample[["latitude", "longitude"]].values.tolist(),
                radius=9, blur=5, max_zoom=14, min_opacity=0.45, gradient=WHALE_GRADIENT,
            ).add_to(wg)
            wg.add_to(m)

        # Hotspots (all ships)
        if not hotspots.empty:
            hg = folium.FeatureGroup(name=f"\U0001F534 Hotspots All - {rname}", show=(region_key == "us_west_coast"))
            rmax = hotspots["risk_score"].max()
            rmin = hotspots["risk_score"].min()
            for _, row in hotspots.iterrows():
                norm = (row["risk_score"] - rmin) / (rmax - rmin + 1e-10)
                rad = 10 + norm * 16
                fc = "#d32f2f" if norm > 0.7 else "#f57c00" if norm > 0.3 else "#fdd835"
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=rad, color="#ffffff", fill=True, fill_color=fc,
                    fill_opacity=0.75, opacity=1.0, weight=2,
                    popup=folium.Popup(
                        f"<div style='font-family:Segoe UI,sans-serif;min-width:160px'>"
                        f"<b>Hotspot ({rname})</b><br>"
                        f"Risk: <b>{row['risk_score']:.3f}</b><br>"
                        f"Ships: {row['ship_density']:,.0f} | Whales: {row['whale_density']:,.0f}</div>",
                        max_width=220,
                    ),
                ).add_to(hg)
            hg.add_to(m)

        # Hotspots (cargo/tanker)
        if not hotspots_ct.empty:
            hcg = folium.FeatureGroup(name=f"\U0001F534 Hotspots Cargo/Tanker - {rname}", show=False)
            ctmax = hotspots_ct["risk_score"].max()
            ctmin = hotspots_ct["risk_score"].min()
            for _, row in hotspots_ct.iterrows():
                norm = (row["risk_score"] - ctmin) / (ctmax - ctmin + 1e-10)
                rad = 10 + norm * 16
                fc = "#d32f2f" if norm > 0.7 else "#f57c00" if norm > 0.3 else "#fdd835"
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=rad, color="#000000", fill=True, fill_color=fc,
                    fill_opacity=0.75, opacity=1.0, weight=2,
                    popup=folium.Popup(
                        f"<div style='font-family:Segoe UI,sans-serif;min-width:160px'>"
                        f"<b>Cargo/Tanker Hotspot ({rname})</b><br>"
                        f"Risk: <b>{row['risk_score']:.3f}</b><br>"
                        f"Ships: {row['ship_density']:,.0f} | Whales: {row['whale_density']:,.0f}</div>",
                        max_width=220,
                    ),
                ).add_to(hcg)
            hcg.add_to(m)

        # Encounters (cargo/tanker - more useful default)
        if not encounters_ct.empty:
            ecg = folium.FeatureGroup(name=f"\u26A0 Encounters Cargo/Tanker - {rname}", show=False)
            et = encounters_ct.head(500)
            ecl = MarkerCluster(options={"maxClusterRadius": 35})
            for _, row in et.iterrows():
                sp = row.get("whale_species", "Unknown")
                color = SPECIES_COLORS.get(sp, "#757575")
                folium.CircleMarker(
                    location=[row["whale_lat"], row["whale_lon"]],
                    radius=4, color=color, fill=True, fill_color=color, fill_opacity=0.7,
                    popup=f"<b>{sp}</b><br>Dist: {row['distance_km']:.2f}km<br>Ship: {row.get('ship_name','?')}",
                ).add_to(ecl)
            ecl.add_to(ecg)
            ecg.add_to(m)

        # Slow-down zones
        if region_key in precomputed and precomputed[region_key]["slow_zones"]:
            szg = folium.FeatureGroup(name=f"\U0001F6A8 Slow Zones - {rname}", show=False)
            for poly in precomputed[region_key]["slow_zones"]:
                folium.Polygon(
                    locations=poly,
                    color="#ff1744", weight=2, fill=True, fill_color="#ff1744",
                    fill_opacity=0.15,
                    popup=f"<b>Recommended Slow Zone</b><br>{rname}<br>Reduce to 10 knots",
                ).add_to(szg)
            szg.add_to(m)

    # --- Confidence/Uncertainty Layer (all regions) ---
    print("Building confidence layer...")
    conf_group = folium.FeatureGroup(name="\U0001F4CA Confidence / Data Coverage", show=False)
    CONF_STEP = 3
    CONF_CELL = 0.1 * CONF_STEP
    for region_key in ACTIVE_REGIONS:
        if region_key not in precomputed:
            continue
        pc = precomputed[region_key]
        lons = pc["lon_centers"]
        lats = pc["lat_centers"]
        conf = pc["confidence"]
        for i in range(0, min(len(conf), len(lons)), CONF_STEP):
            for j in range(0, min(len(conf[i]) if i < len(conf) else 0, len(lats)), CONF_STEP):
                val = conf[i][j]
                if val > 0.25:
                    opacity = min(val * 0.5, 0.4)
                    color = "#4caf50" if val > 0.6 else "#ffeb3b" if val > 0.3 else "#f44336"
                    folium.Rectangle(
                        bounds=[[lats[j] - CONF_CELL/2, lons[i] - CONF_CELL/2],
                                [lats[j] + CONF_CELL/2, lons[i] + CONF_CELL/2]],
                        color=color, weight=0, fill=True, fill_color=color,
                        fill_opacity=opacity,
                    ).add_to(conf_group)
    conf_group.add_to(m)

    # --- Draw tool for route simulation ---
    Draw(
        draw_options={
            "polyline": {"shapeOptions": {"color": "#00e5ff", "weight": 4}},
            "polygon": False, "rectangle": False, "circle": False,
            "marker": False, "circlemarker": False,
        },
        edit_options={"edit": False},
    ).add_to(m)

    # --- Layer control ---
    folium.LayerControl(collapsed=True).add_to(m)

    # --- Embed precomputed data for client-side route analysis ---
    print("Embedding route analysis engine...")
    grid_data_for_js = {}
    for rk in ACTIVE_REGIONS:
        if rk not in precomputed:
            continue
        pc = precomputed[rk]
        grid_data_for_js[rk] = {
            "bounds": pc["bounds"],
            "lon_centers": pc["lon_centers"],
            "lat_centers": pc["lat_centers"],
            "whale_density": pc["whale_density"],
            "ship_density": pc["ship_density"],
            "name": REGION_NAMES.get(rk, rk),
        }

    stats_json = json.dumps(all_stats)
    grid_json = json.dumps(grid_data_for_js)

    # Build the big JS/HTML block
    route_analysis_html = f"""
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>

    <div id="route-panel" style="
        display:none; position:fixed; bottom:20px; left:50%; transform:translateX(-50%);
        z-index:1500; background:rgba(255,255,255,0.97); border-radius:12px;
        padding:16px 20px; font-family:'Segoe UI',sans-serif; font-size:13px;
        box-shadow:0 4px 20px rgba(0,0,0,0.3); max-width:820px; width:92%;
        max-height:52vh; overflow-y:auto;
    ">
        <div style="float:right;">
            <span onclick="clearRoute()" style="cursor:pointer;font-size:12px;color:#1565c0;margin-right:12px;text-decoration:underline;">Clear Route</span>
            <span onclick="closeRoutePanel()" style="cursor:pointer;font-size:18px;color:#999;">&times;</span>
        </div>
        <div style="font-weight:700;font-size:16px;color:#1a237e;margin-bottom:8px;">Route Risk Analysis</div>
        <div id="route-summary"></div>
        <canvas id="routeChart" height="90" style="margin-top:8px;"></canvas>
        <div id="route-reroute" style="margin-top:10px;"></div>
        <div id="speed-slider-wrap" style="display:none;margin-top:10px;background:#f5f5f5;border-radius:8px;padding:10px;">
            <div style="display:flex;align-items:center;gap:10px;">
                <span style="font-weight:600;font-size:12px;color:#333;white-space:nowrap;">Speed:</span>
                <input type="range" id="speed-slider" min="6" max="22" step="0.5" value="14"
                    style="flex:1;accent-color:#1565c0;cursor:pointer;">
                <span id="speed-val" style="font-weight:700;font-size:14px;color:#1a237e;min-width:52px;text-align:right;">14 kn</span>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:9px;color:#999;margin-top:2px;padding:0 2px;">
                <span>6 kn (slow)</span><span>14 kn (normal)</span><span>22 kn (fast)</span>
            </div>
        </div>
        <div id="route-econ" style="margin-top:10px;"></div>
    </div>

    <div id="dashboard-toggle" onclick="toggleDashboard()" style="
        position:fixed; top:60px; right:12px; z-index:900;
        background:rgba(26,35,126,0.92); border-radius:8px; padding:8px 16px;
        color:white; font-family:'Segoe UI',sans-serif; font-size:13px;
        cursor:pointer; box-shadow:0 2px 8px rgba(0,0,0,0.3); user-select:none;
    ">Dashboard &#9776;</div>

    <div id="dashboard" style="
        position:fixed; top:0; right:0; z-index:1999;
        width:370px; height:100vh; overflow-y:auto;
        background:rgba(255,255,255,0.97); font-family:'Segoe UI',sans-serif;
        box-shadow:-4px 0 20px rgba(0,0,0,0.15); padding:16px 18px;
        transform:translateX(100%); transition:transform 0.3s ease;
        font-size:12.5px;
    ">
        <div onclick="toggleDashboard()" style="cursor:pointer;float:right;font-size:20px;color:#999;">&times;</div>
        <div style="font-size:20px;font-weight:800;color:#1a237e;margin-bottom:2px;">Under the Sea</div>
        <div style="color:#666;margin-bottom:14px;font-size:11.5px;">Multi-Region Whale-Ship Risk Analysis &mdash; 2024</div>
        <div id="dash-stats"></div>
        <div style="margin-top:14px;font-weight:700;margin-bottom:4px;">How to Use</div>
        <div style="font-size:11.5px;color:#555;line-height:1.5;">
            <b>Draw a route:</b> Click the polyline tool (top-left) to draw a ship route. Click each waypoint, double-click to finish. The tool analyzes risk, suggests a <span style="color:#00c853;font-weight:700;">safer reroute</span> (green dashed line) if one exists, and compares the economic cost of both paths.<br><br>
            <b>Layers:</b> Toggle ship traffic, whale sightings, hotspots, slow zones, and confidence overlay from the layer control (top-right).<br><br>
            <b>Slow Zones</b> (red polygons): Auto-generated areas where ships should reduce to 10 knots based on high whale-ship overlap.<br><br>
            <b>Confidence</b>: Green = lots of data, yellow = moderate, red = sparse. Low-confidence areas may have whales we don't know about.
        </div>
        <div style="margin-top:12px;padding-top:8px;border-top:1px solid #e0e0e0;color:#999;font-size:10px;">
            Ship data: NOAA AIS 2024 (12 monthly snapshots)<br>
            Whale data: OBIS 2000&ndash;2024<br>
            Regions: West Coast, East Coast, Gulf of Mexico, Hawaii
        </div>
    </div>

    <div style="
        position:fixed; bottom:30px; left:12px; z-index:900;
        background:rgba(255,255,255,0.93); border-radius:10px;
        padding:10px 14px; font-family:'Segoe UI',sans-serif;
        font-size:11.5px; box-shadow:0 2px 10px rgba(0,0,0,0.2);
        border:1px solid #e0e0e0; max-width:185px;
    ">
        <div style="font-weight:700;font-size:12px;color:#1a237e;margin-bottom:5px;">Legend</div>
        <div style="margin-bottom:3px;">
            <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#d32f2f;border:2px solid #fff;vertical-align:middle;"></span> High risk
            <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#f57c00;border:2px solid #fff;vertical-align:middle;margin-left:4px;"></span> Moderate
            <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#fdd835;border:2px solid #fff;vertical-align:middle;margin-left:4px;"></span> Lower
        </div>
        <div style="font-size:10px;color:#888;margin-bottom:4px;">White border = all ships | Black = cargo/tanker</div>
        <div style="margin-bottom:2px;"><span style="color:#e65100;">&#11044;</span> Ships <span style="color:#00838f;margin-left:8px;">&#11044;</span> Whales</div>
        <div style="margin-bottom:2px;"><span style="color:#ff1744;">&#9644;</span> Slow zone (10kn recommended)</div>
        <div style="margin-bottom:2px;"><span style="color:#00c853;">- - -</span> Safer reroute</div>
        <div><span style="color:#4caf50;">&#9632;</span> High conf <span style="color:#ffeb3b;">&#9632;</span> Med <span style="color:#f44336;">&#9632;</span> Low</div>
    </div>

    <div style="
        position:fixed; top:12px; left:50%; transform:translateX(-50%);
        z-index:900; background:rgba(26,35,126,0.92); border-radius:10px;
        padding:9px 24px; font-family:'Segoe UI',sans-serif;
        color:white; font-size:16px; font-weight:700;
        box-shadow:0 2px 10px rgba(0,0,0,0.3); pointer-events:none;
    ">Under the Sea &mdash; Whale-Ship Risk Map</div>

    <script>
    const GRID_DATA = {grid_json};
    const ALL_STATS = {stats_json};
    let routeChart = null;
    let routeLayer = null;
    let dashOpen = false;

    function toggleDashboard() {{
        dashOpen = !dashOpen;
        document.getElementById('dashboard').style.transform = dashOpen ? 'translateX(0)' : 'translateX(100%)';
    }}

    let _theMap = null;
    let _drawnLine = null;
    let _lastOrig = null;
    let _lastSafer = null;
    let _lastRerouteImproved = false;
    function closeRoutePanel() {{ document.getElementById('route-panel').style.display = 'none'; }}
    function clearRoute() {{
        if (routeLayer && _theMap) {{ _theMap.removeLayer(routeLayer); routeLayer = null; }}
        if (_drawnLine && _theMap) {{ _theMap.removeLayer(_drawnLine); _drawnLine = null; }}
        _lastOrig = null; _lastSafer = null; _lastRerouteImproved = false;
        closeRoutePanel();
    }}

    // Build dashboard stats
    (function() {{
        let html = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:10px;">';
        let totalShips = 0, totalWhales = 0, totalEnc = 0, totalVessels = 0;
        for (let r in ALL_STATS) {{
            totalShips += ALL_STATS[r].ships;
            totalWhales += ALL_STATS[r].whales;
            totalEnc += ALL_STATS[r].encounters;
            totalVessels += ALL_STATS[r].unique_vessels;
        }}
        html += `<div style="background:#e8eaf6;border-radius:6px;padding:8px;text-align:center;">
            <div style="font-size:18px;font-weight:700;color:#283593;">${{totalShips.toLocaleString()}}</div>
            <div style="font-size:9px;color:#666;">Ship Positions</div></div>`;
        html += `<div style="background:#e0f2f1;border-radius:6px;padding:8px;text-align:center;">
            <div style="font-size:18px;font-weight:700;color:#00695c;">${{totalWhales.toLocaleString()}}</div>
            <div style="font-size:9px;color:#666;">Whale Sightings</div></div>`;
        html += `<div style="background:#fff3e0;border-radius:6px;padding:8px;text-align:center;">
            <div style="font-size:18px;font-weight:700;color:#e65100;">${{totalVessels.toLocaleString()}}</div>
            <div style="font-size:9px;color:#666;">Unique Vessels</div></div>`;
        html += `<div style="background:#fce4ec;border-radius:6px;padding:8px;text-align:center;">
            <div style="font-size:18px;font-weight:700;color:#c62828;">4</div>
            <div style="font-size:9px;color:#666;">Regions Analyzed</div></div>`;
        html += '</div>';

        for (let r in ALL_STATS) {{
            let s = ALL_STATS[r];
            let name = r.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());
            html += `<div style="background:#f5f5f5;border-radius:6px;padding:6px 8px;margin-bottom:4px;font-size:11.5px;">
                <b>${{name}}</b>: ${{s.ships.toLocaleString()}} ships, ${{s.whales.toLocaleString()}} whales, ${{s.encounters.toLocaleString()}} encounters</div>`;
        }}
        document.getElementById('dash-stats').innerHTML = html;
    }})();

    // Grid interpolation — returns whale density, ship density, combined risk
    function sampleGrid(lat, lon) {{
        for (let rk in GRID_DATA) {{
            let g = GRID_DATA[rk];
            let [minLon, minLat, maxLon, maxLat] = g.bounds;
            if (lon >= minLon && lon <= maxLon && lat >= minLat && lat <= maxLat) {{
                let lons = g.lon_centers, lats = g.lat_centers;
                let i = 0, j = 0;
                for (let k = 0; k < lons.length - 1; k++) {{ if (lon >= lons[k]) i = k; }}
                for (let k = 0; k < lats.length - 1; k++) {{ if (lat >= lats[k]) j = k; }}
                let wd = 0, sd = 0;
                if (i < g.whale_density.length && j < g.whale_density[i].length) wd = g.whale_density[i][j];
                if (g.ship_density && i < g.ship_density.length && j < g.ship_density[i].length) sd = g.ship_density[i][j];
                return {{ density: wd, shipDensity: sd, risk: wd * sd, region: g.name }};
            }}
        }}
        return {{ density: 0, shipDensity: 0, risk: 0, region: "Open Ocean" }};
    }}

    function haversineKm(lat1, lon1, lat2, lon2) {{
        const R = 6371;
        const dLat = (lat2 - lat1) * Math.PI / 180;
        const dLon = (lon2 - lon1) * Math.PI / 180;
        const a = Math.sin(dLat/2)**2 + Math.cos(lat1*Math.PI/180)*Math.cos(lat2*Math.PI/180)*Math.sin(dLon/2)**2;
        return 2 * R * Math.asin(Math.sqrt(a));
    }}

    function riskColor(density) {{
        if (density > 0.6) return '#d32f2f';
        if (density > 0.3) return '#ff9800';
        if (density > 0.1) return '#fdd835';
        return '#4caf50';
    }}

    function riskLabel(density) {{
        if (density > 0.6) return 'CRITICAL';
        if (density > 0.3) return 'HIGH';
        if (density > 0.1) return 'MODERATE';
        return 'LOW';
    }}

    // Economic model — distKm at a given speed, with fuel and charter costs
    const ECON_NORMAL_SPEED_KN = 14;
    const ECON_FUEL_TONS_DAY = 40;
    const ECON_FUEL_COST_TON = 600;
    const ECON_CHARTER_DAY = 25000;

    function computeEconForRoute(distKm, speedKn) {{
        const distNm = distKm * 0.539957;
        const timeHrs = distNm / speedKn;
        const fuelTons = (timeHrs / 24) * ECON_FUEL_TONS_DAY;
        const fuelCost = fuelTons * ECON_FUEL_COST_TON;
        const charterCost = (timeHrs / 24) * ECON_CHARTER_DAY;
        return {{ distNm, timeHrs, fuelTons, fuelCost, charterCost, totalCost: fuelCost + charterCost }};
    }}

    // ---------- REROUTING ALGORITHM ----------
    // For each segment of the original route that passes through high whale density,
    // try lateral offsets perpendicular to the bearing to find a lower-risk waypoint.
    // Then smooth the result.
    function computeSaferRoute(originalPoints) {{
        const RISK_THRESHOLD = 0.15;
        const MAX_OFFSET_DEG = 1.5;
        const OFFSET_STEPS = 12;
        const STEP_SIZE = MAX_OFFSET_DEG / OFFSET_STEPS;

        let saferPoints = [originalPoints[0]];

        for (let k = 1; k < originalPoints.length - 1; k++) {{
            let pt = originalPoints[k];
            let s = sampleGrid(pt.lat, pt.lng);
            if (s.density <= RISK_THRESHOLD) {{
                saferPoints.push(pt);
                continue;
            }}

            let prev = originalPoints[k - 1];
            let next = originalPoints[k + 1];
            let bearLat = next.lat - prev.lat;
            let bearLng = next.lng - prev.lng;
            let mag = Math.sqrt(bearLat * bearLat + bearLng * bearLng);
            if (mag < 1e-8) {{ saferPoints.push(pt); continue; }}
            let perpLat = -bearLng / mag;
            let perpLng = bearLat / mag;

            let bestPt = pt;
            let bestDensity = s.density;
            let bestPenalty = s.density;

            for (let sign = -1; sign <= 1; sign += 2) {{
                for (let step = 1; step <= OFFSET_STEPS; step++) {{
                    let offLat = pt.lat + sign * perpLat * STEP_SIZE * step;
                    let offLng = pt.lng + sign * perpLng * STEP_SIZE * step;
                    let os = sampleGrid(offLat, offLng);
                    let distPenalty = step * STEP_SIZE * 0.05;
                    let penalty = os.density + distPenalty;
                    if (penalty < bestPenalty) {{
                        bestPenalty = penalty;
                        bestDensity = os.density;
                        bestPt = L.latLng(offLat, offLng);
                    }}
                }}
            }}
            saferPoints.push(bestPt);
        }}
        saferPoints.push(originalPoints[originalPoints.length - 1]);

        // Smooth pass: average neighboring offsets to avoid zig-zags
        let smoothed = [saferPoints[0]];
        for (let k = 1; k < saferPoints.length - 1; k++) {{
            let pLat = (saferPoints[k-1].lat + saferPoints[k].lat * 2 + saferPoints[k+1].lat) / 4;
            let pLng = (saferPoints[k-1].lng + saferPoints[k].lng * 2 + saferPoints[k+1].lng) / 4;
            smoothed.push(L.latLng(pLat, pLng));
        }}
        smoothed.push(saferPoints[saferPoints.length - 1]);

        return smoothed;
    }}

    function analyzePointList(points) {{
        let totalDist = 0;
        let densities = [];
        let waypoints = [];
        for (let k = 0; k < points.length; k++) {{
            let p = points[k];
            let s = sampleGrid(p.lat, p.lng);
            if (k > 0) totalDist += haversineKm(points[k-1].lat, points[k-1].lng, p.lat, p.lng);
            densities.push(s.density);
            waypoints.push({{ lat: p.lat, lng: p.lng, density: s.density, dist: totalDist, region: s.region }});
        }}
        let avgDensity = densities.length ? densities.reduce((a,b) => a+b, 0) / densities.length : 0;
        let maxDensity = densities.length ? Math.max(...densities) : 0;
        let highRiskPct = densities.length ? densities.filter(d => d > 0.3).length / densities.length * 100 : 0;
        let criticalPct = densities.length ? densities.filter(d => d > 0.6).length / densities.length * 100 : 0;
        return {{ waypoints, totalDist, avgDensity, maxDensity, highRiskPct, criticalPct, densities }};
    }}

    // Listen for drawn routes
    window.addEventListener('load', function() {{
        let mapObj = null;
        document.querySelectorAll('.folium-map').forEach(el => {{
            if (el._leaflet_id) mapObj = el;
        }});

        // Get the leaflet map instance
        let theMap = null;
        for (let key in window) {{
            if (window[key] && window[key]._container && window[key].getCenter) {{
                theMap = window[key];
                break;
            }}
        }}
        if (!theMap) {{
            let maps = Object.values(window).filter(v => v && v._leaflet_id && v.getCenter);
            if (maps.length > 0) theMap = maps[0];
        }}

        if (theMap) {{
            _theMap = theMap;
            theMap.on('draw:created', function(e) {{
                if (_drawnLine) {{ theMap.removeLayer(_drawnLine); }}
                _drawnLine = e.layer;
                theMap.addLayer(_drawnLine);
                let latlngs = e.layer.getLatLngs();
                analyzeRoute(latlngs, theMap);
            }});
        }}
    }});

    function interpolateRoute(latlngs) {{
        let interpPoints = [];
        for (let k = 0; k < latlngs.length; k++) {{
            interpPoints.push(latlngs[k]);
            if (k < latlngs.length - 1) {{
                let segDist = haversineKm(latlngs[k].lat, latlngs[k].lng, latlngs[k+1].lat, latlngs[k+1].lng);
                let nInterp = Math.max(1, Math.floor(segDist / 5));
                for (let s = 1; s < nInterp; s++) {{
                    let frac = s / nInterp;
                    interpPoints.push(L.latLng(
                        latlngs[k].lat + frac * (latlngs[k+1].lat - latlngs[k].lat),
                        latlngs[k].lng + frac * (latlngs[k+1].lng - latlngs[k].lng)
                    ));
                }}
            }}
        }}
        return interpPoints;
    }}

    function analyzeRoute(latlngs, theMap) {{
        if (routeLayer) theMap.removeLayer(routeLayer);
        routeLayer = L.layerGroup().addTo(theMap);

        // --- Original route ---
        let interpPoints = interpolateRoute(latlngs);
        let orig = analyzePointList(interpPoints);

        // Draw colored segments for original route
        for (let k = 1; k < orig.waypoints.length; k++) {{
            let color = riskColor(orig.waypoints[k].density);
            L.polyline(
                [[orig.waypoints[k-1].lat, orig.waypoints[k-1].lng],
                 [orig.waypoints[k].lat, orig.waypoints[k].lng]],
                {{ color: color, weight: 5, opacity: 0.85 }}
            ).addTo(routeLayer);
        }}
        orig.waypoints.forEach(wp => {{
            if (wp.density > 0.3) {{
                L.circleMarker([wp.lat, wp.lng], {{
                    radius: 4 + wp.density * 8, color: riskColor(wp.density),
                    fill: true, fillColor: riskColor(wp.density), fillOpacity: 0.7, weight: 1
                }}).bindPopup(`Whale density: ${{(wp.density*100).toFixed(1)}}%<br>Risk: ${{riskLabel(wp.density)}}`).addTo(routeLayer);
            }}
        }});

        // --- Safer reroute ---
        let saferPts = computeSaferRoute(interpPoints);
        let saferInterp = interpolateRoute(saferPts);
        let safer = analyzePointList(saferInterp);

        let hasHighRisk = orig.densities.some(d => d > 0.15);
        let rerouteImproved = safer.avgDensity < orig.avgDensity * 0.85 && hasHighRisk;

        if (rerouteImproved) {{
            let saferLatLngs = safer.waypoints.map(w => [w.lat, w.lng]);
            L.polyline(saferLatLngs, {{
                color: '#00c853', weight: 4, opacity: 0.85,
                dashArray: '10,8'
            }}).bindPopup('<b>Suggested Safer Route</b><br>Lower whale encounter risk').addTo(routeLayer);

            L.circleMarker(saferLatLngs[0], {{
                radius: 6, color: '#00c853', fill: true, fillColor: '#00c853', fillOpacity: 1, weight: 2
            }}).addTo(routeLayer);
            L.circleMarker(saferLatLngs[saferLatLngs.length - 1], {{
                radius: 6, color: '#00c853', fill: true, fillColor: '#00c853', fillOpacity: 1, weight: 2
            }}).addTo(routeLayer);
        }}

        // --- Summary cards ---
        let origEcon = computeEconForRoute(orig.totalDist, ECON_NORMAL_SPEED_KN);
        let summaryHtml = `
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:6px;margin-bottom:8px;">
                <div style="background:#e8eaf6;border-radius:6px;padding:6px;text-align:center;">
                    <div style="font-size:16px;font-weight:700;">${{orig.totalDist.toFixed(0)}} km</div>
                    <div style="font-size:9px;color:#666;">Distance</div></div>
                <div style="background:#ffebee;border-radius:6px;padding:6px;text-align:center;">
                    <div style="font-size:16px;font-weight:700;">${{orig.highRiskPct.toFixed(1)}}%</div>
                    <div style="font-size:9px;color:#666;">High+ Risk</div></div>
                <div style="background:#fff3e0;border-radius:6px;padding:6px;text-align:center;">
                    <div style="font-size:16px;font-weight:700;">${{(orig.maxDensity*100).toFixed(0)}}%</div>
                    <div style="font-size:9px;color:#666;">Peak Density</div></div>
                <div style="background:#e8f5e9;border-radius:6px;padding:6px;text-align:center;">
                    <div style="font-size:16px;font-weight:700;">${{origEcon.distNm.toFixed(1)}} nm</div>
                    <div style="font-size:9px;color:#666;">Nautical Miles</div></div>
            </div>`;
        document.getElementById('route-summary').innerHTML = summaryHtml;

        // --- Reroute comparison ---
        let rerouteHtml = '';
        if (rerouteImproved) {{
            let saferEcon = computeEconForRoute(safer.totalDist, ECON_NORMAL_SPEED_KN);
            let extraDist = safer.totalDist - orig.totalDist;
            let extraDistNm = extraDist * 0.539957;
            let extraTime = saferEcon.timeHrs - origEcon.timeHrs;
            let extraFuel = saferEcon.fuelCost - origEcon.fuelCost;
            let extraCharter = saferEcon.charterCost - origEcon.charterCost;
            let extraTotal = saferEcon.totalCost - origEcon.totalCost;
            let riskReduction = ((1 - safer.avgDensity / (orig.avgDensity + 1e-10)) * 100);

            rerouteHtml = `
            <div style="background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:8px;padding:10px;border:1px solid #a5d6a7;">
                <div style="font-weight:700;color:#2e7d32;margin-bottom:6px;">
                    &#9989; Safer Route Available <span style="font-weight:400;color:#558b2f;font-size:11px;">(dashed green line)</span>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;font-size:12px;margin-bottom:6px;">
                    <div style="background:white;border-radius:4px;padding:5px;text-align:center;">
                        <div style="font-size:14px;font-weight:700;color:#2e7d32;">${{riskReduction.toFixed(0)}}%</div>
                        <div style="font-size:9px;color:#666;">Risk Reduction</div></div>
                    <div style="background:white;border-radius:4px;padding:5px;text-align:center;">
                        <div style="font-size:14px;font-weight:700;color:#1565c0;">${{safer.totalDist.toFixed(0)}} km</div>
                        <div style="font-size:9px;color:#666;">Safer Distance</div></div>
                    <div style="background:white;border-radius:4px;padding:5px;text-align:center;">
                        <div style="font-size:14px;font-weight:700;color:#e65100;">${{safer.highRiskPct.toFixed(1)}}%</div>
                        <div style="font-size:9px;color:#666;">High Risk (was ${{orig.highRiskPct.toFixed(1)}}%)</div></div>
                </div>
                <div style="font-size:11px;color:#555;line-height:1.6;">
                    Extra distance: <b>+${{extraDist.toFixed(1)}} km (${{extraDistNm.toFixed(1)}} nm)</b> &bull;
                    Extra time: <b>+${{extraTime.toFixed(1)}} hrs</b> &bull;
                    Peak density: <b>${{(safer.maxDensity*100).toFixed(0)}}%</b> (was ${{(orig.maxDensity*100).toFixed(0)}}%)
                </div>
            </div>`;
        }} else if (hasHighRisk) {{
            rerouteHtml = `
            <div style="background:#fff3e0;border-radius:8px;padding:8px;border:1px solid #ffe0b2;font-size:12px;color:#e65100;">
                <b>No significantly safer reroute found.</b> The route may already be near-optimal, or whale activity is widespread in this corridor.
                Consider slowing to 10 knots in high-risk segments.
            </div>`;
        }} else {{
            rerouteHtml = `
            <div style="background:#e8f5e9;border-radius:8px;padding:8px;border:1px solid #c8e6c9;font-size:12px;color:#2e7d32;">
                <b>Route is low-risk.</b> No reroute needed &mdash; whale encounter probability is minimal along this path.
            </div>`;
        }}
        document.getElementById('route-reroute').innerHTML = rerouteHtml;

        // --- Economic comparison ---
        let econHtml = '';
        if (rerouteImproved) {{
            let saferEcon = computeEconForRoute(safer.totalDist, ECON_NORMAL_SPEED_KN);
            let extraTotal = saferEcon.totalCost - origEcon.totalCost;
            econHtml = `
            <div style="background:#263238;color:white;border-radius:8px;padding:10px;">
                <div style="font-weight:700;margin-bottom:8px;">Economic Impact Comparison</div>
                <table style="width:100%;font-size:11.5px;border-collapse:collapse;">
                    <tr style="border-bottom:1px solid #455a64;">
                        <td style="padding:4px 6px;"></td>
                        <td style="padding:4px 6px;font-weight:700;color:#ff8a80;">Original Route</td>
                        <td style="padding:4px 6px;font-weight:700;color:#69f0ae;">Safer Route</td>
                        <td style="padding:4px 6px;font-weight:700;color:#fff176;">Delta</td>
                    </tr>
                    <tr style="border-bottom:1px solid #37474f;">
                        <td style="padding:4px 6px;color:#90a4ae;">Distance</td>
                        <td style="padding:4px 6px;">${{origEcon.distNm.toFixed(1)}} nm</td>
                        <td style="padding:4px 6px;">${{saferEcon.distNm.toFixed(1)}} nm</td>
                        <td style="padding:4px 6px;">+${{(saferEcon.distNm - origEcon.distNm).toFixed(1)}} nm</td>
                    </tr>
                    <tr style="border-bottom:1px solid #37474f;">
                        <td style="padding:4px 6px;color:#90a4ae;">Transit Time</td>
                        <td style="padding:4px 6px;">${{origEcon.timeHrs.toFixed(1)}} hrs</td>
                        <td style="padding:4px 6px;">${{saferEcon.timeHrs.toFixed(1)}} hrs</td>
                        <td style="padding:4px 6px;">+${{(saferEcon.timeHrs - origEcon.timeHrs).toFixed(1)}} hrs</td>
                    </tr>
                    <tr style="border-bottom:1px solid #37474f;">
                        <td style="padding:4px 6px;color:#90a4ae;">Fuel Cost</td>
                        <td style="padding:4px 6px;">$${{Math.round(origEcon.fuelCost).toLocaleString()}}</td>
                        <td style="padding:4px 6px;">$${{Math.round(saferEcon.fuelCost).toLocaleString()}}</td>
                        <td style="padding:4px 6px;">+$${{Math.round(saferEcon.fuelCost - origEcon.fuelCost).toLocaleString()}}</td>
                    </tr>
                    <tr style="border-bottom:1px solid #37474f;">
                        <td style="padding:4px 6px;color:#90a4ae;">Charter Cost</td>
                        <td style="padding:4px 6px;">$${{Math.round(origEcon.charterCost).toLocaleString()}}</td>
                        <td style="padding:4px 6px;">$${{Math.round(saferEcon.charterCost).toLocaleString()}}</td>
                        <td style="padding:4px 6px;">+$${{Math.round(saferEcon.charterCost - origEcon.charterCost).toLocaleString()}}</td>
                    </tr>
                    <tr>
                        <td style="padding:6px;color:#90a4ae;font-weight:700;">Total Cost</td>
                        <td style="padding:6px;font-weight:700;font-size:13px;">$${{Math.round(origEcon.totalCost).toLocaleString()}}</td>
                        <td style="padding:6px;font-weight:700;font-size:13px;">$${{Math.round(saferEcon.totalCost).toLocaleString()}}</td>
                        <td style="padding:6px;font-weight:700;font-size:13px;color:#fff176;">+$${{Math.round(extraTotal).toLocaleString()}}</td>
                    </tr>
                </table>
                <div style="margin-top:8px;font-size:11px;color:#b0bec5;">
                    The safer route adds <b>$${{Math.round(extraTotal).toLocaleString()}}</b> in operating costs but
                    reduces whale encounter risk by <b>${{((1 - safer.avgDensity / (orig.avgDensity + 1e-10)) * 100).toFixed(0)}}%</b>.
                </div>
                <div style="font-size:10px;color:#78909c;margin-top:4px;">Based on 14kn cruising speed, 40t/day fuel burn, $600/t bunker, $25k/day charter rate</div>
            </div>`;
        }} else {{
            econHtml = `
            <div style="background:#263238;color:white;border-radius:8px;padding:10px;">
                <div style="font-weight:700;margin-bottom:6px;">Economic Summary (Original Route at 14kn)</div>
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;font-size:12px;">
                    <div>Transit: <b>${{origEcon.timeHrs.toFixed(1)}} hrs</b></div>
                    <div>Fuel: <b>$${{Math.round(origEcon.fuelCost).toLocaleString()}}</b></div>
                    <div>Charter: <b>$${{Math.round(origEcon.charterCost).toLocaleString()}}</b></div>
                </div>
                <div style="margin-top:6px;font-size:14px;">Total voyage cost: <b style="color:#80cbc4;">$${{Math.round(origEcon.totalCost).toLocaleString()}}</b></div>
                <div style="font-size:10px;color:#78909c;margin-top:4px;">Based on 14kn cruising speed, 40t/day fuel burn, $600/t bunker, $25k/day charter rate</div>
            </div>`;
        }}
        document.getElementById('route-econ').innerHTML = econHtml;

        // --- Chart: original vs safer density profile ---
        if (routeChart) routeChart.destroy();
        let labels = orig.waypoints.map(w => w.dist.toFixed(0));
        let origData = orig.waypoints.map(w => (w.density * 100).toFixed(1));
        let datasets = [{{
            label: 'Original Route',
            data: origData,
            borderColor: '#ff5252',
            backgroundColor: 'rgba(255,82,82,0.12)',
            fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2,
        }}];

        if (rerouteImproved) {{
            let saferLabels = safer.waypoints.map(w => w.dist.toFixed(0));
            let saferData = [];
            let si = 0;
            for (let oi = 0; oi < labels.length; oi++) {{
                let origKm = parseFloat(labels[oi]);
                while (si < safer.waypoints.length - 1 && safer.waypoints[si].dist < origKm) si++;
                saferData.push((safer.waypoints[Math.min(si, safer.waypoints.length-1)].density * 100).toFixed(1));
            }}
            datasets.push({{
                label: 'Safer Route',
                data: saferData,
                borderColor: '#00c853',
                backgroundColor: 'rgba(0,200,83,0.10)',
                fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2,
                borderDash: [6, 3],
            }});
        }}

        routeChart = new Chart(document.getElementById('routeChart'), {{
            type: 'line',
            data: {{ labels: labels, datasets: datasets }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ display: rerouteImproved, labels: {{ font: {{ size: 10 }} }} }} }},
                scales: {{
                    x: {{ title: {{ display: true, text: 'Distance (km)', font: {{ size: 10 }} }}, ticks: {{ maxTicksLimit: 8, font: {{ size: 9 }} }} }},
                    y: {{ title: {{ display: true, text: 'Whale Density %', font: {{ size: 10 }} }}, min: 0, ticks: {{ font: {{ size: 9 }} }} }}
                }}
            }}
        }});

        _lastOrig = orig;
        _lastSafer = safer;
        _lastRerouteImproved = rerouteImproved;
        document.getElementById('speed-slider').value = 14;
        document.getElementById('speed-val').textContent = '14 kn';
        document.getElementById('speed-slider-wrap').style.display = 'block';
        document.getElementById('route-panel').style.display = 'block';
    }}

    // --- Speed slider: live-update economics ---
    document.getElementById('speed-slider').addEventListener('input', function() {{
        let speed = parseFloat(this.value);
        document.getElementById('speed-val').textContent = speed.toFixed(1) + ' kn';
        if (!_lastOrig) return;
        updateEconForSpeed(speed);
    }});

    function updateEconForSpeed(speedKn) {{
        let orig = _lastOrig;
        let safer = _lastSafer;
        let rerouteImproved = _lastRerouteImproved;
        let origEcon = computeEconForRoute(orig.totalDist, speedKn);

        let econHtml = '';
        if (rerouteImproved) {{
            let saferEcon = computeEconForRoute(safer.totalDist, speedKn);
            let extraTotal = saferEcon.totalCost - origEcon.totalCost;
            econHtml = `
            <div style="background:#263238;color:white;border-radius:8px;padding:10px;">
                <div style="font-weight:700;margin-bottom:8px;">Economic Impact at ${{speedKn.toFixed(1)}} kn</div>
                <table style="width:100%;font-size:11.5px;border-collapse:collapse;">
                    <tr style="border-bottom:1px solid #455a64;">
                        <td style="padding:4px 6px;"></td>
                        <td style="padding:4px 6px;font-weight:700;color:#ff8a80;">Original</td>
                        <td style="padding:4px 6px;font-weight:700;color:#69f0ae;">Safer</td>
                        <td style="padding:4px 6px;font-weight:700;color:#fff176;">Delta</td>
                    </tr>
                    <tr style="border-bottom:1px solid #37474f;">
                        <td style="padding:4px 6px;color:#90a4ae;">Transit Time</td>
                        <td style="padding:4px 6px;">${{origEcon.timeHrs.toFixed(1)}} hrs</td>
                        <td style="padding:4px 6px;">${{saferEcon.timeHrs.toFixed(1)}} hrs</td>
                        <td style="padding:4px 6px;">+${{(saferEcon.timeHrs - origEcon.timeHrs).toFixed(1)}} hrs</td>
                    </tr>
                    <tr style="border-bottom:1px solid #37474f;">
                        <td style="padding:4px 6px;color:#90a4ae;">Fuel Cost</td>
                        <td style="padding:4px 6px;">$${{Math.round(origEcon.fuelCost).toLocaleString()}}</td>
                        <td style="padding:4px 6px;">$${{Math.round(saferEcon.fuelCost).toLocaleString()}}</td>
                        <td style="padding:4px 6px;">+$${{Math.round(saferEcon.fuelCost - origEcon.fuelCost).toLocaleString()}}</td>
                    </tr>
                    <tr>
                        <td style="padding:6px;color:#90a4ae;font-weight:700;">Total</td>
                        <td style="padding:6px;font-weight:700;font-size:13px;">$${{Math.round(origEcon.totalCost).toLocaleString()}}</td>
                        <td style="padding:6px;font-weight:700;font-size:13px;">$${{Math.round(saferEcon.totalCost).toLocaleString()}}</td>
                        <td style="padding:6px;font-weight:700;font-size:13px;color:#fff176;">+$${{Math.round(extraTotal).toLocaleString()}}</td>
                    </tr>
                </table>
                <div style="font-size:10px;color:#78909c;margin-top:4px;">40t/day fuel burn &middot; $600/t bunker &middot; $25k/day charter</div>
            </div>`;
        }} else {{
            econHtml = `
            <div style="background:#263238;color:white;border-radius:8px;padding:10px;">
                <div style="font-weight:700;margin-bottom:6px;">Economics at ${{speedKn.toFixed(1)}} kn</div>
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;font-size:12px;">
                    <div>Transit: <b>${{origEcon.timeHrs.toFixed(1)}} hrs</b></div>
                    <div>Fuel: <b>$${{Math.round(origEcon.fuelCost).toLocaleString()}}</b></div>
                    <div>Charter: <b>$${{Math.round(origEcon.charterCost).toLocaleString()}}</b></div>
                </div>
                <div style="margin-top:6px;font-size:14px;">Total voyage cost: <b style="color:#80cbc4;">$${{Math.round(origEcon.totalCost).toLocaleString()}}</b></div>
                <div style="font-size:10px;color:#78909c;margin-top:4px;">40t/day fuel burn &middot; $600/t bunker &middot; $25k/day charter</div>
            </div>`;
        }}
        document.getElementById('route-econ').innerHTML = econHtml;
    }}
    </script>
    """

    m.get_root().html.add_child(Element(route_analysis_html))

    m.save(str(OUTPUT))
    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"\nMap saved to {OUTPUT} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    make_map()
