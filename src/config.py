"""Project configuration: regions, species, vessel types, and analysis parameters."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_SHIPS_DIR = DATA_DIR / "raw" / "ships"
RAW_WHALES_DIR = DATA_DIR / "raw" / "whales"
PROCESSED_DIR = DATA_DIR / "processed"

AIS_BASE_URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{month:02d}_{day:02d}.zip"
OBIS_API_URL = "https://api.obis.org/v3/occurrence"

# Major whale species to track (scientific name -> common name)
WHALE_SPECIES = {
    "Balaenoptera musculus": "Blue Whale",
    "Megaptera novaeangliae": "Humpback Whale",
    "Eubalaena glacialis": "North Atlantic Right Whale",
    "Balaenoptera physalus": "Fin Whale",
    "Eschrichtius robustus": "Gray Whale",
    "Physeter macrocephalus": "Sperm Whale",
    "Balaenoptera acutorostrata": "Minke Whale",
}

# AIS vessel type codes for large commercial vessels
LARGE_VESSEL_TYPES = {
    60: "Passenger",
    61: "Passenger, Hazardous A",
    62: "Passenger, Hazardous B",
    63: "Passenger, Hazardous C",
    64: "Passenger, Hazardous D",
    69: "Passenger, No additional info",
    70: "Cargo",
    71: "Cargo, Hazardous A",
    72: "Cargo, Hazardous B",
    73: "Cargo, Hazardous C",
    74: "Cargo, Hazardous D",
    79: "Cargo, No additional info",
    80: "Tanker",
    81: "Tanker, Hazardous A",
    82: "Tanker, Hazardous B",
    83: "Tanker, Hazardous C",
    84: "Tanker, Hazardous D",
    89: "Tanker, No additional info",
}
LARGE_VESSEL_TYPE_RANGE = range(60, 90)

# Key whale hotspot regions: (name, min_lon, min_lat, max_lon, max_lat)
REGIONS = {
    "us_west_coast": {
        "name": "US West Coast (CA/OR/WA)",
        "bounds": (-130, 30, -115, 50),
        "description": "Blue, humpback, gray whale migration corridor",
    },
    "us_east_coast": {
        "name": "US East Coast (MA to FL)",
        "bounds": (-82, 24, -65, 45),
        "description": "North Atlantic right whale calving/feeding grounds",
    },
    "gulf_of_mexico": {
        "name": "Gulf of Mexico",
        "bounds": (-98, 18, -80, 31),
        "description": "Sperm whale habitat, major shipping lanes",
    },
    "hawaii": {
        "name": "Hawaii",
        "bounds": (-162, 17, -153, 24),
        "description": "Humpback whale breeding grounds",
    },
    "alaska": {
        "name": "Alaska / Gulf of Alaska",
        "bounds": (-170, 50, -130, 65),
        "description": "Humpback, gray whale feeding grounds",
    },
}

# Spatial analysis parameters
PROXIMITY_THRESHOLD_KM = 5.0       # Distance to flag a "close encounter"
GRID_CELL_SIZE_DEG = 0.1           # ~11km grid cells for density analysis
ENCOUNTER_RISK_RADIUS_KM = 10.0    # Radius for encounter risk prediction

# Ship speed filter: only consider moving ships (knots)
MIN_SHIP_SPEED_KNOTS = 1.0
