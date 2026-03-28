# Neural Network Collision Risk Prediction Model

## Overview

This document describes a PyTorch neural network designed to predict the risk of whale-ship collisions across four major US maritime regions. The model ingests spatial, temporal, environmental, and biological features to produce a continuous risk score for any given ocean grid cell in any month of the year.

The model is fully standalone — it does not depend on or modify the existing spatial analysis pipeline. It trains from the same processed data (ship AIS records and OBIS whale sightings) but builds its own feature set and learns its own patterns.

---

## Data Sources

| Source | Description | Records |
|--------|-------------|---------|
| NOAA AIS 2024 | Ship positions from 12 monthly snapshots (Jan–Dec 2024), filtered to large vessels (types 60–89) | ~7.7M across 4 regions |
| OBIS | Ocean Biodiversity Information System whale sightings (2000–2024), 7 major species | ~269K across 4 regions |
| Encounters | Pre-computed close encounters (ship within 5 km of whale sighting) | ~13M across 4 regions |

### Regions Covered

| Region | Bounding Box (lon, lat) | Why It Matters |
|--------|------------------------|----------------|
| US West Coast | -130 to -115, 30 to 50 | Blue, humpback, gray whale migration corridor |
| US East Coast | -82 to -65, 24 to 45 | North Atlantic right whale calving/feeding grounds |
| Gulf of Mexico | -98 to -80, 18 to 31 | Sperm whale habitat, major shipping lanes |
| Hawaii | -162 to -153, 17 to 24 | Humpback whale breeding grounds |

---

## Feature Engineering

The ocean is divided into a grid of 0.25° cells (~28 km). For each cell in each month, we compute 18 features by querying KD-trees built from the raw ship and whale data. A search radius of 0.375° around each cell center captures nearby observations.

### Feature Table

| # | Feature | Category | Description | Why It Matters |
|---|---------|----------|-------------|----------------|
| 1 | `lat_norm` | Spatial | Latitude normalized to [0,1] within the region | Captures north-south habitat gradients |
| 2 | `lon_norm` | Spatial | Longitude normalized to [0,1] within the region | Captures offshore vs. nearshore patterns |
| 3 | `month_sin` | Temporal | sin(2π × month / 12) | Cyclical encoding so Dec (12) is close to Jan (1), not far away |
| 4 | `month_cos` | Temporal | cos(2π × month / 12) | Second component of the cyclical pair — together they uniquely identify any month |
| 5 | `season_sin` | Temporal | sin(2π × quarter / 4) | Coarser seasonal signal (winter/spring/summer/fall) |
| 6 | `season_cos` | Temporal | cos(2π × quarter / 4) | Second component of seasonal encoding |
| 7 | `ship_count` | Traffic | Number of ship AIS positions in the cell that month | Direct measure of traffic intensity |
| 8 | `avg_sog` | Traffic | Mean speed over ground (knots) of ships in cell | Faster ships are more lethal in collisions |
| 9 | `std_sog` | Traffic | Standard deviation of ship speeds | High variance = mixed traffic (maneuvering near ports, speed transitions) |
| 10 | `avg_ship_length` | Traffic | Mean length of vessels in cell (meters) | Larger ships pose greater strike risk |
| 11 | `cargo_tanker_frac` | Traffic | Fraction of ships that are cargo or tanker type | These vessel types are the most dangerous for whale strikes |
| 12 | `unique_vessels` | Traffic | Count of distinct MMSIs (vessel identifiers) | Distinguishes heavy single-ship traffic from many different vessels |
| 13 | `whale_count` | Biological | Whale sightings in cell across ALL time (full OBIS dataset) | Represents whether the cell is whale habitat at all |
| 14 | `whale_month_count` | Biological | Whale sightings in cell for that specific month only | Captures seasonal whale presence — migration, breeding |
| 15 | `n_species` | Biological | Number of distinct whale species observed in cell | Higher biodiversity = more whale activity, diverse risk |
| 16 | `avg_sst` | Environmental | Mean sea surface temperature from whale records (°C) | Whales prefer specific temperature ranges; SST drives prey availability |
| 17 | `avg_bathymetry` | Environmental | Mean ocean depth from whale records (meters) | Whale species have depth preferences; continental shelf edges are high-risk |
| 18 | `avg_individual_count` | Biological | Mean number of individuals per sighting event | Areas with group sightings have higher collision probability per transit |

### Why Cyclical Encoding?

Standard integer encoding (month = 1, 2, ..., 12) tells the network that December (12) is far from January (1). In reality, they're adjacent. Cyclical encoding using sin/cos maps the 12 months onto a circle, so the network can learn smooth seasonal transitions. The same applies to season encoding.

### Whale Habitat vs. Monthly Activity

OBIS whale sighting data spans 2000–2024 but is very unevenly distributed across months. Some months may have only a handful of records for a region. To handle this:

- **`whale_count`** (habitat): Uses the full dataset regardless of month. This tells the model "whales use this area."
- **`whale_month_count`** (activity): Uses only sightings from that specific month. This tells the model "whales are specifically active here in July."

The model learns to combine both signals — a cell with high habitat count but low monthly activity in January still has some risk, but less than a cell with high activity in January.

---

## Target Variable

The target is the **number of close encounters** (ship within 5 km of whale sighting) observed in each grid cell, transformed as:

```
y = log(1 + encounter_count) / max(log(1 + encounter_count))
```

This produces a value in [0, 1] where:
- 0 = no encounters ever observed in this cell
- 1 = the cell with the most encounters across all regions and months

The log transform is critical because encounter counts span many orders of magnitude (0 to 4M+). Without it, the network would only learn to distinguish "zero" from "astronomical" and ignore the important gradations in between.

---

## Network Architecture

```
Input (18 features)
    │
    ▼
Linear(18 → 128) → BatchNorm → ReLU → Dropout(0.2)
    │
    ▼
Linear(128 → 256) → BatchNorm → ReLU → Dropout(0.2)
    │
    ▼
Linear(256 → 128) → BatchNorm → ReLU → Dropout(0.15)
    │
    ▼
Linear(128 → 64) → ReLU
    │
    ▼
Linear(64 → 1)  →  Output (collision risk score)
```

**Total parameters**: ~69,000

### Layer-by-Layer Explanation

| Layer | Purpose |
|-------|---------|
| **Linear** | Standard fully-connected layer: `output = input × weights + bias`. Each neuron computes a weighted sum of all inputs. |
| **BatchNorm** | Normalizes activations within each mini-batch to have zero mean and unit variance. This stabilizes training, allows higher learning rates, and acts as mild regularization. |
| **ReLU** | Rectified Linear Unit: `f(x) = max(0, x)`. Introduces non-linearity so the network can learn complex patterns. Without it, stacking linear layers would just produce another linear function. |
| **Dropout** | During training, randomly zeroes out 20% (or 15%) of neurons per forward pass. Forces the network to not rely on any single neuron, improving generalization to unseen data. Disabled during inference. |

### Why This Architecture?

- **Expanding then contracting** (128 → 256 → 128 → 64): The wider middle layers give the network capacity to discover complex interactions between features. The narrowing forces it to compress those patterns into a compact representation.
- **5 layers deep**: Deep enough to capture non-linear feature interactions (e.g., "high whale count AND high ship speed AND winter month = high risk") but not so deep that training becomes unstable.
- **No sigmoid on output**: The target is already in [0, 1] via normalization. Omitting a sigmoid lets the network output unconstrained values during training, which are clipped to [0, 1] at prediction time.

---

## Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Optimizer** | Adam (lr=0.001, weight_decay=1e-5) | Adam adapts learning rates per-parameter; weight decay provides L2 regularization to prevent overfitting |
| **Scheduler** | Cosine Annealing (T_max=120) | Smoothly decays the learning rate from 0.001 to ~0, allowing fine-grained convergence in later epochs |
| **Loss function** | MSE (Mean Squared Error) | Standard regression loss; penalizes large errors quadratically, which is appropriate since we want to be especially accurate for high-risk cells |
| **Batch size** | 512 | Large enough for stable BatchNorm statistics, small enough for good stochastic gradient estimates |
| **Epochs** | 120 | Determined empirically — validation loss plateaus around epoch 80–100 |
| **Train/Val split** | 80/20 | Fixed random seed (42) for reproducibility |

### Training Process

1. **Shuffle** all 63,746 samples with a fixed seed
2. **Split**: 80% train (50,997), 20% validation (12,749)
3. **Z-score normalize** all 18 features: `x' = (x - mean) / std`
4. Each epoch:
   - Iterate through training data in batches of 512
   - Forward pass → compute MSE loss → backward pass → update weights
   - Evaluate on validation set (no gradient computation)
   - Step the learning rate scheduler
5. After 120 epochs, freeze the model for prediction

---

## Feature Importance

Feature importance is computed via **gradient-based attribution**:

1. Pass all training samples through the model with gradients enabled on the input
2. Compute the sum of all outputs
3. Backpropagate to get `∂output/∂input` for every sample and feature
4. Take the mean absolute gradient per feature across all samples
5. Normalize so importances sum to 100%

This tells us: "If I perturb this feature slightly, how much does the predicted risk change on average?" Features with high importance have a strong influence on the model's predictions.

This is more reliable than permutation importance for neural networks because it captures the actual learned sensitivity of the network to each input dimension.

---

## Prediction and Output

After training, the model predicts risk for every (cell, month) combination in the dataset. The outputs are then aggregated:

- **Monthly profiles**: Mean and max predicted risk per month per region, revealing seasonal patterns
- **Spatial map**: For each cell, take the peak risk across all 12 months and display on an interactive map
- **Top hotspots**: The 30 highest-risk cells per region, ranked by predicted risk with actual encounter counts shown alongside for validation

### Interpreting the Risk Score

| Score Range | Label | Meaning |
|-------------|-------|---------|
| 0.0 – 0.1 | Low | Minimal whale-ship overlap predicted |
| 0.1 – 0.3 | Moderate | Some overlap; standard caution recommended |
| 0.3 – 0.6 | High | Significant overlap; speed reduction advised |
| 0.6 – 1.0 | Critical | Highest predicted collision risk; rerouting recommended |

---

## What the Neural Network Discovers That Rules Can't

The key advantage of using a neural network over hand-coded rules is its ability to learn **non-linear interactions** between features automatically:

1. **Seasonal × Spatial interactions**: The model learns that the same ocean cell has very different risk levels in January vs. July, without us manually encoding migration calendars for each species.

2. **Traffic pattern nuances**: A cell with 100 slow-moving cargo ships has a different risk profile than a cell with 100 fast-moving tankers. The network learns these distinctions from the combination of `ship_count`, `avg_sog`, `cargo_tanker_frac`, and `avg_ship_length`.

3. **Environmental thresholds**: Whale species congregate at specific SST and bathymetry ranges. The network can learn that "SST between 14–17°C AND depth of 200–1000m AND month 6–9" is a high-risk combination without us knowing this a priori.

4. **Cross-region transfer**: By training on all 4 regions simultaneously, the model can transfer patterns. If it learns that "high whale habitat count + high cargo fraction + warm month = high risk" on the West Coast, it can apply that logic to similar cells on the East Coast even if East Coast data is sparser.

5. **Diminishing returns**: The relationship between ship count and risk isn't linear. Going from 0 to 100 ships matters more than going from 1000 to 1100. The network's non-linear activations naturally capture this.

---

## Limitations and Future Work

- **Temporal mismatch**: Ship data is from 2024 only; whale data spans 2000–2024. The model assumes whale habitat patterns are relatively stable, which is approximately true but doesn't account for climate-driven shifts.
- **No real-time data**: Predictions are based on historical patterns, not live AIS feeds or real-time whale detections.
- **Grid resolution**: 0.25° cells (~28 km) are coarse. Higher resolution would improve accuracy but dramatically increase compute time and memory.
- **No ocean current or prey data**: Whale presence is strongly influenced by prey (krill, fish) driven by ocean currents. Adding satellite chlorophyll-a or current data could improve predictions.
- **Single output**: The model predicts aggregate risk, not per-species risk. A multi-output network could predict species-specific collision probabilities.
