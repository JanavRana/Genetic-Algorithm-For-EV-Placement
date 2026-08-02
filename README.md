# EV Charging Station Optimiser

Optimal placement of EV charging stations in Surat, India using a
**Genetic Algorithm** with real city road data and interactive map visualisation.



## Project Structure


ev_charging_project/
├── main.py                # Orchestrator — run this
├── population_density.py  # Synthetic population density generation
├── genetic_algorithm.py   # Full GA implementation (no external GA libs)
├── ga_integration.py      # Convert GA output → lat/lon station dicts
├── map_renderer.py        # Folium map + density overlay + markers
├── requirements.txt       # Python dependencies
└── README.md              # This file




## Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```


### 2 — Run

```bash
python main.py
```

You will be prompted:


How many EV charging stations do you want? (e.g. 10): 12


The script will then:

1. Download the Surat road network from OpenStreetMap (~5–15 s, internet required)
2. Generate a synthetic multi-hotspot population density surface
3. Run the Genetic Algorithm for 150 generations (~5–20 s depending on CPU)
4. Build an interactive HTML map
5. Automatically open `ev_charging_map.html` in your browser

## Configuration

All GA parameters live in `main.py` and can be tuned:

config = GAConfig(
    n_stations      = <from user>,
    population_size = 80,      # larger → better quality, slower
    n_generations   = 150,     # more → better quality, slower
    mutation_rate   = 0.15,    # 0–1; higher → more exploration
    elite_fraction  = 0.10,    # fraction of population kept unchanged
    coverage_scale  = 15.0,    # grid cells; controls station "reach"
    seed            = None,    # None = random each run (BONUS feature)
)

## Dependencies

| Package      | Purpose                                     |
|--------------|---------------------------------------------|
| `numpy`      | Array maths, GA operations                  |
| `scipy`      | Gaussian smoothing of density               |
| `matplotlib` | Render density PNG for map overlay          |
| `folium`     | Interactive Leaflet.js map                  |
| `osmnx`      | Download & parse OpenStreetMap road network |



# ga_integration.py
-----------------
Bridge between the Genetic Algorithm (grid indices) and the real-world
map coordinates (latitude / longitude).

Responsibilities
  1. Convert GA chromosome (row, col) → (lat, lon) for each station
  2. Optionally snap each station to the nearest real road node in the
     OSMnx graph (improves realism; gracefully skipped if graph is None)
  3. Return a list of dicts ready to be consumed by map_renderer.py

# genetic_algorithm.py
--------------------
Pure-Python / NumPy genetic algorithm for optimal EV-charging-station placement.

Chromosome representation
  A chromosome is a list of `n_stations` (row, col) integer tuples
  representing grid indices inside the density matrix.

Fitness function
  We want stations to be placed where population density is HIGH and
  where each high-density cell is *covered* by at least one nearby station.

  fitness(chromosome) = Σ_{cell} density[cell] * exp(-d_min(cell) / scale)

  where d_min(cell) is the Euclidean grid distance to the nearest station.
  Higher fitness ← stations closer to dense cells.

# GA operators
  Selection  : tournament selection (size 3)
  Crossover  : uniform crossover on the list of station indices
  Mutation   : random re-position of a randomly chosen station
  Elitism    : top-k individuals always survive unchanged


# main.py
-------
Entry point — orchestrates the full EV charging station optimisation pipeline.

Pipeline
  1. Ask user for number of stations
  2. Download Surat road network via osmnx (defines bounding box)
  3. Generate synthetic population density
  4. Run Genetic Algorithm
  5. Convert GA result to lat/lon stations (road-snapped)
  6. Build folium map with density overlay + markers
  7. Save HTML and open in browser



# map_renderer.py
---------------
All folium / matplotlib map-rendering logic lives here.

Key design decisions
  • Population density is rendered as a smooth continuous PNG overlay
    (NOT folium HeatMap which creates ugly circular blobs).
  • The PNG is generated with matplotlib using a perceptually uniform
    colormap, alpha-composited so the base OSM tiles show through.
  • EV station markers use a ⚡ emoji DivIcon so they stand out clearly.


# population_density.py
---------------------
Generates a realistic synthetic population density surface for the city.

Strategy:
- Scatter multiple random hotspots across the bounding box
- Each hotspot has its own random position, amplitude, and spread
- Combine all hotspots into a single density surface using 2D Gaussians
- Apply Gaussian smoothing to blend everything naturally
- Add low-level structured noise for texture/realism
- Result: no dominant centre, no circular blobs, looks like real urban fabric
