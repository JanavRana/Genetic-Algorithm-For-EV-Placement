"""
main.py
-------
Entry point: configure Surat's population centers, run the density
generator, and render the final folium map with optional GA stations.

Run
---
    python main.py

Output
------
    surat_ev_map.html   — Open in any browser
"""

from population_density import PopulationDensityGenerator, PopulationCenter, DensityConfig
from map_renderer import EVChargingMapRenderer

# ---------------------------------------------------------------------------
# 1. Geographic bounding box for Surat, Gujarat, India
# ---------------------------------------------------------------------------
#    Roughly covers the urban agglomeration + surroundings
SURAT_BBOX = (21.08, 72.74, 21.28, 72.95)   # (south, west, north, east)

# ---------------------------------------------------------------------------
# 2. Population centers (multi-hotspot model)
#    Based on Surat's real urban geography:
#    - Old city core around Rander Rd / Chowk Bazar area
#    - Adajan & Vesu (western high-density residential)
#    - Udhna & Sagrampura (industrial + working class)
#    - Katargam (dense north)
#    - Varachha (dense east)
#    - Dindoli & Althan (south growth corridors)
# ---------------------------------------------------------------------------
SURAT_CENTERS = [
    # (lat,    lon,    weight, spread_lat, spread_lon)  # area name
    PopulationCenter(21.195, 72.831, weight=1.00, spread_lat=0.018, spread_lon=0.020),  # City Core / Chowk
    PopulationCenter(21.180, 72.790, weight=0.85, spread_lat=0.016, spread_lon=0.018),  # Adajan
    PopulationCenter(21.162, 72.812, weight=0.75, spread_lat=0.014, spread_lon=0.015),  # Vesu
    PopulationCenter(21.210, 72.858, weight=0.90, spread_lat=0.017, spread_lon=0.019),  # Varachha
    PopulationCenter(21.228, 72.840, weight=0.70, spread_lat=0.013, spread_lon=0.016),  # Katargam
    PopulationCenter(21.175, 72.855, weight=0.80, spread_lat=0.015, spread_lon=0.017),  # Udhna
    PopulationCenter(21.155, 72.840, weight=0.60, spread_lat=0.012, spread_lon=0.014),  # Dindoli
    PopulationCenter(21.200, 72.810, weight=0.65, spread_lat=0.013, spread_lon=0.015),  # Sagrampura
    PopulationCenter(21.240, 72.870, weight=0.50, spread_lat=0.011, spread_lon=0.013),  # Limbayat
    PopulationCenter(21.135, 72.800, weight=0.45, spread_lat=0.010, spread_lon=0.012),  # Althan
    PopulationCenter(21.190, 72.875, weight=0.55, spread_lat=0.012, spread_lon=0.014),  # Kapodra
]

# ---------------------------------------------------------------------------
# 3. Density generation config
#    sigma_smooth=18 is the key parameter that kills circular artifacts
# ---------------------------------------------------------------------------
DENSITY_CONFIG = DensityConfig(
    grid_resolution=500,    # Higher = sharper PNG, more memory
    sigma_smooth=18.0,      # Main smoothing — increase to blend more
    noise_scale=0.09,       # Organic texture intensity
    noise_seed=7,
)

# ---------------------------------------------------------------------------
# 4. (Optional) Plug in your GA output here
#    Replace with actual output from your Genetic Algorithm
# ---------------------------------------------------------------------------
EXAMPLE_GA_STATIONS = [
    # (lat,    lon)     — locations selected by GA
    (21.195, 72.831),   # City core
    (21.180, 72.790),   # Adajan junction
    (21.210, 72.858),   # Varachha road
    (21.228, 72.840),   # Katargam
    (21.162, 72.812),   # Vesu
    (21.175, 72.855),   # Udhna
    (21.240, 72.870),   # Limbayat
]

EXAMPLE_GA_SCORES = [0.94, 0.88, 0.91, 0.76, 0.72, 0.81, 0.68]

# ---------------------------------------------------------------------------
# 5. Build and save the map
# ---------------------------------------------------------------------------

def main(output_path: str = "surat_ev_map.html"):
    print("[1/4] Initialising population density generator …")
    generator = PopulationDensityGenerator(
        bbox=SURAT_BBOX,
        centers=SURAT_CENTERS,
        config=DENSITY_CONFIG,
    )

    print("[2/4] Setting up folium map renderer …")
    renderer = EVChargingMapRenderer(
        bbox=SURAT_BBOX,
        city_name="Surat, Gujarat",
        zoom=13,
    )

    print("[3/4] Generating density surface + encoding PNG overlay …")
    (
        renderer
        .add_density_layer(generator, colormap="custom", alpha_max=0.72)
        .add_charging_stations(EXAMPLE_GA_STATIONS, scores=EXAMPLE_GA_SCORES)
        .add_title()
        .add_legend()
        .finalize()
    )

    print("[4/4] Saving …")
    renderer.save(output_path)
    print(f"\n✅  Done! Open '{output_path}' in your browser.\n")


if __name__ == "__main__":
    main()
