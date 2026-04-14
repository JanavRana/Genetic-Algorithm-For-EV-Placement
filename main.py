"""
main.py
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
"""

import sys
import webbrowser
from pathlib import Path
import time

# ── project modules ────────────────────────────────────────────────────────
from population_density import generate_density_matrix
from genetic_algorithm   import GAConfig, run_genetic_algorithm
from ga_integration      import chromosome_to_stations, random_stations
from map_renderer        import build_map


# ── constants ──────────────────────────────────────────────────────────────
CITY            = "Surat, India"
NETWORK_DIST_M  = 7_000          # radius in metres (keeps download fast)
GRID_SIZE       = 200            # density grid resolution
OUTPUT_FILE     = "ev_charging_map.html"


# ── helpers ────────────────────────────────────────────────────────────────

def get_n_stations() -> int:
    """Prompt user for the desired number of EV charging stations."""
    while True:
        raw = input("\n⚡  How many EV charging stations do you want? (e.g. 10): ").strip()
        try:
            n = int(raw)
            if 1 <= n <= 100:
                return n
            print("   Please enter a number between 1 and 100.")
        except ValueError:
            print("   Invalid input — please enter a whole number.")


def download_city_graph(city: str, dist: int):
    """
    Download the drivable road network for the city within `dist` metres.
    Returns (graph, bbox) where bbox = (south, north, west, east).
    """
    import osmnx as ox
    print(f"\n📡  Downloading road network for '{city}' (radius {dist/1000:.1f} km)…")
    t0 = time.time()
    G = ox.graph_from_place(city, network_type="drive", dist=dist)
    elapsed = time.time() - t0
    print(f"    ✓ Downloaded in {elapsed:.1f}s  "
          f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")

    # Derive bounding box from graph node coordinates
    lats = [d["y"] for _, d in G.nodes(data=True)]
    lons = [d["x"] for _, d in G.nodes(data=True)]
    bbox = (min(lats), max(lats), min(lons), max(lons))
    print(f"    Bounding box: S={bbox[0]:.4f}  N={bbox[1]:.4f}  "
          f"W={bbox[2]:.4f}  E={bbox[3]:.4f}")
    return G, bbox


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  ⚡  EV Charging Station Optimiser  —  Genetic Algorithm")
    print("=" * 60)

    # ── 1. User input ──────────────────────────────────────────────────────
    n_stations = get_n_stations()
    print(f"\n  Will optimise placement of {n_stations} EV charging station(s).")

    # ── 2. Download city graph ─────────────────────────────────────────────
    try:
        graph, bbox = download_city_graph(CITY, NETWORK_DIST_M)
    except Exception as e:
        print(f"\n⚠️  Could not download graph: {e}")
        print("   Falling back to a hard-coded bounding box for Surat.")
        graph = None
        # Approximate bbox for central Surat
        bbox = (21.184, 21.244, 72.788, 72.848)

    # ── 3. Population density ──────────────────────────────────────────────
    print("\n🌆  Generating synthetic population density…", end=" ", flush=True)
    density = generate_density_matrix(
        bbox=bbox,
        grid_size=GRID_SIZE,
        n_hotspots=20,
        seed=None,          # fully random each run  ← BONUS: different each time
    )
    print("done ✓")

    # ── 4. Genetic Algorithm ───────────────────────────────────────────────
    config = GAConfig(
        n_stations      = n_stations,
        population_size = 50,
        n_generations   = 100,
        mutation_rate   = 0.15,
        elite_fraction  = 0.10,
        coverage_scale  = 15.0,
        seed            = None,    # random each run
    )

    print(f"\n🧬  Running Genetic Algorithm "
          f"(pop={config.population_size}, gen={config.n_generations})…\n")
    t0     = time.time()
    result = run_genetic_algorithm(density, config, verbose=True)
    elapsed = time.time() - t0

    print(f"\n  ✓ GA finished in {elapsed:.1f}s  |  "
          f"Best fitness: {result.best_fitness:.4f}")

    # ── 5. Convert chromosome to station coordinates ───────────────────────
    print("\n📍  Mapping stations to coordinates…", end=" ", flush=True)
    ga_stations = chromosome_to_stations(
        chromosome=result.best_chromosome,
        bbox=bbox,
        grid_size=GRID_SIZE,
        graph=graph,
    )

    # Baseline: same number of random stations for visual comparison
    rand_st = random_stations(
        n_stations=n_stations,
        bbox=bbox,
        grid_size=GRID_SIZE,
        seed=7,
        graph=graph,
    )
    print("done ✓")

    # Print a summary table
    print("\n  GA-Optimised Station Locations:")
    print(f"  {'#':<4} {'Latitude':>10}  {'Longitude':>11}")
    print("  " + "-" * 28)
    for st in ga_stations:
        print(f"  {st['id']:<4} {st['lat']:>10.5f}  {st['lon']:>11.5f}")

    # ── 6. Build map ───────────────────────────────────────────────────────
    print("\n🗺️  Building interactive map…", end=" ", flush=True)
    m = build_map(
        bbox=bbox,
        density=density,
        ga_stations=ga_stations,
        rand_stations=rand_st,
    )
    print("done ✓")

    # ── 7. Save & open ─────────────────────────────────────────────────────
    output_path = Path(OUTPUT_FILE).resolve()
    m.save(str(output_path))
    print(f"\n💾  Map saved → {output_path}")

    print("🌐  Opening in browser…")
    webbrowser.open(output_path.as_uri())

    print("\n" + "=" * 60)
    print("  ✅  Done!  Enjoy your optimised EV charging map.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
