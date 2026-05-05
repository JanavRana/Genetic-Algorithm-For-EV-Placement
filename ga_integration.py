from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Optional

from population_density import density_to_latlon
from genetic_algorithm import Chromosome


# Public API

"""
Convert a GA chromosome to a list of station dicts.

Parameters

chromosome : list of (row, col) tuples
bbox       : (south, north, west, east)
grid_size  : size of the density grid
graph      : osmnx graph (optional); if provided, each station is
snapped to the nearest road intersection

Returns

list of dicts, each with keys:
    id    : int   (1-based)
    lat   : float
    lon   : float
    label : str
"""

def chromosome_to_stations(
    chromosome: Chromosome,
    bbox:       tuple[float, float, float, float],
    grid_size:  int,
    graph=None,           # optional osmnx graph for road-snapping
) -> List[Dict[str, Any]]:

    stations = []

    for idx, (row, col) in enumerate(chromosome, start=1):
        lat, lon = density_to_latlon(row, col, bbox, grid_size)

        # optional road-snapping 
        if graph is not None:
            lat, lon = _snap_to_road(graph, lat, lon)

        stations.append({
            "id":    idx,
            "lat":   lat,
            "lon":   lon,
            "label": f"EV Station {idx}",
        })

    return stations


def random_stations(
    n_stations: int,
    bbox:       tuple[float, float, float, float],
    grid_size:  int,
    seed:       int | None = None,
    graph=None,
) -> List[Dict[str, Any]]:

    # Generate `n_stations` random placements (used as baseline comparison).

    # Same signature as chromosome_to_stations so results are comparable.
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, grid_size, size=n_stations)
    cols = rng.integers(0, grid_size, size=n_stations)
    dummy_chromosome: Chromosome = list(zip(rows.tolist(), cols.tolist()))
    return chromosome_to_stations(dummy_chromosome, bbox, grid_size, graph)


# Internal helpers

def _snap_to_road(graph, lat: float, lon: float) -> tuple[float, float]:
    # Snap a coordinate to the nearest node in the OSMnx graph.
    try:
        import osmnx as ox
        node_id = ox.distance.nearest_nodes(graph, lon, lat)
        node    = graph.nodes[node_id]
        return float(node["y"]), float(node["x"])
    except Exception:
        return lat, lon
