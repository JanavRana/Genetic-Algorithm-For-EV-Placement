"""
ga_integration.py
-----------------
Adapter layer: converts your existing Genetic Algorithm output
into the format expected by EVChargingMapRenderer.

This module is intentionally thin — it just bridges your GA's
coordinate system (grid indices OR lat/lon) to the renderer.
"""

import numpy as np
from typing import List, Tuple, Union, Optional


class GAOutputAdapter:
    """
    Converts GA chromosome output → (lat, lon) pairs for the map renderer.

    Supports two input modes
    -----------------------
    A) Grid-index mode : GA returns (row, col) grid indices.
       You provide the bounding box and grid shape, and the adapter
       maps indices to geographic coordinates via linear interpolation.

    B) Coordinate mode : GA returns (lat, lon) directly (no conversion needed).
    """

    def __init__(
        self,
        bbox: Tuple[float, float, float, float],
        grid_shape: Optional[Tuple[int, int]] = None,
    ):
        """
        Parameters
        ----------
        bbox       : (south, west, north, east) — same bbox used in your osmnx load
        grid_shape : (n_rows, n_cols) of your GA grid. Required only for grid-index mode.
        """
        self.south, self.west, self.north, self.east = bbox
        self.grid_shape = grid_shape

    def from_grid_indices(
        self,
        indices: List[Tuple[int, int]],
        fitness_scores: Optional[List[float]] = None,
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Convert (row, col) grid indices → (lat, lon) pairs.

        Parameters
        ----------
        indices       : List of (row, col) tuples from GA chromosome
        fitness_scores: Optional per-station fitness values

        Returns
        -------
        stations : List of (lat, lon) tuples
        scores   : Normalised fitness scores (0–1), or uniform if not provided
        """
        if self.grid_shape is None:
            raise ValueError("grid_shape must be set for grid-index mode.")

        n_rows, n_cols = self.grid_shape
        stations = []

        for row, col in indices:
            lat = self.south + (row / (n_rows - 1)) * (self.north - self.south)
            lon = self.west  + (col / (n_cols - 1)) * (self.east  - self.west)
            stations.append((lat, lon))

        scores = self._normalize_scores(fitness_scores, len(stations))
        return stations, scores

    def from_latlon(
        self,
        latlon_pairs: List[Tuple[float, float]],
        fitness_scores: Optional[List[float]] = None,
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Pass-through for GA output already in lat/lon.

        Parameters
        ----------
        latlon_pairs  : List of (lat, lon) tuples
        fitness_scores: Optional fitness values

        Returns
        -------
        stations, scores
        """
        scores = self._normalize_scores(fitness_scores, len(latlon_pairs))
        return list(latlon_pairs), scores

    def top_k(
        self,
        stations: List[Tuple[float, float]],
        scores: List[float],
        k: int,
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """Return the top-k stations by fitness score."""
        paired = sorted(zip(scores, stations), reverse=True)[:k]
        top_scores, top_stations = zip(*paired)
        return list(top_stations), list(top_scores)

    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_scores(scores: Optional[List[float]], n: int) -> List[float]:
        if scores is None:
            return [1.0] * n
        arr = np.array(scores, dtype=float)
        rng = arr.max() - arr.min()
        if rng == 0:
            return [1.0] * n
        return ((arr - arr.min()) / rng).tolist()


# ---------------------------------------------------------------------------
# Example: wiring GA output into the map (copy-paste into your GA script)
# ---------------------------------------------------------------------------

def render_ga_result(
    ga_grid_solutions: List[Tuple[int, int]],
    ga_fitness_scores: List[float],
    bbox: Tuple[float, float, float, float],
    grid_shape: Tuple[int, int],
    output_html: str = "surat_ev_map.html",
):
    """
    One-shot convenience function: takes raw GA output, renders full map.

    Parameters
    ----------
    ga_grid_solutions : list of (row, col) chosen by your GA
    ga_fitness_scores : corresponding fitness values
    bbox              : (south, west, north, east)
    grid_shape        : (n_rows, n_cols) of your GA grid
    output_html       : output filename
    """
    from population_density import PopulationDensityGenerator, PopulationCenter, DensityConfig
    from map_renderer import EVChargingMapRenderer
    from main import SURAT_CENTERS, DENSITY_CONFIG  # reuse Surat config

    adapter = GAOutputAdapter(bbox=bbox, grid_shape=grid_shape)
    stations, scores = adapter.from_grid_indices(ga_grid_solutions, ga_fitness_scores)

    generator = PopulationDensityGenerator(
        bbox=bbox, centers=SURAT_CENTERS, config=DENSITY_CONFIG
    )

    renderer = EVChargingMapRenderer(bbox=bbox, city_name="Surat, Gujarat", zoom=13)
    (
        renderer
        .add_density_layer(generator)
        .add_charging_stations(stations, scores=scores)
        .add_title("EV Charging Optimization — GA Result")
        .add_legend()
        .finalize()
        .save(output_html)
    )
