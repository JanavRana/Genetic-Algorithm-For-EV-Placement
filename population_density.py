"""
population_density.py
---------------------
Generates smooth, realistic synthetic population density surfaces
for a given geographic bounding box. Uses multi-center Gaussian
mixture models + scipy smoothing to eliminate circular artifacts.

Author: Generated for EV Charging Station Optimization Project
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PopulationCenter:
    """
    Represents one urban density hotspot.

    Attributes
    ----------
    lat, lon   : Geographic coordinates of the center
    weight     : Relative population intensity (0–1 scale, will be normalized)
    spread_lat : Std-dev spread in the latitude direction (degrees)
    spread_lon : Std-dev spread in the longitude direction (degrees)
    """
    lat: float
    lon: float
    weight: float = 1.0
    spread_lat: float = 0.015   # ~1.5 km at India latitudes
    spread_lon: float = 0.018


@dataclass
class DensityConfig:
    """
    Hyper-parameters controlling grid resolution and smoothing.

    Attributes
    ----------
    grid_resolution : Number of grid cells along each axis (higher → sharper)
    sigma_smooth    : Gaussian smoothing radius (in grid cells).
                      Larger values remove blob artifacts at the cost of detail.
    noise_scale     : Fraction of max density added as structured noise (realism)
    noise_seed      : Random seed for reproducibility
    """
    grid_resolution: int = 400
    sigma_smooth: float = 18.0
    noise_scale: float = 0.08
    noise_seed: int = 42


# ---------------------------------------------------------------------------
# Core density generator
# ---------------------------------------------------------------------------

class PopulationDensityGenerator:
    """
    Builds a smooth 2-D population density surface from a list of
    PopulationCenter objects over a geographic bounding box.

    Usage
    -----
    >>> gen = PopulationDensityGenerator(bbox, centers, config)
    >>> density = gen.generate()          # np.ndarray, shape (H, W), values 0–1
    >>> rgba_img = gen.to_rgba(density)   # np.ndarray uint8 for ImageOverlay
    """

    def __init__(
        self,
        bbox: Tuple[float, float, float, float],
        centers: List[PopulationCenter],
        config: Optional[DensityConfig] = None,
    ):
        """
        Parameters
        ----------
        bbox    : (south, west, north, east) in decimal degrees
        centers : List of PopulationCenter objects
        config  : DensityConfig (uses defaults if None)
        """
        self.south, self.west, self.north, self.east = bbox
        self.centers = centers
        self.cfg = config or DensityConfig()

        # Build coordinate grids
        self.lats = np.linspace(self.south, self.north, self.cfg.grid_resolution)
        self.lons = np.linspace(self.west,  self.east,  self.cfg.grid_resolution)
        self.LON, self.LAT = np.meshgrid(self.lons, self.lats)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> np.ndarray:
        """
        Returns a 2-D float array (values 0–1) representing normalized
        population density. Rows = latitude (south→north), cols = longitude.
        """
        surface = self._gaussian_mixture()
        surface = self._add_structured_noise(surface)
        surface = self._smooth(surface)
        surface = self._normalize(surface)
        return surface

    def to_rgba(
        self,
        density: np.ndarray,
        colormap_name: str = "YlOrRd",
        alpha_max: float = 0.75,
        threshold: float = 0.05,
    ) -> np.ndarray:
        """
        Convert a normalized density array → RGBA uint8 image suitable
        for folium.ImageOverlay.

        Parameters
        ----------
        density        : Output of generate()
        colormap_name  : Any matplotlib colormap name
        alpha_max      : Maximum opacity (0–1). Keeps map visible beneath.
        threshold      : Density below this value becomes fully transparent.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        cmap = plt.get_cmap(colormap_name)
        norm = Normalize(vmin=0, vmax=1)

        # Apply colormap → (H, W, 4) float32 in [0,1]
        rgba = cmap(norm(density)).astype(np.float32)

        # Build alpha channel: proportional to density, hard cutoff at threshold
        alpha = np.where(density < threshold, 0.0, density * alpha_max)
        # Soft edges: blend the alpha so there's no sharp boundary ring
        alpha = gaussian_filter(alpha, sigma=self.cfg.sigma_smooth * 0.5)
        rgba[:, :, 3] = np.clip(alpha, 0, alpha_max)

        # Flip vertically: ImageOverlay expects north at row-0
        rgba = np.flipud(rgba)

        return (rgba * 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _gaussian_mixture(self) -> np.ndarray:
        """Sum weighted, anisotropic Gaussians for each population center."""
        surface = np.zeros_like(self.LAT)
        total_weight = sum(c.weight for c in self.centers)

        for center in self.centers:
            w = center.weight / total_weight
            dlat = (self.LAT - center.lat) / center.spread_lat
            dlon = (self.LON - center.lon) / center.spread_lon
            # Anisotropic Gaussian (different sigma per axis → no perfect circles)
            surface += w * np.exp(-0.5 * (dlat**2 + dlon**2))

        return surface

    def _add_structured_noise(self, surface: np.ndarray) -> np.ndarray:
        """
        Add low-frequency Perlin-like noise to break up uniform gradients.
        Uses multiple octaves of smoothed random noise for realism.
        """
        rng = np.random.default_rng(self.cfg.noise_seed)
        noise = np.zeros_like(surface)
        n = self.cfg.grid_resolution

        # Three octaves: coarse, medium, fine
        for sigma, amplitude in [(n * 0.15, 1.0), (n * 0.06, 0.4), (n * 0.02, 0.15)]:
            raw = rng.standard_normal((n, n))
            octave = gaussian_filter(raw, sigma=sigma)
            octave = (octave - octave.min()) / (octave.max() - octave.min())
            noise += amplitude * octave

        noise = (noise - noise.min()) / (noise.max() - noise.min())
        return surface + self.cfg.noise_scale * noise * surface  # modulate by density

    def _smooth(self, surface: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian smoothing to eliminate circular blob artifacts.
        This is the key step: a single large sigma blurs away the
        per-center rings while preserving the overall gradient shape.
        """
        return gaussian_filter(surface, sigma=self.cfg.sigma_smooth)

    @staticmethod
    def _normalize(surface: np.ndarray) -> np.ndarray:
        mn, mx = surface.min(), surface.max()
        if mx == mn:
            return np.zeros_like(surface)
        return (surface - mn) / (mx - mn)
