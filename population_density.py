import numpy as np
from scipy.ndimage import gaussian_filter

"""
Generate a 2-D population-density grid.

Parameters

bbox        : (south, north, west, east) in degrees
grid_size   : number of cells along each axis
n_hotspots  : number of urban density clusters to place
seed        : random seed for reproducibility (None = fully random)

Returns

density : (grid_size, grid_size) float array, values in [0, 1]
"""
def generate_density_matrix(
    bbox: tuple[float, float, float, float],
    grid_size: int = 200,
    n_hotspots: int = 18,
    seed: int | None = None,
) -> np.ndarray:

    rng = np.random.default_rng(seed)

    south, north, west, east = bbox
    density = np.zeros((grid_size, grid_size), dtype=np.float64)

    #  coordinate grids (fractional position 0‥1) 
    yi = np.linspace(0.0, 1.0, grid_size)   # latitude axis (rows)
    xi = np.linspace(0.0, 1.0, grid_size)   # longitude axis (cols)
    yy, xx = np.meshgrid(yi, xi, indexing="ij")  # shape (grid_size, grid_size)

    #  scatter hotspots across the whole bounding box 
    for _ in range(n_hotspots):
        # random centre – uniform across the grid (no centre bias)
        cy = rng.uniform(0.05, 0.95)
        cx = rng.uniform(0.05, 0.95)

        # random spread – mix of tight (commercial) and wide (residential) zones
        sy = rng.uniform(0.04, 0.18)
        sx = rng.uniform(0.04, 0.18)

        # random peak amplitude
        amplitude = rng.uniform(2.0, 5.0)

        # anisotropic Gaussian: different spread in each direction
        blob = amplitude * np.exp(
            -(((yy - cy) ** 2) / (2 * sy ** 2) + ((xx - cx) ** 2) / (2 * sx ** 2))
        )
        density += blob

    #  low-level structured noise (perlin-like via filtered random) 
    noise_raw = rng.random((grid_size, grid_size))
    noise_smooth = gaussian_filter(noise_raw, sigma=grid_size * 0.04)
    noise_smooth /= noise_smooth.max() + 1e-9
    density += 0.15 * noise_smooth  # subtle texture layer

    # final smoothing to remove any remaining hard edges 
    density = gaussian_filter(density, sigma=grid_size * 0.025)

    #  normalise to [0, 1] 
    density -= density.min()
    density /= density.max() + 1e-9

    return density

"""
Convert a (row, col) grid index back to (latitude, longitude).

Row 0 → south edge, row grid_size-1 → north edge
Col 0 → west  edge, col grid_size-1 → east  edge
"""
def density_to_latlon(
    row: int,
    col: int,
    bbox: tuple[float, float, float, float],
    grid_size: int,
) -> tuple[float, float]:

    south, north, west, east = bbox
    lat = south + (row / (grid_size - 1)) * (north - south)
    lon = west  + (col / (grid_size - 1)) * (east  - west)
    return lat, lon
