"""
map_renderer.py
---------------
Renders the population density surface as a smooth ImageOverlay
on a folium map. Also handles GA charging station markers.

Usage
-----
    from map_renderer import EVChargingMapRenderer
    renderer = EVChargingMapRenderer(bbox, city_name="Surat, India")
    renderer.add_density_layer(density_generator)
    renderer.add_charging_stations(stations)   # Optional: GA output
    renderer.save("output_map.html")
"""

import io
import base64
import numpy as np
import folium
from folium import plugins
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from population_density import PopulationDensityGenerator, DensityConfig


# ---------------------------------------------------------------------------
# Custom colormaps (more realistic than stock YlOrRd)
# ---------------------------------------------------------------------------

def make_urban_colormap():
    """
    A custom colormap that mimics real urban density maps:
    low density → soft blue-green → yellow → orange-red → deep red.
    """
    colors = [
        (0.05, 0.25, 0.50, 0),    # deep blue, fully transparent
        (0.13, 0.50, 0.70, 0.3),  # teal
        (0.56, 0.82, 0.31, 0.5),  # lime-green
        (1.00, 0.90, 0.20, 0.65), # yellow
        (1.00, 0.55, 0.10, 0.75), # orange
        (0.85, 0.10, 0.10, 0.80), # red
        (0.45, 0.00, 0.20, 0.85), # deep maroon
    ]
    positions = [0, 0.15, 0.35, 0.55, 0.72, 0.88, 1.0]
    cmap = LinearSegmentedColormap.from_list(
        "urban_density",
        list(zip(positions, [(r, g, b) for r, g, b, _ in colors])),
        N=512,
    )
    return cmap, [a for _, _, _, a in colors], positions


# ---------------------------------------------------------------------------
# Main renderer class
# ---------------------------------------------------------------------------

class EVChargingMapRenderer:
    """
    Orchestrates the full map: basemap → density overlay → GA stations.

    Parameters
    ----------
    bbox      : (south, west, north, east) bounding box
    city_name : String label for the map title
    zoom      : Initial folium zoom level
    """

    def __init__(
        self,
        bbox: Tuple[float, float, float, float],
        city_name: str = "City",
        zoom: int = 12,
    ):
        self.south, self.west, self.north, self.east = bbox
        self.city_name = city_name
        self.center_lat = (self.south + self.north) / 2
        self.center_lon = (self.west  + self.east)  / 2

        self.map = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=zoom,
            tiles="CartoDB positron",   # clean light basemap
            control_scale=True,
        )
        self._add_layer_control_placeholder()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_density_layer(
        self,
        generator: PopulationDensityGenerator,
        colormap: str = "custom",
        alpha_max: float = 0.72,
        layer_name: str = "Population Density",
    ) -> "EVChargingMapRenderer":
        """
        Generate the density surface and add it as an ImageOverlay.

        Parameters
        ----------
        generator  : Configured PopulationDensityGenerator instance
        colormap   : "custom" uses the urban colormap; any matplotlib name works
        alpha_max  : Max overlay opacity
        layer_name : Name shown in the LayerControl
        """
        density = generator.generate()

        if colormap == "custom":
            rgba = self._apply_custom_colormap(density, alpha_max)
        else:
            rgba = generator.to_rgba(density, colormap_name=colormap, alpha_max=alpha_max)

        # Encode as PNG in memory (no temp file needed)
        png_b64 = self._array_to_b64_png(rgba)

        bounds = [[self.south, self.west], [self.north, self.east]]
        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{png_b64}",
            bounds=bounds,
            opacity=1.0,          # alpha baked into RGBA; keep folium opacity at 1
            name=layer_name,
            zindex=1,
            interactive=False,
        ).add_to(self.map)

        # Store for legend
        self._density_added = True
        return self

    def add_charging_stations(
        self,
        stations: List[Tuple[float, float]],
        scores: Optional[List[float]] = None,
        layer_name: str = "EV Charging Stations (GA)",
    ) -> "EVChargingMapRenderer":
        """
        Plot GA-selected charging station locations.

        Parameters
        ----------
        stations : List of (lat, lon) tuples from the GA output
        scores   : Optional fitness scores per station (used to size markers)
        layer_name : LayerControl label
        """
        fg = folium.FeatureGroup(name=layer_name, show=True)

        if scores is None:
            scores = [1.0] * len(stations)

        max_score = max(scores) if scores else 1.0

        for i, ((lat, lon), score) in enumerate(zip(stations, scores)):
            normalized = score / max_score if max_score else 1.0
            radius = 8 + 10 * normalized

            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                color="#00FF88",
                fill=True,
                fill_color="#00DD77",
                fill_opacity=0.85,
                weight=2.5,
                popup=folium.Popup(
                    f"<b>Station #{i+1}</b><br>"
                    f"Lat: {lat:.5f}<br>Lon: {lon:.5f}<br>"
                    f"Score: {score:.3f}",
                    max_width=200,
                ),
                tooltip=f"⚡ Station #{i+1}",
            ).add_to(fg)

            # Inner dot for visibility
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color="#FFFFFF",
                fill=True,
                fill_color="#FFFFFF",
                fill_opacity=1.0,
                weight=0,
            ).add_to(fg)

        fg.add_to(self.map)
        return self

    def add_title(self, title: Optional[str] = None) -> "EVChargingMapRenderer":
        """Inject a floating title card into the map HTML."""
        title = title or f"EV Charging Optimization — {self.city_name}"
        html = f"""
        <div style="
            position: fixed;
            top: 14px; left: 50%; transform: translateX(-50%);
            z-index: 9999;
            background: rgba(15,15,30,0.88);
            color: #F0F4FF;
            padding: 10px 22px;
            border-radius: 8px;
            font-family: 'Segoe UI', sans-serif;
            font-size: 15px;
            font-weight: 600;
            letter-spacing: 0.5px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            border: 1px solid rgba(100,180,255,0.25);
            pointer-events: none;
        ">
            ⚡ {title}
        </div>
        """
        self.map.get_root().html.add_child(folium.Element(html))
        return self

    def add_legend(self) -> "EVChargingMapRenderer":
        """Add a density legend + station key."""
        html = """
        <div style="
            position: fixed; bottom: 40px; right: 14px; z-index: 9998;
            background: rgba(15,15,30,0.88); color: #E8EDF5;
            padding: 14px 16px; border-radius: 8px; font-size: 12px;
            font-family: 'Segoe UI', sans-serif;
            box-shadow: 0 4px 16px rgba(0,0,0,0.5);
            border: 1px solid rgba(100,180,255,0.2);
            min-width: 155px;
        ">
            <div style="font-weight:700; font-size:13px; margin-bottom:8px;
                        border-bottom:1px solid rgba(255,255,255,0.15); padding-bottom:6px;">
                Legend
            </div>
            <div style="font-weight:600; margin-bottom:6px;">Population Density</div>
            <div style="
                width: 130px; height: 14px; border-radius: 3px;
                background: linear-gradient(to right,
                    rgba(34,128,178,0.4),
                    rgba(143,209,79,0.6),
                    rgba(255,230,51,0.75),
                    rgba(255,140,26,0.8),
                    rgba(216,26,26,0.85),
                    rgba(115,0,51,0.9));
                margin-bottom: 3px;
            "></div>
            <div style="display:flex; justify-content:space-between;
                        font-size:10px; color:#AAB4C8; margin-bottom:10px;">
                <span>Low</span><span>High</span>
            </div>
            <div style="font-weight:600; margin-bottom:5px;">GA Stations</div>
            <div style="display:flex; align-items:center; gap:7px;">
                <div style="width:14px; height:14px; border-radius:50%;
                            background:#00DD77; border:2px solid #00FF88;"></div>
                <span>Selected location</span>
            </div>
        </div>
        """
        self.map.get_root().html.add_child(folium.Element(html))
        return self

    def finalize(self) -> "EVChargingMapRenderer":
        """Add LayerControl. Call after all layers are added."""
        folium.LayerControl(position="topright", collapsed=False).add_to(self.map)
        return self

    def save(self, path: str) -> str:
        """Save the map to an HTML file and return the path."""
        self.map.save(path)
        print(f"[✓] Map saved → {path}")
        return path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_custom_colormap(
        self, density: np.ndarray, alpha_max: float
    ) -> np.ndarray:
        """Use the urban density colormap with blended alpha."""
        from scipy.ndimage import gaussian_filter

        cmap, _, _ = make_urban_colormap()
        rgba = cmap(density).astype(np.float32)           # (H, W, 4) in [0,1]

        # Custom alpha: sigmoid-shaped, smoothed to avoid rings
        alpha_raw = np.where(density < 0.04, 0.0, (density ** 0.6) * alpha_max)
        alpha_smooth = gaussian_filter(alpha_raw, sigma=10)
        rgba[:, :, 3] = np.clip(alpha_smooth, 0, alpha_max)

        rgba = np.flipud(rgba)                             # north at top
        return (rgba * 255).astype(np.uint8)

    @staticmethod
    def _array_to_b64_png(rgba: np.ndarray) -> str:
        """Encode RGBA uint8 array as base64 PNG string."""
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=False)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _add_layer_control_placeholder(self):
        """Dummy FeatureGroup so LayerControl always has ≥1 entry."""
        folium.FeatureGroup(name="Basemap", show=True).add_to(self.map)
