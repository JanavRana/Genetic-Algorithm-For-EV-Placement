"""
map_renderer.py
---------------
All folium / matplotlib map-rendering logic lives here.

Key design decisions
  • Population density is rendered as a smooth continuous PNG overlay
    (NOT folium HeatMap which creates ugly circular blobs).
  • The PNG is generated with matplotlib using a perceptually uniform
    colormap, alpha-composited so the base OSM tiles show through.
  • EV station markers use a ⚡ emoji DivIcon so they stand out clearly.
"""

from __future__ import annotations

import io
import base64
from typing import List, Dict, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")                   # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import folium
from folium.raster_layers import ImageOverlay


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Colourmap for the density overlay
_DENSITY_CMAP = "YlOrRd"          # yellow → orange → red (population heat)
_OVERLAY_OPACITY = 0.55            # transparency of the density layer
_MAP_ZOOM = 12


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def density_to_png_base64(density: np.ndarray) -> str:
    """
    Convert a (H, W) float density matrix to a transparent PNG encoded as
    a base64 string suitable for embedding in a folium ImageOverlay.

    Steps
    -----
    1. Map density values → RGBA using the chosen colourmap
    2. Set alpha channel proportional to density (transparent where unpopulated)
    3. Save to an in-memory BytesIO buffer as PNG
    4. Encode as base64
    """
    # --- normalise just in case ---
    d = density.astype(np.float64)
    d = (d - d.min()) / (d.max() - d.min() + 1e-9)

    # --- apply colourmap to get RGBA (values in [0,1]) ---
    cmap = plt.get_cmap(_DENSITY_CMAP)
    rgba = cmap(d)                       # shape (H, W, 4)

    # --- custom alpha: transparent at zero, opaque at 1 ---
    # Use a non-linear alpha so mid-density areas are still visible
    rgba[..., 3] = np.clip(d ** 0.6 * _OVERLAY_OPACITY, 0.0, _OVERLAY_OPACITY)

    # --- save to PNG in memory ---
    fig, ax = plt.subplots(figsize=(density.shape[1] / 100, density.shape[0] / 100), dpi=100)
    ax.imshow(rgba, origin="upper", aspect="auto", interpolation="bilinear")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)

    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def build_map(
    bbox:          tuple[float, float, float, float],
    density:       np.ndarray,
    ga_stations:   List[Dict[str, Any]],
    rand_stations: List[Dict[str, Any]] | None = None,
) -> folium.Map:
    """
    Build and return the complete folium Map object.

    Parameters
    ----------
    bbox          : (south, north, west, east)
    density       : 2-D population density array
    ga_stations   : list of station dicts from ga_integration
    rand_stations : optional baseline random stations to compare

    Returns
    -------
    folium.Map ready to be saved as HTML
    """
    south, north, west, east = bbox
    centre_lat = (south + north) / 2
    centre_lon = (west  + east)  / 2

    # ------------------------------------------------------------------ #
    # 1. Base map
    # ------------------------------------------------------------------ #
    m = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=_MAP_ZOOM,
        tiles="CartoDB positron",
        prefer_canvas=True,
    )

    # ------------------------------------------------------------------ #
    # 2. Population density overlay
    # ------------------------------------------------------------------ #
    png_b64 = density_to_png_base64(density)

    ImageOverlay(
        image=png_b64,
        bounds=[[south, west], [north, east]],
        opacity=1.0,          # alpha already baked into the PNG
        name="Population Density",
        interactive=False,
        cross_origin=False,
        zindex=1,
    ).add_to(m)

    # ------------------------------------------------------------------ #
    # 3. Optional: random-placement baseline (grey markers)
    # ------------------------------------------------------------------ #
    if rand_stations:
        rand_group = folium.FeatureGroup(name="Random Placement (baseline)", show=True)
        for st in rand_stations:
            _add_station_marker(
                group=rand_group,
                lat=st["lat"],
                lon=st["lon"],
                label=f"Random {st['id']}",
                emoji="📍",
                color="#888888",
            )
        rand_group.add_to(m)

    # ------------------------------------------------------------------ #
    # 4. GA-optimised stations (⚡ markers)
    # ------------------------------------------------------------------ #
    ga_group = folium.FeatureGroup(name="GA-Optimised EV Stations", show=True)
    for st in ga_stations:
        _add_station_marker(
            group=ga_group,
            lat=st["lat"],
            lon=st["lon"],
            label=st["label"],
            emoji="⚡",
            color="#00cc44",
        )
    ga_group.add_to(m)

    # ------------------------------------------------------------------ #
    # 5. Layer control + title
    # ------------------------------------------------------------------ #
    folium.LayerControl(collapsed=False).add_to(m)
    _add_title(m, n_stations=len(ga_stations))
    _add_legend(m)

    return m


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_station_marker(
    group:  folium.FeatureGroup,
    lat:    float,
    lon:    float,
    label:  str,
    emoji:  str,
    color:  str,
) -> None:
    """Add a styled DivIcon marker to a FeatureGroup."""
    icon_html = f"""
    <div style="
        font-size: 22px;
        line-height: 1;
        text-align: center;
        text-shadow: 0 0 6px {color}, 0 0 12px {color};
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.5));
    ">{emoji}</div>
    """
    folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(
            html=icon_html,
            icon_size=(28, 28),
            icon_anchor=(14, 14),
        ),
        popup=folium.Popup(
            f"""
            <div style="font-family:sans-serif; font-size:13px; padding:4px;">
              <b style="color:{color};">{emoji} {label}</b><br>
              <span style="color:#555;">Lat: {lat:.5f}</span><br>
              <span style="color:#555;">Lon: {lon:.5f}</span>
            </div>
            """,
            max_width=200,
        ),
        tooltip=label,
    ).add_to(group)


def _add_title(m: folium.Map, n_stations: int) -> None:
    """Inject a floating title card into the map HTML."""
    title_html = f"""
    <div style="
        position: fixed;
        top: 16px; left: 50%; transform: translateX(-50%);
        z-index: 1000;
        background: rgba(10,10,20,0.88);
        color: #00ffaa;
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 15px;
        font-weight: 700;
        padding: 10px 22px;
        border-radius: 30px;
        border: 1.5px solid #00cc66;
        box-shadow: 0 4px 20px rgba(0,200,100,0.25);
        letter-spacing: 0.5px;
        pointer-events: none;
        white-space: nowrap;
    ">
      ⚡ Optimal EV Charging Station Placement — {n_stations} Stations (Surat, India)
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))


def _add_legend(m: folium.Map) -> None:
    """Add a small legend explaining the map layers."""
    legend_html = """
    <div style="
        position: fixed;
        bottom: 40px; right: 16px;
        z-index: 1000;
        background: rgba(10,10,20,0.88);
        color: #eee;
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 12px;
        padding: 10px 14px;
        border-radius: 10px;
        border: 1px solid #444;
        line-height: 1.9;
        pointer-events: none;
    ">
      <b style="color:#00ffaa;">Map Legend</b><br>
      🟡→🔴&nbsp; Population density (low → high)<br>
      ⚡&nbsp; GA-optimised EV station<br>
      📍&nbsp; Random baseline station
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
