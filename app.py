import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from flask_caching import Cache

import os
import argparse
import configparser
import numpy as np
import pandas as pd
from wcs_height_map import WCSHeightMap
from wms_image import WMSImage
import pyproj
from fresnel import fresnel_zone_radius
import tempfile

from layout import (
    build_layout,
    placeholder_figure,
    PLOT_HEIGHT_2D,
    PLOT_HEIGHT_3D,
    PLOT_MARGIN,
)


class DistanceExceededError(Exception):
    pass


MAX_DISTANCE = 50000

config = configparser.ConfigParser(
    defaults=os.environ, interpolation=configparser.ExtendedInterpolation()
)
config.read("config.ini")

theme_url = getattr(dbc.themes, config["dashboard"]["theme"].upper())
app = dash.Dash(
    __name__,
    external_stylesheets=[theme_url, dbc.icons.FONT_AWESOME],
    prevent_initial_callbacks=True,
)
app.title = "LoRaWAN line of sight helper"

WMS_CACHE_DIR = tempfile.TemporaryDirectory(prefix="wms_cache_")
WCS_CACHE_DIR = tempfile.TemporaryDirectory(prefix="wcs_cache_")
FLASK_CACHE_DIR = tempfile.TemporaryDirectory(prefix="flaskcache_")

cache = Cache(
    app.server,
    config={"CACHE_TYPE": "filesystem", "CACHE_DIR": FLASK_CACHE_DIR.name},
)

stations = pd.read_csv(config["dashboard"]["stations"])


def lerp(a, b, t):
    return t * b + (1.0 - t) * a


@cache.memoize(timeout=600)
def generate_data(lon1, lat1, lon2, lat2, spm, view_width):
    geod = pyproj.Geod(ellps="clrk66")
    azi1, azi2, dist = geod.inv(lon1, lat1, lon2, lat2)
    if dist > MAX_DISTANCE:
        raise DistanceExceededError

    heightmap = WCSHeightMap(
        url=config["heightmap"]["url"]
        + "&token="
        + config["heightmap"]["token"],
        layer=config["heightmap"]["layer"],
        tile_size=int(config["heightmap"]["tile_size"]),
        resolution=int(config["heightmap"]["resolution"]),
        cache_dir=WCS_CACHE_DIR.name,
    )

    image = WMSImage(
        url=config["image"]["url"] + "&token=" + config["image"]["token"],
        layer=config["image"]["layer"],
        tile_size=int(config["image"]["tile_size"]),
        resolution=int(config["image"]["resolution"]),
        cache_dir=WMS_CACHE_DIR.name,
    )

    npts_x = round(dist * spm)
    npts_y = int(round(view_width * spm * 0.5) * 2 + 1)

    out_lon = []
    out_lat = []
    out_offset = []

    inter = geod.inv_intermediate(
        lon1, lat1, lon2, lat2, npts=npts_x, initial_idx=0, terminus_idx=0
    )

    for (
        ilon,
        ilat,
    ) in zip(inter.lons, inter.lats):
        azi_fwd, azi_bwd, d1 = geod.inv(lon1, lat1, ilon, ilat)
        flon1, flat1, faz1 = geod.fwd(
            ilon, ilat, azi_bwd - 90.0, view_width / 2.0
        )
        flon2, flat2, faz2 = geod.fwd(
            ilon, ilat, azi_bwd + 90.0, view_width / 2.0
        )

        rinter = geod.inv_intermediate(
            flon1,
            flat1,
            flon2,
            flat2,
            npts=npts_y,
            initial_idx=0,
            terminus_idx=0,
        )

        out_lon.extend(rinter.lons)
        out_lat.extend(rinter.lats)
        out_offset.extend(
            np.linspace(-view_width / 2.0, view_width / 2.0, npts_y)
        )

    out_height = heightmap.get_heights(out_lon, out_lat)
    out_color = image.get_pixels(out_lon, out_lat)

    height_start = out_height[int(npts_y) // 2]
    height_end = out_height[int((npts_x - 1) * npts_y + npts_y // 2)]

    return dict(
        dist=dist,
        npts_x=npts_x,
        npts_y=npts_y,
        offset=out_offset,
        height=out_height,
        color=out_color,
        height_start=height_start,
        height_end=height_end,
    )


@cache.memoize(timeout=600)
def generate_mesh_indices(n: int, m: int):
    t1, t2, t3 = [], [], []
    for i in range(n - 1):
        for j in range(m - 1):
            k = i * m + j

            t1.append(k)
            t2.append(k + m)
            t3.append(k + 1)

            t1.append(k + 1)
            t2.append(k + m)
            t3.append(k + m + 1)
    return t1, t2, t3


@app.callback(
    Output("data", "data"),
    Output("data_loading", "children"),
    Output("sidebar_error", "children"),
    Output("sidebar_error", "is_open"),
    Input("session_update", "n_clicks"),
    State("gateway_id", "value"),
    State("node_id", "value"),
    State("spm", "value"),
    State("view_width", "value"),
)
def update_data(
    n_clicks: int, gateway_id: str, node_id: str, spm: float, view_width: float
):
    lon1, lat1 = stations.query("station == @gateway_id").iloc[0][
        ["lon", "lat"]
    ]
    lon2, lat2 = stations.query("station == @node_id").iloc[0][["lon", "lat"]]

    try:
        result = generate_data(lon1, lat1, lon2, lat2, spm, view_width)
    except DistanceExceededError:
        return (
            None,
            "",
            (
                "Distance between gateway and node"
                " cannot exceed {MAX_DISTANCE} meters."
            ),
            True,
        )

    data = dict(
        params=dict(
            gateway_id=gateway_id,
            node_id=node_id,
            spm=spm,
            view_width=view_width,
            lon1=lon1,
            lat1=lat1,
            lon2=lon2,
            lat2=lat2,
        ),
        result=result,
    )
    return data, "", "", False


@app.callback(
    Output("graph_2d", "figure"),
    Output("graph_2d", "clickData"),
    Input("data", "data"),
    Input("gateway_offset", "value"),
    Input("node_offset", "value"),
    Input("frequency", "value"),
)
def update_graph_2d(data, gateway_offset, node_offset, frequency):
    if data is None:
        return placeholder_figure("", PLOT_HEIGHT_2D), None

    result = data["result"]

    npts_x = result["npts_x"]
    npts_y = result["npts_y"]
    d1 = np.linspace(0, result["dist"], npts_x)
    height = result["height"][npts_y // 2::npts_y]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=d1,
            y=height,
            name="Surface height",
            mode="lines",
            hovertemplate="Gateway dist.: %{x}<br>Terrain height: %{y}",
        )
    )

    gateway_height = result["height_start"] + gateway_offset
    node_height = result["height_end"] + node_offset

    fresnel_steps_x = int(
        round(result["dist"] * float(config["dashboard"]["fresnel_ppm_x"]))
    )

    d1 = np.linspace(0, result["dist"], fresnel_steps_x)
    h = np.linspace(gateway_height, node_height, fresnel_steps_x)
    r = fresnel_zone_radius(d1, result["dist"] - d1, frequency)

    figure.add_traces(
        [
            go.Scatter(
                x=d1,
                y=h - r,
                name="Fresnel zone (lower)",
                mode="lines",
                hoverinfo="skip",
            ),
            go.Scatter(
                x=d1,
                y=h + r,
                name="Fresnel zone (upper)",
                mode="lines",
                hoverinfo="skip",
            ),
        ]
    )

    figure.update_layout(
        height=PLOT_HEIGHT_2D,
        margin=PLOT_MARGIN,
        showlegend=False,
        hovermode="x",
    )
    return figure, None


@app.callback(
    Output("graph-cross", "figure"),
    Input("data", "data"),
    Input("graph_2d", "clickData"),
    Input("gateway_offset", "value"),
    Input("node_offset", "value"),
    Input("frequency", "value"),
)
def update_graph_cross(
    data, clickData, gateway_offset, node_offset, frequency
):
    if data is None:
        return placeholder_figure("", PLOT_HEIGHT_2D)

    if clickData is None or len(clickData["points"]) == 0:
        return placeholder_figure(
            "Click height curve to show cross section", PLOT_HEIGHT_2D
        )

    result = data["result"]

    data_start = result["npts_y"] * clickData["points"][0]["pointIndex"]
    data_end = data_start + result["npts_y"]

    d1 = clickData["points"][0]["x"]
    r = fresnel_zone_radius(d1, result["dist"] - d1, frequency)
    height = result["height"][data_start:data_end]
    offset = result["offset"][data_start:data_end]

    gateway_height = result["height_start"] + gateway_offset
    node_height = result["height_end"] + node_offset
    t = d1 / result["dist"]
    los_height = lerp(gateway_height, node_height, t)

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=offset, y=height, mode="lines"))
    figure.add_shape(
        type="circle",
        xref="x",
        yref="y",
        x0=-r,
        y0=los_height - r,
        x1=r,
        y1=los_height + r,
    )
    figure.update_layout(height=PLOT_HEIGHT_2D, margin=PLOT_MARGIN)
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    return figure


@app.callback(
    Output("graph_3d", "figure"),
    Input("data", "data"),
    Input("graph_2d", "clickData"),
    Input("enable_graph_3d", "value"),
    Input("gateway_offset", "value"),
    Input("node_offset", "value"),
    Input("frequency", "value"),
    Input("3d_view_window", "value"),
)
def update_graph_3d(
    data,
    clickData,
    enable_graph_3d,
    gateway_offset,
    node_offset,
    frequency,
    window,
):
    if not enable_graph_3d:
        return placeholder_figure(
            "3D terrain view is disabled", PLOT_HEIGHT_3D
        )

    if data is None:
        return placeholder_figure("", PLOT_HEIGHT_3D)

    if clickData is None or len(clickData["points"]) == 0:
        return placeholder_figure(
            "Click height curve to show cross section", PLOT_HEIGHT_3D
        )

    result = data["result"]
    npts_x = result["npts_x"]
    npts_y = result["npts_y"]

    # Compute data slice window
    window_pts = int(round(npts_x / result["dist"] * window))
    xi_start = int(clickData["points"][0]["pointIndex"] - window_pts // 2)
    xi_end = xi_start + window_pts

    xi_start = max(xi_start, 0)
    xi_end = min(xi_end, npts_x - 1)
    window_pts = xi_end - xi_start

    data_start = xi_start * npts_y
    data_end = xi_end * npts_y

    # Compute d1 distance falues
    t_start = xi_start / npts_x
    t_end = (xi_end - 1) / npts_x
    d1 = np.repeat(
        np.linspace(
            t_start * result["dist"], t_end * result["dist"], window_pts
        ),
        npts_y,
    )

    # Extract result data slices
    offset = result["offset"][data_start:data_end]
    height = result["height"][data_start:data_end]
    color = result["color"][data_start:data_end]

    # Add 3D terrain trace
    t1, t2, t3 = generate_mesh_indices(window_pts, result["npts_y"])
    color_rgb = ["#{:02x}{:02x}{:02x}".format(*c) for c in color]

    figure = go.Figure()
    figure.add_trace(
        go.Mesh3d(
            x=d1,
            y=offset,
            z=height,
            i=t1,
            j=t2,
            k=t3,
            vertexcolor=color_rgb,
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
        )
    )

    # Add 3D fresnel zone trace
    fresnel_steps_x = int(
        round(window * float(config["dashboard"]["fresnel_ppm_x"]))
    )
    fresnel_steps_y = int(config["dashboard"]["fresnel_steps_y"])

    gateway_height = result["height_start"] + gateway_offset
    node_height = result["height_end"] + node_offset

    height_start = lerp(gateway_height, node_height, t_start)
    height_end = lerp(gateway_height, node_height, t_end)

    x, y, z = [], [], []
    angles = np.arange(fresnel_steps_y) / fresnel_steps_y * np.pi * 2.0
    angles_cos = np.cos(angles)
    angles_sin = np.sin(angles)
    for h, d in zip(
        np.linspace(height_start, height_end, fresnel_steps_x),
        np.linspace(d1[0], d1[-1], fresnel_steps_x),
    ):
        r = fresnel_zone_radius(d, result["dist"] - d, frequency)
        x.extend(np.repeat(d, len(angles)))
        y.extend(angles_cos * r)
        z.extend(h + angles_sin * r)

    figure.add_trace(
        go.Mesh3d(x=x, y=y, z=z, alphahull=0, opacity=0.3, hoverinfo="none")
    )

    # Draw ring around fresnel at clicked point
    d1_click = clickData["points"][0]["x"]
    t_click = d1_click / result["dist"]
    h_click = lerp(gateway_height, node_height, t_click)
    r_click = fresnel_zone_radius(
        d1_click, result["dist"] - d1_click, frequency
    )
    angles = np.linspace(0, 2.0 * np.pi, fresnel_steps_y + 1)
    figure.add_trace(
        go.Scatter3d(
            x=np.repeat(d1_click, len(angles)),
            y=np.cos(angles) * r_click,
            z=h_click + np.sin(angles) * r_click,
            mode="lines",
            line=dict(color="#0000ff"),
        )
    )

    figure.update_layout(
        margin=PLOT_MARGIN,
        height=PLOT_HEIGHT_3D,
        scene=dict(
            aspectmode="data",
            camera_projection_type="orthographic",
            camera_eye=dict(x=-2.5, y=-2.5, z=2.5),
        ),
    )

    return figure


@app.callback(
    Output("collapse_graph_3d", "is_open"), Input("enable_graph_3d", "value")
)
def update_collapse_graph_3d(enable_graph_3d):
    return enable_graph_3d


app.layout = build_layout(stations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=80)
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()

    cache.clear()
    app.run_server(host="0.0.0.0", port=args.port, debug=args.debug)
