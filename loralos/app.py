import dash
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

import os
import argparse
import configparser
import numpy as np
import pandas as pd
import pyproj
from geo import los_height_range, los_height_single
from terrain import generate_data, DistanceExceededError
from fresnel import fresnel_zone_radius
from geometry import plane_mesh_indices, tube_mesh_indices

from layout import (
    build_layout,
    placeholder_figure,
    PLOT_HEIGHT_2D,
    PLOT_HEIGHT_3D,
    PLOT_MARGIN,
)

config = configparser.ConfigParser(
    defaults=os.environ, interpolation=configparser.ExtendedInterpolation()
)
config.read("config.ini")

theme_url = getattr(dbc.themes, config["dashboard"]["theme"].upper())
app = dash.Dash(
    __name__,
    external_stylesheets=[
        theme_url,
        dbc.icons.FONT_AWESOME,
        "/assets/style.css"
    ],
    prevent_initial_callbacks=True,
)
app.title = "LoRaWAN line of sight helper"

stations = pd.read_csv(config["dashboard"]["stations"])


def lerp(a, b, t):
    """Linear interpolation."""
    return t * b + (1.0 - t) * a


def google_maps_link(lat: float, lon: float) -> html.A:
    return html.A(
        f"{lat:.5f}, {lon:.5f}",
        href=f"https://www.google.com/maps/search/?api=1&query={lat},{lon}",
        target="_blank",
    )


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
        data = generate_data(config, lon1, lat1, lon2, lat2, spm, view_width)
    except DistanceExceededError:
        return (
            None,
            "",
            "Distance between gateway and node cannot exceed {MAX_DISTANCE} meters.",  # noqa: W505
            True,
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

    npts_x = data["npts_x"]
    npts_y = data["npts_y"]
    d1 = np.linspace(0, data["dist"], npts_x)
    height = data["height"][npts_y // 2 :: npts_y]

    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=d1,
            y=height,
            name="Surface height",
            mode="lines",
            hovertemplate="Gateway dist.: %{x}<br>Terrain height: %{y}",
        )
    )

    gateway_height = data["height_start"] + gateway_offset
    node_height = data["height_end"] + node_offset

    fresnel_steps_x = int(
        round(data["dist"] * float(config["dashboard"]["fresnel_ppm_x"]))
    )

    d1 = np.linspace(0, data["dist"], fresnel_steps_x)
    h = los_height_range(
        data["lon1"],
        data["lat1"],
        gateway_height,
        data["lon2"],
        data["lat2"],
        node_height,
        fresnel_steps_x,
    )
    r = fresnel_zone_radius(d1, data["dist"] - d1, frequency)

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
        hovermode="closest",
        hoverdistance=-1,
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

    data_start = data["npts_y"] * clickData["points"][0]["pointIndex"]
    data_end = data_start + data["npts_y"]

    d1 = clickData["points"][0]["x"]
    r = fresnel_zone_radius(d1, data["dist"] - d1, frequency)
    height = data["height"][data_start:data_end]
    offset = data["offset"][data_start:data_end]

    gateway_height = data["height_start"] + gateway_offset
    node_height = data["height_end"] + node_offset
    t = d1 / data["dist"]
    los_height = los_height_single(
        data["lon1"],
        data["lat1"],
        gateway_height,
        data["lon2"],
        data["lat2"],
        node_height,
        t,
    )

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
        raise PreventUpdate

    if data is None:
        return placeholder_figure("", PLOT_HEIGHT_3D)

    if clickData is None or len(clickData["points"]) == 0:
        return placeholder_figure(
            "Click height curve to show cross section", PLOT_HEIGHT_3D
        )

    npts_x = data["npts_x"]
    npts_y = data["npts_y"]

    # Compute data slice window
    window_pts = int(round(npts_x / data["dist"] * window))
    xi_start = int(clickData["points"][0]["pointIndex"] - window_pts // 2)
    xi_end = xi_start + window_pts

    # Clamp values to range
    xi_start = max(xi_start, 0)
    xi_end = min(xi_end, npts_x - 1)
    window_pts = xi_end - xi_start

    data_start = xi_start * npts_y
    data_end = xi_end * npts_y

    # Compute d1 distance values
    t_start = xi_start / npts_x
    t_end = (xi_end - 1) / npts_x
    d1 = np.repeat(
        np.linspace(t_start * data["dist"], t_end * data["dist"], window_pts),
        npts_y,
    )

    # Extract result data slices
    offset = data["offset"][data_start:data_end]
    height = data["height"][data_start:data_end]
    color = data["color"][data_start:data_end]

    # Add 3D terrain trace
    t1, t2, t3 = plane_mesh_indices(window_pts, data["npts_y"])
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

    gateway_height = data["height_start"] + gateway_offset
    node_height = data["height_end"] + node_offset

    x, y, z = [], [], []
    angles = np.arange(fresnel_steps_y) / fresnel_steps_y * np.pi * 2.0
    angles_cos = np.cos(angles)
    angles_sin = np.sin(angles)
    for t, d in zip(
        np.linspace(t_start, t_end, fresnel_steps_x),
        np.linspace(d1[0], d1[-1], fresnel_steps_x),
    ):
        h = los_height_single(
            data["lon1"],
            data["lat1"],
            gateway_height,
            data["lon2"],
            data["lat2"],
            node_height,
            t,
        )
        r = fresnel_zone_radius(d, data["dist"] - d, frequency)

        x.extend(np.repeat(d, len(angles)))
        y.extend(angles_cos * r)
        z.extend(h + angles_sin * r)
    t1, t2, t3 = tube_mesh_indices(fresnel_steps_x, fresnel_steps_y)

    figure.add_trace(
        go.Mesh3d(
            x=x, y=y, z=z, i=t1, j=t2, k=t3, opacity=0.3, hoverinfo="none"
        )
    )

    # Draw ring around fresnel at clicked point
    d1_click = clickData["points"][0]["x"]
    t_click = d1_click / data["dist"]
    h_click = lerp(gateway_height, node_height, t_click)
    r_click = fresnel_zone_radius(d1_click, data["dist"] - d1_click, frequency)
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
    Output("session_gateway_location", "children"),
    Output("session_node_location", "children"),
    Output("session_distance", "children"),
    Output("session_azimuth", "children"),
    Output("session_compass", "style"),
    Input("data", "data"),
)
def update_session_info(data):
    if data is None:
        return "N/A", "N/A", "N/A", "N/A", {"visibility": "hidden"}

    gateway_loc = google_maps_link(data["lat1"], data["lon1"])
    node_loc = google_maps_link(data["lat2"], data["lon2"])
    distance = f"{data['dist']:.1f} meters"
    azimuth = f"{data['azi12']:.2f} °"
    compass = {"transform": f"rotate({-data['azi12']}deg)"}
    return gateway_loc, node_loc, distance, azimuth, compass


@app.callback(
    Output("link_clicked_point", "children"),
    Input("data", "data"),
    Input("graph_2d", "clickData"),
)
def update_link_clicked_point(data, clickData):
    if data is None:
        return "N/A", None

    if clickData is None or len(clickData["points"]) == 0:
        return "N/A", None

    d1 = clickData["points"][0]["x"]
    azi12 = data["azi12"]
    lon1, lat1 = data["lon1"], data["lat1"]

    g = pyproj.Geod(ellps="WGS84")
    lon, lat, azi_bwd = g.fwd(lon1, lat1, azi12, d1)

    return google_maps_link(lat, lon)


@app.callback(
    Output("collapse_graph_3d", "is_open"), Input("enable_graph_3d", "value")
)
def update_collapse_graph_3d(enable_graph_3d):
    return enable_graph_3d


app.layout = build_layout(app, stations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=int(config["dashboard"]["port"]))
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()

    app.run_server(host="0.0.0.0", port=args.port, debug=args.debug)
