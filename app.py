import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from flask_caching import Cache

import os
import configparser
import numpy as np
import pandas as pd
from wcs_height_map import WCSHeightMap
from wms_image import WMSImage
import pyproj
from fresnel import fresnel_zone_radius


class DistanceExceededError(Exception):
    pass


PLOT_HEIGHT_2D = 250
PLOT_HEIGHT_3D = 400
MARGIN_NONE = dict(t=0, r=0, b=0, l=0)
MAX_DISTANCE = 50000

config = configparser.ConfigParser()
config.read("config.defaults.ini")
if os.path.exists("config.ini"):
    config.read("config.ini")

theme_url = getattr(dbc.themes, config["dashboard"]["theme"].upper())
app = dash.Dash(__name__, external_stylesheets=[theme_url], prevent_initial_callbacks=True)
app.title = "LoRaWAN line of sight helper"

cache = Cache(app.server, config={
    "CACHE_TYPE": "filesystem",
    "CACHE_DIR": config["flask"]["cache_dir"]
})

stations = pd.read_csv(config["dashboard"]["stations"])


def lerp(a, b, t):
    return t * b + (1.0 - t) * a


@cache.memoize(timeout=600)
def generate_data(lon1, lat1, lon2, lat2, frequency, spm, padding):
    geod = pyproj.Geod(ellps="clrk66")
    azi1, azi2, dist = geod.inv(lon1, lat1, lon2, lat2)
    if dist > MAX_DISTANCE:
        raise DistanceExceededError

    heightmap = WCSHeightMap(
        url=config["heightmap"]["url"] + "&token=" + config["heightmap"]["token"],
        layer=config["heightmap"]["layer"],
        tile_size=int(config["heightmap"]["tile_size"]),
        resolution=int(config["heightmap"]["resolution"])
    )

    image = WMSImage(
        url=config["image"]["url"] + "&token=" + config["image"]["token"],
        layer=config["image"]["layer"],
        tile_size=int(config["image"]["tile_size"]),
        resolution=int(config["image"]["resolution"])
    )

    max_r = fresnel_zone_radius(dist / 2.0, dist / 2.0, frequency) + 2.0 * padding
    npts_x = round(dist * spm)
    npts_y = int(round(max_r * spm * 0.5) * 2 + 1)

    out_lon = []
    out_lat = []
    out_d1 = []
    out_offset = []

    inter = geod.inv_intermediate(lon1, lat1, lon2, lat2, npts=npts_x, initial_idx=0, terminus_idx=0)

    for ilon, ilat, in zip(inter.lons, inter.lats):
        azi_fwd, azi_bwd, d1 = geod.inv(lon1, lat1, ilon, ilat)
        flon1, flat1, faz1 = geod.fwd(ilon, ilat, azi_bwd - 90.0, max_r)
        flon2, flat2, faz2 = geod.fwd(ilon, ilat, azi_bwd + 90.0, max_r)

        rinter = geod.inv_intermediate(
            flon1, flat1, flon2, flat2,
            npts=npts_y,
            initial_idx=0,
            terminus_idx=0,
        )

        out_lon.extend(rinter.lons)
        out_lat.extend(rinter.lats)
        out_d1.extend(np.repeat(d1, npts_y))
        out_offset.extend(np.linspace(-max_r, max_r, npts_y))

    out_height = heightmap.get_heights(out_lon, out_lat)
    out_color = image.get_pixels(out_lon, out_lat)

    height_start = out_height[int(npts_y) // 2]
    height_end = out_height[int((npts_x - 1) * npts_y + npts_y // 2)]

    return dict(
        dist=dist,
        npts_x=npts_x,
        npts_y=npts_y,
        lon=out_lon,
        lat=out_lat,
        d1=out_d1,
        offset=out_offset,
        height=out_height,
        color=out_color,
        height_start=height_start,
        height_end=height_end
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
    State("frequency", "value"),
    State("spm", "value"),
    State("padding", "value")
)
def update_data(
    n_clicks: int,
    gateway_id: str,
    node_id: str,
    frequency: float,
    spm: float,
    padding: float
):
    lon1, lat1 = stations.query("station == @gateway_id").iloc[0][["lon", "lat"]]
    lon2, lat2 = stations.query("station == @node_id").iloc[0][["lon", "lat"]]

    try:
        result = generate_data(lon1, lat1, lon2, lat2, frequency, spm, padding)
    except DistanceExceededError:
        return None, "", f"Distance between gateway and node cannot exceed {MAX_DISTANCE} meters.", True

    data = dict(
        params=dict(
            gateway_id=gateway_id,
            node_id=node_id,
            frequency=frequency,
            spm=spm,
            padding=padding,
            lon1=lon1,
            lat1=lat1,
            lon2=lon2,
            lat2=lat2
        ),
        result=result
    )
    return data, "", "", False


@app.callback(
    Output("graph-2d", "figure"),
    Output("graph-2d", "clickData"),
    Input("data", "data"),
    Input("gateway_offset", "value"),
    Input("node_offset", "value"),
)
def update_graph_2d(data, gateway_offset, node_offset):
    if data is None:
        return placeholder_figure("", PLOT_HEIGHT_2D), None

    result = data["result"]
    params = data["params"]

    npts_y = result["npts_y"]
    d1 = result["d1"][npts_y // 2::npts_y]
    height = result["height"][npts_y // 2::npts_y]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=d1,
            y=height,
            name="Surface height",
            mode="lines",
            hovertemplate="Gateway dist.: %{x}<br>Terrain height: %{y}"
        )
    )

    gateway_height = result["height_start"] + gateway_offset
    node_height = result["height_end"] + node_offset

    fresnel_steps_x = int(round(result["dist"] * float(config["dashboard"]["fresnel_ppm_x"])))

    d1 = np.linspace(0, result["dist"], fresnel_steps_x)
    h = np.linspace(gateway_height, node_height, fresnel_steps_x)
    r = fresnel_zone_radius(d1, result["dist"] - d1, params["frequency"])

    figure.add_traces([
        go.Scatter(x=d1, y=h-r, name="Fresnel zone (lower)", mode="lines", hoverinfo="skip"),
        go.Scatter(x=d1, y=h+r, name="Fresnel zone (upper)", mode="lines", hoverinfo="skip")
    ])

    figure.update_layout(
        height=PLOT_HEIGHT_2D,
        margin=MARGIN_NONE,
        showlegend=False,
        hovermode="x"
    )
    return figure, None


@app.callback(
    Output("graph-cross", "figure"),
    Input("data", "data"),
    Input("graph-2d", "clickData"),
    Input("gateway_offset", "value"),
    Input("node_offset", "value")
)
def update_graph_cross(data, clickData, gateway_offset, node_offset):
    if data is None:
        return placeholder_figure("", PLOT_HEIGHT_2D)

    if clickData is None or len(clickData["points"]) == 0:
        return placeholder_figure("Click height curve to show cross section", PLOT_HEIGHT_2D)

    result = data["result"]
    params = data["params"]

    data_start = result["npts_y"] * clickData["points"][0]["pointIndex"]
    data_end = data_start + result["npts_y"]

    d1 = clickData["points"][0]["x"]
    r = fresnel_zone_radius(d1, result["dist"] - d1, params["frequency"])
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
        xref="x", yref="y",
        x0=-r, y0=los_height - r,
        x1=r, y1=los_height + r
    )
    figure.update_layout(
        height=PLOT_HEIGHT_2D,
        margin=MARGIN_NONE
    )
    figure.update_yaxes(
        scaleanchor="x",
        scaleratio=1
    )
    return figure


@app.callback(
    Output("graph-3d", "figure"),
    Input("data", "data"),
    Input("graph-2d", "clickData"),
    Input("enable-3d-graph", "value"),
    Input("gateway_offset", "value"),
    Input("node_offset", "value"),
    Input("3d_view_window", "value")
)
def update_graph_3d(data, clickData, enable_3d_graph, gateway_offset, node_offset, window):
    if not enable_3d_graph:
        return placeholder_figure("3D terrain view is disabled", PLOT_HEIGHT_3D)

    if data is None:
        return placeholder_figure("", PLOT_HEIGHT_3D)

    if clickData is None or len(clickData["points"]) == 0:
        return placeholder_figure("Click height curve to show cross section", PLOT_HEIGHT_3D)

    result = data["result"]
    params = data["params"]
    npts_x = result["npts_x"]
    npts_y = result["npts_y"]

    window_pts = int(round(npts_x / result["dist"] * window))
    xi_mid = int(clickData["points"][0]["pointIndex"])
    xi_start = xi_mid - window_pts // 2
    xi_end = xi_start + window_pts

    xi_start = max(xi_start, 0)
    xi_end = min(xi_end, npts_x-1)
    window_pts = xi_end - xi_start

    data_start = xi_start * npts_y
    data_end = xi_end * npts_y

    d1 = result["d1"][data_start:data_end]
    offset = result["offset"][data_start:data_end]
    height = result["height"][data_start:data_end]
    color = result["color"][data_start:data_end]

    # Add 3D terrain trace
    t1, t2, t3 = generate_mesh_indices(window_pts, result["npts_y"])
    color_rgb = ["#{:02x}{:02x}{:02x}".format(*c) for c in color]

    figure = go.Figure()
    figure.add_trace(
        go.Mesh3d(
            x=d1, y=offset, z=height,
            i=t1, j=t2, k=t3,
            vertexcolor=color_rgb,
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0)
        )
    )

    # Add 3D fresnel zone trace
    fresnel_steps_x = int(round(window * float(config["dashboard"]["fresnel_ppm_x"])))
    fresnel_steps_y = int(config["dashboard"]["fresnel_steps_y"])

    height_start = result["height_start"] + gateway_offset
    height_end = result["height_end"] + node_offset

    t0 = d1[0] / result["dist"]
    t1 = d1[-1] / result["dist"]

    height0 = lerp(height_start, height_end, t0)
    height1 = lerp(height_start, height_end, t1)

    x, y, z = [], [], []
    angles = np.arange(fresnel_steps_y) / fresnel_steps_y * np.pi * 2.0
    angles_cos = np.cos(angles)
    angles_sin = np.sin(angles)
    for h, d in zip(
        np.linspace(height0, height1, fresnel_steps_x),
        np.linspace(d1[0], d1[-1], fresnel_steps_x)
    ):
        r = fresnel_zone_radius(d, result["dist"] - d, params["frequency"])
        x.extend(np.repeat(d, len(angles)))
        y.extend(angles_cos * r)
        z.extend(h + angles_sin * r)

    figure.add_trace(
        go.Mesh3d(x=x, y=y, z=z, alphahull=0, opacity=0.4, hoverinfo="none")
    )

    # Draw ring around fresnel at clicked point
    d1_click = clickData["points"][0]["x"]
    t_click = d1_click / result["dist"]
    h_click = lerp(height_start, height_end, t_click)
    r_click = fresnel_zone_radius(d1_click, result["dist"]-d1_click, params["frequency"])
    angles = np.linspace(0, 2.0*np.pi, fresnel_steps_y+1)
    figure.add_trace(
        go.Scatter3d(
            x=np.repeat(d1_click, len(angles)),
            y=np.cos(angles) * r_click,
            z=h_click + np.sin(angles) * r_click,
            mode="lines",
            line=dict(color="#0000ff")
        )
    )

    figure.update_layout(
        margin=MARGIN_NONE,
        height=PLOT_HEIGHT_3D,
        scene=dict(
            aspectmode="data",
            camera_projection_type="orthographic",
            camera_eye=dict(x=-2.5, y=-2.5, z=2.5)
        )
    )

    return figure


def placeholder_figure(text: str = "", height: int = 100):
    figure = go.Figure()
    figure.update_layout(
        height=height,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis_visible=False,
        yaxis_visible=False,
        annotations=[
            dict(
                text=text,
                xref="paper",
                yref="paper",
                showarrow=False,
                font_size=14,
                font_color="#808080"
            )
        ]
    )
    return figure


def sidebar_numeric_input(
    id: str, label: str, unit: str, min: float, max: float, value: float
) -> html.Div:
    return html.Div([
        dbc.Label(label, html_for=id),
        dbc.InputGroup([
            dbc.Input(id=id, type="number", min=min, max=max, value=value),
            dbc.InputGroupText(unit),
        ]),
    ], className="mb-3")


numeric_inputs = [
    dict(id="frequency", label="Frequency", unit="GHz", min=0, max=10, value=0.858),
    dict(id="spm", label="Sample resolution", unit="per meter", min=0.01, max=2, value=0.8),
    dict(id="padding", label="View padding", unit="m", min=0, max=100, value=10),
]

navbar = dbc.NavbarSimple(
    brand="LoRaWAN line of sight helper",
    brand_href="#",
    fluid=True,
    dark=True,
    color="dark",
)

options_stations = [dict(label=id, value=id) for id in stations.station]
sidebar = dbc.Form(
    [
        html.H5("Session settings"),
        html.Div([
            dbc.Label("Gateway location", html_for="gateway_id"),
            dbc.Select(id="gateway_id", options=options_stations, value="FGV"),
        ], className="mb-3"),
        html.Div([
            dbc.Label("Node location", html_for="node_id"),
            dbc.Select(id="node_id", options=options_stations, value="FGD"),
        ], className="mb-3"),
        html.Div([
            sidebar_numeric_input(**args)
            for args in numeric_inputs
        ]),
        dbc.Button("Update", id="session_update", type="submit", className="mb-3"),
        dbc.Alert(id="sidebar_error", color="danger", is_open=False),
        dcc.Loading(html.Div(id="data_loading")),
    ],
    className="pt-3"
),

container = dbc.Container(
    dbc.Row(
        [
            dbc.Col(sidebar, md=3, lg=2),
            dbc.Col(
                [
                    html.H5("Fresnel zone"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Gateway height", width="auto"),
                            dbc.InputGroup([
                                dbc.Input(id="gateway_offset", type="number", value=12.0),
                                dbc.InputGroupText("m")
                            ])
                        ], sm=6, lg=3, xl=2),
                        dbc.Col([
                            dbc.Label("Node height", width="auto"),
                            dbc.Input(id="node_offset", type="number", value=2.0)
                        ], sm=6, lg=3, xl=2)
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.H5("Height curve"),
                            html.Div(
                                dcc.Loading(dcc.Graph(id="graph-2d", figure=placeholder_figure("", PLOT_HEIGHT_2D))),
                                className="border"
                            ),
                        ], lg=9),
                        dbc.Col([
                            html.H5("Cross section"),
                            html.Div(
                                dcc.Loading(dcc.Graph(id="graph-cross", figure=placeholder_figure("", PLOT_HEIGHT_2D))),
                                className="border"
                            ),
                            html.P(id="test")
                        ], lg=3)
                    ], className="mb-3"),
                    html.H5("3D terrain"),
                    dbc.Row([
                        dbc.Col(
                            dbc.Switch(id="enable-3d-graph", label="Show 3D terrain", value=False),
                            sm=6, lg=3, xl=2
                        ),
                        dbc.Col([
                            dbc.Label("View window", html_for="3d_view_window"),
                            dbc.InputGroup([
                                dbc.Input(id="3d_view_window", type="number", min=10, max=5000, value=400),
                                dbc.InputGroupText("m")
                            ])
                        ], sm=6, lg=3, xl=2)
                    ], className="mb-3"),
                    html.Div(dcc.Loading(dcc.Graph(id="graph-3d", figure=placeholder_figure("", PLOT_HEIGHT_3D))), className="border")
                ],
                md=9,
                lg=10,
                className="py-3"
            ),
            html.Footer(
                html.Small("Energinet | LoRaWAN line of sight helper", className="text-muted"),
                className="border-top py-3"
            )
        ]
    ),
    fluid=True
)

app.layout = html.Div([navbar, container, dcc.Store(id="data")])


if __name__ == "__main__":
    cache.clear()
    app.run_server(debug=True)
