import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from flask_caching import Cache

import json
import numpy as np
import pandas as pd
from operator import itemgetter
from wcs_height_map import WCSHeightMap
from wms_image import WMSImage
import pyproj
from fresnel import fresnel_zone_radius


with open("config.json", "r") as fp:
    config = json.load(fp)

theme_url = getattr(dbc.themes, config["dashboard"]["theme"].upper())
app = dash.Dash(__name__, external_stylesheets=[theme_url], prevent_initial_callbacks=True)
app.title = "LoRaWAN line of sight helper"

cache = Cache(app.server, config={
    "CACHE_TYPE": "filesystem",
    "CACHE_DIR": config["flask"]["cache_dir"]
})

stations = pd.read_csv(config["stations"]["path"])

heightmap = WCSHeightMap(
    url=config["heightmap"]["url"] + "&token=" + config["heightmap"]["token"],
    layer=config["heightmap"]["layer"],
    tile_size=config["heightmap"]["tile_size"],
    resolution=config["heightmap"]["resolution"]
)

photo = WMSImage(
    url=config["photo"]["url"] + "&token=" + config["photo"]["token"],
    layer=config["photo"]["layer"],
    tile_size=config["photo"]["tile_size"],
    resolution=config["photo"]["resolution"]
)


@cache.memoize(timeout=600)
def generate_data(lon1, lat1, lon2, lat2, frequency, spm, padding):
    geod = pyproj.Geod(ellps="clrk66")
    azi1, azi2, dist = geod.inv(lon1, lat1, lon2, lat2)

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

    out_height = [heightmap.get_height(lon, lat) for lon, lat in zip(out_lon, out_lat)]
    out_color = [photo.get_pixel(lon, lat) for lon, lat in zip(out_lon, out_lat)]

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


def placeholder_figure(text: str, height: int = 100):
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


def make_numeric_input(
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
    dict(id="gateway_height", label="Gateway height", unit="m", min=0, max=100, value=12),
    dict(id="node_height", label="Node height", unit="m", min=0, max=100, value=2),
    dict(id="frequency", label="Frequency", unit="GHz", min=0, max=10, value=0.858),
    dict(id="spm", label="Samples per meter", unit="per meter", min=0.01, max=2, value=0.5),
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
        html.Div([
            dbc.Label("Gateway location", html_for="gateway_id"),
            dbc.Select(id="gateway_id", options=options_stations, value="FGV"),
        ], className="mb-3"),
        html.Div([
            dbc.Label("Node location", html_for="node_id"),
            dbc.Select(id="node_id", options=options_stations, value="FGD"),
        ], className="mb-3"),
        html.Div([
            make_numeric_input(**args)
            for args in numeric_inputs
        ]),
        dbc.Switch(id="enable-3d-graph", label="Generate 3D terrain", value=True),
        dbc.Button("Submit", id="submit", type="submit", className="mt-3"),
    ],
    className="mt-3"
),

container = dbc.Container(
    dbc.Row(
        [
            dbc.Col(sidebar, md=3, lg=2),
            dbc.Col(
                [
                    dbc.Row([
                        dbc.Col([
                            html.H3("Height curve"),
                            html.Div(
                                dcc.Loading(dcc.Graph(id="graph-2d", figure=placeholder_figure("No results computed yet", 250))),
                                className="border"
                            ),
                        ], md=9),
                        dbc.Col([
                            html.H3("Cross section"),
                            html.Div(
                                dcc.Loading(dcc.Graph(id="graph-cross", figure=placeholder_figure("Hover over height curve to show cross section", 250))),
                                className="border"
                            ),
                            html.P(id="test")
                        ], md=3)
                    ], className="mb-3"),
                    html.H3("3D terrain"),
                    html.Div(dcc.Loading(dcc.Graph(id="graph-3d", figure=placeholder_figure("No results computed yet"))), className="border")
                ],
                md=9,
                lg=10,
                className="mt-3"
            ),
        ]
    ),
    fluid=True
)

app.layout = html.Div([navbar, container])


@app.callback(
    Output("graph-2d", "figure"),
    Input("submit", "n_clicks"),
    State("gateway_id", "value"),
    State("node_id", "value"),
    State("frequency", "value"),
    State("gateway_height", "value"),
    State("node_height", "value"),
    State("spm", "value"),
    State("padding", "value")
)
def update_graph_2d(
    n_clicks: int,
    gateway_id: str,
    node_id: str,
    frequency: float,
    gateway_height: float,
    node_height: float,
    spm: float,
    padding: float
):
    if n_clicks is None or n_clicks == 0:
        return placeholder_figure("This shouldn't happen. :(")

    # Get location coordinates
    lon1, lat1 = stations.query("station == @gateway_id").iloc[0][["lon", "lat"]]
    lon2, lat2 = stations.query("station == @node_id").iloc[0][["lon", "lat"]]

    result = generate_data(lon1, lat1, lon2, lat2, frequency, spm, padding)
    indices = range(int(result["npts_y"] // 2), result["npts_x"]*result["npts_y"], result["npts_y"])
    getter = itemgetter(*indices)
    # TODO: Replace getter with slices

    d1 = getter(result["d1"])
    height = getter(result["height"])

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=d1,
            y=height,
            customdata=d1,
            name="Surface height",
            mode="lines"
        )
    )

    height_start = result["height_start"] + gateway_height
    height_end = result["height_end"] + node_height

    fresnel_npts_x = int(config["fresnel"]["steps_x"])
    d1 = np.linspace(0, result["dist"], fresnel_npts_x)
    h = np.linspace(height_start, height_end, fresnel_npts_x)
    r = fresnel_zone_radius(d1, result["dist"] - d1, frequency)

    figure.add_trace(
        go.Scatter(x=d1, y=h-r, name="Fresnel zone (lower)", mode="lines")
    )
    figure.add_trace(
        go.Scatter(x=d1, y=h+r, name="Fresnel zone (upper)", mode="lines")
    )

    figure.update_layout(
        height=250,
        margin=dict(t=0, r=0, b=0, l=0),
        showlegend=False
    )
    return figure


@app.callback(
    Output("graph-3d", "figure"),
    Input("submit", "n_clicks"),
    State("enable-3d-graph", "value"),
    State("gateway_id", "value"),
    State("node_id", "value"),
    State("frequency", "value"),
    State("gateway_height", "value"),
    State("node_height", "value"),
    State("spm", "value"),
    State("padding", "value"),
)
def update_graph_3d(
    n_clicks: int,
    enable_3d_graph: bool,
    gateway_id: str,
    node_id: str,
    frequency: float,
    gateway_height: float,
    node_height: float,
    spm: float,
    padding: float,
):
    if n_clicks is None or n_clicks == 0:
        return placeholder_figure("This shouldn't happen. :(")

    if not enable_3d_graph:
        return placeholder_figure("3D terrain generation is disabled")
    
    # Get location coordinates
    lon1, lat1 = stations.query("station == @gateway_id").iloc[0][["lon", "lat"]]
    lon2, lat2 = stations.query("station == @node_id").iloc[0][["lon", "lat"]]

    result = generate_data(lon1, lat1, lon2, lat2, frequency, spm, padding)
    dist = result["dist"]
    t1, t2, t3 = generate_mesh_indices(result["npts_x"], result["npts_y"])
    color_rgb = ["#{:02x}{:02x}{:02x}".format(*c) for c in result["color"]]

    figure = go.Figure()
    figure.add_trace(
        go.Mesh3d(
            x=result["d1"],
            y=result["offset"],
            z=result["height"],
            i=t1,
            j=t2,
            k=t3,
            vertexcolor=color_rgb,
            lighting=dict(
                ambient=1.0,
                diffuse=0.0,
                specular=0.0
            )
        )
    )

    x, y, z = [], [], []
    fresnel_npts_x = int(config["fresnel"]["steps_x"])
    fresnel_npts_y = int(config["fresnel"]["steps_y"])
    height_start = result["height_start"] + gateway_height
    height_end = result["height_end"] + node_height
    for h, d1 in zip(
        np.linspace(height_start, height_end, fresnel_npts_x),
        np.linspace(0, dist, fresnel_npts_x)
    ):
        r = fresnel_zone_radius(d1, dist - d1, frequency)
        for i in range(fresnel_npts_y):
            angle = i / fresnel_npts_y * np.pi * 2.0
            x.append(d1)
            y.append(np.cos(angle) * r)
            z.append(h + np.sin(angle) * r)

    figure.add_trace(
        go.Mesh3d(x=x, y=y, z=z, alphahull=0, opacity=0.5, hoverinfo="none")
    )

    figure.update_layout(
        margin=dict(t=0, r=0, b=0, l=0),
        height=500,
        scene=dict(
            aspectmode="data",
            camera_projection_type="orthographic",
            camera_eye=dict(x=0, y=-2.5, z=2.5),
        ),
    )

    return figure


@app.callback(
    Output("graph-cross", "figure"),
    Input("graph-2d", "hoverData"),
    State("gateway_id", "value"),
    State("node_id", "value"),
    State("frequency", "value"),
    State("gateway_height", "value"),
    State("node_height", "value"),
    State("spm", "value"),
    State("padding", "value"),
)
def update_graph_cross(
    hoverData,
    gateway_id: str,
    node_id: str,
    frequency: float,
    gateway_height: float,
    node_height: float,
    spm: float,
    padding: float,
):
    if hoverData["points"][0]["curveNumber"] != 0:
        raise PreventUpdate

    lon1, lat1 = stations.query("station == @gateway_id").iloc[0][["lon", "lat"]]
    lon2, lat2 = stations.query("station == @node_id").iloc[0][["lon", "lat"]]
    result = generate_data(lon1, lat1, lon2, lat2, frequency, spm, padding)

    point_index = hoverData["points"][0]["pointIndex"]
    data_start = result["npts_y"] * point_index

    d1 = hoverData["points"][0]["customdata"]
    r = fresnel_zone_radius(d1, result["dist"] - d1, frequency)
    height = result["height"][data_start:data_start+result["npts_y"]]
    offset = result["offset"][data_start:data_start+result["npts_y"]]

    t = d1 / result["dist"]
    height_start = result["height_start"] + gateway_height
    height_end = result["height_end"] + node_height
    los_height = height_end * t + (1.0 - t) * height_start

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=offset,
            y=height,
            mode="lines"
        )
    )
    figure.add_shape(
        type="circle",
        xref="x",
        yref="y",
        x0=-r, y0=los_height-r, x1=r, y1=los_height+r
    )
    figure.update_layout(
        height=250,
        margin=dict(t=0, r=0, b=0, l=0),
    )
    figure.update_yaxes(
        scaleanchor="x",
        scaleratio=1
    )
    return figure


if __name__ == "__main__":
    cache.clear()
    app.run_server(debug=True)
