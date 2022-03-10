import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import json

from wcs_height_map import WCSHeightMap
from wms_image import WMSImage
import pyproj
import plotly.graph_objects as go
import numpy as np
from fresnel import fresnel_zone_radius
import pandas as pd


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
                font_size=18
            )
        ]
    )
    return figure


with open("config.json", "r") as fp:
    config = json.load(fp)

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

numeric_inputs = [
    dict(id="gateway_height", label="Gateway height", unit="m", min=0, max=100, value=12),
    dict(id="node_height", label="Node height", unit="m", min=0, max=100, value=2),
    dict(id="frequency", label="Frequency", unit="GHz", min=0, max=10, value=0.858),
    dict(id="spm", label="Samples per meter", unit="per meter", min=0.01, max=2, value=0.5),
    dict(id="padding", label="View padding", unit="m", min=0, max=100, value=10),
]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN])
app.title = "LoRaWAN line of sight helper"

navbar = dbc.NavbarSimple(
    brand="LoRaWAN line of sight helper",
    brand_href="#",
    fluid=True,
    dark=True,
    color="dark",
)

options_stations = [dict(label=id, value=id) for id in stations.station]
sidebar = html.Div(
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
        dbc.Button("Submit", id="submit", className="mt-3"),
    ],
    className="mt-3"
),

container = dbc.Container(
    dbc.Row(
        [
            dbc.Col(sidebar, md=3, lg=2),
            dbc.Col(
                [
                    html.H2("Height curve"),
                    html.Hr(),
                    html.H2("3D terrain"),
                    html.Div(dcc.Loading(dcc.Graph(id="graph-3d")), className="border")
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
        return placeholder_figure("")

    if not enable_3d_graph:
        return placeholder_figure("3D terrain generation is disabled.")
    
    # Get location coordinates
    lon1, lat1 = stations.query("station == @gateway_id").iloc[0][["lon", "lat"]]
    lon2, lat2 = stations.query("station == @node_id").iloc[0][["lon", "lat"]]

    geod = pyproj.Geod(ellps="clrk66")
    azi1, azi2, dist = geod.inv(lon1, lat1, lon2, lat2)
    max_r = fresnel_zone_radius(dist / 2.0, dist / 2.0, frequency) + 2.0 * padding
    npts_x = round(dist * spm)
    npts_y = round(max_r * spm)

    inter = geod.inv_intermediate(lon1, lat1, lon2, lat2, npts=npts_x, initial_idx=0, terminus_idx=0)
    data = []

    for ilon, ilat in zip(inter.lons, inter.lats):
        azi_fwd, azi_bwd, d1 = geod.inv(lon1, lat1, ilon, ilat)
        d2 = np.clip(dist - d1, 0, np.inf)

        r = fresnel_zone_radius(d1, d2, 0.868)
        flon1, flat1, faz1 = geod.fwd(ilon, ilat, azi_fwd - 90.0, max_r)
        flon2, flat2, faz2 = geod.fwd(ilon, ilat, azi_fwd + 90.0, max_r)

        rinter = geod.inv_intermediate(
            flon1, flat1, flon2, flat2,
            npts=npts_y,
            initial_idx=0,
            terminus_idx=0,
        )

        for i, lon, lat in zip(range(npts_y), rinter.lons, rinter.lats):
            h = heightmap.get_height(lon, lat)
            pixel = photo.get_pixel(lon, lat)

            color = "rgb(" + ",".join(map(str, pixel)) + ")"
            offset = (i / (npts_y - 1)) * max_r * 2 - max_r

            data.append(dict(
                d1=d1,
                offset=offset,
                lon=lon,
                lat=lat1,
                height=h,
                color=color,
            ))
    

    data = pd.DataFrame(data)
    t1, t2, t3 = generate_mesh_indices(npts_x, npts_y)

    figure = go.Figure()
    figure.add_trace(
        go.Mesh3d(
            x=data["d1"],
            y=data["offset"],
            z=data["height"],
            i=t1,
            j=t2,
            k=t3,
            vertexcolor=data["color"],
            customdata=data[["lat", "lon"]],
        )
    )

    height_start = heightmap.get_height(lon1, lat1) + gateway_height
    height_end = heightmap.get_height(lon2, lat2) + node_height

    x, y, z = [], [], []
    fresnel_npts_x = round(dist / 50)
    fresnel_npts_y = 12
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
        scene=dict(
            aspectmode="data",
            camera_projection_type="orthographic",
            camera_eye=dict(x=0, y=-2.5, z=2.5),
        ),
        height=500,
    )

    return figure


if __name__ == "__main__":
    app.run_server(debug=True)
