from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from __version__ import __version__ as VERSION


PLOT_HEIGHT_2D = 250
PLOT_HEIGHT_3D = 400
PLOT_MARGIN = dict(t=0, r=0, b=0, l=0)
GITHUB_URL = "https://github.com/SimonLarsen/loralos"
DOCKER_URL = "https://hub.docker.com/repository/docker/simonlarsen/loralos"


def placeholder_figure(text: str = "", height: int = 100) -> go.Figure:
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
                font_color="#808080",
            )
        ],
    )
    return figure


def build_layout(app, stations: pd.DataFrame):
    navbar = dbc.NavbarSimple(
        brand="LoRaWAN line of sight helper",
        children=[
            dbc.NavItem(
                dbc.NavLink(
                    href=GITHUB_URL,
                    className="fab fa-github p-0",
                    style={"font-size": "24px"},
                    target="_blank",
                )
            ),
            dbc.NavItem(
                dbc.NavLink(
                    href=DOCKER_URL,
                    className="fab fa-docker p-0",
                    style={"font-size": "24px"},
                    target="_blank",
                )
            ),
        ],
        brand_href="#",
        fluid=True,
        dark=True,
        color="dark",
    )

    options_stations = [dict(label=id, value=id) for id in stations.station]
    sidebar = dbc.Form(
        [
            html.H5("Session settings"),
            html.Div(
                [
                    dbc.Label("Gateway location", html_for="gateway_id"),
                    dbc.Select(
                        id="gateway_id", options=options_stations, value="FGV", required=True
                    ),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Label("Node location", html_for="node_id"),
                    dbc.Select(
                        id="node_id", options=options_stations, value="FGD", required=True
                    ),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Label("Sample resolution", html_for="spm"),
                    dcc.Slider(
                        id="spm",
                        min=0.2,
                        max=1.0,
                        value=0.8,
                        marks={0.2: "20%", 0.6: "60%", 0.99: "100%"}
                    )
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Label("View width", html_for="view_width"),
                    dbc.InputGroup(
                        [
                            dbc.Input(
                                id="view_width",
                                type="number",
                                min=5,
                                max=100,
                                value=50,
                                required=True
                            ),
                            dbc.InputGroupText("m"),
                        ]
                    ),
                ],
                className="mb-3",
            ),
            dbc.Button(
                "Update", id="session_update", type="submit", className="mb-3"
            ),
            dbc.Alert(id="sidebar_error", color="danger", is_open=False),
            dcc.Loading(html.Div(id="data_loading")),
        ],
        className="pt-3",
    )

    fresnel_zone_controls = [
        dbc.Col(
            [
                dbc.Label(
                    "Gateway height",
                    html_for="gateway_offset",
                ),
                dbc.InputGroup(
                    [
                        dbc.Input(
                            id="gateway_offset",
                            type="number",
                            value=12.0,
                        ),
                        dbc.InputGroupText("m"),
                    ]
                ),
            ],
            md=4,
        ),
        dbc.Col(
            [
                dbc.Label(
                    "Node height",
                    html_for="node_offset",
                ),
                dbc.InputGroup(
                    [
                        dbc.Input(
                            id="node_offset",
                            type="number",
                            value=2.0,
                        ),
                        dbc.InputGroupText("m"),
                    ]
                ),
            ],
            md=4,
        ),
        dbc.Col(
            [
                dbc.Label("Frequency", html_for="frequency"),
                dbc.InputGroup(
                    [
                        dbc.Input(
                            id="frequency",
                            type="number",
                            min=0.001,
                            max=10,
                            value=0.868,
                        ),
                        dbc.InputGroupText("GHz"),
                    ]
                ),
            ],
            md=4,
        ),
    ]

    graph_3d_controls = [
        dbc.Col(
            [
                dbc.Label(
                    "View window",
                    html_for="3d_view_window",
                ),
                dbc.InputGroup(
                    [
                        dbc.Input(
                            id="3d_view_window",
                            type="number",
                            min=10,
                            max=5000,
                            value=400,
                        ),
                        dbc.InputGroupText("m"),
                    ]
                ),
            ],
            md=4,
        )
    ]

    container = dbc.Container(
        fluid=True,
        children=dbc.Row(
            [
                dbc.Col(sidebar, md=3, xl=2),
                dbc.Col(
                    md=9,
                    xl=10,
                    children=[
                        html.H5("Session info"),
                        html.Div(
                            [
                                html.Div(
                                    html.Div(
                                        html.Div(
                                            style={
                                                "width": "128px",
                                                "height": "128px",
                                                "background-image": f"url(\"{app.get_asset_url('compass_fg.png')}\")",
                                                "background-size": "cover",
                                            }
                                        ),
                                        id="session_compass",
                                        style={"visibility": "hidden"},
                                    ),
                                    className="me-3",
                                    style={
                                        "width": "128px",
                                        "height": "128px",
                                        "background-image": f"url(\"{app.get_asset_url('compass_bg.png')}\")",
                                        "background-size": "cover",
                                    },
                                ),
                                html.Div(
                                    html.Ul(
                                        [
                                            html.Li(
                                                [
                                                    html.B(
                                                        "Gateway location: "
                                                    ),
                                                    html.Span(
                                                        "N/A",
                                                        id="session_gateway_location",
                                                    ),
                                                ]
                                            ),
                                            html.Li(
                                                [
                                                    html.B(
                                                        "Gateway location: "
                                                    ),
                                                    html.Span(
                                                        "N/A",
                                                        id="session_node_location",
                                                    ),
                                                ]
                                            ),
                                            html.Li(
                                                [
                                                    html.B("Distance: "),
                                                    html.Span(
                                                        "N/A",
                                                        id="session_distance",
                                                    ),
                                                ]
                                            ),
                                            html.Li(
                                                [
                                                    html.B("North azimuth: "),
                                                    html.Span(
                                                        "N/A",
                                                        id="session_azimuth",
                                                    ),
                                                ]
                                            ),
                                        ],
                                        className="list-unstyled",
                                    ),
                                ),
                            ],
                            className="d-flex mb-3",
                        ),
                        html.H5("Fresnel zone"),
                        dbc.Row(fresnel_zone_controls, className="mb-3"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    md=9,
                                    children=[
                                        html.H5("Height curve"),
                                        html.Div(
                                            dcc.Loading(
                                                dcc.Graph(
                                                    id="graph_2d",
                                                    figure=placeholder_figure(
                                                        "", PLOT_HEIGHT_2D
                                                    ),
                                                )
                                            ),
                                            className="border mb-2",
                                        ),
                                        html.P(
                                            [
                                                "Clicked point: ",
                                                html.A(
                                                    "N/A",
                                                    id="link_clicked_point",
                                                    target="_blank",
                                                ),
                                            ]
                                        ),
                                    ],
                                ),
                                dbc.Col(
                                    md=3,
                                    children=[
                                        html.H5("Cross section"),
                                        html.Div(
                                            dcc.Loading(
                                                dcc.Graph(
                                                    id="graph-cross",
                                                    figure=placeholder_figure(
                                                        "", PLOT_HEIGHT_2D
                                                    ),
                                                )
                                            ),
                                            className="border",
                                        ),
                                    ],
                                ),
                            ],
                            className="mb-3",
                        ),
                        html.H5("3D terrain"),
                        dbc.Switch(
                            id="enable_graph_3d",
                            label="Show 3D terrain",
                            value=False,
                        ),
                        dbc.Collapse(
                            [
                                dbc.Row(graph_3d_controls, className="mb-3"),
                                html.Div(
                                    dcc.Loading(dcc.Graph(id="graph_3d")),
                                    className="border",
                                ),
                            ],
                            id="collapse_graph_3d",
                            is_open=False,
                        ),
                    ],
                    className="py-3",
                ),
                html.Footer(
                    html.Small(
                        (
                            "Energinet | LoRaWAN line of sight helper |"
                            f" Version {VERSION}",
                        ),
                        className="text-muted",
                    ),
                    className="border-top py-3",
                ),
            ]
        ),
    )

    layout = html.Div(
        [
            navbar,
            container,
            dcc.Store(id="data", storage_type="memory", clear_data=True),
        ]
    )

    return layout
