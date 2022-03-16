from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd


VERSION = "0.0.2"
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


def build_layout(stations: pd.DataFrame):
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
                        id="gateway_id", options=options_stations, value="FGV"
                    ),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Label("Node location", html_for="node_id"),
                    dbc.Select(
                        id="node_id", options=options_stations, value="FGD"
                    ),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Label("Sample resolution", html_for="spm"),
                    dbc.InputGroup(
                        [
                            dbc.Input(
                                id="spm",
                                type="number",
                                min=0.01,
                                max=2.0,
                                value=0.8,
                            ),
                            dbc.InputGroupText("samples/m"),
                        ]
                    ),
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
            sm=4,
            xl=3,
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
            sm=4,
            xl=3,
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
            sm=4,
            xl=3,
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
            sm=4,
            xl=3,
        )
    ]

    container = dbc.Container(
        dbc.Row(
            [
                dbc.Col(sidebar, md=3, lg=2),
                dbc.Col(
                    [
                        html.H5("Session info"),
                        html.Ul(
                            [
                                html.Li(
                                    [
                                        html.B("Distance: "),
                                        html.Span(
                                            "N/A", id="session_distance"
                                        ),
                                    ]
                                ),
                                html.Li(
                                    [
                                        html.B("Azimuth angle: "),
                                        html.Span("N/A", id="session_azimuth"),
                                    ]
                                ),
                            ],
                            className="list-unstyled",
                        ),
                        html.H5("Fresnel zone"),
                        dbc.Row(fresnel_zone_controls, className="mb-3"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
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
                                            className="border",
                                        ),
                                    ],
                                    lg=9,
                                ),
                                dbc.Col(
                                    [
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
                                        html.P(id="test"),
                                    ],
                                    lg=3,
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
                    md=9,
                    lg=10,
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
        fluid=True,
    )

    layout = html.Div(
        [
            navbar,
            container,
            dcc.Store(id="data", storage_type="memory", clear_data=True),
        ]
    )

    return layout
