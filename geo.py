import numpy as np
import pyproj
from typing import Tuple


def geo_radius(lat: float, geod: pyproj.Geod = pyproj.Geod(ellps="WGS84")):
    """
    Compute geocentric radius.
    See: https://en.wikipedia.org/wiki/Earth_radius#Geocentric_radius.

    Parameters
    ----------
    lat : float
        Latitude in degrees.
    ellps : str
        Ellipsis model.

    Returns
    -------
    float
        Earth radius in meters.
    """
    a, b = geod.a, geod.b

    phi = np.deg2rad(lat)
    return np.sqrt(
        ((a**2 * np.cos(phi)) ** 2 + (b**2 * np.sin(phi)) ** 2)
        / ((a * np.cos(phi)) ** 2 + (b * np.sin(phi)) ** 2)
    )


def geo_to_ecef(
    lon: float,
    lat: float,
    h: float,
    geod: pyproj.Geod = pyproj.Geod(ellps="WGS84"),
) -> Tuple[float, float, float]:
    """
    Convert lon/lat/height to ECEF coordinates.
    See: https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates.  # noqa: W505

    Parameters
    ----------
    lon : float
        Longitude in degrees.
    lat : float
        Latitude in degrees.
    h : float
        Height above geocentric radius.
    ellps : str
        Ellipsis model.

    Returns
    -------
    x, y, z
        ECEF coordinates.
    """
    a, b = geod.a, geod.b

    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon)
    e2 = 1.0 - b**2 / a**2

    N = a / np.sqrt(1.0 - e2 * (np.sin(phi)) ** 2)
    x = (N + h) * np.cos(phi) * np.cos(lam)
    y = (N + h) * np.cos(phi) * np.sin(lam)
    z = ((b**2 / a**2) * N + h) * np.sin(phi)
    return np.array((x, y, z))


def ecef_to_geo(
    x: float,
    y: float,
    z: float,
    geod: pyproj.Geod = pyproj.Geod(ellps="WGS84"),
) -> Tuple[float, float]:
    """
    Convert ECEF coordinates to lon/lat/height.
    See: https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#The_application_of_Ferrari%27s_solution.  # noqa: W505

    Parameters
    ----------
    x : float
        X-coordinate
    y : float
        Y-coordinate
    z : float
        Z-coordinate

    Returns
    -------
    float, float, float
        longitude, latitude, height
    """
    a, b = geod.a, geod.b
    e2 = (a**2 - b**2) / a**2
    ep2 = (a**2 - b**2) / b**2
    e4 = e2**2
    p = np.sqrt(x**2 + y**2)
    F = 54.0 * b**2 * z**2
    G = p**2 + (1 - e2) * z**2 - e2 * (a**2 - b**2)
    c = (e4 * F * p**2) / (G**3)
    s = np.power(1 + c + np.sqrt(c**2 + 2 * c), 1 / 3)
    k = s + 1 + 1 / s
    P = F / (3 * k**2 * G**2)
    Q = np.sqrt(1 + 2 * e4 * P)
    r0 = ((-P * e2 * p) / (1 + Q)) + np.sqrt(
        1 / 2 * a**2 * (1 + 1 / Q)
        - (P * (1 - e2) * z**2) / (Q * (1 + Q))
        - 1 / 2 * P * p**2
    )
    U = np.sqrt((p - e2 * r0) ** 2 + z**2)
    V = np.sqrt((p - e2 * r0) ** 2 + (1 - e2) * z**2)
    z0 = (b**2 * z) / (a * V)
    h = U * (1 - (b**2) / (a * V))
    phi = np.arctan((z + ep2 * z0) / p)
    lam = np.arctan2(y, x)
    return np.array((np.rad2deg(lam), np.rad2deg(phi), h))


def los_height_range(
    lon1: float,
    lat1: float,
    height1: float,
    lon2: float,
    lat2: float,
    height2: float,
    npts: int,
    geod: pyproj.Geod = pyproj.Geod(ellps="WGS84"),
):
    """
    Compute line of sight height for points between to coordinates.

    Parameters
    ----------
    lon1 : float
        Longitude for initial point.
    lat1 : float
        Latitude of initial point.
    height1 : float
        Height above sea level of initial point.
    lon2 : float
        Longitude of terminus point.
    lat2 : float
        Latitude of terminus point.
    height2 : float
        Height above sea level of terminus point.
    npts : int
        Number of points along line to generate.
    geod : pyproj.Geod
        Ellipsis model.

    Returns
    -------
    List of floats
        Height above sea level for each point in line of sight.
    """
    p1 = geo_to_ecef(lon1, lat1, height1, geod)
    p2 = geo_to_ecef(lon2, lat2, height2, geod)

    x = np.linspace(p1[0], p2[0], npts)
    y = np.linspace(p1[1], p2[1], npts)
    z = np.linspace(p1[2], p2[2], npts)

    los_radius = np.sqrt(x**2 + y**2 + z**2)

    lons, lats, heights = ecef_to_geo(x, y, z, geod)
    r = [geo_radius(lat, geod) for lat in lats]

    los_heights = los_radius - r
    return los_heights


def los_height_single(
    lon1: float,
    lat1: float,
    height1: float,
    lon2: float,
    lat2: float,
    height2: float,
    t: float,
    geod: pyproj.Geod = pyproj.Geod(ellps="WGS84"),
):
    """
    Compute line of sight height for point between two coordinates.

    Parameters
    ----------
    lon1 : float
        Longitude for initial point.
    lat1 : float
        Latitude of initial point.
    height1 : float
        Height above sea level of initial point.
    lon2 : float
        Longitude of terminus point.
    lat2 : float
        Latitude of terminus point.
    height2 : float
        Height above sea level of terminus point.
    t : float
        Interpolation value between initial and terminus points.
        Must be value between 0.0 and 1.0.
    geod : pyproj.Geod
        Ellipsis model.

    Returns
    -------
    List of floats
        Height above sea level for each point in line of sight.
    """
    p1 = geo_to_ecef(lon1, lat1, height1, geod)
    p2 = geo_to_ecef(lon2, lat2, height2, geod)

    p = t * p2 + (1.0 - t) * p1
    los_radius = np.sqrt(p.dot(p))

    lon, lat, height = ecef_to_geo(*p, geod)
    r = geo_radius(lat, geod)

    return los_radius - r
