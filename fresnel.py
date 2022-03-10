import numpy as np


SPEED_OF_LIGHT = 299792458


def fresnel_zone_radius(d1: float, d2: float, f: float, n: int = 1) -> float:
    """
    Computes Fresnel zone radius at a specific point between two antennas.

    Parameters
    ----------
    d1 : float
        Distance from first antenna to point in meters.
    d2 : float
        Distance from second antenna to point in meters.
    f : float
        Frequency in GHz.
    
    Returns
    -------
    float
        Radius in meters.
    """
    assert n == 1, "Only first Fresnel supported for now."
    d1km = d1 / 1000
    d2km = d2 / 1000
    return np.sqrt(SPEED_OF_LIGHT * d1km * d2km / ((d1km + d2km) * f)) / 1000
