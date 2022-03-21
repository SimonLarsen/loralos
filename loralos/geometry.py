import numpy as np


def plane_mesh_indices(npts_x: int, npts_y: int):
    """
    Generate vertex indices for a triangle plane mesh.

    Parameters
    ----------
    npts_x : int
        Number of vertices on major axis.
    npts_y : int
        Number of vertices on minor axis.

    Returns
    -------
    3-tuple of lists of integers
        Vertex indices for triangles.
    """
    v_x = np.repeat(np.arange(npts_x - 1), npts_y - 1)
    v_y = np.tile(np.arange(npts_y - 1), npts_x - 1)
    v = v_x * npts_y + v_y

    t1 = np.concatenate((v, v))
    t2 = np.concatenate((v + npts_y, v + npts_y + 1))
    t3 = np.concatenate((v + npts_y + 1, v + 1))
    return t1, t2, t3


def tube_mesh_indices(npts_x: int, npts_y: int):
    """
    Generate vertex indices for tube mesh.

    Parameters
    ----------
    npts_x : int
        Number of vertices along tube length (major axis).
    npts_y : int
        Number of vertices around tube circumference (minor axis).

    Returns
    -------
    3-tuple of lists of integers.
        Vertex indices for triangles.
    """
    v_x = np.repeat(np.arange(npts_x - 1), npts_y)
    v_y = np.tile(np.arange(npts_y), npts_x - 1)
    v = v_x * npts_y + v_y
    v1 = v_x * npts_y + (v_y + 1) % npts_y

    t1 = np.concatenate((v, v))
    t2 = np.concatenate((v1, v1 + npts_y))
    t3 = np.concatenate((v1 + npts_y, v + npts_y))
    return t1, t2, t3
