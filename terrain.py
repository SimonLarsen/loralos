import tempfile
import pyproj
import numpy as np
from wcs_height_map import WCSHeightMap
from wms_image import WMSImage


MAX_DISTANCE = 50000
WMS_CACHE_DIR = tempfile.TemporaryDirectory(prefix="wms_cache_")
WCS_CACHE_DIR = tempfile.TemporaryDirectory(prefix="wcs_cache_")


class DistanceExceededError(Exception):
    pass


def generate_data(config, lon1, lat1, lon2, lat2, spm, view_width):
    geod = pyproj.Geod(ellps="WGS84")
    azi12, azi21, dist = geod.inv(lon1, lat1, lon2, lat2)
    if dist > MAX_DISTANCE:
        raise DistanceExceededError

    heightmap = WCSHeightMap(
        url=config["heightmap"]["url"],
        token=config["heightmap"]["token"],
        layer=config["heightmap"]["layer"],
        tile_size=int(config["heightmap"]["tile_size"]),
        resolution=int(config["heightmap"]["resolution"]),
        cache_dir=WCS_CACHE_DIR.name,
    )

    image = WMSImage(
        url=config["image"]["url"],
        token=config["image"]["token"],
        layer=config["image"]["layer"],
        tile_size=int(config["image"]["tile_size"]),
        resolution=int(config["image"]["resolution"]),
        cache_dir=WMS_CACHE_DIR.name,
    )

    # Number of samples from gateway to node
    npts_x = round(dist * spm)
    # Number of samples across (always odd)
    npts_y = int(round(view_width * spm * 0.5) * 2 + 1)

    out_lon = []
    out_lat = []
    out_offset = []

    inter = geod.inv_intermediate(
        lon1, lat1, lon2, lat2, npts=npts_x, initial_idx=0, terminus_idx=0
    )

    for ilon, ilat in zip(inter.lons, inter.lats):
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
        lon1=lon1,
        lat1=lat1,
        lon2=lon2,
        lat2=lat2,
        spm=spm,
        view_width=view_width,
        azi12=azi12,
        azi21=azi21,
        dist=dist,
        npts_x=npts_x,
        npts_y=npts_y,
        offset=out_offset,
        height=out_height,
        color=out_color,
        height_start=height_start,
        height_end=height_end,
    )
