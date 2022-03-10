import pyproj
from owslib.wcs import WebCoverageService
import rasterio
import os.path


FORMAT_ENDINGS = {
    "GTiff": "tif"
}


class WCSHeightMap:
    def __init__(
        self,
        url: str,
        layer: str,
        tile_size: int = 1000,
        resolution: int = 500,
        format: str = "GTiff",
        crs: str = "epsg:25832"
    ):
        self.url = url
        self.layer = layer
        self.tile_size = tile_size
        self.resolution = resolution
        self.format = format
        self.crs = crs

        self.wcs = WebCoverageService(self.url)
        self.trans = pyproj.Transformer.from_crs(
            "wgs84",
            self.crs,
            always_xy=True
        )
        self.cached_file = None
        self.cached_bbox = None
        self.band = None

    def point_in_bounding_box(self, x: float, y: float) -> bool:
        return (
            self.cached_bbox is not None
            and x >= self.cached_bbox[0]
            and x <= self.cached_bbox[2]
            and y >= self.cached_bbox[1]
            and y <= self.cached_bbox[3]
        )

    def load_tile(self, x: float, y: float) -> None:
        if self.point_in_bounding_box(x, y):
            return

        tx = int(x // self.tile_size) * self.tile_size
        ty = int(y // self.tile_size) * self.tile_size

        bbox = (tx, ty, tx + self.tile_size, ty + self.tile_size)

        cache_file = f"mapcache/wcs_{self.layer}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}_{self.tile_size}_{self.resolution}.{FORMAT_ENDINGS[self.format]}"

        if not os.path.exists(cache_file):
            res = self.wcs.getCoverage(
                identifier=self.layer,
                crs=self.crs,
                bbox=bbox,
                format=self.format,
                width=self.resolution,
                height=self.resolution,
            )
            data = res.read()

            with open(cache_file, "wb") as fp:
                fp.write(data)

        self.cached_file = rasterio.open(cache_file)
        self.band = self.cached_file.read(1)
        self.cached_bbox = bbox

    def get_height(self, lon: float, lat: float) -> float:
        x, y = self.trans.transform(lon, lat)
        self.load_tile(x, y)

        row, col = self.cached_file.index(x, y)
        return self.band[row, col]
