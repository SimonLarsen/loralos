import pyproj
from owslib.wcs import WebCoverageService
import rasterio
import os.path
from typing import List


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
        self.cached_image = None

    def load_tile(self, x: float, y: float) -> None:
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
        self.cached_image = self.cached_file.read(1)

    def get_heights(
        self,
        lons: List[float],
        lats: List[float]
    ) -> List[float]:
        points = [None] * len(lons)
        tiles = [None] * len(lons)
        for i in range(len(lons)):
            x, y = self.trans.transform(lons[i], lats[i])
            points[i] = (x, y)

            tx = int(x // self.tile_size * self.tile_size)
            ty = int(y // self.tile_size * self.tile_size)
            tiles[i] = (tx, ty)
        
        order = list(range(len(lons)))
        order.sort(key=lambda i: tiles[i])
    
        prev_tile = None
        out = [None] * len(lons)
        for i in order:
            tile = tiles[i]
            if tile != prev_tile:
                self.load_tile(*tile)
                prev_tile = tile
            
            x, y = points[i]
            row, col = self.cached_file.index(x, y)
            out[i] = self.cached_image[row, col]

        return out