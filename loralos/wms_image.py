from owslib.wms import WebMapService
import pyproj
from PIL import Image
from typing import Tuple, List, Dict, Any
import os.path
from pathlib import Path


FORMAT_ENDINGS = {"image/jpeg": "jpg"}


class WMSImage:
    def __init__(
        self,
        url: str,
        layer: str,
        cache_dir: str,
        style: str = "default",
        tile_size: int = 1000,
        resolution: int = 500,
        format: str = "image/jpeg",
        crs: str = "epsg:25832",
        headers: Dict[str, Any] = None,
    ) -> None:
        self.url = url
        self.layer = layer
        self.cache_dir = cache_dir
        self.style = style
        self.tile_size = tile_size
        self.resolution = resolution
        self.format = format
        self.crs = crs
        self.headers = headers

        self.wms = WebMapService(self.url, headers=self.headers)
        self.trans = pyproj.Transformer.from_crs(
            "wgs84", self.crs, always_xy=True
        )
        self.cached_image = None

    def load_tile(self, x: float, y: float) -> None:
        tx = int(x // self.tile_size * self.tile_size)
        ty = int(y // self.tile_size * self.tile_size)
        bbox = (tx, ty, tx + self.tile_size, ty + self.tile_size)

        cache_file = Path(self.cache_dir) / (
            f"wms_{self.layer}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
            f"_{self.tile_size}_{self.resolution}"
            f".{FORMAT_ENDINGS[self.format]}"
        )
        if not os.path.exists(cache_file):
            res = self.wms.getmap(
                layers=[self.layer],
                styles=[self.style],
                srs=self.crs,
                bbox=bbox,
                size=(self.resolution, self.resolution),
                format=self.format,
            )
            with open(cache_file, "wb") as fp:
                fp.write(res.read())

        image = Image.open(cache_file)
        self.cached_image = image.load()

    def get_pixels(
        self, lons: List[float], lats: List[float]
    ) -> List[Tuple[float, float, float]]:
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
            px = round((x - tile[0]) / self.tile_size * (self.resolution - 1))
            py = round(
                (1.0 - (y - tile[1]) / self.tile_size) * (self.resolution - 1)
            )
            out[i] = self.cached_image[px, py]

        return out
