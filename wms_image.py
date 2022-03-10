from owslib.wms import WebMapService
import pyproj
from PIL import Image
from typing import Tuple
import os.path


FORMAT_ENDINGS = {
    "image/jpeg": "jpg"
}


class WMSImage:
    def __init__(
        self,
        url: str,
        layer: str,
        style: str = "default",
        tile_size: int = 1000,
        resolution: int = 500,
        format: str = "image/jpeg",
        crs: str = "epsg:25832"
    ) -> None:
        self.url = url
        self.layer = layer
        self.style = style
        self.tile_size = tile_size
        self.resolution = resolution
        self.format = format
        self.crs = crs

        self.wms = WebMapService(self.url)
        self.trans = pyproj.Transformer.from_crs(
            "wgs84",
            self.crs,
            always_xy=True
        )
        self.cached_bbox = None
        self.image = None
    
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

        cache_file = f"mapcache/wms_{self.layer}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}_{self.tile_size}_{self.resolution}.{FORMAT_ENDINGS[self.format]}"

        if not os.path.exists(cache_file):
            res = self.wms.getmap(
                layers=[self.layer],
                styles=[self.style],
                srs=self.crs,
                bbox=bbox,
                size=(self.resolution, self.resolution),
                format=self.format
            )
            with open(cache_file, "wb") as fp:
                fp.write(res.read())

        image = Image.open(cache_file)
        self.image = image.load()
        self.cached_bbox = bbox

    def get_pixel(self, lon: float, lat: float) -> Tuple[float, float, float]:
        x, y = self.trans.transform(lon, lat)
        self.load_tile(x, y)

        bbox = self.cached_bbox
        px = round((x - bbox[0]) / self.tile_size * (self.resolution - 1))
        py = round((1.0 - (y - bbox[1]) / self.tile_size) * (self.resolution - 1))
        return self.image[px, py]
