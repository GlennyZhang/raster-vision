from rastervision.data.label import Labels

import numpy as np
from rasterio.features import rasterize
import shapely


class SemanticSegmentationLabels(Labels):
    """A set of spatially referenced labels.
    """

    def __init__(self, windows, label_fn, aoi_polygons=None):
        """Constructor

        Args:
        """
        self.windows = windows
        self.label_fn = label_fn
        self.aoi_polygons = aoi_polygons

    def __add__(self, other):
        """Add labels to these labels.

        Returns a concatenation of this and the other labels.
        """
        return SemanticSegmentationLabels(
            self.windows + other.windows,
            self.label_fn,
            aoi_polygons=self.aoi_polygons)

    def filter_by_aoi(self, aoi_polygons):
        return SemanticSegmentationLabels(
            self.windows, self.label_fn, aoi_polygons=aoi_polygons)

    def add_window(self, window):
        self.windows.append(window)

    def get_windows(self):
        return self.windows

    def get_label_arr(self, window, clip_extent=None):
        label_arr = self.label_fn(window)
        window_geom = window.to_shapely()

        if self.aoi_polygons:
            # For each aoi_polygon, intersect with window, and put in window frame of
            # reference.
            window_aois = []
            for aoi in self.aoi_polygons:
                window_aoi = aoi.intersection(window_geom)
                if not window_aoi.is_empty:
                    def transform_shape(x, y, z=None):
                        return (x - window.xmin, y - window.ymin)

                    window_aoi = shapely.ops.transform(transform_shape, window_aoi)
                    window_aois.append(window_aoi)

            if window_aois:
                # Set pixels not in the AOI polygon to 0, so they are ignored during
                # eval.
                mask = rasterize(
                    [(p, 0) for p in window_aois],
                    out_shape=label_arr.shape,
                    fill=1,
                    dtype=np.uint8)
                label_arr[mask] = 0

        if clip_extent is not None:
            clip_window = window.intersection(clip_extent)
            label_arr = label_arr[0:clip_window.get_height(), 0:clip_window.get_width()]

        return label_arr
