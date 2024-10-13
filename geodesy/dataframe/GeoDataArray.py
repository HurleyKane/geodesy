from __future__ import annotations

import pandas as pd
import numpy as np

from typing import Any
from collections.abc import (
    Hashable,
    Mapping,
    Sequence,
)
from xarray.core import dtypes
from xarray import DataArray
from xarray.core.indexes import Index

from xarray.plot.accessor import DataArrayPlotAccessor
from xarray.core.utils import UncachedAccessor
from matplotlib.image import AxesImage
from xarray.plot.facetgrid import FacetGrid
from geodesy.plotting import MatplotToGeoformat as MTG

import matplotlib.pyplot as plt
# from
class GeoDataArrayPlotAccessor:
    def __init__(self, darray: GeoDataArray) -> None:
        self._da = darray

    def imshow(self, *args, cmap="coolwarm", label_fontsize=None, **kwargs) -> AxesImage | FacetGrid[DataArray]:
        obj = self._da.plot.imshow(*args, cmap=cmap, **kwargs)
        # 检查子图的数量
        try:
            num_subplots = len(obj.axes.flat)  # 获取子图数量
            # 批量处理每个子图
            for i, ax in enumerate(obj.axes.flat):
                MTG.ticks_transform_to_lat_lon(ax=ax)
                ax.set_aspect("equal")
                ax.set_xlabel("")
                ax.set_ylabel("")
                if label_fontsize is not None:
                    ax.tick_params(labelsize=label_fontsize)
        except:
            if "ax" in kwargs:
                if kwargs["ax"] is not None:
                    ax = kwargs["ax"]
                else:
                    ax = plt.gca()
            else:
                ax = plt.gca()
            MTG.ticks_transform_to_lat_lon(ax=ax)
            ax.set_aspect("equal")
            ax.set_xlabel("")
            ax.set_ylabel("")
            if label_fontsize is not None:
                ax.tick_params(labelsize=label_fontsize)
        return obj

class GeoDataArray(DataArray):
    __slots__ = ()
    plot = UncachedAccessor(DataArrayPlotAccessor)
    geo_plot = UncachedAccessor(GeoDataArrayPlotAccessor)
    def __init__(
            self,
            data: Any = dtypes.NA,
            coords: Sequence[Sequence[Any] | pd.Index | DataArray]
                    | Mapping[Any, Any]
                    | None = None,
            dims: Hashable | Sequence[Hashable] | None = None,
            name: Hashable | None = None,
            attrs: Mapping | None = None,
            # internal parameters
            indexes: Mapping[Any, Index] | None = None,
            fastpath: bool = False,
    ) -> None:
        super().__init__(data, coords, dims, name, attrs, indexes, fastpath)

    @property
    def region(self) -> list[float]:
        region = [np.nanmin(self.clon), np.nanmax(self.clon),
                  np.nanmin(self.clat), np.nanmax(self.clat)]
        return region

    @property
    def fitting_accuracy(self) -> float:
        fitting_accuracy = abs(self.clon[1].values - self.clon[0].values)
        return fitting_accuracy
