from __future__ import annotations

import warnings
# 忽略 FutureWarning 类型的警告
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
from typing import Any
from collections.abc import (
    Mapping,
)

import pandas as pd
import numpy as np
from xarray import Dataset
from xarray.core.types import DataVars
from xarray.core.utils import UncachedAccessor
from xarray.plot.accessor import DatasetPlotAccessor

class BaseDataset(Dataset):
    """
    A class to store time series data in xarray Dataset format.
    """
    plot = UncachedAccessor(DatasetPlotAccessor)

    def __init__(
            self,
            data_vars: DataVars | None = None,
            coords: Mapping[Any, Any] | None = None,
            attrs: Mapping[Any, Any] | None = None,
    ) -> None:
        super().__init__(data_vars, coords, attrs)

    def save_to_txt(self, filename, fmt="%.6f"):
        results = []
        lon, lat = self.get_WGS84_coords()
        results.append(lon.ravel())
        results.append(lat.ravel())
        for var in self.data_vars.keys():
            temp_var = self[var].values.ravel()
            results.append(temp_var)
        results = np.stack(results).T
        with open(filename, "w") as f:
            np.savetxt(f, results, fmt=fmt, delimiter=" ")

