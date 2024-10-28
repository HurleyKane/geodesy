from __future__ import annotations

import sys
from typing import Any
from collections.abc import (
    Mapping,
)

import numpy as np
import pandas as pd
from xarray.core.types import DataVars

from geodesy.dataframe.BaseDataset import BaseDataset

class TimeSeriesDataset(BaseDataset):
    """
    A class to store time series data in xarray Dataset format.
    """
    __slots__ = ()
    necessary_coords = ["clon", "clat"]
    def __init__(
            self,
            data_vars: DataVars | None = None,
            coords: Mapping[Any, Any] | None = None,
            attrs: Mapping[Any, Any] | None = None,
    ) -> None:
        super().__init__(data_vars, coords, attrs)
        if not self._is_exist_necessary_parameters(self):
            sys.exit()
    @classmethod
    def _create_from_txt(cls, filename: str, columns: list = None, delimeter=" "):
        df = pd.read_csv(filename, sep=delimeter, header=None)
        return cls.creat_from_numpy_or_pandas(df, columns)

    @staticmethod
    def create_from_txt(filename: str, columns: list = None, delimeter=" ") -> TimeSeriesDataset:
        return TimeSeriesDataset._create_from_txt(filename, columns, delimeter)

    @classmethod
    def creat_from_numpy_or_pandas(cls, data: pd.DataFrame or np.ndarray, columns:list | None= None):
        if type(data) == np.ndarray:
            data = pd.DataFrame(data)
        if columns is None:
            columns = ["lon", "lat"] + [str(i) for i in range(data.values.shape[1] - 2)]
        elif len(columns) < data.shape[1]:
            columns += [str(i) for i in range(data.shape[1] - len(columns))]
        elif len(columns) > data.shape[1]:
            columns = columns[:data.shape[1]]
        data.columns = np.array(columns)
        data = data.sort_values(by=["lat", "lon"], ascending=[False, True])
        df2 = data.set_index(["lon", "lat"], drop=False).unstack(level=0).loc[::-1, :]

        # 定义dataset
        data_vars = {}
        for i in range(0, len(data.columns)):
            if data.columns[i] == "lon" or data.columns[i] == "lat":
                continue
            if data.columns[i] != "error":
                data_vars[data.columns[i]] = (["clat", "clon"], df2[data.columns[i]])
        dataset = Dataset(data_vars=data_vars,
                          coords={
                              "clon": (["clon"], df2.lon.strike_slip_segments_columns.values),
                              "clat": (["clat"], df2.lon.index.values)
                          }
                          )
        return cls(dataset)

    @staticmethod
    def create_from_txt(filename:str, columns:list=None, delimeter=" "):
        df = pd.read_csv(filename, sep=delimeter, header=None)
        return GeoDataset.creat_from_numpy_or_pandas(df, columns)
