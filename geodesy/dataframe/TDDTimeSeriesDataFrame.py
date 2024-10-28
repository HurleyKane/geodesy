from __future__ import annotations
import sys
import numpy as np
from geodesy.dataframe.TimeSeriesDataFrame import TimeSeriesDataFrame

class TDDTimeSeriesDataFrame(TimeSeriesDataFrame):
    neccessary_columns = TimeSeriesDataFrame.neccessary_columns + ["ew_def", "ns_def", "v_def"]
    def __init__(
            self,
            data=None,
            index = None,
            columns = None,
            dtype = None,
            copy = False,
    ) -> None:
        super().__init__(data, index, columns, dtype, copy)