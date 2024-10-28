from __future__ import annotations

import sys
from geodesy.dataframe.BaseDataset import BaseDataset

class TDDTimeSeriesDataset(BaseDataset):
    neccessary_columns = ["time", "ew_def", "ns_def", "v_def"]
    __slots__ = tuple(neccessary_columns) + BaseDataset.__slots__
    def __init__(
            self,
            data=None,
            index = None,
            columns = None,
            dtype = None,
            copy = False,
            **kwargs
    ) -> None:
        super().__init__(data, index, columns, dtype, copy)
        if columns is None and self.shape[1] >= 2:
            columns_list = self.columns.tolist()
            self.columns = columns_list
        if not self.is_exit_necessary_columns(self.columns):
            sys.exit()

