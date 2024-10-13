import numpy as np
import pandas as pd
from geodesy.dataframe.InSARDataset import InSARDataset
from geodesy.dataframe.GeoDataFrame import GeoDataFrame

class InSARDataFrame(GeoDataFrame):
    __slots__ = ()
    def __init__(self, data:np.ndarray or pd.DataFrame=None, columns:list=None):
        if columns is None:
            self._columns = ["lon", "lat", "ele", "look_e", "look_n", "look_u", "defor"]
        else:
            self._columns = list(columns)
        super().__init__(data=data, columns=self.columns)

    @property
    def columns(self):
        return self._columns

    def to_InSARData2D(self):
        InSARDataset.creat_from_numpy_or_pandas(data=self.dataframe.values, columns=self.columns)
