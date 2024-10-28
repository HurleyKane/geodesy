import warnings
import pandas as pd
import numpy as np
from geodesy.dataframe.BaseDataFrame import BaseDataFrame

class TimeSeriesDataFrame(BaseDataFrame):
    neccessary_columns = ["time", ]
    _B = []
    def __init__(
            self,
            data=None,
            index = None,
            columns = None,
            dtype = None,
            copy = False,
    ) -> None:
        super().__init__(data, index, columns, dtype, copy)

    @classmethod
    def read_from_csv(cls, filename, delimiter: str = " ", columns: list = None, header=None, str_col:list=None,
                      skiprows=None, **kwargs):
        dataframe = pd.read_csv(filename, delimiter=delimiter, header=header, skiprows=skiprows, dtype=float,**kwargs)
        dataframe = cls(data=dataframe.values, columns=columns)

        if columns is None:
            columns = list(dataframe.columns)
        dtype_dict = {}
        for column in columns:
            dtype_dict[column] = "float"
        if str_col is not None:
            for col in str_col:
                dtype_dict[columns[col]] = "str"
        for key in dtype_dict.keys():
            dataframe[key] = dataframe[key].astype(dtype_dict[key])
        dataframe.set_index(keys="time", inplace=True)
        dataframe.index = dataframe.index.astype(float)
        return dataframe

    @property
    def B(self):
        """模型系数矩阵"""
        return np.hstack(self._B)

    @property
    def ti(self):
        ti = self.index.values.reshape(-1, 1).astype(float)
        return ti

    def add_time_series_model(self, tg:tuple=None):
        if not isinstance(tg, tuple):
            tg = (tg, )
        B = []
        ti = self.ti
        B.append(np.ones_like(ti))
        B.append(ti)
        B.append(np.sin(2*np.pi*ti))
        B.append(np.cos(2*np.pi*ti))
        B.append(np.sin(4*np.pi*ti))
        B.append(np.cos(4*np.pi*ti))
        self._B = B
    def add_step_term(self, tg:tuple):
        # 阶跃项
        if tg is not None:
            for i in range(len(tg)):
                self._B.append(np.where(self.ti>tg[i], 1, 0))

    def weighted_least_square(self, Y):
        from geodesy.algorithm import weighted_least_square
        return weighted_least_square(self.B, Y)

    """************************************地震类相关分量****************************************************"""
    def add_postseismic_vicoelastic_rheology(self, tk: tuple, tao: float):
        """震后粘弹性体指数模型"""
        ti = self.index.values.reshape(-1, 1).astype(float)
        for i in range(len(tk)):
            self._B.append((-1 * np.exp(-((ti - tk[i]) / tao)) + 1) * np.where(ti > tk[i], 1, 0))

