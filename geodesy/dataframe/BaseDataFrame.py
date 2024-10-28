from __future__ import annotations

import warnings

# 忽略 FutureWarning 类型的警告
warnings.filterwarnings("ignore", category=FutureWarning)

import sys

import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from pandas._typing import Axes, IndexLabel, Axis, Level, IgnoreRaise
from pandas.plotting import PlotAccessor

from geodesy.dataframe.DataFrameCachedAccessor import DataFrameCachedAccessor

class BaseDataFrame(DataFrame):
    neccessary_columns = None
    def __init__(
            self,
            data=None,
            index = None,
            columns = None,
            dtype = None,
            copy = False,
    ) -> None:
        super().__init__(data, index, columns=columns,dtype=dtype, copy=copy)
        if not self._is_exist_necessary_parameters(self):
            sys.exit()

    def plot(self, *args, **kwargs):
        return super().plot(*args, **kwargs)

    @property
    def dataframe(self):
        return self

    @classmethod
    def _is_exist_necessary_parameters(cls, obj:BaseDataFrame) -> bool:
        if cls.neccessary_columns is None:
            return True
        for coord in tuple(cls.neccessary_columns):
            if coord not in list(obj.columns):
                print(f"coord not in Base.necessary_columns, including: {cls.neccessary_columns} !")
                return False
        return True

    @classmethod
    def read_from_csv(cls, filename, delimiter: str = " ", columns: list = None, header=None):
        dataframe = pd.read_csv(filename, delimiter=delimiter, header=header)
        return cls(data=dataframe.values, columns=columns)

    """*****************************返回类类型方法**********************************************************"""
    @classmethod
    def _drop(
            cls,
            labels: IndexLabel | None = None,
            *,
            axis: Axis = 0,
            index: IndexLabel | None = None,
            columns: IndexLabel | None = None,
            level: Level | None = None,
            inplace: bool = False,
            errors: IgnoreRaise = "raise",
    ) -> BaseDataFrame | None:
        """
        删除行：drop(index)
        删除列：drop(df.columns[0], axis=1)
        """
        obj = super().drop(labels, axis=axis, index=index, columns=columns, level=level, inplace=inplace, errors=errors)
        if inplace:
            return None
        else:
            return BaseDataFrame(obj)

    def drop(
            self,
            labels: IndexLabel | None = None,
            *,
            axis: Axis = 0,
            index: IndexLabel | None = None,
            columns: IndexLabel | None = None,
            level: Level | None = None,
            inplace: bool = False,
            errors: IgnoreRaise = "raise",
    ) -> BaseDataFrame | None:
        """
        删除行：drop(index)
        删除列：drop(df.columns[0], axis=1)
        """
        return self._drop(labels, axis=axis, index=index, columns=columns, level=level, inplace=inplace, errors=errors)

    @classmethod
    def _filter_data_nan(cls, dataframe:DataFrame, columnNum=3):
        """
        根据对应进行None值的筛选
        """
        sub = np.argwhere(dataframe.values[:, columnNum] != dataframe.values[:, columnNum])
        if len(sub) != 0:
            Onedb = dataframe.drop(index=sub)
            return cls(Onedb)
        else:
            return cls(dataframe)

    def filter_data_nan(self, columnNum=3):
        return self._filter_data_nan(self.dataframe, columnNum)

    """**************************************通用方法***************************************************************"""
    def append(self, row:list):
        """在GPS数据中加入一行"""
        self._dataframe.loc[len(self.dataframe.index)] = row

    def show(self):
        # times new roman
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.show(block=True)