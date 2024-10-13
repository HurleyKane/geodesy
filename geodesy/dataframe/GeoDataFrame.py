from __future__ import annotations
import sys

import utm
import numpy as np
import pandas as pd
from pyproj import Transformer
from pandas import DataFrame
from pandas._typing import Axes, Dtype, IndexLabel, Axis, Level, IgnoreRaise
from pandas.plotting import PlotAccessor

from geodesy.dataframe.DataFrameCachedAccessor import DataFrameCachedAccessor

# noinspection PyCompatibility
class GeoDataFrame(DataFrame):
    __slots__ = ()
    # Add plotting methods to GeoDataFrame
    plot = DataFrameCachedAccessor("plot", PlotAccessor)
    neccessary_columns = ["lon", "lat"]
    def __init__(
            self,
            data=None,
            index = None,
            columns = None,
            dtype = None,
            copy = False,
    ) -> None:
        super().__init__(data, index, columns, dtype, copy)
        if columns is None and self.shape[1] >= 2:
            columns_list = self.columns.tolist()
            columns_list[0] = "lon"
            columns_list[1] = "lat"
            self.columns = columns_list
        if not self.is_exit_necessary_columns(self.columns):
            sys.exit()

    @staticmethod
    def is_exit_necessary_columns(columns: Axes | None) -> bool:
        # 判断columns是否在必要的列
        for column in GeoDataFrame.neccessary_columns:
            if column not in columns:
                print(f"columns not in GeoDataFrame.neccessary_columns, including: {GeoDataFrame.neccessary_columns} !")
                return False
        return True


    @property
    def dataframe(self):
        return self

    @classmethod
    def read_from_csv(cls, filename, delimiter: str = " ", columns: list = None, header=None):
        dataframe = pd.read_csv(filename, delimiter=delimiter, header=header)
        if columns:
            dataframe.columns = columns
        return cls(data=dataframe, columns=columns)

    """*****************************返回类类型方法**********************************************************"""
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
    ) -> GeoDataFrame | None:
        """
        删除行：drop(index)
        删除列：drop(df.columns[0], axis=1)
        """
        obj = super().drop(labels, axis=axis, index=index, columns=columns, level=level, inplace=inplace, errors=errors)
        if inplace:
            return None
        else:
            return GeoDataFrame(obj)

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

    @classmethod
    def _extract_extra_data(cls, dataframe:DataFrame, region:list) -> GeoDataFrame:
        left, right, bottom, top = region
        lon_subs, = np.where((dataframe.dataframe.lon>left)&(dataframe.dataframe.lon<right))
        lat_subs, = np.where((dataframe.dataframe.lat>bottom)&(dataframe.dataframe.lat<top))
        subs = np.intersect1d(lon_subs, lat_subs)
        return cls(data=dataframe.dataframe.values[subs, :])

    def extract_area_data(self, region):
        # 对region范围内的点进行提取
        return self._extract_extra_data(self.dataframe, region)

    """**************************************通用方法***************************************************************"""
    def append(self, row:list):
        """在GPS数据中加入一行"""
        self._dataframe.loc[len(self.dataframe.index)] = row

    def get_region(self):
        return [np.min(self.dataframe.lon), np.max(self.dataframe.lon), np.min(self.dataframe.lat), np.max(self.dataframe.lat)]

    def get_geodetic_coordinate(self):
        """
        有关WGS84和UTM坐标系的转化
        """
        zone = utm.from_latlon(self.lat.values[0], self.lon.values[0])[2] + 32600
        transformer = Transformer.from_crs("epsg:4326", "epsg:{}".format(zone))
        geo_x, geo_y = transformer.transform(self.dataframe.lat, self.dataframe.lon)
        return geo_x, geo_y

    def find_neighbor_points(self, point, number=6):
        """
        point: 输入点的x, y的坐标
        number: 寻找最临近6个点
        """
        d = np.sqrt((self.dataframe.lon-point[0])**2 + (self.dataframe.lat-point[1])**2)
        # 找出距离最近的几个点的下标
        sub = np.argsort(d)[0:number]
        return sub