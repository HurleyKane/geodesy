from __future__ import annotations
import sys

import utm
import numpy as np
from pyproj import Transformer
from pandas.plotting import PlotAccessor

from geodesy.dataframe.DataFrameCachedAccessor import DataFrameCachedAccessor
from geodesy.dataframe.BaseDataFrame import BaseDataFrame

# noinspection PyCompatibility
class GeoDataFrame(BaseDataFrame):
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


    """*****************************返回类类型方法**********************************************************"""
    def filter_data_nan(self, columnNum=3):
        return self._filter_data_nan(self.dataframe, columnNum)

    @classmethod
    def _extract_extra_data(cls, dataframe:BaseDataFrame, region:list) -> GeoDataFrame:
        left, right, bottom, top = region
        lon_subs, = np.where((dataframe.dataframe.lon>left)&(dataframe.dataframe.lon<right))
        lat_subs, = np.where((dataframe.dataframe.lat>bottom)&(dataframe.dataframe.lat<top))
        subs = np.intersect1d(lon_subs, lat_subs)
        return cls(data=dataframe.dataframe.values[subs, :])

    def extract_area_data(self, region):
        # 对region范围内的点进行提取
        return self._extract_extra_data(self.dataframe, region)

    """**************************************通用方法***************************************************************"""
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