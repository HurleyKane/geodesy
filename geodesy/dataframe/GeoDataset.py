from __future__ import annotations

import sys
import utm
import pandas as pd
import numpy as np
import logging
from copy import deepcopy
from pyproj import Transformer
from typing import Any
from collections.abc import (
    Hashable,
    Mapping,
    Iterable
)

from tqdm import tqdm
from xarray import Dataset, open_dataset
from xarray.core.types import DataVars
from xarray.core.types import Self
from xarray.plot.accessor import DatasetPlotAccessor
from xarray import load_dataset as ld

import matplotlib.pyplot as plt
from xarray.core.utils import UncachedAccessor
from matplotlib.image import AxesImage

"""自建库"""
from geodesy.dataframe.GeoDataArray import GeoDataArray
from geodesy.dataframe.BaseDataset import BaseDataset
from geodesy.plotting.MatplotToGeoformat import add_shared_colorbar

class GeoDataDatasetPlotAccessor:
    def __init__(self, dataset: GeoDataset) -> None:
        self._ds = dataset
    def imshow_properties(
            self, *args,
            properties : list[str] = None,
            cmap="coolwarm",
            colorbar_name : str = None,
            axes : list[AxesImage] = None,
            add_xticks :bool = True,
            add_yticks :bool = True,
            axis = 0,
            **kwargs
            ):
        if "size" in kwargs:
            size = kwargs["size"]
            del kwargs["size"]
        else:
            size = "3%"
        if "pad" in kwargs:
            pad = kwargs["pad"]
            del kwargs["pad"]
        else:
            pad = 0.005
        if "tick_fontsize" in kwargs:
            tick_fontsize = kwargs["tick_fontsize"]
            del kwargs["tick_fontsize"]
        else:
            tick_fontsize = None
        if "orientation" in kwargs:
            orientation = kwargs["orientation"]
            del kwargs["orientation"]
        else:
            orientation = "vertical"

        objs = []

        if axes is None:
            axes = plt.subplots(1, len(properties))[1]
        for index, property in enumerate(properties):
            temp_obj = self._ds[property].geo_plot.imshow(*args, ax=axes[index], cmap=cmap, add_colorbar=False, **kwargs)
            # 如果不是第一个图，则隐藏 y 轴刻度
            if (index > 0 and axis==0) or add_yticks == False:
                axes[index].set_yticklabels([])  # 隐藏 y 轴刻度标签
                axes[index].set_yticks([])
            if (index < 1 and axis==1 ) or add_xticks == False:
                axes[index].set_xticklabels([])  # 隐藏 x 轴刻度标签
                axes[index].set_xticks([])
            objs.append(temp_obj)
            axes[index].tick_params(axis="x", labelsize=tick_fontsize)
            axes[index].tick_params(axis="y", labelsize=tick_fontsize)
        if colorbar_name is not None:
            add_shared_colorbar(
                temp_obj, axes = axes, label=colorbar_name, size=size,
                pad=pad, orientation=orientation, extend="both")
        plt.subplots_adjust(right=0.85)  # 右侧增加边距
        return objs

class GeoDataset(BaseDataset):
    """以经纬度为单位的二维数据的通用数据类"""
    __slots__ = ()
    necessary_coords = ["clon", "clat"]
    necessary_vars = []
    select_vars = []

    geo_plot = UncachedAccessor(GeoDataDatasetPlotAccessor)

    def __init__(
            self,
            data_vars: DataVars | None = None,
            coords: Mapping[Any, Any] | None = None,
            attrs: Mapping[Any, Any] | None = None,
    ) -> None:
        super().__init__(data_vars, coords, attrs)
        if not self._is_exist_necessary_parameters(self):
            sys.exit()

    def __getitem__(
            self, key: Mapping[Any, Any] | Hashable | Iterable[Hashable]
    ) -> Self | GeoDataArray:
        """
        通过 [] 进行访问获得GeoDataArray
        """
        DataArray = super().__getitem__(key)
        return GeoDataArray(DataArray)

    @classmethod
    def _is_exist_necessary_parameters(cls, obj: GeoDataset) -> bool:
        for coord in tuple(cls.necessary_coords):
            if coord not in list(obj.coords.keys()):
                print(f"coord not in GeoDataset.necessary_coords, including: {cls.necessary_coords} !")
                return False

        if len(cls.necessary_vars) != 0:
            for var in cls.necessary_vars:
                if var not in list(obj.data_vars):
                    print(f"vars not in GeoDataset.necessary_vars, including: {cls.necessary_vars} !")
                    return False
        return True

    @property
    def dataset(self):
        return self

    @property
    def region(self) -> list[float]:
        region = [np.nanmin(self.clon), np.nanmax(self.clon),
                  np.nanmin(self.clat), np.nanmax(self.clat)]
        return region

    @property
    def fitting_accuracy(self) -> float:
        fitting_accuracy = abs(self.clon[1].values - self.clon[0].values)
        return fitting_accuracy

    @classmethod
    def _load_dataset(cls, filename_or_obj, **kwargs) -> GeoDataset:
        return cls(ld(filename_or_obj, **kwargs))

    @classmethod
    def open_dataset(cls, filename:str):
        dataset = open_dataset(filename)
        return cls(dataset)


    @classmethod
    def create_empty_obj(cls, clon, clat, attributes:tuple=("ele",)):
        ds = Dataset()
        ds.assign_coords(clon=("clon", clon), lat=("clat", clat))
        for attribute in attributes:
            ds[attribute] = ("clat", "clon"), np.full((len(clat), len(clon)), np.nan)
        return cls(ds)

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

    @classmethod
    def read_tiff(cls, filename:str):
        from geodesy.io import read_tiff
        return cls(read_tiff(filename))

    @classmethod
    def read_tiffs(cls, filenames:list[str], properties_name:list[list[str]]=None):
        from geodesy.io import read_tiffs
        return cls(read_tiffs(filenames, properties_name=properties_name))

    @classmethod
    def _interp_dataset(cls, dataset:Dataset, region, fitting_accuracy, method="nearest")->GeoDataset:
        """
        method : {"linear", "nearest", "zero", "slinear", "quadratic", "cubic", "polynomial",
                  "barycentric", "krog", "pchip", "spline", "akima"}, default: "linear"
        """
        row = np.arange(region[3], region[2] - fitting_accuracy, -fitting_accuracy)
        col = np.arange(region[0], region[1] + fitting_accuracy, fitting_accuracy)
        interp_dataset = dataset.interp(clon=col, clat=row, method_non_numeric=method)
        return cls(interp_dataset)

    def interp_dataset(self, region, fitting_accuracy, method="nearest"):
        return self._interp_dataset(self, region, fitting_accuracy, method=method)


    @classmethod
    def _downsampling(cls, dataset:GeoDataset, step=2) -> GeoDataset:
        """
        对结果安装经纬度进行排序，并重新初始化
        """
        # 进行等距离的降采样
        print("Data is being downsamping......")
        clon = [i for i in range(0, len(dataset.clon), step)]
        clat = [i for i in range(0, len(dataset.clat), step)]
        downsampling_dataset = dataset.isel(clon=clon, clat=clat)
        return cls(downsampling_dataset)

    def downsampling(self, step=2) -> GeoDataset:
        return self._downsampling(self, step=step)

    def split_block(self, m):
        # 将一个矩阵shape按m行划分成m份,返回每一份的行/列索引
        n = len(self.clat.values)

        # 确保n > m
        if n <= m:
            raise ValueError("n must be greater than m")

        # 确定每块的最小行/列大小
        split_size = n // m
        extra = n % m  # 余数部分

        splits = []
        row_start = 0

        # 按行划分为m份
        for i in range(m):
            row_end = row_start + split_size

            # 若有多余部分，给每个块多加一行，直到余数用完
            if extra > 0:
                row_end += 1
                extra -= 1

            # 添加该份的行范围
            splits.append([0, len(self.clon.values), row_start, row_end])  # [col1, col2, row1, row2]

            # 更新下一块的起始行
            row_start = row_end

        return splits

    def set_dataset_attribute_values(self, attribute:str, row, col, value):
        """更改dataset对应行列的属性的值"""
        self._dataset[attribute][row, col] = value

    @classmethod
    def __quardtree_downsampling(cls, obj:GeoDataset, threshold, according_variable:str, method="default",
                               plt_show=True):
        """对结果记性四叉树下采样
        采样逻辑：对叶节点的区域中心点进行采样
        according_varibable：变量名
        method: default, drop_track
        """
        lon, lat = obj.get_WGS84_coords()
        from geodesy.algorithm import QuadTree
        qt = QuadTree(obj.dataset)
        qt.decompose(threshold, according_variable, method)  # 进行四叉树分割

        if plt_show:
            qt.matplot_img(lon=lon, lat=lat, plt_var=according_variable)
            import matplotlib.pyplot as plt
            plt.show()
        leaf_nodes = qt.visit_leaf_nodes(statistics=True)

        # 提取叶节点中的对应节点的值
        result_dataset = deepcopy(obj.dataset)
        result_dataset[according_variable][:,:] = np.nan
        for node in leaf_nodes:
            ## 采取中心叶节点的中心点进行采样处理
            row = int((node.row_limits[0] + node.row_limits[1])/2)
            column = int((node.column_limits[0] + node.column_limits[1])/2)
            result_dataset[according_variable][dict(clat=row, clon=column)] = \
                obj.dataset[according_variable][dict(clat=row, clon=column)]
        return cls(result_dataset)

    def quardtree_downsampling(self, threshold, according_variable:str, method="default",
                               plt_show=True, **kwargs) -> GeoDataset:
        return self.__quardtree_downsampling(self, threshold, according_variable, method, plt_show)

    def to_InSARData2D(self):
        try:
            from InSARlib import InSARDataset
            return InSARDataset(self.dataset)
        except ImportError as e:
            logging.error(f"ImportError: {e}")
            print(
                "InSARlib is not installed or there is an issue with the version. Please install it using 'pip install InSARlib'.")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            print("An unexpected error occurred. Please check your environment and try again.")
            return None
    def saveNetcdf(self, fnames:str):
        self.dataset.to_netcdf(path=fnames)

    def get_WGS84_coords(self):
        lon, lat = np.meshgrid(self.dataset.clon, self.dataset.clat)
        return lon, lat

    def get_geo_coords(self):
        lon, lat = self.get_WGS84_coords()
        zone = utm.from_latlon(self.dataset.clat[0],
                               self.dataset.clon[0])[2] + 32600
        transformer = Transformer.from_crs("epsg:4326", "epsg:{}".format(zone))
        geo_x, geo_y = transformer.transform(lat, lon)  # self.x点的纵坐标， self.y点的横坐标
        geo_x[np.isinf(geo_x)] = np.nan
        geo_y[np.isinf(geo_y)] = np.nan
        return geo_x, geo_y

    def get_intersection_region(self, region) ->  list[float] | None:
        # 获取两个对象的相交区域
        lon0_1, lon1_1, lat0_1, lat1_1 = self.region
        lon0_2, lon1_2, lat0_2, lat1_2 = region

        # 判断两个区域是否相交
        if (lon0_1 >= lon1_2 or lon0_2 >= lon1_1 or lat0_1 >= lat1_2 or lat0_2 >= lat1_1):
            # 没有相交的情况
            return None
        else:
            # 计算相交区域
            intersection_lon0 = max(lon0_1, lon0_2)
            intersection_lon1 = min(lon1_1, lon1_2)
            intersection_lat0 = max(lat0_1, lat0_2)
            intersection_lat1 = min(lat1_1, lat1_2)

            return [intersection_lon0, intersection_lon1, intersection_lat0, intersection_lat1]

    def subtract_region(self, subtracted_region) -> list[float]:
        # 从 region 中剔除 subtracted_region 部分
        lon0, lon1, lat0, lat1 = self.region
        inter_lon0, inter_lon1, inter_lat0, inter_lat1 = subtracted_region

        # 计算剔除后的区域
        non_intersection_lon0 = lon0 if lon0 >= inter_lon1 else inter_lon1
        non_intersection_lon1 = lon1 if lon1 <= inter_lon0 else inter_lon0
        non_intersection_lat0 = lat0 if lat0 >= inter_lat1 else inter_lat1
        non_intersection_lat1 = lat1 if lat1 <= inter_lat0 else inter_lat0

        return [non_intersection_lon0, non_intersection_lon1, non_intersection_lat0, non_intersection_lat1]

    def calculate_union(self, region) -> list[float]:
        # 取∪逻辑
        union_lon0 = min(self.region[0], region[0])
        union_lon1 = max(self.region[1], region[1])
        union_lat0 = min(self.region[2], region[2])
        union_lat1 = max(self.region[3], region[3])

        return [union_lon0, union_lon1, union_lat0, union_lat1]

    def find_closest_point(self, lon, lat):
        col = np.argmin(np.abs(self.clon.values - lon))
        row = np.argmin(np.abs(self.clat.values - lat))
        return row, col


if __name__ == '__main__':
    import InSARlib.core.config as cf
    import os

    os.chdir(cf.get_rootPath())

    # %% 三维形变数据的读取
    TDDdataFile = "InSARlib/Datas/results/SMVCE_result.tdd"
    TDDcolumns = ["lon", "lat", "ele", "ew_def", "ns_def", "v_def"]
    TDDAarray = GeoDataset.create_from_txt(TDDdataFile, columns=TDDcolumns)

    # %% 创建空数据
    clon = TDDAarray.clon.values
    clat = TDDAarray.clat.values
    attributes = ("strain_ee", "strain_eb")
    TDDEmpty = GeoDataset._create_empty_obj(clon, clat, attributes=attributes)
    GeoDataset._add_dynamic_properties(attribute_names=attributes)
    print(TDDEmpty)


