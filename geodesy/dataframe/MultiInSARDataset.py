from __future__ import annotations
import sys

import matplotlib.pyplot as plt
from xarray import load_dataset as ld
# 第三方库
import numpy as np
import xarray as xr
from tqdm import tqdm

from xarray.core.types import DataVars
from collections.abc import Mapping
from typing import Any

# 自建库
from geodesy.dataframe.InSARDataset import InSARDataset


def interp(databases, coordinates, fitting_accuracy=None):
    """
    确定研究区域内参与计算的点的值
    如果几个文件的坐标不统一则需要进行坐标的统一工作
    parameters
    ___________
    method : {"linear", "nearest", "zero", "slinear", "quadratic", "cubic", "polynomial", \
        "barycentric", "krog", "pchip", "spline", "akima"}, default: "linear"
    """
    from geodesy.dataframe.InSARDataset import InSARDataset
    interp_databases = []
    for i in tqdm(range(len(databases)), desc="aligning dataset", ncols=100):
        interp_databases.append(
            InSARDataset.__interp_database(databases[i].dataset, coordinates=coordinates, fitting_accuracy=fitting_accuracy))
    return MultiInSARDataset.create_from_databases(datasets=interp_databases)

class MultiInSARDataset(InSARDataset):
    __slots__ = ("_ele")
    necessary_coords = InSARDataset.necessary_coords + ["num"]
    necessary_vars = InSARDataset.necessary_vars

    def __init__(
            self,
            data_vars: DataVars | None = None,
            coords: Mapping[Any, Any] | None = None,
            attrs: Mapping[Any, Any] | None = None,
            select_vars: str | None = "para",
    ) -> None:
        super().__init__(data_vars, coords, attrs, select_vars)

    @property
    def ele(self):
        if self._ele is None:
            self._ele = np.zeros_like(self._defor.values[:, :, 0])
        return self._ele

    @property
    def filelen(self):
        return len(self.num)

    @property
    def dataset(self):
        return self

    @property
    def fitting_accuracy(self):
        return self.fitting_accuracy

    @classmethod
    def load_dataset(cls, filename_or_obj: str, **kwargs) -> MultiInSARTDDInversion:
        return cls._load_dataset(filename_or_obj, **kwargs)

    @classmethod
    def interp_datasets(cls, datasets:list[InSARDataset], coordinates, fitting_accuracy=None) -> MultiInSARDataset:
        """
        确定研究区域内参与计算的点的值
        如果几个文件的坐标不统一则需要进行坐标的统一工作
        parameters
        ___________
        method : {"linear", "nearest", "zero", "slinear", "quadratic", "cubic", "polynomial", \
            "barycentric", "krog", "pchip", "spline", "akima"}, default: "linear"
        """
        from geodesy.dataframe.InSARDataset import InSARDataset
        interp_databases = []
        for i in tqdm(range(len(datasets)), desc="aligning dataset", ncols=100):
            interp_databases.append(InSARDataset.__interp_database(datasets[i].dataset, coordinates=coordinates, fitting_accuracy=fitting_accuracy))
        return cls.create_from_databases(datasets=interp_databases)

    @classmethod
    def create_from_lltndes(cls, fnames:list, columns=None, region=None, fitting_accuracy=None):
        databases = []  # 存放不同文件的InSARDatabase类型数据库的列表
        for i in range(0, len(fnames)):
            databases.append(InSARDataset.create_from_txt(fnames[i], columns=columns,
                                                          region=region, fitting_accuracy=fitting_accuracy))
        return cls.create_from_databases(datasets=databases,
                                         region=region, fitting_accuracy=fitting_accuracy)

    @classmethod
    def create_from_databases(cls, datasets:list, region=None, fitting_accuracy=None,
                              select_vars="para"
                              ):
        db_len = len(datasets)
        if region is None:
            lon0, lon1, lat0, lat1 = [], [], [], []
            for i in range(db_len):
                lon0.append(datasets[i].region[0])
                lon1.append(datasets[i].region[1])
                lat0.append(datasets[i].region[2])
                lat1.append(datasets[i].region[3])
            region = [min(lon0), max(lon1), min(lat0), max(lat1)]

        """判断是否对齐"""
        mark = True
        clon, clat = datasets[0].dataset.clon, datasets[0].dataset.clat
        for databse in datasets:
            try:
                a = clon.values != databse.dataset.clon.values
                b = clat.values != databse.dataset.clat.values
            except:
                mark = False
                break
            if type(a) == np.ndarray: a = a.any()
            if type(b) == np.ndarray: b = b.any()
            if a or b:
                mark = False
                break
        """对databases中的影像进行对齐"""
        # 1. 根据region和fitting_accuracy进行对齐
        if mark == False:  # 没对齐
            if fitting_accuracy is None:
                fitting_accuracy = round(datasets[0].dataset.clon[1].values - datasets[0].dataset.clon[0].values, 6)
            interp_other = MultiInSARDataset.interp_datasets(datasets=datasets, coordinates=region,
                                                             fitting_accuracy=fitting_accuracy)
            return cls(interp_other.dataset, select_vars=select_vars)
        else: # 对齐
            temp_datasets = []
            for i in range(len(datasets)):  # 添加新维度num,区分不同轨道下的值
                temp_datasets.append(datasets[i].dataset.assign_coords(num=[i]))
            dataset = xr.concat(temp_datasets, dim="num").transpose("clat", "clon", "num")  # 维度重新排列
        # 2. 对齐后检查，fitting_accuracy是否正确
        temp_fitting_accuracy = abs(datasets[0].dataset.clon[0] - datasets[0].dataset.clon[1]).data
        if fitting_accuracy is None:
            fitting_accuracy = temp_fitting_accuracy
        if fitting_accuracy != temp_fitting_accuracy:
            other = interp(datasets, coordinates=region, fitting_accuracy=fitting_accuracy)
            return cls(other.dataset)
        return cls(dataset)

    def __len__(self):
        return len(self.dataset.num)

    @classmethod
    def load_databases(cls, fnames:list, region=None, fitting_accuracy=None):
        databases = []
        for fname in fnames:
            temp_database = InSARDataset.load_txt(fname)
            databases.append(temp_database)
        return cls.create_from_databases(datasets=databases, region=region, fitting_accuracy=fitting_accuracy)

    def get_WGS84_coords(self):
        return self.to_InSARData2Ds()[0].get_WGS84_coords()

    def get_geo_coords(self):
        return self.to_InSARData2Ds()[0].get_geo_coords()

    def to_lltndes(self, fnames, fmt: str = "%0.6f", save_path="./",):
        if self.filelen != len(fnames):
            print("please imput right fnames!")
            sys.exit()
        for i in range(self.filelen):
            InSARDataset(self.dataset.isel(num=i)).to_lltnde(fname=fnames[i], fmt=fmt, save_path=save_path)

    def to_InSARData2Ds(self):
        InSARDatabases = []
        for i in range(self.filelen):
            data=self.dataset.sel(num=i).drop_vars("num")
            InSARDatabases.append(InSARDataset(data))
        return InSARDatabases

    def extract_region_datas(self, region:list):
        """
        提取范围内的点，以新DataAnalysis类型的数据进行返回
        """
        return self.__extract_region_data(self, region=region)

    @classmethod
    def __extract_region_data(cls, obj:MultiInSARDataset, region:list):
        clat = obj.dataset.clat
        clon = obj.dataset.clon
        lon0, lon1, lat0, lat1 = region
        lat_sub = clat[np.logical_and(clat>=lat0, clat<=lat1)]
        lon_sub = clon[np.logical_and(clon>=lon0, clon<=lon1)]
        dataset = obj.dataset.sel(clat=lat_sub, clon=lon_sub)
        return cls(dataset)

    def set_dtypes(self, dtypes:list):
        for d in dtypes:
            if d == "InSAR" or d == "GPS":
                pass
            else:
                print("The input values should be:'InSAR' or 'GPS' for set_dtypes()'s dtypes")
                sys.exit()
        self.dataset.coords["dtypes"] = dtypes

    def plot_figure(self, plot_fig:str = "defor", savefig=False, show_mark=True, num:int = None,
                    vmin=None, vmax=None,
                    ):
        # 当plot_fig为"touyin"时，num需要进行定义
        """画图查看公共域, 图是从上到下，从左到右的排序方式"""
        lon, lat = self.get_WGS84_coords()

        fig = plt.figure(figsize=(18, 8))
        ax = plt.subplot2grid((2, round(self.filelen/2)+2),
                              (0, round(self.filelen/2)), rowspan=2, colspan=2)  # 定义子图的位置
        color = ['r', 'orange', 'yellow', 'g', 'c', 'b', 'purple', 'r',
                 'orange', 'yellow', 'g', 'c', 'b', 'purple']  # 当数据量大于14个,颜色就需要重新定义，不够智能化
        if plot_fig == "touyin":
            touyin_xishu = ["look_e, look_n", "look_u"]
        else: touyin_xishu = None
        for i in tqdm(range(0, self.filelen), desc="绘制图像", ncols=100):
            ax_temp = plt.subplot2grid((2, round(self.filelen/2)+2),
                                       (i % 2, int(i/2)))  # 定义子图的位置
            try:
                if plot_fig == "defor":
                    scatter = ax_temp.scatter(lon, lat, vmin=vmin, vmax=vmax,
                                              c=self.dataset.sel(num=i).defor, cmap="jet")
                elif plot_fig == "touyin":
                    scatter = exec("scatter = ax_temp.scatter(lon, lat,"
                                   + "c=self.dataset.ssel(num=i)." +touyin_xishu[i] + ", cmap='jet')")
                else: scatter = None
                ax.plot(lon, lat, "o", color=color[i], alpha=0.05)
                plt.colorbar(scatter, ax=ax_temp)
            except AttributeError:
                scatter = ax_temp.scatter(self.dataset.isel(num=i).lon, self.dataset.isel(num=i).lat,
                                          c=self.dataset.isel(num=i).defor, cmap="jet", vmin=-1.5, vmax=1.5)
                ax.plot(self.dataset.isel(num=i).lon, self.dataset.isel(num=i).lat, "o", color=color[i], alpha=0.05)
                plt.colorbar(scatter, ax=ax_temp)

        if show_mark is True:
            plt.show()
        if savefig is True:
            print("图像保存中......")
            plt.savefig("./results/卫星影像", dpi=200)
        return ax

    def imshow(self, vmin=None, vmax=None, figsize=None):
        fig, axes = plt.subplots(1, self.filelen, figsize=figsize)
        for index, ax in enumerate(axes):
            ax.imshow(self.dataset.defor[:,:,index], cmap="jet", vmin=vmin, vmax=vmax)

    def plot_touyin_xishu_muti(self, i:int):
        fig,axes = plt.subplots(1,3)
        fig.set_size_inches(10, 3)
        scatter1 = axes[0].scatter(self.dataset.lon[:, :, i], self.dataset.lat[:, :, i], c=self.dataset.look_e[:, :, i], cmap='jet')
        scatter2 = axes[1].scatter(self.dataset.lon[:, :, i], self.dataset.lat[:, :, i], c=self.dataset.look_n[:, :, i], cmap='jet')
        scatter3 = axes[2].scatter(self.dataset.lon[:, :, i], self.dataset.lat[:, :, i], c=self.dataset.look_u[:, :, i], cmap='jet')
        plt.colorbar(scatter1, ax = axes[0])
        plt.colorbar(scatter2, ax = axes[1])
        plt.colorbar(scatter3, ax = axes[2])
        plt.show()

    @classmethod
    def add_guassian_noise_cls(cls, instance:MultiInSARDataset, sigmas:list):
        if len(sigmas) != instance.filelen:
            print("please input ", instance.filelen, " length list")
            sys.exit()
        else:
            import numpy.random as nr
            import copy
            temp_da = copy.deepcopy(instance)
            results = np.zeros_like(instance.dataset.defor)
            for i in range(instance.filelen):
                value_gaussian_noise = nr.normal(0, sigmas[i], instance.dataset.defor[:, :, i].shape) #高斯噪声
                results[:, :, i] = value_gaussian_noise
            temp_da.dataset["defor"] = instance.dataset.defor + results
        return cls(temp_da)

    def add_gaussian_noise(self,sigmas:list):
        return self.add_guassian_noise_cls(self, sigmas=sigmas)

    def select_study_area(self, savefig=False, nums:list=None):
        """
        选择研究区域进行解算，默认利用鼠标在图像中选择区域进行选择计算域
        coordinates研究区域的角点：x1,x2,y1,y2,默认为None
        """
        def on_press(event):
            """鼠标按键按下时响应函数"""
            nonlocal  scope_x1, scope_y1
            scope_x1 = event.xdata
            scope_y1 = event.ydata
            print(u"     请点击鼠标滑动框选取所需计算域范围！")
            print(u"     窗口的一个端点的坐标为：", scope_x1, scope_y1)

        def on_release(event):
            """鼠标释放时响应函数"""
            nonlocal  scope_x2, scope_y2
            scope_x2 = event.xdata
            scope_y2 = event.ydata
            print(u"     窗口的另一个端点的坐标为：", scope_x2, scope_y2)

        """选取点，建立窗口"""
        scope_x1, scope_x2, scope_y1, scope_y2 = 0, 0, 0, 0
        fig, ax = plt.subplots(1, 1)
        cmap_names = [
            'viridis', 'plasma', 'inferno', 'magma',  # 渐变色映射
            'cividis', 'turbo', 'nipy_spectral', 'twilight',  # 新颜色映射
            'tab10', 'tab20', 'tab20b', 'tab20c',  # 分类色映射
            'cool', 'coolwarm', 'hot', 'autumn'  # 颜色映射
        ]
        if nums is None:
            nums = range(self.filelen)
        for index in nums:
            plt.imshow(self.dataset.sel(num=index).defor, extent=self.region, cmap=cmap_names[index], alpha=1)
        ax.figure.canvas.mpl_connect("button_press_event", on_press)
        ax.figure.canvas.mpl_connect("button_release_event", on_release)  # 鼠标释放时响应事件，用于录入scope的边界点
        plt.show()

        """建立窗口"""
        if scope_x1 > scope_x2:
            a = scope_x2
            b = scope_x1
        else:
            a = scope_x1
            b = scope_x2
        if scope_y1 > scope_y2:
            c = scope_y2
            d = scope_y1
        else:
            c = scope_y1
            d = scope_y2
        region = [a, b, c, d]

        return region  # 返回选择区域的坐标

    def imshow_defor(self, vmin: float = None, vmax: float = None, cmap: str = "jet", axes=None, axis=0):
        from typing import List
        plt.rcParams["font.family"] = "Times New Roman"
        databases : List[InSARDataset] = self.to_InSARData2Ds()
        if axes is None:
            axes = [None] * self.filelen
        for index, database in enumerate(databases):
            if axes[0] != None:
                ax = axes[index]
            else:
                ax = plt.subplot(1, self.filelen, index+1)
            """绘图"""
            databaseVisual = database.to_InSARDatasetVisual(database.dataset)
            databaseVisual.imshow_defor(vmin=vmin, vmax=vmax, cmap=cmap, ax=ax)
            """更改刻度显示方向"""
            if axis == 0: # 沿x轴方向横向出图
                plt.gca().xaxis.set_ticks_position("top")
            else:
                plt.gca().yaxis.set_ticks_position("left")
            """隐藏多余刻度信息"""
            if index != 0:
                if axis == 0: plt.yticks([])
                else:
                    plt.xticks([])
