import os, sys
import numpy as np
import xarray as xr
from scipy import signal
import matplotlib.pyplot as plt

from xarray.core.types import DataVars
from collections.abc import Mapping
from typing import Any
from geodesy.dataframe.GeoDataset import GeoDataset

class InSARDataset(GeoDataset):
    __slots__ = ()
    necessary_vars = ["defor"] + GeoDataset.necessary_vars
    para_vars = ["look_e", "look_n", "look_u"]
    angle_vars = ["incidence", "azimuth"]
    def __init__(
            self,
            data_vars: DataVars | None = None,
            coords: Mapping[Any, Any] | None = None,
            attrs: Mapping[Any, Any] | None = None,
            select_vars: str | None = "para",
    ) -> None:
        """
        数据结构采用xarray:https://docs.xarray.dev/en/stable/getting-started-guide/quick-overview.html
        """
        if select_vars is None:
            select_vars = "para"
        if select_vars not in ["para", "azimuth"]:
            sys.exit("select_vars must be either 'para' or 'angle'")
        elif select_vars == "para":
            InSARDataset.necessary_vars = InSARDataset.necessary_vars + InSARDataset.para_vars
        else:
            InSARDataset.necessary_vars = InSARDataset.necessary_vars + InSARDataset.angle_vars
        super().__init__(data_vars, coords, attrs)

    @property
    def ele(self):
        return self._ele

    @classmethod
    def create_from_txt(cls, filename:str, columns:list=None, region:list=None, fitting_accuracy:float=None,
                        select_vars:str="para"
                        ):
        if columns is None:
            columns = ["lon", "lat", "ele", "look_e", "look_n", "look_u", "defor"]
        obj = super(InSARDataset, cls).create_from_txt(filename=filename,
                                                       columns=columns)
        if region is not None:
            if fitting_accuracy is None:
                fitting_accuracy = abs(obj.dataset.clon[1] - obj.dataset.clon[0])
            obj = obj.interp_dataset(coordinates=region, fitting_accuracy=fitting_accuracy)
        return cls(obj, select_vars=select_vars)

    def __add__(self, other):
        # 两个SAR影像相加逻辑 A+B=A+(B-B_B∩A)
        total_region = self.calculate_union(other.region)
        ds1 = InSARDataset.interp(self.dataset, coordinates=total_region, fitting_accuracy=self.fitting_accuracy)
        ds2 = InSARDataset.interp(other.dataset, coordinates=total_region, fitting_accuracy=self.fitting_accuracy)
        return InSARDataset(ds1.dataset.combine_first(ds2.dataset))

    def gps_correct_insar(self, gps_file, columns:list, skiprows:int=0, usecols:list=None, fitting_accuracy=0.005):
        """
        利用GPS数据对InSAR数据进行去轨道的改正
        skip_points:为list类型数据,设置不需要读入的数据
        """
        from geodesy.dataframe.GNSSDataFrame import GNSSDataFrame

        gps_db = GNSSDataFrame.read_csv(filename=gps_file, columns=columns)  # 读入数据
        # 遍历查询GPS站点附近的点
        points_ds = []  # 存放第i站点周围点的信息
        for i in range(len(gps_db.dataframe.lon)):  # 遍历第一个点
            try:
                coords = self.dataset.sel(clon=gps_db.dataframe.lon[i], clat=gps_db.dataframe.lat[i], method="nearest",
                                          tolerance=0.01).coords
            except KeyError:
                continue
            clon = [round(i,5) for i in np.arange(coords["clon"].values-2*fitting_accuracy, coords["clon"].
                                                  values+2*fitting_accuracy, fitting_accuracy)]
            clat = [round(i,5) for i in np.arange(coords["clat"].values-2*fitting_accuracy, coords["clat"].
                                                  values+2*fitting_accuracy, fitting_accuracy)]
            point_temp_ds = self.dataset.sel(clon=clon, clat=clat)
            points_ds.append(point_temp_ds)

        # 对得到的站点周围点数据进行处理
        # # 算出GPS站点周围点对应系数下的GPS站点对应的azi方向上的数值
        insar_azi_value = []
        gps_azi_value = []
        for i in range(len(points_ds)):  # 遍历第一个点
            look_e = np.average(points_ds[i].look_e)
            look_n = np.average(points_ds[i].look_n)
            look_u = np.average(points_ds[i].look_u)
            value = np.nansum(points_ds[i].defor.values)/np.sum(points_ds[i].defor.values==points_ds[i].defor.values)
            if value == value:
                insar_azi_value.append(value)
                gps_azi_value.append(look_e * gps_db.dataframe.ew_def[i] + look_n * gps_db.dataframe.ns_def[i] +look_u
                                     * gps_db.dataframe.v_def[i])
        # 利用 n = c+bm+am^2 对数据进行拟合
        m = np.array(insar_azi_value)
        B = np.matrix(np.c_[m*m, m, np.ones_like(m)])
        n = np.matrix(gps_azi_value).T
        X = np.linalg.solve(B.T*B,B.T*n)
        X = np.array(X).reshape(1,-1)

        # 根据修正值对整个GPS场进行修正
        a, b, c = X[0]
        da_def = a * self.dataset.defor ** 2 + b * self.dataset.defor + c  # da--->xr.DataArray类型
        return da_def

    def transform_angle_to_lookenu(self, projectType):
        # 将入射角和方位角转换为东西、南北和垂直向的系数
        new_data_vars = {}
        vars_keys = [i.name for i in self.dataset.var()]
        if "incidence" in vars_keys and "azimuth" in vars_keys:
            if "look_e" not in vars_keys and "look_n" not in vars_keys and "look_u" not in vars_keys:
                if projectType == "LOS":
                    look_e = -np.sin(self.dataset.incidence) * np.cos(self.dataset.azimuth)
                    look_n = np.sin(self.dataset.incidence) * np.sin(self.dataset.azimuth)
                    look_u = np.cos(self.dataset.incidence)
                elif projectType == "AZI":
                    look_e = np.sin(self.dataset.azimuth)
                    look_n = np.cos(self.dataset.azimuth)
                    look_u = np.full_like(look_e, 0)
                else:
                    print("project type is error!")
                    sys.exit()
                new_data_vars["look_e"] = (["clat", "clon"], look_e)
                new_data_vars["look_n"] = (["clat", "clon"], look_n)
                new_data_vars["look_u"] = (["clat", "clon"], look_u)
            else:
                print("lookenu is in here!")
        else:
            print("angle is not in here")

    def quardtree_downsampling(obj, threshold, according_variable: str, method="default",
                               plt_show=True, **kwargs):
        """对结果记性四叉树下采样"""
        """
        according_varibable：根据某一变量进行四叉树分解的下采样
        method: default, drop_track
        """
        obj = super().quardtree_downsampling(threshold=threshold, according_variable=according_variable,
                                             method=method, plt_show=plt_show,
                                             convert_type="InSAR"
                                             )
        return obj

    def plot_scatter(self, plt_var:str="defor", save_fig_name="defor", dimension="2d", vmin:float=None, vmax:float=None):
        # plt_var : defor, touyin
        print("打印散点数据中......")
        lon, lat = self.get_WGS84_coords()
        if dimension=="2d":

            fig,ax = plt.subplots(1,1)
            # exec("_axe_a = ax.scatter(self.dataset.lon, self.dataset.lat, c=self.dataset."+plt_var+
            #      ", vmax=1, vmin=-1, cmap='jet')")
            exec("_axe_a = ax.scatter(lon, lat, c=self.dataset."+plt_var+
                 ", cmap='jet', vmin=vmin, vmax=vmax)")
            axe_a = locals()["_axe_a"]
            plt.colorbar(axe_a,ax = ax)
        elif dimension=="3d":
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            exec("_axe_a = ax.scatter(lon, lat, self.dataset."+plt_var+".values,"
                 + "c=self.dataset."+plt_var+".values, cmap='jet', vmin=vmin, vmax=vmax)")
            axe_a = locals()["_axe_a"]
            plt.colorbar(axe_a, ax=ax)
        if save_fig_name != None:
            plt.savefig(save_fig_name,dpi=500)
        plt.show()

    def figure(self):
        plt.figure()

    def imshow(self, vmin=None, vmax=None, extent=None):
        plt.imshow(self.dataset.defor.data, vmin=vmin, vmax=vmax, cmap="coolwarm", extent=extent)

    def show(self):
        plt.show()

    def plot_touyin_xishu(self):
        fig,axes = plt.subplots(1,3)
        fig.set_size_inches(10, 3)
        scatter1 = axes[0].scatter(self.dataset.lon, self.dataset.lat, c=self.dataset.look_e, cmap='jet')
        scatter2 = axes[1].scatter(self.dataset.lon, self.dataset.lat, c=self.dataset.look_n, cmap='jet')
        scatter3 = axes[2].scatter(self.dataset.lon, self.dataset.lat, c=self.dataset.look_u, cmap='jet')
        plt.colorbar(scatter1, ax = axes[0])
        plt.colorbar(scatter2, ax = axes[1])
        plt.colorbar(scatter3, ax = axes[2])
        plt.show()

    def add_gaussian_noise(self,sigma):
        """加入sigma大小的标准差的高斯噪声（高斯分布）"""
        print("adding gaussian noise...")
        from numpy import random as nr
        value_gaussian_noise = nr.normal(0,sigma,self.dataset.defor.shape) #高斯噪声
        dataset_defor = self.dataset.defor + value_gaussian_noise # add guassion niose
        return dataset_defor

    def to_lltnde(self, fname, fmt:str="%0.6f", save_path="./"):
        # 该代码需要更新
        filepath = os.path.join(save_path, fname)
        with open(filepath, "w") as f:
            lon, lat = self.get_WGS84_coords()
            lon = lon.reshape(-1, 1)
            lat = lat.reshape(-1, 1)
            try:
                ele = self.dataset.ele.values.ravel().reshape(-1, 1)
            except:
                ele = np.zeros_like(lon)
            look_e = self.dataset.look_e.values.ravel().reshape(-1, 1)
            look_n = self.dataset.look_n.values.ravel().reshape(-1, 1)
            look_u = self.dataset.look_u.values.ravel().reshape(-1, 1)
            defor = self.dataset.defor.values.ravel().reshape(-1, 1)
            results = np.hstack([lon, lat, ele, look_e, look_n, look_u, defor])
            subs = np.argwhere(lon == lon)[:, 0]
            results = results[subs]
            np.savetxt(f, results, fmt=fmt)

    def to_InSARData1D(self):
        import InSARDataFrame
        lon, lat = self.get_WGS84_coords()
        lon = lon.reshape(-1, 1)
        lat = lat.reshape(-1, 1)
        try:
            ele = self.dataset.ele.values.ravel().reshape(-1, 1)
        except:
            ele = np.zeros_like(lon)
        look_e = self.dataset.look_e.values.ravel().reshape(-1, 1)
        look_n = self.dataset.look_n.values.ravel().reshape(-1, 1)
        look_u = self.dataset.look_u.values.ravel().reshape(-1, 1)
        defor = self.dataset.defor.values.ravel().reshape(-1, 1)
        results = np.hstack([lon, lat, ele, look_e, look_n, look_u, defor])
        subs = np.argwhere(lon == lon)[:, 0]
        results = results[subs]
        return InSARDataFrame.InSARDataFrame(data=results)


    def convolve2d(self, kernel:np.ndarray, mode:str="same", atol=0.5, plot_mark:bool=False):
        """
        mode:
            1."same",同维卷积方法，目前卷积方法共有三种，该种方法是卷积核核心元素生效
            2.“full",完全卷积方法，该种方法是卷积核任意元素生效
            3.“valid”,有效卷积，卷积核数组每个元素生效
        atol: absolute tolerance,绝对容差
        """
        # 实现一个二维的卷积操作
        if len(kernel.shape) == 1:
            print("convolution kernel is one dimention")
            sys.exit()
        values = self.dataset.defor.values
        values = np.where(values != values, 0, values)
        results = signal.convolve2d(values, kernel, mode=mode)
        results = np.where(abs(results)<atol, 0, 1)  # 小于阈值则对该地区设为0

        if plot_mark:
            # plt.scatter(x=self.dataset.lon, y=self.dataset.lat, c=results, s=10, cmap="jet")
            plt.imshow(results, cmap="jet")
        return results

