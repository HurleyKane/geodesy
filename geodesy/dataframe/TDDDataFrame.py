import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

from geodesy.dataframe.GeoDataFrame import GeoDataFrame


class TDDDataFrame(GeoDataFrame):
    """
    格式要求：lon, lat, ele, ew_def, ns_def, v_def
    """
    __slots__ = ()

    def __init__(self, data: np.ndarray or pd.DataFrame, columns=None):
        """获取形变后的值"""
        if columns is None:
            columns = ["lon", "lat", "ele", "ew_def", "ns_def", "v_def"]
        super().__init__(data=data, columns=columns)

    def griddata(self, region, fitting_accuracy=0.005):
        row = np.arange(region[3], region[2] - fitting_accuracy, -fitting_accuracy)
        col = np.arange(region[0], region[1] + fitting_accuracy, fitting_accuracy)
        xg, yg = np.meshgrid(col, row)
        scope = np.c_[xg.ravel(), yg.ravel()]
        values = []
        for i in range(2, self.dataframe.shape[1]):
            temp = griddata((self.dataframe.lon, self.dataframe.lat), self.dataframe.iloc[:, i], scope,
                            method="nearest")
            values.append(temp)
        new_datas = np.vstack([xg.ravel(), yg.ravel()] + values).T
        return TDDDataFrame(data=new_datas)

    def plot_figure(self, ew: list = None, ns: list = None, v: list = None, gps_file=None):
        """
        ew, ns, v 分别指对应方向的vmin,vmax的取值，放入list中从小到大排列
        """
        if ew is None: ew = [None, None]
        if ns is None: ns = [None, None]
        if v is None: v = [None, None]
        plt.figure(figsize=(11, 4))
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)
        ew = ax1.scatter(x=self.dataframe.lon, y=self.dataframe.lat, c=self.dataframe.ew_def, vmin=ew[0], vmax=ew[1],
                         cmap="jet")
        ns = ax2.scatter(x=self.dataframe.lon, y=self.dataframe.lat, c=self.dataframe.ns_def, vmin=ns[0], vmax=ns[1],
                         cmap="jet")
        u = ax3.scatter(x=self.dataframe.lon, y=self.dataframe.lat, c=self.dataframe.v_def, vmin=v[0], vmax=v[1],
                        cmap="jet")
        plt.colorbar(ew, ax=ax1)
        plt.colorbar(ns, ax=ax2)
        plt.colorbar(u, ax=ax3)
        plt.show()

    def sort(self):
        return self.dataframe.sort_values(by=["lon", "lat"], ascending=[True, False])

    def to_gps_db(self):
        from geodesy.dataframe.GNSSDataFrame import GNSSDataFrame
        return GNSSDataFrame(data=self.dataframe.values)

    def project_to_LOS(self, theta, alpha, filename="POT.lltnde"):
        """
        theta:入射角
        alpha:方位角
        """
        from geodesy.dataframe.InSARDataset import InSARDataset

        theta = theta / 180 * np.pi  # 入射角
        alpha = alpha / 180 * np.pi
        los_look = [-np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), np.cos(theta)]
        los_look = [np.full_like(self.dataframe.ew_def, direction_look) for direction_look in los_look]
        los_defor = self.dataframe.ew_def * los_look[0] + self.dataframe.ns_def * los_look[1] + self.dataframe.v_def * \
                    los_look[2]
        temp_data = [self.dataframe.lon, self.dataframe.lat, self.dataframe.ele] + los_look + [los_defor]
        los_data = np.vstack(temp_data).T
        if filename is not None:
            with open(filename, "w") as f:
                np.savetxt(f, los_data, fmt="%0.6f")
        columns = ["lon", "lat", "ele", "look_e", "look_n", "look_u", "defor"]
        return InSARDataset.creat_from_numpy_or_pandas(data=los_data, columns=columns)


    def project_to_AZI(self, theta, alpha, filename="AZI.lltnde"):
        """
        theta:入射角
        alpha:方位角
        """
        from geodesy.dataframe.InSARDataset import InSARDataset

        alpha = alpha / 180 * np.pi
        azi_look = [np.sin(alpha), np.cos(alpha), 0]
        azi_look = [np.full_like(self.dataframe.ew_def, direction_look) for direction_look in azi_look]
        azi_defor = self.dataframe.ew_def * azi_look[0] + self.dataframe.ns_def * azi_look[1] + self.dataframe.v_def * \
                    azi_look[2]
        azi_data = np.vstack([self.dataframe.lon, self.dataframe.lat, self.dataframe.ele] + azi_look + [azi_defor]).T
        with open(filename, "w") as f:
            np.savetxt(f, azi_data, fmt="%0.6f")
        columns = ["lon", "lat", "ele", "look_e", "look_n", "look_u", "defor"]
        return InSARDataset.creat_from_numpy_or_pandas(data=azi_data, columns=columns)
