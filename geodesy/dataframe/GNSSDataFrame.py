import os, sys
import logging
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt

from geodesy.dataframe.GeoDataFrame import GeoDataFrame

class GNSSDataFrame(GeoDataFrame):
    def __init__(self, data:np.ndarray or pd.DataFrame=None, columns:list=None):
        '''
        usercols:为选择数据列的列表：列表中的数据分应分别为lon,lat,ele,ew_deformation,ns_deformation,v_def,如果为None则默认数据
            分别为0,1,2,3,4,5
        columns:请从下列数据进行选择："lon","lat","ele","ew_def","ns_def","v_def", "sites"
        '''
        if not self._are_reasonable_column(columns):
            print("please input columns in 'lon', 'lat', 'ele', 'ew_def', 'ns_def', 'v_def', 'sites'!")
            sys.exit()
        super().__init__(data=data, columns=columns)

    def _are_reasonable_column(self, columns):
        for column in columns:
            if column in ['lon','lat','ele','ew_def','ns_def','v_def','sites']:
                return True
            else: return False

    def delete(self, rows:list):
        # 删除并重新进行初始化
        return GNSSDataFrame(data = super().drop(index=rows).dataframe)

    def add_gaussian_noise(self,sigmas:list):
        """加入sigma大小的标准差的高斯噪声（高斯分布）"""
        # print("\nadding gaussian noise...")
        from numpy import random as nr
        new_obj = deepcopy(self)
        shape = self.dataframe.lon.shape
        ew_gaussian_noise = nr.normal(0, sigmas[0], shape) #高斯噪声
        ns_guassian_noise = nr.normal(0, sigmas[1], shape)
        v_guassian_noise = nr.normal(0, sigmas[2], shape)
        new_obj.dataframe.ew_def = self.dataframe.ew_def + ew_gaussian_noise
        new_obj.dataframe.ns_def = self.dataframe.ns_def + ns_guassian_noise
        new_obj.dataframe.v_def  = self.dataframe.v_def  + v_guassian_noise
        return new_obj

    def to_InSAR1Ds(self):
        try:
            from InSARlib.core.InSAR import InSARDataFrame
        except ImportError as e:
            logging.error(f"ImportError: {e}")
            print(
                "InSARlib is not installed or there is an issue with the version. Please install it using 'pip install InSARlib'.")
            return None
        if self.dataframe.ele is None:
            self.dataframe.ele = np.zeros_like(self.dataframe.lon)
        ew = np.c_[self.dataframe.lon, self.dataframe.lat, self.dataframe.ele,
        np.ones_like(self.dataframe.lon), np.zeros_like(self.dataframe.lon),
        np.zeros_like(self.dataframe.lon), self.dataframe.ew_def]
        ns = np.c_[self.dataframe.lon, self.dataframe.lat, self.dataframe.ele,
        np.zeros_like(self.dataframe.lon), np.ones_like(self.dataframe.lon),
        np.zeros_like(self.dataframe.lon), self.dataframe.ns_def]
        u = np.c_[self.dataframe.lon, self.dataframe.lat, self.dataframe.ele,
        np.zeros_like(self.dataframe.lon), np.zeros_like(self.dataframe.lon),
        np.ones_like(self.dataframe.lon), self.dataframe.v_def]
        ew = InSARDataFrame(data=ew)
        ns = InSARDataFrame(data=ns)
        u  = InSARDataFrame(data=u)
        return ew, ns, u

    def plot_gps(self, ew=None, ns=None, v=None):
        """s: be familiar with s's function in plt.scatter() function"""
        if ew is None: ew=[None, None]
        if ns is None: ns=[None, None]
        if v is None: v=[None, None]
        fig, axes = plt.subplots(1,3)
        ax1, ax2, ax3 = axes
        ax1.scatter(self.dataframe.lon, self.dataframe.lat, c=self.dataframe.ew_def, cmap="jet", vmin=ew[0], vmax=ew[1])
        ax2.scatter(self.dataframe.lon, self.dataframe.lat, c=self.dataframe.ns_def, cmap="jet", vmin=ns[0], vmax=ns[1])
        ax3.scatter(self.dataframe.lon, self.dataframe.lat, c=self.dataframe.v_def,  cmap="jet", vmin=v[0], vmax=v[1])

    def save_to_tdd(self, filepath="./results/r.tdd"):
        filepath = os.path.join(".", filepath)
        np.savetxt(filepath, self.dataframe.values, fmt="%0.8f")

