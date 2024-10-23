from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geodesy.dataframe.GeoDataset import GeoDataset
from geodesy.dataframe.GNSSDataFrame import GNSSDataFrame
from xarray.core.types import DataVars
from collections.abc import Mapping
from typing import Any


class TDDDataset(GeoDataset):
    __slots__ = ("_initial_mark", "_dataset", "_region", "_fitting_accuracy", "_columns", "_clon", "_clat")
    def __init__(
            self,
            data_vars: DataVars | None = None,
            coords: Mapping[Any, Any] | None = None,
            attrs: Mapping[Any, Any] | None = None,
            select_para: str | None = None,
    ) -> None:
        super().__init__(data_vars, coords, attrs)
        if select_para == "strain":
            TDDDataset.necessary_vars += TDDDataset.strain_and_rotation_parameter
            self._initial_strain_and_rotation_parameters()
            self._add_three_strain_parameters()
        self._initial_mark = False

    strain_and_rotation_parameter = ("strain_ee", "strain_en", "strain_ev", "strain_nn", "strain_nv", "strain_vv",
                                     "rotation_en", "rotation_ev", "rotation_nv")
    necessary_vars = GeoDataset.necessary_vars + ["ew_def", "ns_def", "v_def"]

    def __sub__(self, other : TDDDataset):
        if not isinstance(other, TDDDataset):
            raise TypeError(f"Unsupported operand type(s) for -: '{type(self)}' and '{type(other)}'")

        # 使用父类的减法操作
        result_ds = super().__sub__(other)

        # 返回一个新的 TDDataset 实例
        return TDDDataset(result_ds)

    @property
    def dilatation_strain(self):
        if not self._initial_mark:
            return None
        else:
            # value = (self.dataset["strain_ee"]+self.dataset["strain_nn"]+self.dataset["strain_vv"])/3
            value = (self.dataset["strain_ee"]+self.dataset["strain_nn"])
            return value

    @property
    def differential_rotation_magnitude(self):
        if not self._initial_mark:
            return None
        else:
            value = self.dataset["rotation_en"]
            return value

    @property
    def maximum_shear_strain(self):
        if not self._initial_mark:
            return None
        else:
            value = np.sqrt((self.dataset["strain_ee"]-self.dataset["strain_nn"])**2/4+self.dataset["strain_en"]**2)
            return value

    @classmethod
    def _create_empty_obj(cls, clon, clat, attributes:tuple=None):
        base_attributes = ("ele", "ew_def", "ns_def", "v_def")
        if attributes is None:
            attributes = base_attributes
        else:
            attributes = tuple(list(base_attributes) + list(attributes))
        return super()._create_empty_obj(clon, clat, attributes=attributes)

    @staticmethod
    def create_from_txt(filename, extra_attributes:str=None, delimeter:str= " ") -> TDDDataset:
        df = pd.read_csv(filename, sep=delimeter, header=None)
        if df.shape[1] == 6 and extra_attributes == "strain":
            new_cols = TDDDataset.strain_and_rotation_parameter
            for i in range(len(new_cols)):
                df[new_cols[i]] = 0
        obj = TDDDataset.creat_from_numpy_or_pandas(df, select_para=extra_attributes)
        return obj

    @classmethod
    def creat_from_numpy_or_pandas(cls, data: pd.DataFrame or np.ndarray, select_para:str=None):
        if select_para is None:
            columns = ["lon", "lat", "ele", "ew_def", "ns_def", "v_def"]
        elif select_para == "strain":
            columns = ["lon", "lat", "ele", "ew_def", "ns_def", "v_def"] + list(cls.strain_and_rotation_parameter)
        else:
            print("No such attribute in columns")
            columns = None
        obj = super().creat_from_numpy_or_pandas(data=data, columns= columns)
        return TDDDataset(obj, select_para=select_para)

    def _initial_strain_and_rotation_parameters(self):
        """初始化应变和旋转参数属性, 并将三个应变不变量传入Dataset"""
        for attribute_name in self.strain_and_rotation_parameter:
            if attribute_name not in list(self.dataset.data_vars.keys()):
                self._dataset[attribute_name] = ("clat", "clon"), np.full((len(self.clat), len(self.clon)), np.nan)
        self._initial_mark = True

    def _add_three_strain_parameters(self):
        """dataset中加入三个应变不变量"""
        self["dilatation_strain"] = self.dilatation_strain
        self["differential_rotation_magnitude"] = self.differential_rotation_magnitude
        self["maximum_shear_strain"] = self.maximum_shear_strain

    def get_GNSS_sites_indexes_in_region(self, GNSSdb:GNSSDataFrame):
        indexes = []
        lon0, lon1, lat0, lat1 = self.region
        for i in range(GNSSdb.lon.shape[0]):
            if GNSSdb.lon[i] <= lon1  and GNSSdb.lon[i] >= lon0 and GNSSdb.lat[i] <= lat1 and GNSSdb.lat[i] >= lat0:
                indexes.append(i)
        return indexes

    def geo_plot_3D_deformation(self, *args, colorbar_name : str = "deformation/m", **kwargs):
        return super().geo_plot.imshow_properties(
            *args, properties=["ew_def", "ns_def", "v_def"], colorbar_name = colorbar_name,
            **kwargs
        )

    def geo_plot_strain_parameters(self, *args, colorbar_name : str = "", **kwargs):
        return super().geo_plot.imshow_properties(
            *args, properties=["dilatation_strain", "differential_rotation_magnitude", "maximum_shear_strain"],
            colorbar_name = colorbar_name,
            **kwargs
        )


    def calculate_residual_with_GNSS(self, GNSSdb: GNSSDataFrame):
        import pandas as pd
        columns = ["sites", "lon", "lat", "ew(cm)", "ns(cm)", "v(cm)"]
        results = pd.DataFrame([], columns=columns)
        indexes = self.get_GNSS_sites_indexes_in_region(GNSSdb)
        if len(indexes) == 0:
            print("No GNSS in this region")
        else:
            ew_subs, ns_subs, v_subs = [], [], []
            for i, index in enumerate(indexes):
                row, col = self.find_closest_point(lon=GNSSdb.lon[indexes[i]], lat=GNSSdb.lat[indexes[i]])
                # print(self.clat[row], self.clon[col])
                mean_ew_def = np.nanmean(self.ew_def[row - 3:row + 3, col - 3:col + 3])
                mean_ns_def = np.nanmean(self.ns_def[row - 3:row + 3, col - 3:col + 3])
                mean_v_def = np.nanmean(self.v_def[row - 3:row + 3, col - 3:col + 3])
                ew_sub = (mean_ew_def - GNSSdb.ew_def[index]) * 100
                ns_sub = (mean_ns_def - GNSSdb.ns_def[index]) * 100
                v_sub = (mean_v_def - GNSSdb.v_def[index]) * 100
                ew_subs.append(ew_sub)
                ns_subs.append(ns_sub)
                v_subs.append(v_sub)
            ew_subs = np.array(ew_subs)
            ns_subs = np.array(ns_subs)
            v_subs = np.array(v_subs)
            ew_ridual = np.sqrt(np.sum((ew_subs - np.mean(ew_subs)) ** 2) / ew_subs.shape[0])
            ns_ridual = np.sqrt(np.sum((ns_subs - np.mean(ns_subs)) ** 2) / ns_subs.shape[0])
            v_risual = np.sqrt(np.sum((v_subs - np.mean(v_subs)) ** 2) / v_subs.shape[0])
            for column in ["sites", "lon", "lat"]:
                results[column] = GNSSdb[column]
            results["ew(cm)"] = ew_subs
            results["ns(cm)"] = ns_subs
            results["v(cm)"] = v_subs
            results.loc[results.shape[0]] = ["RMSE", 0 , 0, ew_ridual, ns_ridual, v_risual]
            return results


    def calculte_3D_deformation_rmse(self, real_TDDdata:TDDDataset):
        from geodesy.algorithm import calculate_rmse
        ew_rmse = calculate_rmse(real_TDDdata.ew_def.values.ravel(), self.ew_def.values.ravel())
        ns_rmse = calculate_rmse(real_TDDdata.ns_def.values.ravel(), self.ns_def.values.ravel())
        v_rmse  = calculate_rmse(real_TDDdata.v_def.values.ravel(),  self.v_def.values.ravel())
        return ew_rmse, ns_rmse, v_rmse

if __name__ == "__main__":
    #%%
    import InSARlib.core.config as cf
    import os
    os.chdir(cf.get_rootPath())

    #%% 三维形变数据的读取
    # TDDdataFile = "./tempResults/niaho.tdd"
    TDDdataFile = "Datas/results/SMVCE_result.tdd"
    TDDcolumns = ["lon", "lat", "ele", "ew_def", "ns_def", "v_def"]
    TDDAarray = TDDDataset.create_from_txt(TDDdataFile, extra_attributes="strain")
