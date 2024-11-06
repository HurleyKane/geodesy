from __future__ import annotations
import pandas as pd
import numpy as np
from geodesy.dataframe.BaseDataFrame import BaseDataFrame

class MultiTimeSeriesDataFrame():
    def __init__(self, datas:dict[str, TimeSeriesDataFrame]):
        self.datas = datas

    @classmethod
    def read_from_folder(cls, path, delimiter: str = " ", columns: list = None, header=None, str_col: list = None,
                      skiprows=None, **kwargs):
        import os
        if os.path.exists(path):
            files = os.listdir(path)
        else:
            raise FileNotFoundError
        files_path = [os.path.join(path, file, "data.txt") for file in files]
        datas = {}
        for index, file in enumerate(files_path):
            try:
                data = TimeSeriesDataFrame.read_from_csv(file, delimiter, columns, header, str_col, skiprows, **kwargs)
                datas[files[index]] = (data)
            except Exception as e:
                print(e)
                print(f"file: {file} read failed")
                # 输出log文件
                with open("log.txt", "a") as f:
                    f.write(f"file: {file} read failed\n")
                continue
        return cls(datas)

    def weighted_least_square(self):
        from tqdm import tqdm
        with tqdm(total=len(self.datas)) as pbar:
            for data in self.datas.values():
                try:
                    data.weighted_least_square()
                    pbar.update(1)
                except:
                    pbar.update(1)

class TimeSeriesDataFrame(BaseDataFrame):
    neccessary_columns = ["time", ]
    _B = []
    _X = []
    def __init__(
            self,
            data=None,
            index = None,
            columns = None,
            dtype = None,
            copy = False,
    ) -> None:
        super().__init__(data, index, columns, dtype, copy)
        self.set_index("time", inplace=True)

    """*******************************************数据的存取********************************************************"""
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
            if key == "time": continue
            dataframe[key] = dataframe[key].astype(dtype_dict[key])
        dataframe.index = dataframe.index.astype(float)
        return dataframe

    @property
    def B(self):
        """模型系数矩阵"""
        return np.hstack(self._B)

    @property
    def X(self):
        return np.hstack(self._X)

    @property
    def ti(self):
        ti = self.index.values.reshape(-1, 1).astype(float)
        return ti

    def add_time_series_model(self):
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

    def weighted_least_square(self):
        from geodesy.algorithm import weighted_least_square
        for column in self.columns:
            X = weighted_least_square(self.B, self[column].values.reshape(-1, 1))
            self._X.append(X)
        return np.hstack(self._X)

    @property
    def Y(self):
        Y = self.B @ self.X
        Y = np.hstack([self.ti, Y])
        columns = ["fit_" + column for column in self.columns]
        columns = ["time"] + columns
        Y = TimeSeriesDataFrame(data=Y, columns=columns)
        return Y

    """************************************地震类相关分量****************************************************"""
    def add_postseismic_vicoelastic_rheology(self, tk: tuple, tao: float):
        """震后粘弹性体指数模型"""
        ti = self.index.values.reshape(-1, 1).astype(float)
        for i in range(len(tk)):
            self._B.append((-1 * np.exp(-((ti - tk[i]) / tao)) + 1) * np.where(ti > tk[i], 1, 0))
