from __future__ import annotations
from pandas import DataFrame, read_csv
import os
from io import StringIO


class FaultGeometry(DataFrame):
    # 定义列名
    default_columns = ["no", "slip", "lon", "lat", "depth", "length", "width", "strike", "dip", "rake"]

    def __init__(
            self,
            data=None,
            index=None,
            dtype=None,
            copy=False,
    ) -> None:
        # 验证输入数据的格式，并处理
        if isinstance(data, (list, dict)):
            # 如果 data 是列表或字典，构造 DataFrame
            super().__init__(data=data, index=index, dtype=dtype, copy=copy)
        elif isinstance(data, DataFrame):
            # 如果 data 已经是 DataFrame，直接传递
            super().__init__(data=data.values, index=index, dtype=dtype, copy=copy)
        else:
            raise ValueError("Data must be a list, dict, or DataFrame")

        # 设置列名
        self.columns = FaultGeometry.default_columns

    @classmethod
    def read_from_csv(cls, source, delimiter: str = " ", header=None, **kwargs):
        try:
            # 判断 source 是文件路径还是 StringIO 对象
            if isinstance(source, str) and os.path.isfile(source):
                # 读取文件
                dataframe = read_csv(source, delimiter=delimiter, header=header, names=cls.default_columns, **kwargs)
            elif isinstance(source, str) and not os.path.isfile(source):
                source = StringIO(source)
                dataframe = read_csv(source, delimiter=delimiter, header=header, names=cls.default_columns, **kwargs)
            elif isinstance(source, StringIO):
                # 读取 StringIO 对象
                dataframe = read_csv(source, delimiter=delimiter, header=header, names=cls.default_columns, **kwargs)
            else:
                raise ValueError("Source must be a valid file path or a StringIO object")
        except Exception as e:
            raise IOError(f"Error reading source: {e}")

        return cls(data=dataframe)
    # 转换为矩形坐标
    def get_rectangle_coords(self):
        pass


if __name__ == "__main__":
    # 示例数据
    str_data = """1,2,3,4,5,6,7,8,9,0
2,3,4,5,6,7,8,9,0,1"""
    data = StringIO(str_data)

    try:
        # 读取数据
        fg = FaultGeometry.read_from_csv(data, delimiter=",")
        print(fg)
    except Exception as e:
        print(f"Error: {e}")
