from geodesy.config import get_rootPath
import platform

import numpy as np
import pandas as pd
import re, os

# 数据的路径确定工作
data_path = os.path.join(get_rootPath(), "io", "CN-faults")
if not os.path.exists(data_path):
    raise FileNotFoundError("The data path does not exist. please download datas from" +
                            " https://github.com/gmt-china/china-geospatial-data/releases")
# 确定工作平台
platform_name = platform.system()
if platform_name == "Windows":
    data_path = os.path.join(data_path, "china-geospatial-data-UTF8")
elif platform_name == "Linux" or "Darwin":
    data_path = os.path.join(data_path, "china-geospatial-data-GB2312")
else:
    raise ValueError("The platform is not supported.")

class GMTData:
    def __init__(self, fault_data, fault_df, path):
        self.fault_dict = fault_data
        self.fault_df = fault_df
        self.path = path

def read_gmt(filename:str, columns:int or tuple[str]=None):
    if not filename.endswith("gmt"):
        raise ValueError("The filename must end with .gmt")
    filepath = os.path.join(data_path, filename)
    if not os.path.isfile(filepath):
        raise FileNotFoundError("The file does not exist.")

    # 初始化数据存储结构
    faults_attributes = []
    faults_data = {}
    columns = []
    number = -1

    # 读取文件（假设数据存储在'text_file.txt'）
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip() # 去除行首的空格
            # 提取列名
            if line.startswith("# @NOBJECTID"):
                columns = line.split("|")[1:]
            # 匹配以@D开头的数据行，提取断层编号、名称等属性信息
            elif line.startswith(">"):
                number += 1
                # 初始化当前断层的坐标数据
                if columns is None:
                    columns = [str(i) for i in range(20)]
                faults_data[number] = []
                fault_info = {}
            elif line.startswith('# @D'):
                # 提取断层编码和属性信息
                parts = line[2:].split('|')[1:]
                # 添加断层属性到列表
                for index, column in enumerate(columns):
                    fault_info[column] = parts[index] if index < len(parts) else None
                faults_attributes.append(fault_info)
            # 提取坐标数据
            elif re.match(r'^[0-9\.]+ [0-9\.]+$', line):
                coords = list(map(float, line.split()))
                faults_data[number].append(coords)
    for key, data in faults_data.items():
        faults_data[key] = np.array(data)
    faults_df = pd.DataFrame(faults_attributes)
    return GMTData(fault_data=faults_data, fault_df=faults_df, path=filepath)

CN_fault = read_gmt("CN-faults.gmt")
CN_border_L1 = read_gmt("CN-border-L1.gmt")
CN_border_La = read_gmt("CN-border-La.gmt")
CN_block_L1 = read_gmt("CN-block-L1.gmt")
CN_block_L2 = read_gmt("CN-block-L2.gmt")
CN_block_L1_deduced = read_gmt("CN-block-L1-deduced.gmt")
geo3al = read_gmt("geo3al.gmt")