import os
filename = "./datas/Simulation experiment's initial database.nc"
from geodesy import MultiInSARDataset
data = MultiInSARDataset.loadtxt(filename)
data1 = data.extract_region_datas(data.region)