from __future__ import annotations

import sys
import numpy as np
from osgeo import gdal
import xarray as xr


def read_tiff(
        filename,
        properties_name : list[str] | None = None,
) -> xr.Dataset:
    geo_data = gdal.Open(filename)
    GeoTransform = geo_data.GetGeoTransform()
    bandnum = geo_data.RasterCount
    if properties_name is None or len(properties_name) == bandnum:
        pass
    else:
        print("please check properties_name parameter!")
        sys.exit()

    # 获取经纬度
    XSize = geo_data.RasterXSize
    YSize = geo_data.RasterYSize
    clon = np.array([GeoTransform[0] + x * GeoTransform[1] for x in range(XSize)])
    clat = np.array([GeoTransform[3] + y * GeoTransform[5] for y in range(YSize)])

    data_vars = {}
    filename = filename.split(".")[-2]
    for i in range(bandnum):
        temp_var = geo_data.GetRasterBand(i+1).ReadAsArray()[:, :]
        if properties_name is not None:
            name = properties_name[i]
        else:
            name = f"{filename}_band{i}"
        data_vars[name] = (["clat", "clon"], temp_var)
    dataset = xr.Dataset(
        data_vars = data_vars,
        coords = {
            "clat" : clat,
            "clon" : clon
        }
    )
    return dataset

def read_tiffs(
        filenames : list[str],
        properties_name : list[list[str]] | None = None,
) -> xr.Dataset:
    dataarrays = []
    for i, filename in enumerate(filenames):
        temp_dataarray = read_tiff(filename, properties_name[i])
        dataarrays.append(temp_dataarray)
    return xr.merge(dataarrays)


if __name__ == "__main__":
    filenames = [
        "results/coseismic/000-east.grd",
        "results/coseismic/000-north.grd",
        "results/coseismic/000-up.grd",
    ]
    dataset = read_tiffs(filenames, [["east"], ["north"], ["up"]])