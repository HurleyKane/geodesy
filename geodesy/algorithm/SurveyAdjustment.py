"""
测量平差基础库
"""
import numpy as np

def calculate_rmse(y_true, y_pred):
    """
    计算均方根误差（RMSE）。

    参数：
    - y_true: 实际值数组或列表，可能包含 NaN 值。
    - y_pred: 预测值数组或列表，可能包含 NaN 值。

    返回：
    - RMSE 值。
    """
    # 将输入转换为 numpy 数组（如果它们不是的话）
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 确保 y_true 和 y_pred 的形状相同
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"

    # 计算差值，并忽略 NaN 值
    diff = y_true - y_pred
    diff[~np.isfinite(diff)] = 0  # 将非有限值（包括 NaN 和无穷大）替换为 0

    # 计算均方误差（MSE），使用 np.nanmean 忽略 NaN 值
    mse = np.nanmean(diff ** 2)

    # 计算 RMSE
    rmse = np.sqrt(mse)

    return rmse

