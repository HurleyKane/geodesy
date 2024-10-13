"""
将matplotlib绘制的图像转换成地理学中常用的一些格式功能
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def _format_lat_lon_x(value, pos=None):
    """
    自定义函数，将 x 轴坐标值转换为带有东/西方向的经度字符串
    value: x 轴坐标值
    pos: 坐标的索引
    """
    if value >= 0:
        return f'{value:.2f}°E'
    else:
        return f'{-value:.2f}°W'


def _format_lat_lon_y(value, pos=None):
    """
    自定义函数，将 y 轴坐标值转换为带有北/南方向的纬度字符串
    value: y 轴坐标值
    pos: 坐标的索引
    """
    if value >= 0:
        return f'{value:.2f}°N'
    else:
        return f'{-value:.2f}°S'

def ticks_transform_to_lat_lon(ax = None):
    """
    将坐标轴刻度格式化为带有经度和纬度方向的字符串
    ax: matplotlib 的坐标轴对象
    """
    # 设置 x 轴刻度格式化器为经度格式化函数
    if ax is not None:
        formatter_x = FuncFormatter(_format_lat_lon_x)
        ax.xaxis.set_major_formatter(formatter_x)

        # 设置 y 轴刻度格式化器为纬度格式化函数
        formatter_y = FuncFormatter(_format_lat_lon_y)
        ax.yaxis.set_major_formatter(formatter_y)

class GeoDataArray_plot_setting:
    xticks_num = 3
    yticks_num = 3
    def __init__(
            self,
            geodataarray,
            ax : plt.Axes | None  = None,
        ) -> None:
        self._gda = geodataarray

        # 将刻度转换为经纬度
        ticks_transform_to_lat_lon(ax)

        # 调整ticks的可见刻度数量
        self.ticks_setting(xticks_num=GeoDataArray_plot_setting.xticks_num,
                           yticks_num=GeoDataArray_plot_setting.yticks_num)

    def ticks_setting(
            self,
            xticks_num : int | None = None,
            yticks_num : int | None = None,
        ):
        lon0, lon1, lat0, lat1 = self._gda.region
        x_step = (lon1 - lon0) / (xticks_num + 1)
        y_step = (lat1 - lat0) / (yticks_num + 1)

        lon0_hat = lon0 + x_step/2
        lon1_hat = lon1 - x_step/2
        lat0_hat = lat0 + y_step/2
        lat1_hat = lat1 - y_step/2
        x_step_hat = (lon1_hat - lon0_hat) / (xticks_num - 1)
        y_step_hat = (lat1_hat - lat0_hat) / (yticks_num - 1)
        x_coord_hat = [(lon0_hat + i * x_step_hat) for i in range(0, xticks_num)]
        y_coord_hat = [(lat0_hat + i * y_step_hat) for i in range(0, yticks_num)]
        plt.xticks(x_coord_hat)
        plt.yticks(y_coord_hat)

def add_shared_colorbar(im, axes, label='deformation/m', size="5%", pad=0.05, orientation='vertical', extend=None):
    """
    为多个子图添加一个共享的等高或等宽 colorbar。

    参数：
    - im: 由 `imshow` 或其他绘图函数返回的图像对象。
    - axes: 子图的轴 (`Axes`) 对象列表或二维数组。
    - label: colorbar 标签，默认为 'Colorbar'。
    - size: colorbar 的宽度或高度，默认为 '5%'。
    - pad: colorbar 和子图之间的间距，默认为 0.05。
    - orientation: colorbar 的方向，默认为 'vertical'（垂直），也可以设置为 'horizontal'（水平）。
    - extend: 设置 colorbar 的延伸部分，可以为 'both', 'min', 'max', 或 'neither'。
    """
    # 确保axes是一个展平的一维列表
    if isinstance(axes, (list, np.ndarray)):
        if isinstance(axes[0], (list, np.ndarray)):
            axes = np.array(axes).flatten()

    # 获取figure对象
    fig = axes[0].figure

    if orientation == 'vertical':
        # 找到所有子图的最右侧和最底部位置
        x0 = max(ax.get_position().x1 for ax in axes) + pad
        x1 = x0 + float(size.strip('%')) / 100 * (axes[0].get_position().x1 - axes[0].get_position().x0)
        y0 = min(ax.get_position().y0 for ax in axes)
        height = max(ax.get_position().y1 for ax in axes) - y0
        cax = fig.add_axes([x0, y0, x1 - x0, height])
    else:  # orientation == 'horizontal'
        # 找到所有子图的最底部和最右侧位置
        y0 = min(ax.get_position().y0 for ax in axes) - pad
        y1 = y0 - float(size.strip('%')) / 100 * (axes[0].get_position().y1 - axes[0].get_position().y0)
        x0 = min(ax.get_position().x0 for ax in axes)
        x1 = max(ax.get_position().x1 for ax in axes)
        cax = fig.add_axes([x0, y1, x1 - x0, abs(y0 - y1)])

    cbar = fig.colorbar(im, cax=cax, orientation=orientation, extend=extend)
    cbar.set_label(label)
    return cbar

def add_label_subplots(fig_or_axes, labels=None, label_args=None, offset=(0.03, 0.88)):
    """
    为给定的图形或子图列表中的子图添加顺序标签（a, b, c, d等）。
    偏移量按子图的百分比调整，(0, 0) 代表子图左下角，(1, 1) 代表右上角，标签保持在子图范围内。

    参数:
        fig_or_axes (matplotlib.figure.Figure 或 list of matplotlib.axes.Axes): 包含子图的图形或子图的列表。
        labels (list of str, 可选): 自定义标签。如果为 None，则使用默认标签（'a', 'b', 'c'等）。
        label_args (dict, 可选): 传递给 plt.text 的其他参数（例如，fontsize, fontweight）。
        offset (tuple of float, 可选): 标签在子图中的相对位置（x, y），默认位置为子图左上角（0.05, 0.95）。
    """
    # 判断输入是图形对象还是子图列表
    if isinstance(fig_or_axes, plt.Figure):
        axes = fig_or_axes.axes  # 如果是图形对象，获取所有子图
    else:
        axes = fig_or_axes  # 如果是子图列表，直接使用
        if isinstance(axes, (list, np.ndarray)):
            if isinstance(axes[0], (list, np.ndarray)):
               axes = np.array(axes).flatten()

    # 如果没有提供自定义标签，则生成默认的 'a', 'b', 'c' 等标签
    if labels is None:
        labels = [f"({chr(i)})" for i in range(97, 97 + len(axes))]

    # 如果没有提供 label_args，则使用默认的字体大小和粗细
    if label_args is None:
        label_args = {'fontsize': 14}

    # 为每个子图添加标签
    for ax, label in zip(axes, labels):
        ax_coords = ax.get_position()  # 获取子图的位置

        # 根据偏移量和子图的大小计算标签的实际位置
        x_pos = ax_coords.x0 + offset[0] * (ax_coords.x1 - ax_coords.x0)
        y_pos = ax_coords.y0 + offset[1] * (ax_coords.y1 - ax_coords.y0)

        # 添加标签并确保其在子图范围内
        ax.figure.text(x_pos, y_pos, label, **label_args)


if __name__ == '__main__':
    # 创建示例数据
    lons = [115.8575, 116.4074, 121.4737]
    lats = [39.9042, 39.9042, 31.2304]

    # 绘制经纬度坐标点
    plt.scatter(lons, lats)

    # 设置坐标轴的格式化器
    ticks_transform_to_lat_lon(plt.gca())

    # 添加标签和标题
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Locations')

    # 显示图形
    plt.show()
