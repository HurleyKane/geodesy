"""
处理霍夫变换识别的断层线
"""
from __future__ import annotations
import re
import json, pickle
import numpy as np
from sklearn.cluster import KMeans
from piecewise_regression.main import Fit
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import sympy as sp
from geodesy.earthquake.FaultLines import FaultLines

def polyline_expression():
    # 初始化 LaTeX 打印
    from sympy.abc import i
    sp.init_printing(use_latex="mathjax")

    # 定义一个函数来创建表达式
    # 定义符号变量
    xx = sp.Symbol('x')
    constant = sp.Symbol('constant')
    alpha = sp.IndexedBase('alpha')
    breakpoint = sp.IndexedBase('breakpoint')
    k = sp.symbols('k', integer=True)

    # 构建表达式
    expr = constant + alpha[1] * xx

    # 添加求和项
    sum_expr = sp.Sum((alpha[i + 1] - alpha[i]) * sp.Max(xx - breakpoint[i], 0), (i, 1, k))
    expr += sum_expr

    # # 添加误差项
    # expr += epsilon
    return expr

class FitLine:
    def __init__(
            self,
            expression,
            difinational_domain,
            data_vars : dict,
            fit_parameters : Fit = None,
    ) -> None:
        self.expression = expression
        self.difinational_domain = difinational_domain
        self.data_vars = data_vars
        self.fit_parameters : Fit = fit_parameters

    @property
    def numpy_function(self):
        x = sp.symbols("x")
        func = sp.lambdify(x, self.expression, modules={"Max": np.maximum})
        return func

    def plot_line(self, linewidth = None, color : str = None):
        x_value, y_value = self.get_xy_values()
        plt.plot(x_value, y_value, linewidth=linewidth, color=color)
        plt.gca().set_aspect('equal')
        plt.gca().invert_yaxis()

    def get_xy_values(self):
        x_value = np.arange(self.difinational_domain[0], self.difinational_domain[1]+1)
        y_value = self.numpy_function(x_value)
        return x_value, y_value

    def encoding(
            self,
            shape : tuple[int] | list[int] = None
    ) -> np.ndarray:
        """对单一直线进行编码"""
        quadrature_array = np.full(shape, "None", dtype = "U30")
        x_values, y_values = np.meshgrid( np.arange(shape[1]), np.arange(shape[0]))
        y = self.numpy_function(x_values)
        in_difinational_domain = np.logical_and(self.difinational_domain[0] < x_values, x_values < self.difinational_domain[1])
        not_in_difinational_domain = np.logical_not(in_difinational_domain)
        quadrature_array[np.logical_and(y_values < y,  in_difinational_domain)] = "0"  # 在直线上方, 定义域内
        quadrature_array[np.logical_and(y_values >= y, in_difinational_domain)] = "1"  # 在直线下方, 定义域内
        quadrature_array[np.logical_and(y_values < y,  not_in_difinational_domain)] = "2"  # 在直线上方, 定义域外
        quadrature_array[np.logical_and(y_values >= y, not_in_difinational_domain)] = "3"  # 在直线下方, 定义域外
        return quadrature_array
    
    def encoding_and_decoding(
            self,
            shape
    ):
        quadrature_array = self.encoding(shape)

        def trinary_to_decimal(trinary_str):
            return int(trinary_str, base=4)

        vectorized_ternary_to_decimal = np.vectorize(trinary_to_decimal)
        decimal_array = vectorized_ternary_to_decimal(quadrature_array)
        return decimal_array


class FitLines:
    def __init__(
            self,
            fitlines: list[FitLine],
    ) -> None:
        self.values = fitlines
        self._fitlines_position = None
        self._code_array = None

    @property
    def numpy_functions(self):
        functions = []
        for fitline in self.values:
            functions.append(fitline.numpy_function)
        return functions

    @property
    def size(self):
        return len(self.values)

    @property
    def fitlines_position(self):
        # 获取拟合的各个线段的表达式对应的像素
        return self._fitlines_position

    @fitlines_position.setter
    def fitlines_position(self, shape):
        if self._fitlines_position is None:
            self._fitlines_position = self.get_the_fitlines_numbers(shape=shape)

    @property
    def code_array(self):
        return self._code_array

    @code_array.setter
    def code_array(self, shape):
        if self._code_array is None:
            self._code_array = self.encoding(shape)

    def __getitem__(self, item):
        return self.values[item]


    def plot_lines(self, linewitdh=None, color : list[str] | str = None):
        length = len(self.values)
        for index, fitline in enumerate(self.values):
            if color is not None:
                if type(color) is list:
                    fitline.plot_line(linewidth=linewitdh, color=color[index])
                elif type(color) is str:
                    fitline.plot_line(linewidth=linewitdh, color=color)
                else:
                    raise ValueError("colorbar must be list or str")
            else:
                fitline.plot_line(linewidth=linewitdh)
        plt.gca().set_aspect('equal')
        plt.gca().invert_yaxis()

    def plot_code_images(self, shape:tuple[int] | list[int]):
        decimal_codes = self.encoding_and_decoding(shape)
        plt.imshow(decimal_codes)

    def encoding(
            self,
            shape : tuple[int] | list[int] = None
    ) -> np.ndarray :
        """进行编码"""
        quadrature_array = self.values[0].encoding(shape)
        if self.size > 1:
            for index in range(1, self.size):
                temp_image = self.values[index].encoding(shape)
                quadrature_array = np.char.add(quadrature_array, temp_image)
        return quadrature_array
        
    def encoding_and_decoding(
            self,
            shape : tuple[int] | list[int] = None,
    ) -> np.ndarray:
        """编码后解码，获取四进制的数值"""
        quadrature = self.values[0].encoding(shape)
        if self.size > 1:
            for index in range(1, self.size):
                temp_image = self.values[index].encoding(shape)
                quadrature = np.char.add(quadrature, temp_image)

        def trinary_to_decimal(trinary_str):
            return int(trinary_str, base = 4)
    
        vectorized_ternary_to_decimal = np.vectorize(trinary_to_decimal)
        decimal_array = vectorized_ternary_to_decimal(quadrature)
        return decimal_array

    def get_the_fitlines_numbers(
            self,
            shape : tuple[int] | list[int] = None
    ) -> np.ndarray:
        """对拟合的各个线段的表达式对应的像素进行编号"""
        fitline_numbers = np.zeros(shape, dtype = "int")
        for index, fitline in enumerate(self.values):
            x_value, y_value = fitline.get_xy_values()
            y_value = np.int32(y_value)
            fitline_numbers[y_value, x_value] = index + 1
        return fitline_numbers

    def save_to_pickle(self, fname:str) -> None:
        with open(fname, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_pickle(fname:str) -> FitLines:
        with open(fname, "rb") as f:
            faultlines = pickle.load(f)
        return faultlines

    def get_homogeneousPointsubs_and_status_about_fault(self, r, c, gwin_r, gwin_c):
        fault_line_num = np.unique(self.fitlines_position[gwin_r, gwin_c])
        if len(fault_line_num) == 0: # 证明该点处不在断层附近
            is_homogeneousPoint = np.full_like(gwin_r.ravel(), True)
            return is_homogeneousPoint
        else: # 证明该点为断层附近的点
            gwin_code_erea = self.code_array[gwin_r, gwin_c].ravel()
            fault_line_num = fault_line_num[1:] - 1
            code = self.code_array[r, c]  # 取出该点的编码
            match_module = ""
            for index, c in enumerate(code):
                if index in fault_line_num:
                    match_module += c
                else:
                    match_module += "."
                # 定义正则表达式模式
            pattern = re.compile(match_module)

            # 使用 np.vectorize 和正则表达式匹配
            match_func = np.vectorize(lambda x: bool(pattern.search(x)))

            # 找出匹配模式的布尔数组
            is_homogeneousPoint = match_func(gwin_code_erea)

        return is_homogeneousPoint

def draw_fault_lines(data:np.ndarray=None, vmin=None, vmax=None, save_json:str=None, fig=None, ax=None):
    """绘制断层线"""
    # 创建一个按钮的回调函数
    def on_button_click(event):
        # 这里执行您的响应事件
        nonlocal event_active, lines, button1
        if event_active:
            print("已断开事件")
            event_active = False
            button1.label.set_text("execute")
            points.pop()
            if len(points) >= 2: lines.pop().remove()
            if len(points) == 0: scatters.pop().remove()
            fig.canvas.draw()  # 刷新图形
        else:
            print("已激活事件")
            update_fig()
            event_active = True
            button1.label.set_text("paused")

    # 定义一个绘制函数
    def on_draw_figure_click(event):
        if event_active:
            nonlocal lines, scatters
            points.append((event.xdata, event.ydata))
            if len(points) == 1:
                pair_points = points[-1:]
                x, y = zip(*pair_points)  # 将点坐标分开
                scatter = ax.scatter(x, y, color=rainbow_colors[fault_line_number-1])
                scatters.append(scatter)
            elif len(points) >= 2:
                pair_points = points[-2:]
                x, y = zip(*pair_points)  # 将点坐标分开
                line = ax.plot(x, y, rainbow_colors[fault_line_number-1])  # 绘制直线
                lines.append(*line)
                fig.canvas.draw()  # 刷新图形

    def on_button_back_last_step_click(event):
        nonlocal event_active, lines, button2, points
        if event_active:
            # 预先删除多余
            points.pop()
            if len(points) >= 2: lines.pop().remove()
            # 开始判断
        if len(points) > 0:
            points.pop()
        if len(lines) > 0:
            lines.pop().remove()
        if len(points) == 0:
            scatters.pop().remove()
        update_fig()

    def update_fig():
        nonlocal fig, ax
        ax.set_title("fault line:" + str(fault_line_number), loc="left")
        fig.canvas.draw()  # 刷新图形

    # 定义一个用来切换采集类别的键盘响应函数
    def switch_fault_line_on_key(event):
        nonlocal fault_line_number, total_points, points, lines
        temp_points = points[:]
        if event.key == "enter" and len(points) != 0:
            print("fault line:" + str(fault_line_number), " ", "points:", len(points), " _lines:", len(lines))
            fault_line_number += 1
            total_points.append(temp_points)
            points.clear()
            lines.clear()
            update_fig()
            print("切换断层线")

    # 一些全局变量参数
    points = []
    lines = []
    scatters = []
    total_points = []
    rainbow_colors = ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#0000FF', '#4B0082', '#9400D3'] * 20 # 彩虹颜色组
    fault_line_number = 1
    event_active = False # 初始化按钮状态

    # 创建一个Matplotlib图形
    if fig == None and ax == None:
        fig, ax = plt.subplots()

    # 外部输入参数进行绘图
    if data is not None:
        plt.imshow(data, cmap="jet", vmin=vmin, vmax=vmax)

    #定义一个开始采集数据的按钮
    button_ax1 = plt.axes([0.45, 0.901, 0.1, 0.075])
    button1 = Button(button_ax1, 'execute')

    # 定义一个返回上一步的采集断裂线的按钮
    button_back_ax2 = plt.axes([0.56, 0.901, 0.1, 0.075])
    button2 = Button(button_back_ax2, "back")

    # 活动链接
    button1.on_clicked(on_button_click)
    button2.on_clicked(on_button_back_last_step_click)
    fig.canvas.mpl_connect('button_press_event', on_draw_figure_click)
    fig.canvas.mpl_connect("key_press_event", switch_fault_line_on_key)

    # 图层设置
    if data is None:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
    plt.show(block=True)

    if save_json is not None:
        with open(save_json, "w") as f:
            json.dump(total_points, f)
        print("数据已保存")

    return total_points


# 现在需要创建一个
class faultLinesfit:
    """对获取的断裂线进行一系列的处理"""
    def __init__(
            self,
            lines : FaultLines | list | tuple = None,
            linesFilePath = None,
            pointTypeNum = 3,
            isImshow = False
    ) -> None :
        """
        _lines : [[(x1, x2)), (y1, y2)]]
        """
        if type(lines) is list:
            self.lines = FaultLines(lines=lines)
        else:
            self.lines = lines
        self.pointTypeNum = pointTypeNum
        self.isImshow =isImshow
        if linesFilePath is not None:
            self.lines = FaultLines.load_from_json(linesFilePath)
        self.fitLineMinXs = []
        self.fitLineMaxXs = []
        self.fitLineCofficients = []  # 拟合直线相关系数

    def kmeans(
            self,
            plot : bool = False,
            extent : list = None
        ) -> dict[str, FaultLines]:
        KB = self.lines.getKB()
        ## 利用kmeans算法对线段进行分类
        kmeans = KMeans(n_clusters=self.pointTypeNum)
        kmeans.fit(KB)
        labels = np.array(kmeans.labels_)
        lines_dict = {}
        for i in range((self.pointTypeNum)):
            lines_dict[i] = []
        for index, line in enumerate(self.lines):
            lines_dict[labels[index]].append(line)
        labels_unique = np.unique(labels)
        for index in labels_unique:
            lines_dict[index] = FaultLines(lines_dict[index])
        if plot:
            color = ["blue", "red", "green", "yellow", "cyan", "magenta"]
            color += color
            temp = None

            for key in lines_dict.keys():
                if key != temp:
                    temp = key
                    label = "line_" + str(key)
                else:
                    label = None
                lines_dict[key].plot_lines(color=color[key], label=label)
            plt.legend()
            plt.title("kmeans lines")
            plt.gca().invert_yaxis()
            plt.gca().set_aspect("equal")
            if extent is not None:
                plt.axis(extent)
        return lines_dict

    def get_different_lines_coords(self, lines_dict: dict[str, FaultLines]):
        xs, ys = [], []
        for key in sorted(lines_dict.keys()):
            lines = lines_dict[key]
            x, y = lines.get_coords_from_lines()
            xs.append(x)
            ys.append(y)
        return xs, ys

    def fitLines_to_one_curve_line(self, plot : bool = False):
        """
        将直线拟合成一条高阶的直线, 这里采用了二阶的曲线形式
        """
        lines_dict = self.kmeans()
        xs, ys = self.get_different_lines_coords(lines_dict)
        for i in range(self.pointTypeNum):
            x = np.array(xs[i]).reshape(-1, 1)
            l = np.array(ys[i]).reshape(-1, 1)
            b = np.hstack([x**2, x, np.ones_like(x)])
            p = np.eye(len(x))
            from geodesy import robust_estimation
            results =  robust_estimation(A=b, P=p, P_total=[], l=l, atol=0.01)
            cofficient = results[0]
            self.fitLineCofficients.append(cofficient)
            x2 = np.array([i for i in range(np.min(x), np.max(x)+1)])
            y2 = cofficient[0] * (x2**2) + cofficient[1] * x2 +cofficient[2]
            if self.isImshow:
                plt.plot(x2, y2, linewidth=2, label="fitted fault line", color="red")
            self.fitLineMinXs.append(np.min(x))
            self.fitLineMaxXs.append(np.max(x))

    @staticmethod
    def fit_to_ployline(
            lines : FaultLines,
            strat_values = None,
            n_breakpoints = None
    ) -> Fit:
        """
        对lines对象拟合一条多段想
        """
        try :
            import piecewise_regression
        except:
            print("please install piecewise_regression module !")
            return None
        # 第一步：进行拟合
        x, y = lines.get_coords_from_lines()
        pw_fit = piecewise_regression.Fit(x, y, strat_values,  n_breakpoints=n_breakpoints)
        results = pw_fit.get_results()["estimates"]
        keys = sorted(list(results.keys()))
        alphas_key = sorted([item for item in keys if item.startswith('alpha')])
        breakpoints_key = sorted([item for item in keys if item.startswith('breakpoint')])
        k_value = pw_fit.n_breakpoints

        # 第二步：给出多段线表达式
        ## 创建变量
        k = sp.symbols('k', integer=True)
        constant = sp.Symbol('constant')
        alpha = sp.IndexedBase('alpha')
        breakpoint = sp.IndexedBase("breakpoint")

        ## 将值传入表达式
        subs_dict = {}
        # subs_dict.update({k : k_value})
        constant_value = results[f"const"]["estimate"]
        subs_dict.update({constant : constant_value})
        for key in alphas_key:
            num = int(key[len("alpha"):])
            subs_dict.update({alpha[num] : results[f"alpha{num}"]["estimate"]})
        for key in breakpoints_key:
            num = int(key[len("breakpoint"):])
            subs_dict.update({breakpoint[num] : results[f"breakpoint{num}"]["estimate"]})

        expression = polyline_expression().subs({k : k_value}).doit()
        expression = expression.subs(subs_dict)

        fitline = FitLine(expression = expression, difinational_domain = lines.definitional_domain, data_vars=subs_dict,
                          fit_parameters=pw_fit
                          )
        return fitline

    @staticmethod
    def fit_to_polylines(
            lines_dict: dict[str, FaultLines],
            breakpoint_list : list[int],
    ) -> FitLines:
        """
        拟合多段线
        """
        fitLines_lists = []
        for index, key in enumerate(sorted(lines_dict.keys())):
            temp_expression = faultLinesfit.fit_to_ployline(lines_dict[key], n_breakpoints=breakpoint_list[index])
            fitLines_lists.append(temp_expression)
        fitLines = FitLines(fitLines_lists)
        return fitLines


    def getHomogeneousPoints(self, rowLength, columnLength, isImshow:bool=False):
        """0代表断裂线的位置"""
        homogeneousPoints = np.ones((rowLength, columnLength))
        for i in range(self.pointTypeNum):
            xs = np.int32(np.arange(self.fitLineMinXs[i], self.fitLineMaxXs[i]+1))
            cofficients = self.fitLineCofficients[i]
            for x in np.nditer(xs):
                y0 = np.int32(cofficients[0, 0] * x**2 + cofficients[1, 0] * x + cofficients[2, 0])
                y1 = np.int32(cofficients[0, 0] * (x+1)**2 + cofficients[1, 0] * (x+1) + cofficients[2, 0])
                if y1 < y0:
                    temp = y0
                    y0 = y1
                    y1 = temp
                for y in range(y0, y1+1):
                    homogeneousPoints[y, x] = 0
        if isImshow:
            fig = plt.figure()
            fig.set_size_inches(5, 5)
            ax = plt.subplot()
            plt.imshow(homogeneousPoints)
            plt.show()
        return homogeneousPoints


if __name__ == "__main__":
    #%%
    filename = "./test/fault_lines.json"
    faultlines = FaultLines.load_from_json(filename)
    faultLinesFitbandle = faultLinesfit(faultlines, pointTypeNum=2, isImshow=True)
    lines_dict = faultLinesFitbandle.kmeans(True)
    plt.show()
    #%%  拟合直线
    fitlines = faultLinesFitbandle.fit_to_polylines(lines_dict, breakpoint_list=[5, 1])
    fitlines.plot_lines()
    #%% 保存直线
    fitlines.save_to_pickle("./test/fitfaultlines.pk1")
    #%%
    from InSARlib.core import FaultLines
    fitlines = FaultLines.load_from_json("./test/fitfaultlines.pk1")
    fitlines.plot_lines()
