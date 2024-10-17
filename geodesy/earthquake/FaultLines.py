from __future__ import annotations
import sys
import numpy as np
import matplotlib.pyplot as plt


def _get_coords_from_two_points(point1, point2, isImshow:bool=False) -> list[tuple]:
    """计算两直线之间经过的像素的坐标"""
    # pixel_coordinates = []
    x_list, y_list = [], []

    # 计算直线的斜率和截距
    if point2[0] - point1[0] != 0:
        m = (point2[1] - point1[1]) / (point2[0] - point1[0])
        b = point1[1] - m * point1[0]

        # 计算直线的 x 和 y 坐标范围
        xs = np.int32(np.array([point1[0], point2[0]]))
        ys = np.int32(np.array([point1[1], point2[1]]))
        x_range = np.int32(np.arange(np.ceil(np.min(xs)), np.ceil(np.max(xs)) + 1))
        y0_range = np.ceil(m * x_range + b)
        y1_range = np.ceil(m * (x_range + 1) + b)

        # 返回像素坐标的元组列表
        # pixel_coordinates = [(int(x), int(y)) for x, y in zip(x_range, y_range)]
        # 对y进行调整
        if (y1_range < y0_range).any():
            temp = y0_range
            y0_range = y1_range
            y1_range = temp
        logical = y1_range > np.max(ys)
        y1_range[logical] = np.max(ys)
        for index in range(len(x_range)):
            x = x_range[index]
            try:
                for y in range(int(y0_range[index]), int(y1_range[index]) + 1):
                    x_list.append(x)
                    y_list.append(y)
            except:
                pass

        if isImshow:
            image = np.zeros([int(np.max(y1_range) + 1), int(np.max(x_range) + 1)])
            image[y_list][x_list] = 1
            plt.subplots(1, 1)
            plt.imshow(image)
            plt.show()
    else:
        y0 = min(point1[1], point2[1])
        y1 = max(point1[1], point2[1])
        for y in range(int(y0), int(y1)):
            x_list.append(point1[0])
            y_list.append(int(y))

    return x_list, y_list

def _are_similar_lines(line1, line2, threshold=3):
    # 计算线段之间的位置差异
    x1_diff = abs(line1[0][0] - line2[0][0])
    y1_diff = abs(line1[0][1] - line2[0][1])
    x2_diff = abs(line1[1][0] - line2[1][0])
    y2_diff = abs(line1[1][1] - line2[1][1])

    # 如果任何一个端点的差异超过阈值，则认为不相似
    if x1_diff > threshold or y1_diff > threshold or x2_diff > threshold or y2_diff > threshold:
        return False
    else:
        return True


class FaultLines:
    """断层线对象"""
    __slots__ = ["_lines", "index"]
    def __init__(
            self,
            lines : list | tuple | FaultLines
    ) -> None:
        """"
        _lines' structure : [point1, point2], point = [x, y]
        """
        if type(lines) == FaultLines:
            self._lines = lines.values
        if self.checkFaultLine(lines):
            self._lines = lines
            self._correct_points()
        else:
            print('Fault Line is error!')
            sys.exit()

    @property
    def values(self):
        return self._lines

    def __add__(
            self,
            other : FaultLines
    ) -> FaultLines :
        values = self.values + other.values
        return FaultLines(lines=values)

    def __repr__(self):
        return "Fault_Lines : " + str(self.values) + "\n"

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self):
            values = self[self.index]
            self.index += 1
            return values
        else:
            raise StopIteration

    def __getitem__(self, item):
        return self.values[item]

    def __len__(self):
        return len(self.values)

    @property
    def size(self):
        return len(self.values)

    @property
    def definitional_domain(self):
        x, _ = self.get_coords_from_lines()
        x0 = np.min(x)
        x1 = np.max(x)
        return [x0, x1]

    @property
    def value_domain(self):
        _, y = self.get_coords_from_lines()
        y0 = np.min(y)
        y1 = np.max(y)
        return [y0, y1]

    @staticmethod
    def load_from_json(filename : str) -> FaultLines :
        import json
        lines = json.load(open(filename, "rb"))
        return FaultLines(lines)

    def save_to_json(self, filename : str) -> None:
        import json
        json.dump(self.values, open(filename, "wb"))

    @staticmethod
    def checkFaultLine(lines) -> bool:
        if type(lines) == FaultLines:
            lines = lines.values
        try:
            line_nd = np.array(lines)
            if len(line_nd.shape) != 3:
                return False
            else:
                if line_nd.shape[1] == 2 and line_nd.shape[2] == 2:
                    return True
                else:
                    return False
        except:
            return False

    def _correct_points(self):
        new_lines = []
        for point in self._lines:
            point1, point2 = point
            x1, y1 = point1
            x2, y2 = point2
            new_lines.append([[x1, y1], [x2, y2]])
        self._lines = new_lines

    def getKB(self):
        KB = []
        ## 绘制线段
        for line in self._lines:
            # y = kx + b1
            x1, y1 = line[0]
            x2, y2 = line[1]
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            ## 算出直线的k和b1,斜率theta
            k = (y2-y1)/(x2-x1)
            b1 = (x2*y1-x1*y2)/(x2-x1)
            theta = np.arctan((y2-y1)/(x2-x1)) * 180 / np.pi
            ## 记录映射到(k, b)的参数空间中，每个直线可以用点来表示
            KB.append([k, b1])
        return np.array(KB)
    def plot_lines_kb(self, color : str = 'blue'):
        KBs = self.getKB()
        plt.scatter(KBs[:, 0], KBs[:, 1], color = color)

    def plot_lines(
            self,
            color : str = 'blue',
            linewidth : float=1,
            label : str | None = None
        ) -> None:
        for index, point in enumerate(self):
            try:
                (x1, y1), (x2, y2) = point
            except:
                point1, point2 = point
                x1, y1 = point1
                x2, y2 = point2
            if index != 0:
                label = None
            plt.plot((x1, x2), (y1, y2), color=color, linewidth=linewidth, label=label)
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')

    def remove_line(self, line:list | tuple) -> None:
        self.values.remove(line)

    def filter_similar_line(self, line, threshold=3) -> bool:
        """过滤掉相似线段"""
        status = False
        for target_line in self.values:
            are_similar = _are_similar_lines(line, target_line, threshold)
            if are_similar:
                self.remove_line(target_line)
                status = True
        return status

    def filter_similar_lines(self, lines : list | tuple | FaultLines, threshold=3):
        """过滤掉复数的线段"""
        if type(lines) == FaultLines:
            lines = lines.values
        for line in lines:
            self.filter_similar_line(line, threshold)


    def get_coords_from_lines(self):
        x, y = [], []
        total_points = [np.ceil(points) for points in self.values]
        for points in total_points:
            for i in range(len(points)-1):
                x_list, y_list = _get_coords_from_two_points(point1=points[i], point2=points[i + 1])
                x += x_list
                y += y_list
        return x, y


    def get_pixels_from_lines(self, shape:tuple=None, isImshow:bool=False):
        """
        根据不同的点，找出对应的shape对应的位置
        total_points: 存放点的列表
        """
        xs, ys = self.get_coords_from_lines()
        if shape is not None:
            image = np.ones(shape)
        else:
            image = np.ones([int(np.max(xs) + 1), int(np.max(ys) + 1)])
        try:
            image[int(xs)][int(ys)] = 0  # 0代表线段存在的位置
        except:
            pass
        if isImshow:
            plt.imshow(image)
            plt.show(block=True)
        return image

    # def


class MutiFaultLines:
    def __init__(
            self,
            faultlines_list : list[FaultLines],
    ):
        self.faultlines_list = faultlines_list
