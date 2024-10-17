"""
这是一个获取断层线的库,依赖于cv2.
"""
from __future__ import  annotations
import numpy as np
from scipy import ndimage, signal
import matplotlib.pyplot as plt
from tqdm import tqdm

from geodesy.earthquake import FaultLines

def non_maximization_suppress(gradient_magnitude:np.ndarray, gradient_direction:np.ndarray):
    """
    遍历像素点并执行非极大值抑制, 理论参考：https://zhuanlan.zhihu.com/p/447565904
    """
    new_magnitude = np.zeros_like(gradient_magnitude)
    with tqdm(range((gradient_magnitude.shape[0]-1)*(gradient_magnitude.shape[1]-1)), desc="non maximization suppress", ncols=100) as pbar:
        for i in range(1, gradient_magnitude.shape[0] - 1):
            for j in range(1, gradient_magnitude.shape[1] - 1):
                pbar.update(1)
                if new_magnitude[i, j]!=new_magnitude[i, j] or gradient_magnitude[i, j]!=gradient_magnitude[i ,j]:
                    new_magnitude[i, j] = np.nan
                    continue
                direction = gradient_direction[i, j]
                if 0 < np.tan(direction) < 1:
                    Ga = np.tan(direction) * gradient_magnitude[i-1, j] + (1 - np.tan(direction))*gradient_magnitude[i-1, j-1]
                    Gc = np.tan(direction) * gradient_magnitude[i+1, j] + (1 - np.tan(direction))*gradient_magnitude[i+1, j+1]
                elif 1<= np.tan(direction):
                    Ga = gradient_magnitude[i, j-1]*1/np.tan(direction) + gradient_magnitude[i-1, j-1]*(1 - 1/np.tan(direction))
                    Gc = gradient_magnitude[i, j+1]*1/np.tan(direction) + gradient_magnitude[i+1, j+1]*(1 - 1/np.tan(direction))
                elif -1 < np.tan(direction) < 0:
                    Ga = -np.tan(direction) * gradient_magnitude[i-1, j] + (1 + np.tan(direction))*gradient_magnitude[i-1, j+1]
                    Gc = -np.tan(direction) * gradient_magnitude[i+1, j] + (1 + np.tan(direction))*gradient_magnitude[i+1, j-1]
                else:
                    Ga = -gradient_magnitude[i, j + 1]*1/np.tan(direction)+gradient_magnitude[i-1,j+1]*(1+1/np.tan(direction))
                    Gc = -gradient_magnitude[i, j - 1]*1/np.tan(direction)+gradient_magnitude[i+1,j-1]*(1+1/np.tan(direction))
                gradient_magnitude_i_j = gradient_magnitude[i, j]
                if gradient_magnitude_i_j >= Gc and gradient_magnitude_i_j >= Ga: # 证明该点为局部极大值点
                    new_magnitude[i, j] = gradient_magnitude[i, j]
    return new_magnitude, gradient_direction

class getfaultlines:
    def __init__(
            self,
            data:np.ndarray,
            # banary_vmin : float = 0.5,
            # banary_max : float = 3,
    ):
        self.data = data
        self.data_vars = {}
        #
        # banary_gradient, _ = self._get_deformation_gradients_NMS_barnary(vmin = banary_vmin, vmax = banary_max)

    def _get_deformation_gradients(self, apply_gaussian_filter=True, sigma=0.8, radius=5):
        masked_defor_array = np.ma.masked_invalid(self.data)
        # Apply Gaussian filter if the flag is set to True
        if apply_gaussian_filter:
            guss_filtered_array = ndimage.gaussian_filter(masked_defor_array, sigma=sigma, truncate=radius)
        else:
            guss_filtered_array = masked_defor_array
        hengxian_kernel = np.array([-1, 1]).reshape(1, -1)
        zongxian_kernel = np.array([-1, 1]).reshape(-1, 1)
        sobel_array_x = signal.convolve2d(guss_filtered_array, hengxian_kernel, mode="same", boundary="symm")
        sobel_array_y = signal.convolve2d(guss_filtered_array, zongxian_kernel, mode="same", boundary="symm")

        gradient_magnitude = np.hypot(sobel_array_x, sobel_array_y)
        gradient_direction = np.arctan2(sobel_array_y, sobel_array_x)  # 弧度制
        self.data_vars["gradient_magnitude"] = gradient_magnitude
        self.data_vars["gradient_direction"] = gradient_direction
        return gradient_magnitude, gradient_direction

    def _get_deformation_gradients_NMS(self):
        gradient_magnitude, gradient_direction = self._get_deformation_gradients()
        gradient_magnitude_NMS, gradient_direction = non_maximization_suppress(gradient_magnitude, gradient_direction)
        self.data_vars["gradient_magnitude_NMS"] = gradient_magnitude_NMS
        return gradient_magnitude

    def _get_deformation_gradients_NMS_barnary(
            self, vmin=0.5, vmax=3,
            apply_gaussian_filter=False, sigma=0.8, radius=5,
            ):
        gradient_magnitude, gradient_direction = self._get_deformation_gradients(apply_gaussian_filter, sigma, radius)
        gradient_magnitude_NMS, gradient_direction = non_maximization_suppress(gradient_magnitude, gradient_direction)
        self.data_vars["gradient_magnitude_NMS"] = gradient_magnitude_NMS

        # 二值化
        from copy import deepcopy
        gradient_magnitude_NMS_banary = deepcopy(gradient_magnitude_NMS)
        gradient_magnitude_NMS_banary[gradient_magnitude_NMS_banary < vmin] = 0
        gradient_magnitude_NMS_banary[gradient_magnitude_NMS_banary > vmax] = 0
        gradient_magnitude_NMS_banary[gradient_magnitude_NMS_banary > 0] = 255
        self.data_vars["gradient_magnitude_banary"] = gradient_magnitude_NMS_banary
        gradient_magnitude_NMS_clipped = np.clip(gradient_magnitude_NMS_banary, 0, 255)
        gradient_magnitude_NMS_clipped[np.isnan(gradient_magnitude_NMS_clipped)] = 0
        gradient_magnitude_NMS_uint8 = gradient_magnitude_NMS_clipped.astype("uint8")
        self.data_vars["gradient_magnitude_NMS_uint8"] = gradient_magnitude_NMS_uint8
        return gradient_magnitude_NMS_uint8, gradient_direction

    def _HoughLinesP(
            self,
            banary_data,
            rho = 3,
            theta = np.pi/180,
            threshold = 30,
            minLineLength = 6,
            maxLineGap = 10,
            imshow=True
    ) -> FaultLines:
        import cv2
        lines = cv2.HoughLinesP(banary_data, rho, theta, threshold=threshold,
                                minLineLength=minLineLength, maxLineGap=maxLineGap)
        total_points = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                points = [(int(x1), int(y1)), (int(x2), int(y2))]
                if x1 == x2:
                    continue
                total_points.append(points)
        lines = FaultLines(total_points)
        self.data_vars["lines"] = lines
        return lines

    def plot_HoughLinesP_lines(self, extent : list = None):
        try:
            lines = self.data_vars["lines"]
        except:
            print("HoughlinesP don't run!")
            return 0
        if lines is not None:
            for line in lines:
                x1, y1 = line[0]
                x2, y2 = line[1]
                plt.plot((x1, x2), (y1, y2))
        plt.gca().invert_yaxis()
        plt.gca().set_aspect("equal")
        if extent is not None:
            plt.axis(extent)

    def plot_property(self, property:str, vmin=None, vmax=None, ax = None):
        plt.figure()
        plt.imshow(self.data_vars[property], cmap="gray", vmin=vmin, vmax=vmax, ax=ax)
        plt.legend()
        plt.colorbar()


