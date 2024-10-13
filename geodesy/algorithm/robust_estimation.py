"""
库名：稳健估计
参考文献：陈健平, 沈云中, 2020. 相关观测值双因子抗差估计的改进算法. 大地测量与地球动力学 40, 507–511.
        https://doi.org/10.14075/j.jgg.2020.05.013
"""
import sys
import numpy as np

def robust_estimation(A:np.ndarray, P:np.ndarray, l:np.ndarray,
                      atol:float, k0:float=1.5, k1:float=4.5, iterationThreshold=30, iteration_number:int = 0,
                      P_total:list = [],
                      hat_x=None, P_ones=None,
                      method = "variance",
                      ):
    """
    稳健估计算法
    A: 系数矩阵
    P: 权阵
    l: 观测值
    P_total:用于存储每次迭代过程中的权重
    iteration_number: 不是输入变量，只是用来计数，不是迭代阈值！！！！！！！！！！
    method: "variance":只考虑方差, 权阵为对角阵的情况
            "variance_and_covariance": 考虑方差和协方差的情况，权阵为非对角阵，不考虑内部权重
    step: hat_x, P_new新的两个变量，到时候进行解释
    """
    # 一次稳健估计迭代
    if hat_x is not None and P_ones is not None:
        hat_x_before = hat_x
    elif hat_x is not None and P_ones is None:
        print("please input P_ones")
        sys.exit()
    else:
        hat_x_before = np.linalg.pinv(A.T @ P @ A) @ (A.T @ P @ l)  # 一次迭代前的估值
    # print(iteration_number)
    # if iteration_number == 25:
    #     pass
    v = A @ hat_x_before - l
    t = np.sum(np.diagonal(P) == 0)
    r = (A.shape[0] - A.shape[1] - t)
    sigma0_2 = abs((v.T @ P @ v) / r)
    # D_l = np.linalg.pinv(P) * sigma0_2
    D_hatx = np.linalg.pinv(A.T @ P @ A) * sigma0_2
    # 确定降权因子gamma
    if method == "variance_and_covariance":  # 该部分的逻辑可以修改为numpy数组中整体计算，有缘继续优化
        gamma = np.zeros_like(P)
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                if P[i, j] != 0:
                    frac_vi_sigmai = v[i, 0]/np.sqrt(sigma0_2)  # vi标准化残差
                    frac_vj_sigmaj = v[j, 0]/np.sqrt(sigma0_2)
                    gamma_i_i = shrinkage_factor(frac_vi_sigmai, k0, k1)
                    gamma_j_j = shrinkage_factor(frac_vj_sigmaj, k0, k1)
                    gamma[i, j] = np.sqrt(gamma_j_j * gamma_i_i)
    elif method == "variance":
        frac_vi_sigma = v / np.sqrt(sigma0_2)
        gamma = shrinkage_factor(frac_vi_sigma, k0, k1)
    else:
        gamma = None
        print("please check the robust estimation method!")
        sys.exit()
    if P_ones is not None:
        P_new = gamma * P_ones
        P_total.append(P_new)
    else:
        P_new = gamma * P  # 更新权阵P, 等于P和gamma权阵的哈达玛积
        P_total.append(P_new)
    try:
        hat_x_after = np.linalg.pinv(A.T @ P_new  @ A) @ (A.T @ P_new @ l)
    except np.linalg.LinAlgError:
        return hat_x_before, P_new, iteration_number
    #  确实是否满足迭代条件：前后两次估值变化小于一定阈值则迭代结束
    x_befor_after_error = np.abs(hat_x_after - hat_x_before)
    if atol == atol:
        mark  = (np.asarray(x_befor_after_error) < atol).all()
        iteration_number += 1
        if not mark and iteration_number < iterationThreshold:   #  mark is False且小于迭代阈值则继续进行迭代
            return robust_estimation(A=A, P=P_new, l=l, k0=k0, k1=k1, atol=atol, iteration_number=iteration_number,
                                     P_total=P_total
                                     )
        else:
            return hat_x_after, P_total, iteration_number
    else:
        iteration_number += 1
        return hat_x_after, P_total, iteration_number


def shrinkage_factor(frac_vi_sigma:float or np.ndarray, k0:float, k1:float):
    # 降权因子gamma
    frac_vi_sigma = np.abs(frac_vi_sigma)
    if type(frac_vi_sigma) is float or type(frac_vi_sigma) is np.float_:
        if np.abs(frac_vi_sigma) <= k0:
            gamma = 1
        elif k0 < np.abs(frac_vi_sigma) <= k1:
            gamma = k0 / np.abs(frac_vi_sigma) * ((k1 - np.abs(frac_vi_sigma)) / (k1 - k0)) ** 2
        else:
            gamma = 0
        return gamma
    elif type(frac_vi_sigma) is np.ndarray:
        gamma_diag = frac_vi_sigma
        gamma_diag[frac_vi_sigma < k0] = 1
        logical = np.logical_and(frac_vi_sigma >k0, frac_vi_sigma <= k1)
        gamma_diag[logical] = k0 / np.abs(frac_vi_sigma[logical]) * ((k1 - np.abs(frac_vi_sigma[logical])) / (k1 - k0)) ** 2
        gamma_diag[frac_vi_sigma > k1] = 0
        try:
            gamma = np.diag(gamma_diag.ravel())
        except:
            pass
        return gamma


if __name__=="__main__":
    # # 示例：模拟y=ax+b直线
    """改进的稳健估计拟合方法"""
    import numpy as np
    import matplotlib.pyplot as plt

    def generate_noisy_outliers_data(n, a, b, noise_std, outlier_prob, outlier_range):
        # 生成等间距的 x 值
        x = np.linspace(0, 50, n)

        # 计算拟合数据 y
        y = a * x + b

        # 添加高斯噪声,随机噪声
        np.random.seed(32)
        noise = np.random.normal(0, noise_std, n)   # 正态分布
        y_noisy = y + noise

        # 添加粗差
        num_outliers = int(outlier_prob * n)
        outlier_indices = np.random.choice(np.arange(n), size=num_outliers, replace=False)
        y_outliers = np.random.uniform(*outlier_range, size=num_outliers)
        y_noisy[outlier_indices] = y_outliers

        return x.reshape(-1, 1), y.reshape(-1, 1), y_noisy.reshape(-1, 1)

    # 示例使用
    n = 100  # 数据点数量
    a = 2.5  # 斜率
    b1 = 10  # 截距
    b2 = -10
    noise_std = 2.0  # 噪声的标准差
    outlier_prob = 0.1  # 粗差的概率
    outlier_range = (30, 70)  # 粗差的取值范围

    x1, y_real1, y_noisy1 = generate_noisy_outliers_data(n, a, b1, noise_std, outlier_prob, outlier_range)
    x2, y_real2, y_noisy2 = generate_noisy_outliers_data(n, a, b2, noise_std, outlier_prob, outlier_range)
    x = np.vstack((x1, x2+50))
    y_real = np.vstack((y_real1, y_real2))
    y_noisy = np.vstack((y_noisy1, y_noisy2))
    plt.plot(x, y_noisy, label="Simulation Line", color="red")

    # 设置图形标题和坐标轴标签
    plt.title('Fitted Data with Noise and Outliers')
    plt.xlabel('x')
    plt.ylabel('y')
    B = np.hstack([x, np.ones_like(x)])
    P = np.zeros((len(y_noisy), len(y_noisy)))
    P[0, 0] = 1
    P[1, 1] = 1
    P[2, 2] = 1
    P[3, 3] = 1
    P[10, 10] = 1
    P_ones = np.eye(len(y_noisy))
    hat_x, P_new, iter_num = robust_estimation(A=B, P=P, l=y_noisy, atol=0.01)
    hat_x_2, P, iter_num = robust_estimation(A=B, P=P_new, l=y_noisy, atol=0.01, hat_x=hat_x, P_ones=P_ones, k1=60)
    hat_x_wei = np.linalg.pinv(B.T @ P_ones @ B) @ (B.T @ P_ones @ y_noisy)  # 一次迭代前的估值
    print(hat_x_wei)
    print(hat_x)
    print(iter_num)
    plt.plot(x, hat_x[0]*x+hat_x[1], color="blue", label="Fintted Line")
    plt.plot(x, hat_x_2[0]*x+hat_x_2[1], color="orange", label="Fintted Line2")
    plt.plot(x, hat_x_wei[0]*x+hat_x_wei[1], color="green", label="WLS Line")
    plt.legend()
    plt.show()