"""
库名：Helmert方差分量估计算法
建立时间：2021-11-8，正确性以检验
变量命名参考文献：
    刘计洪,胡俊,李志伟,朱建军.InSAR三维同震地表形变监测——窗口优化的SM-VCE算法[J].测绘学报,2021,50(09):1222-1239.

  希纳字符读法：https://blog.csdn.net/openme_openwrt/article/details/7908643
"""
import numpy as np
def allclose(a,atol): #atol为 absolute tolerance parameter 绝对容差
    """判断a矩阵中的数值是否近似等于一个阈值"""
    mark = True
    for i in range(len(a)):
        for j in range(len(a)):
            if not np.allclose(a[i],a[j],atol=atol):
                mark = False
    if mark:
        return True
    elif not mark:
        return False

def normal_equation(L_list,B_list,W_list):
    """法方程;B_list,L_list,P_list为对应不同类独立观测值的系数B,L,P且数据类型为np.array的列表"""
    N_list = [B_list[i].T @ W_list[i] @ B_list[i] for i in range(len(L_list))]
    U_list = [B_list[i].T @ W_list[i] @ L_list[i] for i in range(len(L_list))]
    N = sum(N_list)
    U = sum(U_list)
    try:
        l = np.linalg.pinv(N) @ U #未知参数平差值
    except np.linalg.LinAlgError:
        return np.linalg.LinAlgError, np.nan, np.nan, np.nan
    # l = np.linalg.solve(N, U)  # 未知数参数平差值
    delta = np.array(np.zeros((len(L_list),1)))
    v_list = [B_list[i] @ l - L_list[i] for i in range(len(L_list))]
    delta  = np.array([v_list[i].T @ W_list[i] @ v_list[i] for i in range(len(L_list))]).reshape(-1, 1)
    return N, N_list, delta, l

def Helmert_one(N,N_list,delta,n_list):
    """迭代一次的赫尔摩特算法,n_list为不同类型观测值的个数"""
    gamma = np.array(np.zeros((len(N_list),len(N_list))))
    try:
        # 求伪逆可对所有点的数据进行解算
        inv_N = np.linalg.pinv(N)
    except np.linalg.LinAlgError:
        return np.linalg.LinAlgError, np.nan, np.nan

    for m in range(len(N_list)):
        for n in range(len(N_list)):
            if m == n:
                gamma[m,n] = n_list[m] - 2*(inv_N @ N_list[m]).trace() + \
                             (inv_N @ N_list[n] @ inv_N @ N_list[n]).trace()
            elif m != n:
                gamma[m,n] = (inv_N @ N_list[m] @ inv_N @ N_list[n]).trace()
    sigma2 = np.linalg.pinv(gamma) @ delta
    return sigma2,gamma,delta 

def Helmert(L_list, B_list, W_list, n_list, atol, iterationNumThreshold):
    """
    1. L = Bl:
        L_list: 为不同类ndarray类型的观测值的列表
        B_list: 系数矩阵列表
        n_list: 观测值数量列表
    2.外部主要调用函数:
        atol: 迭代精度阈值，
        iterationNumThreshold: 迭代次数阈值
        当atol和iterationNumThreshold同时为None时，不进行迭代，进行最小二乘计算
    """
    N,N_list,delta,l = normal_equation(L_list,B_list,W_list)
    if N is np.linalg.LinAlgError:
        return np.linalg.LinAlgError, np.nan, np.nan
    if iterationNumThreshold == None and atol == None:  # 该种情况为一次最小二乘计算
        return l, 0, 0

    sigma2,gamma,delta = Helmert_one(N,N_list,delta,n_list)
    if sigma2 is np.linalg.LinAlgError:
        return np.linalg.LinAlgError, np.nan, np.nan

    loop_num = 1 #迭代次数
    while abs(max(sigma2) - min(sigma2)) > atol:
        sub = np.argwhere(sigma2!=0)
        unit_sigma2 = sigma2[sub[0][0], sub[0][1]]
        W_list = [unit_sigma2/sigma2[i, 0] * W_list[i] if sigma2[i, 0] != 0 else W_list[i] for i in range(len(L_list))]
        N,N_list,delta,l = normal_equation(L_list,B_list,W_list)
        if N is np.linalg.LinAlgError:
            return np.linalg.LinAlgError, np.nan, np.nan

        sigma2,gamma,delta = Helmert_one(N,N_list,delta,n_list)
        if sigma2 is np.linalg.LinAlgError:
            return np.linalg.LinAlgError, np.nan, np.nan
        loop_num += 1
        if loop_num > iterationNumThreshold:
            #  print("迭代次数过大,停止迭代")
            return l, loop_num, W_list
    return l,loop_num, W_list

if __name__ == "__main__":
    """参考算例：广义测量平差，崔希璋，武汉大学出版社算例3-2-1"""
    #测角系数矩阵B1 12*4
    B1 = np.array([
        [0.5532,-0.8100,0,0],
        [0.2434,0.5528,0,0],
        [-0.7966,0.2572,0,0],
        [-0.2434,-0.5528,0,0],
        [0.6298,0.6368,0,0],
        [-0.3864,-0.0840,0,0],
        [0.7966,-0.2572,-0.2244,-0.3379],
        [-0.8350,-0.1523,0.0384,0.4095],
        [0.0384,0.4095,0.1860,-0.0716],
        [-0.0384,-0.4095,0.2998,0.1901],
        [-0.3480,0.3255,-0.0384,-0.4095],
        [0.3864,0.0840,-0.2614,0.2194]
        ])
    #测边系数矩阵B2 6*4
    B2=np.array([
        [0.3072,0.9516,0,0],
        [-0.9152,0.4030,0,0],
        [0.2124,-0.9722,0,0],
        [0,0,-0.6429,-0.7660],
        [0,0,-0.8330,0.5532],
        [0.9956,-0.0934,-0.9956,0.0934]
        ])
    B = [B1,B2]

    #测角常数项L1 12*1
    L1 = np.array([0.18,-0.53,3.15,0.23,-2.44,1.01,2.68,-4.58,2.80,-3.10,8.04,-1.14]).reshape(-1,1)
    # 测边常数项L2 6*1
    L2 = np.array([-0.84,1.54,-3.93,2.15,-12.58,-8.21]).reshape(-1,1)
    L = [L1,L2]

    #测角和测边个数
    n1 = 12
    n2 = 6
    n = [n1,n2]
    
    #初始定权，测角中误差=1.5，测边中误差=2.0
    sita = [1.5*1.5,2.0*2.0]
    P1 = np.array(np.eye(12,12))
    P2 = np.array(sita[0]/sita[1] * np.eye(6,6))
    P = [P1,P2]

    #Helmert方差分量估计算法
    l,loop_num = Helmert(L,B,P,n, atol=0.01, iterationNumThreshold=30)
    print(f"迭代{loop_num}次\n未知参数平差值：\n{l}")
