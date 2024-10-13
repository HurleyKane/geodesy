"""
库名：四叉树（一种降采样的图像处理方法）
更新功能：
    3-0-0: 实现最大值和最小值之差满足阈值的四叉树分解
    3-0-1: 实现以los向的形变值的正负值进行四叉树分解的方式
    3-0-4: 改进四叉树分解绘图效果,增加四叉树节点的编号
    3-2-0: 利用四叉树进行分割时利用去轨道的办法减去趋势值
"""
import os
import sys
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_path)

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
# from data_visualization import set_lonlat
# from Mingkai_lib.Result_visualization import set_lonlat

class QuadTreeNode:
    """四叉树节点"""
    def __init__(self):
        self.top_right_child    = None          #右上区域
        self.top_left_child     = None          #左上区域
        self.lower_right_child  = None          #右下区域
        self.lower_left_child   = None          #坐下区域
        self.row_limits         = [None,None]   #该节点的行的范围, 闭区间
        self.column_limits      = [None,None]   #该节点的列的范围, 闭区间
        self.displacement_type  = None          #形变位移的类型
        self.column_len         = None          #节点范围内列的长度
        self.row_len            = None          #节点范围内行的长度
        self.number             = ""            #四叉树节点的编号,左上、右上、左下和右下分别记为:1,2,3,4

class QuadTree:
    """四叉树"""
    def __init__(self, dataset_n:xr.Dataset):
        """
        注：此处的行为横轴经度对应的点的数量，列为纵轴维度对应的点的数量
        """
        self.dataset_n   = dataset_n  # 用于降采样的数据类型
        self.row_len  = dataset_n.clat.size
        self.column_len = dataset_n.clon.size
        self.nodes_number = 1                   #节点的数量
        self.root = QuadTreeNode()              #创建根节点
        self.root.row_limits    = [0,self.row_len - 1]    #根节点行的范围, 此范围为闭区间
        self.root.column_limits = [0,self.column_len - 1] #根节点列的范围
        self.root.row_len    = self.root.row_limits[1]    - self.root.row_limits[0] + 1    #节点范围内行的长度
        self.root.number += ''                  #根节点的编号
        self.root.column_len = self.root.column_limits[1] - self.root.column_limits[0] + 1 #节点范围内列的长度
        self.points_subs = []
                #记录点的坐标(x,y)的列表方便进行绘图,四个点为一组，以正上方开始，逆时针装入点

    def add_children(self,parent):
        """
        四叉树分解中，给父亲节点加入四个孩子节点，parent为父节点
        """
        """child function"""
        def top_left_child(parent,parent_x1,parent_x2,parent_y1,parent_y2):
            """左上孩子"""
            parent.top_left_child = QuadTreeNode()
            parent.top_left_child.row_limits       = [parent_x1,int((parent_x1+parent_x2)/2)]
            parent.top_left_child.column_limits    = [parent_y1,int((parent_y1+parent_y2)/2)]
            parent.top_left_child.row_len          = \
                    parent.top_left_child.row_limits[1] - parent.top_left_child.row_limits[0] + 1
            parent.top_left_child.column_len       = \
                    parent.top_left_child.column_limits[1] - parent.top_left_child.column_limits[0] + 1
            parent.top_left_child.number = parent.number+"1"
            self.nodes_number += 1

        def top_right_child(parent,parent_x1,parent_x2,parent_y1,parent_y2):
            """右上孩子"""
            parent.top_right_child = QuadTreeNode()
            parent.top_right_child.row_limits      = [parent_x1,int((parent_x1+parent_x2)/2)]
            parent.top_right_child.column_limits   = [int((parent_y1+parent_y2)/2)+1,parent_y2]
            parent.top_right_child.row_len         = \
                parent.top_right_child.row_limits[1] - parent.top_right_child.row_limits[0] + 1
            parent.top_right_child.column_len      = \
                parent.top_right_child.column_limits[1] - parent.top_right_child.column_limits[0] + 1
            parent.top_right_child.number = parent.number+"2"
            self.nodes_number += 1

        def lower_left_child(parent,parent_x1,parent_x2,parent_y1,parent_y2):
            """左下孩子"""
            parent.lower_left_child = QuadTreeNode()
            parent.lower_left_child.row_limits     = [int((parent_x1+parent_x2)/2)+1,parent_x2]
            parent.lower_left_child.column_limits  = [parent_y1,int((parent_y1+parent_y2)/2)]
            parent.lower_left_child.row_len        = \
                parent.lower_left_child.row_limits[1] - parent.lower_left_child.row_limits[0] + 1
            parent.lower_left_child.column_len     = \
                parent.lower_left_child.column_limits[1] - parent.lower_left_child.column_limits[0] + 1
            parent.lower_left_child.number = parent.number+"3"
            self.nodes_number += 1

        def lower_right_child(parent,parent_x1,parent_x2,parent_y1,parent_y2):
            """右下孩子"""
            parent.lower_right_child = QuadTreeNode()
            parent.lower_right_child.row_limits    = [int((parent_x1+parent_x2)/2)+1,parent_x2]
            parent.lower_right_child.column_limits = [int((parent_y1+parent_y2)/2)+1,parent_y2]
            parent.lower_right_child.row_len         = \
                parent.lower_right_child.row_limits[1] - parent.lower_right_child.row_limits[0] + 1
            parent.lower_right_child.column_len      = \
                parent.lower_right_child.column_limits[1] - parent.lower_right_child.column_limits[0] + 1
            parent.lower_right_child.number = parent.number+"4"
            self.nodes_number += 1

        """准备数据"""
        parent_x1 = parent.row_limits[0]
        parent_x2 = parent.row_limits[1]
        parent_y1 = parent.column_limits[0]
        parent_y2 = parent.column_limits[1]

        if parent.column_len > 1 and parent.row_len > 1:
            top_left_child(parent,parent_x1,parent_x2,parent_y1,parent_y2)
            top_right_child(parent,parent_x1,parent_x2,parent_y1,parent_y2)
            lower_left_child(parent,parent_x1,parent_x2,parent_y1,parent_y2)
            lower_right_child(parent,parent_x1,parent_x2,parent_y1,parent_y2)
        elif parent.row_len == 1 and parent.column_len > 1:
            top_left_child(parent,parent_x1,parent_x2,parent_y1,parent_y2)
            top_right_child(parent,parent_x1,parent_x2,parent_y1,parent_y2)
        elif parent.row_len > 1 and parent.column_len == 1:
            top_left_child(parent,parent_x1,parent_x2,parent_y1,parent_y2)
            lower_left_child(parent,parent_x1,parent_x2,parent_y1,parent_y2)
        else: print("this is a error,please examine")

    def visit_leaf_nodes(self, statistics=False):
        """访问叶子节点，并返回包含所有节点的列表"""
        leaf_nodes = []
        queue   = [self.root] #根节点进入队列，将队列作为一个循环判断的工具
        while queue:
            cur_node = queue.pop(0)
            """如果有一个孩子节点不为空，则加入队列继续遍历"""
            if cur_node.top_left_child != None:    queue.append(cur_node.top_left_child)
            if cur_node.top_right_child != None:   queue.append(cur_node.top_right_child)
            if cur_node.lower_left_child != None:  queue.append(cur_node.lower_left_child)
            if cur_node.lower_right_child != None: queue.append(cur_node.lower_right_child)
            if cur_node.top_left_child == None and cur_node.top_right_child == None\
                    and cur_node.lower_left_child == None and cur_node.lower_right_child == None:
                """如果四个节点都为None，则证明该节点为叶节点，装入leaf_nodes"""
                leaf_nodes.append(cur_node)
        if statistics:
            print(f"    leaf_nodes' len:{len(leaf_nodes)}")
        return leaf_nodes

    def matplot_img(self, lat, lon, QTdecomp_threshold=None, number="", colorbar = False, plt_var:str="defor",
                    img_inverse:bool=False
                    ):
        """
        利用matplotlib对四叉树分解后的图像进行绘图
        region表示绘图范围[lon0,lon1,lat0,lat1]
        """
        from InSARlib.plot import set_lonlat
        leaf_nodes = self.visit_leaf_nodes()
        if img_inverse:
            region = [self.dataset_n.clon[0], self.dataset_n.clon[-1], self.dataset_n.clat[-1], self.dataset_n.clat[0]]
        else:
            region = [self.dataset_n.clon[0], self.dataset_n.clon[-1], self.dataset_n.clat[0], self.dataset_n.clat[-1]]

        from mpl_toolkits.basemap import Basemap
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 加入字体
        region = np.round(region, 2)
        fig = plt.figure(figsize=[12, 4])
        ax1 = plt.subplot(121)
        plt.xlim(region[0], region[2])
        plt.ylim(region[1], region[3])
        ax1.invert_yaxis()
        ax2 = plt.subplot(122)
        plt.xlim(region[0], region[2])
        plt.ylim(region[1], region[3])
        ax2.invert_yaxis()

        #  plt.tight_layout() #调整整体空白
        plt.subplots_adjust(wspace=0, hspace=0.05)  # 调整子图间距

        ax1.set_title("(a) 原始方位向形变图", y=-0.1)
        if QTdecomp_threshold != None and number != '':
            ax2.set_title(number + "阈值为" + str(QTdecomp_threshold) + "m的区域分割", y=-0.1)


        map1 = Basemap(region[0], region[2], region[1], region[3], ax=ax1)
        map2 = Basemap(region[0], region[2], region[1], region[3], ax=ax2)
        set_lonlat(map1, ax1, np.arange(region[0] + 0.1, region[2] + 0.05, 0.3),
                   np.arange(region[1] + 0.05, region[3] + 0.05, 0.3), [0, 0, 1, 0], [1, 0, 0, 0], 14)
        set_lonlat(map2, ax2, np.arange(region[0] + 0.1, region[2] + 0.05, 0.3),
                   np.arange(region[1] + 0.05, region[3] + 0.05, 0.3), [0, 0, 1, 0], [0, 0, 0, 0], 14)

        # exec("map1.scatter(self.dataset_n.lon, self.dataset_n.lat, c=self.dataset_n."+plt_var +
        #      ", cmap='jet', vmax=4, vmin=-6)")
        # exec("_points2 = map2.scatter(self.dataset_n.lon, self.dataset_n.lat, c=self.dataset_n."+plt_var +
        #      ", cmap='jet', vmax=4, vmin=-6)")
        exec("map1.scatter(lon, lat, c=self.dataset_n."+plt_var +
             ", cmap='jet')")
        exec("_points2 = map2.scatter(lon, lat, c=self.dataset_n."+plt_var +
             ", cmap='jet')")
        points2 = locals()["_points2"]
        if colorbar == True:
            map2.colorbar(points2)

        def find_points(number,row_limits,column_limits,len_0=0):
            """根据编号寻找该节点绘图时用的row_limits和column_limits"""
            if len_0 == len(number):
                return row_limits,column_limits
            elif len_0 < len(number):
                if number[len_0] == '1':
                    start_row = row_limits[0]
                    start_columns = column_limits[0]
                    end_row  = int((row_limits[0]+row_limits[1])/2)
                    end_columns = int((column_limits[0]+column_limits[1])/2)
                elif number[len_0] == '2':
                    start_row = row_limits[0]
                    start_columns = int((column_limits[0]+column_limits[1])/2)
                    end_row  = int((row_limits[0]+row_limits[1])/2)
                    end_columns = column_limits[1]
                elif number[len_0] == '3':
                    start_row = int((row_limits[0]+row_limits[1])/2)
                    start_columns = column_limits[0]
                    end_row  = row_limits[1]
                    end_columns = int((column_limits[1]+column_limits[0])/2)
                elif number[len_0] == '4':
                    start_row = int((row_limits[1]+row_limits[0])/2)
                    start_columns = int((column_limits[0]+column_limits[1])/2)
                    end_row = row_limits[1]
                    end_columns = column_limits[1]
                else:
                    print("节点数量不是4，请检查")
                    return 0

                new_row_limits = [start_row,end_row]
                new_column_limits = [start_columns,end_columns]
                return find_points(number,new_row_limits,new_column_limits,len_0+1) \
                        #如果number没遍历完，再次进入递归寻找节点绘图时的边界,并返回find的返回值

        if leaf_nodes != None:
            """根据叶节点进行图像的绘制"""
            for cur_node in leaf_nodes:
                row_limits,column_limits = find_points(cur_node.number,
                        self.root.row_limits,self.root.column_limits)
                    #找出节点绘图是的左上，右下两个点
                """绘制矩形"""
                x = [self.dataset_n.clon.isel(clon=column_limits[0]),
                     self.dataset_n.clon.isel(clon=column_limits[1]),
                     self.dataset_n.clon.isel(clon=column_limits[1]),
                     self.dataset_n.clon.isel(clon=column_limits[0]),
                     self.dataset_n.clon.isel(clon=column_limits[0])]
                y = [self.dataset_n.clat.isel(clat=row_limits[0]),
                     self.dataset_n.clat.isel(clat=row_limits[0]),
                     self.dataset_n.clat.isel(clat=row_limits[1]),
                     self.dataset_n.clat.isel(clat=row_limits[1]),
                     self.dataset_n.clat.isel(clat=row_limits[0])]
                if cur_node.displacement_type == "negative":   color = "black"
                elif cur_node.displacement_type == "positive": color = "r"
                elif cur_node.displacement_type == "nan":      color = 'y'
                else: color = "black"
                ax2.plot(x,y,color,linewidth=0.4)

    def threshold_judgement(self, cur_node, according_to:str, threshold=None, method=None):
        """
        四叉树分解判断是否进行分割，小于阈值返回TRUE进行分割，大于阈值返回False不进行分割
        threshold: 分割设定阈值
        cur_node: 输入目前需要进行判断的节点
        method: default：  为默认的最大值与最小值之差,
                drop_track去轨道方法：对区域内的形变值拟合一个平面,拟合曲面方程 l = a + bx + cy + dxy + ex**2 + fy**2,区域内点减去平
                面进行去轨道后，再进行最大值和最小值之差的计算
                downsampling:  进行下采样
        """
        import warnings
        warnings.filterwarnings("ignore") #忽略使用np.nanmax有关nan值的报错信息
        row_limits = cur_node.row_limits
        column_limits = cur_node.column_limits

        # 找到当前节点内所有点的下标,和下标对应的点的值
        node_win_r = range(row_limits[0], row_limits[1] + 1)
        node_win_c = range(column_limits[0], column_limits[1] + 1)
        dataset_win_rc = self.dataset_n.isel(clat=node_win_r, clon=node_win_c)
        str = according_to + " = dataset_win_rc." + according_to  # 形如：defor = dataset_win_rc.defor
        exec(str)
        variable = locals()[according_to]

        # 对范围内的点进行阈值判断
        if method == "default" or method == "drop_track":
            if method == "default":
                if threshold is None:
                    print("请检查阈值是否设定")
                else:  # 根据图像范围内的最大值和最小值的插值与阈值比较作出分解判断
                    # print(defor)
                    max = np.nanmax(variable)
                    min = np.nanmin(variable)
                    if max != max: max = 0
                    if min != min: min = 0
                    if max - min  < threshold or max - min == 0 : return False  # 小于阈值或为nan值，不需要进行分割
                    else: return True  # 确认需要进行分割
            elif method == 'drop_track':
                # 去轨道
                geo_x = dataset_win_rc.geo_x
                geo_y = dataset_win_rc.geo_y
                if variable.size > 50:  # 如果区域内点的数据很少，则不需要进行去平面
                    # 拟合曲面方程 l = a + bx + cy + dxy + ex**2 + fy**2
                    l = []  # 区域内点的形变值
                    B = []  # 方程系数
                    # 计算拟合平面的值
                    for r in range(variable.clat.size):
                        for c in range(variable.clon.size):
                            if variable.isel(clat=r, clon=c):
                                l.append(variable.isel(clat=r, clon=c))
                                x_i = geo_x.isel(clat=r, clon=c)
                                y_i = geo_y.isel(clat=r, clon=c)
                                B.append([1,x_i, y_i, x_i*y_i, x_i**2, y_i**2])
                    # # # l和B确定之后，还需要对其中的nan值进行过滤处理
                    subs = []
                    for i in range(len(l)):
                        if l[i] != l[i]:  # 如果该值为nan
                            subs.append(i)
                    l = np.delete(l, subs, axis=0)
                    B = np.delete(B, subs, axis=0)
                    l = np.matrix(l).T
                    B = np.matrix(B)
                    try:
                        X = np.linalg.solve(B.T * B, B.T * l)
                        # # 进行平面拟合
                        a, b, c, d, e, f = np.array(X)
                        surface = a + b * geo_x + c * geo_y + d * geo_x * geo_y + e * (geo_x ** 2) + f * (geo_y ** 2)
                        variable -= surface
                    except np.linalg.LinAlgError:  # 如果矩阵出现奇异，证明该矩阵内的点不足以进行曲面拟合，则交给max-min进行判断
                        pass
                max = np.nanmax(variable)
                min = np.nanmin(variable)
                if max != max: max = 0
                if min != min: min = 0
                if max - min < threshold or max - min == 0:
                    return False  # 小于阈值或为nan值，不需要进行分割
                else:
                    return True  # 确认需要进行分割
        elif method == "var":
            var = np.nanvar(variable)
            if var < threshold:
                return False
            else:
                return True

    def extract_nodes_subscipt(self,nodes):
        """提取节点的下标并利用列表依次返回"""
        total_points_number = 0
        subs_r_c = []
        for node in nodes:
            r_sub = range(node.row_limits[0], node.row_limits[1] + 1)
            c_sub = range(node.column_limits[0], node.column_limits[1] + 1)
            subs_r_c.append([r_sub, c_sub])
            total_points_number += len(r_sub) * len(c_sub)
        print("    all nodes' points number is:{}".format(total_points_number))
        return subs_r_c

    def decompose(self, threshold, according_variable, method="default"): #分解、聚合、正负判断
        """进行四叉树分解"""
        print("QuadTree is decomposing......")
        queue = [self.root] #根节点装入队列
        while queue: #当队列不为空就继续循环
            cur_node = queue.pop(0) #取出队列中第一个元素
            if self.threshold_judgement(cur_node, according_variable, threshold, method) == True \
                    and (cur_node.row_len > 1 or cur_node.column_len > 1) :
                #如果大于阈值或者行列能继续分解时进行分解
                self.add_children(cur_node)
                points = [
                    (cur_node.row_limits[0],int((cur_node.column_limits[0] + cur_node.column_limits[1])/2)),
                    (int((cur_node.row_limits[0] + cur_node.row_limits[1])/2),cur_node.column_limits[0]),
                    (cur_node.row_limits[1],int((cur_node.column_limits[0] + cur_node.column_limits[1])/2)),
                    (int((cur_node.row_limits[0] + cur_node.row_limits[1])/2),cur_node.column_limits[1])
                ]
                self.points_subs.append(points)
                """四个节点入队"""
                if cur_node.top_left_child != None:    queue.append(cur_node.top_left_child)
                if cur_node.top_right_child != None:   queue.append(cur_node.top_right_child)
                if cur_node.lower_left_child != None:  queue.append(cur_node.lower_left_child)
                if cur_node.lower_right_child != None: queue.append(cur_node.lower_right_child)