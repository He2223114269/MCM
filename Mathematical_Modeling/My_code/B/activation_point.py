import math
import numpy as np
import pandas as pd
def activation_point(x,y,radius_matrix):
    """
    正式版本——给出一个点和半径矩阵，计算出所有他们能激活的点,不需要激活矩阵
    :param x: 激活点x坐标
    :param y: 激活点y坐标
    :param adjugate_matrix: 激活矩阵
    :return: 返回所有激活点
    """
    activation_point = []
    rows, cols = radius_matrix.shape
    max_radius = int(np.max(radius_matrix))
    y_max = min(cols - 1, y + max_radius)
    y_min = max(0, y - max_radius)
    x_min = max(0, x - max_radius)
    x_max = min(rows - 1, x + max_radius)
    #             print("y_max: {}, y_min: {}, x_min: {}, x_max: {}".format(y_max, y_min, x_min, x_max))
    for ii in range(x_min, x_max + 1):
        for jj in range(y_min, y_max + 1):
            # print("ii: {}, jj: {}, radius_matrix :{}".format(ii,jj,radius_matrix[ii, jj]))
            # print("距离: {} radius_matrix: {} ".format(math.sqrt((ii - x) ** 2 + (jj - y) ** 2),radius_matrix[ii, jj]))
            if math.sqrt((ii - x) ** 2 + (jj - y) ** 2) <= radius_matrix[ii, jj]/ 37.04:  #
                point = list((ii,jj))
                activation_point.append(point)

