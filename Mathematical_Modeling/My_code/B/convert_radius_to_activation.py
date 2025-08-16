import numpy as np
import math


def convert_radius_to_activation(radius_matrix):
    # 获取半径矩阵的行数和列数
    rows, cols = radius_matrix.shape
    print(rows, cols)
    # 初始化激活矩阵，全部置零
    activation_matrix = np.zeros((rows, cols))

    # 遍历半径矩阵的每个元素
    max_radius = int(np.max(radius_matrix))
    print(max_radius)
    for i in range(rows):
        for j in range(cols):
            print("--------------i,j: {} {}".format(i, j))
            activation_count = 0
            # 扩展半径范围，包括边缘情况
            y_max = min(cols - 1, j + max_radius)
            y_min = max(0, j - max_radius)
            x_min = max(0, i - max_radius)
            x_max = min(rows - 1, i + max_radius)
            print("y_max: {}, y_min: {}, x_min: {}, x_max: {}".format(y_max, y_min, x_min, x_max))
            for ii in range(x_min, x_max + 1):
                for jj in range(y_min, y_max + 1):
                    print("ii: {}, jj: {}, radius_matrix :{}".format(ii,jj,radius_matrix[ii, jj]))
                    print("距离: {} radius_matrix: {} ".format(math.sqrt((ii - i) ** 2 + (jj - j) ** 2),
                                                             radius_matrix[ii, jj]))
                    if math.sqrt((ii - i) ** 2 + (jj - j) ** 2) <= radius_matrix[ii, jj] / 37.04 : #
                        activation_count =  activation_count + 1
                        print("activation_count : {}".format(activation_count))

            # 在激活范围内更新激活矩阵的值
            activation_matrix[i, j] = activation_count
            print(activation_count)
            print(activation_matrix)

    return activation_matrix

### -----------------------测试需要修改30行的备注------------------------
r_m = [[1, 2, 1, 2, 1],
       [3, 0, 2, 1, 3],
       [1, 2, 1, 2, 10]]
r_m = np.array(r_m)
ctivation_Matrix = convert_radius_to_activation(r_m)
print("Radius Matrix:")
print(r_m)
print("Activation Matrix:")
print(ctivation_Matrix)
