import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_adjugate_matrix(matrix):
    """
    创建初值伴随矩阵
    :param matrix: 原始矩阵
    :return: 初值伴随矩阵，值为0
    """
    rows = len(matrix)
    cols = len(matrix[0])
    zero_matrix = np.zeros((rows, cols))
    return zero_matrix


def initial_point(activation_matrix):
    """
    初始化出发点，根据激活矩阵确定当前的最优初始点
    :param activation_matrix: 激活矩阵
    :return:
    """
    max_index = np.unravel_index(np.argmax(activation_matrix), activation_matrix.shape)
    max_index = list(max_index)
    return max_index


def update_point(next, point):
    """
    确定下一个点
    :param next:  下次前进的方向
    :param point: 当前点
    :return: 下个位置
    """
    point = list(point)
    if next == "x_left":
        point[0] = point[0] - 1
    elif next == "x_right":
        point[0] = point[0] + 1
    elif next == "y_up":
        point[1] = point[1] - 1
    elif next == "y_down":
        point[1] = point[1] + 1
    return point


def check_matrix_positive(matrix):
    return np.any(matrix == 0)


def Translate_Forward_Strategy_test(radius_matrix, adjugate_matrix, x, y):
    """
    前进预测函数，输出四个前进方向
    :param radius_matrix: 半径矩阵
    :param adjugate_matrix: 邻接矩阵
    :param x: 当前位置的x值
    :param y: 当前位置的y值
    :return:
    """

    rows = len(radius_matrix)
    cols = len(radius_matrix[0])
    max_radius = int(np.max(radius_matrix))
    x_left = x_right = y_up = y_down = 0
    x_temp = x - 1
    if x_temp >= 0 and adjugate_matrix[x_temp, y] == 0:
        x_left_ponit = activation_point_test(x_temp, y, radius_matrix, adjugate_matrix)
        x_left = len(x_left_ponit)

    x_temp = x + 1
    if x_temp <= rows - 1 and adjugate_matrix[x_temp, y] == 0:
        x_right_ponit = activation_point_test(x_temp, y, radius_matrix, adjugate_matrix)
        x_right = len(x_right_ponit)

    y_temp = y - 1
    if y_temp >= 0 and adjugate_matrix[x, y_temp] == 0:
        y_up_ponit = activation_point_test(x, y_temp, radius_matrix, adjugate_matrix)
        y_up = len(y_up_ponit)

    if x_left ==0 and  x_right ==0 and y_up==0 and y_down == 0:
        return 0

    y_temp = y + 1
    if y_temp <= cols - 1 and adjugate_matrix[x, y_temp] == 0:
        y_down_ponit = activation_point_test(x, y_temp, radius_matrix, adjugate_matrix)
        y_down = len(y_down_ponit)
    import random

    def select_max_variable(variables):
        max_value = max(variables.values())
        max_variables = [var for var, value in variables.items() if value == max_value]
        selected_variable = random.choice(max_variables)
        return selected_variable

    variables = {"x_left": x_left, "x_right": x_right, "y_up": y_up, "y_down": y_down}
    # 调用函数选择最大的数值对应的变量
    selected_variable = select_max_variable(variables)
    if selected_variable == "x_left":
        adjugate_matrix = update_adjugate_matrix(x_left_ponit, adjugate_matrix)
    if selected_variable == "x_right":
        adjugate_matrix = update_adjugate_matrix(x_right_ponit, adjugate_matrix)
    if selected_variable == "y_up":
        adjugate_matrix = update_adjugate_matrix(y_up_ponit, adjugate_matrix)
    if selected_variable == "y_down":
        adjugate_matrix = update_adjugate_matrix(y_down_ponit, adjugate_matrix)
    # print("--------------------------------------------------------")
    return selected_variable


def convert_radius_to_activation_test(radius_matrix):
    """
    测试版本-计算测试数据
    :param radius_matrix: 半径矩阵
    :return: 输出激活矩阵
    """
    # 获取半径矩阵的行数和列数
    rows, cols = radius_matrix.shape
    #     print(rows, cols)
    # 初始化激活矩阵，全部置零
    activation_matrix = np.zeros((rows, cols))

    # 遍历半径矩阵的每个元素
    max_radius = int(np.max(radius_matrix))
    #     print(max_radius)
    for i in range(rows):
        for j in range(cols):
            #             print("--------------i,j: {} {}".format(i, j))
            activation_count = 0
            # 扩展半径范围，包括边缘情况
            y_max = min(cols - 1, j + max_radius)
            y_min = max(0, j - max_radius)
            x_min = max(0, i - max_radius)
            x_max = min(rows - 1, i + max_radius)
            #             print("y_max: {}, y_min: {}, x_min: {}, x_max: {}".format(y_max, y_min, x_min, x_max))
            for ii in range(x_min, x_max + 1):
                for jj in range(y_min, y_max + 1):
                    # print("ii: {}, jj: {}, radius_matrix :{}".format(ii,jj,radius_matrix[ii, jj]))
                    # print("距离: {} radius_matrix: {} ".format(math.sqrt((ii - i) ** 2 + (jj - j) ** 2),radius_matrix[ii, jj]))
                    if math.sqrt((ii - i) ** 2 + (jj - j) ** 2) <= radius_matrix[ii, jj] and radius_matrix[
                        ii, jj] != 0:  # / 37.04
                        activation_count = activation_count + 1
            #                         print("activation_count : {}".format(activation_count))

            # 在激活范围内更新激活矩阵的值
            activation_matrix[i, j] = activation_count
    #             print(activation_count)
    #             print(activation_matrix)
    return activation_matrix


def calculateDistance_test(Radius_Matrix, Activation_Matrix, Adjugate_Matrix):
    """
    求一次测线经过的点
    :param Radius_Matrix:  半径矩阵
    :param Activation_Matrix: 激活矩阵
    :param Adjugate_Matrix:  邻接矩阵
    :return: 计算一条测线的点分布
    """
    detection_line = []
    point = initial_point(Activation_Matrix)
    print("initial_point{}".format(point))
    detection_line.append(point)

    while True:
        Activation_Matrix = convert_radius_to_activation_test(Radius_Matrix)
        point = initial_point(Activation_Matrix)
        if point[0] == 0 or point[0] == (len(Activation_Matrix) - 1) or point[1] == 0 or point[1] == (
                len(Activation_Matrix[0]) - 1):
            detection_line.append(point)
            for point in detection_line:
                p = []
                li = activation_point_test(point[0], point[1], Radius_Matrix, Activation_Matrix)
                for value in li:
                    p.append(value)
            p = list(p)
            update_adjugate_matrix(p, Adjugate_Matrix)
            Radius_Matrix = update_radius_matrix(Radius_Matrix, Adjugate_Matrix)
            Activation_Matrix = convert_radius_to_activation_test(Radius_Matrix)

            print("--------------------------------")
            print("半径矩阵: \n{} \n\n  激活矩阵:\n {}\n\n 伴随矩阵:\n {}" \
                  .format(Radius_Matrix, Activation_Matrix, Adjugate_Matrix))
            detection_line = calculateDistance_test(Radius_Matrix, Activation_Matrix, Adjugate_Matrix)

            Activation_Matrix = convert_radius_to_activation_test(Radius_Matrix)
            print("end")
            print(f"Adjugate_Matrix:\n {Adjugate_Matrix}")
            break
        # print(point)
        next = Translate_Forward_Strategy_test(Radius_Matrix, Adjugate_Matrix, point[0], point[1])
        if next == 0:
            print("End")
            break
        point = (update_point(next, point))
        detection_line.append(point)

    return (detection_line)


def activation_point_test(x, y, radius_matrix, adjugate_matrix):
    """
    测试版本-仅适用于测试数据，给出一个点和半径矩阵，计算出所有他们能激活的点
    :param x: 激活点x坐标
    :param y: 激活点y坐标
    :param radius_matrix: 半径矩阵
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
            if math.sqrt((ii - x) ** 2 + (jj - y) ** 2) <= radius_matrix[ii, jj] and adjugate_matrix[
                ii, jj] == 0:  # / 37.04
                point = list((ii, jj))
                activation_point.append(point)

    return activation_point


def activation_point(x, y, radius_matrix, adjugate_matrix):
    """
    激活矩阵版本不重复激活——给出一个点和半径矩阵，计算出所有他们能激活的点
    :param x: 激活点x坐标
    :param y: 激活点y坐标
    :param radius_matrix: 半径矩阵
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
            if math.sqrt((ii - x) ** 2 + (jj - y) ** 2) <= radius_matrix[ii, jj] / 37.04 and adjugate_matrix[
                ii, jj] == 0:  #
                point = list((ii, jj))
                activation_point.append(point)

    return activation_point


def update_adjugate_matrix(activation_points, adjugate_matrix):
    """
    更新伴随矩阵，将激活的坐标在伴随矩阵上加一
    :param activation_points:激活点
    :param adjugate_matrix:激活前伴随矩阵
    :return:激活后的伴随矩阵
    """
    for point in activation_points:
        row, col = point
        adjugate_matrix[row][col] += 1

    return adjugate_matrix


def update_radius_matrix(radius_matrix, Adjugate_Matrix):
    assert radius_matrix.shape == Adjugate_Matrix.shape, "输入矩阵尺寸不匹配"

    updated_radius_matrix = np.where(Adjugate_Matrix == 0, radius_matrix, 0)

    return updated_radius_matrix

def endfounction_test(radius_matrix, Activation_Matrix):
    i = 1
    radius_matrix = np.array(radius_matrix)
    Activation_Matrix = np.array(Activation_Matrix)
    Adjugate_Matrix = create_adjugate_matrix(radius_matrix)
    plt.imshow(radius_matrix, cmap='viridis')  # 使用'viridis'颜色映射
    while (Adjugate_Matrix == 0).any():
        print(f"---------------第{i}次迭代----------------")
        detection_line = calculateDistance_test(radius_matrix, Activation_Matrix, Adjugate_Matrix)
        radius_matrix = update_radius_matrix(radius_matrix, Adjugate_Matrix)
        Activation_Matrix = convert_radius_to_activation_test(radius_matrix)

        i = i + 1
        plt.imshow(radius_matrix, cmap='viridis')  # 使用'viridis'颜色映射
        # 绘制图像
        plt.figure()
        plt.imshow(radius_matrix, cmap='viridis')
        plt.colorbar()
        lines_arr = np.array(detection_line)
        x_coords = lines_arr[:, 0]
        y_coords = lines_arr[:, 1]
        plt.plot(x_coords, y_coords, color='red')
        plt.show()


r_m = [[1.4886142, 0.8456213, 2.136997, 4.941508, 2.985246, 1.932116, 0.53975916, ],
       [1.7710884, 2.8999279, 1.5591438, 2.5739675, 0.36148265, 4.0796123, 0.194687,],
       [3.2411544, 1.4165895, 1.9165912, 1.760249, 4.946924, 4.0427003, 4.0804954, ],
       [3.668133, 2.023943, 4.670376, 1.4255198, 0.9299506, 0.5426398, 1.6190097, ],
       [1.7753079, 4.4532833, 1.8346862, 0.8180791, 4.368987, 1.8748162, 3.3748724,],
       [4.5770545, 3.7520282, 0.26316088, 0.32538202, 3.827253, 1.0943422, 2.193553],
       [2.9307108, 4.358497, 4.0927663, 1.3401148, 1.9592247, 3.0486445, 4.8886576,],
       [3.688964, 3.4612098, 1.2939997, 1.3174696, 0.9073608, 2.6526885, 2.472786, ],
       [0.7336474, 2.9115882, 0.19073708, 2.27222, 1.4719074, 0.9961275, 4.165045, ],
       [2.447018, 1.0646493, 3.3767536, 2.263373, 2.313686, 1.7354474, 1.9270022, ],
       [1.1160762, 1.8740808, 3.1195884, 1.3742502, 2.7280214, 1.960381, 2.7193341,],
       [4.630166, 0.84973407, 2.9772265, 2.353126, 4.3826966, 4.2475386, 3.7644026,]]
r_m = np.array(r_m)
c_m = convert_radius_to_activation_test(r_m)
a_m = create_adjugate_matrix(r_m)
# endfounction_test(r_m, c_m)


lines_point = calculateDistance_test(r_m,c_m,a_m)
r_m = np.array(r_m)
c_m = convert_radius_to_activation_test(r_m)
# endfounction_test(r_m,c_m)

a_m = np.zeros((len(r_m), len(r_m[0])))
print("Radius Matrix:")
print(r_m)
print("Activation Matrix:")
print(c_m)

point = initial_point(c_m)
print(point)

detection_line = calculateDistance_test(r_m,c_m,a_m)
print("detection_line: {}".format(detection_line))
print(a_m)

# 转换为numpy数组
lines_arr = np.array(detection_line)

# 分别获取x和y坐标数组
x_coords = lines_arr[:, 0]
y_coords = lines_arr[:, 1]

plt.imshow(r_m, cmap='viridis')  # 使用'viridis'颜色映射

# 绘制直线
plt.plot(x_coords, y_coords, color='red')
plt.colorbar()  # 添加颜色条
# plt.show()

r_m = update_radius_matrix(r_m,a_m)
c_m = convert_radius_to_activation_test(r_m)
print("Radius Matrix:")
print(r_m)
print("Activation Matrix:")
print(c_m)

detection_line = calculateDistance_test(r_m,c_m,a_m)
print("detection_line: {}".format(detection_line))
print(a_m)
# 转换为numpy数组
lines_arr = np.array(detection_line)

# 分别获取x和y坐标数组
x_coords = lines_arr[:, 0]
y_coords = lines_arr[:, 1]
# 绘制直线
plt.plot(x_coords, y_coords, color='red')
plt.colorbar()  # 添加颜色条
plt.show()
