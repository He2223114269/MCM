import numpy as np


def update_radius_matrix(radius_matrix, Adjugate_Matrix):
    assert radius_matrix.shape == Adjugate_Matrix.shape, "输入矩阵尺寸不匹配"

    updated_radius_matrix = np.where(Adjugate_Matrix == 0, radius_matrix, 0)

    return updated_radius_matrix
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


Radius_Matrix = np.array([[1, 2, 3],
                         [2, 3, 5],
                         [2, 5, 1]])

Adjugate_Matrix = np.array([[1, 2, 3],
                         [2, 0, 5],
                         [2, 5, 1]])
Radius_Matrix = update_radius_matrix(Radius_Matrix, Adjugate_Matrix)
print(Radius_Matrix)