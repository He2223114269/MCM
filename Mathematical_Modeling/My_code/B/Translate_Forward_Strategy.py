import numpy as np
import pandas as pd
import math
def Translate_Forward_Strategy(radius_matrix,adjoint_matrix,x,y):
    rows = len(radius_matrix)
    cols = len(radius_matrix[0])
    max_radius = int(np.max(radius_matrix))
    x_left = x_right = y_up = y_down = 0
    x_temp = x - 1
    if  adjoint_matrix[x_temp,y] == 0 and x_temp-max_radius>0:
        y_min = max(0,y - max_radius)
        y_max = min(cols,y + max_radius)
        increase_count = 0
        for j in range(y_min,y_max):
           if math.sqrt(max_radius ** 2 + (y - j) ** 2) <= radius_matrix[x_temp - max_radius, j]/ 37.04:
               increase_count = increase_count + 1

        x_left = increase_count

    x_temp = x + 1
    if  adjoint_matrix[x_temp,y] == 0 and x_temp + max_radius < rows-1:
        y_min = max(0,y - max_radius)
        y_max = min(cols,y + max_radius)
        increase_count = 0
        for j in range(y_min,y_max):
           if math.sqrt(max_radius ** 2 + (y - j) ** 2) <= radius_matrix[x_temp + max_radius, j]/ 37.04:
               increase_count = increase_count + 1

        x_right = increase_count

    y_temp = y - 1
    if adjoint_matrix[x, y_temp] == 0 and y_temp - max_radius > 0:
        x_min = max(0, x - max_radius)
        x_max = min(cols, x + max_radius)
        increase_count = 0
        for i in range(x_min, x_max):
            if math.sqrt(max_radius ** 2 + (x - i) ** 2) <= radius_matrix[y_temp - max_radius, i] / 37.04:
                increase_count = increase_count + 1

        y_up = increase_count

    y_temp = y + 1
    if adjoint_matrix[x, y_temp] == 0 and y_temp + max_radius < cols -1:
        x_min = max(0, x - max_radius)
        x_max = min(cols, x + max_radius)
        increase_count = 0
        for i in range(x_min, x_max):
            if math.sqrt(max_radius ** 2 + (x - i) ** 2) <= radius_matrix[y_temp + max_radius, i] / 37.04:
                increase_count = increase_count + 1

        y_down = increase_count

    import random

    def select_max_variable(variables):
        max_value = max(variables.values())
        max_variables = [var for var, value in variables.items() if value == max_value]
        selected_variable = random.choice(max_variables)
        return selected_variable

    variables = {"x_left": x_left, "x_right": x_right, "y_up": y_up, "y_down": y_down}
    # 调用函数选择最大的数值对应的变量
    selected_variable = select_max_variable(variables)

    return selected_variable

