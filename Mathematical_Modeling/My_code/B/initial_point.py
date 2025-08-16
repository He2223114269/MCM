import numpy as np
import pandas as pd

def initial_point(radius_matrix):
    max_index = np.unravel_index(np.argmax(radius_matrix), radius_matrix.shape)
    return max_index