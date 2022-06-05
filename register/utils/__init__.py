import numpy as np


def get_distance_squared(x, y):
    diff = x - y
    return np.einsum('...i,...i->...', diff, diff)
