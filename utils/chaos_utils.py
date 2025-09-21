import numpy as np


def logistic_tent_map(key_vec, shape):
    # 简化实现，实际可根据key_vec参数化
    x = key_vec[0] % 1 or 0.123
    r = 3.99
    mu = 0.5
    arr = np.zeros(shape, dtype=np.uint8)
    for i in range(np.prod(shape)):
        x = r * x * (1 - x)
        if x < mu:
            x = x / mu
        else:
            x = (1 - x) / (1 - mu)
        arr.flat[i] = int((x * 255) % 256)
    return arr


def henon_map(key_vec, shape):
    # 简化实现，实际可根据key_vec参数化
    x = key_vec[1] % 1 or 0.234
    y = key_vec[2] % 1 or 0.345
    a = 1.4
    b = 0.3
    arr = np.zeros(shape, dtype=np.uint8)
    for i in range(np.prod(shape)):
        x_new = 1 - a * x * x + y
        y_new = b * x
        x, y = x_new, y_new
        arr.flat[i] = int((abs(x) * 255) % 256)
    return arr 