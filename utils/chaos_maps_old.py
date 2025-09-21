import numpy as np

def logistic_map(x, r=4.0):
    """一维Logistic映射"""
    return r * x * (1 - x)

def tent_map(x, mu=2.0):
    """一维Tent映射"""
    if x < 0.5:
        return mu * x
    else:
        return mu * (1 - x)

def logistic_tent_system(x0, r, mu, size):
    """
    生成Logistic-Tent混沌序列
    k = {x0, r, mu}
    """
    sequence = np.zeros(size)
    x = x0
    for i in range(size):
        if x < 0.5:
            x = logistic_map(x, r)
        else:
            x = tent_map(x, mu)
        sequence[i] = x
    return sequence.reshape((int(np.sqrt(size)), -1))

def henon_map(x, y, a=1.4, b=0.3):
    """二维Henon映射"""
    x_next = 1 - a * x**2 + y
    y_next = b * x
    return x_next, y_next

def henon_system(x0, y0, a, b, size):
    """
    生成Henon混沌序列
    k = {x0, y0, a, b}
    """
    sequence = np.zeros(size)
    x, y = x0, y0
    for i in range(size):
        x, y = henon_map(x, y)
        sequence[i] = x  # Use one of the dimensions
    return sequence.reshape((int(np.sqrt(size)), -1))

class ChaosMaskGenerator:
    """混沌掩码生成器"""
    def __init__(self, key_vector):
        """
        key_vector: 包含混沌系统所需参数的字典
        e.g., {'lt_x0': 0.1, 'lt_r': 4.0, 'lt_mu': 2.0, 
                'henon_x0': 0.1, 'henon_y0': 0.1, 'henon_a': 1.4, 'henon_b': 0.3}
        """
        self.key_vector = key_vector

    def generate_mask(self, size):
        """
        生成混沌掩码 M = M1 ⊕ M2
        """
        # (1) Logistic–Tent 系统
        m1 = logistic_tent_system(
            self.key_vector['lt_x0'],
            self.key_vector['lt_r'],
            self.key_vector['lt_mu'],
            size=size*size
        )

        # (2) Henon 系统
        m2 = henon_system(
            self.key_vector['henon_x0'],
            self.key_vector['henon_y0'],
            self.key_vector['henon_a'],
            self.key_vector['henon_b'],
            size=size*size
        )
        
        # 归一化到 [0, 255]
        m1 = (m1 - np.min(m1)) / (np.max(m1) - np.min(m1)) * 255
        m2 = (m2 - np.min(m2)) / (np.max(m2) - np.min(m2)) * 255

        # (3) 异或融合
        mask = np.bitwise_xor(m1.astype(np.uint8), m2.astype(np.uint8))
        
        return mask 