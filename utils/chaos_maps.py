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
        # 区域性扰动分配
        mask = np.zeros((size, size), dtype=np.uint8)
        n_regions = 4
        region_size = size // n_regions
        for i in range(n_regions):
            for j in range(n_regions):
                region_key = self.key_vector.copy()
                # 对每个区域扰动不同的key参数
                region_key['lt_x0'] += 0.01 * (i + j)
                region_key['henon_x0'] += 0.01 * (i - j)
                m1 = logistic_tent_system(
                    region_key['lt_x0'], region_key['lt_r'], region_key['lt_mu'], size=region_size*region_size)
                m2 = henon_system(
                    region_key['henon_x0'], region_key['henon_y0'], region_key['henon_a'], region_key['henon_b'], size=region_size*region_size)
                m1 = (m1 - np.min(m1)) / (np.max(m1) - np.min(m1)) * 255
                m2 = (m2 - np.min(m2)) / (np.max(m2) - np.min(m2)) * 255
                region_mask = np.bitwise_xor(m1.astype(np.uint8), m2.astype(np.uint8))
                mask[i*region_size:(i+1)*region_size, j*region_size:(j+1)*region_size] = region_mask
        return mask 

    def generate_mask_float(self, size):
        """
        生成 [0,1] 连续掩码，用于逐元素乘法绑定 M(k) ∈ (0,1]
        """
        mask = np.zeros((size, size), dtype=np.float32)
        n_regions = 4
        region_size = size // n_regions
        eps = 1e-3
        for i in range(n_regions):
            for j in range(n_regions):
                region_key = self.key_vector.copy()
                region_key['lt_x0'] += 0.01 * (i + j)
                region_key['henon_x0'] += 0.01 * (i - j)
                m1 = logistic_tent_system(
                    region_key['lt_x0'], region_key['lt_r'], region_key['lt_mu'], size=region_size*region_size)
                m2 = henon_system(
                    region_key['henon_x0'], region_key['henon_y0'], region_key['henon_a'], region_key['henon_b'], size=region_size*region_size)
                # 归一化到 (0,1]
                m1 = (m1 - np.min(m1)) / (np.max(m1) - np.min(m1) + 1e-8)
                m2 = (m2 - np.min(m2)) / (np.max(m2) - np.min(m2) + 1e-8)
                region_mask = 0.5 * (m1 + m2)
                region_mask = np.clip(region_mask, eps, 1.0)
                mask[i*region_size:(i+1)*region_size, j*region_size:(j+1)*region_size] = region_mask.astype(np.float32)
        return mask