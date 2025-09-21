import numpy as np
import cv2
from typing import Tuple, Optional
import pywt
from scipy import ndimage

class LogisticTentMap:
    """
    Logistic-Tent混合混沌映射
    用于生成密钥敏感的混沌扰动序列
    """
    
    def __init__(self, r: float = 3.9, a: float = 0.7, key: int = 42):
        """
        初始化混沌映射参数
        Args:
            r: Logistic映射参数
            a: Tent映射参数
            key: 密钥种子
        """
        self.r = r
        self.a = a
        self.key = key
        np.random.seed(key)
        self.x = np.random.random()
    
    def logistic_map(self, x: float) -> float:
        """Logistic映射"""
        return self.r * x * (1 - x)
    
    def tent_map(self, x: float) -> float:
        """Tent映射"""
        if x < self.a:
            return x / self.a
        else:
            return (1 - x) / (1 - self.a)
    
    def hybrid_map(self, x: float) -> float:
        """混合映射"""
        logistic_val = self.logistic_map(x)
        tent_val = self.tent_map(x)
        return 0.5 * (logistic_val + tent_val)
    
    def generate_sequence(self, length: int) -> np.ndarray:
        """
        生成混沌序列
        Args:
            length: 序列长度
        Returns:
            混沌序列
        """
        sequence = np.zeros(length)
        x = self.x
        
        for i in range(length):
            x = self.hybrid_map(x)
            sequence[i] = x
        
        return sequence
    
    def generate_2d_chaos(self, height: int, width: int) -> np.ndarray:
        """
        生成2D混沌扰动矩阵
        Args:
            height: 图像高度
            width: 图像宽度
        Returns:
            2D混沌矩阵
        """
        # 生成两个独立的混沌序列
        seq1 = self.generate_sequence(height)
        seq2 = self.generate_sequence(width)
        
        # 外积生成2D矩阵
        chaos_2d = np.outer(seq1, seq2)
        
        # 归一化到[-1, 1]
        chaos_2d = 2 * chaos_2d - 1
        
        return chaos_2d

class WaveletChaosPerturbation:
    """
    小波混沌扰动模块
    结合离散小波变换和混沌映射进行结构域扰动
    """
    
    def __init__(self, wavelet: str = 'db4', key: int = 42):
        """
        初始化小波混沌扰动
        Args:
            wavelet: 小波基函数
            key: 混沌映射密钥
        """
        self.wavelet = wavelet
        self.chaos_generator = LogisticTentMap(key=key)
    
    def wavelet_decompose(self, image: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        对图像进行小波分解
        Args:
            image: 输入图像 [H, W, C]
        Returns:
            小波系数 (LL, LH, HL, HH)
        """
        if len(image.shape) == 3:
            # 彩色图像，分别处理每个通道
            coeffs_list = []
            for c in range(image.shape[2]):
                coeffs = pywt.wavedec2(image[:, :, c], self.wavelet, level=1)
                coeffs_list.append(coeffs)
            
            # 提取各子带
            LL = np.stack([coeffs[0] for coeffs in coeffs_list], axis=-1)
            LH = np.stack([coeffs[1][0] for coeffs in coeffs_list], axis=-1)
            HL = np.stack([coeffs[1][1] for coeffs in coeffs_list], axis=-1)
            HH = np.stack([coeffs[1][2] for coeffs in coeffs_list], axis=-1)
            
            return LL, LH, HL, HH
        else:
            # 灰度图像
            coeffs = pywt.wavedec2(image, self.wavelet, level=1)
            return coeffs[0], coeffs[1][0], coeffs[1][1], coeffs[1][2]
    
    def wavelet_reconstruct(self, LL: np.ndarray, LH: np.ndarray, 
                          HL: np.ndarray, HH: np.ndarray) -> np.ndarray:
        """
        小波重构
        Args:
            LL, LH, HL, HH: 小波系数
        Returns:
            重构图像
        """
        if len(LL.shape) == 3:
            # 彩色图像
            reconstructed_channels = []
            for c in range(LL.shape[2]):
                coeffs = (LL[:, :, c], (LH[:, :, c], HL[:, :, c], HH[:, :, c]))
                reconstructed = pywt.waverec2(coeffs, self.wavelet)
                reconstructed_channels.append(reconstructed)
            return np.stack(reconstructed_channels, axis=-1)
        else:
            # 灰度图像
            coeffs = (LL, (LH, HL, HH))
            return pywt.waverec2(coeffs, self.wavelet)
    
    def apply_chaos_perturbation(self, HH: np.ndarray, strength: float, 
                                key: int) -> np.ndarray:
        """
        对高频子带应用混沌扰动
        Args:
            HH: 高频子带
            strength: 扰动强度 [0, 1]
            key: 密钥
        Returns:
            扰动后的高频子带
        """
        # 更新混沌生成器的密钥
        self.chaos_generator.key = key
        
        # 生成混沌扰动
        height, width = HH.shape[:2]
        chaos_perturb = self.chaos_generator.generate_2d_chaos(height, width)
        
        # 应用扰动
        if len(HH.shape) == 3:
            # 彩色图像
            HH_perturbed = HH.copy()
            for c in range(HH.shape[2]):
                HH_perturbed[:, :, c] = HH[:, :, c] + strength * chaos_perturb
        else:
            # 灰度图像
            HH_perturbed = HH + strength * chaos_perturb
        
        return HH_perturbed
    
    def perturb_image(self, image: np.ndarray, strength: float, 
                     key: int, alpha: float = 0.5) -> np.ndarray:
        """
        对图像应用小波混沌扰动，支持alpha分配LL/HH扰动强度
        Args:
            image: 输入图像
            strength: 扰动强度 [0, 1]
            key: 密钥
            alpha: LL子带扰动分配因子 [0,1]
        Returns:
            扰动后的图像
        """
        # 确保图像值在[0, 1]范围内
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # 小波分解
        LL, LH, HL, HH = self.wavelet_decompose(image)

        # 对LL/HH子带分别应用混沌扰动
        LL_perturbed = self.apply_chaos_perturbation(LL, alpha * strength, key)
        HH_perturbed = self.apply_chaos_perturbation(HH, (1 - alpha) * strength, key)
        
        # 其余子带不变
        perturbed_image = self.wavelet_reconstruct(LL_perturbed, LH, HL, HH_perturbed)
        perturbed_image = np.clip(perturbed_image, 0, 1)
        return perturbed_image
    
    def visualize_wavelet_bands(self, image: np.ndarray) -> dict:
        """
        可视化小波子带
        Args:
            image: 输入图像
        Returns:
            包含各子带的字典
        """
        LL, LH, HL, HH = self.wavelet_decompose(image)
        
        # 归一化子带用于可视化
        def normalize_band(band):
            band_norm = (band - band.min()) / (band.max() - band.min() + 1e-8)
            return np.clip(band_norm, 0, 1)
        
        return {
            'LL': normalize_band(LL),
            'LH': normalize_band(LH),
            'HL': normalize_band(HL),
            'HH': normalize_band(HH)
        } 