import numpy as np
import cv2
from typing import Tuple, Optional, Union
import torch
from utils.wavelet_chaos import WaveletChaosPerturbation
from utils.diffusion_perturb import DiffusionPerturbation
import hashlib
from utils.chaos_utils import logistic_tent_map, henon_map


class DCEModule:
    """
    DCE (Dual-domain Cryptographic Encryption) 模块
    集成结构域扰动（小波混沌）与语义域扰动（条件扩散生成）的混合机制
    """

    def __init__(self, wavelet_type: str = 'db4', image_size: Union[int, Tuple[int, int]] = 256,
                 key: int = 42, device: str = 'cpu'):
        """
        初始化DCE模块
        Args:
            wavelet_type: 小波基函数类型
            image_size: 图像尺寸
            key: 密钥
            device: 计算设备
        """
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = tuple(image_size)

        self.wavelet_type = wavelet_type
        self.image_size = image_size
        self.key = key
        self.device = device

        # 初始化扰动模块
        self.wavelet_chaos = WaveletChaosPerturbation(wavelet=wavelet_type, key=key)
        self.diffusion_perturb = DiffusionPerturbation(image_size=image_size)

        print(f"DCE模块初始化完成 - 小波类型: {wavelet_type}, 图像尺寸: {image_size}, 密钥: {key}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        # 转RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        target_h, target_w = self.image_size
        if image.shape[:2] != (target_h, target_w):
            image = cv2.resize(image, (target_w, target_h))  # 注意cv2.resize传(w,h)

        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        return image

    def apply_wavelet_chaos_perturbation(self, image: np.ndarray,
                                         strength: float, alpha: float = 0.5) -> np.ndarray:
        """
        应用小波混沌扰动（结构域），支持alpha参数
        Args:
            image: 输入图像
            strength: 扰动强度 [0, 1]
            alpha: LL子带扰动分配因子 [0,1]
        Returns:
            扰动后的图像
        """
        if strength == 0:
            return image
        return self.wavelet_chaos.perturb_image(image, strength, self.key, alpha=alpha)

    def apply_diffusion_perturbation(self, image: np.ndarray,
                                     strength: float) -> np.ndarray:
        """
        应用扩散模型扰动（语义域）
        Args:
            image: 输入图像
            strength: 扰动强度 [0, 1]
        Returns:
            扰动后的图像
        """
        if strength == 0:
            return image
        return self.diffusion_perturb.embedded_diffusion_perturb(image, strength, self.key)

    def apply_color_perturbation(self, image: np.ndarray, strength: float, seed: Optional[int] = None) -> np.ndarray:
        """
        颜色空间扰动（Lab + PCA 主方向注入）：ΔC = S · δ · u
        - image: [H,W,3] in [0,1]
        - strength S ∈ [0,1]
        """
        if strength <= 0:
            return image

        # 转 uint8 便于 Lab 转换
        img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)

        h, w, _ = lab.shape
        flat = lab.reshape(-1, 3)

        # 计算协方差与主方向
        cov = np.cov(flat, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argmax(eigvals)
        u = eigvecs[:, idx]  # 主方向，shape [3]

        # 设定 δ 标准差（使用 L 通道标准差作为尺度）
        sigma = float(np.std(flat[:, 0]) + 1e-6)

        # 可复现实验的随机源
        rs = np.random.RandomState((self.key if seed is None else int(seed)) % 2**32)
        delta = rs.normal(loc=0.0, scale=sigma)

        dC = float(strength) * float(delta) * u.astype(np.float32)

        # 应用扰动
        lab_pert = lab.reshape(-1, 3) + dC[None, :]
        # 限幅到 uint8 范围
        lab_pert = np.clip(lab_pert, 0, 255).reshape(h, w, 3).astype(np.uint8)

        rgb_pert = cv2.cvtColor(lab_pert, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        return np.clip(rgb_pert, 0.0, 1.0)

    def alpha_sigmoid(self, S, S0=0.5, gamma=12):
        """
        结构-语义扰动分配Sigmoid函数
        α(S) = 1 / (1 + exp[-γ(S - S0)])
        """
        return 1.0 / (1.0 + np.exp(-gamma * (S - S0)))

    def hybrid_perturbation(self, image: np.ndarray, strength: float,
                            S0: float = 0.5, gamma: float = 12,
                            color_ratio: float = 0.3) -> np.ndarray:
        """
        混合扰动：结合小波混沌和扩散扰动，采用Sigmoid型α(S)分配结构/语义扰动强度
        Args:
            image: 输入图像
            strength: 扰动强度 [0, 1]
            S0: Sigmoid转折点
            gamma: Sigmoid斜率
        Returns:
            混合扰动后的图像
        """
        # 预处理图像
        image_processed = self.preprocess_image(image)

        if strength == 0:
            return image_processed

        # 结构-语义扰动分配
        alpha = self.alpha_sigmoid(strength, S0, gamma)
        SLL = alpha * strength
        SHH = (1 - alpha) * strength
        SCOL = color_ratio * SHH  # 在语义份额中切出颜色扰动比例

        # 小波混沌扰动（结构域，低频LL）
        image_wavelet = self.apply_wavelet_chaos_perturbation(image_processed, SLL, alpha=alpha)

        # 扩散模型扰动（语义域，高频HH）
        image_diffusion = self.apply_diffusion_perturbation(image_wavelet, SHH)

        # 颜色扰动（跨模态 Lab 主方向注入）
        image_color = self.apply_color_perturbation(image_diffusion, SCOL)

        return image_color

    def encrypt_image(self, image: np.ndarray, strength: float,
                      perturbation_type: str = 'hybrid') -> np.ndarray:
        """
        图像加密主函数
        Args:
            image: 输入图像
            strength: 扰动强度 [0, 1]
            perturbation_type: 扰动类型 ('wavelet', 'diffusion', 'hybrid')
        Returns:
            加密后的图像
        """
        if perturbation_type == 'wavelet':
            return self.apply_wavelet_chaos_perturbation(image, strength)
        elif perturbation_type == 'diffusion':
            return self.apply_diffusion_perturbation(image, strength)
        elif perturbation_type == 'color':
            return self.apply_color_perturbation(self.preprocess_image(image), strength)
        elif perturbation_type == 'hybrid':
            return self.hybrid_perturbation(image, strength)
        else:
            raise ValueError(f"不支持的扰动类型: {perturbation_type}")

    def visualize_perturbation_effects(self, image: np.ndarray,
                                       strength: float = 0.5) -> dict:
        """
        可视化不同扰动类型的效果
        Args:
            image: 输入图像
            strength: 扰动强度
        Returns:
            包含不同扰动结果的字典
        """
        # 预处理图像
        image_processed = self.preprocess_image(image)

        # 应用不同类型的扰动
        results = {
            'original': image_processed,
            'wavelet': self.apply_wavelet_chaos_perturbation(image_processed, strength),
            'diffusion': self.apply_diffusion_perturbation(image_processed, strength),
            'hybrid': self.hybrid_perturbation(image_processed, strength)
        }

        return results

    def analyze_perturbation_strength(self, image: np.ndarray,
                                      strengths: np.ndarray) -> dict:
        """
        分析不同扰动强度下的效果
        Args:
            image: 输入图像
            strengths: 扰动强度数组
        Returns:
            分析结果
        """
        image_processed = self.preprocess_image(image)
        results = {}

        for strength in strengths:
            # 应用混合扰动
            perturbed = self.hybrid_perturbation(image_processed, strength)

            # 计算扰动强度指标
            mse = np.mean((perturbed - image_processed) ** 2)
            psnr_val = -10 * np.log10(mse + 1e-8)

            results[strength] = {
                'perturbed_image': perturbed,
                'mse': mse,
                'psnr': psnr_val
            }

        return results

    def generate_chaos_mask(self, shape, key_vec):
        """
        生成混沌掩码（Logistic-Tent和Henon映射异或融合）
        key_vec: 密钥向量
        shape: 掩码形状
        """
        M1 = logistic_tent_map(key_vec, shape)
        M2 = henon_map(key_vec, shape)
        M = np.bitwise_xor(M1, M2)
        return M

    def fingerprint_tensor(self, delta_P, delta_f, S, S1, S2, gamma=10):
        """
        生成连续融合指纹张量 Φ(S)
        """
        Sc = (S1 + S2) / 2
        omega = 1 / (1 + np.exp(-gamma * (S - Sc)))
        Phi = omega * delta_P + (1 - omega) * delta_f
        return Phi

    def encode_fingerprint(self, Phi, key_vec, mu_t, sigma_t):
        """
        指纹张量归一化量化并混沌掩码加密
        """
        mask = self.generate_chaos_mask(Phi.shape, key_vec)
        Phi_enc = np.bitwise_xor(Phi.astype(np.uint8), mask.astype(np.uint8))
        Phi_norm = (Phi_enc - mu_t) / (sigma_t + 1e-8) * 64 + 128
        Phi_quant = np.clip(np.round(Phi_norm), 0, 255).astype(np.uint8)
        return Phi_quant

    def hash_fingerprint(self, Phi_quant, key_vec, r=None, hash_len=256):
        """
        哈希盲化映射，输出二进制指纹
        """
        if r is None:
            r = np.random.randint(0, 256, size=len(key_vec), dtype=np.uint8)
        key_blind = np.bitwise_xor(np.array(key_vec, dtype=np.uint8), r)
        concat = np.concatenate([Phi_quant.flatten(), key_blind])
        h = hashlib.sha256(concat.tobytes()).digest()
        bits = np.unpackbits(np.frombuffer(h, dtype=np.uint8))[:hash_len]
        return bits

    def generate_fingerprint(self, delta_P, delta_f, S, S1, S2, key_vec, mu_t, sigma_t, hash_len=256):
        """
        一站式生成密钥绑定视觉指纹
        """
        Phi = self.fingerprint_tensor(delta_P, delta_f, S, S1, S2)
        Phi_quant = self.encode_fingerprint(Phi, key_vec, mu_t, sigma_t)
        bits = self.hash_fingerprint(Phi_quant, key_vec, hash_len=hash_len)
        return bits


class DCEController:
    """
    DCE控制器
    管理扰动强度与密钥的协同控制
    """

    def __init__(self, base_key: int = 42):
        """
        初始化DCE控制器
        Args:
            base_key: 基础密钥
        """
        self.base_key = base_key
        self.dce_modules = {}

    def create_dce_module(self, module_id: str, wavelet_type: str = 'db4',
                          image_size: Union[int, Tuple[int, int]] = 256) -> DCEModule:
        key = self.base_key + hash(module_id) % 1000
        dce_module = DCEModule(
            wavelet_type=wavelet_type,
            image_size=image_size,
            key=key
        )
        self.dce_modules[module_id] = dce_module
        return dce_module

    def get_dce_module(self, module_id: str) -> Optional[DCEModule]:
        """
        获取DCE模块
        Args:
            module_id: 模块ID
        Returns:
            DCE模块实例
        """
        return self.dce_modules.get(module_id)

    def batch_encrypt(self, images: list, strengths: list,
                      module_id: str = 'default') -> list:
        """
        批量加密
        Args:
            images: 图像列表
            strengths: 扰动强度列表
            module_id: 模块ID
        Returns:
            加密后的图像列表
        """
        if module_id not in self.dce_modules:
            self.create_dce_module(module_id)

        dce_module = self.dce_modules[module_id]
        encrypted_images = []

        for image, strength in zip(images, strengths):
            encrypted = dce_module.encrypt_image(image, strength, 'hybrid')
            encrypted_images.append(encrypted)

        return encrypted_images

    def adaptive_encryption(self, image: np.ndarray, target_psnr: float = 30.0,
                            module_id: str = 'default') -> Tuple[np.ndarray, float]:
        """
        自适应加密：根据目标PSNR调整扰动强度
        Args:
            image: 输入图像
            target_psnr: 目标PSNR
            module_id: 模块ID
        Returns:
            (加密图像, 实际扰动强度)
        """
        if module_id not in self.dce_modules:
            self.create_dce_module(module_id)

        dce_module = self.dce_modules[module_id]

        # 二分搜索找到合适的扰动强度
        low, high = 0.0, 1.0
        best_strength = 0.5
        best_psnr_diff = float('inf')

        for _ in range(10):  # 最多迭代10次
            strength = (low + high) / 2
            encrypted = dce_module.encrypt_image(image, strength, 'hybrid')

            # 计算PSNR
            mse = np.mean((encrypted - image) ** 2)
            psnr_val = -10 * np.log10(mse + 1e-8)

            psnr_diff = abs(psnr_val - target_psnr)

            if psnr_diff < best_psnr_diff:
                best_psnr_diff = psnr_diff
                best_strength = strength

            if psnr_val > target_psnr:
                high = strength
            else:
                low = strength

        # 使用最佳强度进行最终加密
        final_encrypted = dce_module.encrypt_image(image, best_strength, 'hybrid')

        return final_encrypted, best_strength
