import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Dict, Optional, List
import hashlib
import struct
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
import warnings

# 导入新的混沌映射模块
from utils.chaos_maps import ChaosMaskGenerator

warnings.filterwarnings('ignore')


class ResidualModeling:
    """
    残差信号建模
    实现几何和语义残差的提取与融合
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device

    def extract_depth_residual(self, image_orig: np.ndarray,
                               image_perturbed: np.ndarray,
                               monodepth_model) -> np.ndarray:
        """
        提取深度残差信号
        Args:
            image_orig: 原始图像
            image_perturbed: 扰动图像
            monodepth_model: 深度估计模型
        Returns:
            深度残差图 ΔP(S) = (D_enc - D_raw)^2
        """
        # 预测深度图
        depth_orig = monodepth_model.estimate_depth(image_orig)
        depth_perturbed = monodepth_model.estimate_depth(image_perturbed)

        # 计算深度残差
        depth_residual = (depth_perturbed - depth_orig) ** 2
        return depth_residual

    def extract_semantic_residual(self, image_orig: np.ndarray,
                                  image_perturbed: np.ndarray,
                                  clip_model) -> Dict[str, np.ndarray]:
        """
        提取多层次语义残差信号
        Args:
            image_orig: 原始图像
            image_perturbed: 扰动图像
            clip_model: CLIP模型
        Returns:
            各层语义残差字典
        """
        # 提取第4、8、12层的语义特征
        layers = [4, 8, 12]
        semantic_residuals = {}

        for layer in layers:
            features_orig = clip_model.extract_features(image_orig, layer=layer)
            features_perturbed = clip_model.extract_features(image_perturbed, layer=layer)

            features_orig_mean = np.mean(features_orig, axis=1).squeeze()
            features_pert_mean = np.mean(features_perturbed, axis=1).squeeze()
            
            # 使用1-cosine相似度作为残差
            semantic_residual = 1 - np.dot(features_orig_mean, features_pert_mean) / \
                                (np.linalg.norm(features_orig_mean) * np.linalg.norm(features_pert_mean) + 1e-8)

            semantic_residuals[f'layer_{layer}'] = semantic_residual

        return semantic_residuals

    def fuse_semantic_residuals(self, semantic_residuals: Dict[str, np.ndarray]) -> np.ndarray:
        """
        融合多层语义残差信号
        """
        residuals = list(semantic_residuals.values())
        if not residuals:
            return np.array(0.0)
        
        # 简单的加权平均
        # weights = [0.2, 0.3, 0.5] # 示例权重
        # fused_semantic = np.average(residuals, weights=weights)
        fused_semantic = np.mean(residuals)
        return fused_semantic

    def adaptive_fusion_weight(self, S: float, S_c: float = 0.14, beta: float = 0.5) -> float:
        """
        计算相变驱动的模态权重 ω(S)
        ω(S) = 1/2 * [1 + tanh(β * (S - Sc))]
        """
        omega = 0.5 * (1 + np.tanh(beta * (S - S_c)))
        return omega

    def fuse_residuals(self, depth_residual: np.ndarray,
                       fused_semantic_residual: float,
                       S: float, S_c: float = 0.14, beta: float = 0.5) -> np.ndarray:
        """
        融合深度和语义残差信号
        Φ(S) = ω(S) * ΔP(S) + (1 - ω(S)) * Δf(S)
        """
        omega = self.adaptive_fusion_weight(S, S_c, beta)

        # 调整语义残差使其与深度残差形状匹配
        semantic_residual_map = np.full_like(depth_residual, fused_semantic_residual)

        # 模态融合
        joint_residual = omega * depth_residual + (1 - omega) * semantic_residual_map
        return joint_residual


class ChaoticScrambling:
    """
    基于双混沌系统的密钥绑定扰动
    """
    def __init__(self, key_vector: dict):
        self.chaos_generator = ChaosMaskGenerator(key_vector)

    def apply_mask(self, fingerprint_tensor: np.ndarray) -> np.ndarray:
        """
        应用混沌掩码
        Φ_enc = Φ(S) ⊕ M
        """
        h, w = fingerprint_tensor.shape
        mask = self.chaos_generator.generate_mask(size=h) # h and w should be same
        
        # 确保数据类型和范围一致
        fp_int = (fingerprint_tensor * 255).astype(np.uint8)
        
        # 异或操作
        perturbed_fp = np.bitwise_xor(fp_int, mask)
        
        return perturbed_fp.astype(np.float32) / 255.0


class DynamicQuantization:
    """
    动态归一化与量化编码
    """
    def __init__(self, window_size: int = 32):
        self.window_size = window_size
        self.history = []

    def _update_history(self, data: np.ndarray):
        self.history.append(data)
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def sliding_window_quantize(self, fingerprint: np.ndarray) -> np.ndarray:
        """
        使用滑动窗口统计量进行量化
        """
        self._update_history(fingerprint)
        
        # 计算历史统计量
        if len(self.history) > 1:
            mean = np.mean(self.history)
            std = np.std(self.history)
        else:
            mean = np.mean(fingerprint)
            std = np.std(fingerprint)
        
        # 归一化和量化
        quantized = ((fingerprint - mean) / (std + 1e-8)) * 64 + 128
        quantized = np.clip(quantized, 0, 255)
        
        return quantized.astype(np.uint8)

    def quantile_robust_quantize(self, fingerprint: np.ndarray) -> np.ndarray:
        """
        使用分位数进行鲁棒量化
        """
        q_low = np.quantile(fingerprint, 0.1)
        q_high = np.quantile(fingerprint, 0.9)
        
        # 归一化和量化
        quantized = (fingerprint - q_low) / (q_high - q_low + 1e-8) * 255
        quantized = np.clip(quantized, 0, 255)

        return quantized.astype(np.uint8)


class LightweightEncoder(nn.Module):
    """
    轻量级深度编码网络
    基于MobileViTv3的混合式编码器
    """

    def __init__(self, input_channels: int = 2, output_dim: int = 128):
        super(LightweightEncoder, self).__init__()

        # 简化的MobileViTv3风格编码器
        self.features = nn.Sequential(
            # 初始卷积层
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # 深度可分离卷积块
            self._make_depthwise_block(16, 32),
            self._make_depthwise_block(32, 64),
            self._make_depthwise_block(64, 128),

            # 通道注意力机制
            ChannelAttention(128),

            # 全局平均池化
            nn.AdaptiveAvgPool2d((1, 1)),

            # 全连接层
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim),
            nn.Tanh()  # 输出范围[-1, 1]
        )

        # 可逆特征变换模块
        self.reversible_transform = ReversibleTransform(output_dim)

    def _make_depthwise_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """
        构建深度可分离卷积块
        """
        return nn.Sequential(
            # 深度卷积
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            # 点卷积
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # 下采样
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        features = self.features(x)
        # 应用可逆特征变换
        transformed = self.reversible_transform(features)
        return transformed


class ChannelAttention(nn.Module):
    """
    通道注意力机制
    """

    def __init__(self, channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return x * self.sigmoid(out).view(x.size(0), x.size(1), 1, 1)


class ReversibleTransform(nn.Module):
    """
    可逆特征变换模块
    """

    def __init__(self, feature_dim: int):
        super(ReversibleTransform, self).__init__()
        self.feature_dim = feature_dim

        # 可逆变换参数
        self.transform_matrix = nn.Parameter(torch.randn(feature_dim, feature_dim))
        self.bias = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用可逆变换
        """
        # 确保变换矩阵可逆
        transform = torch.matrix_exp(self.transform_matrix - self.transform_matrix.t())
        transformed = torch.matmul(x, transform) + self.bias

        return transformed


class VisualFingerprintGenerator:
    """
    视觉指纹生成器
    """
    def __init__(self, device: str = 'cpu', key: int = 42, key_vector: dict = None):
        self.device = device
        self.key = key
        self.residual_modeling = ResidualModeling(device=device)
        
        if key_vector is None:
            # 提供一个默认的密钥向量
            key_vector = {
                'lt_x0': 0.1, 'lt_r': 3.9, 'lt_mu': 1.9, 
                'henon_x0': 0.1, 'henon_y0': 0.1, 'henon_a': 1.4, 'henon_b': 0.3
            }
        self.chaotic_scrambler = ChaoticScrambling(key_vector)
        self.quantizer = DynamicQuantization()

    def generate_fingerprint(self, image_orig: np.ndarray,
                             image_perturbed: np.ndarray,
                             perturbation_strength: float,
                             monodepth_model,
                             clip_model,
                             S_c: float = 0.14,
                             quantization_method: str = 'quantile') -> Dict:
        """
        生成完整的视觉指纹
        """
        # 1. 提取残差
        depth_residual = self.residual_modeling.extract_depth_residual(
            image_orig, image_perturbed, monodepth_model
        )
        semantic_residuals = self.residual_modeling.extract_semantic_residual(
            image_orig, image_perturbed, clip_model
        )
        fused_semantic = self.residual_modeling.fuse_semantic_residuals(semantic_residuals)

        # 2. 模态自适应融合
        joint_residual = self.residual_modeling.fuse_residuals(
            depth_residual, fused_semantic, perturbation_strength, S_c=S_c
        )

        # 3. 混沌扰动掩码
        encrypted_residual = self.chaotic_scrambler.apply_mask(joint_residual)

        # 4. 动态归一化与量化
        if quantization_method == 'quantile':
            quantized_fingerprint = self.quantizer.quantile_robust_quantize(encrypted_residual)
        else:
            quantized_fingerprint = self.quantizer.sliding_window_quantize(encrypted_residual)

        # 5. 哈希盲化
        fingerprint_hash = self.compute_blinded_hash(quantized_fingerprint)

        return {
            'fingerprint_tensor': encrypted_residual,
            'quantized_fingerprint': quantized_fingerprint,
            'fingerprint_hash': fingerprint_hash,
            'joint_residual': joint_residual,
            'depth_residual': depth_residual,
            'semantic_residuals': semantic_residuals
        }

    def compute_blinded_hash(self, fingerprint: np.ndarray, blinding_factor: int = 12345) -> str:
        """
        计算密钥驱动的盲化哈希
        f = H(Φ_quant || (k ⊕ r))
        """
        # 准备数据
        data = fingerprint.tobytes()
        
        # 盲化因子
        blinding_data = struct.pack('I', self.key ^ blinding_factor)
        
        # 哈希
        hasher = hashlib.sha256()
        hasher.update(data)
        hasher.update(blinding_data)
        
        # 返回十六进制摘要
        return hasher.hexdigest()


class FingerprintEvaluator:
    """
    指纹评估器
    """
    def __init__(self, key: int = 42):
        self.key = key
        # self.residual_modeling = ResidualModeling()

    def compute_cmcs_star(self, S: float,
                          geometric_residual: float,
                          semantic_residual: float,
                          S_c: float = 0.14,
                          alpha: float = 0.5) -> float:
        """
        计算新的CMCS*指标
        CMCS* = Var[φ(S)] / (Sc - S), for S < Sc
        这里简化计算，不直接计算序参量的方差，而是使用一个代理
        """
        if S >= S_c:
            return 0.0
        
        # 序参量 φ(S)
        order_parameter = alpha * semantic_residual - (1 - alpha) * geometric_residual
        
        # 简化版：使用序参量绝对值作为涨落强度代理
        fluctuation_strength = np.abs(order_parameter)
        
        # CMCS*
        cmcs_star = fluctuation_strength / (S_c - S + 1e-8)
        
        return cmcs_star
    
    def compute_kdt_multi(self, base_hash: str, perturbed_hashes: Dict[str, str]) -> float:
        """
        计算密钥敏感度测试 KDT-multi
        """
        def hamming_distance(h1, h2):
            # 将十六进制哈希转换为二进制字符串
            h1_bin = bin(int(h1, 16))[2:].zfill(256)
            h2_bin = bin(int(h2, 16))[2:].zfill(256)
            return sum(c1 != c2 for c1, c2 in zip(h1_bin, h2_bin))

        total_bits = 256  # SHA-256
        key_entropy = 8 # 假设密钥熵
        
        score = 0
        
        # 扰动 {1, 4, 8} bits
        for j in [1, 4, 8]:
            key_name = f'{j}_bit'
            if key_name in perturbed_hashes:
                hd = hamming_distance(base_hash, perturbed_hashes[key_name])
                normalized_hd = hd / total_bits
                
                # 阈值 tau_j
                tau_j = (key_entropy * j) / (8 * total_bits) # 调整分母
                
                if normalized_hd > tau_j:
                    score += 1
                    
        return score / 3.0

    def evaluate_fingerprint(self, fingerprint_result: Dict, S: float) -> Dict:
        """
        评估单个指纹的质量
        """
        # CMCS* 计算
        # 简化计算，使用残差的均值
        geo_res_mean = np.mean(fingerprint_result['depth_residual'])
        sem_res_fused = np.mean(list(fingerprint_result['semantic_residuals'].values()))

        cmcs_star = self.compute_cmcs_star(S, geo_res_mean, sem_res_fused)

        return {
            'cmcs_star': cmcs_star,
            'fingerprint_hash': fingerprint_result['fingerprint_hash']
            # KDT-multi 需要多个指纹，在实验层面计算
        }
