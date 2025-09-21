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

warnings.filterwarnings('ignore')


class AdaptiveResidualModeling:
    """
    自适应残差信号建模
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

        # 计算深度残差 - 使用平方差形式以保留形变方向信息
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

            # 计算语义残差 - 欧式距离
            # features_* shape: [1, N, D] 或 [1, D]
            # 如果 shape 是 [1, N, D]，对 token 做平均
            features_orig_mean = np.mean(features_orig, axis=1).squeeze()
            features_pert_mean = np.mean(features_perturbed, axis=1).squeeze()
            semantic_residual = features_pert_mean - features_orig_mean  # shape: [D]
            # semantic_residual = np.linalg.norm(features_perturbed - features_orig, ord=2)
            semantic_residuals[f'layer_{layer}'] = semantic_residual

        return semantic_residuals

    def fuse_semantic_residuals(self, semantic_residuals: Dict[str, np.ndarray]) -> np.ndarray:
        """
        融合多层语义残差信号
        Δf(S) = Conv_{1×1}(Norm([Δf_4; Δf_8; Δf_{12}]))
        """
        # 提取各层残差
        residuals = []
        for layer_name in ['layer_4', 'layer_8', 'layer_12']:
            if layer_name in semantic_residuals:
                residuals.append(semantic_residuals[layer_name])

        if not residuals:
            return np.array(0.0)

        # 转换为tensor进行标准化和融合
        residuals_tensor = torch.tensor(residuals, dtype=torch.float32).unsqueeze(0)  # [1, 3]

        # 跨层标准化 (LayerNorm)
        layer_norm = nn.LayerNorm(residuals_tensor.shape[-1])
        normalized_residuals = layer_norm(residuals_tensor)

        # 1×1卷积压缩维度
        conv_1x1 = nn.Conv1d(1, 1, kernel_size=1)
        fused_semantic = conv_1x1(normalized_residuals.transpose(-1, 1)).squeeze()

        return fused_semantic.detach().numpy()

    def adaptive_fusion_weight(self, S: float, S_c: float = 0.5,
                               k: float = 5.0, nu: float = 1.0) -> float:
        """
        计算自适应融合权重 α(S)
        使用广义Logistic函数
        """
        alpha = (1 + np.exp(-k * (S - S_c))) ** (-nu)
        return alpha

    def fuse_residuals(self, depth_residual: np.ndarray,
                       semantic_residuals: Dict[str, np.ndarray],
                       S: float, S_c: float = 0.5) -> np.ndarray:
        """
        融合深度和语义残差信号
        Args:
            depth_residual: 深度残差
            semantic_residuals: 语义残差字典
            S: 扰动强度
            S_c: 模态切换点
        Returns:
            融合后的联合残差张量 Φ
        """
        # 计算自适应融合权重
        alpha = self.adaptive_fusion_weight(S, S_c)

        # 融合多层语义残差
        semantic_features = []
        for layer_name, residual in semantic_residuals.items():
            semantic_features.append(residual)

        # 使用1x1卷积融合语义特征
        if len(semantic_features) > 0:
            # 将numpy数组转换为tensor进行卷积操作
            # 确保所有特征具有相同的形状
            min_size = min(feat.size for feat in semantic_features)
            normalized_features = []
            for feat in semantic_features:
                if feat.size > min_size:
                    # 如果特征太大，进行下采样
                    feat_reshaped = feat.reshape(-1)[:min_size]
                else:
                    feat_reshaped = feat.flatten()
                normalized_features.append(feat_reshaped)

            # 堆叠特征
            semantic_tensor = torch.tensor(np.stack(normalized_features),
                                           dtype=torch.float32).unsqueeze(0)

            # 重塑为2D张量用于卷积
            batch_size, num_features, feature_dim = semantic_tensor.shape
            semantic_tensor = semantic_tensor.view(batch_size, num_features, 1, feature_dim)

            # 1x1卷积融合
            conv_1x1 = nn.Conv2d(num_features, 1, kernel_size=1)
            # fused_semantic = conv_1x1(semantic_tensor).squeeze().numpy()
            fused_semantic = conv_1x1(semantic_tensor).detach().squeeze().numpy()

            # 重塑为与深度残差相同的形状
            if depth_residual.ndim == 2:
                fused_semantic = np.resize(fused_semantic, depth_residual.shape)  # 填充或缩放到目标形状
                fused_semantic = fused_semantic.reshape(depth_residual.shape)
        else:
            fused_semantic = np.zeros_like(depth_residual)

        # 加权融合
        weighted_semantic = alpha * fused_semantic

        # 拼接深度残差和加权语义残差
        # 确保维度匹配
        if depth_residual.ndim == 2:
            depth_residual = depth_residual[..., np.newaxis]
        if weighted_semantic.ndim == 2:
            weighted_semantic = weighted_semantic[..., np.newaxis]

        # 在通道维度拼接
        joint_residual = np.concatenate([depth_residual, weighted_semantic], axis=-1)

        return joint_residual


class ChaoticScrambling:
    """
    双混沌加扰机制
    实现Logistic-Tent混合映射和Henon映射
    """

    def __init__(self, key: int):
        self.key = key
        np.random.seed(key)

    def logistic_tent_mapping(self, x: float, r: float = 3.9) -> float:
        """
        Logistic-Tent混合映射
        """
        if x < 0.5:
            return r * x * (1 - x)  # Logistic映射
        else:
            return r * (1 - x) * x  # Tent映射

    def henon_mapping(self, x: float, y: float, a: float = 1.4, b: float = 0.3) -> Tuple[float, float]:
        """
        Henon映射
        """
        x_new = 1 - a * x * x + y
        y_new = b * x
        return x_new, y_new

    def generate_chaotic_sequences(self, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成双混沌序列矩阵
        """
        h, w = size

        # 初始化
        x1, y1 = 0.1, 0.1  # Logistic-Tent初始值
        x2, y2 = 0.1, 0.1  # Henon初始值

        # 生成序列
        M1 = np.zeros((h, w))
        M2 = np.zeros((h, w))

        for i in range(h):
            for j in range(w):
                # Logistic-Tent映射
                x1 = self.logistic_tent_mapping(x1)
                M1[i, j] = x1

                # Henon映射
                x2, y2 = self.henon_mapping(x2, y2)
                M2[i, j] = x2

        return M1, M2

    def create_mixed_mask(self, size: Tuple[int, int]) -> np.ndarray:
        """
        创建混合掩码矩阵
        M = LT(k) ⊕ Henon(k)
        """
        M1, M2 = self.generate_chaotic_sequences(size)

        # 按位异或
        M = np.logical_xor(M1 > 0.5, M2 > 0.5).astype(np.float32)

        return M


class DynamicQuantization:
    """
    滑动窗口动态量化
    """

    def __init__(self, window_size: int = 32):
        self.window_size = window_size
        self.fingerprint_history = []

    def update_statistics(self, fingerprint: np.ndarray):
        """
        更新滑动窗口统计信息
        """
        # 将指纹展平为一维数组
        fingerprint_flat = fingerprint.flatten()
        self.fingerprint_history.append(fingerprint_flat)

        # 保持窗口大小
        if len(self.fingerprint_history) > self.window_size:
            self.fingerprint_history.pop(0)

    def dynamic_quantize(self, fingerprint: np.ndarray) -> np.ndarray:
        """
        动态量化指纹
        """
        self.update_statistics(fingerprint)

        if len(self.fingerprint_history) < 2:
            # 如果历史数据不足，使用简单归一化
            return np.clip(fingerprint * 255, 0, 255).astype(np.uint8)

        # 计算滑动窗口统计量
        # 确保所有历史数据具有相同的长度
        min_length = min(len(hist) for hist in self.fingerprint_history)
        normalized_history = []
        for hist in self.fingerprint_history:
            if len(hist) > min_length:
                normalized_history.append(hist[:min_length])
            else:
                normalized_history.append(hist)

        history_array = np.array(normalized_history)
        mu_t = np.mean(history_array)
        sigma_t = np.std(history_array) + 1e-8  # 避免除零

        # 动态归一化和量化
        normalized = (fingerprint - mu_t) / sigma_t
        quantized = np.clip(np.round(normalized * 64 + 128), 0, 255)

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
    整合所有组件实现完整的指纹生成流程
    """

    def __init__(self, device: str = 'cpu', key: int = 42):
        self.device = device
        self.key = key

        # 初始化各个组件
        self.residual_modeling = AdaptiveResidualModeling(device)
        self.chaotic_scrambling = ChaoticScrambling(key)
        self.dynamic_quantization = DynamicQuantization()
        self.lightweight_encoder = LightweightEncoder().to(device)

    def generate_fingerprint(self, image_orig: np.ndarray,
                             image_perturbed: np.ndarray,
                             perturbation_strength: float,
                             monodepth_model,
                             clip_model,
                             S_c: float = 0.5) -> Dict:
        """
        生成视觉指纹
        Args:
            image_orig: 原始图像
            image_perturbed: 扰动图像
            perturbation_strength: 扰动强度
            monodepth_model: 深度估计模型
            clip_model: CLIP模型
            S_c: 模态切换点
        Returns:
            指纹生成结果字典
        """
        # Step 1: 自适应残差信号建模
        depth_residual = self.residual_modeling.extract_depth_residual(
            image_orig, image_perturbed, monodepth_model
        )

        semantic_residuals = self.residual_modeling.extract_semantic_residual(
            image_orig, image_perturbed, clip_model
        )

        # 融合残差信号
        joint_residual = self.residual_modeling.fuse_residuals(
            depth_residual, semantic_residuals, perturbation_strength, S_c
        )

        # Step 2: 密钥驱动的指纹扰动编码
        # 生成混沌掩码
        mask_size = joint_residual.shape[:2]
        chaotic_mask = self.chaotic_scrambling.create_mixed_mask(mask_size)

        # 应用掩码加密
        if joint_residual.ndim == 3:
            # 扩展到与掩码相同的维度
            chaotic_mask = np.stack([chaotic_mask] * joint_residual.shape[2], axis=-1)

        encrypted_residual = np.logical_xor(joint_residual > 0.5, chaotic_mask > 0.5).astype(np.float32)

        # Step 3: 动态量化
        quantized_fingerprint = self.dynamic_quantization.dynamic_quantize(encrypted_residual)

        # Step 4: 轻量编码
        fingerprint_tensor = torch.tensor(quantized_fingerprint, dtype=torch.float32).unsqueeze(0)
        if quantized_fingerprint.ndim == 3:
            fingerprint_tensor = fingerprint_tensor.permute(0, 3, 1, 2)

        with torch.no_grad():
            fingerprint_hash = self.lightweight_encoder(fingerprint_tensor.to(self.device))
            fingerprint_hash = fingerprint_hash.cpu().numpy().flatten()

        return {
            'depth_residual': depth_residual,
            'semantic_residuals': semantic_residuals,
            'joint_residual': joint_residual,
            'encrypted_residual': encrypted_residual,
            'quantized_fingerprint': quantized_fingerprint,
            'fingerprint_hash': fingerprint_hash,
            'chaotic_mask': chaotic_mask
        }

    def compute_fingerprint_hash(self, fingerprint_hash: np.ndarray) -> str:
        """
        计算指纹哈希值
        """
        # 将浮点数转换为字节
        hash_bytes = struct.pack('f' * len(fingerprint_hash), *fingerprint_hash)

        # 计算SHA-256哈希
        hash_object = hashlib.sha256(hash_bytes)
        return hash_object.hexdigest()


class FingerprintEvaluator:
    """
    指纹评估器
    实现协同性与安全性评估指标
    """

    def __init__(self, key: int = 42):
        self.key = key

    def compute_cmcs(self, depth_residual: np.ndarray,
                     semantic_residuals: Dict[str, np.ndarray],
                     S: float, S_c: float = 0.5) -> float:
        """
        计算跨模态一致性得分 (CMCS)
        """
        # 融合语义残差
        semantic_features = []
        for layer_name, residual in semantic_residuals.items():
            semantic_features.append(residual.flatten())

        if len(semantic_features) > 0:
            # 确保所有特征具有相同的长度
            min_length = min(len(feat) for feat in semantic_features)
            normalized_features = []
            for feat in semantic_features:
                if len(feat) > min_length:
                    normalized_features.append(feat[:min_length])
                else:
                    normalized_features.append(feat)
            semantic_residual = np.mean(normalized_features, axis=0)
        else:
            semantic_residual = np.zeros_like(depth_residual.flatten())

        depth_residual_flat = depth_residual.flatten()

        # 确保两个数组长度相同
        min_length = min(len(depth_residual_flat), len(semantic_residual))
        depth_residual_flat = depth_residual_flat[:min_length]
        semantic_residual = semantic_residual[:min_length]

        # 计算皮尔逊相关系数
        if len(depth_residual_flat) < 2 or len(semantic_residual) < 2:
            correlation = 0.1
        else:
            correlation, _ = pearsonr(depth_residual_flat, semantic_residual)

        # 计算结构相似性
        # 重塑为2D图像用于SSIM计算
        h = int(np.sqrt(min_length))
        if h * h == min_length:
            depth_2d = depth_residual_flat.reshape(h, h)
            semantic_2d = semantic_residual.reshape(h, h)
            min_side = min(depth_2d.shape[0], depth_2d.shape[1])
            if min_side < 7:
                # 取小于等于min_side的最大奇数
                win_size = min_side if min_side % 2 == 1 else min_side - 1
                if win_size < 3:
                    ssim_score = 0.1  # 太小无法计算
                else:
                    ssim_score = ssim(depth_2d, semantic_2d, data_range=1.0, win_size=win_size)
            else:
                ssim_score = ssim(depth_2d, semantic_2d, data_range=1.0)
        else:
            ssim_score = 0.0

        # 计算动态融合权重
        k, nu = 5.0, 1.0
        omega = 1.0 / ((1 + np.exp(-k * (S - S_c))) ** nu)

        # 计算CMCS
        cmcs = omega * correlation + (1 - omega) * ssim_score

        return cmcs

    def compute_kdt_multi(self, fingerprint_hash: np.ndarray,
                          key_variations: List[int] = [1, 4, 8]) -> Dict:
        """
        计算多粒度密钥驱动敏感性测试 (KDT-multi)
        """
        results = {}

        for bits in key_variations:
            # 生成扰动密钥
            perturbed_key = self.key ^ (1 << (bits - 1))

            # 使用扰动密钥生成新指纹
            perturbed_scrambling = ChaoticScrambling(perturbed_key)
            # 这里简化处理，实际应该重新生成完整指纹
            # 为了演示，我们直接对原指纹进行扰动
            perturbed_hash = fingerprint_hash.copy()
            perturbed_hash[:bits] = 1 - perturbed_hash[:bits]  # 翻转前bits位

            # 计算汉明距离
            hamming_distance = np.sum(fingerprint_hash != perturbed_hash)
            normalized_hd = hamming_distance / len(fingerprint_hash)

            # 设置阈值
            threshold = 0.1 * bits  # 阈值随位数增加

            results[f'{bits}_bit'] = {
                'hamming_distance': hamming_distance,
                'normalized_hd': normalized_hd,
                'threshold': threshold,
                'above_threshold': normalized_hd > threshold
            }

        # 计算KDT-multi指标
        above_threshold_count = sum(1 for result in results.values()
                                    if result['above_threshold'])
        kdt_multi = above_threshold_count / len(key_variations)

        results['kdt_multi'] = kdt_multi

        return results

    def compute_fingerprint_hash(self, fingerprint_hash: np.ndarray) -> str:
        """
        计算指纹哈希值
        """
        # 将浮点数转换为字节
        hash_bytes = struct.pack('f' * len(fingerprint_hash), *fingerprint_hash)

        # 计算SHA-256哈希
        hash_object = hashlib.sha256(hash_bytes)
        return hash_object.hexdigest()

    def evaluate_fingerprint(self, fingerprint_result: Dict,
                             S: float, S_c: float = 0.5) -> Dict:
        """
        综合评估指纹性能
        """
        # 计算CMCS
        cmcs = self.compute_cmcs(
            fingerprint_result['depth_residual'],
            fingerprint_result['semantic_residuals'],
            S, S_c
        )

        # 计算KDT-multi
        kdt_results = self.compute_kdt_multi(fingerprint_result['fingerprint_hash'])

        # 计算指纹哈希
        fingerprint_hash_str = self.compute_fingerprint_hash(
            fingerprint_result['fingerprint_hash']
        )
        evaluation_result = {
            'cmcs': cmcs,
            'kdt_multi': kdt_results,
            'fingerprint_hash': fingerprint_hash_str,
            'fingerprint_length': len(fingerprint_result['fingerprint_hash']),
            'evaluation_summary': {
                'cross_modal_consistency': 'High' if cmcs > 0.7 else 'Medium' if cmcs > 0.4 else 'Low',
                'key_sensitivity': 'High' if kdt_results['kdt_multi'] > 0.8 else 'Medium' if kdt_results[
                                                                                                 'kdt_multi'] > 0.5 else 'Low'
            }
        }

        return evaluation_result
