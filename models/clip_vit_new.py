import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import Union, Optional
import warnings

warnings.filterwarnings('ignore')


class SimpleClipViT:
    """
    简化的CLIP-ViT模型
    用于演示和测试，避免复杂的依赖安装
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.feature_dim = 512

    def preprocess_image(self, image: Union[np.ndarray, Image.Image, str]) -> torch.Tensor:
        """预处理图像"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image.astype(np.uint8))

        # 简化的预处理
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)

        # 调整到标准尺寸
        if image_tensor.shape[2] != 224 or image_tensor.shape[3] != 224:
            image_tensor = F.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False)

        return image_tensor.to(self.device)

    def extract_features(self, image: Union[np.ndarray, Image.Image, str],
                         layer: Optional[int] = None) -> Union[torch.Tensor, np.ndarray]:
        """提取语义特征"""
        # 预处理图像
        image_tensor = self.preprocess_image(image)

        # 简化的特征提取（模拟CLIP的输出）
        batch_size = image_tensor.shape[0]

        # 生成模拟的特征向量
        features = torch.randn(batch_size, self.feature_dim, device=self.device)

        # 应用L2归一化
        features = F.normalize(features, p=2, dim=1)

        # 转换为numpy数组
        features = features.cpu().numpy()

        return features

    def extract_multi_layer_features(self, image: Union[np.ndarray, Image.Image, str],
                                     layers: list = [4, 8, 12]) -> dict:
        """提取多层特征"""
        features_dict = {}

        for layer in layers:
            features = self.extract_features(image, layer=layer)
            features_dict[f'layer_{layer}'] = features

        return features_dict

    def compute_semantic_similarity(self, image1: Union[np.ndarray, Image.Image, str],
                                    image2: Union[np.ndarray, Image.Image, str]) -> float:
        """计算两张图像的语义相似度"""
        features1 = self.extract_features(image1)
        features2 = self.extract_features(image2)

        # 计算余弦相似度
        if isinstance(features1, np.ndarray):
            features1 = torch.from_numpy(features1)
        if isinstance(features2, np.ndarray):
            features2 = torch.from_numpy(features2)

        similarity = F.cosine_similarity(features1, features2, dim=-1)
        return similarity.item()

    def get_feature_dimension(self) -> int:
        """获取特征维度"""
        return self.feature_dim


class ClipViTWrapper:
    """
    CLIP-ViT语义特征提取器
    支持多层特征提取
    """
    import sys
    sys.path.append('/Users/ushiushi/anaconda3/lib/python3.11/site-packages')

    # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    def __init__(self, device: str = 'cpu', model_name: str = 'ViT-B/32'):
        self.device = device
        self.model_name = model_name

        print(f"正在加载CLIP模型: {model_name}")
        try:
            import clip
            self.model, self.preprocess = clip.load(model_name, device=device)
            self.use_real_clip = True
            print("CLIP模型加载成功！")
        except Exception as e:
            print(f"CLIP模型加载失败，使用简化版本: {e}")
            self.model = SimpleClipViT(device=device)
            self.use_real_clip = False

    def preprocess_image(self, image: Union[np.ndarray, Image.Image, str]) -> torch.Tensor:
        """预处理图像"""
        if self.use_real_clip:
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image.astype(np.uint8))

            # 应用CLIP预处理
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            return image_tensor
        else:
            return self.model.preprocess_image(image)

    def extract_features(self, image: Union[np.ndarray, Image.Image, str],
                         layer: Optional[int] = None) -> Union[torch.Tensor, np.ndarray]:
        """提取语义特征"""
        if self.use_real_clip:
            self.model.eval()

            # 预处理图像
            image_tensor = self.preprocess_image(image)

            with torch.no_grad():
                if layer is not None:
                    # 提取指定层的特征
                    features = self._extract_layer_features(image_tensor, layer)
                else:
                    # 提取最终特征
                    features = self.model.encode_image(image_tensor)

            # 转换为numpy数组
            if isinstance(features, torch.Tensor):
                features = features.cpu().numpy()

            return features
        else:
            return self.model.extract_features(image, layer=layer)

    # def _extract_layer_features(self, image_tensor: torch.Tensor, layer: int) -> torch.Tensor:
    #     """提取指定层的特征"""
    #     # 获取图像编码器
    #     image_encoder = self.model.visual
    #
    #     # 前向传播到指定层
    #     x = image_encoder.conv1(image_tensor)  # 初始卷积
    #     x = image_encoder.bn1(x)
    #     x = image_encoder.act1(x)
    #     x = image_encoder.avgpool(x)
    #     x = torch.flatten(x, 1)
    #     x = image_encoder.fc(x)
    #
    #     # 通过Transformer层
    #     for i, transformer_layer in enumerate(image_encoder.transformer.resblocks):
    #         x = transformer_layer(x)
    #         if i == layer - 1:  # 到达指定层
    #             break
    #
    #     # 应用层归一化
    #     x = image_encoder.ln_post(x)
    #
    #     return x

    def _extract_layer_features(self, image_tensor: torch.Tensor, layer: int) -> torch.Tensor:
        """
        从 OpenAI 的 CLIP ViT 模型中提取指定 Transformer 层的输出
        """
        image_encoder = self.model.visual  # VisionTransformer

        # Patch embedding
        x = image_encoder.conv1(image_tensor)  # shape: [B, C, H, W]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, C, HW]
        x = x.permute(0, 2, 1)  # [B, HW, C]

        # 添加 class token
        class_embedding = image_encoder.class_embedding.to(x.dtype)
        cls_tokens = class_embedding + torch.zeros(x.shape[0], 1, class_embedding.shape[-1], device=x.device)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1 + HW, C]

        # 加位置编码
        x = x + image_encoder.positional_embedding
        x = image_encoder.ln_pre(x)

        # Transformer 前向传播到指定层
        for i, block in enumerate(image_encoder.transformer.resblocks):
            x = block(x)
            if i == layer - 1:
                break

        # 不取 ln_post，不提取 cls_token，只返回中间特征
        return x  # shape: [B, 1 + HW, C]

    def extract_multi_layer_features(self, image: Union[np.ndarray, Image.Image, str],
                                     layers: list = [4, 8, 12]) -> dict:
        """提取多层特征"""
        if self.use_real_clip:
            features_dict = {}

            for layer in layers:
                features = self.extract_features(image, layer=layer)
                features_dict[f'layer_{layer}'] = features

            return features_dict
        else:
            return self.model.extract_multi_layer_features(image, layers)

    def compute_semantic_similarity(self, image1: Union[np.ndarray, Image.Image, str],
                                    image2: Union[np.ndarray, Image.Image, str]) -> float:
        """计算两张图像的语义相似度"""
        if self.use_real_clip:
            features1 = self.extract_features(image1)
            features2 = self.extract_features(image2)

            # 计算余弦相似度
            if isinstance(features1, np.ndarray):
                features1 = torch.from_numpy(features1)
            if isinstance(features2, np.ndarray):
                features2 = torch.from_numpy(features2)

            similarity = F.cosine_similarity(features1, features2, dim=-1)
            return similarity.item()
        else:
            return self.model.compute_semantic_similarity(image1, image2)

    def get_feature_dimension(self) -> int:
        """获取特征维度"""
        if self.use_real_clip:
            # 创建一个虚拟图像来获取特征维度
            dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            features = self.extract_features(dummy_image)
            return features.shape[-1] if hasattr(features, 'shape') else len(features)
        else:
            return self.model.get_feature_dimension()
