import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from typing import Union, List


class ClipViTWrapper(nn.Module):
    """
    CLIP-ViT模型包装器
    用于提取图像的多层语义特征
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 device: str = 'cpu'):
        super(ClipViTWrapper, self).__init__()
        self.device = device
        self.model_name = model_name

        print(f"正在加载CLIP模型: {model_name}...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        print("CLIP模型加载完成！")

    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """
        图像预处理
        Args:
            image: 输入图像 (numpy数组或PIL图像)
        Returns:
            预处理后的图像张量
        """
        if isinstance(image, np.ndarray):
            # 将numpy数组 [0,1] 转换为PIL图像
            image = Image.fromarray((image * 255).astype(np.uint8))

        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        return inputs['pixel_values'].to(self.device)

    def extract_features(self, image: Union[np.ndarray, Image.Image],
                         layers: List[int] = [4, 8, 12]) -> torch.Tensor:
        """
        提取并融合指定层的语义特征
        Args:
            image: 输入图像
            layers: 需要提取的ViT层索引
        Returns:
            融合后的特征向量
        """
        self.model.eval()
        image_tensor = self.preprocess_image(image)

        with torch.no_grad():
            outputs = self.model.vision_model(
                pixel_values=image_tensor,
                output_hidden_states=True
            )

            # hidden_states包含输入嵌入和所有Transformer层的输出
            # 第0层是输入嵌入，第1-12层是Transformer层输出
            hidden_states = outputs.hidden_states

            # 提取指定层的[CLS] token特征
            # hidden_states[i] -> [batch_size, sequence_length, hidden_size]
            # 我们取CLS token (第一个token)
            selected_features = [hidden_states[i][:, 0, :] for i in layers]

            # 融合特征（拼接）
            fused_features = torch.cat(selected_features, dim=-1)

        return fused_features.cpu()

    def calculate_semantic_residual(self, image_raw: np.ndarray,
                                    image_enc: np.ndarray,
                                    metric: str = 'l2') -> float:
        """
        计算两张图像之间的语义残差
        Args:
            image_raw: 原始图像
            image_enc: 加密图像
            metric: 度量方式 ('l2' 或 'cosine')
        Returns:
            语义残差
        """
        features_raw = self.extract_features(image_raw)
        features_enc = self.extract_features(image_enc)

        if metric == 'l2':
            # L2距离
            residual = torch.norm(features_raw - features_enc, p=2).item()
        elif metric == 'cosine':
            # 余弦差异 (1 - cosine_similarity)
            cos = nn.CosineSimilarity(dim=1)
            similarity = cos(features_raw, features_enc).item()
            residual = 1 - similarity
        else:
            raise ValueError(f"不支持的度量方式: {metric}")

        return residual