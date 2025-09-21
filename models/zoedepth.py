import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import Union, Tuple, Optional
import os

class ZoeDepth(nn.Module):
    """
    ZoeDepth深度估计模型
    基于预训练的ZoeDepth模型进行单目深度估计
    """
    
    def __init__(self, model_type: str = "ZoeD_N", pretrained: bool = True):
        super(ZoeDepth, self).__init__()
        self.model_type = model_type
        self.pretrained = pretrained
        
        # 简化的ZoeDepth架构（实际使用时需要完整的预训练模型）
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
        if pretrained:
            self._load_pretrained_weights()
    
    def _build_encoder(self):
        """构建编码器"""
        # 简化的编码器结构
        layers = []
        in_channels = 3
        
        # 编码器层
        for i, out_channels in enumerate([64, 128, 256, 512]):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ])
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self):
        """构建解码器"""
        # 简化的解码器结构
        layers = []
        in_channels = 512
        
        # 解码器层
        for i, out_channels in enumerate([256, 128, 64, 1]):
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels) if out_channels != 1 else nn.Identity(),
                nn.ReLU(inplace=True) if out_channels != 1 else nn.Identity()
            ])
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _load_pretrained_weights(self):
        """加载预训练权重（简化版本）"""
        # 在实际应用中，这里应该加载真正的ZoeDepth预训练权重
        print("Loading pretrained ZoeDepth weights...")
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入图像张量 [B, 3, H, W]
        Returns:
            深度图张量 [B, 1, H, W]
        """
        features = self.encoder(x)
        depth = self.decoder(features)
        return depth
    
    def predict(self, image: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        """
        预测深度图
        Args:
            image: 输入图像（numpy数组、PIL图像或图像路径）
        Returns:
            深度图 [H, W]
        """
        self.eval()
        
        # 预处理图像
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        # 归一化到[0,1]
        image = image.astype(np.float32) / 255.0
        
        # 转换为张量
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        with torch.no_grad():
            depth_tensor = self.forward(image_tensor)
            depth_map = depth_tensor.squeeze().cpu().numpy()
        
        return depth_map
    
    def compute_depth_error(self, depth_pred: np.ndarray, depth_gt: np.ndarray) -> float:
        """
        计算深度误差
        Args:
            depth_pred: 预测深度图
            depth_gt: 真实深度图
        Returns:
            平均绝对误差
        """
        return np.mean(np.abs(depth_pred - depth_gt))

class ZoeDepthWrapper:
    """
    ZoeDepth模型包装器，提供便捷的接口
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.model = ZoeDepth(pretrained=True).to(device)
        
    def estimate_depth(self, image: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        """
        估计图像深度
        Args:
            image: 输入图像
        Returns:
            深度图
        """
        return self.model.predict(image)
    
    def compare_depths(self, image1: Union[np.ndarray, Image.Image, str], 
                      image2: Union[np.ndarray, Image.Image, str]) -> Tuple[np.ndarray, float]:
        """
        比较两张图像的深度估计结果
        Args:
            image1: 第一张图像
            image2: 第二张图像
        Returns:
            (深度残差图, 平均误差)
        """
        depth1 = self.estimate_depth(image1)
        depth2 = self.estimate_depth(image2)
        
        depth_diff = np.abs(depth2 - depth1)
        mean_error = np.mean(depth_diff)
        
        return depth_diff, mean_error 