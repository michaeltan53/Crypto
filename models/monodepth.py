import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import Union, Tuple, Optional
import os
import urllib.request
import zipfile
from pathlib import Path
import torchvision.models as models

class ResnetEncoder(nn.Module):
    """ResNet编码器"""
    
    def __init__(self, num_layers, pretrained=True):
        super(ResnetEncoder, self).__init__()
        
        resnets = {
            18: torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained),
            34: torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=pretrained),
            50: torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=pretrained),
            101: torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=pretrained),
            152: torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=pretrained)
        }
        
        resnet = resnets[num_layers]
        
        # 提取各层特征
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu))
        self.encoder.append(resnet.layer1)
        self.encoder.append(resnet.layer2)
        self.encoder.append(resnet.layer3)
        self.encoder.append(resnet.layer4)
        
        # 冻结编码器权重
        if pretrained:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        features = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            features.append(x)
        return features

class DepthDecoder(nn.Module):
    """深度解码器"""
    
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()
        
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        
        # 解码器层
        self.convs = nn.ModuleDict()
        for i in range(4, -1, -1):
            # 上采样
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            
            # 跳跃连接
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
        
        # 输出层
        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
    
    def forward(self, input_features):
        self.outputs = {}
        
        # 解码器前向传播
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.convs[("dispconv", i)](x)
        
        return self.outputs

def upsample(x):
    """上采样"""
    return F.interpolate(x, scale_factor=2, mode="nearest", align_corners=None)

class ConvBlock(nn.Module):
    """卷积块"""
    
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class Conv3x3(nn.Module):
    """3x3卷积"""
    
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()
        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)
    
    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class SimpleMonodepth(nn.Module):
    """
    简化的Monodepth深度估计模型
    使用ResNet编码器和简单的解码器
    """
    
    def __init__(self, encoder_type='resnet18', pretrained=True):
        super(SimpleMonodepth, self).__init__()
        self.encoder_type = encoder_type
        
        # 构建编码器
        if encoder_type == 'resnet18':
            self.encoder = models.resnet18(pretrained=pretrained)
            # 移除最后的分类层
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        elif encoder_type == 'resnet50':
            self.encoder = models.resnet50(pretrained=pretrained)
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        
        # 构建解码器
        self.decoder = self._build_decoder()
        
        # 冻结编码器权重
        if pretrained:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def _build_decoder(self):
        """构建简单的解码器"""
        return nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # 输出深度值在[0,1]范围内
        )
    
    def forward(self, x):
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
        
        # 调整到标准尺寸 (256x256)
        if image_tensor.shape[2] != 256 or image_tensor.shape[3] != 256:
            image_tensor = F.interpolate(image_tensor, size=(256, 256), mode='bilinear', align_corners=False)
        
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

class MonodepthWrapper:
    """
    Monodepth模型包装器，提供便捷的接口
    """
    
    def __init__(self, device: str = 'cpu', encoder_type: str = 'resnet18'):
        self.device = device
        self.model = SimpleMonodepth(encoder_type=encoder_type, pretrained=True).to(device)
        
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