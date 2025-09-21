import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union
import math


class SimpleDiffusionModel(nn.Module):
    """
    简化的扩散模型
    用于条件扩散生成和嵌入式扩散扰动
    """
    
    def __init__(self, image_size: int = 256, channels: int = 3, 
                 time_embed_dim: int = 128, key_embed_dim: int = 64):
        super(SimpleDiffusionModel, self).__init__()
        self.image_size = image_size
        self.channels = channels
        self.time_embed_dim = time_embed_dim
        self.key_embed_dim = key_embed_dim
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # 密钥嵌入
        self.key_embed = nn.Sequential(
            nn.Linear(1, key_embed_dim),
            nn.ReLU(),
            nn.Linear(key_embed_dim, key_embed_dim)
        )
        
        # U-Net风格的编码器-解码器
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
        # 条件融合层
        self.condition_fusion = nn.Sequential(
            nn.Conv2d(channels + time_embed_dim + key_embed_dim, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
    
    def _build_encoder(self):
        """构建编码器"""
        layers = []
        in_channels = self.channels
        
        # 编码器层
        for out_channels in [64, 128, 256]:
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
        layers = []
        in_channels = 256
        
        # 解码器层
        for out_channels in [128, 64, self.channels]:
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels) if out_channels != self.channels else nn.Identity(),
                nn.ReLU(inplace=True) if out_channels != self.channels else nn.Identity()
            ])
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, 
                key: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入图像 [B, C, H, W]
            t: 时间步 [B, 1]
            key: 密钥 [B, 1]
        Returns:
            预测噪声 [B, C, H, W]
        """
        batch_size = x.shape[0]
        
        # 时间嵌入
        t_embed = self.time_embed(t)  # [B, time_embed_dim]
        t_embed = t_embed.unsqueeze(-1).unsqueeze(-1)  # [B, time_embed_dim, 1, 1]
        t_embed = t_embed.expand(-1, -1, x.shape[2], x.shape[3])  # [B, time_embed_dim, H, W]
        
        # 密钥嵌入
        key_embed = self.key_embed(key)  # [B, key_embed_dim]
        key_embed = key_embed.unsqueeze(-1).unsqueeze(-1)  # [B, key_embed_dim, 1, 1]
        key_embed = key_embed.expand(-1, -1, x.shape[2], x.shape[3])  # [B, key_embed_dim, H, W]
        
        # 条件融合
        condition_input = torch.cat([x, t_embed, key_embed], dim=1)
        condition_output = self.condition_fusion(condition_input)
        
        # 编码-解码
        features = self.encoder(condition_output)
        output = self.decoder(features)
        
        return output

class DiffusionPerturbation:
    """
    扩散模型扰动模块
    实现条件扩散生成和嵌入式扩散扰动
    """
    
    def __init__(self, image_size: int = 256, num_timesteps: int = 1000, 
                 beta_start: float = 1e-4, beta_end: float = 0.02):
        """
        初始化扩散扰动
        Args:
            image_size: 图像尺寸
            num_timesteps: 时间步数
            beta_start: 初始噪声调度参数
            beta_end: 最终噪声调度参数
        """
        self.image_size = image_size
        self.num_timesteps = num_timesteps
        
        # 噪声调度
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 扩散模型
        self.model = SimpleDiffusionModel(image_size=image_size)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, 
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向扩散过程 q(x_t | x_0)
        Args:
            x_start: 初始图像
            t: 时间步
            noise: 可选噪声
        Returns:
            加噪后的图像
        """
        if t[0] == 0:
            return x_start
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, 
                 key: torch.Tensor) -> torch.Tensor:
        """
        反向去噪过程 p(x_{t-1} | x_t)
        Args:
            x_t: 当前时间步的图像
            t: 时间步
            key: 密钥
        Returns:
            去噪后的图像
        """
        with torch.no_grad():
            # 预测噪声
            predicted_noise = self.model(x_t, t.float().unsqueeze(-1), key.float().unsqueeze(-1))
            
            # 计算去噪参数
            alpha_t = self.alphas[t].view(-1, 1, 1, 1)
            alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            beta_t = self.betas[t].view(-1, 1, 1, 1)
            
            # 去噪公式
            x_prev = (1 / alpha_t.sqrt()) * (x_t - (beta_t / (1 - alpha_cumprod_t).sqrt()) * predicted_noise)
            
            if t[0] > 0:
                noise = torch.randn_like(x_t)
                x_prev = x_prev + (beta_t.sqrt() * noise)
            
            return x_prev
    
    def conditional_diffusion(self, x_start: torch.Tensor, strength: float, 
                            key: int, num_steps: int = 50) -> torch.Tensor:
        """
        条件扩散生成
        Args:
            x_start: 初始图像
            strength: 扰动强度 [0, 1]
            key: 密钥
            num_steps: 扩散步数
        Returns:
            扰动后的图像
        """
        device = x_start.device
        batch_size = x_start.shape[0]
        
        # 计算起始时间步
        start_t = int(strength * self.num_timesteps)
        # start_t = int((1 - strength) * self.num_timesteps)
        start_t = max(0, min(start_t, self.num_timesteps - 1))
        
        # 前向加噪
        t = torch.tensor([start_t], device=device).repeat(batch_size)
        x_t = self.q_sample(x_start, t)
        
        # 反向去噪
        key_tensor = torch.tensor([key], device=device).repeat(batch_size)
        
        for i in range(num_steps):
            t = torch.tensor([start_t - i], device=device).repeat(batch_size)
            t = torch.clamp(t, 0, self.num_timesteps - 1)
            x_t = self.p_sample(x_t, t, key_tensor)
        
        return x_t
    
    def embedded_diffusion_perturb(self, image: np.ndarray, strength: float, 
                                  key: int) -> np.ndarray:
        """
        嵌入式扩散扰动
        Args:
            image: 输入图像 [H, W, C]
            strength: 扰动强度 [0, 1]
            key: 密钥
        Returns:
            扰动后的图像
        """
        # 预处理
        # if image.max() > 1.0:
        #     image = image.astype(np.float32) / 255.0
        image = image.astype(np.float32) / 255.0
        
        # 转换为张量
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        # 条件扩散
        perturbed_tensor = self.conditional_diffusion(image_tensor, strength, key)
        
        # 转换为numpy
        perturbed_image = perturbed_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        
        # 确保输出在[0, 1]范围内
        perturbed_image = np.clip(perturbed_image, 0, 1)
        
        return perturbed_image
    
    def train_step(self, x_start: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        训练步骤
        Args:
            x_start: 初始图像
            key: 密钥
        Returns:
            损失值
        """
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # 随机时间步
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        
        # 随机噪声
        noise = torch.randn_like(x_start)
        
        # 前向扩散
        x_t = self.q_sample(x_start, t, noise)
        
        # 预测噪声
        predicted_noise = self.model(x_t, t.float().unsqueeze(-1), key.float().unsqueeze(-1))
        
        # 计算损失
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss 