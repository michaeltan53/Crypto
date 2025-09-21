import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FingerprintAuthenticator(nn.Module):
    """
    轻量级双因子认证 MLP
    输入：指纹1、指纹2（如 Φ(S,k')、Φ(S,k)），输出：是否匹配（0/1）
    """

    def __init__(self, input_dim=256, hidden_dim=512, output_dim=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        # 更好的初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, fp1, fp2):
        # fp1, fp2: [B, D]
        x = torch.cat([fp1, fp2], dim=-1)
        logits = self.mlp(x)
        return torch.sigmoid(logits)


class PhaseConsistencyLoss(nn.Module):
    """
    相变感知损失 L_phase
    """

    def __init__(self, beta=0.5, S_c=0.14):
        super().__init__()
        self.beta = beta
        self.S_c = S_c

    def forward(self, omega, S):
        # omega: [B]，S: [B]
        target = torch.sigmoid(self.beta * (S - self.S_c))
        return F.mse_loss(omega, target)


class BayesianStrengthRegressor(nn.Module):
    """
    轻量贝叶斯扰动强度回推网络（L-BAC）
    输入：指纹，输出：扰动强度 S 及置信区间
    """

    def __init__(self, input_dim=256, latent_dim=32, dropout_p=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出扰动强度 S
        )
        self.dropout_p = dropout_p
        # 更好的初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, fp, mc_dropout=10):
        # fp: [B, D]
        preds = []
        self.train()  # 启用 dropout
        for _ in range(mc_dropout):
            z = self.encoder(fp)
            s_hat = self.decoder(z)
            # 约束输出在[0,1]范围内
            s_hat = torch.sigmoid(s_hat)
            preds.append(s_hat)
        preds = torch.stack(preds, dim=0)  # [mc, B, 1]
        mean = preds.mean(dim=0).squeeze(-1)  # [B]
        std = preds.std(dim=0).squeeze(-1)  # [B]
        return mean, std


# 物理先验正则项
class PhysicalPriorLoss(nn.Module):
    def __init__(self, beta=0.5, S_c=0.14):
        super().__init__()
        self.beta = beta
        self.S_c = S_c

    def forward(self, pred_S, phi_S):
        # phi_S: [B]，pred_S: [B]
        # 物理先验：在 S_c 附近抑制漂移
        penalty = torch.exp(-((pred_S - self.S_c) ** 2) / (2 * 0.01 ** 2))
        return (penalty * phi_S ** 2).mean()


# 添加一个简单的认证器，用于演示
class SimpleAuthenticator:
    """
    简单的基于距离的认证器，用于演示
    """

    def __init__(self, threshold=0.8, metric="cosine"):
        self.threshold = threshold
        self.metric = metric

    def authenticate(self, fp1, fp2):
        """
        基于余弦相似度的认证
        """
        # 计算余弦相似度
        # cos_sim = F.cosine_similarity(fp1, fp2, dim=-1)
        # # 转换为认证得分 (0-1)
        # auth_score = (cos_sim + 1) / 2
        # return auth_score
        if self.metric == "cosine":
            cos_sim = F.cosine_similarity(fp1, fp2, dim=-1)
            return (cos_sim > self.threshold).float()
        elif self.metric == "euclidean":
            dist = torch.norm(fp1 - fp2, dim=-1)
            return (dist < self.threshold).float()
