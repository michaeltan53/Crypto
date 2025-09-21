#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示脚本：展示DCE模块和相变模型的使用方法
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
import torch

# 导入自定义模块
from models.monodepth import MonodepthWrapper
from models.dce_module import DCEController
from models.clip_vit_new import ClipViTWrapper
from utils.metrics import DualModalAnalyzer, PhaseTransitionModel
from utils.wavelet_chaos import WaveletChaosPerturbation
from utils.diffusion_perturb import DiffusionPerturbation


def demo_basic_perturbation():
    """演示基本扰动功能"""
    print("=== 基本扰动演示 ===")

    # 创建示例图像
    image = np.random.random((256, 256, 3)).astype(np.float32)

    # 初始化DCE模块
    dce_controller = DCEController(base_key=42)
    dce_module = dce_controller.create_dce_module('demo', image_size=256)

    # 应用不同强度的扰动
    strengths = [0.0, 0.25, 0.5, 0.75, 1.0]

    fig, axes = plt.subplots(2, len(strengths), figsize=(15, 6))

    for i, strength in enumerate(strengths):
        # 应用扰动
        perturbed = dce_module.encrypt_image(image, strength, 'hybrid')

        # 显示结果
        axes[0, i].imshow(image)
        axes[0, i].set_title(f'原始图像')
        axes[0, i].axis('off')

        axes[1, i].imshow(perturbed)
        axes[1, i].set_title(f'扰动强度 {strength}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('demo_basic_perturbation.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("基本扰动演示完成！")


def demo_wavelet_chaos():
    """演示小波混沌扰动"""
    print("\n=== 小波混沌扰动演示 ===")

    # 创建示例图像
    image = np.random.random((256, 256, 3)).astype(np.float32)

    # 初始化小波混沌扰动
    wavelet_chaos = WaveletChaosPerturbation(wavelet='db4', key=42)

    # 可视化小波子带
    bands = wavelet_chaos.visualize_wavelet_bands(image)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(bands['LL'])
    axes[0, 0].set_title('LL (低频)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(bands['LH'])
    axes[0, 1].set_title('LH (水平高频)')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(bands['HL'])
    axes[1, 0].set_title('HL (垂直高频)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(bands['HH'])
    axes[1, 1].set_title('HH (对角高频)')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('demo_wavelet_bands.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 应用不同强度的混沌扰动
    strengths = [0.0, 0.3, 0.6, 0.9]

    fig, axes = plt.subplots(2, len(strengths), figsize=(12, 6))

    for i, strength in enumerate(strengths):
        perturbed = wavelet_chaos.perturb_image(image, strength, 42)

        axes[0, i].imshow(image)
        axes[0, i].set_title(f'原始图像')
        axes[0, i].axis('off')

        axes[1, i].imshow(perturbed)
        axes[1, i].set_title(f'混沌扰动 {strength}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('demo_wavelet_chaos.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("小波混沌扰动演示完成！")


def demo_diffusion_perturbation():
    """演示扩散模型扰动"""
    print("\n=== 扩散模型扰动演示 ===")

    # 创建示例图像
    image = np.random.random((256, 256, 3)).astype(np.float32)

    # 初始化扩散扰动
    diffusion_perturb = DiffusionPerturbation(image_size=256)

    # 应用不同强度的扩散扰动
    strengths = [0.0, 0.3, 0.6, 0.9]

    fig, axes = plt.subplots(2, len(strengths), figsize=(12, 6))

    for i, strength in enumerate(strengths):
        perturbed = diffusion_perturb.embedded_diffusion_perturb(image, strength, 42)

        axes[0, i].imshow(image)
        axes[0, i].set_title(f'原始图像')
        axes[0, i].axis('off')

        axes[1, i].imshow(perturbed)
        axes[1, i].set_title(f'扩散扰动 {strength}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('demo_diffusion_perturbation.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("扩散模型扰动演示完成！")


def demo_phase_transition_model():
    """演示相变模型拟合"""
    print("\n=== 相变模型拟合演示 ===")

    # 生成模拟数据
    np.random.seed(42)
    S = np.linspace(0, 1, 50)

    # 模拟双阶段响应
    S_th = 0.5
    k1 = 0.1
    k2 = 0.01
    beta = 2.0

    # 生成理论值
    delta_P_theory = np.where(S <= S_th,
                              k1 * S,
                              k2 * np.exp(beta * S))

    # 添加噪声
    noise = np.random.normal(0, 0.01, len(S))
    delta_P_noisy = delta_P_theory + noise

    # 拟合相变模型
    phase_model = PhaseTransitionModel()
    fit_result = phase_model.fit_model(S, delta_P_noisy)

    if fit_result is not None:
        print(f"拟合优度 R²: {fit_result['r_squared']:.4f}")
        print(f"相变阈值 S_th: {fit_result['params']['S_th']:.4f}")
        print(f"线性阶段斜率 k₁: {fit_result['params']['k1']:.4f}")
        print(f"指数增长率 β: {fit_result['params']['beta']:.4f}")

        # 可视化拟合结果
        phase_model.plot_fit(S, delta_P_noisy, save_path='demo_phase_transition.png')

    print("相变模型拟合演示完成！")


def demo_depth_estimation():
    """演示深度估计"""
    print("\n=== 深度估计演示 ===")

    # 创建包含几何结构的示例图像
    image = np.zeros((256, 256, 3), dtype=np.float32)

    # 添加一些几何形状
    cv2.rectangle(image, (50, 50), (150, 150), (0.7, 0.3, 0.1), -1)
    cv2.circle(image, (200, 100), 50, (0.1, 0.7, 0.3), -1)
    pts = np.array([[300, 50], [350, 150], [250, 150]], np.int32)
    cv2.fillPoly(image, [pts], (0.3, 0.1, 0.7))

    # 添加纹理
    noise = np.random.normal(0, 0.1, (256, 256, 3))
    image = np.clip(image + noise, 0, 1)

    # 初始化深度估计模型
    monodepth = MonodepthWrapper(device='cpu')

    # 估计深度
    depth_map = monodepth.estimate_depth(image)

    # 可视化结果
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(image)
    axes[0].set_title('原始图像')
    axes[0].axis('off')

    im = axes[1].imshow(depth_map, cmap='viridis')
    axes[1].set_title('深度图')
    axes[1].axis('off')

    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    plt.savefig('demo_depth_estimation.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("深度估计演示完成！")


def demo_dual_modal():
    """演示双模态（深度+语义）感知退化分析"""
    print("\n=== 双模态（深度+语义）感知退化分析演示 ===")

    # 创建示例图像
    image = np.zeros((256, 256, 3), dtype=np.float32)
    cv2.rectangle(image, (50, 50), (150, 150), (0.7, 0.3, 0.1), -1)
    cv2.circle(image, (200, 100), 50, (0.1, 0.7, 0.3), -1)
    noise = np.random.normal(0, 0.1, (256, 256, 3))
    image = np.clip(image + noise, 0, 1)

    # 初始化组件
    monodepth = MonodepthWrapper(device='cpu')
    clip_vit = ClipViTWrapper(device='cpu')
    dce_controller = DCEController(base_key=42)
    dce_module = dce_controller.create_dce_module('demo', image_size=256)
    analyzer = DualModalAnalyzer()

    # 生成扰动强度序列
    strengths = np.linspace(0, 1, 10)
    depth_errors = []
    semantic_errors = []

    # 计算原始深度和语义特征
    depth_orig = monodepth.estimate_depth(image)
    features_orig = clip_vit.extract_features(image)

    print("运行双模态扰动强度实验...")
    for strength in strengths:
        # 应用扰动
        perturbed = dce_module.encrypt_image(image, strength, 'hybrid')

        # 深度估计
        depth_pert = monodepth.estimate_depth(perturbed)
        depth_error = np.mean(np.abs(depth_pert - depth_orig))
        depth_errors.append(depth_error)

        # 语义特征
        features_pert = clip_vit.extract_features(perturbed)
        semantic_error = torch.norm(features_orig - features_pert, p=2).item()
        semantic_errors.append(semantic_error)

        print(f"强度 {strength:.2f}: 深度误差 {depth_error:.6f}，语义误差 {semantic_error:.6f}")

    # 分析结果
    analysis_result = analyzer.analyze_dual_modal_response(
        np.array(strengths), np.array(depth_errors), np.array(semantic_errors)
    )

    if analysis_result is not None:
        print("\n分析结果:")
        if analysis_result['crossover_point'] is not None:
            print(f"主导模态切换点 S_c: {analysis_result['crossover_point']:.4f}")
        else:
            print("未找到主导模态切换点")
        # 可视化结果
        analyzer.plot_dual_modal_response(save_path='demo_dual_modal_response.png')

    print("双模态感知退化分析演示完成！")


def main():
    """主函数"""
    print("DCE模块和相变模型演示")
    print("=" * 50)

    # 运行各个演示
    demo_basic_perturbation()
    demo_wavelet_chaos()
    demo_diffusion_perturbation()
    demo_phase_transition_model()
    demo_depth_estimation()
    demo_dual_modal()

    print("\n" + "=" * 50)
    print("所有演示完成！")
    print("结果图像已保存到当前目录")


if __name__ == "__main__":
    main()
