#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三阶段扰动响应模型测试脚本
验证论文3.1节描述的几何残差ΔP(S)和语义残差Δf(S)的三阶段分段演化特性

作者：基于论文实现
日期：2024
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# 导入自定义模块
from models.monodepth import MonodepthWrapper
from models.clip_vit_new import ClipViTWrapper
from models.dce_module import DCEController
from utils.metrics import ThreeStageResponseModel, ModalDominanceSwitching


class ThreeStageResponseTest:
    """
    三阶段扰动响应测试
    """
    
    def __init__(self, image_size: int = 256, device: str = 'cpu', key: int = 42):
        self.image_size = image_size
        self.device = device
        self.key = key
        
        print("正在初始化三阶段响应测试组件...")
        
        # 初始化基础模型
        self.monodepth = MonodepthWrapper(device=device)
        self.clip_vit = ClipViTWrapper(device=device)
        self.dce_controller = DCEController(base_key=key)
        
        # 初始化三阶段响应模型和模态切换机制
        self.three_stage_model = ThreeStageResponseModel()
        self.modal_switching = ModalDominanceSwitching()
        
        # 创建DCE模块
        self.dce_module = self.dce_controller.create_dce_module(
            'three_stage_test', image_size=image_size
        )
        
        print("三阶段响应测试组件初始化完成！")
    
    def _generate_test_image(self) -> np.ndarray:
        """生成测试图像"""
        image = np.full((self.image_size, self.image_size, 3), (0.8, 0.9, 1.0), dtype=np.float32)
        
        # 添加几何结构
        cv2.rectangle(image, (50, 100), (150, 200), (0.7, 0.3, 0.1), -1)  # 建筑物
        cv2.circle(image, (200, 150), 40, (0.1, 0.7, 0.3), -1)  # 圆形物体
        cv2.line(image, (20, 50), (180, 80), (0.3, 0.1, 0.7), 3)  # 线条
        
        # 添加纹理
        noise = np.random.normal(0, 0.05, (self.image_size, self.image_size, 3))
        image = np.clip(image + noise, 0, 1)
        
        return image
    
    def test_geometric_residual(self, image: np.ndarray, num_samples: int = 50) -> dict:
        """测试几何残差ΔP(S)的三阶段响应"""
        print("\n=== 测试几何残差ΔP(S)响应 ===")
        
        strengths = np.linspace(0, 1, num_samples)
        geometric_residuals = []
        
        # 计算原始深度图
        depth_orig = self.monodepth.estimate_depth(image)
        
        print("计算不同扰动强度下的几何残差...")
        for strength in tqdm(strengths, desc="几何残差分析"):
            # 生成扰动图像
            perturbed_image = self.dce_module.encrypt_image(image, strength, 'hybrid')
            
            # 计算几何残差 ΔP(S) = (D_enc - D_raw)^2
            depth_pert = self.monodepth.estimate_depth(perturbed_image)
            geometric_residual = (depth_pert - depth_orig) ** 2
            geometric_residuals.append(np.mean(geometric_residual))
        
        return {
            'strengths': strengths,
            'geometric_residuals': np.array(geometric_residuals)
        }
    
    def test_semantic_residual(self, image: np.ndarray, num_samples: int = 50) -> dict:
        """测试语义残差Δf(S)的饱和响应"""
        print("\n=== 测试语义残差Δf(S)响应 ===")
        
        strengths = np.linspace(0, 1, num_samples)
        semantic_residuals = []
        
        print("计算不同扰动强度下的语义残差...")
        for strength in tqdm(strengths, desc="语义残差分析"):
            # 生成扰动图像
            perturbed_image = self.dce_module.encrypt_image(image, strength, 'hybrid')
            
            # 计算语义残差 Δf(S) - 多层融合
            semantic_residuals_dict = {}
            for layer in [4, 8, 12]:
                features_orig_layer = self.clip_vit.extract_features(image, layer=layer)
                features_pert_layer = self.clip_vit.extract_features(perturbed_image, layer=layer)
                # 计算欧式距离
                semantic_residual = np.linalg.norm(features_pert_layer - features_orig_layer, ord=2)
                semantic_residuals_dict[f'layer_{layer}'] = semantic_residual
            
            # 融合多层语义残差
            fused_semantic = self._fuse_semantic_residuals(semantic_residuals_dict)
            semantic_residuals.append(fused_semantic)
        
        return {
            'strengths': strengths,
            'semantic_residuals': np.array(semantic_residuals)
        }
    
    def _fuse_semantic_residuals(self, semantic_residuals: dict) -> float:
        """融合多层语义残差信号"""
        residuals = []
        for layer_name in ['layer_4', 'layer_8', 'layer_12']:
            if layer_name in semantic_residuals:
                residuals.append(semantic_residuals[layer_name])
        
        if not residuals:
            return 0.0
        
        # 简单的平均融合
        return np.mean(residuals)
    
    def test_three_stage_fitting(self, geometric_data: dict, semantic_data: dict) -> dict:
        """测试三阶段模型拟合"""
        print("\n=== 测试三阶段模型拟合 ===")
        
        strengths = geometric_data['strengths']
        geometric_residuals = geometric_data['geometric_residuals']
        semantic_residuals = semantic_data['semantic_residuals']
        
        # 拟合三阶段响应模型
        fit_params = self.three_stage_model.fit_three_stage_model(
            strengths, geometric_residuals, semantic_residuals
        )
        
        # 统计显著性检验
        significance_results = self.three_stage_model.statistical_significance_test(
            strengths, geometric_residuals, semantic_residuals
        )
        
        # 计算模态融合响应
        fusion_response, lambda_f = self.modal_switching.residual_fusion_response(
            strengths, geometric_residuals, semantic_residuals
        )
        
        return {
            'fit_params': fit_params,
            'significance_results': significance_results,
            'fusion_response': fusion_response,
            'lambda_f': lambda_f
        }
    
    def visualize_test_results(self, geometric_data: dict, semantic_data: dict, 
                              fitting_results: dict, save_dir: str = "three_stage_test_results"):
        """可视化测试结果"""
        print("\n=== 可视化测试结果 ===")
        os.makedirs(save_dir, exist_ok=True)
        
        strengths = geometric_data['strengths']
        geometric_residuals = geometric_data['geometric_residuals']
        semantic_residuals = semantic_data['semantic_residuals']
        
        # 绘制三阶段响应曲线
        three_stage_plot_path = os.path.join(save_dir, "three_stage_test_response.png")
        self.three_stage_model.plot_three_stage_response(
            strengths, geometric_residuals, semantic_residuals, 
            save_path=three_stage_plot_path
        )
        print(f"三阶段响应测试图已保存到: {three_stage_plot_path}")
        
        # 绘制模态融合响应
        if 'fusion_response' in fitting_results:
            self._plot_modal_fusion_test(fitting_results, save_dir)
        
        # 保存数值结果
        self._save_test_results(geometric_data, semantic_data, fitting_results, save_dir)
    
    def _plot_modal_fusion_test(self, fitting_results: dict, save_dir: str):
        """绘制模态融合测试结果"""
        fusion_response = fitting_results['fusion_response']
        lambda_f = fitting_results['lambda_f']
        
        # 这里需要从之前的测试中获取strengths
        strengths = np.linspace(0, 1, len(fusion_response))
        
        plt.figure(figsize=(12, 8))
        
        # 子图1: 融合权重函数
        plt.subplot(2, 2, 1)
        plt.plot(strengths, lambda_f, 'g-^', label='λ_f(S)', linewidth=2)
        plt.axvline(x=0.637, color='purple', linestyle='--', alpha=0.5, label='S_c=0.637')
        plt.xlabel('扰动强度 S')
        plt.ylabel('融合权重 λ_f(S)')
        plt.title('动态融合权重函数测试')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: 融合响应函数
        plt.subplot(2, 2, 2)
        plt.plot(strengths, fusion_response, 'm-d', label='R(S)', linewidth=2)
        plt.axvline(x=0.637, color='purple', linestyle='--', alpha=0.5, label='S_c=0.637')
        plt.xlabel('扰动强度 S')
        plt.ylabel('融合响应 R(S)')
        plt.title('残差融合响应函数测试')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        fusion_test_plot_path = os.path.join(save_dir, "modal_fusion_test.png")
        plt.savefig(fusion_test_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"模态融合测试图已保存到: {fusion_test_plot_path}")
    
    def _save_test_results(self, geometric_data: dict, semantic_data: dict, 
                          fitting_results: dict, save_dir: str):
        """保存测试结果"""
        import json
        
        results = {
            'geometric_data': {
                'strengths': geometric_data['strengths'].tolist(),
                'residuals': geometric_data['geometric_residuals'].tolist()
            },
            'semantic_data': {
                'strengths': semantic_data['strengths'].tolist(),
                'residuals': semantic_data['semantic_residuals'].tolist()
            },
            'fitting_results': {
                'fit_params': fitting_results.get('fit_params', {}),
                'significance_results': fitting_results.get('significance_results', {}),
                'fusion_response': fitting_results.get('fusion_response', []).tolist(),
                'lambda_f': fitting_results.get('lambda_f', []).tolist()
            }
        }
        
        results_path = os.path.join(save_dir, "three_stage_test_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"测试结果已保存到: {results_path}")
    
    def run_complete_test(self, num_samples: int = 50):
        """运行完整的三阶段响应测试"""
        print("=" * 60)
        print("开始三阶段扰动响应测试")
        print("=" * 60)
        
        # 生成测试图像
        image = self._generate_test_image()
        
        # 测试几何残差响应
        geometric_data = self.test_geometric_residual(image, num_samples)
        
        # 测试语义残差响应
        semantic_data = self.test_semantic_residual(image, num_samples)
        
        # 测试三阶段模型拟合
        fitting_results = self.test_three_stage_fitting(geometric_data, semantic_data)
        
        # 可视化结果
        self.visualize_test_results(geometric_data, semantic_data, fitting_results)
        
        # 打印关键结果
        print("\n" + "=" * 60)
        print("三阶段响应测试结果总结")
        print("=" * 60)
        
        if fitting_results.get('fit_params'):
            geo_params = fitting_results['fit_params'].get('geometric', {})
            sem_params = fitting_results['fit_params'].get('semantic', {})
            
            print(f"几何残差拟合优度 R²: {geo_params.get('r2', 0):.4f}")
            print(f"语义残差拟合优度 R²: {sem_params.get('r2', 0):.4f}")
        
        if fitting_results.get('significance_results'):
            print("\n统计显著性检验结果:")
            for test_name, result in fitting_results['significance_results'].items():
                p_value = result.get('p_value', 1.0)
                significance = "显著" if p_value < 0.01 else "不显著"
                print(f"  {test_name}: p={p_value:.4f} ({significance})")
        
        print("\n三阶段分界点:")
        print(f"  S₁ = 0.15 (高敏感阶段)")
        print(f"  S₂ = 0.60 (线性鲁棒阶段)")
        print(f"  S_c = 0.637 (模态切换点)")
        
        print("\n测试完成！")


def main():
    """主函数"""
    # 创建测试实例
    test = ThreeStageResponseTest(image_size=256, device='cpu', key=42)
    
    # 运行完整测试
    test.run_complete_test(num_samples=30)


if __name__ == "__main__":
    main() 