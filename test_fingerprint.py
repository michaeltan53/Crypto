#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉指纹系统测试脚本
验证3.2节核心功能

作者：基于论文实现
日期：2024
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """测试模块导入"""
    print("测试视觉指纹模块导入...")
    
    try:
        from models.visual_fingerprint import (
            AdaptiveResidualModeling, ChaoticScrambling, DynamicQuantization,
            LightweightEncoder, VisualFingerprintGenerator, FingerprintEvaluator
        )
        from models.monodepth import MonodepthWrapper
        from models.clip_vit_new import ClipViTWrapper
        print("✓ 所有视觉指纹模块导入成功")
        return True
    except Exception as e:
        print(f"✗ 视觉指纹模块导入失败: {e}")
        return False

def test_adaptive_residual_modeling():
    """测试自适应残差信号建模"""
    print("\n测试自适应残差信号建模...")
    
    try:
        from models.visual_fingerprint import AdaptiveResidualModeling
        
        # 创建测试数据
        depth_residual = np.random.random((64, 64)).astype(np.float32)
        semantic_residuals = {
            'layer_4': np.random.random(512).astype(np.float32),
            'layer_8': np.random.random(512).astype(np.float32),
            'layer_12': np.random.random(512).astype(np.float32)
        }
        
        # 初始化模型
        residual_modeling = AdaptiveResidualModeling(device='cpu')
        
        # 测试自适应融合权重
        alpha = residual_modeling.adaptive_fusion_weight(S=0.5, S_c=0.5)
        assert 0 <= alpha <= 1
        
        # 测试残差融合
        joint_residual = residual_modeling.fuse_residuals(
            depth_residual, semantic_residuals, S=0.5, S_c=0.5
        )
        assert joint_residual.shape[2] == 2  # 深度 + 语义通道
        
        print("✓ 自适应残差信号建模测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 自适应残差信号建模测试失败: {e}")
        return False

def test_chaotic_scrambling():
    """测试双混沌加扰机制"""
    print("\n测试双混沌加扰机制...")
    
    try:
        from models.visual_fingerprint import ChaoticScrambling
        
        # 初始化混沌加扰
        chaotic_scrambling = ChaoticScrambling(key=42)
        
        # 测试混沌映射
        x1 = chaotic_scrambling.logistic_tent_mapping(0.3)
        assert 0 <= x1 <= 1
        
        x2, y2 = chaotic_scrambling.henon_mapping(0.1, 0.1)
        assert isinstance(x2, float) and isinstance(y2, float)
        
        # 测试混沌序列生成
        M1, M2 = chaotic_scrambling.generate_chaotic_sequences((32, 32))
        assert M1.shape == (32, 32) and M2.shape == (32, 32)
        
        # 测试混合掩码
        mask = chaotic_scrambling.create_mixed_mask((32, 32))
        assert mask.shape == (32, 32)
        assert np.all(np.logical_or(mask == 0, mask == 1))
        
        print("✓ 双混沌加扰机制测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 双混沌加扰机制测试失败: {e}")
        return False

def test_dynamic_quantization():
    """测试滑动窗口动态量化"""
    print("\n测试滑动窗口动态量化...")
    
    try:
        from models.visual_fingerprint import DynamicQuantization
        
        # 初始化量化器
        quantizer = DynamicQuantization(window_size=8)
        
        # 测试量化
        test_fingerprint = np.random.random((32, 32)).astype(np.float32)
        quantized = quantizer.dynamic_quantize(test_fingerprint)
        
        assert quantized.shape == test_fingerprint.shape
        assert quantized.dtype == np.uint8
        assert np.all(quantized >= 0) and np.all(quantized <= 255)
        
        # 测试滑动窗口更新
        for i in range(10):
            test_fp = np.random.random((16, 16)).astype(np.float32)
            quantizer.dynamic_quantize(test_fp)
        
        assert len(quantizer.fingerprint_history) <= quantizer.window_size
        
        print("✓ 滑动窗口动态量化测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 滑动窗口动态量化测试失败: {e}")
        return False

def test_lightweight_encoder():
    """测试轻量级深度编码网络"""
    print("\n测试轻量级深度编码网络...")
    
    try:
        from models.visual_fingerprint import LightweightEncoder
        import torch
        
        # 初始化编码器
        encoder = LightweightEncoder(input_channels=2, output_dim=128)
        
        # 创建测试输入
        test_input = torch.randn(1, 2, 64, 64)
        
        # 前向传播
        with torch.no_grad():
            output = encoder(test_input)
        
        assert output.shape == (1, 128)
        assert torch.all(output >= -1) and torch.all(output <= 1)  # Tanh输出范围
        
        print("✓ 轻量级深度编码网络测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 轻量级深度编码网络测试失败: {e}")
        return False

def test_fingerprint_generator():
    """测试视觉指纹生成器"""
    print("\n测试视觉指纹生成器...")
    
    try:
        from models.visual_fingerprint import VisualFingerprintGenerator
        from models.monodepth import MonodepthWrapper
        from models.clip_vit_new import ClipViTWrapper
        
        # 创建测试图像
        image_orig = np.random.random((256, 256, 3)).astype(np.float32)
        image_perturbed = np.random.random((256, 256, 3)).astype(np.float32)
        
        # 初始化组件
        fingerprint_generator = VisualFingerprintGenerator(device='cpu', key=42)
        monodepth = MonodepthWrapper(device='cpu')
        clip_vit = ClipViTWrapper(device='cpu')
        
        # 生成指纹
        fingerprint_result = fingerprint_generator.generate_fingerprint(
            image_orig, image_perturbed, 0.5, monodepth, clip_vit
        )
        
        # 检查结果
        required_keys = [
            'depth_residual', 'semantic_residuals', 'joint_residual',
            'encrypted_residual', 'quantized_fingerprint', 'fingerprint_hash'
        ]
        
        for key in required_keys:
            assert key in fingerprint_result
        
        # 检查指纹哈希
        fingerprint_hash = fingerprint_generator.compute_fingerprint_hash(
            fingerprint_result['fingerprint_hash']
        )
        assert len(fingerprint_hash) == 64  # SHA-256哈希长度
        
        print("✓ 视觉指纹生成器测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 视觉指纹生成器测试失败: {e}")
        return False

def test_fingerprint_evaluator():
    """测试指纹评估器"""
    print("\n测试指纹评估器...")
    
    try:
        from models.visual_fingerprint import FingerprintEvaluator
        
        # 创建测试数据
        depth_residual = np.random.random((64, 64)).astype(np.float32)
        semantic_residuals = {
            'layer_4': np.random.random(512).astype(np.float32),
            'layer_8': np.random.random(512).astype(np.float32),
            'layer_12': np.random.random(512).astype(np.float32)
        }
        fingerprint_hash = np.random.random(128).astype(np.float32)
        
        # 初始化评估器
        evaluator = FingerprintEvaluator(key=42)
        
        # 测试CMCS计算
        cmcs = evaluator.compute_cmcs(depth_residual, semantic_residuals, S=0.5)
        assert isinstance(cmcs, float)
        assert -1 <= cmcs <= 1  # 相关系数范围
        
        # 测试KDT-multi计算
        kdt_results = evaluator.compute_kdt_multi(fingerprint_hash)
        assert 'kdt_multi' in kdt_results
        assert 0 <= kdt_results['kdt_multi'] <= 1
        
        # 测试指纹哈希计算
        hash_str = evaluator.compute_fingerprint_hash(fingerprint_hash)
        assert len(hash_str) == 64  # SHA-256哈希长度
        
        print("✓ 指纹评估器测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 指纹评估器测试失败: {e}")
        return False

def test_integration():
    """测试集成功能"""
    print("\n测试集成功能...")
    
    try:
        from models.visual_fingerprint import VisualFingerprintGenerator, FingerprintEvaluator
        from models.monodepth import MonodepthWrapper
        from models.clip_vit_new import ClipViTWrapper
        from models.dce_module import DCEController
        
        # 创建测试图像
        image = np.random.random((256, 256, 3)).astype(np.float32)
        
        # 初始化所有组件
        fingerprint_generator = VisualFingerprintGenerator(device='cpu', key=42)
        fingerprint_evaluator = FingerprintEvaluator(key=42)
        monodepth = MonodepthWrapper(device='cpu')
        clip_vit = ClipViTWrapper(device='cpu')
        dce_controller = DCEController(base_key=42)
        dce_module = dce_controller.create_dce_module('test', image_size=256)
        
        # 生成扰动图像
        perturbed = dce_module.encrypt_image(image, 0.5, 'hybrid')
        
        # 生成指纹
        fingerprint_result = fingerprint_generator.generate_fingerprint(
            image, perturbed, 0.5, monodepth, clip_vit
        )
        
        # 评估指纹
        evaluation = fingerprint_evaluator.evaluate_fingerprint(fingerprint_result, 0.5)
        
        # 检查评估结果
        assert 'cmcs' in evaluation
        assert 'kdt_multi' in evaluation
        assert 'fingerprint_hash' in evaluation
        assert 'evaluation_summary' in evaluation
        
        print("✓ 集成功能测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 集成功能测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("开始运行视觉指纹系统测试...")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_adaptive_residual_modeling,
        test_chaotic_scrambling,
        test_dynamic_quantization,
        test_lightweight_encoder,
        test_fingerprint_generator,
        test_fingerprint_evaluator,
        test_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"测试 {test.__name__} 出现异常: {e}")
            results.append(False)
    
    # 汇总结果
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 60)
    print(f"视觉指纹系统测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有视觉指纹系统测试通过！系统可以正常运行。")
        return True
    else:
        print("⚠️  部分视觉指纹系统测试失败，请检查相关模块。")
        return False

if __name__ == "__main__":
    run_all_tests() 