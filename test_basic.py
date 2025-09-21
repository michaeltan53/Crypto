#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基本功能测试脚本
验证核心模块是否能正常工作
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def test_imports():
    """测试模块导入"""
    print("测试模块导入...")

    try:
        from models.zoedepth import ZoeDepthWrapper
        from models.dce_module import DCEController
        from utils.metrics import PerturbationAnalyzer, PhaseTransitionModel
        from utils.wavelet_chaos import WaveletChaosPerturbation
        from utils.diffusion_perturb import DiffusionPerturbation
        print("✓ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"✗ 模块导入失败: {e}")
        return False


def test_wavelet_chaos():
    """测试小波混沌扰动"""
    print("\n测试小波混沌扰动...")

    try:
        from utils.wavelet_chaos import WaveletChaosPerturbation

        # 创建测试图像
        image = np.random.random((128, 128, 3)).astype(np.float32)

        # 初始化扰动模块
        wavelet_chaos = WaveletChaosPerturbation(wavelet='db4', key=42)

        # 应用扰动
        perturbed = wavelet_chaos.perturb_image(image, 0.5, 42)

        # 检查结果
        assert perturbed.shape == image.shape
        assert perturbed.dtype == image.dtype
        assert np.all(perturbed >= 0) and np.all(perturbed <= 1)

        print("✓ 小波混沌扰动测试通过")
        return True

    except Exception as e:
        print(f"✗ 小波混沌扰动测试失败: {e}")
        return False


def test_diffusion_perturbation():
    """测试扩散模型扰动"""
    print("\n测试扩散模型扰动...")

    try:
        from utils.diffusion_perturb import DiffusionPerturbation

        # 创建测试图像
        image = np.random.random((128, 128, 3)).astype(np.float32)

        # 初始化扰动模块
        diffusion_perturb = DiffusionPerturbation(image_size=128)

        # 应用扰动
        perturbed = diffusion_perturb.embedded_diffusion_perturb(image, 0.5, 42)

        # 检查结果
        assert perturbed.shape == image.shape
        assert perturbed.dtype == image.dtype
        assert np.all(perturbed >= 0) and np.all(perturbed <= 1)

        print("✓ 扩散模型扰动测试通过")
        return True

    except Exception as e:
        print(f"✗ 扩散模型扰动测试失败: {e}")
        return False


def test_dce_module():
    """测试DCE模块"""
    print("\n测试DCE模块...")

    try:
        from models.dce_module import DCEController

        # 创建测试图像
        image = np.random.random((128, 128, 3)).astype(np.float32)

        # 初始化DCE控制器
        dce_controller = DCEController(base_key=42)
        dce_module = dce_controller.create_dce_module('test', image_size=128)

        # 测试不同扰动类型
        for pert_type in ['wavelet', 'diffusion', 'hybrid']:
            perturbed = dce_module.encrypt_image(image, 0.5, pert_type)
            assert perturbed.shape == image.shape
            assert perturbed.dtype == image.dtype
            assert np.all(perturbed >= 0) and np.all(perturbed <= 1)

        print("✓ DCE模块测试通过")
        return True

    except Exception as e:
        print(f"✗ DCE模块测试失败: {e}")
        return False


def test_zoedepth():
    """测试ZoeDepth模型"""
    print("\n测试ZoeDepth模型...")

    try:
        from models.zoedepth import ZoeDepthWrapper

        # 创建测试图像
        image = np.random.random((128, 128, 3)).astype(np.float32)

        # 初始化深度估计模型
        zoedepth = ZoeDepthWrapper(device='cpu')

        # 估计深度
        depth_map = zoedepth.estimate_depth(image)

        # 检查结果
        assert depth_map.shape == (128, 128)
        assert depth_map.dtype == np.float32

        print("✓ ZoeDepth模型测试通过")
        return True

    except Exception as e:
        print(f"✗ ZoeDepth模型测试失败: {e}")
        return False


def test_phase_transition_model():
    """测试相变模型"""
    print("\n测试相变模型...")

    try:
        from utils.metrics import PhaseTransitionModel

        # 生成测试数据
        S = np.linspace(0, 1, 20)
        delta_P = 0.1 * S + 0.01 * np.exp(2 * S) + np.random.normal(0, 0.01, 20)

        # 初始化模型
        phase_model = PhaseTransitionModel()

        # 拟合模型
        fit_result = phase_model.fit_model(S, delta_P)

        # 检查结果
        assert fit_result is not None
        assert 'r_squared' in fit_result
        assert 'params' in fit_result
        assert fit_result['r_squared'] > 0

        print("✓ 相变模型测试通过")
        return True

    except Exception as e:
        print(f"✗ 相变模型测试失败: {e}")
        return False


def test_metrics():
    """测试评估指标"""
    print("\n测试评估指标...")

    try:
        from utils.metrics import DepthMetrics

        # 创建测试数据
        depth_pred = np.random.random((64, 64)).astype(np.float32)
        depth_gt = depth_pred + np.random.normal(0, 0.1, (64, 64))

        # 计算指标
        metrics = DepthMetrics.compute_all_metrics(depth_pred, depth_gt)

        # 检查结果
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'rel' in metrics
        assert 'ssim' in metrics
        assert all(isinstance(v, (int, float)) for v in metrics.values())

        print("✓ 评估指标测试通过")
        return True

    except Exception as e:
        print(f"✗ 评估指标测试失败: {e}")
        return False


def test_analyzer():
    """测试扰动分析器"""
    print("\n测试扰动分析器...")

    try:
        from utils.metrics import PerturbationAnalyzer

        # 生成测试数据
        strengths = np.linspace(0, 1, 15)
        depth_errors = 0.1 * strengths + 0.01 * np.exp(2 * strengths) + np.random.normal(0, 0.01, 15)

        # 初始化分析器
        analyzer = PerturbationAnalyzer()

        # 分析结果
        analysis_result = analyzer.analyze_perturbation_response(strengths, depth_errors)

        # 检查结果
        assert analysis_result is not None
        assert 'fit_result' in analysis_result
        assert 'statistics' in analysis_result
        assert 'phase_transition' in analysis_result

        # 生成报告
        report = analyzer.generate_report()
        assert len(report) > 0

        print("✓ 扰动分析器测试通过")
        return True

    except Exception as e:
        print(f"✗ 扰动分析器测试失败: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("开始运行基本功能测试...")
    print("=" * 50)

    tests = [
        test_imports,
        test_wavelet_chaos,
        test_diffusion_perturbation,
        test_dce_module,
        test_zoedepth,
        test_phase_transition_model,
        test_metrics,
        test_analyzer
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

    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！系统可以正常运行。")
        return True
    else:
        print("⚠️  部分测试失败，请检查相关模块。")
        return False


if __name__ == "__main__":
    run_all_tests()
