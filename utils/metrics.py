import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, brentq
from scipy.interpolate import interp1d
from scipy.stats import pearsonr, ttest_ind, mannwhitneyu
from typing import Tuple, List, Dict, Optional
import cv2
from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import peak_signal_to_noise_ratio as psnr
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthMetrics:
    """
    深度估计评估指标
    """

    @staticmethod
    def mean_absolute_error(depth_pred: np.ndarray, depth_gt: np.ndarray) -> float:
        """
        平均绝对误差 (MAE)
        """
        return np.mean(np.abs(depth_pred - depth_gt))

    @staticmethod
    def root_mean_square_error(depth_pred: np.ndarray, depth_gt: np.ndarray) -> float:
        """
        均方根误差 (RMSE)
        """
        return np.sqrt(np.mean((depth_pred - depth_gt) ** 2))

    @staticmethod
    def relative_error(depth_pred: np.ndarray, depth_gt: np.ndarray) -> float:
        """
        相对误差
        """
        return np.mean(np.abs(depth_pred - depth_gt) / (depth_gt + 1e-8))

    @staticmethod
    def delta_accuracy(depth_pred: np.ndarray, depth_gt: np.ndarray,
                       threshold: float = 1.25) -> float:
        """
        Delta准确率 (δ < threshold)
        """
        ratio = np.maximum(depth_pred / (depth_gt + 1e-8),
                           depth_gt / (depth_pred + 1e-8))
        return np.mean(ratio < threshold)

    @staticmethod
    def structural_similarity(depth_pred: np.ndarray, depth_gt: np.ndarray) -> float:
        """
        结构相似性 (SSIM)
        """
        # 归一化到[0, 1]
        pred_norm = (depth_pred - depth_pred.min()) / (depth_pred.max() - depth_pred.min() + 1e-8)
        gt_norm = (depth_gt - depth_gt.min()) / (depth_gt.max() - depth_gt.min() + 1e-8)

        return ssim(pred_norm, gt_norm, data_range=1.0)

    @staticmethod
    def compute_all_metrics(depth_pred: np.ndarray, depth_gt: np.ndarray) -> Dict[str, float]:
        """
        计算所有深度评估指标
        """
        return {
            'mae': DepthMetrics.mean_absolute_error(depth_pred, depth_gt),
            'rmse': DepthMetrics.root_mean_square_error(depth_pred, depth_gt),
            'rel': DepthMetrics.relative_error(depth_pred, depth_gt),
            'delta_1.25': DepthMetrics.delta_accuracy(depth_pred, depth_gt, 1.25),
            'delta_1.25^2': DepthMetrics.delta_accuracy(depth_pred, depth_gt, 1.25 ** 2),
            'delta_1.25^3': DepthMetrics.delta_accuracy(depth_pred, depth_gt, 1.25 ** 3),
            'ssim': DepthMetrics.structural_similarity(depth_pred, depth_gt)
        }


class PhaseTransitionModel:
    """
    相变模型拟合
    实现分段连续的双态函数模型
    """

    def __init__(self):
        self.params = None
        self.r_squared = None
        self.threshold = None

    def phase_transition_function(self, S: np.ndarray, k1: float, k2: float,
                                  beta: float, S_th: float) -> np.ndarray:
        """
        分段连续的双态函数模型
        ΔP(S) = {
            k₁S,           S ≤ S_th
            k₂e^(βS),      S > S_th
        }
        满足连续性约束：k₁S_th = k₂e^(βS_th)
        """
        result = np.zeros_like(S)

        # 线性阶段
        linear_mask = S <= S_th
        result[linear_mask] = k1 * S[linear_mask]

        # 指数阶段
        exp_mask = S > S_th
        result[exp_mask] = k2 * np.exp(beta * S[exp_mask])

        return result

    def fit_model(self, S: np.ndarray, delta_P: np.ndarray,
                  initial_guess: Optional[List[float]] = None) -> Dict:
        """
        拟合相变模型
        Args:
            S: 扰动强度数组
            delta_P: 感知退化量数组
            initial_guess: 初始参数猜测 [k1, k2, beta, S_th]
        Returns:
            拟合结果字典
        """
        if initial_guess is None:
            initial_guess = [0.1, 0.01, 2.0, 0.5]

        # 定义拟合函数（包含连续性约束）
        def fit_function(S, k1, k2, beta, S_th):
            # 应用连续性约束
            k2_constrained = k1 * S_th / np.exp(beta * S_th)
            return self.phase_transition_function(S, k1, k2_constrained, beta, S_th)

        try:
            # 拟合参数
            popt, pcov = curve_fit(fit_function, S, delta_P, p0=initial_guess,
                                   bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, 1]))

            k1, k2, beta, S_th = popt
            k2_constrained = k1 * S_th / np.exp(beta * S_th)

            # 计算拟合优度
            y_pred = fit_function(S, k1, k2, beta, S_th)
            ss_res = np.sum((delta_P - y_pred) ** 2)
            ss_tot = np.sum((delta_P - np.mean(delta_P)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # 计算相关系数
            correlation, p_value = pearsonr(S, delta_P)

            self.params = {
                'k1': k1,
                'k2': k2_constrained,
                'beta': beta,
                'S_th': S_th
            }
            self.r_squared = r_squared
            self.threshold = S_th

            return {
                'params': self.params,
                'r_squared': r_squared,
                'correlation': correlation,
                'p_value': p_value,
                'pcov': pcov
            }

        except Exception as e:
            print(f"拟合失败: {e}")
            return None

    def predict(self, S: np.ndarray) -> np.ndarray:
        """
        使用拟合的模型进行预测
        """
        if self.params is None:
            raise ValueError("模型尚未拟合，请先调用fit_model方法")

        k1, k2, beta, S_th = (self.params['k1'], self.params['k2'],
                              self.params['beta'], self.params['S_th'])

        return self.phase_transition_function(S, k1, k2, beta, S_th)

    def plot_fit(self, S: np.ndarray, delta_P: np.ndarray,
                 save_path: Optional[str] = None) -> None:
        """
        绘制拟合结果
        """
        if self.params is None:
            raise ValueError("模型尚未拟合，请先调用fit_model方法")

        plt.figure(figsize=(10, 6))

        # 原始数据
        plt.scatter(S, delta_P, alpha=0.6, label='实验数据', color='blue')

        # 拟合曲线
        S_smooth = np.linspace(0, 1, 1000)
        y_fit = self.predict(S_smooth)
        plt.plot(S_smooth, y_fit, 'r-', linewidth=2, label='拟合曲线')

        # 相变点
        S_th = self.params['S_th']
        y_th = self.predict(np.array([S_th]))
        plt.axvline(x=S_th, color='green', linestyle='--',
                    label=f'相变点 S_th = {S_th:.3f}')
        plt.scatter([S_th], y_th, color='red', s=100, zorder=5)

        # 分段显示
        S_linear = S_smooth[S_smooth <= S_th]
        S_exp = S_smooth[S_smooth > S_th]

        if len(S_linear) > 0:
            y_linear = self.predict(S_linear)
            plt.plot(S_linear, y_linear, 'g-', linewidth=3,
                     label='线性阶段 (鲁棒性)')

        if len(S_exp) > 0:
            y_exp = self.predict(S_exp)
            plt.plot(S_exp, y_exp, 'orange', linewidth=3,
                     label='指数阶段 (激增)')

        plt.xlabel('扰动强度 S')
        plt.ylabel('感知退化量 ΔP(S)')
        plt.title(f'相变模型拟合结果 (R² = {self.r_squared:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


class DualModalAnalyzer:
    """
    双模态（深度+语义）扰动分析器
    """

    def __init__(self):
        self.results = {}
        self.crossover_point = None

    def _normalize_errors(self, errors: np.ndarray) -> np.ndarray:
        """
        将误差归一化到[0, 1]范围
        """
        min_e = errors.min()
        max_e = errors.max()
        if max_e == min_e:
            return np.zeros_like(errors)
        return (errors - min_e) / (max_e - min_e)

    def _find_crossover_point(self, S: np.ndarray, depth_errors: np.ndarray,
                              semantic_errors: np.ndarray) -> Optional[float]:
        """
        寻找深度和语义误差曲线的交叉点
        """
        # 归一化误差以便于比较
        norm_depth_err = self._normalize_errors(depth_errors)
        norm_semantic_err = self._normalize_errors(semantic_errors)

        # 创建插值函数
        f_depth = interp1d(S, norm_depth_err, fill_value="extrapolate")
        # 将 norm_semantic_err 转换为一维数组
        # norm_semantic_err = norm_semantic_err.mean(axis=1)
        # print(S.shape, norm_semantic_err.shape)
        f_semantic = interp1d(S, norm_semantic_err, fill_value="extrapolate")

        # 定义差值函数，寻找其根
        f_diff = lambda s: f_depth(s) - f_semantic(s)

        # 寻找交叉点
        crossover_candidates = []
        for i in range(len(S) - 1):
            s_start, s_end = S[i], S[i + 1]
            if f_diff(s_start) * f_diff(s_end) < 0:
                try:
                    crossover = brentq(f_diff, s_start, s_end)
                    crossover_candidates.append(crossover)
                except (ValueError, RuntimeError):
                    continue

        # 返回第一个找到的有效交叉点
        return crossover_candidates[0] if crossover_candidates else None

    def analyze_dual_modal_response(self, strengths: np.ndarray,
                                    depth_errors: np.ndarray,
                                    semantic_errors: np.ndarray) -> Dict:
        """
        分析双模态扰动响应
        """
        # 寻找交叉点
        self.crossover_point = self._find_crossover_point(strengths, depth_errors, semantic_errors)

        analysis_result = {
            'crossover_point': self.crossover_point,
            'strengths': strengths,
            'depth_errors': depth_errors,
            'semantic_errors': semantic_errors,
        }

        if self.crossover_point is not None:
            print(f"感知主导切换点 (Sc): {self.crossover_point:.4f}")
        else:
            print("未找到明确的感知主导切换点。")

        self.results = analysis_result
        return analysis_result

    def plot_dual_modal_response(self, save_path: Optional[str] = None):
        """
        绘制双模态响应曲线和交叉点
        """
        if not self.results:
            raise ValueError("请先运行 analyze_dual_modal_response")

        S = self.results['strengths']
        depth_errors = self.results['depth_errors']
        semantic_errors = self.results['semantic_errors']

        # 归一化误差用于绘图
        norm_depth = self._normalize_errors(depth_errors)
        norm_semantic = self._normalize_errors(semantic_errors)

        plt.figure(figsize=(12, 7))
        plt.plot(S, norm_depth, 'r-o', label='geomerty diff $\Delta P(S)$', alpha=0.7)
        plt.plot(S, norm_semantic, 'b-s', label='segment diff $\Delta f(S)$', alpha=0.7)

        if self.crossover_point is not None:
            Sc = self.crossover_point
            f_depth_interp = interp1d(S, norm_depth)
            crossover_y = f_depth_interp(Sc)

            plt.axvline(x=Sc, color='purple', linestyle='--', label=f'cross point $S_c={Sc:.3f}$')
            plt.scatter([Sc], [crossover_y], color='purple', s=150, zorder=5, marker='*')

            # 填充区域
            plt.fill_between(S, 0, 1, where=S < Sc, color='blue', alpha=0.1, label='segment area')
            plt.fill_between(S, 0, 1, where=S >= Sc, color='red', alpha=0.1, label='geometry area')

        plt.xlabel('Disturbance intensity S')
        plt.ylabel('Normalized diff')
        plt.title('Bimodal perception residual response curve')
        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.ylim(0, 1.05)
        plt.xlim(0, 1)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def generate_report(self) -> str:
        """
        生成分析报告
        """
        if not self.results:
            return "暂无分析结果"

        report = "=== 双模态扰动响应分析报告 ===\n\n"

        if self.crossover_point is not None:
            Sc = self.crossover_point
            report += f"【交叉点】\n"
            report += f"- 感知主导模态切换点 (交叉点) Sc: {Sc:.4f}\n\n"
            report += "【模态主导阶段分析】\n"
            report += f"- 弱扰动 (S < {Sc:.3f}): 语义变化主导感知退化。\n"
            report += f"  此阶段，语义残差增长较快，建议采用深度+语义双通道视觉指纹。\n\n"
            report += f"- 强扰动 (S >= {Sc:.3f}): 几何结构崩解主导感知退化。\n"
            report += f"  此阶段，深度误差成为主要退化信号，建议采用单通道深度指纹以降低冗余。\n"
        else:
            report += "【核心发现】\n"
            report += "- 未在[0,1]区间内找到明确的模态切换点。\n"
            report += "- 可能原因：在所有扰动强度下，某一模态始终占主导地位。\n"

        return report

    def three_stage_response(self, S, delta_P, delta_f, S1, S2, gamma=10):
        """
        三阶段分段响应函数 R(S)
        S: 扰动强度数组
        delta_P: 几何残差数组
        delta_f: 语义残差数组
        S1, S2: 阶段分界点
        gamma: Sigmoid陡度
        返回: R(S) 响应数组
        """
        R = []
        for i, s in enumerate(S):
            if s <= S1:
                R.append(delta_f[i])
            elif S1 < s <= S2:
                alpha = 1 / (1 + np.exp(-gamma * (s - (S1 + S2) / 2)))
                R.append([delta_P[i], alpha * delta_f[i]])
            else:
                R.append(delta_P[i])
        return R

    def continuous_fusion_response(self, S, delta_P, delta_f, S1, S2, gamma=10):
        """
        连续模态调控融合函数 Φ(S)
        S: 扰动强度数组
        delta_P: 几何残差数组
        delta_f: 语义残差数组
        S1, S2: 阶段分界点
        gamma: Sigmoid陡度
        返回: Φ(S) 融合数组
        """
        Sc = (S1 + S2) / 2
        # change it to tanh
        omega = 1 / (1 + np.exp(-gamma * (S - Sc)))
        Phi = omega * delta_P + (1 - omega) * delta_f
        return Phi, omega

    def cmcs(self, S, delta_D, delta_f, S1, S2, gamma=10):
        """
        计算CMCS指标
        S: 扰动强度数组
        delta_D: 几何残差数组
        delta_f: 语义残差数组
        S1, S2: 阶段分界点
        gamma: Sigmoid陡度
        返回: CMCS数组
        """
        Sc = (S1 + S2) / 2
        # change it to tanh
        omega = 1 / (1 + np.exp(-gamma * (S - Sc)))
        cmcs_list = []
        for i in range(len(S)):
            # Pearson相关系数
            try:
                rho = pearsonr(delta_D[:i + 1], delta_f[:i + 1])[0]
            except Exception:
                rho = 0
            # SSIM
            try:
                ssim_val = ssim(delta_D[:i + 1], delta_f[:i + 1])
            except Exception:
                ssim_val = 0
            cmcs_val = omega[i] * rho + (1 - omega[i]) * ssim_val
            cmcs_list.append(cmcs_val)
        return np.array(cmcs_list)

    def kdt_multi(self, Fk, Fk_1, Fk_4, Fk_8, tau1=0.3, tau4=0.3, tau8=0.3):
        """
        计算KDT-multi指标
        Fk: 原始指纹
        Fk_1, Fk_4, Fk_8: 1/4/8-bit扰动下的指纹
        tau1, tau4, tau8: 各级阈值
        返回: KDT-multi分数
        """

        def hamming_dist(a, b):
            return np.sum(a != b)

        n = len(Fk)
        hd1 = hamming_dist(Fk, Fk_1) / n
        hd4 = hamming_dist(Fk, Fk_4) / n
        hd8 = hamming_dist(Fk, Fk_8) / n
        score = (int(hd1 > tau1) + int(hd4 > tau4) + int(hd8 > tau8)) / 3
        return score


class ThreeStageResponseModel:
    """
    三阶段扰动响应模型
    实现论文3.1.2节描述的三阶段分段演化特性
    """

    def __init__(self):
        self.S1 = 0.14  # 高敏感阶段与线性鲁棒阶段的分界点
        self.S2 = 0.60  # 线性鲁棒阶段与结构主导阶段的分界点
        self.gamma = 1.8  # 幂指数增长参数
        self.params = {}

    def geometric_residual_model(self, S: np.ndarray, k1: float = 0.5,
                                 k2: float = 0.3, k3: float = 0.1,
                                 beta: float = 2.5, b: float = 0.05) -> np.ndarray:
        """
        几何残差ΔP(S)的三段式拟合模型
        ΔP(S) = {
            k₁S^γ,         S ≤ S₁
            k₂S + b,       S₁ < S ≤ S₂  
            k₃e^(βS),      S > S₂
        }
        """
        result = np.zeros_like(S)

        # 高敏感阶段 (S ≤ S₁)
        mask1 = S <= self.S1
        result[mask1] = k1 * (S[mask1] ** self.gamma)

        # 线性鲁棒阶段 (S₁ < S ≤ S₂)
        mask2 = (S > self.S1) & (S <= self.S2)
        result[mask2] = k2 * S[mask2] + b

        # 结构主导阶段 (S > S₂)
        mask3 = S > self.S2
        result[mask3] = k3 * np.exp(beta * S[mask3])

        return result

    def semantic_residual_model(self, S: np.ndarray, k_sat: float = 0.8,
                                alpha: float = 3.0) -> np.ndarray:
        """
        语义残差Δf(S)的饱和响应模型
        Δf(S) = k_sat * (1 - e^(-αS))
        """
        return k_sat * (1 - np.exp(-alpha * S))

    def fit_three_stage_model(self, S: np.ndarray, delta_P: np.ndarray,
                              delta_f: np.ndarray) -> Dict:
        """
        拟合三阶段响应模型
        """
        try:
            # 拟合几何残差模型
            def geometric_fit_func(S, k1, k2, k3, beta, b):
                return self.geometric_residual_model(S, k1, k2, k3, beta, b)

            popt_geo, pcov_geo = curve_fit(geometric_fit_func, S, delta_P,
                                           p0=[0.5, 0.3, 0.1, 2.5, 0.05],
                                           bounds=([0, 0, 0, 0, -1],
                                                   [np.inf, np.inf, np.inf, 10, 1]))

            # 拟合语义残差模型
            def semantic_fit_func(S, k_sat, alpha):
                return self.semantic_residual_model(S, k_sat, alpha)

            popt_sem, pcov_sem = curve_fit(semantic_fit_func, S, delta_f,
                                           p0=[0.8, 3.0],
                                           bounds=([0, 0], [np.inf, np.inf]))

            # 计算拟合优度
            y_pred_geo = geometric_fit_func(S, *popt_geo)
            y_pred_sem = semantic_fit_func(S, *popt_sem)

            r2_geo = 1 - np.sum((delta_P - y_pred_geo) ** 2) / np.sum((delta_P - np.mean(delta_P)) ** 2)
            r2_sem = 1 - np.sum((delta_f - y_pred_sem) ** 2) / np.sum((delta_f - np.mean(delta_f)) ** 2)

            self.params = {
                'geometric': {
                    'k1': popt_geo[0], 'k2': popt_geo[1], 'k3': popt_geo[2],
                    'beta': popt_geo[3], 'b': popt_geo[4], 'r2': r2_geo
                },
                'semantic': {
                    'k_sat': popt_sem[0], 'alpha': popt_sem[1], 'r2': r2_sem
                }
            }

            return self.params

        except Exception as e:
            print(f"三阶段模型拟合失败: {e}")
            return None

    def statistical_significance_test(self, S: np.ndarray, delta_P: np.ndarray,
                                      delta_f: np.ndarray) -> Dict:
        """
        对临界点S₁和S₂进行显著性检验
        """
        results = {}

        # 检验S₁点 (S=0.15)
        S1_idx = np.argmin(np.abs(S - 0.15))
        if 0 < S1_idx < len(S) - 1:
            # 取前后小区间样本
            before_S1 = delta_P[:S1_idx]
            after_S1 = delta_P[S1_idx:]

            # t检验
            t_stat, p_value = ttest_ind(before_S1, after_S1)
            results['S1_t_test'] = {'t_stat': t_stat, 'p_value': p_value}

            # Mann-Whitney U检验
            u_stat, u_p_value = mannwhitneyu(before_S1, after_S1, alternative='two-sided')
            results['S1_mann_whitney'] = {'u_stat': u_stat, 'p_value': u_p_value}

        # 检验S₂点 (S=0.60)
        S2_idx = np.argmin(np.abs(S - 0.60))
        if 0 < S2_idx < len(S) - 1:
            before_S2 = delta_P[:S2_idx]
            after_S2 = delta_P[S2_idx:]

            t_stat, p_value = ttest_ind(before_S2, after_S2)
            results['S2_t_test'] = {'t_stat': t_stat, 'p_value': p_value}

            u_stat, u_p_value = mannwhitneyu(before_S2, after_S2, alternative='two-sided')
            results['S2_mann_whitney'] = {'u_stat': u_stat, 'p_value': u_p_value}

        return results

    def plot_three_stage_response(self, S: np.ndarray, delta_P: np.ndarray,
                                  delta_f: np.ndarray, save_path: Optional[str] = None):
        """
        绘制三阶段响应曲线
        """
        plt.figure(figsize=(15, 10))

        # 子图1: 几何残差响应
        plt.subplot(2, 2, 1)
        plt.scatter(S, delta_P, alpha=0.6, label='Experiment Data', color='blue')

        # 拟合曲线
        if self.params:
            S_smooth = np.linspace(0, 1, 1000)
            geo_params = self.params['geometric'].copy()
            geo_params.pop('r2')  # 移除 r2
            y_fit_geo = self.geometric_residual_model(S_smooth, **geo_params)
            # y_fit_geo = self.geometric_residual_model(S_smooth, **self.params['geometric'])
            plt.plot(S_smooth, y_fit_geo, 'r-', linewidth=2, label='Fitting Curve')

        # 标记阶段分界点
        plt.axvline(x=self.S1, color='green', linestyle='--', label=f'S₁={self.S1}')
        # plt.axvline(x=self.S2, color='orange', linestyle='--', label=f'S₂={self.S2}')

        plt.xlabel('Strengths S')
        plt.ylabel('Geometry Difference ΔP(S)')
        plt.title('Triple-stage and modal switch')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图2: 语义残差响应
        plt.subplot(2, 2, 2)
        # print(S.shape, delta_f.shape)
        # delta_f = delta_f.mean(axis=1)
        plt.scatter(S, delta_f, alpha=0.6, label='Experiment Data', color='red')

        if self.params:
            S_smooth = np.linspace(0, 1, 1000)
            sem_params = self.params['semantic'].copy()
            sem_params.pop('r2')  # 移除 r2
            y_fit_sem = self.semantic_residual_model(S_smooth, **sem_params)
            # y_fit_sem = self.semantic_residual_model(S_smooth, **self.params['semantic'])
            plt.plot(S_smooth, y_fit_sem, 'r-', linewidth=2, label='Fitting Curve')

        plt.xlabel('Strength S')
        plt.ylabel('Semantic Difference Δf(S)')
        plt.title('Semantic Difference Response')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图3: 双模态对比
        plt.subplot(2, 2, 3)
        plt.plot(S, delta_P, 'b-o', label='Geometry Differnece ΔP(S)', alpha=0.7)
        plt.plot(S, delta_f, 'r-s', label='Semantic Difference Δf(S)', alpha=0.7)
        plt.axvline(x=self.S1, color='green', linestyle='--', alpha=0.5)
        # plt.axvline(x=self.S2, color='orange', linestyle='--', alpha=0.5)

        plt.xlabel('Strength S')
        plt.ylabel('Difference Signal')
        plt.title('Dual-modal difference comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图4: 阶段划分
        plt.subplot(2, 2, 4)
        plt.fill_between(S, 0, 1, where=S <= self.S1, color='lightblue', alpha=0.3, label='High Sensative Stage')
        plt.fill_between(S, 0, 1, where=(S > self.S1) & (S <= self.S2), color='lightgreen', alpha=0.3,
                         label='Linear Robust stage')
        plt.fill_between(S, 0, 1, where=S > self.S2, color='lightcoral', alpha=0.3, label='Structure Leading Stage')

        plt.xlabel('strength S')
        plt.ylabel('Fusion Split')
        plt.title('Triple-stage Split')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


class ModalDominanceSwitching:
    """
    模态主导性切换机制
    实现论文3.1.3节描述的残差融合响应函数R(S)
    """

    def __init__(self, S_c: float = 0.637):
        self.S_c = S_c  # 模态切换临界点
        self.gamma_0 = 2.0  # 动态Sigmoid参数
        self.gamma_1 = 1.5

    def lambda_f_function(self, S: np.ndarray) -> np.ndarray:
        """
        动态融合权重函数λ_f(S)
        λ_f(S) = 1 / (1 + e^(-(γ₀ + γ₁S)(S - S_c)))
        """
        gamma_dynamic = self.gamma_0 + self.gamma_1 * S
        return 1 / (1 + np.exp(-gamma_dynamic * (S - self.S_c)))

    def residual_fusion_response(self, S: np.ndarray, delta_P: np.ndarray,
                                 delta_f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        统一的残差融合响应函数R(S)
        R(S) = {
            ΔP(S) + λ_f(S) · Δf(S),  S < S_c
            ΔP(S),                    S ≥ S_c
        }
        """
        lambda_f = self.lambda_f_function(S)
        delta_P = np.repeat(delta_P[:, np.newaxis], 3, axis=1)
        lambda_f = np.repeat(lambda_f[:, np.newaxis], 3, axis=1)

        delta_f = np.tile(delta_f[:, np.newaxis], (1, 3))
        condition_exp = np.repeat((S < self.S_c)[:, np.newaxis], delta_f.shape[1], axis=1)  # (30, 3)

        # 应用分段函数
        # print(delta_P.shape)
        # print(lambda_f.shape)
        # print(delta_f.shape)
        # print((lambda_f * delta_f).shape)
        # print((delta_P + lambda_f * delta_f).shape)

        R = np.where(condition_exp,
                     delta_P + lambda_f * delta_f,
                     delta_P)

        return R, lambda_f


class CrossModalConsistencyLoss:
    """
    跨模态结构一致性约束损失
    实现论文3.1.3节描述的多项一致性约束
    """

    def __init__(self, alpha1: float = 1.0, alpha2: float = 1.0, alpha3: float = 1.0):
        self.alpha1 = alpha1  # 边缘位置一致性损失权重
        self.alpha2 = alpha2  # 结构相似性损失权重  
        self.alpha3 = alpha3  # 梯度方向对齐损失权重

    def edge_consistency_loss(self, delta_P: torch.Tensor, delta_f: torch.Tensor) -> torch.Tensor:
        """
        边缘位置一致性损失
        L_cmc = ||∇(ΔP) ⊙ ∇(Δf)||₁
        """
        grad_P = torch.gradient(delta_P, dim=(0, 1))
        grad_f = torch.gradient(delta_f, dim=(0, 1))

        # 计算梯度幅度
        grad_P_mag = torch.sqrt(grad_P[0] ** 2 + grad_P[1] ** 2)
        grad_f_mag = torch.sqrt(grad_f[0] ** 2 + grad_f[1] ** 2)

        # 计算重叠程度
        overlap = grad_P_mag * grad_f_mag
        return torch.norm(overlap, p=1)

    def structural_similarity_loss(self, delta_P: torch.Tensor, delta_f: torch.Tensor) -> torch.Tensor:
        """
        结构相似性损失
        L_ssim = 1 - SSIM(ΔP, Δf)
        """
        # 归一化到[0,1]
        delta_P_norm = (delta_P - delta_P.min()) / (delta_P.max() - delta_P.min() + 1e-8)
        delta_f_norm = (delta_f - delta_f.min()) / (delta_f.max() - delta_f.min() + 1e-8)

        ssim_score = ssim(delta_P_norm.cpu().numpy(), delta_f_norm.cpu().numpy(), data_range=1.0)
        return 1 - torch.tensor(ssim_score, dtype=torch.float32)

    def gradient_direction_loss(self, delta_P: torch.Tensor, delta_f: torch.Tensor) -> torch.Tensor:
        """
        梯度方向对齐损失
        L_dir = ||∠(∇ΔP) - ∠(∇Δf)||₂
        """
        grad_P = torch.gradient(delta_P, dim=(0, 1))
        grad_f = torch.gradient(delta_f, dim=(0, 1))

        # 计算梯度方向角度
        angle_P = torch.atan2(grad_P[1], grad_P[0])
        angle_f = torch.atan2(grad_f[1], grad_f[0])

        # 计算角度差异
        angle_diff = torch.abs(angle_P - angle_f)
        return torch.norm(angle_diff, p=2)

    def compute_total_loss(self, delta_P: torch.Tensor, delta_f: torch.Tensor) -> torch.Tensor:
        """
        计算总的一致性损失
        L_fuse = α₁L_cmc + α₂L_ssim + α₃L_dir
        """
        L_cmc = self.edge_consistency_loss(delta_P, delta_f)
        L_ssim = self.structural_similarity_loss(delta_P, delta_f)
        L_dir = self.gradient_direction_loss(delta_P, delta_f)

        total_loss = (self.alpha1 * L_cmc +
                      self.alpha2 * L_ssim +
                      self.alpha3 * L_dir)

        return total_loss
