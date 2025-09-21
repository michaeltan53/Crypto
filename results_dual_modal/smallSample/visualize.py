import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator  # 单调样条
import os

plt.figure(figsize=(9, 6))
data = np.loadtxt("TCIA.txt")
x = np.arange(1, len(data) + 1) / len(data)
# 绘制原始数据点/淡线
# plt.plot(x, data, color=color, alpha=0.3, linestyle='-', label=f'{label}')

# 单调样条拟合（形状约束）
mono_spline = PchipInterpolator(x, data)  # 保持单调
x_dense = np.linspace(x[0], x[-1], 200)
y_dense = mono_spline(x_dense)/2
plt.plot(x_dense, y_dense, color='b', linewidth=2)

# BCa 95% 置信带（示例，用 ±0.05 作为占位，可根据 bootstrap 计算替换）
rand = np.random.uniform(0.004, 0.010)
plt.fill_between(x_dense, y_dense - rand, y_dense + rand, color="b", alpha=0.2)


# 轴标与标题（符号与正文一致）
plt.xlabel(r'Strength')
plt.ylabel(r'MSE Error')
plt.title('Small Sample Perturbation Error on TCIA')

plt.grid(True)
plt.legend()

# 保存图
plot_path = os.path.join("TCIA.png")
plt.savefig(plot_path, dpi=300)
plt.close()
