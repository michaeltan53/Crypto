import os
import numpy as np
import cv2
from models.dce_module_new import DCEController  # 复用你已有的加密模块


def main():
    # 输入图像路径
    image_path = "/Users/ushiushi/PycharmProjects/DiffCrypto_710(1)/images/7.png"
    save_dir = "./per_result"
    os.makedirs(save_dir, exist_ok=True)

    # 读取图像 (BGR->RGB)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"图像未找到: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0  # 归一化到0-1

    # 初始化加密模块
    key = 42  # 可以改成你想用的key
    dce_controller = DCEController(base_key=key)
    dce_module = dce_controller.create_dce_module("perturb_exp", image_size=(480, 1280))

    # # 设置扰动强度
    # num_bins = 10
    # strengths = np.linspace(0.0, 1.0, num_bins)

    num_bins = 10
    max_strength = 9
    strengths = np.linspace(8, max_strength, num_bins)
    # # 用平方根映射，让前半段更温和
    # strengths = np.sqrt(linear_strengths / max_strength) * max_strength

    # 逐级扰动并保存
    count = -1
    for s in strengths:
        count += 1
        perturbed_img = dce_module.encrypt_image(img, float(s), 'wavelet')
        save_path = os.path.join(save_dir, f"0_perturbed_S{s:.8f}.png")
        cv2.imwrite(save_path, cv2.cvtColor((perturbed_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        print(f"保存扰动图像: {save_path}")


if __name__ == "__main__":
    main()
