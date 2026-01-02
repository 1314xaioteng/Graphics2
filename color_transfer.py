"""
色彩迁移模块 - 按推荐方案实现
================================

Reinhard 色彩迁移算法（核心20行）
简单、高效、效果好
"""

import numpy as np
import cv2


def reinhard_color_transfer(source, reference):
    """
    Reinhard色彩迁移算法（推荐方案 - 核心20行）

    Args:
        source: 源图像 (H, W, 3) RGB
        reference: 参考图像 (H, W, 3) RGB

    Returns:
        迁移后的图像
    """
    # 转换到LAB空间
    source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype(np.float32)

    # 计算统计量
    mean_src = source_lab.mean(axis=(0, 1))
    std_src = source_lab.std(axis=(0, 1))
    mean_ref = reference_lab.mean(axis=(0, 1))
    std_ref = reference_lab.std(axis=(0, 1))

    # 应用变换
    result_lab = (source_lab - mean_src) * (std_ref / std_src) + mean_ref
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)

    # 转回RGB
    result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
    return result


# 兼容旧接口的类包装
class ColorTransfer:
    @staticmethod
    def reinhard_color_transfer(source, reference, strength=1.0, mask=None):
        """兼容旧接口"""
        result = reinhard_color_transfer(source, reference)

        # 支持强度调节
        if strength < 1.0:
            result = (result.astype(np.float32) * strength +
                     source.astype(np.float32) * (1 - strength)).astype(np.uint8)

        return result
