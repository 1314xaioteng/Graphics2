"""
背景合成模块 - 按推荐方案实现
================================

深度图 Alpha 混合 + 高斯羽化（核心15行）
简单、高效、效果好
"""

import numpy as np
import cv2


def composite_with_background(foreground, depth_map, background):
    """
    将渲染结果合成到背景图上（推荐方案 - 核心15行）

    Args:
        foreground: 前景图像 (H, W, 3) RGB
        depth_map: 深度图 (H, W) - 归一化到[0,1]
        background: 背景图像 RGB

    Returns:
        合成后的图像
    """
    h, w = depth_map.shape
    bg_resized = cv2.resize(background, (w, h))

    # 生成Alpha遮罩（前景区域应该完全不透明）
    # 使用更低的阈值确保捕获所有前景像素
    alpha = (depth_map > 0.001).astype(np.float32)

    # 不做任何模糊，保持硬边界
    # alpha值要么是1.0（前景）要么是0（背景）

    alpha = np.expand_dims(alpha, axis=2)

    # 混合
    result = foreground * alpha + bg_resized * (1 - alpha)

    return result.astype(np.uint8)


# 兼容旧接口的类包装
class BackgroundCompositor:
    @staticmethod
    def composite_simple(foreground, depth_map, background, feather_size=5):
        """简单背景合成"""
        h, w = depth_map.shape
        bg_resized = cv2.resize(background, (w, h))

        # 生成Alpha遮罩
        alpha = (depth_map > 0).astype(np.float32)

        # 羽化（使用可调节的核大小）
        if feather_size > 0:
            kernel_size = feather_size * 2 + 1
            alpha = cv2.GaussianBlur(alpha, (kernel_size, kernel_size), 0)

        alpha = np.expand_dims(alpha, axis=2)

        # 混合
        result = foreground * alpha + bg_resized * (1 - alpha)
        return result.astype(np.uint8)

    @staticmethod
    def composite_with_dof(foreground, depth_map, background,
                          focus_depth=0.5, blur_strength=10, feather_size=5):
        """带景深效果的背景合成"""
        h, w = depth_map.shape
        bg_resized = cv2.resize(background, (w, h))

        # 对背景应用模糊（简化版景深）
        if blur_strength > 1:
            kernel_size = blur_strength * 2 + 1
            bg_resized = cv2.GaussianBlur(bg_resized, (kernel_size, kernel_size), 0)

        # 生成Alpha遮罩并羽化
        alpha = (depth_map > 0).astype(np.float32)
        if feather_size > 0:
            kernel_size = feather_size * 2 + 1
            alpha = cv2.GaussianBlur(alpha, (kernel_size, kernel_size), 0)

        alpha = np.expand_dims(alpha, axis=2)

        # 混合
        result = foreground * alpha + bg_resized * (1 - alpha)
        return result.astype(np.uint8)
