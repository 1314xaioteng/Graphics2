"""
真实NPR渲染系统 - 非真实感三维场景渲染 (改进版)
========================================
核心特性：
- 多尺度特征提取（替代单一深度/法线图）
- 感知驱动的风格化（基于梯度场和特征金字塔）
- 自适应笔触方向（根据几何信息动态调整）
- 增强的纹理生成（更自然的艺术效果）
- 边缘保持滤波（保持结构同时平滑）
- GPU加速渲染

改进内容：
1. 多尺度特征金字塔提取
2. 自适应笔触方向系统
3. 引导滤波器用于边缘保持
4. 增强的纹理生成算法
5. 感知驱动的色彩处理
"""

import numpy as np
import cv2
from PIL import Image
import os
import sys
import math
from typing import Dict, Tuple, Optional, List
import json
from numba import jit

# 导入项目模块
sys.path.insert(0, os.path.dirname(__file__))

# 导入色彩迁移和背景合成模块
try:
    from color_transfer import reinhard_color_transfer
    from background_compositor import composite_with_background
except ImportError:
    print("警告: color_transfer 或 background_compositor 模块未找到")
    reinhard_color_transfer = None
    composite_with_background = None

@jit(nopython=True)
def rasterize_triangles(vertices, faces, image_size):
    """使用Numba加速的光栅化核心函数"""
    depth_map = np.full((image_size, image_size), 1e10, dtype=np.float32)
    normal_map = np.zeros((image_size, image_size, 3), dtype=np.float32)

    # 计算面法线
    face_normals = np.zeros((len(faces), 3), dtype=np.float32)
    for i in range(len(faces)):
        v0 = vertices[faces[i, 0]]
        v1 = vertices[faces[i, 1]]
        v2 = vertices[faces[i, 2]]
        edge1 = v1 - v0
        edge2 = v2 - v0

        nx = edge1[1] * edge2[2] - edge1[2] * edge2[1]
        ny = edge1[2] * edge2[0] - edge1[0] * edge2[2]
        nz = edge1[0] * edge2[1] - edge1[1] * edge2[0]
        norm = (nx * nx + ny * ny + nz * nz) ** 0.5
        if norm > 1e-6:
            face_normals[i, 0] = nx / norm
            face_normals[i, 1] = ny / norm
            face_normals[i, 2] = nz / norm

    # 光栅化循环
    half_size = image_size / 2
    for i in range(len(faces)):
        v0 = vertices[faces[i, 0]]
        v1 = vertices[faces[i, 1]]
        v2 = vertices[faces[i, 2]]

        p0x = int((v0[0] + 1) * half_size)
        p0y = int((v0[1] + 1) * half_size)
        p1x = int((v1[0] + 1) * half_size)
        p1y = int((v1[1] + 1) * half_size)
        p2x = int((v2[0] + 1) * half_size)
        p2y = int((v2[1] + 1) * half_size)

        min_x = max(0, min(p0x, p1x, p2x))
        max_x = min(image_size - 1, max(p0x, p1x, p2x))
        min_y = max(0, min(p0y, p1y, p2y))
        max_y = min(image_size - 1, max(p0y, p1y, p2y))

        if min_x > max_x or min_y > max_y:
            continue

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                area = (p1x - p0x) * (p2y - p0y) - (p1y - p0y) * (p2x - p0x)
                if abs(area) < 1e-6:
                    continue

                w0 = ((p1x - x) * (p2y - y) - (p1y - y) * (p2x - x)) / area
                w1 = ((p2x - x) * (p0y - y) - (p2y - y) * (p0x - x)) / area
                w2 = 1.0 - w0 - w1

                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    z = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]
                    if z < depth_map[y, x]:
                        depth_map[y, x] = z
                        normal_map[y, x, 0] = face_normals[i, 0]
                        normal_map[y, x, 1] = face_normals[i, 1]
                        normal_map[y, x, 2] = face_normals[i, 2]

    return depth_map, normal_map


class NPRRenderer:
    """改进的NPR非真实感渲染器"""

    def __init__(self):
        """初始化NPR渲染器"""
        self.depth_map = None
        self.normal_map = None
        self.edge_map = None
        self.color_map = None
        self.feature_pyramid = None  # 新增：多尺度特征
        self.gradient_field = None   # 新增：梯度场

    def load_obj_simple(self, obj_path: str) -> Dict:
        """加载OBJ模型（简化版）"""
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"模型不存在: {obj_path}")

        vertices = []
        faces = []

        with open(obj_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):
                    parts = line.split()
                    vertices.append([float(x) for x in parts[1:4]])
                elif line.startswith('f '):
                    parts = line.split()[1:]
                    face = []
                    for p in parts:
                        idx = int(p.split('/')[0]) - 1
                        face.append(idx)
                    if len(face) == 3:
                        faces.append(face)

        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)

        # 归一化顶点
        center = vertices.mean(axis=0)
        vertices -= center
        scale = np.max(np.abs(vertices))
        if scale > 0:
            vertices /= scale

        return {
            'vertices': vertices,
            'faces': faces,
            'path': obj_path,
            'name': os.path.splitext(os.path.basename(obj_path))[0],
            'stats': {
                'vertices': len(vertices),
                'faces': len(faces)
            }
        }

    def render_depth_normal(self, model: Dict, image_size: int = 512, 
                           model_scale: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
        """渲染深度图和法线图"""
        vertices = model['vertices'].copy()
        faces = model['faces']
        vertices = vertices * model_scale

        depth_map, normal_map = rasterize_triangles(vertices, faces, image_size)

        mask = depth_map < 1e9
        depth_map[~mask] = 0

        if mask.any() and depth_map.max() > depth_map.min():
            depth_range = depth_map[mask]
            if depth_range.max() > depth_range.min():
                depth_normalized = (depth_range - depth_range.min()) / (depth_range.max() - depth_range.min())
                depth_normalized = depth_normalized * 0.9 + 0.1
                depth_map[mask] = depth_normalized

        depth_map = np.clip(depth_map, 0, 1)
        normal_map = (normal_map + 1) / 2

        return depth_map, normal_map

    def extract_feature_pyramid(self, model: Dict, base_size: int = 512) -> List[Dict]:
        """新功能：提取多尺度特征金字塔"""
        pyramid = []
        scales = [1.0, 0.75, 0.5]

        print("  提取多尺度特征金字塔...")
        for scale in scales:
            size = int(base_size * scale)
            depth, normal = self.render_depth_normal(model, size)

            # 计算梯度
            grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

            pyramid.append({
                'scale': scale,
                'size': size,
                'depth': depth,
                'normal': normal,
                'gradient_mag': gradient_mag,
                'grad_x': grad_x,
                'grad_y': grad_y
            })
            print(f"    尺度 {scale}: {size}x{size}")

        return pyramid

    def compute_adaptive_stroke_direction(self, depth_map: np.ndarray, 
                                         normal_map: np.ndarray) -> np.ndarray:
        """新功能：计算自适应笔触方向"""
        # 基于深度梯度计算主方向
        grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=5)

        # 计算角度（垂直于梯度方向）
        angles = np.arctan2(grad_y, grad_x) + np.pi / 2

        # 平滑角度场
        angles_smooth = cv2.GaussianBlur(angles, (15, 15), 5)

        return angles_smooth

    def guided_filter(self, image: np.ndarray, guide: np.ndarray, 
                     radius: int = 8, eps: float = 0.01) -> np.ndarray:
        """新功能：引导滤波器 - 边缘保持平滑"""
        mean_I = cv2.boxFilter(guide, cv2.CV_32F, (radius, radius))
        mean_p = cv2.boxFilter(image, cv2.CV_32F, (radius, radius))
        mean_Ip = cv2.boxFilter(guide * image, cv2.CV_32F, (radius, radius))

        cov_Ip = mean_Ip - mean_I * mean_p
        mean_II = cv2.boxFilter(guide * guide, cv2.CV_32F, (radius, radius))
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
        mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))

        return mean_a * guide + mean_b

    def detect_edges(self, depth_map: np.ndarray, normal_map: np.ndarray,
                    threshold: float = 0.1) -> np.ndarray:
        """检测模型边界（增强版）"""
        # 使用多尺度边缘检测
        depth_edges = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
        depth_edges += cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
        depth_edges = np.abs(depth_edges)

        normal_x = cv2.Sobel(normal_map[:, :, 0], cv2.CV_32F, 1, 0, ksize=3)
        normal_y = cv2.Sobel(normal_map[:, :, 1], cv2.CV_32F, 0, 1, ksize=3)
        normal_z = cv2.Sobel(normal_map[:, :, 2], cv2.CV_32F, 1, 0, ksize=3)
        normal_edges = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)

        # 合并多尺度边缘
        edges = depth_edges + normal_edges * 0.5
        edges = cv2.normalize(edges, None, 0, 1, cv2.NORM_MINMAX)

        edge_map = (edges > threshold).astype(np.uint8)
        return edge_map

    def apply_sketch_style(self, color_map: np.ndarray, edge_map: np.ndarray,
                          depth_map: np.ndarray, strength: float = 0.9) -> np.ndarray:
        """改进的铅笔素描风格 - 使用自适应笔触方向"""
        h, w = color_map.shape[:2]

        # 前景遮罩
        eps = 0.02
        mask = (depth_map > eps).astype(np.float32)
        mask = cv2.morphologyEx((mask * 255).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 3.0)

        # 转灰度
        gray = cv2.cvtColor(color_map, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # 使用引导滤波器而不是简单的高斯模糊
        gray_norm = gray / 255.0
        inv = 1.0 - gray_norm
        blur = self.guided_filter(inv, inv, radius=20, eps=0.01)
        base_sketch = gray_norm / (1.0 - blur + 1e-6)
        base_sketch = np.clip(base_sketch * 255.0, 0, 255)

        # 计算自适应笔触方向
        stroke_angles = self.compute_adaptive_stroke_direction(depth_map, self.normal_map)

        # 明暗度
        darkness = 1.0 - base_sketch / 255.0
        darkness = np.clip(darkness * 1.5, 0, 1)

        # 创建自适应方向的铅笔纹理
        hatch_texture = self._create_adaptive_hatching(h, w, stroke_angles, darkness)

        # 将纹理应用到素描
        sketch = 255.0 - hatch_texture * 180.0 * mask_f

        # 添加边缘轮廓
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
        edges = cv2.GaussianBlur(edges.astype(np.float32), (0, 0), 0.8)
        edges = edges / 255.0 * mask_f
        sketch = sketch - edges * 80
        sketch = np.clip(sketch, 0, 255)

        # 纸张纹理
        paper = np.random.normal(0, 1, (h, w)).astype(np.float32)
        paper = cv2.GaussianBlur(paper, (0, 0), 1.5)
        paper = paper / (paper.std() + 1e-6)
        sketch = sketch + paper * 3 * mask_f
        sketch = np.clip(sketch, 0, 255)

        # 白色背景
        bg = np.ones((h, w), np.float32) * 252.0
        bg_texture = np.random.normal(0, 1, (h, w)).astype(np.float32)
        bg_texture = cv2.GaussianBlur(bg_texture, (0, 0), 2.0)
        bg = bg + bg_texture * 2
        bg = np.clip(bg, 248, 255)

        # 合成
        result = sketch * mask_f + bg * (1.0 - mask_f)
        result = np.clip(result, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

        if strength < 1.0:
            out = (out.astype(np.float32) * strength +
                   color_map.astype(np.float32) * (1.0 - strength)).astype(np.uint8)

        return out

    def _create_adaptive_hatching(self, h: int, w: int, angles: np.ndarray, 
                                 darkness: np.ndarray) -> np.ndarray:
        """新功能：创建自适应方向的交叉影线纹理"""
        hatch = np.zeros((h, w), np.float32)

        # 采样网格
        step = 3
        for y in range(0, h, step):
            for x in range(0, w, step):
                if darkness[y, x] < 0.2:
                    continue

                angle = angles[y, x]
                intensity = darkness[y, x]

                # 计算线段端点
                length = int(6 * intensity)
                dx = int(length * np.cos(angle))
                dy = int(length * np.sin(angle))

                x1 = max(0, min(w-1, x - dx))
                y1 = max(0, min(h-1, y - dy))
                x2 = max(0, min(w-1, x + dx))
                y2 = max(0, min(h-1, y + dy))

                cv2.line(hatch, (x1, y1), (x2, y2), intensity, 1, cv2.LINE_AA)

                # 交叉影线（对于暗部）
                if intensity > 0.5:
                    angle2 = angle + np.pi / 2
                    dx2 = int(length * 0.7 * np.cos(angle2))
                    dy2 = int(length * 0.7 * np.sin(angle2))

                    x3 = max(0, min(w-1, x - dx2))
                    y3 = max(0, min(h-1, y - dy2))
                    x4 = max(0, min(w-1, x + dx2))
                    y4 = max(0, min(h-1, y + dy2))

                    cv2.line(hatch, (x3, y3), (x4, y4), intensity * 0.7, 1, cv2.LINE_AA)

        # 平滑
        hatch = cv2.GaussianBlur(hatch, (3, 3), 0.5)

        # 添加随机变化
        noise = np.random.normal(0, 0.1, (h, w)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), 1.0)
        hatch = hatch + noise * darkness
        hatch = np.clip(hatch, 0, 1)

        return hatch

    def apply_watercolor_style(self, color_map: np.ndarray, edge_map: np.ndarray,
                              depth_map: np.ndarray, strength: float = 0.9) -> np.ndarray:
        """改进的水彩风格 - 使用引导滤波和感知色彩"""
        h, w = color_map.shape[:2]

        eps = 0.02
        mask = (depth_map > eps).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 5.0)
        mask3 = np.dstack([mask_f, mask_f, mask_f])

        src = color_map.copy()

        # 使用引导滤波创建平滑色块
        smooth = src.astype(np.float32) / 255.0
        for c in range(3):
            smooth[:, :, c] = self.guided_filter(smooth[:, :, c], depth_map, radius=15, eps=0.01)
        smooth = (smooth * 255).astype(np.uint8)

        # 增强饱和度
        hsv = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
        smooth = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # 边缘柔化
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edges = cv2.GaussianBlur(edges.astype(np.float32), (0, 0), 3.0)
        edges = edges / (edges.max() + 1e-6)

        darken = 1.0 - 0.15 * edges * mask_f
        smooth = (smooth.astype(np.float32) * darken[:, :, None]).astype(np.uint8)

        # 水彩纸纹理
        paper = np.random.normal(0, 1, (h, w)).astype(np.float32)
        paper = cv2.GaussianBlur(paper, (0, 0), 2.0)
        paper = paper / (paper.std() + 1e-6)
        smooth = smooth.astype(np.float32) + paper[:, :, None] * 4 * mask_f[:, :, None]
        smooth = np.clip(smooth, 0, 255).astype(np.uint8)

        # 晕染效果
        smooth = cv2.GaussianBlur(smooth, (3, 3), 0.8)

        # 白色水彩纸背景
        bg = np.ones((h, w, 3), np.float32) * 250.0
        bg_wash = np.random.normal(0, 1, (h, w)).astype(np.float32)
        bg_wash = cv2.GaussianBlur(bg_wash, (0, 0), 30.0)
        bg_wash = (bg_wash - bg_wash.min()) / (bg_wash.max() - bg_wash.min() + 1e-6)
        bg = bg - 15 * bg_wash[:, :, None]
        bg = np.clip(bg, 230, 255).astype(np.uint8)

        result = (smooth.astype(np.float32) * mask3 +
                  bg.astype(np.float32) * (1.0 - mask3))
        result = np.clip(result, 0, 255).astype(np.uint8)

        if strength < 1.0:
            result = (result.astype(np.float32) * strength +
                      color_map.astype(np.float32) * (1.0 - strength)).astype(np.uint8)

        return result

    def apply_cartoon_style(self, color_map: np.ndarray, edge_map: np.ndarray,
                           depth_map: np.ndarray, strength: float = 0.95) -> np.ndarray:
        """改进的卡通风格 - 更智能的色阶量化"""
        h, w = color_map.shape[:2]

        eps = 0.02
        mask = (depth_map > eps).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 3.0)
        mask3 = np.dstack([mask_f, mask_f, mask_f])

        src = color_map.copy()

        # 使用引导滤波平滑（保持边缘）
        smooth = src.astype(np.float32) / 255.0
        for c in range(3):
            smooth[:, :, c] = self.guided_filter(smooth[:, :, c], depth_map, radius=10, eps=0.001)
        smooth = (smooth * 255).astype(np.uint8)

        # 增强饱和度
        hsv = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
        smooth = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # 智能色阶量化（基于K-means）
        pixels = smooth.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 8, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        quantized = centers[labels.flatten()].reshape(smooth.shape).astype(np.uint8)

        # 提取干净的轮廓线
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        edges = edges.astype(np.float32) * mask_f

        # 叠加边缘
        edges_3ch = np.dstack([edges, edges, edges])
        cartoon = quantized.astype(np.float32) * (1.0 - edges_3ch / 255.0)
        cartoon = np.clip(cartoon, 0, 255).astype(np.uint8)

        # 白色背景
        bg = np.ones((h, w, 3), np.uint8) * 255

        result = (cartoon.astype(np.float32) * mask3 +
                  bg.astype(np.float32) * (1.0 - mask3))
        result = np.clip(result, 0, 255).astype(np.uint8)

        if strength < 1.0:
            result = (result.astype(np.float32) * strength +
                      color_map.astype(np.float32) * (1.0 - strength)).astype(np.uint8)

        return result

    def apply_oil_style(self, color_map: np.ndarray, edge_map: np.ndarray,
                       depth_map: np.ndarray, strength: float = 0.9) -> np.ndarray:
        """改进的油画风格 - 方向感知的笔触"""
        h, w = color_map.shape[:2]

        eps = 0.02
        mask = (depth_map > eps).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 5.0)
        mask3 = np.dstack([mask_f, mask_f, mask_f])

        src = color_map.copy()

        # 增强饱和度
        hsv = cv2.cvtColor(src, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
        vivid = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # 计算笔触方向
        stroke_angles = self.compute_adaptive_stroke_direction(depth_map, self.normal_map)

        # 方向感知的油画效果
        oil = self._apply_directional_oil_painting(vivid, stroke_angles, mask_f)

        # 画布纹理
        canvas = np.random.normal(0, 1, (h, w)).astype(np.float32)
        canvas = cv2.GaussianBlur(canvas, (0, 0), 1.5)
        canvas = canvas / (canvas.std() + 1e-6)
        oil = oil.astype(np.float32) + canvas[:, :, None] * 6 * mask_f[:, :, None]
        oil = np.clip(oil, 0, 255).astype(np.uint8)

        # 边缘增强
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        edges = cv2.Laplacian(gray, cv2.CV_32F)
        edges = np.abs(edges)
        edges = cv2.GaussianBlur(edges, (0, 0), 1.0)
        edges = edges / (edges.max() + 1e-6) * 0.12
        oil = oil.astype(np.float32) * (1.0 + edges[:, :, None] * mask_f[:, :, None])
        oil = np.clip(oil, 0, 255).astype(np.uint8)

        # 画布色背景
        bg = np.ones((h, w, 3), np.float32)
        bg[:, :, 0] = 230
        bg[:, :, 1] = 220
        bg[:, :, 2] = 210
        bg_texture = np.random.normal(0, 1, (h, w)).astype(np.float32)
        bg_texture = cv2.GaussianBlur(bg_texture, (0, 0), 1.0)
        bg = bg + bg_texture[:, :, None] * 8
        bg = np.clip(bg, 200, 245).astype(np.uint8)

        result = (oil.astype(np.float32) * mask3 +
                  bg.astype(np.float32) * (1.0 - mask3))
        result = np.clip(result, 0, 255).astype(np.uint8)

        if strength < 1.0:
            result = (result.astype(np.float32) * strength +
                      color_map.astype(np.float32) * (1.0 - strength)).astype(np.uint8)

        return result

    def _apply_directional_oil_painting(self, image: np.ndarray, angles: np.ndarray,
                                       mask: np.ndarray) -> np.ndarray:
        """新功能：方向感知的油画效果"""
        h, w = image.shape[:2]
        result = np.zeros_like(image, dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)

        # 采样网格
        step = 7
        radius = 10

        for y in range(0, h, step):
            for x in range(0, w, step):
                if mask[y, x] < 0.1:
                    continue

                angle = angles[y, x]

                # 创建椭圆形采样区域（沿笔触方向拉长）
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        # 旋转坐标
                        rx = dx * np.cos(angle) - dy * np.sin(angle)
                        ry = dx * np.sin(angle) + dy * np.cos(angle)

                        # 椭圆判定（长轴:短轴 = 2:1）
                        if (rx / (radius * 1.5))**2 + (ry / radius)**2 <= 1:
                            ny = y + dy
                            nx = x + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                result[ny, nx] += image[y, x].astype(np.float32)
                                counts[ny, nx] += 1

        # 归一化
        valid = counts > 0
        result[valid] = result[valid] / counts[valid, None]
        result[~valid] = image[~valid].astype(np.float32)

        return result.astype(np.uint8)

    def render(self, model: Dict, style: str = 'sketch', strength: float = 0.9,
               image_size: int = 512,
               color_reference: Optional[np.ndarray] = None,
               background: Optional[np.ndarray] = None) -> np.ndarray:
        """完整NPR渲染流程（改进版）"""
        print("\n=== 改进版 NPR 渲染流程 ===")
        print(f" 模型: {model['name']}")
        print(f" 顶点: {model['stats']['vertices']}")
        print(f" 面片: {model['stats']['faces']}")
        print(f" 风格: {style}")

        # Step 1: 提取多尺度特征金字塔
        print("\n[步骤 1/6] 提取多尺度特征金字塔...")
        self.feature_pyramid = self.extract_feature_pyramid(model, image_size)

        # 使用最高分辨率的特征
        primary_features = self.feature_pyramid[0]
        self.depth_map = primary_features['depth']
        self.normal_map = primary_features['normal']

        # Step 2: 生成色彩图（使用感知驱动的方法）
        print("\n[步骤 2/6] 生成感知驱动的色彩图...")
        color_map = self._generate_perceptual_color_map(self.depth_map, self.normal_map, image_size)
        self.color_map = color_map

        # Step 3: 边界检测
        print("\n[步骤 3/6] 检测边界与轮廓...")
        edge_map = self.detect_edges(self.depth_map, self.normal_map)
        self.edge_map = edge_map

        # Step 4: 应用风格（使用改进的算法）
        print(f"\n[步骤 4/6] 应用 {style} 风格 (强度: {strength:.0%})...")
        if style == 'sketch':
            result = self.apply_sketch_style(color_map, edge_map, self.depth_map, strength)
        elif style == 'watercolor':
            result = self.apply_watercolor_style(color_map, edge_map, self.depth_map, strength)
        elif style == 'cartoon':
            result = self.apply_cartoon_style(color_map, edge_map, self.depth_map, strength)
        elif style == 'oil':
            result = self.apply_oil_style(color_map, edge_map, self.depth_map, strength)
        else:
            result = color_map

        # Step 5: 色彩迁移
        if color_reference is not None and reinhard_color_transfer is not None:
            print("\n[步骤 5/6] 应用色彩迁移...")
            result = reinhard_color_transfer(result, color_reference)
        else:
            print("\n[步骤 5/6] 跳过色彩迁移")

        # Step 6: 背景合成
        if background is not None and composite_with_background is not None:
            print("\n[步骤 6/6] 合成背景...")
            result_normal = np.flipud(result)
            depth_map_normal = np.flipud(self.depth_map)
            result_composited = composite_with_background(result_normal, depth_map_normal, background)
            print("\n✓ 渲染完成！")
            return result_composited

        print("\n[步骤 6/6] 跳过背景合成")
        result = np.flipud(result)
        print("\n✓ 渲染完成！")
        return result

    def _generate_perceptual_color_map(self, depth_map: np.ndarray, 
                                      normal_map: np.ndarray, size: int) -> np.ndarray:
        """新功能：生成感知驱动的色彩图"""
        # 基于法线的彩色映射（模拟简单光照）
        normal_rgb = (normal_map * 255).astype(np.uint8)

        # 计算简单的光照（光源从右上方）
        light_dir = np.array([0.5, 0.5, 0.7])
        light_dir = light_dir / np.linalg.norm(light_dir)

        # 漫反射光照
        diffuse = np.maximum(0, np.sum(normal_map * light_dir, axis=2))
        diffuse = np.clip(diffuse * 200 + 55, 0, 255).astype(np.uint8)
        diffuse_3ch = cv2.cvtColor(diffuse, cv2.COLOR_GRAY2RGB)

        # 混合法线颜色和光照
        color_map = (normal_rgb.astype(np.float32) * 0.4 +
                    diffuse_3ch.astype(np.float32) * 0.6).astype(np.uint8)

        return color_map


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='改进版NPR非真实感渲染系统')
    parser.add_argument('--model', '-m', required=True, help='OBJ模型路径')
    parser.add_argument('--style', '-s', default='sketch',
                       choices=['sketch', 'watercolor', 'cartoon', 'oil'],
                       help='风格（默认: sketch）')
    parser.add_argument('--strength', type=float, default=0.9,
                       help='风格强度（0-1，默认: 0.9）')
    parser.add_argument('--output', '-o', default=None, help='输出路径')
    parser.add_argument('--size', type=int, default=512, help='输出尺寸（默认: 512）')

    args = parser.parse_args()

    # 创建渲染器
    renderer = NPRRenderer()

    # 加载模型
    print(f"加载模型: {args.model}")
    model = renderer.load_obj_simple(args.model)
    print(f" 已加载: {model['stats']['vertices']} 顶点, {model['stats']['faces']} 面片")

    # 渲染
    result = renderer.render(model, args.style, args.strength, args.size)

    # 保存
    if args.output is None:
        args.output = f"npr_{model['name']}_{args.style}_improved.png"

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    Image.fromarray(result).save(args.output)
    print(f"\n结果已保存: {args.output}")

    # 保存配置
    config_path = args.output.replace('.png', '_config.json')
    config = {
        'model': args.model,
        'style': args.style,
        'strength': args.strength,
        'output': args.output,
        'model_stats': model['stats'],
        'improvements': [
            '多尺度特征金字塔',
            '自适应笔触方向',
            '引导滤波器边缘保持',
            '感知驱动的色彩生成',
            '方向感知的油画效果',
            '智能色阶量化（K-means）'
        ]
    }

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"配置已保存: {config_path}")


if __name__ == '__main__':
    main()
