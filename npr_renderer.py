"""
真实NPR渲染系统 - 非真实感三维场景渲染
========================================

核心特性：
- 基于几何信息（深度、法线、边界）的真正NPR渲染
- 四种手绘风格（素描、水彩、卡通、油画）
- 参数可调的风格强度控制
- GPU加速渲染

实现原理：
1. 加载3D模型
2. 渲染深度图和法线图
3. 边界检测（轮廓线提取）
4. 根据风格应用NPR效果
5. 叠加几何约束保留3D结构

"""

import numpy as np
import cv2
from PIL import Image
import os
import sys
from typing import Dict, Tuple, Optional
import json
from numba import jit

# 导入项目模块
sys.path.insert(0, os.path.dirname(__file__))


@jit(nopython=True)
def rasterize_triangles(vertices, faces, image_size):
    """使用Numba加速的光栅化核心函数"""
    depth_map = np.zeros((image_size, image_size), dtype=np.float32)
    # 初始化法线图，默认指向Z轴
    normal_map = np.zeros((image_size, image_size, 3), dtype=np.float32)

    # 计算面法线
    face_normals = np.zeros((len(faces), 3), dtype=np.float32)
    for i in range(len(faces)):
        v0 = vertices[faces[i, 0]]
        v1 = vertices[faces[i, 1]]
        v2 = vertices[faces[i, 2]]

        edge1 = v1 - v0
        edge2 = v2 - v0

        # 手动计算叉乘
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

        # 投影到屏幕坐标
        p0x = int((v0[0] + 1) * half_size)
        p0y = int((v0[1] + 1) * half_size)
        p1x = int((v1[0] + 1) * half_size)
        p1y = int((v1[1] + 1) * half_size)
        p2x = int((v2[0] + 1) * half_size)
        p2y = int((v2[1] + 1) * half_size)

        # 包围盒
        min_x = max(0, min(p0x, p1x, p2x))
        max_x = min(image_size - 1, max(p0x, p1x, p2x))
        min_y = max(0, min(p0y, p1y, p2y))
        max_y = min(image_size - 1, max(p0y, p1y, p2y))

        if min_x > max_x or min_y > max_y:
            continue

        # 遍历包围盒内的像素
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                # 计算重心坐标 (使用叉乘原理)
                area = (p1x - p0x) * (p2y - p0y) - (p1y - p0y) * (p2x - p0x)
                if abs(area) < 1e-6:
                    continue

                w0 = ((p1x - x) * (p2y - y) - (p1y - y) * (p2x - x)) / area
                w1 = ((p2x - x) * (p0y - y) - (p2y - y) * (p0x - x)) / area
                w2 = 1.0 - w0 - w1

                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    z = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]

                    if z > depth_map[y, x] or depth_map[y, x] == 0:
                        depth_map[y, x] = z
                        normal_map[y, x, 0] = face_normals[i, 0]
                        normal_map[y, x, 1] = face_normals[i, 1]
                        normal_map[y, x, 2] = face_normals[i, 2]

    return depth_map, normal_map

class NPRRenderer:
    """真实的NPR非真实感渲染器"""

    def __init__(self):
        """初始化NPR渲染器"""
        self.depth_map = None
        self.normal_map = None
        self.edge_map = None
        self.color_map = None

    def load_obj_simple(self, obj_path: str) -> Dict:
        """
        加载OBJ模型（简化版）

        Args:
            obj_path: OBJ文件路径

        Returns:
            模型数据
        """
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

        # 归一化顶点到[-1, 1]
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

    def render_depth_normal(self, model: Dict, image_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """
        渲染深度图和法线图 (使用 Numba 加速版)
        """
        vertices = model['vertices']
        faces = model['faces']

        # 调用上面定义的 Numba 加速函数
        # 注意：第一次调用时会进行编译，可能需要1-2秒，之后会非常快
        depth_map, normal_map = rasterize_triangles(vertices, faces, image_size)

        # 归一化深度
        if depth_map.max() > depth_map.min():
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

        # 归一化法线到[0, 1]
        normal_map = (normal_map + 1) / 2

        return depth_map, normal_map

    def detect_edges(self, depth_map: np.ndarray, normal_map: np.ndarray,
                     threshold: float = 0.1) -> np.ndarray:
        """
        检测模型边界（轮廓线）

        Args:
            depth_map: 深度图
            normal_map: 法线图
            threshold: 边界检测阈值

        Returns:
            边界图
        """
        # Sobel边界检测
        depth_edges = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
        depth_edges += cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
        depth_edges = np.abs(depth_edges)

        # 法线边界检测
        normal_x = cv2.Sobel(normal_map[:, :, 0], cv2.CV_32F, 1, 0, ksize=3)
        normal_y = cv2.Sobel(normal_map[:, :, 1], cv2.CV_32F, 0, 1, ksize=3)
        normal_z = cv2.Sobel(normal_map[:, :, 2], cv2.CV_32F, 1, 0, ksize=3)
        normal_edges = np.sqrt(normal_x ** 2 + normal_y ** 2 + normal_z ** 2)

        # 合并边界
        edges = depth_edges + normal_edges * 0.5
        edges = cv2.normalize(edges, None, 0, 1, cv2.NORM_MINMAX)

        # 二值化
        edge_map = (edges > threshold).astype(np.uint8)

        return edge_map

    def apply_sketch_style(self, color_map: np.ndarray, edge_map: np.ndarray,
                           depth_map: np.ndarray, strength: float = 0.9) -> np.ndarray:
        """
        应用改进版素描风格
        改进点：增加铅笔纸张纹理、利用深度图增强阴影涂抹感
        """
        # --- 1. 基础转换 ---
        # 转灰度并反转（变成白底黑线条的逻辑）
        gray = cv2.cvtColor(color_map, cv2.COLOR_RGB2GRAY)
        inv_gray = 255 - gray

        # --- 2. 模拟铅笔纹理 (Pencil Texture) ---
        # 生成高斯噪点模拟纸张颗粒
        h, w = gray.shape
        noise = np.zeros((h, w), dtype=np.uint8)
        cv2.randn(noise, 0, 40)  # 噪点强度

        # 将噪点叠加到画面上 (使用混合模式)
        # 这一步让纯平滑的灰色变成粗糙的铅笔质感
        textured = cv2.addWeighted(inv_gray, 0.8, noise, 0.2, 0)

        # --- 3. 增强阴影 (Shading via Depth) ---
        # 素描中，离得远或暗的地方画家会涂得更黑
        # 只有在有深度变化的地方才加重笔触
        shading = (depth_map * 255).astype(np.uint8)
        # 对深度图做阈值处理，只保留暗部作为"重阴影区"
        _, dark_areas = cv2.threshold(shading, 100, 255, cv2.THRESH_BINARY_INV)
        # 模糊一下阴影边缘，模拟大拇指涂抹的效果
        dark_areas = cv2.GaussianBlur(dark_areas, (15, 15), 0)

        # --- 4. 融合 ---
        # 基础图像（反转回来变成白底）
        base_sketch = 255 - textured

        # 叠加轮廓线（让轮廓更黑、更实）
        # edge_map: 1是线，0是背景
        edges = (1 - edge_map) * 255

        # 混合：底图 * 轮廓 * 阴影
        # 这是一个乘法过程：任何一个为黑(0)，结果就为黑
        final_sketch = cv2.multiply(base_sketch.astype(np.float32) / 255.0,
                                    edges.astype(np.float32) / 255.0)

        # 再次叠加深度阴影（让立体感更强）
        shadow_factor = 1.0 - (dark_areas.astype(np.float32) / 255.0 * 0.3)
        final_sketch = final_sketch * shadow_factor

        final_sketch = np.clip(final_sketch * 255, 0, 255).astype(np.uint8)

        # 转回RGB
        result_rgb = cv2.cvtColor(final_sketch, cv2.COLOR_GRAY2RGB)

        # 根据强度混合原图（通常素描不需要混合原图，但保留此接口）
        # 如果 strength < 1，会透出一点点原色，像"彩铅"
        if strength < 1.0:
            result = (result_rgb.astype(np.float32) * strength +
                      color_map.astype(np.float32) * (1 - strength)).astype(np.uint8)
            return result

        return result_rgb

    def apply_watercolor_style(self, color_map: np.ndarray, edge_map: np.ndarray,
                               depth_map: np.ndarray, strength: float = 0.7) -> np.ndarray:
        """
        应用改进版水彩风格
        改进点：更强的晕染流动感、湿边效果、色彩透亮
        """
        # --- 1. 颜色抽象 (Abstraction) ---
        # 使用金字塔均值漂移滤波，这是模拟水彩最经典的方法
        # 它能抹平纹理，但保留色彩边界，产生"一滩一滩"的水渍感
        # 注意：这个函数比较慢，如果卡顿，可以改回 bilateralFilter
        # sp: 空间窗半径, sr: 色彩窗半径
        try:
            painted = cv2.pyrMeanShiftFiltering(color_map, sp=10, sr=40)
        except:
            # 降级方案
            painted = cv2.bilateralFilter(color_map, 9, 75, 75)

        # --- 2. 模拟湿边与加深 (Wet Edges) ---
        # 水彩干燥时边缘会变深。我们利用边缘检测来模拟这个效果。
        # 对边缘进行模糊，产生晕染开的感觉
        edge_soft = cv2.GaussianBlur(edge_map.astype(np.float32), (5, 5), 0)
        # 反转：边缘处数值小（变暗），非边缘处为1
        edge_overlay = 1.0 - (edge_soft * 0.3)  # 0.3是加深程度

        # 叠加到图像上
        watery = painted.astype(np.float32) * np.stack([edge_overlay] * 3, axis=2)
        watery = np.clip(watery, 0, 255).astype(np.uint8)

        # --- 3. 提升通透感 (Transparency) ---
        # 水彩是透明的，我们稍微提高一点亮度和对比度
        lab = cv2.cvtColor(watery, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        # 限制对比度自适应直方图均衡化 (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        watery = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

        # --- 4. 纸张纹理叠加 (Paper Grain) ---
        # 类似素描，加一点点噪点模拟水彩纸的坑洼
        h, w, c = color_map.shape
        noise = np.zeros((h, w), dtype=np.uint8)
        cv2.randn(noise, 0, 10)  # 噪点很弱，隐约可见即可
        noise_rgb = cv2.cvtColor(noise, cv2.COLOR_GRAY2RGB)

        # 减去噪点（模拟纸张凹下去的地方颜料沉积）
        final = cv2.subtract(watery, noise_rgb)

        # --- 5. 混合 ---
        result = (final.astype(np.float32) * strength +
                  color_map.astype(np.float32) * (1 - strength)).astype(np.uint8)

        return result

    def apply_cartoon_style(self, color_map: np.ndarray, edge_map: np.ndarray,
                            depth_map: np.ndarray, strength: float = 0.95) -> np.ndarray:
        """
        应用改进版卡通风格 (Cel Shading)
        改进点：增加色阶量化（色块感）、提升饱和度、锐化轮廓线
        """
        # --- 1. 双边滤波 (平滑但保留边缘) ---
        # 次数越多，画面越像"色块"，杂色越少
        # 为了性能，迭代2次即可，d=9 是邻域直径
        smooth = color_map
        for _ in range(2):
            smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=75, sigmaSpace=75)

        # --- 2. 提升饱和度 (Vibrancy) ---
        # 动漫风格通常色彩比较鲜艳
        hsv = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)  # 饱和度增加30%
        # 稍微提升亮度，让画面更明快
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
        vivid = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # --- 3. 色阶量化 (Color Quantization / Cel Shading) ---
        # 将连续的颜色变成离散的色阶（例如每个通道只保留4-8个值）
        # 這是卡通渲染的核心：消灭渐变
        n_colors = 8  # 每个通道的色阶数量
        div = 256 // n_colors
        quantized = (vivid // div) * div + div // 2

        # --- 4. 处理轮廓线 (Outlines) ---
        # 对输入的几何边界图进行加粗和二值化
        # 膨胀边界，让线条稍微粗一点，更有漫画感
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thick_edges = cv2.dilate(edge_map, kernel, iterations=1)

        # 反转边界：线条为0 (黑)，背景为1 (白)
        edge_mask = (1 - thick_edges).astype(np.float32)
        # 将单通道扩展为3通道
        edge_mask_3ch = np.stack([edge_mask, edge_mask, edge_mask], axis=2)

        # --- 5. 组合 ---
        # 直接相乘：线条处变全黑，非线条处保留原色
        cartoon = quantized.astype(np.float32) * edge_mask_3ch
        cartoon = np.clip(cartoon, 0, 255).astype(np.uint8)

        # --- 6. 最终混合 ---
        result = (cartoon.astype(np.float32) * strength +
                  color_map.astype(np.float32) * (1 - strength)).astype(np.uint8)

        return result

    def apply_oil_style(self, color_map: np.ndarray, edge_map: np.ndarray,
                        depth_map: np.ndarray, strength: float = 0.85) -> np.ndarray:
        """
        应用改进版油画风格
        改进点：增加色彩鲜艳度、更好的笔触涂抹感、叠加画布纹理
        """
        # --- 1. 提升色彩饱和度 (Vivid Color) ---
        # 油画颜料通常比较鲜艳，转换到HSV空间提升S分量
        hsv = cv2.cvtColor(color_map, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.4  # 饱和度增加40%
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        vivid = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # --- 2. 模拟油画笔触 (Paint Strokes) ---
        # 使用保边滤波器 (EPF) 产生涂抹感，比单纯的高斯模糊更像笔触
        # flags=1 是 RECURS_FILTER，速度快且效果好
        # sigma_s 控制邻域大小(笔触大小)，sigma_r 控制色彩差异阈值
        painted = cv2.edgePreservingFilter(vivid, flags=1, sigma_s=60, sigma_r=0.4)

        # 如果觉得不够抽象，可以再叠一层轻微的量化（可选）
        # painted = (painted // 16) * 16

        # --- 3. 生成画布纹理 (Canvas Texture) ---
        # 生成一种类似布料的噪点
        h, w, c = color_map.shape
        noise = np.zeros((h, w), dtype=np.uint8)
        cv2.randn(noise, 128, 30)  # 均值128，标准差30

        # 将噪点稍微模糊，模拟画布纹理的柔和感
        canvas_texture = cv2.GaussianBlur(noise, (3, 3), 0)

        # 将纹理转为3通道并归一化到 [0.9, 1.1] 范围以便叠加
        canvas_overlay = cv2.cvtColor(canvas_texture, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        canvas_overlay = (canvas_overlay - 0.5) * 0.2 + 1.0  # 调整纹理强度

        # --- 4. 融合 ---
        # 叠加纹理
        result_float = painted.astype(np.float32) * canvas_overlay
        result = np.clip(result_float, 0, 255).astype(np.uint8)

        # 最终根据 strength 混合原图
        # 注意：这里我们混合的是"增强过色彩的原图"(vivid)，这样即使强度低也不失真
        final = (result.astype(np.float32) * strength +
                 vivid.astype(np.float32) * (1 - strength)).astype(np.uint8)

        return final

    def render(self, model: Dict, style: str = 'sketch', strength: float = 0.9,
               image_size: int = 512) -> np.ndarray:
        """
        完整NPR渲染流程

        Args:
            model: 模型数据
            style: 风格（sketch/watercolor/cartoon/oil）
            strength: 风格强度（0-1）
            image_size: 输出尺寸

        Returns:
            渲染结果
        """
        print("\nNPR Rendering Pipeline")
        print(f"  Model: {model['name']}")
        print(f"  Vertices: {model['stats']['vertices']}")
        print(f"  Faces: {model['stats']['faces']}")

        # Step 1: 渲染深度和法线
        print("\n  Step 1: Rendering depth & normal maps...")
        depth_map, normal_map = self.render_depth_normal(model, image_size)
        self.depth_map = depth_map
        self.normal_map = normal_map
        print(f"    Depth map: {depth_map.shape}")
        print(f"    Normal map: {normal_map.shape}")

        # Step 2: 生成色彩图（基于深度和法线）
        print("\n  Step 2: Generating color map...")
        color_map = self._generate_color_map(depth_map, normal_map, image_size)
        self.color_map = color_map
        print(f"    Color map: {color_map.shape}")

        # Step 3: 边界检测
        print("\n  Step 3: Detecting edges & outlines...")
        edge_map = self.detect_edges(depth_map, normal_map)
        self.edge_map = edge_map
        print(f"    Edge map: {edge_map.shape}")

        # Step 4: 应用风格
        print(f"\n  Step 4: Applying {style} style...")
        if style == 'sketch':
            result = self.apply_sketch_style(color_map, edge_map, depth_map, strength)
        elif style == 'watercolor':
            result = self.apply_watercolor_style(color_map, edge_map, depth_map, strength)
        elif style == 'cartoon':
            result = self.apply_cartoon_style(color_map, edge_map, depth_map, strength)
        elif style == 'oil':
            result = self.apply_oil_style(color_map, edge_map, depth_map, strength)
        else:
            result = color_map

        print(f"    {style.capitalize()} style applied (strength: {strength:.1%})")
        result = np.flipud(result)
        return result

    def _generate_color_map(self, depth_map: np.ndarray, normal_map: np.ndarray,
                            size: int) -> np.ndarray:
        """生成色彩图（基于深度和法线）"""
        # 转换法线为[0, 255]范围
        normal_rgb = (normal_map * 255).astype(np.uint8)

        # 基于深度的灰度
        depth_gray = (depth_map * 255).astype(np.uint8)
        depth_3ch = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2RGB)

        # 混合
        color_map = (normal_rgb.astype(np.float32) * 0.6 +
                     depth_3ch.astype(np.float32) * 0.4).astype(np.uint8)

        return color_map

    @staticmethod
    def _triangle_area(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
        """计算三角形面积"""
        return abs((p1[0] - p0[0]) * (p2[1] - p0[1]) -
                   (p2[0] - p0[0]) * (p1[1] - p0[1])) / 2.0


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='NPR非真实感渲染系统')
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
    print(f"Loading model: {args.model}")
    model = renderer.load_obj_simple(args.model)
    print(f"   Loaded: {model['stats']['vertices']} vertices, {model['stats']['faces']} faces")

    # 渲染
    result = renderer.render(model, args.style, args.strength, args.size)

    # 保存
    if args.output is None:
        args.output = f"npr_{model['name']}_{args.style}.png"

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    Image.fromarray(result).save(args.output)
    print(f"\nResult saved: {args.output}")

    # 保存配置
    config_path = args.output.replace('.png', '_config.json')
    config = {
        'model': args.model,
        'style': args.style,
        'strength': args.strength,
        'output': args.output,
        'model_stats': model['stats']
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {config_path}")


if __name__ == '__main__':
    main()