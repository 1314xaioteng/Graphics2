# 实NPR渲染系统 - 非真实感三维场景渲染

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
        """
        专业铅笔素描风格 - 手绘仿真版 (Bio-mimetic Pencil Sketch)
        
        模拟真实绘画过程：
        1. 纸张纹理作为基底 (Paper Grain)
        2. 智能影调分层 (Tone Separation)
        3. 抖动排线生成 (Stochastic Hatching with Wobble)
        4. 纸笔交互合成 (Graphite-Paper Interaction)
        """
        h, w = color_map.shape[:2]
        
        # --- 内部辅助：生成具有手绘感的抖动排线 ---
        def generate_wobbly_hatch(angle_deg, spacing, wobble_scale=2.0, layer_seed=0):
            np.random.seed(layer_seed)
            # 1. 绘制超大画布以支持旋转裁剪
            diag = int(np.sqrt(h**2 + w**2)) + 100
            canvas = np.zeros((diag, diag), dtype=np.float32)
            
            # 2. 模拟手绘的不规则排线
            pos = -diag // 2
            while pos < diag // 2:
                # 间距随机扰动 (模拟手部肌肉记忆的误差)
                gap = spacing + np.random.uniform(-1.5, 2.0)
                pos += gap
                
                # 线条属性随机
                thickness = 1
                if np.random.random() > 0.7: thickness = 2 # 偶尔用力
                intensity = np.random.uniform(0.6, 1.0) #以此模拟墨色深浅
                
                # 绘制直线 (稍微延伸出画布以防边缘空白)
                p1 = (int(pos + diag//2), 0)
                p2 = (int(pos + diag//2), diag)
                cv2.line(canvas, p1, p2, intensity, thickness, cv2.LINE_AA)
            
            # 3. 旋转排线
            center = (diag // 2, diag // 2)
            M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            rotated = cv2.warpAffine(canvas, M, (diag, diag))
            
            # 4. 裁剪回原图尺寸
            start_x = (diag - w) // 2
            start_y = (diag - h) // 2
            pattern = rotated[start_y:start_y+h, start_x:start_x+w]
            
            # 尺寸安全检查
            if pattern.shape[:2] != (h, w):
                pattern = cv2.resize(pattern, (w, h))
                
            # 5. 施加“手抖”扭曲 (Domain Distortion)
            # 这比直接画曲线更像手绘：它是整体趋势的微小摆动
            noise_res = 40  # 噪声网格大小
            flow_grid = np.random.uniform(-1, 1, (h//noise_res+2, w//noise_res+2, 2)).astype(np.float32)
            flow = cv2.resize(flow_grid, (w, h), interpolation=cv2.INTER_CUBIC) * wobble_scale
            
            grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (grid_x + flow[:,:,0]).astype(np.float32)
            map_y = (grid_y + flow[:,:,1]).astype(np.float32)
            
            # 重映射实现抖动
            wobbly_line = cv2.remap(pattern, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            
            return wobbly_line

        # ========== 1. 构建纸张基底 ==========
        # 生成高频纸纹噪声
        noise = np.random.normal(0, 1, (h, w)).astype(np.float32)
        paper_grain = cv2.GaussianBlur(noise, (0, 0), 0.7)
        # 纸纹影响蒙版：凹陷处(值较小)容易积墨，凸起处(值较大)容易留白
        paper_texture = 1.0 + paper_grain * 0.12

        # ========== 2. 影调映射 (S-Curve Tone Mapping) ==========
        gray = cv2.cvtColor(color_map, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        # 增强对比度，模拟素描对灰度的概括
        tone = 1.0 / (1.0 + np.exp(-9.0 * (gray - 0.5)))
        
        # 分层蒙版 (Layer Masks) - 使用软边缘
        # 亮调 (Hatching Layer 1)
        mask_light = (tone <= 0.85).astype(np.float32)
        mask_light = cv2.GaussianBlur(mask_light, (15, 15), 5.0)
        
        # 中调 (Hatching Layer 2)
        mask_mid = (tone <= 0.60).astype(np.float32)
        mask_mid = cv2.GaussianBlur(mask_mid, (11, 11), 3.0)
        
        # 暗部 (Hatching Layer 3)
        mask_dark = (tone <= 0.35).astype(np.float32)
        mask_dark = cv2.GaussianBlur(mask_dark, (7, 7), 2.0)
        
        # 闭塞点 (Occlusion / Scribble)
        mask_deep = (tone <= 0.15).astype(np.float32)
        
        # ========== 3. 生成排线层 (Generating Strokes) ==========
        # 层1: 稀疏单向排线 (45度) - 基础调子
        hatch1 = generate_wobbly_hatch(45, spacing=7, wobble_scale=2.0, layer_seed=1)
        
        # 层2: 稍密的同向复笔 (40度) - 增加厚度
        hatch2 = generate_wobbly_hatch(40, spacing=6, wobble_scale=2.5, layer_seed=2)
        
        # 层3: 交叉排线 (-60度) - 塑造体积
        hatch3 = generate_wobbly_hatch(-60, spacing=6, wobble_scale=2.0, layer_seed=3)
        
        # 层4: 横向乱序排线 (90度) - 压暗
        hatch4 = generate_wobbly_hatch(90, spacing=4, wobble_scale=3.0, layer_seed=4)
        
        # ========== 4. 纸笔交互合成 (Compositing) ==========
        # 初始化画布为纸白
        canvas = np.ones((h, w), dtype=np.float32) * 0.96
        
        # 添加纸张底纹
        canvas += paper_grain * 0.02
        
        # 计算“吸墨量”：线条强度 * 区域蒙版 * 纸张凹凸
        # 纸张越凹(paper_texture值越小)，越容易积墨 -> ink_factor 越大
        ink_absorption = 1.0 - (paper_grain * 0.25)
        
        # 逐层叠加 (Multiply Blending)
        # 每一层都让纸变黑一点
        
        # 浅调层
        layer1_ink = hatch1 * mask_light * 0.18 * ink_absorption
        canvas *= (1.0 - layer1_ink)
        
        # 中调层
        layer2_ink = hatch2 * mask_mid * 0.25 * ink_absorption
        canvas *= (1.0 - layer2_ink)
        
        # 暗部层
        layer3_ink = hatch3 * mask_dark * 0.35 * ink_absorption
        canvas *= (1.0 - layer3_ink)
        
        # 深部层
        layer4_ink = hatch4 * mask_deep * 0.40 * ink_absorption
        canvas *= (1.0 - layer4_ink)
        
        # ========== 5. 轮廓勾勒 (Sketchy Outlines) ==========
        # 获取基础边缘
        sobel_edges = np.sqrt(cv2.Sobel(gray, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(gray, cv2.CV_32F, 0, 1)**2)
        outline_base = (sobel_edges > 0.15).astype(np.float32)
        
        # 轮廓抖动 (Jitter)
        jx = cv2.resize(np.random.normal(0, 1, (h//8, w//8)), (w, h)) * 1.5
        jy = cv2.resize(np.random.normal(0, 1, (h//8, w//8)), (w, h)) * 1.5
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        
        outline_jittered = cv2.remap(outline_base, (grid_x+jx).astype(np.float32), 
                                     (grid_y+jy).astype(np.float32), cv2.INTER_LINEAR)
        
        # 轮廓也要受纸纹影响，不能是死黑
        outline_ink = outline_jittered * (1.0 - paper_grain * 0.3)
        canvas = np.minimum(canvas, 1.0 - outline_ink * 0.85)

        # ========== 6. 揉擦效果 (Smudging) ==========
        # 模拟手指在暗部抹匀石墨
        smudge_layer = cv2.GaussianBlur(canvas, (25, 25), 10.0)
        # 仅在中间调和暗部混合模糊层
        smudge_mask = mask_mid * 0.3
        canvas = canvas * (1 - smudge_mask) + smudge_layer * smudge_mask
        
        # ========== 7. 最终色阶调整 ==========
        # 确保黑度足够但不过曝
        canvas = np.clip(canvas, 0.05, 1.0)
        
        # 背景处理
        fg_mask = (depth_map > 0.01).astype(np.float32)
        fg_mask = cv2.GaussianBlur(fg_mask, (3, 3), 1.0) # 边缘柔化
        
        # 背景保持纯净纸白 (稍微带点灰度)
        final_sketch = canvas * fg_mask + 0.98 * (1 - fg_mask)
        
        final_sketch = np.clip(final_sketch * 255, 0, 255).astype(np.uint8)
        
        return cv2.cvtColor(final_sketch, cv2.COLOR_GRAY2RGB)
        
        if strength < 1.0:
            out = (out.astype(np.float32) * strength +
                   color_map.astype(np.float32) * (1.0 - strength)).astype(np.uint8)
        
        return out

    def _generate_paper_texture(self, h: int, w: int) -> np.ndarray:
        """
        生成真实的冷压水彩纸纹理
        特点：多尺度纤维结构、凹凸不平的表面
        """
        # 基础纤维纹理 - 多个尺度叠加
        paper = np.zeros((h, w), np.float32)
        
        # 大尺度起伏（纸张整体不平整）
        coarse = np.random.normal(0, 1, (h // 16 + 1, w // 16 + 1)).astype(np.float32)
        coarse = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
        paper += coarse * 0.3
        
        # 中尺度纤维
        medium = np.random.normal(0, 1, (h // 4 + 1, w // 4 + 1)).astype(np.float32)
        medium = cv2.resize(medium, (w, h), interpolation=cv2.INTER_LINEAR)
        paper += medium * 0.4
        
        # 细微纤维纹理
        fine = np.random.normal(0, 1, (h, w)).astype(np.float32)
        fine = cv2.GaussianBlur(fine, (3, 3), 0.8)
        paper += fine * 0.3
        
        # 归一化到0-1
        paper = (paper - paper.min()) / (paper.max() - paper.min() + 1e-6)
        
        return paper
    
    def _compute_ambient_occlusion(self, depth_map: np.ndarray, normal_map: np.ndarray) -> np.ndarray:
        """
        计算屏幕空间环境遮蔽(SSAO)
        用于识别凹陷区域，增加颜料沉积
        """
        h, w = depth_map.shape
        
        # 多尺度深度梯度 - 检测凹陷
        ao = np.zeros((h, w), np.float32)
        
        # 拉普拉斯算子检测凹陷
        laplacian = cv2.Laplacian(depth_map, cv2.CV_32F)
        ao += np.clip(-laplacian * 5, 0, 1)  # 凹陷处为正
        
        # 基于法线的遮蔽
        # 法线朝下/内侧的区域更暗
        normal_z = normal_map[:, :, 2]  # Z分量，朝向相机
        ao += np.clip((1 - normal_z) * 0.5, 0, 1)
        
        # 多尺度模糊检测大范围遮蔽
        for radius in [5, 15, 31]:
            depth_blur = cv2.GaussianBlur(depth_map, (radius, radius), 0)
            local_occlusion = np.clip((depth_map - depth_blur) * 3, 0, 1)
            ao += local_occlusion * 0.3
        
        # 归一化
        ao = np.clip(ao, 0, 1)
        ao = cv2.GaussianBlur(ao, (5, 5), 0)
        
        return ao
    
    def _generate_coffee_stain_edge(self, mask: np.ndarray, edge_width: int = 15) -> np.ndarray:
        """
        生成"咖啡渍"效果的不规则硬边缘
        模拟水彩颜料干燥时边缘颜料聚集的现象
        """
        h, w = mask.shape
        
        # 使用噪声扭曲边缘
        # 生成位移场
        noise_x = np.random.normal(0, 1, (h // 4 + 1, w // 4 + 1)).astype(np.float32)
        noise_y = np.random.normal(0, 1, (h // 4 + 1, w // 4 + 1)).astype(np.float32)
        noise_x = cv2.resize(noise_x, (w, h), interpolation=cv2.INTER_CUBIC)
        noise_y = cv2.resize(noise_y, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # 应用位移（扭曲边缘）
        map_x = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
        map_y = np.arange(h, dtype=np.float32)[:, None].repeat(w, axis=1)
        map_x = map_x + noise_x * 8
        map_y = map_y + noise_y * 8
        
        # 扭曲后的mask
        warped_mask = cv2.remap(mask.astype(np.float32), map_x, map_y, 
                                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        # 计算边缘带 - 使用距离变换
        mask_binary = (warped_mask > 0.5).astype(np.uint8) * 255
        dist_inside = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
        dist_outside = cv2.distanceTransform(255 - mask_binary, cv2.DIST_L2, 5)
        
        # 边缘带：在边界附近的区域
        edge_band = np.clip(1 - dist_inside / edge_width, 0, 1)
        edge_band = edge_band * (dist_inside > 0).astype(np.float32)
        
        # 添加不规则性 - 边缘浓度变化
        edge_noise = np.random.normal(0, 1, (h, w)).astype(np.float32)
        edge_noise = cv2.GaussianBlur(edge_noise, (0, 0), 3)
        edge_noise = (edge_noise - edge_noise.min()) / (edge_noise.max() - edge_noise.min() + 1e-6)
        
        # 咖啡渍效果：边缘处颜料聚集，形成深色环
        coffee_ring = edge_band ** 0.5 * (0.5 + edge_noise * 0.5)
        
        # 二次噪声添加"颗粒"感
        granular = np.random.normal(0, 1, (h, w)).astype(np.float32)
        granular = cv2.GaussianBlur(granular, (3, 3), 0.5)
        granular = (granular > 0.3).astype(np.float32)
        
        coffee_ring = coffee_ring * (0.7 + granular * 0.3)
        
        return np.clip(coffee_ring, 0, 1), warped_mask

    def apply_watercolor_style(self, color_map: np.ndarray, edge_map: np.ndarray,
                              depth_map: np.ndarray, strength: float = 0.9) -> np.ndarray:
        """
        专业水彩风格渲染 (增强版)
        
        新特性：
        1. 咖啡渍硬边缘效果 - 不规则的自然轮廓
        2. 纸张纹理渗透 - 颜料真正"渗入"纸张纤维
        3. AO驱动的颜料沉积 - 凹陷处颜色更深、颗粒更明显
        
        原有特性：半透明叠色、颜色晕染、回边水痕、颗粒感、边缘加深
        """
        h, w = color_map.shape[:2]
        src = color_map.copy()
        
        # === 0. 预计算关键贴图 ===
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # 生成真实纸张纹理（作为渲染基底）
        paper_texture = self._generate_paper_texture(h, w)
        
        # 计算环境遮蔽（用于颜料沉积）
        ao_map = self._compute_ambient_occlusion(depth_map, self.normal_map if self.normal_map is not None 
                                                  else np.zeros((h, w, 3), np.float32))
        
        # 前景遮罩
        fg_mask = (depth_map > 0.02).astype(np.float32)
        fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
        
        # === 1. 生成湿度场 (wetness map) ===
        wetness = gray / 255.0
        wet_noise = np.random.normal(0, 1, (h, w)).astype(np.float32)
        wet_noise = cv2.GaussianBlur(wet_noise, (0, 0), 30.0)
        wet_noise = (wet_noise - wet_noise.min()) / (wet_noise.max() - wet_noise.min() + 1e-6)
        wetness = wetness * 0.6 + wet_noise * 0.4
        wetness = cv2.GaussianBlur(wetness, (0, 0), 20.0)
        
        # === 2. 创建纸张基底（真正的背景层）===
        # 纸张基础色（暖白色）
        paper_color = np.array([252, 248, 240], dtype=np.float32)  # 略带暖调的纸白
        paper_base = np.ones((h, w, 3), np.float32) * paper_color
        
        # 纸张纹理影响 - 产生凹凸明暗
        paper_shading = (paper_texture - 0.5) * 20
        paper_base = paper_base + paper_shading[:, :, None]
        paper_base = np.clip(paper_base, 0, 255)
        
        # === 3. 色彩简化（块面抽象）===
        smooth = cv2.pyrMeanShiftFiltering(src, sp=15, sr=30)
        for _ in range(3):
            smooth = cv2.bilateralFilter(smooth, 9, 60, 60)
        
        # === 4. 颜色扩散/晕染 (bleeding) ===
        bleeding = smooth.copy().astype(np.float32)
        for i in range(3):
            blur_radius = max(3, int(7 + wetness.mean() * 10))
            if blur_radius % 2 == 0:
                blur_radius += 1
            channel = bleeding[:, :, i]
            diffused = cv2.GaussianBlur(channel, (blur_radius, blur_radius), 0)
            bleeding[:, :, i] = channel * (1 - wetness * 0.5) + diffused * (wetness * 0.5)
        bleeding = np.clip(bleeding, 0, 255).astype(np.uint8)
        
        # === 5. 色彩分离效果 ===
        edges_grad = cv2.Laplacian(gray, cv2.CV_32F)
        edges_grad = np.abs(edges_grad)
        edges_grad = cv2.GaussianBlur(edges_grad, (3, 3), 0)
        edges_grad = edges_grad / (edges_grad.max() + 1e-6)
        
        separated = bleeding.copy().astype(np.float32)
        separated[:, :-1, 0] = separated[:, :-1, 0] * (1 - edges_grad[:, :-1] * 0.15) + \
                                separated[:, 1:, 0] * (edges_grad[:, :-1] * 0.15)
        separated[:, 1:, 2] = separated[:, 1:, 2] * (1 - edges_grad[:, 1:] * 0.15) + \
                               separated[:, :-1, 2] * (edges_grad[:, 1:] * 0.15)
        separated = np.clip(separated, 0, 255).astype(np.uint8)
        
        # === 6. 饱和度与明度调整 ===
        hsv = cv2.cvtColor(separated, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
        hsv[:, :, 2] = np.power(hsv[:, :, 2] / 255.0, 0.85) * 255.0
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # === 7. 留白高光 ===
        brightness = hsv[:, :, 2]
        white_mask = (brightness > 200).astype(np.float32)
        white_mask = cv2.GaussianBlur(white_mask, (5, 5), 0)
        
        # === 8. 咖啡渍硬边缘效果 ===
        # 生成不规则边缘和咖啡渍环
        coffee_ring, warped_fg_mask = self._generate_coffee_stain_edge(fg_mask, edge_width=12)
        
        # 咖啡渍颜色：取边缘处颜色的深色版本
        edge_color = cv2.GaussianBlur(result.astype(np.float32), (15, 15), 0)
        # 变暗变饱和
        edge_hsv = cv2.cvtColor(edge_color.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        edge_hsv[:, :, 1] = np.clip(edge_hsv[:, :, 1] * 1.4, 0, 255)
        edge_hsv[:, :, 2] = edge_hsv[:, :, 2] * 0.6
        edge_darkened = cv2.cvtColor(edge_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # 应用咖啡渍效果
        result_f = result.astype(np.float32)
        coffee_intensity = coffee_ring[:, :, None] * 0.7
        result_f = result_f * (1 - coffee_intensity) + edge_darkened.astype(np.float32) * coffee_intensity
        result = np.clip(result_f, 0, 255).astype(np.uint8)
        
        # === 9. 回边/水痕 (backrun/cauliflower) ===
        gray_bin = cv2.threshold(gray.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)[1]
        dist = cv2.distanceTransform(255 - gray_bin, cv2.DIST_L2, 5)
        dist = cv2.normalize(dist, None, 0, 1, cv2.NORM_MINMAX)
        
        backrun_noise = np.random.normal(0, 1, (h, w)).astype(np.float32)
        backrun_noise = cv2.GaussianBlur(backrun_noise, (0, 0), 8.0)
        backrun = dist + backrun_noise * 0.2
        backrun = cv2.GaussianBlur(backrun, (0, 0), 5.0)
        
        ring_mask = ((backrun > 0.2) & (backrun < 0.4)).astype(np.float32)
        ring_mask = cv2.GaussianBlur(ring_mask, (7, 7), 0) * wetness
        
        result = result.astype(np.float32)
        result = result + ring_mask[:, :, None] * 15
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # === 10. AO驱动的颜料沉积 ===
        # 凹陷处颜色加深 + 颗粒聚集
        ao_darken = ao_map ** 1.5  # 增强对比
        ao_darken = cv2.GaussianBlur(ao_darken, (5, 5), 0)
        
        # 在AO区域降低明度
        result_f = result.astype(np.float32)
        darken_strength = ao_darken[:, :, None] * 40  # 凹陷处变暗
        result_f = result_f - darken_strength
        
        # AO区域添加额外颗粒 - 颜料在凹陷处聚集
        ao_granules = np.random.normal(0, 1, (h, w)).astype(np.float32)
        ao_granules = cv2.GaussianBlur(ao_granules, (3, 3), 0.5)
        ao_granule_mask = ao_map * (1 - white_mask)  # AO强度 * 非高光区
        result_f = result_f + ao_granules[:, :, None] * ao_granule_mask[:, :, None] * 25
        result = np.clip(result_f, 0, 255).astype(np.uint8)
        
        # === 11. 边缘处理 ===
        edges = cv2.Canny(gray.astype(np.uint8), 40, 100)
        edges = edges.astype(np.float32) / 255.0
        
        wet_edges = cv2.GaussianBlur(edges * wetness, (7, 7), 0)
        dry_edges = edges * (1 - wetness)
        dry_edges = cv2.dilate(dry_edges, np.ones((2, 2), np.uint8))
        
        edge_darken = wet_edges + dry_edges * 0.5
        edge_darken = cv2.GaussianBlur(edge_darken, (3, 3), 0)
        darken_factor = 1.0 - edge_darken * 0.25
        result = (result.astype(np.float32) * darken_factor[:, :, None])
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # === 12. 颜料渗入纸张纹理 ===
        # 关键改进：让纸张纹理与颜料层真正融合
        result_f = result.astype(np.float32)
        
        # 纸张凹陷处积聚更多颜料（更深）
        # 纸张凸起处颜料更薄（更亮，露出纸白）
        paper_influence = (paper_texture - 0.5) * 2  # -1到1
        
        # 计算颜料浓度（越深越不透明）
        paint_opacity = 1.0 - (brightness / 255.0) ** 0.5
        paint_opacity = paint_opacity * fg_mask  # 只在前景区域
        
        # 在纸张凸起处：颜料变薄，露出纸张颜色
        thin_paint_mask = np.clip(paper_influence, 0, 1) * paint_opacity
        result_f = result_f * (1 - thin_paint_mask[:, :, None] * 0.3) + \
                   paper_base * (thin_paint_mask[:, :, None] * 0.3)
        
        # 在纸张凹陷处：颜料聚集，颜色加深
        pool_mask = np.clip(-paper_influence, 0, 1) * paint_opacity
        result_f = result_f * (1 - pool_mask[:, :, None] * 0.15)
        
        result = np.clip(result_f, 0, 255).astype(np.uint8)
        
        # === 13. 多尺度颗粒感（增强版）===
        granulation = np.zeros((h, w), np.float32)
        for scale in [2, 4, 8, 16]:
            noise = np.random.normal(0, 1, (h // scale + 1, w // scale + 1)).astype(np.float32)
            noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
            noise = cv2.GaussianBlur(noise, (0, 0), scale * 0.5)
            # 大尺度噪声权重更低
            granulation += noise * (1.0 / (scale ** 0.7))
        
        granulation = (granulation - granulation.min()) / (granulation.max() - granulation.min() + 1e-6)
        granulation = (granulation - 0.5) * 2
        
        # 颗粒强度受AO、纸张纹理、暗度共同影响
        grain_mask = 1.0 - (brightness / 255.0) ** 0.7
        grain_mask = grain_mask * (1 - white_mask)
        grain_mask = grain_mask * (1 + ao_map * 0.5)  # AO区域颗粒更多
        grain_mask = grain_mask * (1 + np.clip(-paper_influence, 0, 1) * 0.3)  # 纸张凹陷处颗粒更多
        
        result_f = result.astype(np.float32)
        result_f = result_f + granulation[:, :, None] * grain_mask[:, :, None] * 10
        result = np.clip(result_f, 0, 255).astype(np.uint8)
        
        # === 14. 半透明叠色效果 ===
        base_layer = result.copy()
        overlay = cv2.GaussianBlur(result, (0, 0), 15.0)
        
        alpha = wetness * 0.3 + (1 - brightness / 255.0) * 0.2
        alpha = np.clip(alpha, 0, 0.5)
        
        result = (overlay.astype(np.float32) * alpha[:, :, None] + 
                  base_layer.astype(np.float32) * (1 - alpha[:, :, None]))
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # === 15. 与纸张背景最终合成 ===
        # 使用扭曲后的前景遮罩（包含咖啡渍边缘）
        final_mask = warped_fg_mask[:, :, None]
        
        # 边缘羽化 + 咖啡渍效果
        result_f = result.astype(np.float32)
        paper_base_f = paper_base
        
        # 在边缘处混合（不是硬切）
        blend_mask = cv2.GaussianBlur(final_mask[:, :, 0], (7, 7), 0)[:, :, None]
        result_f = result_f * blend_mask + paper_base_f * (1 - blend_mask)
        
        # 留白区域显示纸张
        result_f = result_f * (1 - white_mask[:, :, None] * 0.7) + \
                   paper_base_f * (white_mask[:, :, None] * 0.7)
        
        result = np.clip(result_f, 0, 255).astype(np.uint8)
        
        # === 最终处理 ===
        result = cv2.GaussianBlur(result, (3, 3), 0.5)
        
        if strength < 1.0:
            result = (result.astype(np.float32) * strength + 
                      color_map.astype(np.float32) * (1.0 - strength)).astype(np.uint8)
        
        return result

    def apply_cartoon_style(self, color_map: np.ndarray, edge_map: np.ndarray,
                           depth_map: np.ndarray, strength: float = 0.95) -> np.ndarray:
        """
        动漫/3D游戏赛璐珞风格 (Anime Cel-Shading)
        特点：
        1. 极度平滑的色块（无渐变过渡）
        2. 明确的亮暗分层（2-3个色阶）
        3. 粗黑色轮廓线
        4. 高饱和度鲜艳色彩
        """
        h, w = color_map.shape[:2]

        # === 前景/背景分离 ===
        eps = 0.02
        mask = (depth_map > eps).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 3.0)
        mask3 = np.dstack([mask_f, mask_f, mask_f])

        src = color_map.copy()

        # === 1. 强力平滑 - 多次双边滤波消除所有细节 ===
        smooth = src.copy()
        for _ in range(6):
            smooth = cv2.bilateralFilter(smooth, 9, 100, 100)

        # === 2. 提高饱和度 - 动漫风格更鲜艳 ===
        hsv = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.6, 0, 255)  # 大幅提升饱和度
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)  # 稍微提亮
        smooth = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # === 3. 硬色阶量化 - 产生明确的色块分层 ===
        # 量化为4个等级，产生动漫风格的分层效果
        levels = 4
        div = 256 // levels
        quantized = (smooth // div) * div + div // 2
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)

        # === 4. 添加简单的明暗分层 ===
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY).astype(np.float32)
        # 阴影层：较暗区域
        shadow_mask = (gray < 100).astype(np.float32)
        shadow_mask = cv2.GaussianBlur(shadow_mask, (9, 9), 0)
        # 阴影区域颜色变暗
        quantized_f = quantized.astype(np.float32)
        quantized_f = quantized_f * (1.0 - shadow_mask[:, :, None] * 0.2)
        quantized = np.clip(quantized_f, 0, 255).astype(np.uint8)

        # === 5. 提取粗黑轮廓线 ===
        gray_blur = cv2.GaussianBlur(cv2.cvtColor(src, cv2.COLOR_RGB2GRAY), (3, 3), 0)
        
        # 使用自适应阈值获得更完整的轮廓
        edges1 = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 9, 2)
        edges1 = 255 - edges1  # 反转
        
        # Canny边缘补充
        edges2 = cv2.Canny(gray_blur, 50, 150)
        
        # 合并边缘
        edges = cv2.bitwise_or(edges1, edges2)
        
        # 膨胀使轮廓线更粗
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        
        # 应用前景遮罩
        edges = (edges.astype(np.float32) * mask_f).astype(np.uint8)

        # === 6. 叠加黑色轮廓线 ===
        edges_3ch = np.dstack([edges, edges, edges]).astype(np.float32) / 255.0
        cartoon = quantized.astype(np.float32) * (1.0 - edges_3ch)
        cartoon = np.clip(cartoon, 0, 255).astype(np.uint8)

        # === 7. 白色背景合成 ===
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
        """
        油画风格 - 快速高效版本
        """
        h, w = color_map.shape[:2]
        src = color_map.copy()
        
        # 1) 增强饱和度
        hsv = cv2.cvtColor(src, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
        vivid = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # 2) 使用双边滤波+中值滤波模拟油画笔触
        oil = vivid
        for _ in range(3):
            oil = cv2.bilateralFilter(oil, 7, 60, 60)
        oil = cv2.medianBlur(oil, 5)
        
        # 3) 色阶量化（轻微）
        div = 32
        oil = (oil // div) * div + div // 2
        
        # 4) 画布纹理
        canvas = np.random.normal(0, 1, (h, w)).astype(np.float32)
        canvas = cv2.GaussianBlur(canvas, (0, 0), 1.0)
        
        oil = oil.astype(np.float32) + canvas[:, :, None] * 6
        oil = np.clip(oil, 0, 255).astype(np.uint8)
        
        # 5) 边缘增强
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        edges = cv2.Laplacian(gray, cv2.CV_32F)
        edges = np.abs(edges)
        edges = cv2.GaussianBlur(edges, (0, 0), 1.0)
        edges = edges / (edges.max() + 1e-6) * 0.12
        
        oil = oil.astype(np.float32) * (1.0 + edges[:, :, None])
        oil = np.clip(oil, 0, 255).astype(np.uint8)
        
        if strength < 1.0:
            oil = (oil.astype(np.float32) * strength + 
                   color_map.astype(np.float32) * (1.0 - strength)).astype(np.uint8)
        
        return oil


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
