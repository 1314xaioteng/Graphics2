# -*- coding: utf-8 -*-
"""
修复四种风格化效果 - 正确处理背景
"""

new_styles = '''
    def apply_sketch_style(self, color_map: np.ndarray, edge_map: np.ndarray,
                           depth_map: np.ndarray, strength: float = 0.9) -> np.ndarray:
        """
        铅笔素描风格 - 白色纸张背景
        """
        h, w = color_map.shape[:2]
        
        # 前景遮罩
        eps = 0.02
        mask = (depth_map > eps).astype(np.float32)
        mask = cv2.morphologyEx((mask * 255).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 3.0)
        
        # 1) 转灰度
        gray = cv2.cvtColor(color_map, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # 2) 反转
        inv = 255.0 - gray
        
        # 3) 高斯模糊反转图像
        blur = cv2.GaussianBlur(inv, (0, 0), 21.0)
        
        # 4) 颜色减淡混合
        sketch = gray / (256.0 - blur + 1e-6) * 256.0
        sketch = np.clip(sketch, 0, 255)
        
        # 5) 增强对比度
        sketch = cv2.normalize(sketch, None, 0, 255, cv2.NORM_MINMAX)
        
        # 6) 添加铅笔纹理（仅在暗部）
        texture = np.zeros((h, w), np.float32)
        for i in range(0, max(h, w) * 2, 4):
            cv2.line(texture, (i, 0), (0, i), 0.5, 1)
        texture = cv2.GaussianBlur(texture, (3, 3), 0.5)
        
        darkness = 1.0 - sketch / 255.0
        sketch = sketch - texture * darkness * 25 * mask_f
        sketch = np.clip(sketch, 0, 255)
        
        # 7) 纸张纹理
        paper = np.random.normal(0, 1, (h, w)).astype(np.float32)
        paper = cv2.GaussianBlur(paper, (0, 0), 1.0)
        sketch = sketch + paper * 2 * mask_f
        sketch = np.clip(sketch, 0, 255)
        
        # 8) 白色背景
        bg = np.ones((h, w), np.float32) * 255.0
        # 轻微纸张纹理
        bg_texture = np.random.normal(0, 1, (h, w)).astype(np.float32)
        bg_texture = cv2.GaussianBlur(bg_texture, (0, 0), 2.0)
        bg = bg + bg_texture * 3
        bg = np.clip(bg, 245, 255)
        
        # 9) 合成前景和背景
        result = sketch * mask_f + bg * (1.0 - mask_f)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # 转RGB
        out = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        
        if strength < 1.0:
            out = (out.astype(np.float32) * strength + 
                   color_map.astype(np.float32) * (1.0 - strength)).astype(np.uint8)
        
        return out

    def apply_watercolor_style(self, color_map: np.ndarray, edge_map: np.ndarray,
                              depth_map: np.ndarray, strength: float = 0.9) -> np.ndarray:
        """
        水彩风格 - 柔和色块、边缘渗透、白色纸张背景
        """
        h, w = color_map.shape[:2]
        
        # 前景遮罩
        eps = 0.02
        mask = (depth_map > eps).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 5.0)
        mask3 = np.dstack([mask_f, mask_f, mask_f])
        
        src = color_map.copy()
        
        # 1) 多次双边滤波创建平滑色块
        smooth = src
        for _ in range(4):
            smooth = cv2.bilateralFilter(smooth, 9, 75, 75)
        
        # 2) 中值滤波进一步简化
        smooth = cv2.medianBlur(smooth, 5)
        
        # 3) 增强饱和度
        hsv = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
        smooth = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # 4) 边缘柔化（水彩边缘效果）
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
        edges = cv2.GaussianBlur(edges.astype(np.float32), (0, 0), 3.0)
        edges = edges / (edges.max() + 1e-6)
        
        # 边缘处加深
        darken = 1.0 - 0.15 * edges * mask_f
        smooth = (smooth.astype(np.float32) * darken[:, :, None]).astype(np.uint8)
        
        # 5) 水彩纸纹理
        paper = np.random.normal(0, 1, (h, w)).astype(np.float32)
        paper = cv2.GaussianBlur(paper, (0, 0), 2.0)
        paper = paper / (paper.std() + 1e-6)
        
        smooth = smooth.astype(np.float32) + paper[:, :, None] * 4 * mask_f[:, :, None]
        smooth = np.clip(smooth, 0, 255).astype(np.uint8)
        
        # 6) 轻微模糊模拟水彩晕染
        smooth = cv2.GaussianBlur(smooth, (3, 3), 0.8)
        
        # 7) 白色/浅灰水彩纸背景
        bg = np.ones((h, w, 3), np.float32) * 250.0
        bg_wash = np.random.normal(0, 1, (h, w)).astype(np.float32)
        bg_wash = cv2.GaussianBlur(bg_wash, (0, 0), 30.0)
        bg_wash = (bg_wash - bg_wash.min()) / (bg_wash.max() - bg_wash.min() + 1e-6)
        bg = bg - 15 * bg_wash[:, :, None]
        bg = np.clip(bg, 230, 255).astype(np.uint8)
        
        # 8) 合成
        result = (smooth.astype(np.float32) * mask3 + 
                  bg.astype(np.float32) * (1.0 - mask3))
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        if strength < 1.0:
            result = (result.astype(np.float32) * strength + 
                      color_map.astype(np.float32) * (1.0 - strength)).astype(np.uint8)
        
        return result

    def apply_cartoon_style(self, color_map: np.ndarray, edge_map: np.ndarray,
                            depth_map: np.ndarray, strength: float = 0.95) -> np.ndarray:
        """
        卡通风格 - 色阶量化 + 干净轮廓线
        """
        h, w = color_map.shape[:2]
        
        # 前景遮罩
        eps = 0.02
        mask = (depth_map > eps).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 3.0)
        mask3 = np.dstack([mask_f, mask_f, mask_f])
        
        src = color_map.copy()
        
        # 1) 多次双边滤波平滑
        smooth = src
        for _ in range(3):
            smooth = cv2.bilateralFilter(smooth, 9, 75, 75)
        
        # 2) 增强饱和度
        hsv = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
        smooth = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # 3) 色阶量化
        n_colors = 8
        div = 256 // n_colors
        quantized = (smooth // div) * div + div // 2
        
        # 4) 提取干净的轮廓线（使用Canny而不是自适应阈值）
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 100, 200)
        
        # 膨胀轮廓线
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        
        # 只保留前景区域的边缘
        edges = edges.astype(np.float32) * mask_f
        
        # 5) 将边缘叠加到量化图像上
        edges_3ch = np.dstack([edges, edges, edges])
        cartoon = quantized.astype(np.float32) * (1.0 - edges_3ch / 255.0)
        cartoon = np.clip(cartoon, 0, 255).astype(np.uint8)
        
        # 6) 白色背景
        bg = np.ones((h, w, 3), np.uint8) * 255
        
        # 7) 合成
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
        油画风格 - 厚涂笔触效果
        """
        h, w = color_map.shape[:2]
        
        # 前景遮罩
        eps = 0.02
        mask = (depth_map > eps).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 5.0)
        mask3 = np.dstack([mask_f, mask_f, mask_f])
        
        src = color_map.copy()
        
        # 1) 增强饱和度
        hsv = cv2.cvtColor(src, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
        vivid = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # 2) 油画效果
        try:
            oil = cv2.xphoto.oilPainting(vivid, 7, 1)
        except:
            try:
                oil = cv2.stylization(vivid, sigma_s=60, sigma_r=0.45)
            except:
                # 备用：中值滤波+双边滤波
                oil = cv2.medianBlur(vivid, 7)
                oil = cv2.bilateralFilter(oil, 9, 75, 75)
        
        # 3) 画布纹理（仅前景）
        canvas = np.random.normal(0, 1, (h, w)).astype(np.float32)
        canvas = cv2.GaussianBlur(canvas, (0, 0), 1.5)
        canvas = canvas / (canvas.std() + 1e-6)
        
        oil = oil.astype(np.float32) + canvas[:, :, None] * 6 * mask_f[:, :, None]
        oil = np.clip(oil, 0, 255).astype(np.uint8)
        
        # 4) 边缘增强
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        edges = cv2.Laplacian(gray, cv2.CV_32F)
        edges = np.abs(edges)
        edges = cv2.GaussianBlur(edges, (0, 0), 1.0)
        edges = edges / (edges.max() + 1e-6) * 0.12
        
        oil = oil.astype(np.float32) * (1.0 + edges[:, :, None] * mask_f[:, :, None])
        oil = np.clip(oil, 0, 255).astype(np.uint8)
        
        # 5) 画布色背景（米色/灰褐色）
        bg = np.ones((h, w, 3), np.float32)
        bg[:, :, 0] = 230  # R
        bg[:, :, 1] = 220  # G
        bg[:, :, 2] = 210  # B
        
        # 画布纹理
        bg_texture = np.random.normal(0, 1, (h, w)).astype(np.float32)
        bg_texture = cv2.GaussianBlur(bg_texture, (0, 0), 1.0)
        bg = bg + bg_texture[:, :, None] * 8
        bg = np.clip(bg, 200, 245).astype(np.uint8)
        
        # 6) 合成
        result = (oil.astype(np.float32) * mask3 + 
                  bg.astype(np.float32) * (1.0 - mask3))
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        if strength < 1.0:
            result = (result.astype(np.float32) * strength + 
                      color_map.astype(np.float32) * (1.0 - strength)).astype(np.uint8)
        
        return result

'''

# 读取文件
with open(r'd:\MyProject\pythonProject\Graphics2-main\Graphics2\npr_renderer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 找到位置
start_idx = content.find('    def apply_sketch_style')
end_marker = '    def render(self, model: Dict'
end_idx = content.find(end_marker)

if start_idx != -1 and end_idx != -1:
    new_content = content[:start_idx] + new_styles + '\n' + content[end_idx:]
    with open(r'd:\MyProject\pythonProject\Graphics2-main\Graphics2\npr_renderer.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print('All styles fixed successfully!')
else:
    print(f'Error: start={start_idx}, end={end_idx}')
