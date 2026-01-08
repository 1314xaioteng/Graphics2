# -*- coding: utf-8 -*-
"""
优化四种风格化效果
"""

new_styles = '''
    def apply_sketch_style(self, color_map: np.ndarray, edge_map: np.ndarray,
                           depth_map: np.ndarray, strength: float = 0.9) -> np.ndarray:
        """
        铅笔素描风格 - 经典颜色减淡+交叉影线
        """
        h, w = color_map.shape[:2]
        
        # 转灰度
        gray = cv2.cvtColor(color_map, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # 1) 颜色减淡混合 - 经典素描效果
        inv = 255.0 - gray
        blur = cv2.GaussianBlur(inv, (0, 0), 30.0)
        dodge = gray * 256.0 / (256.0 - blur + 1e-6)
        dodge = np.clip(dodge, 0, 255)
        
        # 2) 计算明暗度 - 用于控制纹理密度
        # 使用原始灰度而非dodge，这样暗部更明显
        tone = 1.0 - gray / 255.0
        tone = np.clip(tone * 1.3, 0, 1)
        
        # 3) 创建交叉影线纹理
        # 第一层：45度斜线
        hatch1 = np.zeros((h, w), np.float32)
        for i in range(-h, w + h, 2):
            pt1 = (i, 0)
            pt2 = (i - h, h)
            cv2.line(hatch1, pt1, pt2, 1.0, 1, cv2.LINE_AA)
        
        # 第二层：-45度斜线（交叉）
        hatch2 = np.zeros((h, w), np.float32)
        for i in range(-h, w + h, 3):
            pt1 = (i, 0)
            pt2 = (i + h, h)
            cv2.line(hatch2, pt1, pt2, 1.0, 1, cv2.LINE_AA)
        
        # 第三层：水平线（更密的阴影）
        hatch3 = np.zeros((h, w), np.float32)
        for i in range(0, h, 4):
            cv2.line(hatch3, (0, i), (w, i), 1.0, 1, cv2.LINE_AA)
        
        # 模糊使线条更自然
        hatch1 = cv2.GaussianBlur(hatch1, (0, 0), 0.6)
        hatch2 = cv2.GaussianBlur(hatch2, (0, 0), 0.6)
        hatch3 = cv2.GaussianBlur(hatch3, (0, 0), 0.6)
        
        # 4) 根据明暗程度叠加不同层次的纹理
        # 浅色：少量第一层
        # 中间：第一层+第二层
        # 暗色：三层都有
        layer1 = hatch1 * np.clip(tone * 2.5, 0, 1)
        layer2 = hatch2 * np.clip((tone - 0.3) * 2.5, 0, 1)
        layer3 = hatch3 * np.clip((tone - 0.6) * 3.0, 0, 1)
        
        texture = layer1 + layer2 * 0.7 + layer3 * 0.5
        texture = np.clip(texture, 0, 1)
        
        # 5) 边缘线条
        edges = cv2.Canny(gray.astype(np.uint8), 30, 100)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
        edges = cv2.GaussianBlur(edges.astype(np.float32), (0, 0), 1.0)
        edges = edges / 255.0
        
        # 6) 合成素描
        sketch = 255.0 - texture * 150 - edges * 100
        
        # 混合一些dodge效果保留细节
        sketch = sketch * 0.6 + dodge * 0.4
        sketch = np.clip(sketch, 0, 255)
        
        # 7) 纸张纹理
        paper = np.random.normal(0, 1, (h, w)).astype(np.float32)
        paper = cv2.GaussianBlur(paper, (0, 0), 1.0)
        sketch = sketch + paper * 2
        sketch = np.clip(sketch, 0, 255).astype(np.uint8)
        
        out = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
        
        if strength < 1.0:
            out = (out.astype(np.float32) * strength + 
                   color_map.astype(np.float32) * (1.0 - strength)).astype(np.uint8)
        
        return out

    def apply_watercolor_style(self, color_map: np.ndarray, edge_map: np.ndarray,
                              depth_map: np.ndarray, strength: float = 0.9) -> np.ndarray:
        """
        水彩风格 - 柔和色块+边缘渗透+白色背景
        """
        h, w = color_map.shape[:2]
        src = color_map.copy()
        
        # 1) 简化色彩
        smooth = src
        for _ in range(5):
            smooth = cv2.bilateralFilter(smooth, 9, 75, 75)
        smooth = cv2.medianBlur(smooth, 5)
        
        # 2) 增强饱和度
        hsv = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
        smooth = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # 3) 边缘加深
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 120)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
        edges = cv2.GaussianBlur(edges.astype(np.float32), (0, 0), 2.5)
        edges = edges / (edges.max() + 1e-6)
        
        darken = 1.0 - edges * 0.2
        smooth = (smooth.astype(np.float32) * darken[:, :, None]).astype(np.uint8)
        
        # 4) 水彩纹理
        paper = np.random.normal(0, 1, (h, w)).astype(np.float32)
        paper = cv2.GaussianBlur(paper, (0, 0), 2.0)
        
        wet = np.random.normal(0, 1, (h, w)).astype(np.float32)
        wet = cv2.GaussianBlur(wet, (0, 0), 40.0)
        wet = (wet - wet.min()) / (wet.max() - wet.min() + 1e-6)
        
        result = smooth.astype(np.float32)
        result = result + paper[:, :, None] * 4
        result = result + (wet[:, :, None] - 0.5) * 12
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # 5) 轻微模糊
        result = cv2.GaussianBlur(result, (3, 3), 0.5)
        
        if strength < 1.0:
            result = (result.astype(np.float32) * strength + 
                      color_map.astype(np.float32) * (1.0 - strength)).astype(np.uint8)
        
        return result

    def apply_cartoon_style(self, color_map: np.ndarray, edge_map: np.ndarray,
                            depth_map: np.ndarray, strength: float = 0.95) -> np.ndarray:
        """
        卡通风格 - 色阶量化+清晰轮廓
        """
        src = color_map.copy()
        
        # 1) 双边滤波平滑
        smooth = src
        for _ in range(3):
            smooth = cv2.bilateralFilter(smooth, 9, 75, 75)
        
        # 2) 增强饱和度
        hsv = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
        smooth = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # 3) 色阶量化
        n_colors = 6
        div = 256 // n_colors
        quantized = (smooth // div) * div + div // 2
        
        # 4) 提取清晰轮廓线
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 使用Canny获得干净的边缘
        edges = cv2.Canny(gray, 80, 160)
        
        # 膨胀边缘
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        
        # 5) 合并色块和轮廓
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        # 边缘处设为黑色
        cartoon = quantized.copy()
        cartoon[edges_3ch > 0] = 0
        
        if strength < 1.0:
            cartoon = (cartoon.astype(np.float32) * strength + 
                       color_map.astype(np.float32) * (1.0 - strength)).astype(np.uint8)
        
        return cartoon

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
    print('All styles optimized!')
else:
    print(f'Error: start={start_idx}, end={end_idx}')
