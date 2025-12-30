# 基于几何指导的手绘风格三维场景NPR渲染——实验报告

本实验设计并实现了一套基于几何缓冲区（G-Buffer）的非真实感渲染（NPR）系统。不同于传统的二维图像滤镜，本系统在光栅化阶段提取三维场景的深度图（Depth Map）与法线图（Normal Map），以此作为几何指导信息，驱动素描、水彩、卡通及油画四种风格的生成。

## NRP

非真实感渲染（Non-Photorealistic Rendering, NPR）旨在模拟艺术媒介的表现形式。传统基于图像处理（Image-space）的方法往往忽略了场景的三维结构，导致渲染结果缺乏体积感或轮廓断裂。本实验旨在通过提取三维模型的几何特征（深度与法线），构建一个“几何指导”的渲染管线，在保持低计算成本的同时，实现具有精确三维结构特征的手绘风格渲染。

## 系统架构和方法

本系统采用延迟着色（Deferred Shading）的思想，将渲染流程分为几何阶段和风格化阶段。

1. 几何缓冲区的生成

   系统首先通过基于扫描线的光栅化算法（Rasterization）处理 `.obj` 格式的三维网格数据。在此过程中，不进行光照计算，而是生成两张核心特征图：

   - **深度图 (Depth Map):** 记录每个像素点到视平面的距离，用于后续的笔触强度控制和阴影模拟。

     <img src="D:\MyProject\pythonProject\Gra_code_test\demo_output\armadillo_depth.png" alt="armadillo_depth" style="zoom:50%;" />

   - **法线图 (Normal Map):** 记录每个像素点的表面朝向向量 $(n_x, n_y, n_z)$，用于精确捕捉物体表面的曲率变化。

   - **实现细节:** 使用 `Numba` 加速的重心坐标插值算法计算片元属性。

   ```python
   # 基于光栅化的几何缓冲区生成核心算法
   @jit(nopython=True)
   def rasterize_triangles(vertices, faces, image_size):
       # ... (省略初始化代码)
       
       # 遍历所有三角形面片
       for i in range(len(faces)):
           # ... (省略顶点投影和包围盒计算)
           
           # 遍历包围盒内的像素
           for y in range(min_y, max_y + 1):
               for x in range(min_x, max_x + 1):
                   # 计算重心坐标 (Barycentric Coordinates)
                   # ... (省略重心坐标计算公式)
   
                   # 如果点在三角形内
                   if w0 >= 0 and w1 >= 0 and w2 >= 0:
                       # 插值计算深度值 Z
                       z = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]
   
                       # Z-Buffer 测试：保留更近的片元
                       if z > depth_map[y, x] or depth_map[y, x] == 0:
                           depth_map[y, x] = z
                           # 将面法线写入法线缓冲区
                           normal_map[y, x, :] = face_normals[i, :]
   
       return depth_map, normal_map
   ```

   

2. 几何指导的轮廓提取

   为了模拟手绘风格中的硬边缘（轮廓线），本实验摒弃了仅基于颜色梯度的Canny算子，改为基于几何信息的边缘检测：

   $$Edge = \alpha \cdot Sobel(Depth) + \beta \cdot Sobel(Normal)$$

   <img src="D:\MyProject\pythonProject\Gra_code_test\demo_output\armadillo_edges.png" alt="armadillo_edges" style="zoom:50%;" />

   - 对深度图应用 Sobel 算子提取**深度不连续边界**（物体轮廓）。

   - 对法线图应用 Sobel 算子提取**法线不连续边界**（物体内部折痕）。

   - **代码实现:** 

     ```python
     # 基于深度与法线的几何边缘检测
     def detect_edges(self, depth_map, normal_map, threshold=0.1):
         # 1. 深度图边缘检测 (捕捉物体轮廓)
         depth_edges = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
         depth_edges += cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
         
         # 2. 法线图边缘检测 (捕捉物体内部折痕)
         normal_x = cv2.Sobel(normal_map[:, :, 0], cv2.CV_32F, 1, 0, ksize=3)
         normal_y = cv2.Sobel(normal_map[:, :, 1], cv2.CV_32F, 0, 1, ksize=3)
         normal_z = cv2.Sobel(normal_map[:, :, 2], cv2.CV_32F, 1, 0, ksize=3)
         # 计算法线梯度的模长
         normal_edges = np.sqrt(normal_x ** 2 + normal_y ** 2 + normal_z ** 2)
         
         # 3. 融合几何信息
         edges = depth_edges + normal_edges * 0.5
         
         # 二值化输出
         edge_map = (edges > threshold).astype(np.uint8)
         return edge_map
     ```

     

3. 风格化渲染

   系统利用提取的 G-Buffer 数据实现了四种风格算法：

   1. **素描风格 (Sketch):**

      - **纹理:** 使用高斯噪声模拟纸张颗粒感。

      - **几何指导:** 利用 **深度图** 控制阴影涂抹（Shading），深度值大的区域笔触更重；利用 **边缘图** 勾勒轮廓。

      - **合成:** `Result = (PaperNoise * EdgeMap * DepthShading)`。

        <img src="D:\MyProject\pythonProject\Gra_code_test\demo_output\armadillo_sketch.png" alt="armadillo_sketch" style="zoom:50%;" />

        ```python
        # 几何指导的素描风格合成
        def apply_sketch_style(self, color_map, edge_map, depth_map, strength=0.9):
            # ... (省略噪声生成部分)
        
            # 1. 利用深度图生成阴影 (Shading via Depth)
            # 只有深度较大（较远/较暗）的区域才加重阴影
            shading = (depth_map * 255).astype(np.uint8)
            _, dark_areas = cv2.threshold(shading, 100, 255, cv2.THRESH_BINARY_INV)
            
            # 2. 几何边缘叠加
            # edge_map: 1是线，0是背景 -> 转换为乘法掩码
            edges = (1 - edge_map) * 255
        
            # 3. 最终合成：底纹 x 轮廓 x 深度阴影
            final_sketch = cv2.multiply(base_sketch / 255.0, edges / 255.0)
            
            # 再次叠加深度阴影增强体积感
            shadow_factor = 1.0 - (dark_areas / 255.0 * 0.3)
            final_sketch = final_sketch * shadow_factor
        
            return result_rgb
        ```

        

   2. **水彩风格 (Watercolor):**

      - **抽象化:** 使用均值漂移滤波（Mean Shift Filtering）对色彩进行平滑，消除高频纹理，形成色块感。

      - **几何指导:** 利用边缘图模拟“湿边效应”（Wet Edges），即在轮廓处叠加加深的半透明层。

      - **湍流模拟:** 叠加低频噪声模拟水彩颜料在纸张上的不均匀扩散。

        <img src="D:\MyProject\pythonProject\Gra_code_test\demo_output\armadillo_watercolor.png" alt="armadillo_watercolor" style="zoom:50%;" />

   3. **卡通风格 (Cartoon/Cel-Shading):**

      - **双边滤波:** 在保持边缘清晰的同时平滑颜色。

      - **色阶量化:** 将连续的RGB色彩空间离散化（Quantization），例如将 256 色阶降维至 8 色阶，形成明显的明暗交界线。

      - **轮廓增强:** 叠加膨胀处理后的加粗边缘图。

        <img src="D:\MyProject\pythonProject\Gra_code_test\demo_output\armadillo_cartoon.png" alt="armadillo_cartoon" style="zoom:50%;" />

   4. **油画风格 (Oil Painting):**

      - **笔触模拟:** 采用保边滤波器（Edge Preserving Filter）产生类似油画笔触的涂抹感。

      - **画布纹理:** 叠加生成的画布纹理图层，并提升HSV空间中的饱和度分量。

        <img src="D:\MyProject\pythonProject\Gra_code_test\demo_output\armadillo_oil.png" alt="armadillo_oil" style="zoom:50%;" />

      

   

