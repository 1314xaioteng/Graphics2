#!/usr/bin/env python3
"""
第10天汇报演示 - 交互式NPR渲染查看器
==================================

功能：
1. 3D模型交互旋转（鼠标拖拽）
2. 实时风格切换
3. 参数动态调节
4. 结果导出

依赖：
- npr_renderer.py
- numpy, PIL, tkinter (标准库)
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from PIL import Image, ImageTk
import os
import sys
import time
import copy

# 尝试导入渲染核心
try:
    from npr_renderer import NPRRenderer
except ImportError as e:
    print("错误: 无法导入 npr_renderer.py")
    print(f"详细错误: {e}")
    print("\n可能的原因:")
    print("1. 缺少依赖包（cv2, numpy, numba等）")
    print("2. 请运行: pip install opencv-python numpy numba pillow")
    sys.exit(1)


class NPRInteractiveApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NPR 3D Interactive Viewer - Day 10 Demo")
        self.root.geometry("1000x750")

        # --- 后端初始化 ---
        self.renderer = NPRRenderer()
        self.models = {}
        self.current_model_data = None  # 原始数据
        self.display_model_data = None  # 旋转后的数据

        # 色彩迁移和背景合成
        self.color_reference = None  # 色彩参考图
        self.background_image = None  # 背景图

        # 交互状态
        self.rotation = [0.0, 0.0]  # [pitch, yaw]
        self.scale = 1.0  # 缩放比例
        self.last_mouse = [0, 0]
        self.is_dragging = False

        # 默认参数
        self.render_size = 512
        self.preview_size = 128  # 拖拽时的低分辨率

        # --- UI 布局 ---
        self._setup_ui()

        # --- 加载数据 ---
        self._load_model_list()

        # 初始渲染
        if self.models:
            self.combo_model.current(0)
            self._on_model_select()

    def _setup_ui(self):
        """构建界面布局"""
        # 1. 顶部控制栏
        control_frame = ttk.LabelFrame(self.root, text="Render Controls", padding=10)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10, ipadx=5)

        # 模型选择
        ttk.Label(control_frame, text="Select Model:").pack(anchor=tk.W, pady=(0, 5))
        self.combo_model = ttk.Combobox(control_frame, state="readonly")
        self.combo_model.pack(fill=tk.X, pady=(0, 15))
        self.combo_model.bind("<<ComboboxSelected>>", self._on_model_select)

        # 风格选择
        ttk.Label(control_frame, text="Rendering Style:").pack(anchor=tk.W, pady=(0, 5))
        self.styles = ['sketch', 'watercolor', 'cartoon', 'oil']
        self.combo_style = ttk.Combobox(control_frame, values=self.styles, state="readonly")
        self.combo_style.current(0)
        self.combo_style.pack(fill=tk.X, pady=(0, 15))
        self.combo_style.bind("<<ComboboxSelected>>", self._trigger_full_render)

        # 强度调节
        ttk.Label(control_frame, text="Style Strength:").pack(anchor=tk.W, pady=(0, 5))
        self.var_strength = tk.DoubleVar(value=0.9)
        self.scale_strength = ttk.Scale(control_frame, from_=0.0, to=1.0,
                                        variable=self.var_strength, orient=tk.HORIZONTAL)
        self.scale_strength.pack(fill=tk.X, pady=(0, 5))
        self.lbl_strength_val = ttk.Label(control_frame, text="0.90")
        self.lbl_strength_val.pack(anchor=tk.E, pady=(0, 15))
        # 绑定释放事件，避免滑动时频繁渲染
        self.scale_strength.bind("<ButtonRelease-1>", self._trigger_full_render)
        self.scale_strength.bind("<B1-Motion>", self._update_strength_label)

        # 渲染信息
        self.lbl_info = ttk.Label(control_frame, text="Ready", foreground="gray")
        self.lbl_info.pack(fill=tk.X, pady=(20, 0))

        # 色彩迁移和背景合成按钮
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(control_frame, text="Advanced Features:").pack(anchor=tk.W)

        ttk.Button(control_frame, text="Load Color Reference", command=self._load_color_reference).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Load Background", command=self._load_background).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Clear Color/BG", command=self._clear_extras).pack(fill=tk.X, pady=2)

        # 缩放控制
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(control_frame, text="Zoom Control:").pack(anchor=tk.W)
        zoom_frame = ttk.Frame(control_frame)
        zoom_frame.pack(fill=tk.X, pady=5)
        ttk.Button(zoom_frame, text="-", width=3, command=self._zoom_out).pack(side=tk.LEFT, padx=2)
        self.lbl_zoom = ttk.Label(zoom_frame, text="100%")
        self.lbl_zoom.pack(side=tk.LEFT, expand=True)
        ttk.Button(zoom_frame, text="+", width=3, command=self._zoom_in).pack(side=tk.LEFT, padx=2)
        
        # 保存按钮
        ttk.Button(control_frame, text="Save Screenshot", command=self._save_image).pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Button(control_frame, text="Reset View", command=self._reset_view).pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        # 2. 左侧画布区域
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(canvas_frame, bg="#2b2b2b", width=512, height=512)
        self.canvas.pack(expand=True)  # 居中

        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        # 绑定滚轮缩放
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)  # Windows
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)  # Linux 向上滚
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)  # Linux 向下滚

        # 提示文字
        self.canvas.create_text(256, 256, text="Drag to Rotate\n(Load a model first)",
                                fill="white", font=("Arial", 14), justify=tk.CENTER)

    def _load_model_list(self):
        """扫描模型文件"""
        model_dir = os.path.join("data", "models")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            # 创建一个假的演示数据（如果目录为空）
            messagebox.showinfo("提示", f"未找到模型文件。请将 .obj 文件放入 {model_dir}")
            return

        files = [f for f in os.listdir(model_dir) if f.endswith(".obj")]
        if not files:
            messagebox.showwarning("警告", "data/models/ 目录下没有找到 .obj 文件")
            return

        for f in files:
            name = os.path.splitext(f)[0]
            path = os.path.join(model_dir, f)
            self.models[name] = path

        self.combo_model['values'] = list(self.models.keys())
        print(f"Loaded {len(self.models)} models.")

    def _on_model_select(self, event=None):
        """选择模型回调"""
        name = self.combo_model.get()
        path = self.models.get(name)
        if not path:
            return

        self.lbl_info.config(text=f"Loading {name}...")
        self.root.update()

        try:
            # 加载模型
            self.current_model_data = self.renderer.load_obj_simple(path)
            self._update_display_model()  # 应用旋转
            self._trigger_full_render()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _update_strength_label(self, event=None):
        """更新滑块数值显示"""
        val = self.var_strength.get()
        self.lbl_strength_val.config(text=f"{val:.2f}")

    def _reset_view(self):
        """重置视角"""
        self.rotation = [0.0, 0.0]
        self.scale = 1.0
        self._update_zoom_label()
        self._update_display_model()
        self._trigger_full_render()

    def _load_color_reference(self):
        """加载色彩参考图"""
        filepath = filedialog.askopenfilename(
            title="选择色彩参考图",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
        )
        if filepath:
            try:
                img = Image.open(filepath).convert('RGB')
                self.color_reference = np.array(img)
                messagebox.showinfo("成功", f"已加载色彩参考图：{os.path.basename(filepath)}")
                self._trigger_full_render()
            except Exception as e:
                messagebox.showerror("错误", f"加载失败：{str(e)}")

    def _load_background(self):
        """加载背景图"""
        filepath = filedialog.askopenfilename(
            title="选择背景图",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
        )
        if filepath:
            try:
                img = Image.open(filepath).convert('RGB')
                self.background_image = np.array(img)
                messagebox.showinfo("成功", f"已加载背景图：{os.path.basename(filepath)}")
                self._trigger_full_render()
            except Exception as e:
                messagebox.showerror("错误", f"加载失败：{str(e)}")

    def _clear_extras(self):
        """清除色彩参考和背景"""
        self.color_reference = None
        self.background_image = None
        messagebox.showinfo("已清除", "色彩参考和背景已清除")
        self._trigger_full_render()

    # --- 缩放控制方法 ---
    def _zoom_in(self):
        """放大模型"""
        self.scale = min(self.scale * 1.2, 5.0)  # 最大5倍
        self._update_zoom_label()
        self._update_display_model()
        self._trigger_full_render()
    
    def _zoom_out(self):
        """缩小模型"""
        self.scale = max(self.scale / 1.2, 0.2)  # 最小0.2倍
        self._update_zoom_label()
        self._update_display_model()
        self._trigger_full_render()
    
    def _on_mouse_wheel(self, event):
        """鼠标滚轮缩放"""
        if not self.current_model_data:
            return
        
        # Windows 和 macOS
        if event.num == 4 or event.delta > 0:
            # 向上滚动 - 放大
            self.scale = min(self.scale * 1.1, 5.0)
        elif event.num == 5 or event.delta < 0:
            # 向下滚动 - 缩小
            self.scale = max(self.scale / 1.1, 0.2)
        
        self._update_zoom_label()
        self._update_display_model()
        self._perform_render(preview=True)
    
    def _update_zoom_label(self):
        """更新缩放标签显示"""
        self.lbl_zoom.config(text=f"{int(self.scale * 100)}%")

    # --- 鼠标交互逻辑 ---
    def _on_mouse_down(self, event):
        self.last_mouse = [event.x, event.y]
        self.is_dragging = True

    def _on_mouse_drag(self, event):
        if not self.current_model_data:
            return

        # 计算旋转增量
        dx = event.x - self.last_mouse[0]
        dy = event.y - self.last_mouse[1]
        self.last_mouse = [event.x, event.y]

        # 更新旋转角度
        self.rotation[1] += dx * 0.01  # Yaw
        self.rotation[0] += dy * 0.01  # Pitch

        # 实时更新（低质量预览）
        self._update_display_model()
        self._perform_render(preview=True)

    def _on_mouse_up(self, event):
        self.is_dragging = False
        # 停止拖拽后，执行一次高质量渲染
        self._trigger_full_render()

    # --- 3D 变换逻辑 ---
    def _rotate_vertices(self, vertices, pitch, yaw):
        """对顶点应用旋转矩阵"""
        # 绕X轴 (Pitch)
        rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])

        # 绕Y轴 (Yaw)
        ry = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])

        # 组合旋转 R = Ry * Rx
        r = np.dot(ry, rx)

        # 应用旋转: V_new = V * R.T
        return np.dot(vertices, r.T)

    def _update_display_model(self):
        """根据当前旋转角度和缩放更新用于渲染的模型数据"""
        if self.current_model_data is None:
            return

        # 深拷贝以避免修改原始数据
        # 注意：为了性能，只拷贝需要修改的顶层结构和顶点
        self.display_model_data = copy.copy(self.current_model_data)

        # 应用旋转
        rotated_verts = self._rotate_vertices(
            self.current_model_data['vertices'],
            self.rotation[0],
            self.rotation[1]
        )
        
        # 应用缩放
        rotated_verts = rotated_verts * self.scale
        
        if self.background_image is not None:
            rotated_verts[:, 0] *= -1
        self.display_model_data['vertices'] = rotated_verts

    # --- 渲染逻辑 ---
    def _trigger_full_render(self, event=None):
        """触发高质量渲染"""
        if self.display_model_data:
            self._perform_render(preview=False)

    def _perform_render(self, preview=False):
        """执行渲染"""
        if not self.display_model_data:
            return

        style = self.combo_style.get()
        strength = self.var_strength.get()
        size = self.preview_size if preview else self.render_size

        t0 = time.time()

        # 调用核心渲染器
        # 注意:这里我们使用 copy 的模型数据（已旋转）
        result_img = self.renderer.render(
            self.display_model_data,
            style=style,
            strength=strength,
            image_size=size,
            color_reference=self.color_reference,
            background=self.background_image
        )

        # 更新界面
        self._show_image(result_img)

        dt = time.time() - t0
        mode = "Preview" if preview else "Final"
        self.lbl_info.config(text=f"{mode}: {dt:.3f}s | {size}x{size}")

    def _show_image(self, img_array):
        """在Canvas上显示图像"""
        # 转换 numpy -> PIL -> ImageTk
        img = Image.fromarray(img_array)

        # 如果是预览模式，放大到画布大小以便观察
        if img.width != 512:
            img = img.resize((512, 512), Image.NEAREST)

        self.tk_img = ImageTk.PhotoImage(img)  # 必须保持引用

        # 居中显示
        cx, cy = 256, 256  # Canvas中心
        self.canvas.delete("all")
        self.canvas.create_image(cx, cy, image=self.tk_img)

    def _save_image(self):
        """保存当前结果"""
        if not self.display_model_data:
            return

        filename = f"screenshot_{int(time.time())}.png"
        style = self.combo_style.get()

        # 重新渲染一遍高清的（防止保存的是预览图）
        img_array = self.renderer.render(
            self.display_model_data,
            style=style,
            strength=self.var_strength.get(),
            image_size=1024,  # 保存更高清
            color_reference=self.color_reference,
            background=self.background_image
        )

        Image.fromarray(img_array).save(filename)
        messagebox.showinfo("Saved", f"Image saved to:\n{os.path.abspath(filename)}")


def main():
    root = tk.Tk()
    # 设置样式主题
    style = ttk.Style()
    style.theme_use('clam')

    app = NPRInteractiveApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()