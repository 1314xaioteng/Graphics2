"""
简化版背景合成测试 - 不依赖numba
"""
import numpy as np
import cv2
from PIL import Image
import sys

print("=" * 60)
print("简化版背景合成测试")
print("=" * 60)

# 1. 加载背景图
bg_path = r"D:\Colortran\Graphics2\asset\fangao.jpg"
print(f"\n[1] 加载背景图: {bg_path}")
background = np.array(Image.open(bg_path).convert('RGB'))
print(f"    背景尺寸: {background.shape}")

# 2. 创建一个简单的测试深度图（模拟armadillo）
print("\n[2] 创建测试深度图...")
depth_map = np.zeros((512, 512), dtype=np.float32)
# 在中心创建一个圆形区域代表前景
center = 256
radius = 150
y, x = np.ogrid[:512, :512]
mask = (x - center)**2 + (y - center)**2 <= radius**2
depth_map[mask] = 1.0  # 前景区域深度为1

print(f"    深度图范围: [{depth_map.min():.4f}, {depth_map.max():.4f}]")
fg_pixels = np.sum(depth_map > 0.001)
print(f"    前景像素: {fg_pixels}/262144 ({100*fg_pixels/262144:.1f}%)")

# 3. 创建前景图（纯蓝色模拟）
print("\n[3] 创建测试前景图...")
foreground = np.zeros((512, 512, 3), dtype=np.uint8)
foreground[:, :] = [100, 150, 255]  # 蓝色

# 4. 背景合成
print("\n[4] 执行背景合成...")
from background_compositor import composite_with_background
result = composite_with_background(foreground, depth_map, background)

# 5. 保存结果
print("\n[5] 保存结果...")
Image.fromarray(depth_map.astype(np.uint8) * 255).save("test_depth.png")
Image.fromarray(foreground).save("test_foreground.png")
Image.fromarray(cv2.resize(background, (512, 512))).save("test_background.png")
Image.fromarray(result).save("test_result.png")

print("\n✓ 测试完成！生成的文件：")
print("  - test_depth.png (深度图)")
print("  - test_foreground.png (前景)")
print("  - test_background.png (背景)")
print("  - test_result.png (合成结果)")
print("\n请查看 test_result.png，如果能看到背景图透过圆形区域外显示，说明背景合成工作正常。")
print("如果整个图都是背景或都是前景，说明Alpha遮罩有问题。")
print("=" * 60)
