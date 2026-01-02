"""
诊断深度图问题
"""
import numpy as np
import cv2
from PIL import Image
from npr_renderer import NPRRenderer

print("=" * 60)
print("深度图诊断")
print("=" * 60)

# 加载模型
renderer = NPRRenderer()
model = renderer.load_obj_simple("data/models/armadillo.obj")
print(f"\n模型: {model['name']}")
print(f"顶点数: {model['stats']['vertices']}")
print(f"顶点范围: [{model['vertices'].min():.3f}, {model['vertices'].max():.3f}]")

# 渲染深度图
print("\n渲染深度图...")
depth_map, normal_map = renderer.render_depth_normal(model, 512)

print(f"\n深度图统计:")
print(f"  形状: {depth_map.shape}")
print(f"  范围: [{depth_map.min():.4f}, {depth_map.max():.4f}]")
print(f"  均值: {depth_map.mean():.4f}")
print(f"  非零像素: {np.sum(depth_map > 0)}/{depth_map.size} ({100*np.sum(depth_map > 0)/depth_map.size:.1f}%)")
print(f"  > 0.001: {np.sum(depth_map > 0.001)}/{depth_map.size} ({100*np.sum(depth_map > 0.001)/depth_map.size:.1f}%)")
print(f"  > 0.01: {np.sum(depth_map > 0.01)}/{depth_map.size} ({100*np.sum(depth_map > 0.01)/depth_map.size:.1f}%)")
print(f"  > 0.1: {np.sum(depth_map > 0.1)}/{depth_map.size} ({100*np.sum(depth_map > 0.1)/depth_map.size:.1f}%)")

# 查看深度值分布
unique_vals = np.unique(depth_map)
print(f"\n唯一值数量: {len(unique_vals)}")
print(f"前10个值: {unique_vals[:10]}")
print(f"后10个值: {unique_vals[-10:]}")

# 保存深度图可视化
depth_vis = (depth_map * 255).astype(np.uint8)
Image.fromarray(depth_vis).save("depth_diagnostic.png")
print(f"\n深度图已保存: depth_diagnostic.png")
print("  白色 = 前景（深度大）")
print("  黑色 = 背景（深度为0）")

# 检查是否有黑色区域
black_pixels = np.sum(depth_map == 0)
print(f"\n完全黑色的像素（depth=0）: {black_pixels}/{depth_map.size} ({100*black_pixels/depth_map.size:.1f}%)")

if black_pixels < depth_map.size * 0.3:
    print("\n⚠️  警告: 背景区域太少！")
    print("   问题: 模型填满了整个渲染视口")
    print("   原因: 可能是光栅化问题或模型缩放后仍然太大")
else:
    print("\n✓ 背景区域充足")

print("=" * 60)
