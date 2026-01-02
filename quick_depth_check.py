"""
快速检查深度图渲染问题
"""
import numpy as np
from PIL import Image
from npr_renderer import NPRRenderer

print("=" * 60)
print("快速深度图检查")
print("=" * 60)

# 加载模型
renderer = NPRRenderer()
model = renderer.load_obj_simple("data/models/armadillo.obj")

print(f"\n模型信息:")
print(f"  顶点范围: [{model['vertices'].min():.3f}, {model['vertices'].max():.3f}]")
print(f"  顶点数: {model['stats']['vertices']}")

# 渲染深度图
print("\n渲染深度图...")
depth_map, normal_map = renderer.render_depth_normal(model, 512)

# 统计
total = depth_map.size
nonzero = np.sum(depth_map > 0)
print(f"\n深度图统计:")
print(f"  非零像素: {nonzero}/{total} ({100*nonzero/total:.1f}%)")
print(f"  深度范围: [{depth_map.min():.4f}, {depth_map.max():.4f}]")

if nonzero > 0:
    depth_fg = depth_map[depth_map > 0]
    print(f"  前景深度范围: [{depth_fg.min():.4f}, {depth_fg.max():.4f}]")

# 保存深度图可视化
depth_vis = (depth_map * 255).astype(np.uint8)
Image.fromarray(depth_vis).save("quick_depth_check.png")
print(f"\n深度图已保存: quick_depth_check.png")
print("  白色=前景, 黑色=背景")

# 如果前景太少，说明光栅化有问题
if nonzero < total * 0.3:
    print(f"\n⚠️  警告: 前景区域只有 {100*nonzero/total:.1f}%，应该是60%+")
    print("   可能的原因:")
    print("   1. 模型顶点超出[-1,1]范围，被裁剪了")
    print("   2. 光栅化时三角形包围盒计算有问题")
    print("   3. Z-buffer测试有问题")

print("=" * 60)
