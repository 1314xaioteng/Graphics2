"""
背景合成可视化诊断工具
======================

显示背景合成的各个中间步骤：
1. 原始深度图
2. Alpha遮罩（二值化）
3. Alpha遮罩（羽化后）
4. 前景图
5. 背景图（调整大小）
6. 最终合成结果

使用方法：
python debug_background_visual.py
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from npr_renderer import NPRRenderer


def visualize_background_compositing(model_path, background_path, output_prefix="debug"):
    """
    可视化背景合成的各个步骤

    Args:
        model_path: OBJ模型路径
        background_path: 背景图路径
        output_prefix: 输出文件前缀
    """
    print("=" * 60)
    print("背景合成可视化诊断工具")
    print("=" * 60)

    # 1. 加载模型
    print("\n[1/6] 加载模型...")
    renderer = NPRRenderer()
    model = renderer.load_obj_simple(model_path)
    print(f"    模型: {model['name']}")
    print(f"    顶点数: {model['stats']['vertices']}")
    print(f"    面数: {model['stats']['faces']}")

    # 2. 渲染深度图和法线图
    print("\n[2/6] 渲染深度图和法线图...")
    image_size = 512
    depth_map, normal_map = renderer.render_depth_normal(model, image_size)
    print(f"    深度图范围: [{depth_map.min():.4f}, {depth_map.max():.4f}]")
    print(f"    非零像素: {np.sum(depth_map > 0.001)}/{depth_map.size} ({100*np.sum(depth_map > 0.001)/depth_map.size:.1f}%)")

    # 3. 生成前景（NPR渲染）
    print("\n[3/6] 生成前景图...")
    color_map = renderer._generate_color_map(depth_map, normal_map, image_size)
    edge_map = renderer.detect_edges(depth_map, normal_map)
    foreground = renderer.apply_sketch_style(color_map, edge_map, depth_map, strength=0.9)
    foreground = np.flipud(foreground)
    depth_map_flipped = np.flipud(depth_map)  # 翻转深度图以匹配前景

    # 4. 加载背景
    print("\n[4/6] 加载背景图...")
    background = np.array(Image.open(background_path).convert('RGB'))
    print(f"    背景尺寸: {background.shape}")
    bg_resized = cv2.resize(background, (image_size, image_size))

    # 5. 生成Alpha遮罩（步骤分解）
    print("\n[5/6] 生成Alpha遮罩...")

    # 5a. 二值化Alpha
    alpha_binary = (depth_map_flipped > 0.001).astype(np.float32)
    fg_pixels = np.sum(alpha_binary)
    print(f"    二值化阈值: > 0.001")
    print(f"    前景像素: {fg_pixels}/{alpha_binary.size} ({100*fg_pixels/alpha_binary.size:.1f}%)")

    # 5b. 对比度增强
    alpha_enhanced = np.clip(alpha_binary * 1.5, 0, 1)

    # 5c. 羽化
    alpha_feathered = cv2.GaussianBlur(alpha_enhanced, (15, 15), 0)
    print(f"    羽化核大小: 15x15")
    print(f"    Alpha范围: [{alpha_feathered.min():.4f}, {alpha_feathered.max():.4f}]")

    # 6. 合成
    print("\n[6/6] 合成...")
    alpha_3ch = np.expand_dims(alpha_feathered, axis=2)
    result = foreground * alpha_3ch + bg_resized * (1 - alpha_3ch)
    result = result.astype(np.uint8)

    # 创建可视化图表
    print("\n生成可视化图表...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('背景合成诊断 - 各步骤可视化', fontsize=16, fontweight='bold')

    # 第一行：输入
    axes[0, 0].imshow(depth_map_flipped, cmap='gray')
    axes[0, 0].set_title(f'1. 深度图\n范围: [{depth_map_flipped.min():.3f}, {depth_map_flipped.max():.3f}]')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(foreground)
    axes[0, 1].set_title('2. 前景图 (NPR渲染)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(bg_resized)
    axes[0, 2].set_title(f'3. 背景图 (调整大小)\n{image_size}x{image_size}')
    axes[0, 2].axis('off')

    # 第二行：Alpha处理
    axes[1, 0].imshow(alpha_binary, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(f'4. Alpha遮罩 (二值化)\n前景: {100*fg_pixels/alpha_binary.size:.1f}%')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(alpha_enhanced, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('5. Alpha遮罩 (对比度增强)')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(alpha_feathered, cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title(f'6. Alpha遮罩 (羽化)\n范围: [{alpha_feathered.min():.3f}, {alpha_feathered.max():.3f}]')
    axes[1, 2].axis('off')

    # 第三行：合成结果和对比
    axes[2, 0].imshow(result)
    axes[2, 0].set_title('7. 最终合成结果', fontweight='bold')
    axes[2, 0].axis('off')

    # 显示前景和背景的混合权重
    axes[2, 1].imshow(foreground * alpha_3ch)
    axes[2, 1].set_title('8. 前景部分 (foreground × alpha)')
    axes[2, 1].axis('off')

    axes[2, 2].imshow((bg_resized * (1 - alpha_3ch)).astype(np.uint8))
    axes[2, 2].set_title('9. 背景部分 (background × (1-alpha))')
    axes[2, 2].axis('off')

    plt.tight_layout()

    # 保存可视化图表
    viz_path = f"{output_prefix}_visualization.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 可视化图表已保存: {viz_path}")

    # 保存单独的图像
    Image.fromarray((depth_map_flipped * 255).astype(np.uint8)).save(f"{output_prefix}_1_depth.png")
    Image.fromarray(foreground).save(f"{output_prefix}_2_foreground.png")
    Image.fromarray(bg_resized).save(f"{output_prefix}_3_background.png")
    Image.fromarray((alpha_feathered * 255).astype(np.uint8)).save(f"{output_prefix}_4_alpha.png")
    Image.fromarray(result).save(f"{output_prefix}_5_result.png")

    print(f"✓ 单独图像已保存:")
    print(f"  - {output_prefix}_1_depth.png (深度图)")
    print(f"  - {output_prefix}_2_foreground.png (前景)")
    print(f"  - {output_prefix}_3_background.png (背景)")
    print(f"  - {output_prefix}_4_alpha.png (Alpha遮罩)")
    print(f"  - {output_prefix}_5_result.png (最终结果)")

    # 显示图表
    plt.show()

    print("\n" + "=" * 60)
    print("诊断完成！")
    print("=" * 60)

    # 诊断建议
    print("\n【诊断建议】")
    fg_ratio = 100 * fg_pixels / alpha_binary.size
    if fg_ratio < 20:
        print(f"⚠️  前景占比太小 ({fg_ratio:.1f}%)")
        print("   建议：增大模型缩放比例（npr_renderer.py line 158: vertices *= 0.8）")
    elif fg_ratio > 70:
        print(f"⚠️  前景占比太大 ({fg_ratio:.1f}%)")
        print("   建议：减小模型缩放比例（npr_renderer.py line 158: vertices *= 0.4）")
    else:
        print(f"✓ 前景占比合理 ({fg_ratio:.1f}%)")

    if alpha_feathered.max() < 0.8:
        print(f"⚠️  Alpha遮罩最大值过低 ({alpha_feathered.max():.3f})")
        print("   建议：减小羽化核大小或增加对比度增强系数")
    else:
        print(f"✓ Alpha遮罩对比度良好")

    if depth_map_flipped.max() < 0.5:
        print(f"⚠️  深度图动态范围过小 ({depth_map_flipped.max():.3f})")
        print("   建议：检查模型是否过于平坦")
    else:
        print(f"✓ 深度图动态范围正常")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='背景合成可视化诊断工具')
    parser.add_argument('--model', '-m', default='cube.obj', help='OBJ模型路径')
    parser.add_argument('--background', '-b', required=True, help='背景图路径')
    parser.add_argument('--output', '-o', default='debug', help='输出文件前缀')

    args = parser.parse_args()

    visualize_background_compositing(args.model, args.background, args.output)


if __name__ == '__main__':
    main()
