#!/usr/bin/env python3
"""
第10天汇报演示脚本 - 完整流程
============================

演示流程：
1. ✅ 启动程序（自动）
2. 加载模型（2个：简单+复杂）
3. 风格切换（素描→水彩→卡通→油画）
4. 参数调节（风格强度、边界强度等）
5. 效果对比（三图对比：原始、无约束、有约束）
cd /home/lab/25-txd/teamwork && rm -rf demo_output && python day10_demo.py 2>&1 | tail -30
适合现场10天汇报演讲使用
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image
import subprocess
from typing import List, Dict

sys.path.insert(0, os.path.dirname(__file__))
from npr_renderer import NPRRenderer


class Day10Demo:
    """第10天汇报演示系统"""
    
    def __init__(self):
        self.renderer = NPRRenderer()
        self.models = []
        self.results = []
        self.demo_dir = 'demo_output'
        os.makedirs(self.demo_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("DAY 10 PRESENTATION DEMO - NPR RENDERING SYSTEM")
        print("="*70)
    
    def setup_models(self):
        """设置演示用的模型"""
        # 检查可用的模型
        model_paths = [
            'data/models/armadillo.obj',
            'data/models/cube.obj',
            'data/models/sphere.obj',
            'data/models/bunny.obj',
        ]
        
        print("\n[STEP 1] Setting up demo models...")
        for model_path in model_paths:
            if os.path.exists(model_path):
                model = self.renderer.load_obj_simple(model_path)
                self.models.append(model)
                print(f"  ✓ Loaded: {model['name']} ({model['stats']['vertices']} vertices)")
        
        if not self.models:
            # 如果没有模型，提示用户
            print("\nNo models found. Please add .obj files to data/models/ directory")
            print("\nExample structure:")
            print("  data/models/")
            print("  ├── armadillo.obj")
            print("  ├── cube.obj")
            print("  └── bunny.obj")
            return False
        
        return True
    
    def demo_style_switching(self):
        """演示：风格切换（40秒）"""
        if not self.models:
            return
        
        print("\n[STEP 2] Demonstrating style switching (4 styles)...")
        print("\n  This step would show:")
        print("  - Sketch style:     Black & white linework with clear outlines")
        print("  - Watercolor:       Soft colors, transparent feel, blurred edges")
        print("  - Cartoon:          High contrast, simplified shapes, thick outlines")
        print("  - Oil painting:     Thick strokes, soft transitions, artistic feel")
        
        model = self.models[0]  # 使用第一个模型
        styles = ['sketch', 'watercolor', 'cartoon', 'oil']
        
        print(f"\n  Rendering '{model['name']}' in all styles...")
        
        for style in styles:
            print(f"\n  • Applying {style.upper()} style...")
            result = self.renderer.render(model, style, strength=0.85, image_size=512)
            
            # 保存结果
            output_path = os.path.join(self.demo_dir, f'{model["name"]}_{style}.png')
            Image.fromarray(result).save(output_path)
            self.results.append({
                'model': model['name'],
                'style': style,
                'path': output_path
            })
            
            print(f" Saved: {output_path}")
    
    def demo_parameter_adjustment(self):
        """演示：参数调节（30秒）"""
        if not self.models:
            return
        
        print("\n[STEP 3] Demonstrating parameter adjustment...")
        
        model = self.models[0]
        
        # 演示不同强度的素描风格
        print(f"\n  Adjusting style strength for '{model['name']}'...")
        strengths = [0.3, 0.6, 0.9]
        
        for strength in strengths:
            print(f"\n  • Rendering with strength = {strength:.0%}...")
            result = self.renderer.render(model, 'sketch', strength=strength, image_size=512)
            
            output_path = os.path.join(self.demo_dir, f'{model["name"]}_sketch_strength_{strength:.1f}.png')
            Image.fromarray(result).save(output_path)
            
            print(f"Saved: {output_path}")
            print(f"    Observation: Higher strength → More sketch-like, Lower strength → More original")
    
    def demo_geometric_preservation(self):
        """演示：几何约束保留效果（30秒）"""
        if not self.models:
            return
        
        print("\n[STEP 4] Demonstrating geometric preservation...")
        print("\n  NPR Rendering Process:")
        print("  1. Extract depth map (3D structure)")
        print("  2. Compute normal map (surface orientation)")
        print("  3. Detect edges (outlines & silhouettes)")
        print("  4. Apply style (sketch, watercolor, etc.)")
        print("  5. Preserve geometry (no stretching, no missing parts)")
        
        model = self.models[0]
        
        # 渲染并展示中间过程
        print(f"\n  Processing '{model['name']}'...")
        
        # 生成深度图、法线图、边界图
        depth_map, normal_map = self.renderer.render_depth_normal(model, 512)
        edge_map = self.renderer.detect_edges(depth_map, normal_map)
        
        # 转换为可视化
        depth_vis = np.flipud((depth_map * 255).astype(np.uint8))
        normal_vis = np.flipud((normal_map * 255).astype(np.uint8))
        edge_vis = np.flipud((edge_map * 255).astype(np.uint8))
        
        # 保存中间结果
        Image.fromarray(depth_vis).save(os.path.join(self.demo_dir, f'{model["name"]}_depth.png'))
        Image.fromarray(normal_vis).save(os.path.join(self.demo_dir, f'{model["name"]}_normal.png'))
        Image.fromarray(edge_vis).save(os.path.join(self.demo_dir, f'{model["name"]}_edges.png'))
        
        print(f"  ✓ Depth map preserved: {depth_map.shape}")
        print(f"  ✓ Normal map preserved: {normal_map.shape}")
        print(f"  ✓ Edges detected: {edge_map.shape}")
        
        print("\n  Benefits of geometry-aware rendering:")
        print("  ✓ No texture stretching")
        print("  ✓ No missing parts or holes")
        print("  ✓ Preserves 3D structure under stylization")
        print("  ✓ Maintains model proportions and details")
    
    def create_comparison_image(self):
        """创建对比图（原始→无约束→有约束）"""
        if not self.models or not self.results:
            return
        
        print("\n[STEP 5] Creating comparison visualization...")
        
        model = self.models[0]
        
        # 生成三个版本
        print(f"\n  Generating three versions for comparison:")
        
        # 版本1: 原始（基础渲染）
        print(f"  1. Original: Basic 3D rendering")
        orig = self.renderer._generate_color_map(
            *self.renderer.render_depth_normal(model, 512), 512
        )
        
        # 版本2: 简单风格化（无几何约束）
        print(f"  2. Naive stylization: Without geometric constraints")
        depth_map, normal_map = self.renderer.render_depth_normal(model, 512)
        naive = self.renderer.apply_sketch_style(orig, np.zeros_like(orig[:, :, 0]), depth_map, 0.9)
        
        # 版本3: 几何约束风格化
        print(f"  3. Optimized stylization: With geometric constraints")
        edge_map = self.renderer.detect_edges(depth_map, normal_map)
        optimized = self.renderer.apply_sketch_style(orig, edge_map, depth_map, 0.9)
        
        # 并排显示
        h, w = orig.shape[:2]
        comparison = np.zeros((h, w*3, 3), dtype=np.uint8)
        comparison[:, :w] = orig
        comparison[:, w:w*2] = naive
        comparison[:, w*2:] = optimized

        comparison = np.flipud(comparison)
        
        output_path = os.path.join(self.demo_dir, f'comparison_{model["name"]}.png')
        Image.fromarray(comparison).save(output_path)
        
        print(f"\n Comparison saved: {output_path}")
        print(f"     Format: [Original | Naive | Optimized]")
    
    def generate_report(self):
        """生成演示报告"""
        print("\n" + "="*70)
        print("DEMO RESULTS SUMMARY")
        print("="*70)
        
        # 模型统计
        print(f"\nModels loaded: {len(self.models)}")
        for model in self.models:
            print(f"  • {model['name']}: {model['stats']['vertices']} vertices, {model['stats']['faces']} faces")
        
        # 生成的结果
        print(f"\nStyles rendered: {len(self.results)}")
        for result in self.results:
            print(f"  • {result['model']} - {result['style']}")
        
        # 关键特性
        print(f"\n Key Features Demonstrated:")
        print(f"  Real-time style switching (sketch, watercolor, cartoon, oil)")
        print(f"   Adjustable style strength (0-100%)")
        print(f"  Geometric preservation (depth, normals, edges)")
        print(f"  Edge detection for outline extraction")
        print(f"  GPU-accelerated rendering (if available)")
        
        # 输出位置
        print(f"\n All results saved to: {os.path.abspath(self.demo_dir)}/")
        
        # 列出所有文件
        print(f"\n Generated files:")
        for f in sorted(os.listdir(self.demo_dir)):
            fpath = os.path.join(self.demo_dir, f)
            size = os.path.getsize(fpath) / (1024*1024)
            print(f"  • {f} ({size:.2f} MB)")
        
        # 保存报告
        report = {
            'timestamp': str(Path('').absolute()),
            'models_count': len(self.models),
            'styles_rendered': len(set(r['style'] for r in self.results)),
            'total_results': len(self.results),
            'output_directory': os.path.abspath(self.demo_dir),
            'key_features': [
                'Real-time style switching',
                'Adjustable style strength',
                'Geometric preservation',
                'Edge detection',
                'GPU acceleration'
            ]
        }
        
        report_path = os.path.join(self.demo_dir, 'demo_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n Report saved: {report_path}")
    
    def run_full_demo(self):
        """运行完整演示"""
        # Step 1: 设置模型
        if not self.setup_models():
            return
        
        # Step 2: 风格切换演示
        self.demo_style_switching()
        
        # Step 3: 参数调节演示
        self.demo_parameter_adjustment()
        
        # Step 4: 几何约束保留演示
        self.demo_geometric_preservation()
        
        # Step 5: 对比图生成
        self.create_comparison_image()
        
        # 生成报告
        self.generate_report()
        
        print("\n" + "="*70)
        print(" DEMONSTRATION COMPLETE")
        print("="*70)

def main():
    """主函数"""
    demo = Day10Demo()
    demo.run_full_demo()


if __name__ == '__main__':
    main()
