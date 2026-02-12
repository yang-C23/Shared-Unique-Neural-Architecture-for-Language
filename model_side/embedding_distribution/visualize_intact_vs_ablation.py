#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化：Intact 与 Core-language-ablation 模型在三种语言下的 embedding 分布
优化版 V2：适配论文 Fig 2a 展示，高清晰度，图例优化（白底+实心点）
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# 设置全局字体和线宽，确保缩小后依然清晰
plt.rcParams['axes.linewidth'] = 2.5  # 全局边框加粗

def load_model_language_data(model_name, language, llm_family, method, results_dir):
    """加载特定模型的特定语言数据"""
    npz_file = Path(results_dir) / llm_family / model_name / f"{method}_reduced_2d.npz"
    
    if not npz_file.exists():
        raise FileNotFoundError(f"文件不存在: {npz_file}")
    
    data = np.load(npz_file, allow_pickle=True)
    languages = data['language_labels']
    lang_mask = (languages == language)
    
    return data['embeddings_2d'][lang_mask]

def sample_data(data, max_points=5000):
    """采样数据"""
    if len(data) > max_points:
        idx = np.random.choice(len(data), max_points, replace=False)
        return data[idx]
    return data

def main():
    parser = argparse.ArgumentParser(description='可视化 embedding 分布 (论文发表级)')
    parser.add_argument('--results_dir', type=str, 
                       default='/leonardo_work/EUHPC_B24_036/yang/embedding_distribution/results',
                       help='结果根目录')
    parser.add_argument('--output_file', type=str, help='输出图片路径')
    parser.add_argument('--max_points', type=int, default=500, help='采样点数')
    parser.add_argument('--llm_family', type=str, default='qwen', choices=['qwen', 'llama', 'mistral'])
    parser.add_argument('--method', type=str, default='pca_umap', choices=['pca_umap', 'umap', 'tsne'])
    args = parser.parse_args()
    
    # 自动生成输出文件名
    if not args.output_file:
        output_dir = Path(args.results_dir) / args.llm_family
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output_file = str(output_dir / f"intact_vs_ablation_{args.method}_pub_v2.png")
    
    np.random.seed(42)
    
    model_families = {
        'qwen': {'intact': 'qwen2.5-7b', 'core_ablation': 'qwen2.5-7b_Break001'},
        'llama': {'intact': 'llama2_7b', 'core_ablation': 'llama2_7b_Break001'},
        'mistral': {'intact': 'mistral-base', 'core_ablation': 'mistral_Break001'}
    }
    
    models = model_families[args.llm_family]
    languages = ['CN', 'EN', 'FR']
    color_palettes = {
        'intact': {'CN': '#E5B893', 'EN': '#BEDCED', 'FR': '#C0E7DA'},
        'core_ablation': {'CN': '#D26101', 'EN': '#0073B1', 'FR': '#2DA248'}
    }
    model_labels = {'intact': 'Intact', 'core_ablation': 'Core-language-ablation'}
    
    # 1. 保持大尺寸画布
    fig, ax = plt.subplots(figsize=(12, 12.85)) 
    
    all_points = []
    plotted_labels = set()
    
    for model_key in ['intact', 'core_ablation']:
        model_name = models[model_key]
        label_prefix = model_labels[model_key]
        colors = color_palettes[model_key]
        
        # 散点图本身的透明度保持不变，以便观察重叠
        alpha = 0.7 if model_key == 'intact' else 0.4 
        
        print(f"\n加载 {label_prefix} ({model_name})...")
        for lang in languages:
            try:
                data = load_model_language_data(model_name, lang, args.llm_family,
                                                args.method, args.results_dir)
                data = sample_data(data, args.max_points)
                all_points.append(data)
                
                label = f'{label_prefix}-{lang}'
                display_label = label if label not in plotted_labels else None
                
                # 2. 绘制散点 (s=45 保证可见性)
                ax.scatter(
                    data[:, 0], data[:, 1],
                    color=colors[lang],
                    alpha=alpha,
                    marker='o',
                    s=45,  
                    edgecolors='none', 
                    label=display_label
                )
                plotted_labels.add(label)
            except FileNotFoundError:
                print(f"  警告: {lang} 数据未找到")

    if all_points:
        stacked = np.vstack(all_points)
        x_low, x_high = np.percentile(stacked[:, 0], [1, 99])
        y_low, y_high = np.percentile(stacked[:, 1], [1, 99])
        x_range = x_high - x_low
        y_range = y_high - y_low
        
        # 3. 增大坐标轴跨度 (Pad 0.15)
        pad_factor = 0.15 
        pad_x = pad_factor * x_range if x_range > 0 else 1.0
        pad_y = pad_factor * y_range if y_range > 0 else 1.0
        ax.set_xlim(x_low - pad_x, x_high + pad_x)
        ax.set_ylim(y_low - pad_y, y_high + pad_y)

    # 4. 标签与标题设置 (大字号)
    ax.set_xlabel('Dimension 1', fontsize=24, fontweight='bold', labelpad=15)
    ax.set_ylabel('Dimension 2', fontsize=24, fontweight='bold', labelpad=15)
    ax.set_title('Embedding Distribution Comparison', fontsize=28, fontweight='bold', pad=25)
    
    # 5. 刻度设置 (大字号、粗线条)
    ax.tick_params(axis='both', which='major', labelsize=20, width=2.5, length=8, pad=8)
    
    # ================= [关键修改] 图例优化 =================
    leg = ax.legend(
        loc='upper right', 
        fontsize=18, 
        ncol=1, 
        frameon=True,          # 开启边框背景
        facecolor='white',     # 设置背景为纯白
        framealpha=1.0,        # 背景不透明 (1.0)
        edgecolor='#cccccc',   # 边框颜色 (浅灰，不抢眼)
        markerscale=3.0,       # 放大图例中的点
        handletextpad=0.5,
        borderpad=0.8          # 图例内容与边框的距离
    )
    
    # 强制让图例中的点变得不透明 (alpha=1.0)，即使图中的点是半透明的
    # 这样图例看起来颜色更实、更清晰
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
    # ======================================================
    
    # 6. 边框处理
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    # left/bottom 已经在全局 rcParams 中加粗，这里确保一下
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(2.5) 
    
    ax.grid(False)
    fig.tight_layout()
    
    plt.savefig(args.output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ 已保存: {args.output_file}")

if __name__ == '__main__':
    main()