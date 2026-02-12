#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
降维脚本: 应用多种降维方法到预处理后的embeddings

支持三种降维方法：
1. PCA (50D) + UMAP (2D) - 两步法，先线性降维再非线性
2. Direct UMAP (high-D → 2D) - 直接非线性降维
3. t-SNE (high-D → 2D) - 直接非线性降维
"""

import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import argparse
import time


def reduce_pca_umap(embeddings, n_pca_components=50, random_state=42):
    """
    方法1: 先PCA到50维，再UMAP到2维
    
    Args:
        embeddings: 输入embeddings (n_samples, hidden_size)
        n_pca_components: PCA保留的维度数
        random_state: 随机种子
    
    Returns:
        embeddings_2d: 降维后的2D坐标 (n_samples, 2)
    """
    print(f"\n[方法1: PCA+UMAP]")
    print(f"原始维度: {embeddings.shape}")
    
    # 步骤1: PCA降维到50维
    print(f"  步骤1: PCA降维到 {n_pca_components} 维...")
    start_time = time.time()
    pca = PCA(n_components=n_pca_components, random_state=random_state)
    embeddings_pca = pca.fit_transform(embeddings)
    pca_time = time.time() - start_time
    
    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"    - 耗时: {pca_time:.2f}s")
    print(f"    - PCA保留了 {explained_variance:.2%} 的方差")
    print(f"    - PCA后形状: {embeddings_pca.shape}")
    
    # 步骤2: UMAP降维到2维
    print(f"  步骤2: UMAP降维到 2 维...")
    start_time = time.time()
    umap_model = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        random_state=random_state,
        verbose=False
    )
    embeddings_2d = umap_model.fit_transform(embeddings_pca)
    umap_time = time.time() - start_time
    
    print(f"    - 耗时: {umap_time:.2f}s")
    print(f"    - UMAP后形状: {embeddings_2d.shape}")
    print(f"  总耗时: {pca_time + umap_time:.2f}s")
    
    return embeddings_2d


def reduce_direct_umap(embeddings, random_state=42):
    """
    方法2: 直接使用UMAP从高维降到2维
    
    Args:
        embeddings: 输入embeddings (n_samples, hidden_size)
        random_state: 随机种子
    
    Returns:
        embeddings_2d: 降维后的2D坐标 (n_samples, 2)
    """
    print(f"\n[方法2: Direct UMAP]")
    print(f"原始维度: {embeddings.shape}")
    
    print(f"  UMAP降维到 2 维...")
    start_time = time.time()
    umap_model = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        random_state=random_state,
        verbose=False
    )
    embeddings_2d = umap_model.fit_transform(embeddings)
    total_time = time.time() - start_time
    
    print(f"    - 耗时: {total_time:.2f}s")
    print(f"    - 输出形状: {embeddings_2d.shape}")
    
    return embeddings_2d


def reduce_tsne(embeddings, random_state=42):
    """
    方法3: 使用t-SNE从高维降到2维
    
    Args:
        embeddings: 输入embeddings (n_samples, hidden_size)
        random_state: 随机种子
    
    Returns:
        embeddings_2d: 降维后的2D坐标 (n_samples, 2)
    """
    print(f"\n[方法3: t-SNE]")
    print(f"原始维度: {embeddings.shape}")
    
    print(f"  t-SNE降维到 2 维...")
    start_time = time.time()
    tsne_model = TSNE(
        n_components=2,
        # perplexity=200,
        # max_iter=5000,
        # learning_rate=1000,
        random_state=random_state,
        # verbose=1
        perplexity=80,
        max_iter=4000,
        learning_rate=500,     
        init='pca',
        n_iter_without_progress=400,
        early_exaggeration=20.0
    )
    embeddings_2d = tsne_model.fit_transform(embeddings)
    total_time = time.time() - start_time
    
    print(f"    - 耗时: {total_time:.2f}s")
    print(f"    - 输出形状: {embeddings_2d.shape}")
    
    return embeddings_2d


def main():
    parser = argparse.ArgumentParser(
        description='应用降维方法到预处理后的embeddings'
    )
    parser.add_argument(
        '--preprocessed_file',
        type=str,
        required=True,
        help='预处理后的npz文件路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='输出目录 (通常是 results/{llm_family}/{model_name}/)'
    )
    parser.add_argument(
        '--methods',
        nargs='+',
        default=['pca_umap', 'umap', 'tsne'],
        choices=['pca_umap', 'umap', 'tsne'],
        help='要使用的降维方法列表'
    )
    parser.add_argument(
        '--pca_dims',
        type=int,
        default=50,
        help='PCA降维目标维度 (仅用于pca_umap方法)'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='随机种子'
    )
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"降维分析")
    print(f"输入文件: {args.preprocessed_file}")
    print(f"输出目录: {output_dir}")
    print(f"降维方法: {args.methods}")
    print(f"随机种子: {args.random_seed}")
    print("="*80)
    
    # 加载预处理后的数据
    print("\n加载预处理数据...")
    data = np.load(args.preprocessed_file, allow_pickle=True)
    embeddings = data['embeddings']
    language_labels = data['language_labels']
    
    print(f"  - Embeddings形状: {embeddings.shape}")
    print(f"  - 语言标签数量: {len(language_labels)}")
    
    # 应用各种降维方法
    for method in args.methods:
        print("\n" + "="*80)
        print(f"应用方法: {method.upper()}")
        print("="*80)
        
        # 根据方法选择降维函数
        if method == 'pca_umap':
            embeddings_2d = reduce_pca_umap(
                embeddings, 
                n_pca_components=args.pca_dims,
                random_state=args.random_seed
            )
            output_filename = "pca_umap_reduced_2d.npz"
        elif method == 'umap':
            embeddings_2d = reduce_direct_umap(
                embeddings,
                random_state=args.random_seed
            )
            output_filename = "umap_reduced_2d.npz"
        elif method == 'tsne':
            pca = PCA(n_components=50, random_state=args.random_seed)
            embeddings_for_tsne = pca.fit_transform(embeddings)
            embeddings_2d = reduce_tsne(
                embeddings_for_tsne,
                random_state=args.random_seed
            )
            output_filename = "tsne_reduced_2d.npz"
        else:
            print(f"未知方法: {method}")
            continue
        
        # 保存降维结果
        output_path = output_dir / output_filename
        np.savez_compressed(
            output_path,
            embeddings_2d=embeddings_2d,
            language_labels=language_labels
        )
        print(f"\n✓ 降维结果已保存到: {output_path}")
    
    # 打印总结
    print("\n" + "="*80)
    print("所有降维方法完成！")
    print("="*80)
    print(f"输出目录: {output_dir}")
    print("生成的文件:")
    for method in args.methods:
        if method == 'pca_umap':
            filename = "pca_umap_reduced_2d.npz"
        elif method == 'umap':
            filename = "umap_reduced_2d.npz"
        elif method == 'tsne':
            filename = "tsne_reduced_2d.npz"
        print(f"  - {filename}")
    print("="*80)


if __name__ == "__main__":
    main()

