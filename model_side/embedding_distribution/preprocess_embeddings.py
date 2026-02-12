#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预处理模型embeddings: 加载和聚合为word-level embeddings

这个脚本：
1. 加载模型在三种语言（EN, CN, FR）下生成的token级别embeddings
2. 通过mean pooling将subword tokens聚合为word-level embeddings
3. 保存预处理后的数据供后续降维使用
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def load_embeddings_for_language(model_path, language):
    """
    加载特定语言的所有embeddings和对应的CSV文件
    
    Args:
        model_path: 模型embedding目录路径
        language: 语言代码 (EN, CN, FR)
    
    Returns:
        embeddings: 所有token的embeddings数组
        csv_df: 对应的CSV DataFrame
    """
    model_path = Path(model_path)
    
    # 读取CSV文件获取token信息
    csv_file = model_path / f"task-lpp{language}.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV文件不存在: {csv_file}")
    
    csv_df = pd.read_csv(csv_file, index_col=0)
    print(f"  - CSV文件: {len(csv_df)} 个tokens")
    
    # 加载所有runs的embedding文件
    all_embeddings = []
    sections = sorted(csv_df['section'].unique())
    
    for section in sections:
        npy_file = model_path / f"task-lpp{language}_run-{section}.npy"
        if not npy_file.exists():
            print(f"  警告: 文件不存在 {npy_file}")
            continue
        
        emb = np.load(npy_file)  # shape: (1, n_tokens, hidden_size)
        emb = emb.squeeze(0)  # 移除batch维度: (n_tokens, hidden_size)
        all_embeddings.append(emb)
    
    # 拼接所有sections的embeddings
    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"  - Embeddings形状: {embeddings.shape}")
    
    # 验证embeddings和CSV行数是否匹配
    if len(embeddings) != len(csv_df):
        print(f"  警告: embeddings数量({len(embeddings)})与CSV行数({len(csv_df)})不匹配!")
    
    return embeddings, csv_df


def aggregate_subwords_to_words(embeddings, csv_df):
    """
    将subword token embeddings聚合为word-level embeddings
    
    使用mean pooling将属于同一个单词的多个subword tokens的embeddings平均化
    
    Args:
        embeddings: token级别的embeddings (n_tokens, hidden_size)
        csv_df: 包含word_idx信息的DataFrame
    
    Returns:
        word_embeddings: word级别的embeddings (n_words, hidden_size)
        word_indices: 对应的word索引列表
    """
    # 按word_idx分组进行mean pooling
    word_embeddings = []
    word_indices = []
    
    for word_idx, group in csv_df.groupby('word_idx'):
        # 获取该单词对应的所有token的索引
        token_indices = group.index.tolist()
        
        # 提取对应的embeddings并求平均
        word_emb = embeddings[token_indices].mean(axis=0)
        
        word_embeddings.append(word_emb)
        word_indices.append(word_idx)
    
    word_embeddings = np.stack(word_embeddings)
    
    print(f"  - 聚合前: {len(embeddings)} tokens")
    print(f"  - 聚合后: {len(word_embeddings)} words")
    
    return word_embeddings, word_indices


def load_and_preprocess_model_embeddings(model_path, languages=['EN', 'CN', 'FR']):
    """
    加载并预处理模型在所有语言下的embeddings
    
    Args:
        model_path: 模型embedding目录路径
        languages: 要处理的语言列表
    
    Returns:
        all_embeddings: 所有语言的word-level embeddings拼接 (n_total_words, hidden_size)
        language_labels: 每个embedding对应的语言标签
    """
    all_embeddings = []
    language_labels = []
    
    for lang in languages:
        print(f"\n加载 {lang} 语言数据...")
        
        # 加载token级别的embeddings
        token_embeddings, csv_df = load_embeddings_for_language(model_path, lang)
        
        # 聚合为word级别
        word_embeddings, _ = aggregate_subwords_to_words(token_embeddings, csv_df)
        
        all_embeddings.append(word_embeddings)
        language_labels.extend([lang] * len(word_embeddings))
    
    # 拼接所有语言的embeddings
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    language_labels = np.array(language_labels)
    
    print(f"\n总计: {len(all_embeddings)} 个words from {len(languages)} 种语言")
    print(f"Embedding维度: {all_embeddings.shape[1]}")
    
    return all_embeddings, language_labels


def main():
    parser = argparse.ArgumentParser(
        description='预处理模型embedding: 从token级别聚合到word级别'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='模型embedding目录路径 (如: scratch_link/embeddings/qwen2.5-7b)'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='模型名称（用于输出文件名）'
    )
    parser.add_argument(
        '--llm_family',
        type=str,
        default='qwen',
        choices=['qwen', 'llama', 'mistral'],
        help='LLM家族名称 (qwen/llama/mistral)'
    )
    parser.add_argument(
        '--languages',
        nargs='+',
        default=['EN', 'CN', 'FR'],
        help='要分析的语言列表'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='embedding_distribution/results',
        help='输出目录'
    )
    
    args = parser.parse_args()
    
    # 创建输出目录结构: results/{llm_family}/{model_name}/
    output_dir = Path(args.output_dir) / args.llm_family / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"预处理模型: {args.model_name}")
    print(f"LLM家族: {args.llm_family}")
    print(f"模型路径: {args.model_path}")
    print(f"语言: {args.languages}")
    print(f"输出目录: {output_dir}")
    print("="*80)
    
    # 加载和预处理embeddings
    print("\n[步骤 1/2] 加载和预处理embeddings...")
    embeddings, language_labels = load_and_preprocess_model_embeddings(
        args.model_path,
        args.languages
    )
    
    # 保存预处理后的数据
    print("\n[步骤 2/2] 保存预处理数据...")
    preprocessed_path = output_dir / "preprocessed.npz"
    np.savez_compressed(
        preprocessed_path,
        embeddings=embeddings,
        language_labels=language_labels
    )
    print(f"预处理数据已保存到: {preprocessed_path}")
    
    # 打印统计信息
    print("\n" + "="*80)
    print("预处理完成！统计信息：")
    print("="*80)
    for lang in args.languages:
        count = (language_labels == lang).sum()
        print(f"{lang}: {count} words")
    print(f"总计: {len(embeddings)} words")
    print(f"Embedding维度: {embeddings.shape[1]}")
    print(f"\n输出文件: {preprocessed_path}")
    print("="*80)


if __name__ == "__main__":
    main()

