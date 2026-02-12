import torch
from transformers import AutoModelForCausalLM
import csv
from tqdm import tqdm
import random
import os
import argparse
import numpy as np


def jaccard_similarity(tensor1, tensor2):
    # Compute the intersection (common elements)
    intersection = torch.logical_and(tensor1, tensor2).sum()
    # Compute the union (all unique elements)
    union = torch.logical_or(tensor1, tensor2).sum()
    # Calculate the Jaccard Index (IoU)
    iou = intersection.float() / union.float()

    return iou.item()

def compare_bool_matrix(bool_dict1, bool_dict2):
    params_diff_similarity = {}
    for (name1, bool_matrix1), (name2, bool_matrix2) in zip(bool_dict1.items(), bool_dict2.items()):
        assert name1 == name2
        params_diff_similarity[name1] = jaccard_similarity(bool_matrix1,bool_matrix2)
    return params_diff_similarity

def logical_and_bool_matrix(bool_dict1, bool_dict2):
    params_diff_similarity = {}
    for (name1, bool_matrix1), (name2, bool_matrix2) in zip(bool_dict1.items(), bool_dict2.items()):
        assert name1 == name2
        params_diff_similarity[name1] = torch.logical_and(bool_matrix1,bool_matrix2)
    return params_diff_similarity

def logical_or_bool_matrix(bool_dict1, bool_dict2):
    params_diff_similarity = {}
    for (name1, bool_matrix1), (name2, bool_matrix2) in zip(bool_dict1.items(), bool_dict2.items()):
        assert name1 == name2
        params_diff_similarity[name1] = torch.logical_or(bool_matrix1,bool_matrix2)
    return params_diff_similarity

def calculate_row_bool_matrix(bool_dict1, num):
    params_top_row = {}
    params_top_col = {}
    params_bottom_row = {}
    params_bottom_col = {}
    
    for (name1, bool_matrix1) in bool_dict1.items():
        row_sums = torch.sum(bool_matrix1, dim=1)
        col_sums = torch.sum(bool_matrix1, dim=0)
        top_rows_indices = torch.topk(row_sums, k=num).indices.tolist()
        top_cols_indices = torch.topk(col_sums, k=num).indices.tolist()
        min_rows_indices = torch.topk(row_sums, k=num, largest=False).indices.tolist()
        min_cols_indices = torch.topk(col_sums, k=num, largest=False).indices.tolist()
        params_top_row[name1] = top_rows_indices
        params_top_col[name1] = top_cols_indices
        params_bottom_row[name1] = min_rows_indices
        params_bottom_col[name1] = min_cols_indices
    return params_top_row,params_top_col,params_bottom_row,params_bottom_col

def get_top_bottom_tensor(tensor_diff,k):
    # 计算需要记录的点的数量（最大/最小/中间的k%）
    num_points = int(k * tensor_diff.numel())

    # 找到差异幅度最大的前k%的点
    max_points = tensor_diff.view(-1).topk(num_points).indices
    bool_sensor_max = torch.zeros(tensor_diff.shape, dtype=torch.bool)
    bool_sensor_max.view(-1)[max_points] = True

    # 找到差异幅度最小的前k%的点
    min_points = tensor_diff.view(-1).topk(num_points, largest=False).indices
    bool_sensor_min = torch.zeros(tensor_diff.shape, dtype=torch.bool)
    bool_sensor_min.view(-1)[min_points] = True

    # 随机找k%的点
    bool_sensor_random = torch.zeros(tensor_diff.shape, dtype=torch.bool)
    # 随机选择要设置为True的元素的索引
    random_points = random.sample(range(bool_sensor_random.numel()), num_points)
    # 将选定的索引位置设置为True
    bool_sensor_random.view(-1)[random_points] = True

    return bool_sensor_max,bool_sensor_min,bool_sensor_random

def min_max_normalize_tensor(tensor, epsilon=1e-8):
    """
    对tensor进行Min-Max归一化
    normalized_value = (value - min_value) / (max_value - min_value + epsilon)
    """
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    
    # 避免分母为0
    denominator = max_val - min_val + epsilon
    normalized = (tensor - min_val) / denominator
    
    return normalized

def calculate_softmax_language_specific_score(importance_tensors):
    """
    计算基于Softmax加权法的language specific score (带Min-Max归一化)
    importance_tensors: dict with language names as keys and importance tensors as values
    返回每种语言的language specific score
    
    公式：
    Normalized_I_lang(p) = Min-Max Normalization(I_lang(p))
    P(lang|p) = exp(I_lang(p)) / Σ_i exp(I_i(p))
    S_lang(p) = Normalized_I_lang(p) * P(lang|p)
    """
    language_scores = {}
    languages = list(importance_tensors.keys())
    
    # 1. 对每种语言的重要性进行Min-Max归一化
    normalized_importance_tensors = {}
    for lang in languages:
        normalized_importance_tensors[lang] = min_max_normalize_tensor(importance_tensors[lang])
    
    # 2. 将原始重要性tensor堆叠起来用于计算softmax权重
    stacked_tensors = torch.stack([importance_tensors[lang] for lang in languages], dim=0)
    
    # 3. 使用PyTorch内置的softmax函数计算权重 (基于原始重要性)
    softmax_probs = torch.softmax(stacked_tensors, dim=0)
    
    # 4. 计算每种语言的language specific score: S_lang(p) = Normalized_I_lang(p) * P(lang|p)
    for i, lang in enumerate(languages):
        normalized_importance = normalized_importance_tensors[lang]
        relative_specificity = softmax_probs[i]
        language_scores[lang] = normalized_importance * relative_specificity
    
    return language_scores

def calculate_snr_language_specific_score(importance_tensors, epsilon=1e-8):
    """
    计算基于信噪比方法的language specific score
    importance_tensors: dict with language names as keys and importance tensors as values
    epsilon: 防止分母为0的小常数
    返回每种语言的language specific score
    
    公式：
    S_lang(p) = I_lang(p) / ((Σ_{i≠lang} I_i(p)) / (N-1) + ε)
    其中N是语言总数
    """
    language_scores = {}
    languages = list(importance_tensors.keys())
    
    for target_lang in languages:
        # 获取目标语言的重要性
        target_importance = importance_tensors[target_lang]
        
        # 计算其他语言重要性的平均值作为"噪声"
        other_languages = [lang for lang in languages if lang != target_lang]
        noise_sum = torch.zeros_like(target_importance)
        
        for other_lang in other_languages:
            noise_sum += importance_tensors[other_lang]
        
        # 计算平均噪声
        average_noise = noise_sum / len(other_languages)
        
        # 计算信噪比: S_lang(p) = I_lang(p) / (average_noise + ε)
        snr_score = target_importance / (average_noise + epsilon)
        language_scores[target_lang] = snr_score
    
    return language_scores

def calculate_layer_wise_rank_scores(importance_tensors, language_list):
    """
    按层计算rank特异性得分（高效版本）
    
    对每一层的参数分别进行排序，然后计算百分位排名和特异性得分
    
    Args:
        importance_tensors: dict with language names as keys and importance tensors as values
        language_list: list of language names
    
    Returns:
        dict with language names as keys and specificity score tensors as values
    """
    language_scores = {}
    
    # 为每种语言计算特异性得分
    for target_lang in language_list:
        target_tensor = importance_tensors[target_lang]
        
        # 创建结果tensor，保持原始形状
        specificity_tensor = torch.zeros_like(target_tensor, dtype=torch.float32)
        
        # 展平所有tensor用于排序，但保持形状信息用于重构
        flat_tensors = {}
        for lang in language_list:
            flat_tensors[lang] = importance_tensors[lang].flatten().float()
        
        # 计算每种语言在当前层的百分位排名
        percentiles = {}
        for lang in language_list:
            flat_tensor = flat_tensors[lang]
            
            # 对当前层的数据进行排序，获取排名
            sorted_values, sorted_indices = torch.sort(flat_tensor)
            
            # 创建排名tensor（从0开始）
            ranks = torch.zeros_like(flat_tensor)
            ranks[sorted_indices] = torch.arange(len(flat_tensor), dtype=torch.float32)
            
            # 转换为百分位数 (0到1之间)
            if len(flat_tensor) > 1:
                percentiles[lang] = ranks / (len(flat_tensor) - 1)
            else:
                percentiles[lang] = torch.zeros_like(ranks)
        
        # 计算目标语言的特异性得分
        target_percentiles = percentiles[target_lang]
        
        # 计算其他语言百分位的最大值
        other_langs = [lang for lang in language_list if lang != target_lang]
        max_other_percentiles = percentiles[other_langs[0]].clone()
        
        for other_lang in other_langs[1:]:
            max_other_percentiles = torch.maximum(max_other_percentiles, percentiles[other_lang])
        
        # 计算特异性得分: P_target - max(P_others)
        specificity_flat = target_percentiles - max_other_percentiles
        
        # 重新整形回原始形状
        language_scores[target_lang] = specificity_flat.view(target_tensor.shape)
    
    return language_scores


def accumulate_matrix(param_dict1, param_dict2):
    params_diff_accumulate = {}
    for (name1, param_matrix1), (name2, param_matrix2) in zip(param_dict1.items(), param_dict2.items()):
        assert name1 == name2
        params_diff_accumulate[name1] = param_matrix1 + param_matrix2
    return params_diff_accumulate

# Parse command line arguments
parser = argparse.ArgumentParser(description='Extract language specific regions using Softmax weighting or SNR method')
parser.add_argument('--k', type=float, default=0.01, help='Ratio for top/bottom parameters selection (default: 0.01)')
parser.add_argument('--method', type=str, default='softmax', choices=['softmax', 'snr', 'rank'], 
                    help='Method to calculate language specific scores: softmax (default), snr (signal-to-noise ratio), or rank (percentile ranking)')
parser.add_argument('--model_name', type=str, default='qwen2.5-7b', help='Model name for output paths')
parser.add_argument('--model_path', type=str, default='/leonardo_work/EUHPC_B24_036/yang/qwen2.5_7b/models/Qwen2.5-7B', 
                    help='Path to the original model')
parser.add_argument('--data_base_path', type=str, default='/leonardo_work/EUHPC_B24_036/yang/qwen2.5_7b', 
                    help='Base path for input data')
parser.add_argument('--output_dir', type=str, default=None, 
                    help='Output directory for results (default: {data_base_path}/path_to_save_region_selection)')
args = parser.parse_args()


# # 使用默认输出目录 (data_base_path/path_to_save_region_selection)
# python extract_language_specific_region.py --k 0.01

# # 指定自定义输出目录
# python extract_language_specific_region.py --k 0.01 \
#   --output_dir /custom/output/path

# # 完整示例：使用SNR方法并指定输出目录
# python extract_language_specific_region.py \
#   --k 0.01 \
#   --method snr \
#   --model_name language-specific-qwen2.5-7b \
#   --data_base_path /leonardo_work/EUHPC_B24_036/yang/qwen2.5_7b \
#   --output_dir /leonardo_work/EUHPC_B24_036/yang/custom_results

# 配置路径和模型信息
model_name = args.model_name
original_model_path = args.model_path
data_base_path = args.data_base_path

# 构建输入和输出路径
input_data_base = f"{data_base_path}/path_to_save_step_{{language}}/grad-mul-param_checkpoint_{{samples}}"
output_base_path = args.output_dir if args.output_dir is not None else f"{data_base_path}/path_to_save_region_selection"

# 根据方法类型构建输出路径
if args.method == 'softmax':
    method_suffix = "softmax"
elif args.method == 'snr':
    method_suffix = "snr"
else:  # rank
    method_suffix = "rank"

# 加载原版模型
print(f"Loading original model from {original_model_path}...")
original_model = AutoModelForCausalLM.from_pretrained(original_model_path)

# 语言列表
language_list = ['Chinese','English','French']

k = args.k  # ratio

print(f"Method: {args.method}")
print(f"k: {k}")
print(f"Model: {model_name}")
print(f"Data base path: {data_base_path}")
print(f"Output directory: {output_base_path}")

# 为每种语言创建存储字典
language_top_k_params_dict = {lang: {} for lang in language_list}
language_bottom_k_params_dict = {lang: {} for lang in language_list}
language_random_k_params_dict = {lang: {} for lang in language_list}

#for samples in [10000,100000]:
for samples in [10000]:
    print(f"Processing samples: {samples}")
    
    # 统一处理所有方法：逐层处理，内存高效
    print(f"Processing using {args.method} method...")
    with tqdm(total=400) as pbar:
        for (name, params) in original_model.named_parameters():
            if 'layers.' not in name:
                continue
            
            # 加载每种语言的重要性tensor
            importance_tensors = {}
            skip_parameter = False
            
            for language in language_list:
                file_dir = input_data_base.format(language=language, samples=samples)
                save_path = os.path.join(file_dir, '{}.pt'.format(name.replace('module.','')))
                
                # 检查文件是否存在
                if not os.path.exists(save_path):
                    print(f"Warning: Gradient file not found for {name} in {language}: {save_path}")
                    skip_parameter = True
                    break
                    
                importance_tensors[language] = torch.load(save_path, map_location=torch.device('cpu')).abs().cpu()
            
            # 如果任何语言的文件缺失，跳过这个参数
            if skip_parameter:
                print(f"Skipping parameter {name} due to missing gradient files")
                pbar.update(1)
                continue
            
            print(f"Processing parameter: {name}")
            print(f"Tensor shape: {importance_tensors[language_list[0]].shape}")
            print(f"Tensor dtype: {importance_tensors[language_list[0]].dtype}")
            print(f"Tensor device: {importance_tensors[language_list[0]].device}")
            
            # 根据选择的方法计算每种语言的language specific score
            if args.method == 'softmax':
                language_specific_scores = calculate_softmax_language_specific_score(importance_tensors)
            elif args.method == 'snr':
                language_specific_scores = calculate_snr_language_specific_score(importance_tensors)
            else:  # rank - 新的按层排序实现
                language_specific_scores = calculate_layer_wise_rank_scores(importance_tensors, language_list)
            
            # 为每种语言提取top/bottom/random参数
            for language in language_list:
                score_tensor = language_specific_scores[language]
                bool_sensor_max, bool_sensor_min, bool_sensor_random = get_top_bottom_tensor(score_tensor, k)
                language_top_k_params_dict[language][name] = bool_sensor_max
                language_bottom_k_params_dict[language][name] = bool_sensor_min
                language_random_k_params_dict[language][name] = bool_sensor_random
            
            # 立即释放内存
            del importance_tensors, language_specific_scores
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            
            pbar.update(1)

    # 保存结果（所有方法共用）
    for language in language_list:
        # 确保输出目录存在
        os.makedirs(output_base_path, exist_ok=True)

        # 将输出结果保存到CSV文件
        output_file = f"{output_base_path}/{language.lower()}-{method_suffix}-{model_name}-{k}-{samples}.csv"

        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            if args.method == 'softmax':
                method_name = "Softmax-MinMax"
            elif args.method == 'snr':
                method_name = "SNR"
            else:  # rank
                method_name = "Rank-Percentile"
            writer.writerow(["Parameter Name", f"{method_name} Language Specific top {k} Ratio", 
                           f"{method_name} Language Specific bottom {k} Ratio", 
                           f"{method_name} Language Specific random {k} Ratio"])
            for (name, diff_top),(name2, diff_bottom),(name3,diff_random) in zip(
                language_top_k_params_dict[language].items(),
                language_bottom_k_params_dict[language].items(),
                language_random_k_params_dict[language].items()):
                writer.writerow([name, (diff_top.sum()/diff_top.numel()).item(), 
                               (diff_bottom.sum()/diff_bottom.numel()).item(), 
                               (diff_random.sum()/diff_random.numel()).item()])

        # 保存top k参数
        top_dir = f"{output_base_path}/{language.lower()}-{method_suffix}-{model_name}/{samples}/top{k}"
        os.makedirs(top_dir, exist_ok=True)
        for key, values in language_top_k_params_dict[language].items():
            save_path = os.path.join(top_dir, f"{key}.pt")
            torch.save(values, save_path)

        # 保存bottom k参数
        bottom_dir = f"{output_base_path}/{language.lower()}-{method_suffix}-{model_name}/{samples}/bottom{k}"
        os.makedirs(bottom_dir, exist_ok=True)
        for key, values in language_bottom_k_params_dict[language].items():
            save_path = os.path.join(bottom_dir, f"{key}.pt")
            torch.save(values, save_path)

        # 保存random k参数 (只在特定samples下保存)
        if samples == 10000:  # 只为10000 samples保存random
            random_dir = f"{output_base_path}/{language.lower()}-{method_suffix}-{model_name}/{samples}/random{k}"
            os.makedirs(random_dir, exist_ok=True)
            for key, values in language_random_k_params_dict[language].items():
                save_path = os.path.join(random_dir, f"{key}.pt")
                torch.save(values, save_path)

print(f"Language specific region extraction completed using {args.method} method!")
print(f"Results saved to: {output_base_path}")
