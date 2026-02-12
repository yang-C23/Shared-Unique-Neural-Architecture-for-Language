"""
Language Processing Index (LPI) Analysis

This script computes Language Processing Index (LPI) for cross-linguistic brain encoding analysis.
The LPI measures language-specific brain activation patterns using the formula:

LPI_{L1} = (t_{L1} - mean(t_{other})) / (|t_{L1}| + |mean(t_{other})| + ε)

Where:
- t_{L1}: t-values for target language
- t_{other}: t-values for other languages  
- ε: small epsilon to avoid division by zero

Optimized Workflow:
1. Use brain cortical mask to select valid voxels
2. Apply rank normalization (percentile_rank) to each language's t-map within cortical mask
3. Compute LPI ONLY for cortical voxels using target language vs. other languages
4. Apply p-value threshold using target language's p-map 
5. Generate 3D brain visualizations and save results

Features:
- Loads t-map and p-map pairs from specified directory
- Uses MNI152 gray matter mask for cortical voxel selection
- Rank normalization for comparable t-value scales across languages
- Computes LPI for each target language (CN, EN, FR)
- Final thresholding using target language's p-values
- Saves LPI maps as NIfTI files with statistics
- Generates 3D cortical surface visualizations

Usage:
python lpi_analysis.py --tmap_dir path/to/tmaps --target_languages CN EN FR --output_dir lpi_results --p_threshold 0.05 --normalization percentile_rank
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environment
import matplotlib.pyplot as plt
from pathlib import Path
from nilearn.maskers import NiftiMasker
from nilearn import datasets, surface, plotting
from nilearn.plotting import view_surf
import argparse
import os
import glob
from tqdm import tqdm

def load_mni152_mask():
    """Load the MNI152 gray matter mask and create binary mask"""
    # Check different possible paths for the MNI152 mask
    possible_paths = [
        "brain_encoding/atlas/MNI152_template_gm_mask_2mm.nii.gz",  # Original path
        "../../brain_encoding/atlas/MNI152_template_gm_mask_2mm.nii.gz",  # From code directory
        "../brain_encoding/atlas/MNI152_template_gm_mask_2mm.nii.gz"  # Alternative
    ]
    
    mni152_mask_path = None
    for path in possible_paths:
        if os.path.exists(path):
            mni152_mask_path = path
            break
    
    if mni152_mask_path is None:
        raise FileNotFoundError(f"MNI152 gray matter mask not found. Tried paths: {possible_paths}")
    
    print(f"Using MNI152 gray matter mask: {mni152_mask_path}")
    
    mni152_mask_img = nib.load(mni152_mask_path)
    mni152_mask_data = mni152_mask_img.get_fdata().astype(np.float32)
    
    # The MNI152 gray matter mask is already binary (0 and 1), so we can use it directly
    binary_mask_data = mni152_mask_data.astype(np.int8)
    binary_mask_img = nib.Nifti1Image(
        binary_mask_data, mni152_mask_img.affine, mni152_mask_img.header
    )
    
    return binary_mask_img

def find_tmap_pairs(tmap_dir, languages=['CN', 'EN', 'FR']):
    """
    Find t-map and p-map pairs in the specified directory
    
    Parameters:
    -----------
    tmap_dir : str
        Directory containing t-map and p-map files
    languages : list
        List of languages to look for
        
    Returns:
    --------
    tmap_pairs : dict
        Dictionary mapping language to (tmap_path, pmap_path) tuples
    """
    tmap_dir = Path(tmap_dir)
    tmap_pairs = {}
    
    print(f"Searching for t-map/p-map pairs in: {tmap_dir}")
    
    for language in languages:
        # Look for files containing the language code
        tmap_pattern = f"*{language}participants*tmap.nii.gz"
        pmap_pattern = f"*{language}participants*pmap.nii.gz"
        
        tmap_files = list(tmap_dir.glob(tmap_pattern))
        pmap_files = list(tmap_dir.glob(pmap_pattern))
        
        print(f"  {language}: Found {len(tmap_files)} t-maps, {len(pmap_files)} p-maps")
        
        if len(tmap_files) == 1 and len(pmap_files) == 1:
            tmap_pairs[language] = (tmap_files[0], pmap_files[0])
            print(f"    T-map: {tmap_files[0].name}")
            print(f"    P-map: {pmap_files[0].name}")
        elif len(tmap_files) > 1:
            # If multiple files, use the first one or prompt user
            print(f"    WARNING: Multiple t-maps found for {language}, using: {tmap_files[0].name}")
            if len(pmap_files) > 0:
                tmap_pairs[language] = (tmap_files[0], pmap_files[0])
        else:
            print(f"    WARNING: No matching t-map/p-map pair found for {language}")
    
    return tmap_pairs

def load_and_threshold_tmap(tmap_path, pmap_path, p_threshold=0.05, check_pre_thresholded=True):
    """
    Load t-map and apply p-value thresholding if needed
    
    Parameters:
    -----------
    tmap_path : str or Path
        Path to t-map file
    pmap_path : str or Path  
        Path to p-map file
    p_threshold : float
        P-value threshold for filtering
    check_pre_thresholded : bool
        Whether to check if t-map is already thresholded
        
    Returns:
    --------
    tmap_data : np.ndarray
        Thresholded t-map data
    """
    print(f"Loading t-map: {os.path.basename(tmap_path)}")
    
    # Load t-map and p-map
    tmap_img = nib.load(tmap_path)
    pmap_img = nib.load(pmap_path)
    
    tmap_data = tmap_img.get_fdata().astype(np.float32)
    pmap_data = pmap_img.get_fdata().astype(np.float32)
    
    print(f"  Original t-map range: {tmap_data.min():.3f} to {tmap_data.max():.3f}")
    print(f"  Non-zero voxels: {np.count_nonzero(tmap_data)} / {tmap_data.size}")
    
    # Check if t-map is already thresholded
    if check_pre_thresholded:
        zero_voxels = np.count_nonzero(tmap_data == 0)
        total_voxels = tmap_data.size
        zero_ratio = zero_voxels / total_voxels
        
        print(f"  Zero voxels ratio: {zero_ratio:.1%}")
        
        # If more than 70% of voxels are zero, assume it's already thresholded
        if zero_ratio > 0.7:
            print("  T-map appears to be already thresholded (many zero voxels)")
            return tmap_data, tmap_img.affine, tmap_img.header
    
    # Apply p-value thresholding
    print(f"  Applying p-value threshold: p < {p_threshold}")
    significant_mask = pmap_data < p_threshold
    tmap_thresholded = tmap_data.copy()
    tmap_thresholded[~significant_mask] = 0
    
    significant_voxels = np.count_nonzero(significant_mask)
    print(f"  Significant voxels: {significant_voxels} / {tmap_data.size} ({significant_voxels/tmap_data.size:.1%})")
    print(f"  Thresholded t-map range: {tmap_thresholded.min():.3f} to {tmap_thresholded.max():.3f}")
    
    return tmap_thresholded, tmap_img.affine, tmap_img.header

def normalize_tmap(tmap_data, method='minmax', mask=None):
    """
    Normalize t-map data using different methods
    
    Parameters:
    -----------
    tmap_data : np.ndarray
        T-map data to normalize
    method : str
        Normalization method: 'minmax', 'zscore', 'robust', 'percentile', 'percentile_rank', or 'none'
    mask : np.ndarray, optional
        Binary mask to specify which voxels to consider for normalization
        
    Returns:
    --------
    normalized_data : np.ndarray
        Normalized t-map data
    """
    print(f"  Applying {method} normalization...")
    
    if method == 'none':
        print("    No normalization applied")
        return tmap_data
    
    # Create working copy
    data = tmap_data.copy()
    
    # Determine which voxels to use for normalization statistics
    if mask is not None:
        # Use provided mask
        valid_voxels = mask > 0
    else:
        # Use non-zero voxels (assuming zero voxels are non-significant)
        valid_voxels = data != 0
    
    if not np.any(valid_voxels):
        print("    WARNING: No valid voxels found for normalization")
        return data
    
    valid_data = data[valid_voxels]
    
    print(f"    Using {np.sum(valid_voxels)} voxels for normalization statistics")
    print(f"    Original valid data range: {valid_data.min():.3f} to {valid_data.max():.3f}")
    
    if method == 'minmax':
        # Min-max normalization to [0, 1]
        data_min = valid_data.min()
        data_max = valid_data.max()
        
        if data_max == data_min:
            print("    WARNING: All valid values are the same, cannot normalize")
            return data
        
        # Only normalize non-zero voxels, keep zeros as zeros
        data[valid_voxels] = (valid_data - data_min) / (data_max - data_min)
        print(f"    Min-max normalized to [0, 1] using range [{data_min:.3f}, {data_max:.3f}]")
        
    elif method == 'zscore':
        # Z-score normalization (mean=0, std=1)
        data_mean = valid_data.mean()
        data_std = valid_data.std()
        
        if data_std == 0:
            print("    WARNING: Standard deviation is 0, cannot normalize")
            return data
        
        data[valid_voxels] = (valid_data - data_mean) / data_std
        print(f"    Z-score normalized using mean={data_mean:.3f}, std={data_std:.3f}")
        
    elif method == 'robust':
        # Robust normalization using median and MAD
        data_median = np.median(valid_data)
        mad = np.median(np.abs(valid_data - data_median))
        
        if mad == 0:
            print("    WARNING: MAD is 0, cannot normalize")
            return data
        
        data[valid_voxels] = (valid_data - data_median) / mad
        print(f"    Robust normalized using median={data_median:.3f}, MAD={mad:.3f}")
        
    elif method == 'percentile':
        # Percentile normalization (5th to 95th percentile mapped to [0, 1])
        p5 = np.percentile(valid_data, 5)
        p95 = np.percentile(valid_data, 95)
        
        if p95 == p5:
            print("    WARNING: 5th and 95th percentiles are the same, cannot normalize")
            return data
        
        # Clip to percentile range and normalize
        data[valid_voxels] = np.clip((valid_data - p5) / (p95 - p5), 0, 1)
        print(f"    Percentile normalized using 5th-95th percentile range [{p5:.3f}, {p95:.3f}]")
        
    elif method == 'percentile_rank':
        # Percentile rank normalization - sort by t-value and replace with percentile rank
        print("    Computing percentile rank normalization...")
        
        # Sort valid data and compute percentile ranks
        sorted_indices = np.argsort(valid_data)
        rank_data = np.zeros_like(valid_data)
        
        # Assign percentile ranks (0 to 1)
        n_valid = len(valid_data)
        for i, original_idx in enumerate(sorted_indices):
            rank_data[original_idx] = i / (n_valid - 1) if n_valid > 1 else 0.5
        
        # Replace original values with percentile ranks
        data[valid_voxels] = rank_data
        
        print(f"    Percentile rank normalized: {n_valid} voxels ranked from 0 to 1")
        print(f"    Rank distribution: min={rank_data.min():.3f}, max={rank_data.max():.3f}, mean={rank_data.mean():.3f}")
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Report final statistics
    final_valid_data = data[valid_voxels]
    print(f"    Final valid data range: {final_valid_data.min():.3f} to {final_valid_data.max():.3f}")
    print(f"    Final valid data mean: {final_valid_data.mean():.3f}, std: {final_valid_data.std():.3f}")
    
    return data

def compute_lpi_cortical(target_data, other_data_list, cortical_mask, epsilon=1e-8):
    """
    Compute Language Processing Index (LPI) only for cortical voxels
    This is more efficient and logically consistent than computing for all voxels
    
    Parameters:
    -----------
    target_data : np.ndarray
        Target language t-values (normalized and rectified)
    other_data_list : list of np.ndarray
        List of other languages' t-values (normalized and rectified)
    cortical_mask : np.ndarray
        Boolean mask indicating cortical voxels
    epsilon : float
        Small value to avoid division by zero
        
    Returns:
    --------
    lpi_data : np.ndarray
        LPI values (full brain volume, but only cortical voxels computed)
    """
    print("Computing LPI for cortical voxels only...")
    
    # Initialize LPI data as zeros for full brain volume
    lpi_data = np.zeros_like(target_data)
    
    # Extract cortical voxels only
    cortical_voxels = cortical_mask.astype(bool)
    n_cortical = np.sum(cortical_voxels)
    print(f"Computing LPI for {n_cortical} cortical voxels...")
    
    # 2. 矫正目标语言的t值 (将负值设为0) - 只处理皮层体素
    target_cortical = target_data[cortical_voxels]
    target_cortical_rectified = np.maximum(target_cortical, 0)
    print("目标图谱皮层体素已矫正 (负值 -> 0)。")
    
    # 3. 加载并处理其他语言的t值图谱 - 只处理皮层体素
    other_cortical_rectified_list = []
    for i, other_data in enumerate(other_data_list):
        print(f"矫正其他语言皮层体素 {i+1}/{len(other_data_list)}")
        other_cortical = other_data[cortical_voxels]
        other_cortical_rectified = np.maximum(other_cortical, 0)
        other_cortical_rectified_list.append(other_cortical_rectified)
    
    # 4. 计算其他语言的平均t值 - 只在皮层体素
    if not other_cortical_rectified_list:
        raise ValueError("其他语言列表不能为空。")
    
    mean_other_cortical = np.mean(np.stack(other_cortical_rectified_list, axis=-1), axis=-1)
    print("已计算其他语言在皮层体素的平均t值图谱。")
    
    # 5. 计算LPI指数 - 只在皮层体素
    print("正在计算皮层体素的LPI...")
    # 分子: t_target - mean(t_others)
    numerator = target_cortical_rectified - mean_other_cortical
    
    # 分母: t_target + mean(t_others) + epsilon
    denominator = target_cortical_rectified + mean_other_cortical + epsilon
    
    # 计算LPI，处理分母可能为0的情况
    lpi_cortical = np.zeros_like(target_cortical_rectified)
    np.divide(numerator, denominator, out=lpi_cortical, where=denominator!=0)
    
    # 将皮层LPI结果放回到全脑体积中
    lpi_data[cortical_voxels] = lpi_cortical
    
    print(f"LPI 计算完成 (仅皮层体素)。")
    print(f"皮层LPI范围: {lpi_cortical.min():.3f} 到 {lpi_cortical.max():.3f}")
    print(f"皮层LPI均值: {lpi_cortical.mean():.3f}, 标准差: {lpi_cortical.std():.3f}")
    
    return lpi_data

def compute_lpi(target_data, other_data_list, epsilon=1e-8):
    """
    Compute Language Processing Index (LPI)
    
    Parameters:
    -----------
    target_data : np.ndarray
        Target language t-values (normalized and rectified)
    other_data_list : list of np.ndarray
        List of other languages' t-values (normalized and rectified)
    epsilon : float
        Small value to avoid division by zero
        
    Returns:
    --------
    lpi_data : np.ndarray
        LPI values
    """
    print("Computing LPI...")
    
    # 2. 矫正目标语言的t值 (将负值设为0)
    # np.maximum是逐元素比较，返回每个位置上的最大值
    target_data = np.maximum(target_data, 0)
    print("目标图谱已矫正 (负值 -> 0)。")
    
    # 3. 加载并处理其他语言的t值图谱
    other_data_rectified_list = []
    for i, other_data in enumerate(other_data_list):
        print(f"矫正其他语言图谱 {i+1}/{len(other_data_list)}")
        # 同样进行矫正
        other_data_rectified = np.maximum(other_data, 0)
        other_data_rectified_list.append(other_data_rectified)
    
    # 4. 计算其他语言的平均t值
    # np.stack将列表中的数组堆叠成一个新维度的数组 (x, y, z, n_others)
    # np.mean沿着新创建的维度(axis=-1)计算平均值
    if not other_data_rectified_list:
        raise ValueError("其他语言列表不能为空。")
    
    mean_other_data = np.mean(np.stack(other_data_rectified_list, axis=-1), axis=-1)
    print("已计算其他语言的平均t值图谱。")
    
    # 5. 计算LPI指数
    print("正在计算LPI...")
    # 分子: t_target - mean(t_others)
    numerator = target_data - mean_other_data
    
    # 分母: t_target + mean(t_others) + epsilon  
    # 因为已经矫正，所以不需要取绝对值
    denominator = target_data + mean_other_data + epsilon
    
    # 计算LPI，同时处理分母可能为0的情况
    # np.divide可以指定在分母为0时输出什么值(out=...)，以及在哪里进行计算(where=...)
    lpi_data = np.zeros_like(target_data)  # 创建一个和输入形状相同、填满0的数组
    np.divide(numerator, denominator, out=lpi_data, where=denominator!=0)
    
    print(f"LPI 计算完成。范围: {lpi_data.min():.3f} 到 {lpi_data.max():.3f}")
    print(f"LPI 均值: {lpi_data.mean():.3f}, 标准差: {lpi_data.std():.3f}")
    
    return lpi_data

def generate_3d_brain_visualizations(
    map_img, 
    output_dir, 
    file_prefix, 
    threshold=0.0,
    title_suffix="",
    cmap='hot',  # Use heat colormap for 0 to positive values  
    symmetric_cmap=False  # Use 0 to vmax range
):
    """
    Generate 3D brain surface visualizations using nilearn
    
    Parameters:
    -----------
    map_img : nib.Nifti1Image
        Brain map image (LPI map)
    output_dir : str
        Output directory
    file_prefix : str
        Prefix for output files
    threshold : float
        Visualization threshold
    title_suffix : str
        Additional text for titles
    cmap : str
        Colormap to use
    symmetric_cmap : bool
        Whether to use symmetric colormap
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGenerating 3D brain visualizations for {file_prefix} (threshold={threshold})...")
    
    output_files = {}
    
    try:
        # --- Surface-based Visualizations ---
        print("  Fetching fsaverage surface data...")
        fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
        
        print("  Projecting volume map to cortical surface...")
        texture_left = surface.vol_to_surf(map_img, fsaverage.pial_left, radius=6.0, kind='ball')
        texture_right = surface.vol_to_surf(map_img, fsaverage.pial_right, radius=6.0, kind='ball')
        
        # Clean non-finite values
        texture_left = np.nan_to_num(texture_left)
        texture_right = np.nan_to_num(texture_right)
        
        # Calculate color scale for LPI values
        if symmetric_cmap:
            # For LPI, use symmetric scale around 0
            vmax_surf = max(np.abs(np.min([texture_left.min(), texture_right.min()])),
                           np.max([texture_left.max(), texture_right.max()]))
            vmin_surf = -vmax_surf
        else:
            # For non-symmetric maps (0 to vmax) - only show positive values
            vmin_surf = 0
            vmax_surf = max(texture_left.max(), texture_right.max())
            # Remove negative values from texture data to avoid showing them in plots
            texture_left = np.maximum(texture_left, 0)
            texture_right = np.maximum(texture_right, 0)
            # Recalculate vmax after removing negative values
            vmax_surf = max(texture_left.max(), texture_right.max())
        
        # Check if we have significant values for surface visualization
        has_significant_values = np.any(np.abs(texture_left) > threshold) or \
                               np.any(np.abs(texture_right) > threshold)
        
        if not has_significant_values:
            max_val = max(np.abs(texture_left).max(), np.abs(texture_right).max())
            surface_threshold = max(0.0, max_val * 0.1)
            print(f"  WARNING: No values above threshold {threshold}. Using {surface_threshold:.3f} for surface plots.")
        else:
            surface_threshold = threshold
            
        print(f"  Using surface threshold: {surface_threshold:.3f}")
        print(f"  Color range: [{vmin_surf:.3f}, {vmax_surf:.3f}]")
        if not symmetric_cmap:
            print(f"  Note: Negative values removed from display (only showing 0 to positive values)")
        
        # --- Interactive HTML Views (Left Hemisphere) ---
        print("  Creating interactive HTML visualization for left hemisphere...")
        view_left = view_surf(
            surf_mesh=fsaverage.infl_left, 
            surf_map=texture_left,
            bg_map=fsaverage.sulc_left,
            threshold=surface_threshold, 
            cmap=cmap,
            symmetric_cmap=symmetric_cmap,
            vmax=vmax_surf,
            vmin=vmin_surf,  # Always set vmin: 0 for non-symmetric, -vmax for symmetric
            title=f"Left Hemisphere: {file_prefix}{title_suffix}"
        )
        left_html_path = os.path.join(output_dir, f'{file_prefix}_left_3d.html')
        view_left.save_as_html(left_html_path)
        output_files['left_html'] = left_html_path
        print(f"  - Saved Left Hemisphere HTML: {left_html_path}")
        
        # --- Interactive HTML Views (Right Hemisphere) ---
        print("  Creating interactive HTML visualization for right hemisphere...")
        view_right = view_surf(
            surf_mesh=fsaverage.infl_right, 
            surf_map=texture_right,
            bg_map=fsaverage.sulc_right,
            threshold=surface_threshold, 
            cmap=cmap,
            symmetric_cmap=symmetric_cmap,
            vmax=vmax_surf,
            vmin=vmin_surf,  # Always set vmin: 0 for non-symmetric, -vmax for symmetric
            title=f"Right Hemisphere: {file_prefix}{title_suffix}"
        )
        right_html_path = os.path.join(output_dir, f'{file_prefix}_right_3d.html')
        view_right.save_as_html(right_html_path)
        output_files['right_html'] = right_html_path
        print(f"  - Saved Right Hemisphere HTML: {right_html_path}")
        
        print(f"  3D visualizations saved to: {output_dir}")
        return output_files
        
    except Exception as e:
        print(f"  ERROR: 3D visualization generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

def perform_lpi_analysis(tmap_pairs, target_languages, output_dir, p_threshold=0.05, epsilon=1e-8, normalization='percentile_rank', viz_threshold=0.05):
    """
    Perform LPI analysis for specified target languages following the optimized workflow:
    1. Use brain cortical mask to select valid voxels
    2. Apply rank normalization to each language's t-map within cortical mask
    3. Compute LPI ONLY for cortical voxels using target language (more efficient & logical)
    4. Apply p-value threshold using target language's p-map
    5. Generate 3D brain visualizations and save results
    
    Parameters:
    -----------
    tmap_pairs : dict
        Dictionary mapping language to (tmap_path, pmap_path) tuples
    target_languages : list
        List of languages to use as targets
    output_dir : str
        Output directory for results
    p_threshold : float
        P-value threshold for final LPI filtering
    epsilon : float
        Small value to avoid division by zero in LPI calculation
    normalization : str
        Normalization method for t-maps (should be 'percentile_rank' for ranking)
    viz_threshold : float
        Visualization threshold to hide small values in 3D plots
    """
    # Step 1: Load MNI152 cortical mask for selecting valid voxels
    print("\n=== Step 1: Loading Brain Cortical Mask ===")
    binary_mask_img = load_mni152_mask()
    mask_data = binary_mask_img.get_fdata().astype(bool)
    print(f"Cortical mask loaded. Valid voxels: {np.sum(mask_data)} / {mask_data.size}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 2: Load t-maps and p-maps, apply cortical mask and rank normalization
    print("\n=== Step 2: Loading T-maps and Applying Cortical Mask + Rank Normalization ===")
    loaded_tmaps = {}
    loaded_pmaps = {}
    affines = {}
    headers = {}
    
    for language, (tmap_path, pmap_path) in tmap_pairs.items():
        print(f"\n--- Processing {language} ---")
        
        # Load raw t-map and p-map (without thresholding)
        print(f"Loading t-map: {os.path.basename(tmap_path)}")
        tmap_img = nib.load(tmap_path)
        pmap_img = nib.load(pmap_path)
        
        tmap_data = tmap_img.get_fdata().astype(np.float32)
        pmap_data = pmap_img.get_fdata().astype(np.float32)
        
        print(f"  Original t-map range: {tmap_data.min():.3f} to {tmap_data.max():.3f}")
        print(f"  Original p-map range: {pmap_data.min():.6f} to {pmap_data.max():.6f}")
        
        # Apply cortical mask to select valid voxels
        masked_tmap = tmap_data.copy()
        masked_tmap[~mask_data] = 0  # Set non-cortical voxels to 0
        
        valid_voxels_in_cortex = mask_data & (tmap_data != 0)
        print(f"  Valid voxels in cortical mask: {np.sum(valid_voxels_in_cortex)}")
        
        # Apply rank normalization to cortical voxels only
        normalized_tmap = normalize_tmap(masked_tmap, method=normalization, mask=mask_data)
        
        loaded_tmaps[language] = normalized_tmap
        loaded_pmaps[language] = pmap_data
        affines[language] = tmap_img.affine
        headers[language] = tmap_img.header
        
        # Show statistics for normalized data in cortical region
        cortical_normalized = normalized_tmap[mask_data]
        non_zero_cortical = cortical_normalized[cortical_normalized != 0]
        if len(non_zero_cortical) > 0:
            print(f"  Normalized {language} in cortex: mean={non_zero_cortical.mean():.3f}, "
                  f"range=[{non_zero_cortical.min():.3f}, {non_zero_cortical.max():.3f}]")
    
    # Step 3: Perform LPI analysis for each target language
    print(f"\n=== Step 3: Computing LPI for target languages: {target_languages} ===")
    
    for target_lang in target_languages:
        if target_lang not in loaded_tmaps:
            print(f"WARNING: Target language {target_lang} not found in t-maps. Skipping.")
            continue
            
        print(f"\n--- LPI Analysis for {target_lang} as Target Language ---")
        
        # Get target language data and p-map
        target_data = loaded_tmaps[target_lang]
        target_pmap = loaded_pmaps[target_lang]
        
        # Get other languages data
        other_languages = [lang for lang in loaded_tmaps.keys() if lang != target_lang]
        other_data_list = [loaded_tmaps[lang] for lang in other_languages]
        
        print(f"Target language: {target_lang}")
        print(f"Other languages: {other_languages}")
        
        # Compute LPI only for cortical voxels (more efficient and logical)
        print("Computing LPI for cortical voxels only...")
        lpi_data = compute_lpi_cortical(target_data, other_data_list, mask_data, epsilon)
        
        # Step 4: Apply p-value threshold using target language's p-map
        print(f"\n--- Step 4: Applying p-value threshold (p < {p_threshold}) ---")
        significant_mask = target_pmap < p_threshold
        significant_voxels = np.sum(significant_mask)
        print(f"Significant voxels in target {target_lang}: {significant_voxels} / {target_pmap.size}")
        
        # Create final LPI map: apply p-value threshold to cortical LPI results
        # Note: lpi_data already only contains values in cortical regions (others are 0)
        lpi_final = lpi_data.copy()
        lpi_final[~significant_mask] = 0
        
        # Count final significant cortical voxels
        final_significant_cortical = np.sum(significant_mask & mask_data & (lpi_data != 0))
        cortical_lpi_computed = np.sum(mask_data & (lpi_data != 0))
        
        print(f"LPI computed for cortical voxels: {cortical_lpi_computed}")
        print(f"Final significant cortical voxels: {final_significant_cortical}")
        print(f"Final LPI range: {lpi_final.min():.3f} to {lpi_final.max():.3f}")
        
        # Step 5: Save results and generate 3D visualization
        print(f"\n--- Step 5: Saving Results and Generating 3D Visualization ---")
        
        # Convert to brain volume
        lpi_img = nib.Nifti1Image(lpi_final, affines[target_lang], headers[target_lang])
        
        # Save LPI map as NIfTI
        lpi_filename = f"LPI_{target_lang}_target_p{p_threshold}_{normalization}.nii.gz"
        lpi_path = os.path.join(output_dir, lpi_filename)
        nib.save(lpi_img, lpi_path)
        print(f"Final LPI map saved: {lpi_path}")
        
        # Generate 3D visualizations using cortical mask
        print(f"Generating 3D visualization for LPI_{target_lang}...")
        file_prefix = f"LPI_{target_lang}_target_p{p_threshold}_{normalization}"
        title_suffix = f" (LPI {target_lang} target, p<{p_threshold}, {normalization})"
        
        # Use the provided visualization threshold to hide small values
        print(f"Using visualization threshold: {viz_threshold} (increase to hide smaller values)")
        
        output_files = generate_3d_brain_visualizations(
            lpi_img, 
            output_dir, 
            file_prefix, 
            threshold=viz_threshold,
            title_suffix=title_suffix,
            cmap='hot',  # Heat colormap for 0 to positive values
            symmetric_cmap=False  # Use 0 to vmax range
        )
        
        print(f"3D visualizations saved:")
        for key, path in output_files.items():
            print(f"  {key}: {path}")
        
        # Save some statistics
        stats_file = os.path.join(output_dir, f"LPI_{target_lang}_stats.txt")
        with open(stats_file, 'w') as f:
            f.write(f"LPI Analysis Statistics for {target_lang}\n")
            f.write(f"======================================\n\n")
            f.write(f"Target language: {target_lang}\n")
            f.write(f"Other languages: {other_languages}\n")
            f.write(f"P-value threshold: {p_threshold}\n")
            f.write(f"Normalization method: {normalization}\n")
            f.write(f"Epsilon: {epsilon}\n\n")
            f.write(f"Cortical mask voxels: {np.sum(mask_data)}\n")
            f.write(f"LPI computed for cortical voxels: {cortical_lpi_computed}\n")
            f.write(f"Significant voxels (p < {p_threshold}): {significant_voxels}\n")
            f.write(f"Final significant cortical voxels: {final_significant_cortical}\n\n")
            f.write(f"LPI statistics (final thresholded map):\n")
            valid_lpi = lpi_final[lpi_final != 0]
            if len(valid_lpi) > 0:
                f.write(f"  Range: [{valid_lpi.min():.6f}, {valid_lpi.max():.6f}]\n")
                f.write(f"  Mean: {valid_lpi.mean():.6f}\n")
                f.write(f"  Std: {valid_lpi.std():.6f}\n")
            else:
                f.write(f"  No valid LPI values found\n")
        
        print(f"Statistics saved: {stats_file}")

def main():
    """Main function for LPI analysis"""
    print("=== Language Processing Index (LPI) Analysis ===")
    
    parser = argparse.ArgumentParser(description="Compute Language Processing Index (LPI) from t-maps")
    
    parser.add_argument("--tmap_dir", type=str, required=True,
                       help="Directory containing t-map and p-map files")
    parser.add_argument("--target_languages", nargs='+', 
                       choices=['CN', 'EN', 'FR'], default=['CN', 'EN', 'FR'],
                       help="Languages to use as targets for LPI computation")
    parser.add_argument("--output_dir", type=str, default="lpi_results",
                       help="Output directory for LPI results")
    parser.add_argument("--p_threshold", type=float, default=0.01,
                       help="P-value threshold for filtering t-maps (if not pre-thresholded)")
    parser.add_argument("--epsilon", type=float, default=1e-8,
                       help="Small value to avoid division by zero in LPI calculation")
    parser.add_argument("--normalization", type=str, choices=['minmax', 'zscore', 'robust', 'percentile', 'percentile_rank', 'none'],
                       default='percentile_rank', help="Normalization method for t-maps (recommended: percentile_rank for ranking)")
    parser.add_argument("--viz_threshold", type=float, default=0.05,
                       help="Visualization threshold to hide small values in 3D plots (higher = less noise, typical: 0.05-0.15)")
    parser.add_argument("--check_pre_thresholded", action='store_true', default=True,
                       help="Check if t-maps are already thresholded")
    
    args = parser.parse_args()
    
    print(f"Arguments:")
    print(f"  T-map directory: {args.tmap_dir}")
    print(f"  Target languages: {args.target_languages}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  P-value threshold: {args.p_threshold}")
    print(f"  Epsilon: {args.epsilon}")
    print(f"  Normalization: {args.normalization}")
    print(f"  Visualization threshold: {args.viz_threshold}")
    
    try:
        # Find t-map and p-map pairs
        all_languages = ['CN', 'EN', 'FR']
        tmap_pairs = find_tmap_pairs(args.tmap_dir, all_languages)
        
        if len(tmap_pairs) < 2:
            raise ValueError(f"Need at least 2 languages for LPI analysis. Found: {list(tmap_pairs.keys())}")
        
        print(f"\nFound t-map pairs for languages: {list(tmap_pairs.keys())}")
        
        # Perform LPI analysis
        perform_lpi_analysis(
            tmap_pairs, 
            args.target_languages, 
            args.output_dir,
            args.p_threshold,
            args.epsilon,
            args.normalization,
            args.viz_threshold
        )
        
        print(f"\n=== LPI Analysis Complete ===")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()