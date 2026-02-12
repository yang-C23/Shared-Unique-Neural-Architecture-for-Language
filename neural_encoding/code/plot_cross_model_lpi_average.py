"""
Cross-Model LPI Average Analysis

This script computes averaged Language Processing Index (LPI) across multiple models
for robust and stable language-specific brain activation patterns.

Averaging Strategy:
1. All 6 models (llama2-7b, llama2-13b, mistral-8b, mistral-12b, qwen2.5-7b, qwen2.5-14b)
2. 3 smaller models (llama2-7b, mistral-8b, qwen2.5-7b)
3. 3 larger models (llama2-13b, mistral-12b, qwen2.5-14b)

For each:
- 2 ablation methods (rank, snr)
- 3 normalization methods (percentile_rank, zscore, minmax)
- 3 target languages (CN, EN, FR)

Total: 2×3×3 = 18 groups of averaged LPI results

Usage:
python plot_cross_model_lpi_average.py --base_dir scratch_link/figures \
    --ablation_method rank --normalization percentile_rank \
    --model_group all --p_threshold 0.01 \
    --output_dir scratch_link/figures/cross_model_lpi_average
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from nilearn import datasets, surface, plotting
from nilearn.plotting import view_surf
import argparse
import os
from tqdm import tqdm

# Model groups definition
MODEL_GROUPS = {
    'all': ['llama2-7b', 'llama2-13b', 'mistral-8b', 'mistral-12b', 'qwen2.5-7b', 'qwen2.5-14b'],
    'small': ['llama2-7b', 'mistral-8b', 'qwen2.5-7b'],
    'large': ['llama2-13b', 'mistral-12b', 'qwen2.5-14b']
}

def load_lpi_maps(base_dir, models, ablation_method, normalization, target_lang, p_threshold=0.01):
    """
    Load LPI maps from multiple models
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing all LPI results
    models : list
        List of model names
    ablation_method : str
        Ablation method (rank or snr)
    normalization : str
        Normalization method (percentile_rank, zscore, minmax)
    target_lang : str
        Target language (CN, EN, FR)
    p_threshold : float
        P-value threshold used in original analysis
        
    Returns:
    --------
    lpi_maps : list
        List of loaded LPI maps
    affine : np.ndarray
        Affine matrix (same for all maps)
    header : nib.Nifti1Header
        NIfTI header (from first map)
    """
    lpi_maps = []
    affine = None
    header = None
    
    print(f"\nLoading LPI maps for {target_lang} (ablation={ablation_method}, norm={normalization})...")
    
    for model in models:
        # Construct directory path
        lpi_dir = os.path.join(
            base_dir,
            f"{model}_language-specific-Break_{ablation_method}_top001_lpi_results_{normalization}_p001"
        )
        
        # Construct LPI filename
        lpi_filename = f"LPI_{target_lang}_target_p{p_threshold}_{normalization}.nii.gz"
        lpi_path = os.path.join(lpi_dir, lpi_filename)
        
        if not os.path.exists(lpi_path):
            print(f"  WARNING: LPI map not found: {lpi_path}")
            continue
        
        print(f"  Loading {model}: {os.path.basename(lpi_path)}")
        lpi_img = nib.load(lpi_path)
        lpi_data = lpi_img.get_fdata().astype(np.float32)
        
        # Store affine and header from first map
        if affine is None:
            affine = lpi_img.affine
            header = lpi_img.header
        
        lpi_maps.append(lpi_data)
        
        # Print statistics
        non_zero = lpi_data[lpi_data != 0]
        if len(non_zero) > 0:
            print(f"    Non-zero voxels: {len(non_zero)}, "
                  f"Range: [{non_zero.min():.3f}, {non_zero.max():.3f}], "
                  f"Mean: {non_zero.mean():.3f}")
        else:
            print(f"    WARNING: No non-zero voxels found")
    
    if len(lpi_maps) == 0:
        raise ValueError(f"No LPI maps found for {target_lang}")
    
    print(f"  Loaded {len(lpi_maps)} / {len(models)} LPI maps")
    
    return lpi_maps, affine, header

def average_lpi_maps(lpi_maps, method='mean'):
    """
    Average multiple LPI maps
    
    Parameters:
    -----------
    lpi_maps : list
        List of LPI maps (numpy arrays)
    method : str
        Averaging method ('mean' or 'median')
        
    Returns:
    --------
    averaged_map : np.ndarray
        Averaged LPI map
    """
    print(f"\nAveraging {len(lpi_maps)} LPI maps using {method}...")
    
    # Stack all maps
    stacked_maps = np.stack(lpi_maps, axis=-1)
    
    # Compute average
    if method == 'mean':
        # For voxels where all maps are 0, result should be 0
        # For other voxels, compute mean
        mask = np.any(stacked_maps != 0, axis=-1)
        averaged_map = np.zeros(stacked_maps.shape[:-1])
        
        # Compute mean only for non-zero voxels across models
        for i in range(stacked_maps.shape[0]):
            for j in range(stacked_maps.shape[1]):
                for k in range(stacked_maps.shape[2]):
                    voxel_values = stacked_maps[i, j, k, :]
                    non_zero_values = voxel_values[voxel_values != 0]
                    if len(non_zero_values) > 0:
                        averaged_map[i, j, k] = non_zero_values.mean()
    
    elif method == 'median':
        mask = np.any(stacked_maps != 0, axis=-1)
        averaged_map = np.zeros(stacked_maps.shape[:-1])
        
        for i in range(stacked_maps.shape[0]):
            for j in range(stacked_maps.shape[1]):
                for k in range(stacked_maps.shape[2]):
                    voxel_values = stacked_maps[i, j, k, :]
                    non_zero_values = voxel_values[voxel_values != 0]
                    if len(non_zero_values) > 0:
                        averaged_map[i, j, k] = np.median(non_zero_values)
    
    else:
        raise ValueError(f"Unknown averaging method: {method}")
    
    # Print statistics
    non_zero = averaged_map[averaged_map != 0]
    if len(non_zero) > 0:
        print(f"  Averaged map statistics:")
        print(f"    Non-zero voxels: {len(non_zero)}")
        print(f"    Range: [{non_zero.min():.3f}, {non_zero.max():.3f}]")
        print(f"    Mean: {non_zero.mean():.3f}")
        print(f"    Std: {non_zero.std():.3f}")
    
    return averaged_map

def generate_3d_brain_visualizations(
    map_img, 
    output_dir, 
    file_prefix, 
    threshold=0.0,
    title_suffix="",
    cmap='hot',
    symmetric_cmap=False
):
    """
    Generate 3D brain surface visualizations using nilearn
    (Same as in plot_lpi_analysis.py)
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
            vmax_surf = max(np.abs(np.min([texture_left.min(), texture_right.min()])),
                           np.max([texture_left.max(), texture_right.max()]))
            vmin_surf = -vmax_surf
        else:
            # For non-symmetric maps (0 to vmax) - only show positive values
            vmin_surf = 0
            vmax_surf = max(texture_left.max(), texture_right.max())
            # Remove negative values from texture data
            texture_left = np.maximum(texture_left, 0)
            texture_right = np.maximum(texture_right, 0)
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
        
        # --- Interactive HTML Views (Left Hemisphere) ---
        print("  Creating interactive HTML visualization for left hemisphere...")
        view_left = view_surf(
            surf_mesh=fsaverage.infl_left, 
            surf_map=texture_left,
            bg_map=fsaverage.sulc_left,
            threshold=surface_threshold, 
            cmap=cmap,
            symmetric_cmap=symmetric_cmap,
            vmax=0.18,
            # vmax=vmax_surf,
            vmin=vmin_surf,
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
            # vmax=vmax_surf,
            vmax=0.18,
            vmin=vmin_surf,
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

def perform_cross_model_averaging(
    base_dir,
    models,
    ablation_method,
    normalization,
    target_languages,
    output_dir,
    p_threshold=0.01,
    viz_threshold=0.05,
    averaging_method='mean'
):
    """
    Perform cross-model LPI averaging for all target languages
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing all LPI results
    models : list
        List of model names to average
    ablation_method : str
        Ablation method (rank or snr)
    normalization : str
        Normalization method
    target_languages : list
        List of target languages
    output_dir : str
        Output directory for averaged results
    p_threshold : float
        P-value threshold
    viz_threshold : float
        Visualization threshold
    averaging_method : str
        Method for averaging ('mean' or 'median')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print(f"Cross-Model LPI Averaging")
    print("=" * 80)
    print(f"Models: {models}")
    print(f"Ablation method: {ablation_method}")
    print(f"Normalization: {normalization}")
    print(f"Target languages: {target_languages}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Process each target language
    for target_lang in target_languages:
        print(f"\n{'=' * 80}")
        print(f"Processing Target Language: {target_lang}")
        print(f"{'=' * 80}")
        
        try:
            # Load LPI maps from all models
            lpi_maps, affine, header = load_lpi_maps(
                base_dir, models, ablation_method, normalization, 
                target_lang, p_threshold
            )
            
            # Average LPI maps
            averaged_lpi = average_lpi_maps(lpi_maps, method=averaging_method)
            
            # Create NIfTI image
            averaged_lpi_img = nib.Nifti1Image(averaged_lpi, affine, header)
            
            # Save averaged LPI map
            lpi_filename = f"LPI_{target_lang}_averaged_{len(models)}models_{ablation_method}_{normalization}.nii.gz"
            lpi_path = os.path.join(output_dir, lpi_filename)
            nib.save(averaged_lpi_img, lpi_path)
            print(f"\nSaved averaged LPI map: {lpi_path}")
            
            # Generate 3D visualizations
            file_prefix = f"LPI_{target_lang}_averaged_{len(models)}models_{ablation_method}_{normalization}"
            title_suffix = f" (Averaged {len(models)} models, {target_lang} target)"
            
            output_files = generate_3d_brain_visualizations(
                averaged_lpi_img,
                output_dir,
                file_prefix,
                threshold=viz_threshold,
                title_suffix=title_suffix,
                cmap='hot',
                symmetric_cmap=False
            )
            
            # Save statistics
            stats_file = os.path.join(output_dir, f"LPI_{target_lang}_averaged_stats.txt")
            with open(stats_file, 'w') as f:
                f.write(f"Cross-Model Averaged LPI Statistics for {target_lang}\n")
                f.write(f"=" * 60 + "\n\n")
                f.write(f"Target language: {target_lang}\n")
                f.write(f"Models averaged: {models}\n")
                f.write(f"Number of models: {len(models)}\n")
                f.write(f"Ablation method: {ablation_method}\n")
                f.write(f"Normalization method: {normalization}\n")
                f.write(f"P-value threshold: {p_threshold}\n")
                f.write(f"Averaging method: {averaging_method}\n\n")
                
                valid_lpi = averaged_lpi[averaged_lpi != 0]
                if len(valid_lpi) > 0:
                    f.write(f"Averaged LPI statistics:\n")
                    f.write(f"  Non-zero voxels: {len(valid_lpi)}\n")
                    f.write(f"  Range: [{valid_lpi.min():.6f}, {valid_lpi.max():.6f}]\n")
                    f.write(f"  Mean: {valid_lpi.mean():.6f}\n")
                    f.write(f"  Std: {valid_lpi.std():.6f}\n")
                    f.write(f"  Median: {np.median(valid_lpi):.6f}\n")
                    f.write(f"  25th percentile: {np.percentile(valid_lpi, 25):.6f}\n")
                    f.write(f"  75th percentile: {np.percentile(valid_lpi, 75):.6f}\n")
                else:
                    f.write(f"  No valid LPI values found\n")
            
            print(f"Statistics saved: {stats_file}")
            
        except Exception as e:
            print(f"\nERROR processing {target_lang}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'=' * 80}")
    print(f"Cross-Model LPI Averaging Complete")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 80}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Compute cross-model averaged LPI from multiple models"
    )
    
    parser.add_argument("--base_dir", type=str, required=True,
                       help="Base directory containing all LPI results")
    parser.add_argument("--ablation_method", type=str, required=True,
                       choices=['rank', 'snr'],
                       help="Ablation method (rank or snr)")
    parser.add_argument("--normalization", type=str, required=True,
                       choices=['percentile_rank', 'zscore', 'minmax'],
                       help="Normalization method")
    parser.add_argument("--model_group", type=str, required=True,
                       choices=['all', 'small', 'large'],
                       help="Model group to average (all/small/large)")
    parser.add_argument("--target_languages", nargs='+',
                       choices=['CN', 'EN', 'FR'], default=['CN', 'EN', 'FR'],
                       help="Target languages")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for averaged results")
    parser.add_argument("--p_threshold", type=float, default=0.01,
                       help="P-value threshold used in original analysis")
    parser.add_argument("--viz_threshold", type=float, default=0.05,
                       help="Visualization threshold for 3D plots")
    parser.add_argument("--averaging_method", type=str, default='mean',
                       choices=['mean', 'median'],
                       help="Averaging method (mean or median)")
    
    args = parser.parse_args()
    
    # Get models for the specified group
    models = MODEL_GROUPS[args.model_group]
    
    print(f"\nArguments:")
    print(f"  Base directory: {args.base_dir}")
    print(f"  Ablation method: {args.ablation_method}")
    print(f"  Normalization: {args.normalization}")
    print(f"  Model group: {args.model_group}")
    print(f"  Models: {models}")
    print(f"  Target languages: {args.target_languages}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  P-value threshold: {args.p_threshold}")
    print(f"  Visualization threshold: {args.viz_threshold}")
    print(f"  Averaging method: {args.averaging_method}")
    
    try:
        perform_cross_model_averaging(
            args.base_dir,
            models,
            args.ablation_method,
            args.normalization,
            args.target_languages,
            args.output_dir,
            args.p_threshold,
            args.viz_threshold,
            args.averaging_method
        )
        
        print("\n=== Cross-Model LPI Averaging Complete ===")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

