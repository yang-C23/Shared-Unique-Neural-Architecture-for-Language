"""
Cross-Language T-map Analysis Script

This script performs cross-language analysis on t-maps and p-maps:
1. Loads p-maps from three languages (CN, EN, FR)
2. Creates binary masks using specified p-value threshold
3. Computes intersection of the three language masks
4. Applies intersection mask to t-maps from all three languages
5. Computes mean and minimum values across languages in masked regions
6. Saves results as NIfTI files
7. Generates 3D brain visualizations using the same method as plot_brain_tmap_comparison_3d.py

Usage:
python cross_language_tmap_analysis.py \
    --input_dir scratch_link/figures/t-map \
    --p_threshold 0.01 \
    --output_dir cross_language_results \
    --output_type both \
    --comparison_name llama2_7b_vs_llama2_7b_Break001
"""

import warnings
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

warnings.filterwarnings("ignore")

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
    # Convert to int8 for consistency
    binary_mask_data = mni152_mask_data.astype(np.int8)
    binary_mask_img = nib.Nifti1Image(
        binary_mask_data, mni152_mask_img.affine, mni152_mask_img.header
    )
    
    return binary_mask_img

def find_language_files(input_dir, comparison_name, languages=['CN', 'EN', 'FR']):
    """
    Find t-map and p-map files for each language
    
    Parameters:
    -----------
    input_dir : str
        Directory containing the t-map and p-map files
    comparison_name : str
        Comparison name pattern (e.g., 'llama2_7b_vs_llama2_7b_Break001')
    languages : list
        List of language codes to search for
        
    Returns:
    --------
    dict : Dictionary with language codes as keys and file paths as values
    """
    files = {'tmap': {}, 'pmap': {}}
    
    for lang in languages:
        # Search for t-map files
        tmap_pattern = f"{comparison_name}_paired_{lang}participants_*_tmap.nii.gz"
        tmap_files = glob.glob(os.path.join(input_dir, tmap_pattern))
        
        # Search for p-map files
        pmap_pattern = f"{comparison_name}_paired_{lang}participants_*_pmap.nii.gz"
        pmap_files = glob.glob(os.path.join(input_dir, pmap_pattern))
        
        if tmap_files:
            files['tmap'][lang] = tmap_files[0]
            print(f"Found t-map for {lang}: {files['tmap'][lang]}")
        else:
            raise FileNotFoundError(f"No t-map file found for {lang} with pattern: {tmap_pattern}")
            
        if pmap_files:
            files['pmap'][lang] = pmap_files[0]
            print(f"Found p-map for {lang}: {files['pmap'][lang]}")
        else:
            raise FileNotFoundError(f"No p-map file found for {lang} with pattern: {pmap_pattern}")
    
    return files

def create_intersection_mask(pmaps, p_threshold=0.05):
    """
    Create intersection mask from multiple p-maps
    
    Parameters:
    -----------
    pmaps : dict
        Dictionary with language codes as keys and p-map NIfTI images as values
    p_threshold : float
        P-value threshold for creating binary masks
        
    Returns:
    --------
    intersection_mask : np.ndarray
        Binary mask representing intersection of all language masks
    individual_masks : dict
        Individual binary masks for each language
    """
    print(f"\nCreating binary masks with p-threshold: {p_threshold}")
    
    individual_masks = {}
    mask_data_list = []
    
    for lang, pmap_img in pmaps.items():
        pmap_data = pmap_img.get_fdata()
        
        # Create binary mask: 1 where p < threshold, 0 otherwise
        binary_mask = (pmap_data < p_threshold).astype(np.int8)
        individual_masks[lang] = binary_mask
        mask_data_list.append(binary_mask)
        
        significant_voxels = np.sum(binary_mask)
        total_voxels = np.sum(binary_mask >= 0)  # Total non-NaN voxels
        percentage = (significant_voxels / total_voxels) * 100 if total_voxels > 0 else 0
        
        print(f"  {lang}: {significant_voxels} significant voxels ({percentage:.2f}%)")
    
    # Compute intersection: all languages must be significant
    intersection_mask = np.ones_like(mask_data_list[0], dtype=np.int8)
    for mask_data in mask_data_list:
        intersection_mask = intersection_mask * mask_data
    
    intersection_voxels = np.sum(intersection_mask)
    total_voxels = np.sum(intersection_mask >= 0)
    intersection_percentage = (intersection_voxels / total_voxels) * 100 if total_voxels > 0 else 0
    
    print(f"  Intersection: {intersection_voxels} voxels ({intersection_percentage:.2f}%)")
    
    return intersection_mask, individual_masks

def compute_cross_language_statistics(tmaps, intersection_mask, operation='both'):
    """
    Compute cross-language statistics (mean and/or minimum) in masked regions
    
    Parameters:
    -----------
    tmaps : dict
        Dictionary with language codes as keys and t-map data arrays as values
    intersection_mask : np.ndarray
        Binary mask for intersection regions
    operation : str
        'mean', 'min', or 'both'
        
    Returns:
    --------
    results : dict
        Dictionary containing computed statistics
    """
    print(f"\nComputing cross-language statistics: {operation}")
    
    # Stack t-map data from all languages
    tmap_stack = []
    for lang in ['CN', 'EN', 'FR']:  # Fixed order for consistency
        if lang in tmaps:
            tmap_data = tmaps[lang].get_fdata()
            # Apply intersection mask
            masked_tmap = tmap_data * intersection_mask
            tmap_stack.append(masked_tmap)
    
    if not tmap_stack:
        raise ValueError("No valid t-maps found!")
    
    # Convert to numpy array: (n_languages, x, y, z)
    tmap_array = np.stack(tmap_stack, axis=0)
    
    results = {}
    
    if operation in ['mean', 'both']:
        # Compute mean across languages where mask is active
        # Only compute mean where intersection_mask > 0
        mean_tmap = np.zeros_like(intersection_mask, dtype=np.float32)
        mask_indices = intersection_mask > 0
        
        if np.any(mask_indices):
            # Mean across languages (axis=0) only in masked regions
            language_means = np.mean(tmap_array[:, mask_indices], axis=0)
            mean_tmap[mask_indices] = language_means
        
        results['mean'] = mean_tmap
        
        mean_values_in_mask = mean_tmap[intersection_mask > 0]
        if len(mean_values_in_mask) > 0:
            print(f"  Mean t-map statistics in intersection:")
            print(f"    Min: {np.min(mean_values_in_mask):.6f}")
            print(f"    Max: {np.max(mean_values_in_mask):.6f}")
            print(f"    Mean: {np.mean(mean_values_in_mask):.6f}")
    
    if operation in ['min', 'both']:
        # Compute minimum across languages where mask is active
        min_tmap = np.zeros_like(intersection_mask, dtype=np.float32)
        mask_indices = intersection_mask > 0
        
        if np.any(mask_indices):
            # Minimum across languages (axis=0) only in masked regions
            language_mins = np.min(tmap_array[:, mask_indices], axis=0)
            min_tmap[mask_indices] = language_mins
        
        results['min'] = min_tmap
        
        min_values_in_mask = min_tmap[intersection_mask > 0]
        if len(min_values_in_mask) > 0:
            print(f"  Min t-map statistics in intersection:")
            print(f"    Min: {np.min(min_values_in_mask):.6f}")
            print(f"    Max: {np.max(min_values_in_mask):.6f}")
            print(f"    Mean: {np.mean(min_values_in_mask):.6f}")
    
    return results

def generate_3d_brain_visualizations(
    map_img, 
    output_dir, 
    file_prefix, 
    threshold=0.0,
    title_suffix=""
):
    """
    Generate 3D brain surface visualizations using nilearn
    (Same as in plot_brain_tmap_comparison_3d.py)
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
        
        # Calculate color scale - use absolute values for t-statistics
        # Higher absolute t-values should be brighter
        abs_texture_left = np.abs(texture_left)
        abs_texture_right = np.abs(texture_right)
        vmax_surf = max(np.max(abs_texture_left), np.max(abs_texture_right))
        
        # Check if we have significant values for surface visualization
        has_significant_values = np.any(abs_texture_left > threshold) or \
                               np.any(abs_texture_right > threshold)
        
        if not has_significant_values:
            max_val = max(np.max(abs_texture_left), np.max(abs_texture_right))
            surface_threshold = max(0.0, max_val * 0.1)
            print(f"  WARNING: No values above threshold {threshold}. Using {surface_threshold:.3f} for surface plots.")
        else:
            surface_threshold = threshold
        
        # --- Interactive HTML Views (Left Hemisphere) ---
        print("  Creating interactive HTML visualization for left hemisphere...")
        view_left = view_surf(
            surf_mesh=fsaverage.infl_left, 
            surf_map=abs_texture_left,  # Use absolute values for visualization
            bg_map=fsaverage.sulc_left,
            threshold=surface_threshold, 
            cmap='hot',  # Hot colormap: higher values = brighter colors
            symmetric_cmap=False,  # Don't use symmetric colormap for t-stats
            vmax=vmax_surf,
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
            surf_map=abs_texture_right,  # Use absolute values for visualization
            bg_map=fsaverage.sulc_right,
            threshold=surface_threshold, 
            cmap='hot',  # Hot colormap: higher values = brighter colors
            symmetric_cmap=False,  # Don't use symmetric colormap for t-stats
            vmax=vmax_surf,
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

def save_results_and_visualize(results, reference_img, output_dir, comparison_name, 
                             p_threshold, operation='both', visualization_threshold=0.0):
    """
    Save results as NIfTI files and generate 3D visualizations
    
    Parameters:
    -----------
    results : dict
        Dictionary containing computed statistics ('mean' and/or 'min')
    reference_img : nib.Nifti1Image
        Reference NIfTI image for header and affine information
    output_dir : str
        Output directory
    comparison_name : str
        Comparison name for file naming
    p_threshold : float
        P-value threshold used
    operation : str
        'mean', 'min', or 'both'
    visualization_threshold : float
        Threshold for 3D visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving results to: {output_dir}")
    
    saved_files = {}
    
    if 'mean' in results and operation in ['mean', 'both']:
        # Save mean t-map
        mean_img = nib.Nifti1Image(
            results['mean'], 
            reference_img.affine, 
            reference_img.header
        )
        mean_filename = f"{comparison_name}_cross_language_mean_p{p_threshold}.nii.gz"
        mean_path = os.path.join(output_dir, mean_filename)
        nib.save(mean_img, mean_path)
        print(f"  Saved mean t-map: {mean_path}")
        saved_files['mean_nii'] = mean_path
        
        # Generate 3D visualization for mean
        mean_prefix = f"{comparison_name}_cross_language_mean_p{p_threshold}"
        title_suffix = f" (Cross-Language Mean T-map, p<{p_threshold})"
        mean_viz_files = generate_3d_brain_visualizations(
            mean_img, 
            output_dir, 
            mean_prefix, 
            threshold=visualization_threshold,
            title_suffix=title_suffix
        )
        saved_files.update({f"mean_{k}": v for k, v in mean_viz_files.items()})
    
    if 'min' in results and operation in ['min', 'both']:
        # Save min t-map
        min_img = nib.Nifti1Image(
            results['min'], 
            reference_img.affine, 
            reference_img.header
        )
        min_filename = f"{comparison_name}_cross_language_min_p{p_threshold}.nii.gz"
        min_path = os.path.join(output_dir, min_filename)
        nib.save(min_img, min_path)
        print(f"  Saved min t-map: {min_path}")
        saved_files['min_nii'] = min_path
        
        # Generate 3D visualization for min
        min_prefix = f"{comparison_name}_cross_language_min_p{p_threshold}"
        title_suffix = f" (Cross-Language Min T-map, p<{p_threshold})"
        min_viz_files = generate_3d_brain_visualizations(
            min_img, 
            output_dir, 
            min_prefix, 
            threshold=visualization_threshold,
            title_suffix=title_suffix
        )
        saved_files.update({f"min_{k}": v for k, v in min_viz_files.items()})
    
    return saved_files

def main():
    """Main function for cross-language t-map analysis"""
    print("=== Starting Cross-Language T-map Analysis ===")
    
    parser = argparse.ArgumentParser(
        description="Cross-language analysis of t-maps and p-maps"
    )
    
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing t-map and p-map files (e.g., 'scratch_link/figures/t-map')")
    parser.add_argument("--comparison_name", type=str, required=True,
                       help="Comparison name pattern (e.g., 'llama2_7b_vs_llama2_7b_Break001')")
    parser.add_argument("--p_threshold", type=float, default=0.01,
                       help="P-value threshold for creating binary masks (default: 0.05)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--output_type", type=str, choices=['mean', 'min', 'both'], 
                       default='both', help="Type of output to generate (default: both)")
    parser.add_argument("--languages", nargs='+', default=['CN', 'EN', 'FR'],
                       help="List of language codes to analyze (default: CN EN FR)")
    parser.add_argument("--visualization_threshold", type=float, default=1.0,
                       help="Threshold for 3D visualizations (default: 3.0)")
    
    args = parser.parse_args()
    print(f"Parsed arguments: {args}")
    
    try:
        # Load MNI152 mask for masking operations
        binary_mask_img = load_mni152_mask()
        masker = NiftiMasker(mask_img=binary_mask_img).fit()
        
        # Find input files
        print(f"\nSearching for files in: {args.input_dir}")
        print(f"Comparison pattern: {args.comparison_name}")
        files = find_language_files(args.input_dir, args.comparison_name, args.languages)
        
        # Load p-maps and t-maps
        print(f"\nLoading p-maps and t-maps...")
        pmaps = {}
        tmaps = {}
        
        for lang in args.languages:
            if lang in files['pmap'] and lang in files['tmap']:
                pmaps[lang] = nib.load(files['pmap'][lang])
                tmaps[lang] = nib.load(files['tmap'][lang])
                print(f"  Loaded {lang}: {files['tmap'][lang]}")
            else:
                print(f"  WARNING: Missing files for language {lang}")
        
        if len(pmaps) < 2:
            raise ValueError(f"Need at least 2 languages, found only {len(pmaps)}")
        
        # Create intersection mask from p-maps
        intersection_mask, individual_masks = create_intersection_mask(pmaps, args.p_threshold)
        
        # Compute cross-language statistics
        results = compute_cross_language_statistics(tmaps, intersection_mask, args.output_type)
        
        # Save results and generate visualizations
        reference_img = list(tmaps.values())[0]  # Use first t-map as reference
        saved_files = save_results_and_visualize(
            results, 
            reference_img, 
            args.output_dir, 
            args.comparison_name,
            args.p_threshold,
            args.output_type,
            args.visualization_threshold
        )
        
        print(f"\n=== Analysis Complete ===")
        print(f"Generated files:")
        for key, path in saved_files.items():
            print(f"  {key}: {path}")
        
        return results, intersection_mask, individual_masks, saved_files
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()