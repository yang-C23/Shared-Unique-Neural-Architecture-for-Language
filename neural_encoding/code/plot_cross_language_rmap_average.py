"""
Cross-Language R-map Averaging Script

This script performs cross-language averaging on r-maps and p-maps:
1. Loads p-maps from three languages (CN, EN, FR) for a specific model
2. Creates binary significance masks using p-value threshold (default: p<0.01)
3. Provides two methods for averaging:
   - Union: Creates union mask (any language significant), then averages r-maps
   - Conjunction: Creates intersection mask (all languages significant), then averages r-maps
4. Saves averaged r-maps as NIfTI files
5. Generates 3D brain visualizations

Usage:
python plot_cross_language_rmap_average.py \
    --model_name llama2-7b \
    --rmap_dir scratch_link/figures/llama2-7b_r-map_rmap \
    --p_threshold 0.01 \
    --output_dir scratch_link/figures/llama2-7b_cross_language \
    --mask_type both
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

def find_language_files(rmap_dir, model_name, languages=['CN', 'EN', 'FR']):
    """
    Find r-map and p-map files for each language
    
    Parameters:
    -----------
    rmap_dir : str
        Directory containing the r-map and p-map files
    model_name : str
        Model name (e.g., 'llama2-7b')
    languages : list
        List of language codes to search for
        
    Returns:
    --------
    dict : Dictionary with language codes as keys and file paths as values
    """
    files = {'rmap': {}, 'pmap': {}}
    
    for lang in languages:
        # Search for r-map files
        # Pattern: llama2-7b_ENparticipants_*_rmap.nii.gz
        rmap_pattern = f"{model_name}_{lang}participants_*_rmap.nii.gz"
        rmap_files = glob.glob(os.path.join(rmap_dir, rmap_pattern))
        
        # Search for p-map files
        pmap_pattern = f"{model_name}_{lang}participants_*_pmap.nii.gz"
        pmap_files = glob.glob(os.path.join(rmap_dir, pmap_pattern))
        
        if rmap_files:
            files['rmap'][lang] = rmap_files[0]
            print(f"Found r-map for {lang}: {os.path.basename(files['rmap'][lang])}")
        else:
            raise FileNotFoundError(f"No r-map file found for {lang} with pattern: {rmap_pattern}")
            
        if pmap_files:
            files['pmap'][lang] = pmap_files[0]
            print(f"Found p-map for {lang}: {os.path.basename(files['pmap'][lang])}")
        else:
            raise FileNotFoundError(f"No p-map file found for {lang} with pattern: {pmap_pattern}")
    
    return files

def create_significance_masks(pmaps, p_threshold=0.01):
    """
    Create significance masks from p-maps
    
    Parameters:
    -----------
    pmaps : dict
        Dictionary with language codes as keys and p-map NIfTI images as values
    p_threshold : float
        P-value threshold for creating binary masks
        
    Returns:
    --------
    union_mask : np.ndarray
        Binary mask representing union of all language masks (any language significant)
    conjunction_mask : np.ndarray
        Binary mask representing intersection of all language masks (all languages significant)
    individual_masks : dict
        Individual binary masks for each language
    """
    print(f"\nCreating significance masks with p-threshold: {p_threshold}")
    
    individual_masks = {}
    mask_data_list = []
    
    for lang, pmap_img in pmaps.items():
        pmap_data = pmap_img.get_fdata()
        
        # Create binary mask: 1 where p < threshold, 0 otherwise
        binary_mask = (pmap_data < p_threshold).astype(np.int8)
        individual_masks[lang] = binary_mask
        mask_data_list.append(binary_mask)
        
        significant_voxels = np.sum(binary_mask)
        total_voxels = np.prod(binary_mask.shape)
        percentage = (significant_voxels / total_voxels) * 100 if total_voxels > 0 else 0
        
        print(f"  {lang}: {significant_voxels} significant voxels ({percentage:.2f}%)")
    
    # Compute union mask: at least one language is significant
    union_mask = np.zeros_like(mask_data_list[0], dtype=np.int8)
    for mask_data in mask_data_list:
        union_mask = union_mask | mask_data
    
    union_voxels = np.sum(union_mask)
    total_voxels = np.prod(union_mask.shape)
    union_percentage = (union_voxels / total_voxels) * 100 if total_voxels > 0 else 0
    
    print(f"  Union (any language): {union_voxels} voxels ({union_percentage:.2f}%)")
    
    # Compute conjunction mask: all languages must be significant
    conjunction_mask = np.ones_like(mask_data_list[0], dtype=np.int8)
    for mask_data in mask_data_list:
        conjunction_mask = conjunction_mask & mask_data
    
    conjunction_voxels = np.sum(conjunction_mask)
    conjunction_percentage = (conjunction_voxels / total_voxels) * 100 if total_voxels > 0 else 0
    
    print(f"  Conjunction (all languages): {conjunction_voxels} voxels ({conjunction_percentage:.2f}%)")
    
    return union_mask, conjunction_mask, individual_masks

def compute_averaged_rmap(rmaps, mask, mask_type='union'):
    """
    Compute averaged r-map in masked regions
    
    Parameters:
    -----------
    rmaps : dict
        Dictionary with language codes as keys and r-map NIfTI images as values
    mask : np.ndarray
        Binary mask for averaging
    mask_type : str
        Type of mask ('union' or 'conjunction')
        
    Returns:
    --------
    averaged_rmap : np.ndarray
        Averaged r-map values in masked regions
    """
    print(f"\nComputing averaged r-map using {mask_type} mask...")
    
    # Stack r-map data from all languages
    rmap_stack = []
    for lang in ['CN', 'EN', 'FR']:  # Fixed order for consistency
        if lang in rmaps:
            rmap_data = rmaps[lang].get_fdata()
            rmap_stack.append(rmap_data)
    
    if not rmap_stack:
        raise ValueError("No valid r-maps found!")
    
    # Convert to numpy array: (n_languages, x, y, z)
    rmap_array = np.stack(rmap_stack, axis=0)
    
    # Compute average r-map
    averaged_rmap = np.zeros_like(mask, dtype=np.float32)
    mask_indices = mask > 0
    
    if np.any(mask_indices):
        # Average across languages (axis=0) only in masked regions
        language_means = np.mean(rmap_array[:, mask_indices], axis=0)
        averaged_rmap[mask_indices] = language_means
        
        # Print statistics
        print(f"  R-map statistics in {mask_type} mask:")
        print(f"    Min: {np.min(language_means):.6f}")
        print(f"    Max: {np.max(language_means):.6f}")
        print(f"    Mean: {np.mean(language_means):.6f}")
        print(f"    Std: {np.std(language_means):.6f}")
    else:
        print(f"  WARNING: No voxels in {mask_type} mask!")
    
    return averaged_rmap

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
        
        # Calculate color scale for r-maps (positive values)
        vmax_surf = max(np.max(texture_left), np.max(texture_right))
        
        # Check if we have significant values for surface visualization
        has_significant_values = np.any(texture_left > threshold) or \
                               np.any(texture_right > threshold)
        
        if not has_significant_values:
            max_val = max(np.max(texture_left), np.max(texture_right))
            surface_threshold = max(0.0, max_val * 0.1)
            print(f"  WARNING: No values above threshold {threshold}. Using {surface_threshold:.3f} for surface plots.")
        else:
            surface_threshold = threshold
        
        # --- Interactive HTML Views (Left Hemisphere) ---
        print("  Creating interactive HTML visualization for left hemisphere...")
        view_left = view_surf(
            surf_mesh=fsaverage.infl_left, 
            surf_map=texture_left,
            bg_map=fsaverage.sulc_left,
            threshold=surface_threshold, 
            cmap='hot',  # Hot colormap: higher values = brighter colors
            symmetric_cmap=False,
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
            surf_map=texture_right,
            bg_map=fsaverage.sulc_right,
            threshold=surface_threshold, 
            cmap='hot',  # Hot colormap: higher values = brighter colors
            symmetric_cmap=False,
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

def save_results_and_visualize(results, reference_img, output_dir, model_name, 
                             p_threshold, mask_type='both', visualization_threshold=0.05):
    """
    Save results as NIfTI files and generate 3D visualizations
    
    Parameters:
    -----------
    results : dict
        Dictionary containing averaged r-maps ('union' and/or 'conjunction')
    reference_img : nib.Nifti1Image
        Reference NIfTI image for header and affine information
    output_dir : str
        Output directory
    model_name : str
        Model name for file naming
    p_threshold : float
        P-value threshold used
    mask_type : str
        'union', 'conjunction', or 'both'
    visualization_threshold : float
        Threshold for 3D visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving results to: {output_dir}")
    
    saved_files = {}
    
    if 'union' in results and mask_type in ['union', 'both']:
        # Save union r-map
        union_img = nib.Nifti1Image(
            results['union'], 
            reference_img.affine, 
            reference_img.header
        )
        union_filename = f"{model_name}_cross_language_union_p{p_threshold}_rmap.nii.gz"
        union_path = os.path.join(output_dir, union_filename)
        nib.save(union_img, union_path)
        print(f"  Saved union r-map: {union_path}")
        saved_files['union_nii'] = union_path
        
        # Generate 3D visualization for union
        union_prefix = f"{model_name}_cross_language_union_p{p_threshold}"
        title_suffix = f" (Cross-Language Union R-map, p<{p_threshold})"
        union_viz_files = generate_3d_brain_visualizations(
            union_img, 
            output_dir, 
            union_prefix, 
            threshold=visualization_threshold,
            title_suffix=title_suffix
        )
        saved_files.update({f"union_{k}": v for k, v in union_viz_files.items()})
    
    if 'conjunction' in results and mask_type in ['conjunction', 'both']:
        # Save conjunction r-map
        conjunction_img = nib.Nifti1Image(
            results['conjunction'], 
            reference_img.affine, 
            reference_img.header
        )
        conjunction_filename = f"{model_name}_cross_language_conjunction_p{p_threshold}_rmap.nii.gz"
        conjunction_path = os.path.join(output_dir, conjunction_filename)
        nib.save(conjunction_img, conjunction_path)
        print(f"  Saved conjunction r-map: {conjunction_path}")
        saved_files['conjunction_nii'] = conjunction_path
        
        # Generate 3D visualization for conjunction
        conjunction_prefix = f"{model_name}_cross_language_conjunction_p{p_threshold}"
        title_suffix = f" (Cross-Language Conjunction R-map, p<{p_threshold})"
        conjunction_viz_files = generate_3d_brain_visualizations(
            conjunction_img, 
            output_dir, 
            conjunction_prefix, 
            threshold=visualization_threshold,
            title_suffix=title_suffix
        )
        saved_files.update({f"conjunction_{k}": v for k, v in conjunction_viz_files.items()})
    
    return saved_files

def main():
    """Main function for cross-language r-map averaging"""
    print("=== Starting Cross-Language R-map Averaging ===")
    
    parser = argparse.ArgumentParser(
        description="Cross-language averaging of r-maps using significance masks from p-maps"
    )
    
    parser.add_argument("--model_name", type=str, required=True,
                       help="Model name (e.g., 'llama2-7b')")
    parser.add_argument("--rmap_dir", type=str, required=True,
                       help="Directory containing r-map and p-map files (e.g., 'scratch_link/figures/llama2-7b_r-map_rmap')")
    parser.add_argument("--p_threshold", type=float, default=0.01,
                       help="P-value threshold for creating significance masks (default: 0.01)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--mask_type", type=str, choices=['union', 'conjunction', 'both'], 
                       default='both', help="Type of mask to use (default: both)")
    parser.add_argument("--languages", nargs='+', default=['CN', 'EN', 'FR'],
                       help="List of language codes to analyze (default: CN EN FR)")
    parser.add_argument("--visualization_threshold", type=float, default=0.05,
                       help="Threshold for 3D visualizations (default: 0.05)")
    
    args = parser.parse_args()
    print(f"Parsed arguments: {args}")
    
    try:
        # Load MNI152 mask for masking operations
        binary_mask_img = load_mni152_mask()
        masker = NiftiMasker(mask_img=binary_mask_img).fit()
        
        # Find input files
        print(f"\nSearching for files in: {args.rmap_dir}")
        print(f"Model: {args.model_name}")
        files = find_language_files(args.rmap_dir, args.model_name, args.languages)
        
        # Load p-maps and r-maps
        print(f"\nLoading p-maps and r-maps...")
        pmaps = {}
        rmaps = {}
        
        for lang in args.languages:
            if lang in files['pmap'] and lang in files['rmap']:
                pmaps[lang] = nib.load(files['pmap'][lang])
                rmaps[lang] = nib.load(files['rmap'][lang])
                print(f"  Loaded {lang}: {os.path.basename(files['rmap'][lang])}")
            else:
                print(f"  WARNING: Missing files for language {lang}")
        
        if len(pmaps) < 2:
            raise ValueError(f"Need at least 2 languages, found only {len(pmaps)}")
        
        # Create significance masks from p-maps
        union_mask, conjunction_mask, individual_masks = create_significance_masks(pmaps, args.p_threshold)
        
        # Compute averaged r-maps
        results = {}
        
        if args.mask_type in ['union', 'both']:
            results['union'] = compute_averaged_rmap(rmaps, union_mask, 'union')
        
        if args.mask_type in ['conjunction', 'both']:
            results['conjunction'] = compute_averaged_rmap(rmaps, conjunction_mask, 'conjunction')
        
        # Save results and generate visualizations
        reference_img = list(rmaps.values())[0]  # Use first r-map as reference
        saved_files = save_results_and_visualize(
            results, 
            reference_img, 
            args.output_dir, 
            args.model_name,
            args.p_threshold,
            args.mask_type,
            args.visualization_threshold
        )
        
        print(f"\n=== Analysis Complete ===")
        print(f"Generated files:")
        for key, path in saved_files.items():
            print(f"  {key}: {path}")
        
        return results, union_mask, conjunction_mask, individual_masks, saved_files
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()

