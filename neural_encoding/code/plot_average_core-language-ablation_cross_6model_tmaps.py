"""
Cross-Model T-map Averaging with Intersection Mask

This script averages t-maps across 6 models for each language separately.
It uses p < 0.01 threshold to create binary masks from each model's p-map,
computes the intersection of all masks, and averages t-values only within 
the intersection region (setting values outside to 0).

Usage:
python average_cross_model_tmaps.py \
    --base_dir /leonardo_work/EUHPC_B24_036/yang/scratch_link/figures \
    --output_dir cross_model_averaged_tmaps \
    --p_threshold 0.01 \
    --viz_threshold 0.7 \
    --use_percentile_threshold
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
from datetime import datetime

def find_tmap_files_by_language(base_dir, language):
    """
    Find t-map and p-map files for all 6 models for a specific language
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing model subdirectories
    language : str
        Language code (CN, EN, or FR)
        
    Returns:
    --------
    model_files : dict
        Dictionary mapping model names to (tmap_file, pmap_file) tuples
    """
    base_path = Path(base_dir)
    
    # Model directory mappings
    model_dirs = {
        'llama2-7b': 'llama2-7b_break001t-map',
        'llama2-13b': 'llama2-13b_core_region_t-map',
        'mistral-8b': 'mistral-8b_core_region_t-map',
        'mistral-12b': 'mistral-12b_core_region_t-map',
        'qwen2.5-7b': 'qwen2.5-7b_core_region_t-map',
        'qwen2.5-14b': 'qwen2.5-14b_core_region_t-map'
    }
    
    model_files = {}
    
    print(f"\n{'='*80}")
    print(f"Searching for {language} participant files...")
    print(f"{'='*80}")
    
    for model_name, dir_name in model_dirs.items():
        model_dir = base_path / dir_name
        
        if not model_dir.exists():
            print(f"  WARNING: Directory not found: {model_dir}")
            continue
        
        # Find files for this language
        tmap_files = list(model_dir.glob(f"*_{language}participants_*_tmap.nii.gz"))
        pmap_files = list(model_dir.glob(f"*_{language}participants_*_pmap.nii.gz"))
        
        if not tmap_files:
            print(f"  WARNING: No t-map found for {model_name} with {language} participants")
            continue
        if not pmap_files:
            print(f"  WARNING: No p-map found for {model_name} with {language} participants")
            continue
        
        model_files[model_name] = (tmap_files[0], pmap_files[0])
        print(f"  ✓ {model_name}:")
        print(f"      T-map: {tmap_files[0].name}")
        print(f"      P-map: {pmap_files[0].name}")
    
    print(f"\nTotal models found for {language}: {len(model_files)}/6")
    
    return model_files

def average_tmaps_intersection_mask(model_files, p_threshold, language):
    """
    Create intersection mask from all p-maps at p < threshold, then average t-maps
    
    Parameters:
    -----------
    model_files : dict
        Dictionary mapping model names to (tmap_file, pmap_file) tuples
    p_threshold : float
        P-value threshold (e.g., 0.01)
    language : str
        Language code for logging
        
    Returns:
    --------
    averaged_tmap : np.ndarray
        Averaged t-map (zeros outside intersection)
    intersection_mask : np.ndarray
        Binary intersection mask
    reference_affine : np.ndarray
        Reference affine matrix
    reference_header : nibabel header
        Reference header
    statistics : dict
        Detailed statistics
    """
    print(f"\n{'='*80}")
    print(f"Averaging T-maps for {language} Participants")
    print(f"Method: Intersection Mask (p < {p_threshold})")
    print(f"{'='*80}")
    
    tmap_list = []
    mask_list = []
    model_names = []
    reference_affine = None
    reference_header = None
    model_stats = []
    
    # Step 1: Load all t-maps and create binary masks from p-maps
    print("\nStep 1: Loading t-maps and creating individual masks...")
    for model_name, (tmap_file, pmap_file) in sorted(model_files.items()):
        print(f"\n  Processing {model_name}...")
        
        # Load t-map and p-map
        tmap_img = nib.load(tmap_file)
        pmap_img = nib.load(pmap_file)
        
        tmap_data = tmap_img.get_fdata().astype(np.float32)
        pmap_data = pmap_img.get_fdata().astype(np.float32)
        
        # Store reference from first model
        if reference_affine is None:
            reference_affine = tmap_img.affine.copy()
            reference_header = tmap_img.header.copy()
        
        # Create binary mask: 1 where p < threshold
        binary_mask = (pmap_data < p_threshold).astype(np.int8)
        
        # Statistics
        significant_voxels = np.sum(binary_mask)
        tmap_in_mask = tmap_data[binary_mask > 0]
        
        stats = {
            'model': model_name,
            'significant_voxels': significant_voxels,
            'tmap_range': (tmap_data.min(), tmap_data.max()),
            'tmap_in_mask_mean': tmap_in_mask.mean() if len(tmap_in_mask) > 0 else 0,
            'tmap_in_mask_std': tmap_in_mask.std() if len(tmap_in_mask) > 0 else 0
        }
        model_stats.append(stats)
        
        print(f"    T-map range: [{stats['tmap_range'][0]:.3f}, {stats['tmap_range'][1]:.3f}]")
        print(f"    Mask voxels (p < {p_threshold}): {significant_voxels:,}")
        print(f"    T-values in mask: {stats['tmap_in_mask_mean']:.4f} ± {stats['tmap_in_mask_std']:.4f}")
        
        tmap_list.append(tmap_data)
        mask_list.append(binary_mask)
        model_names.append(model_name)
    
    # Step 2: Compute intersection mask
    print(f"\nStep 2: Computing intersection of {len(mask_list)} masks...")
    intersection_mask = np.ones_like(mask_list[0], dtype=np.int8)
    for mask in mask_list:
        intersection_mask = intersection_mask * mask
    
    intersection_voxels = np.sum(intersection_mask)
    print(f"  Intersection voxels: {intersection_voxels:,}")
    
    # Print individual mask sizes for comparison
    print(f"\n  Individual mask sizes:")
    for i, model_name in enumerate(model_names):
        individual_count = np.sum(mask_list[i])
        overlap_pct = (intersection_voxels / individual_count * 100) if individual_count > 0 else 0
        print(f"    {model_name}: {individual_count:,} voxels (intersection = {overlap_pct:.1f}%)")
    
    # Step 3: Average t-maps only within intersection mask
    print(f"\nStep 3: Averaging t-maps within intersection mask...")
    
    tmap_stack = np.stack(tmap_list, axis=-1)
    averaged_tmap = np.zeros_like(tmap_stack[:, :, :, 0], dtype=np.float32)
    
    # Apply intersection mask and average
    mask_indices = intersection_mask > 0
    if np.any(mask_indices):
        averaged_tmap[mask_indices] = tmap_stack[mask_indices].mean(axis=-1)
        print(f"  ✓ Averaged {intersection_voxels:,} voxels")
    else:
        print(f"  WARNING: No voxels in intersection!")
    
    # Step 4: Compute statistics
    non_zero_avg = averaged_tmap[averaged_tmap != 0]
    statistics = {
        'language': language,
        'method': 'intersection_mask',
        'p_threshold': p_threshold,
        'n_models': len(tmap_list),
        'model_names': model_names,
        'model_stats': model_stats,
        'intersection_voxels': int(intersection_voxels),
        'averaged_stats': {
            'non_zero_voxels': len(non_zero_avg),
            'tmap_range': (float(averaged_tmap.min()), float(averaged_tmap.max())),
            'tmap_mean': float(non_zero_avg.mean()) if len(non_zero_avg) > 0 else 0,
            'tmap_std': float(non_zero_avg.std()) if len(non_zero_avg) > 0 else 0
        }
    }
    
    avg_stats = statistics['averaged_stats']
    print(f"\nAveraged T-map Statistics:")
    print(f"  Non-zero voxels: {avg_stats['non_zero_voxels']:,}")
    print(f"  T-map range: [{avg_stats['tmap_range'][0]:.3f}, {avg_stats['tmap_range'][1]:.3f}]")
    print(f"  T-map mean: {avg_stats['tmap_mean']:.4f} ± {avg_stats['tmap_std']:.4f}")
    
    return averaged_tmap, intersection_mask, reference_affine, reference_header, statistics

def generate_3d_brain_visualizations(
    map_img, 
    output_dir, 
    file_prefix, 
    threshold=0.0,
    title_suffix="",
    cmap='hot',
    symmetric_cmap=False,
    use_percentile_threshold=False,
    fixed_vmax=None,
    fixed_vmin_threshold=None
):
    """
    Generate 3D brain surface visualizations using nilearn
    
    Parameters:
    -----------
    use_percentile_threshold : bool
        If True, threshold is interpreted as a percentile (0-1) of the maximum value.
        E.g., threshold=0.7 means show top 30% of values (70% of max to max)
    fixed_vmax : float, optional
        If provided, use this fixed maximum value for color scale instead of auto-calculating
    fixed_vmin_threshold : float, optional
        If provided, use this fixed minimum threshold for display instead of calculating from threshold parameter
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGenerating 3D brain visualizations...")
    print(f"  File prefix: {file_prefix}")
    print(f"  Threshold mode: {'percentile' if use_percentile_threshold else 'absolute'}")
    
    output_files = {}
    
    try:
        # Fetch fsaverage surface
        print("  Fetching fsaverage surface data...")
        fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
        
        # Project to surface
        print("  Projecting volume to cortical surface...")
        texture_left = surface.vol_to_surf(map_img, fsaverage.pial_left, radius=6.0, kind='ball')
        texture_right = surface.vol_to_surf(map_img, fsaverage.pial_right, radius=6.0, kind='ball')
        
        # Clean non-finite values
        texture_left = np.nan_to_num(texture_left)
        texture_right = np.nan_to_num(texture_right)
        
        # Use absolute values for visualization
        abs_texture_left = np.abs(texture_left)
        abs_texture_right = np.abs(texture_right)
        
        # Determine vmax - use fixed value if provided, otherwise auto-calculate
        if fixed_vmax is not None:
            vmax_surf = fixed_vmax
            print(f"  Using fixed vmax: {vmax_surf:.3f}")
        else:
            vmax_surf = max(abs_texture_left.max(), abs_texture_right.max())
            print(f"  Auto-calculated vmax: {vmax_surf:.3f}")
        
        # Determine surface threshold
        if fixed_vmin_threshold is not None:
            # Use fixed minimum threshold if provided
            surface_threshold = fixed_vmin_threshold
            print(f"  Using fixed minimum threshold: {surface_threshold:.3f}")
        elif use_percentile_threshold:
            # Percentile mode: threshold is a fraction of max value
            surface_threshold = vmax_surf * threshold
            print(f"  Using percentile threshold: {threshold:.1%} of max = {surface_threshold:.3f}")
        else:
            # Absolute mode: use threshold as-is, with fallback
            has_significant = np.any(abs_texture_left > threshold) or np.any(abs_texture_right > threshold)
            
            if not has_significant:
                max_val = max(abs_texture_left.max(), abs_texture_right.max())
                surface_threshold = max(0.0, max_val * 0.1)
                print(f"  WARNING: No values above threshold {threshold}. Using {surface_threshold:.3f}")
            else:
                surface_threshold = threshold
        
        print(f"  Surface threshold: {surface_threshold:.3f}")
        print(f"  Color range: [0, {vmax_surf:.3f}]")
        
        # Left hemisphere
        print("  Creating left hemisphere visualization...")
        view_left = view_surf(
            surf_mesh=fsaverage.infl_left, 
            surf_map=abs_texture_left,
            bg_map=fsaverage.sulc_left,
            threshold=surface_threshold, 
            cmap=cmap,
            symmetric_cmap=symmetric_cmap,
            vmax=vmax_surf,
            title=f"Left Hemisphere: {file_prefix}{title_suffix}"
        )
        left_html_path = os.path.join(output_dir, f'{file_prefix}_left_3d.html')
        view_left.save_as_html(left_html_path)
        output_files['left_html'] = left_html_path
        print(f"  ✓ Saved: {left_html_path}")
        
        # Right hemisphere
        print("  Creating right hemisphere visualization...")
        view_right = view_surf(
            surf_mesh=fsaverage.infl_right, 
            surf_map=abs_texture_right,
            bg_map=fsaverage.sulc_right,
            threshold=surface_threshold, 
            cmap=cmap,
            symmetric_cmap=symmetric_cmap,
            vmax=vmax_surf,
            title=f"Right Hemisphere: {file_prefix}{title_suffix}"
        )
        right_html_path = os.path.join(output_dir, f'{file_prefix}_right_3d.html')
        view_right.save_as_html(right_html_path)
        output_files['right_html'] = right_html_path
        print(f"  ✓ Saved: {right_html_path}")
        
        return output_files
        
    except Exception as e:
        print(f"  ERROR: 3D visualization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

def save_results(averaged_tmap, intersection_mask, reference_affine, reference_header, 
                statistics, output_dir, viz_threshold, use_percentile_threshold,
                fixed_vmax=None, fixed_vmin_threshold=None):
    """Save averaged t-map results with statistics and visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    language = statistics['language']
    p_threshold = statistics['p_threshold']
    n_models = statistics['n_models']
    
    print(f"\n{'='*80}")
    print(f"SAVING RESULTS FOR {language} PARTICIPANTS")
    print(f"{'='*80}")
    
    # Create filename prefix
    file_prefix = f"cross_model_avg_{language}participants_p{p_threshold}_intersection"
    
    # Save averaged t-map as NIfTI
    tmap_img = nib.Nifti1Image(averaged_tmap, reference_affine, reference_header)
    tmap_filename = f"{file_prefix}_tmap.nii.gz"
    tmap_path = os.path.join(output_dir, tmap_filename)
    nib.save(tmap_img, tmap_path)
    print(f"✓ Averaged t-map saved: {tmap_path}")
    
    # Save intersection mask
    mask_img = nib.Nifti1Image(intersection_mask.astype(np.int8), reference_affine, reference_header)
    mask_filename = f"{file_prefix}_mask.nii.gz"
    mask_path = os.path.join(output_dir, mask_filename)
    nib.save(mask_img, mask_path)
    print(f"✓ Intersection mask saved: {mask_path}")
    
    # Generate 3D visualizations
    title_suffix = f" ({language} participants, {n_models} models, intersection mask)"
    
    output_files = generate_3d_brain_visualizations(
        tmap_img, 
        output_dir, 
        file_prefix, 
        threshold=viz_threshold,
        title_suffix=title_suffix,
        cmap='hot',
        symmetric_cmap=False,
        use_percentile_threshold=use_percentile_threshold,
        fixed_vmax=fixed_vmax,
        fixed_vmin_threshold=fixed_vmin_threshold
    )
    
    print(f"\n3D visualizations:")
    for key, path in output_files.items():
        print(f"  {key}: {path}")
    
    # Save detailed statistics
    stats_filename = f"{file_prefix}_stats.txt"
    stats_path = os.path.join(output_dir, stats_filename)
    
    with open(stats_path, 'w') as f:
        f.write(f"Cross-Model T-map Averaging Analysis\n")
        f.write(f"="*80 + "\n\n")
        f.write(f"Analysis Configuration:\n")
        f.write(f"  Language: {language}\n")
        f.write(f"  Number of models: {n_models}\n")
        f.write(f"  Models: {', '.join(statistics['model_names'])}\n")
        f.write(f"  Method: {statistics['method']}\n")
        f.write(f"  P-value threshold: {p_threshold}\n")
        f.write(f"  Visualization threshold: {viz_threshold}")
        if use_percentile_threshold:
            f.write(f" (percentile mode)\n")
        else:
            f.write(f" (absolute mode)\n")
        f.write(f"\n")
        
        # Individual model statistics
        f.write(f"Individual Model Statistics:\n")
        f.write(f"-"*80 + "\n")
        for stats in statistics['model_stats']:
            f.write(f"\n{stats['model']}:\n")
            f.write(f"  T-map range: [{stats['tmap_range'][0]:.3f}, {stats['tmap_range'][1]:.3f}]\n")
            f.write(f"  Significant voxels (p < {p_threshold}): {stats['significant_voxels']:,}\n")
            f.write(f"  T-values in mask: {stats['tmap_in_mask_mean']:.4f} ± {stats['tmap_in_mask_std']:.4f}\n")
        
        # Intersection mask statistics
        f.write(f"\nIntersection Mask Statistics:\n")
        f.write(f"-"*80 + "\n")
        f.write(f"  Intersection voxels: {statistics['intersection_voxels']:,}\n")
        
        # Individual mask overlap
        f.write(f"\n  Individual mask overlap with intersection:\n")
        for stats in statistics['model_stats']:
            overlap_pct = (statistics['intersection_voxels'] / stats['significant_voxels'] * 100) \
                          if stats['significant_voxels'] > 0 else 0
            f.write(f"    {stats['model']}: {overlap_pct:.1f}%\n")
        
        # Averaged statistics
        avg_stats = statistics['averaged_stats']
        f.write(f"\nAveraged T-map Statistics:\n")
        f.write(f"-"*80 + "\n")
        f.write(f"  Non-zero voxels: {avg_stats['non_zero_voxels']:,}\n")
        f.write(f"  T-map range: [{avg_stats['tmap_range'][0]:.3f}, {avg_stats['tmap_range'][1]:.3f}]\n")
        f.write(f"  T-map mean: {avg_stats['tmap_mean']:.4f}\n")
        f.write(f"  T-map std: {avg_stats['tmap_std']:.4f}\n")
        
        # Timestamp
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"✓ Statistics saved: {stats_path}")
    
    return {
        'tmap': tmap_path,
        'mask': mask_path,
        'stats': stats_path,
        **output_files
    }

def main():
    """Main function for cross-model t-map averaging"""
    print("="*80)
    print("CROSS-MODEL T-MAP AVERAGING WITH INTERSECTION MASK")
    print("="*80)
    
    parser = argparse.ArgumentParser(
        description="Average t-maps across 6 models for each language using intersection mask"
    )
    
    parser.add_argument("--base_dir", type=str, 
                       default="/leonardo_work/EUHPC_B24_036/yang/scratch_link/figures",
                       help="Base directory containing model subdirectories")
    parser.add_argument("--output_dir", type=str, 
                       default="cross_model_averaged_tmaps",
                       help="Output directory for averaged results")
    parser.add_argument("--p_threshold", type=float, default=0.01,
                       help="P-value threshold for mask creation (default: 0.01)")
    parser.add_argument("--viz_threshold", type=float, default=0.7,
                       help="Visualization threshold for 3D plots (default: 0.7)")
    parser.add_argument("--use_percentile_threshold", action='store_true',
                       help="If set, viz_threshold is percentile (0-1) of max value")
    parser.add_argument("--fixed_vmax", type=float, default=None,
                       help="Fixed maximum value for color scale (e.g., 8.3)")
    parser.add_argument("--fixed_vmin_threshold", type=float, default=None,
                       help="Fixed minimum threshold for display (e.g., 1.5)")
    parser.add_argument("--languages", type=str, nargs='+', 
                       default=['CN', 'EN', 'FR'],
                       help="Languages to process (default: CN EN FR)")
    
    args = parser.parse_args()
    
    print(f"\nConfiguration:")
    print(f"  Base directory: {args.base_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  P-value threshold: {args.p_threshold}")
    print(f"  Visualization threshold: {args.viz_threshold} "
          f"({'percentile' if args.use_percentile_threshold else 'absolute'})")
    if args.fixed_vmax is not None:
        print(f"  Fixed color scale max: {args.fixed_vmax}")
    if args.fixed_vmin_threshold is not None:
        print(f"  Fixed min display threshold: {args.fixed_vmin_threshold}")
    print(f"  Languages: {', '.join(args.languages)}")
    
    all_results = {}
    
    try:
        # Process each language
        for language in args.languages:
            print(f"\n\n{'#'*80}")
            print(f"# Processing {language} Participants")
            print(f"{'#'*80}")
            
            # Find t-map files for this language
            model_files = find_tmap_files_by_language(args.base_dir, language)
            
            if len(model_files) == 0:
                print(f"ERROR: No model files found for {language}!")
                continue
            
            if len(model_files) < 6:
                print(f"WARNING: Only {len(model_files)}/6 models found for {language}")
            
            # Average t-maps using intersection mask
            averaged_tmap, intersection_mask, ref_affine, ref_header, statistics = \
                average_tmaps_intersection_mask(model_files, args.p_threshold, language)
            
            # Save results
            output_files = save_results(
                averaged_tmap, intersection_mask, ref_affine, ref_header, 
                statistics, args.output_dir, args.viz_threshold, 
                args.use_percentile_threshold,
                fixed_vmax=args.fixed_vmax,
                fixed_vmin_threshold=args.fixed_vmin_threshold
            )
            
            all_results[language] = {
                'statistics': statistics,
                'files': output_files
            }
        
        # Print summary
        print(f"\n\n{'='*80}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"\nResults saved to: {args.output_dir}\n")
        
        for language, results in all_results.items():
            stats = results['statistics']
            print(f"{language} participants:")
            print(f"  Models: {len(stats['model_names'])} ({', '.join(stats['model_names'])})")
            print(f"  Intersection voxels: {stats['intersection_voxels']:,}")
            print(f"  Averaged t-map range: [{stats['averaged_stats']['tmap_range'][0]:.3f}, "
                  f"{stats['averaged_stats']['tmap_range'][1]:.3f}]")
            print(f"  Files: {results['files']['tmap']}")
            print()
        
        return all_results
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()

