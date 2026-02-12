"""
Extract Neural Encoding Results to CSV

This script extracts neural encoding results from all models and languages,
computes per-subject mean correlations across voxels, and generates:
1. Detailed CSV: one row per subject with individual results
2. Summary CSV: one row per model-language combination with statistics

Usage:
    python extract_encoding_results.py --output_dir results_csv
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import os

warnings.filterwarnings("ignore")

# Model configurations
MODELS = {
    'llama2-13b': {
        'normal': 'llama2_13b',
        'break': 'llama2_13b_Break001',
        'random_break': 'llama2-13b_random-Break_001'
    },
    'llama2-7b': {
        'normal': 'llama2_7b',
        'break': 'llama2_7b_Break001',
        'random_break': 'llama2-7b_random-Break_001'
    },
    'mistral-8b': {
        'normal': 'mistral-8b',
        'break': 'mistral-8b_Break001',
        'random_break': 'mistral-8b_random-Break_001'
    },
    'mistral-12b': {
        'normal': 'mistral-12b',
        'break': 'mistral-12b_Break001',
        'random_break': 'mistral-12b_random-Break_001'
    },
    'qwen2.5-7b': {
        'normal': 'qwen2.5-7b',
        'break': 'qwen2.5-7b_Break001',
        'random_break': 'qwen2.5-7b_random-Break_001'
    },
    'qwen2.5-14b': {
        'normal': 'qwen2.5-14b',
        'break': 'qwen2.5-14b_Break001',
        'random_break': 'qwen2.5-14b_random-Break_001'
    }
}

LANGUAGES = ['EN', 'CN', 'FR']

# Expected subjects per language (based on observations)
EXPECTED_SUBJECTS = {
    'EN': list(range(57, 116)),  # EN057-EN115 (with some gaps)
    'CN': list(range(1, 38)),    # CN001-CN037 (with some gaps)
    'FR': list(range(1, 31))     # FR001-FR030 (estimated, adjust if needed)
}


def load_subject_result(result_path):
    """
    Load a subject's encoding result from npz file.
    
    Parameters:
    -----------
    result_path : Path
        Path to the npz file
        
    Returns:
    --------
    dict : Dictionary with 'mean_r', 'n_voxels', 'n_folds' or None if failed
    """
    try:
        data = np.load(result_path)
        if 'cv_scores' not in data:
            print(f"  Warning: No 'cv_scores' in {result_path.name}")
            return None
            
        cv_scores = data['cv_scores']  # shape: (n_folds, n_voxels)
        
        # Average across folds
        mean_across_folds = cv_scores.mean(axis=0)  # shape: (n_voxels,)
        
        # Average across all voxels to get single value per subject
        mean_r = mean_across_folds.mean()
        
        return {
            'mean_r': float(mean_r),
            'n_voxels': int(cv_scores.shape[1]),
            'n_folds': int(cv_scores.shape[0])
        }
    except Exception as e:
        print(f"  Error loading {result_path}: {e}")
        return None


def extract_results(base_dir, output_dir, fallback_dir=None):
    """
    Extract all encoding results and generate CSV files.
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing results (e.g., 'scratch_link/result_MNI152')
    output_dir : str
        Output directory for CSV files
    fallback_dir : str, optional
        Fallback directory to check if results not found in base_dir
    """
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fallback_path = Path(fallback_dir) if fallback_dir else None
    
    # Lists to collect data
    detailed_results = []
    missing_subjects = []
    
    print("Extracting neural encoding results...")
    print(f"Base directory: {base_path}")
    if fallback_path:
        print(f"Fallback directory: {fallback_path}")
    print(f"Output directory: {output_path}\n")
    
    # Iterate over all models
    for model_name, model_dirs in tqdm(MODELS.items(), desc="Models"):
        for model_type, dir_name in model_dirs.items():
            
            # Iterate over languages
            for language in LANGUAGES:
                
                # Construct path to results (try base_dir first, then fallback_dir)
                results_dir = base_path / language / dir_name / "derivatives"
                
                if not results_dir.exists() and fallback_path:
                    # Try fallback directory
                    results_dir = fallback_path / language / dir_name / "derivatives"
                
                if not results_dir.exists():
                    print(f"  Warning: Directory not found: {results_dir}")
                    missing_subjects.append({
                        'model_name': model_name,
                        'model_type': model_type,
                        'language': language,
                        'issue': 'Directory not found',
                        'path': str(results_dir)
                    })
                    continue
                
                # Get all subject directories
                subject_dirs = sorted([d for d in results_dir.iterdir() 
                                     if d.is_dir() and d.name.startswith(f"sub-{language}")])
                
                loaded_subjects = []
                
                # Process each subject
                for sub_dir in subject_dirs:
                    subject_id = sub_dir.name.replace('sub-', '')
                    
                    # Find the npz file
                    npz_files = list(sub_dir.glob(f"sub-{subject_id}_desc-*.npz"))
                    if not npz_files:
                        missing_subjects.append({
                            'model_name': model_name,
                            'model_type': model_type,
                            'language': language,
                            'subject_id': subject_id,
                            'issue': 'NPZ file not found',
                            'path': str(sub_dir)
                        })
                        continue
                    
                    result_path = npz_files[0]
                    result = load_subject_result(result_path)
                    
                    if result is None:
                        missing_subjects.append({
                            'model_name': model_name,
                            'model_type': model_type,
                            'language': language,
                            'subject_id': subject_id,
                            'issue': 'Failed to load data',
                            'path': str(result_path)
                        })
                        continue
                    
                    # Add to detailed results
                    detailed_results.append({
                        'model_name': model_name,
                        'model_type': model_type,
                        'language': language,
                        'subject_id': subject_id,
                        'mean_r': result['mean_r'],
                        'n_voxels': result['n_voxels'],
                        'n_folds': result['n_folds']
                    })
                    loaded_subjects.append(subject_id)
                
                # Report progress
                print(f"  {model_name} ({model_type}) - {language}: {len(loaded_subjects)} subjects")
    
    # Convert to DataFrames
    df_detailed = pd.DataFrame(detailed_results)
    df_missing = pd.DataFrame(missing_subjects)
    
    # Save detailed results
    detailed_csv = output_path / "encoding_results_detailed.csv"
    df_detailed.to_csv(detailed_csv, index=False)
    print(f"\n✓ Detailed results saved to: {detailed_csv}")
    print(f"  Total records: {len(df_detailed)}")
    
    # Generate summary statistics
    summary_results = []
    for (model_name, model_type, language), group in df_detailed.groupby(['model_name', 'model_type', 'language']):
        summary_results.append({
            'model_name': model_name,
            'model_type': model_type,
            'language': language,
            'n_subjects': len(group),
            'mean_r_avg': group['mean_r'].mean(),
            'mean_r_std': group['mean_r'].std(),
            'mean_r_min': group['mean_r'].min(),
            'mean_r_max': group['mean_r'].max(),
            'mean_r_median': group['mean_r'].median(),
            'n_voxels': group['n_voxels'].iloc[0] if len(group) > 0 else 0,
            'n_folds': group['n_folds'].iloc[0] if len(group) > 0 else 0
        })
    
    df_summary = pd.DataFrame(summary_results)
    summary_csv = output_path / "encoding_results_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"✓ Summary statistics saved to: {summary_csv}")
    print(f"  Total model-language combinations: {len(df_summary)}")
    
    # Save missing/failed subjects report
    if len(df_missing) > 0:
        missing_csv = output_path / "encoding_results_missing.csv"
        df_missing.to_csv(missing_csv, index=False)
        print(f"\n⚠ Missing/failed subjects report saved to: {missing_csv}")
        print(f"  Total issues: {len(df_missing)}")
    else:
        print(f"\n✓ No missing or failed subjects!")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY BY MODEL AND LANGUAGE")
    print("="*80)
    print(df_summary.to_string(index=False))
    
    return df_detailed, df_summary, df_missing


def main():
    parser = argparse.ArgumentParser(
        description="Extract neural encoding results to CSV"
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        default='scratch_link/result_MNI152',
        help='Base directory containing encoding results'
    )
    parser.add_argument(
        '--fallback_dir',
        type=str,
        default=None,
        help='Fallback directory to check if results not found in base_dir'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='crosslingual-convergence-main/code/encoding_results_csv',
        help='Output directory for CSV files'
    )
    
    args = parser.parse_args()
    
    df_detailed, df_summary, df_missing = extract_results(args.base_dir, args.output_dir, args.fallback_dir)
    
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE!")
    print("="*80)
    print(f"\nGenerated files in {args.output_dir}:")
    print("  1. encoding_results_detailed.csv - Individual subject results")
    print("  2. encoding_results_summary.csv - Model-language statistics")
    if len(df_missing) > 0:
        print("  3. encoding_results_missing.csv - Missing/failed subjects report")
    
    return df_detailed, df_summary, df_missing


if __name__ == "__main__":
    main()

