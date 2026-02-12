#!/usr/bin/env python
import os
import torch
import argparse
from pathlib import Path

def check_tensor_stats(tensor_path):
    """Check statistics of a saved tensor"""
    try:
        tensor = torch.load(tensor_path, map_location='cpu')
        
        # Basic stats
        total_elements = tensor.numel()
        zero_elements = (tensor == 0).sum().item()
        non_zero_elements = total_elements - zero_elements
        
        # Value stats
        if non_zero_elements > 0:
            non_zero_tensor = tensor[tensor != 0]
            min_val = non_zero_tensor.min().item()
            max_val = non_zero_tensor.max().item()
            mean_val = non_zero_tensor.mean().item()
            std_val = non_zero_tensor.std().item()
        else:
            min_val = max_val = mean_val = std_val = 0.0
            
        return {
            'total_elements': total_elements,
            'zero_elements': zero_elements,
            'non_zero_elements': non_zero_elements,
            'zero_percentage': (zero_elements / total_elements) * 100,
            'min_val': min_val,
            'max_val': max_val,
            'mean_val': mean_val,
            'std_val': std_val,
            'shape': tuple(tensor.shape),
            'dtype': str(tensor.dtype)
        }
    except Exception as e:
        return {'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description='Check saved grad_mul_param tensors')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing grad-mul-param_checkpoint_* folders')
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"Directory {checkpoint_dir} does not exist!")
        return
    
    # Find all checkpoint subdirectories
    checkpoint_subdirs = [d for d in checkpoint_dir.iterdir() 
                         if d.is_dir() and d.name.startswith('grad-mul-param_checkpoint_')]
    
    if not checkpoint_subdirs:
        print("No grad-mul-param_checkpoint_* directories found!")
        return
    
    for subdir in sorted(checkpoint_subdirs):
        print(f"\n=== Checking {subdir.name} ===")
        
        # Find all .pt files in the subdirectory
        pt_files = list(subdir.glob('*.pt'))
        
        if not pt_files:
            print("No .pt files found in this checkpoint!")
            continue
        
        total_files = len(pt_files)
        all_zero_files = 0
        partially_zero_files = 0
        non_zero_files = 0
        
        print(f"Found {total_files} tensor files")
        
        for i, pt_file in enumerate(sorted(pt_files)):
            stats = check_tensor_stats(pt_file)
            
            if 'error' in stats:
                print(f"  ERROR loading {pt_file.name}: {stats['error']}")
                continue
            
            if stats['zero_percentage'] == 100.0:
                all_zero_files += 1
                if i < 5:  # Show details for first few files
                    print(f"  {pt_file.name}: ALL ZERO (shape: {stats['shape']})")
            elif stats['zero_percentage'] > 90.0:
                partially_zero_files += 1
                if i < 5:
                    print(f"  {pt_file.name}: {stats['zero_percentage']:.1f}% zero, "
                          f"range: [{stats['min_val']:.6f}, {stats['max_val']:.6f}]")
            else:
                non_zero_files += 1
                if i < 5:
                    print(f"  {pt_file.name}: {stats['zero_percentage']:.1f}% zero, "
                          f"range: [{stats['min_val']:.6f}, {stats['max_val']:.6f}], "
                          f"mean: {stats['mean_val']:.6f}, std: {stats['std_val']:.6f}")
        
        print(f"\nSummary for {subdir.name}:")
        print(f"  All-zero files: {all_zero_files}/{total_files} ({all_zero_files/total_files*100:.1f}%)")
        print(f"  Partially-zero files (>90% zero): {partially_zero_files}/{total_files}")
        print(f"  Files with significant non-zero values: {non_zero_files}/{total_files}")
        
        if all_zero_files == total_files:
            print("  ⚠️  WARNING: ALL FILES ARE COMPLETELY ZERO!")
        elif all_zero_files > total_files * 0.8:
            print("  ⚠️  WARNING: Most files are completely zero!")

if __name__ == "__main__":
    main() 