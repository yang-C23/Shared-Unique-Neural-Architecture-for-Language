import torch
import argparse
from transformers import AutoModelForCausalLM
import os
from tqdm import tqdm

def check_random_ablation(original_model_path, perturbed_model_path, target_ratio=0.01, tolerance=0.001):
    print(f"Loading original model from {original_model_path}...")
    original_model = AutoModelForCausalLM.from_pretrained(original_model_path, torch_dtype=torch.float16, device_map="cpu")
    
    print(f"Loading perturbed model from {perturbed_model_path}...")
    perturbed_model = AutoModelForCausalLM.from_pretrained(perturbed_model_path, torch_dtype=torch.float16, device_map="cpu")
    
    print("\nChecking parameters...")
    
    total_params_checked = 0
    total_zeros = 0
    
    all_ratios = []
    
    for (name_orig, param_orig), (name_pert, param_pert) in zip(original_model.named_parameters(), perturbed_model.named_parameters()):
        # Ensure we are comparing the same parameters
        assert name_orig == name_pert, f"Parameter mismatch: {name_orig} vs {name_pert}"
        
        # Skip non-layer parameters (embeddings, norms, etc. typically aren't masked in this logic unless specified)
        # Adjust this condition based on exactly what your script targets.
        # usually 'layers.' is the key, and norms are often excluded.
        if 'layers.' not in name_orig or 'norm' in name_orig:
            continue
            
        # Calculate difference
        # In the perturbed model, masked weights should be 0 (if zero_out was used) 
        # OR they should match the original if the logic was "preserve frozen, train others" but we are saving the INITIAL state.
        
        # Let's clarify the logic from save_perturbed_model.py:
        # weight_to_freeze = random_mask (1% True)
        # param_clone_masked = weight_to_freeze * param_clone  (The 1% random weights)
        # param.mul_(~weight_to_freeze) (Zero out the 1% random weights in the model)
        # param.add_(saved_weights[name].data) (Add them back? Wait.)
        
        # Re-reading save_perturbed_model.py logic:
        # param_clone_masked = weight_to_freeze * param_clone  <-- This captures the values at the 1% locations
        # param.mul_(~weight_to_freeze)                        <-- This ZEROS OUT the 1% locations
        # param.add_(saved_weights[name].data)                 <-- This ADDS BACK the 1% locations?
        
        # IF the logic adds them back, then the saved model is IDENTICAL to the original? 
        # Wait, let me re-read the code carefully.
        
        # Ah, in region_freeze_train.py (the training script), the goal is to FREEZE those parameters.
        # But usually 'ablation' implies REMOVING them (setting to zero).
        
        # In the save_perturbed_model.py provided:
        # param.mul_(~weight_to_freeze)  -> Sets the masked (frozen) area to 0.
        # param.add_(saved_weights[name].data) -> Adds the original values BACK to the masked area.
        
        # So the result is... the original parameters?
        # If so, then "save_perturbed_model.py" is just saving the original model but preparing it for training?
        # OR did I misunderstand the "perturbation" intent?
        
        # Usually "ablation" means setting to zero.
        # If the user wants "1% random ablation", they likely want 1% of parameters set to ZERO.
        
        # Let's check the code I edited in save_perturbed_model.py again.
        # It follows the logic that was already there.
        
        # Let's assume the user wants to verify that 1% of the weights are somehow "treated".
        # If the script restores them, then we can't detect it by comparing values to original.
        # BUT if the script was meant to "zero out" the 1% (ablation), then the logic `param.add_` might be wrong for "ablation", 
        # but correct for "freeze during training".
        
        # However, the user called it "random ablation" model. 
        # If it is ablation, those weights should be zero.
        
        # Let's check if the weights are zero or equal to original.
        
        is_zero = (param_pert == 0)
        is_diff = (param_pert != param_orig)
        
        # Count zeros in the perturbed parameter
        num_zeros = is_zero.sum().item()
        num_params = param_pert.numel()
        ratio_zeros = num_zeros / num_params
        
        # Check if the values are different from original
        # If they are different, it means they were modified.
        # If they are zero and original was not zero, it's ablation.
        
        # Let's count how many positions are (Zero in Perturbed AND Non-Zero in Original)
        # This is the strict definition of "ablated weights".
        ablated_mask = (param_pert == 0) & (param_orig != 0)
        num_ablated = ablated_mask.sum().item()
        ratio_ablated = num_ablated / num_params
        
        print(f"Layer: {name_orig}")
        print(f"  Shape: {param_pert.shape}")
        print(f"  Total Zeros in Perturbed: {num_zeros} ({ratio_zeros:.4%})")
        print(f"  Strictly Ablated (Orig!=0 -> Pert=0): {num_ablated} ({ratio_ablated:.4%})")
        
        all_ratios.append(ratio_ablated)
        
        # Heuristic check
        if abs(ratio_ablated - target_ratio) > tolerance:
            print(f"  [WARNING] Ablation ratio {ratio_ablated:.4%} deviates from target {target_ratio:.4%}")
        else:
            print(f"  [OK] Ablation ratio matches target.")
            
    avg_ratio = sum(all_ratios) / len(all_ratios) if all_ratios else 0
    print(f"\nOverall Average Ablation Ratio: {avg_ratio:.4%}")
    if abs(avg_ratio - target_ratio) < tolerance:
        print("SUCCESS: Model appears to be correctly ablated.")
    else:
        print("FAILURE: Model ablation ratio does not match target.")
        print("Note: If the ratio is 0.00%, it means the script might be restoring the weights (Freezing logic) instead of Zeroing them (Ablation logic).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_model_path", type=str, required=True)
    parser.add_argument("--perturbed_model_path", type=str, required=True)
    parser.add_argument("--ratio", type=float, default=0.01)
    args = parser.parse_args()
    
    check_random_ablation(args.original_model_path, args.perturbed_model_path, args.ratio)
