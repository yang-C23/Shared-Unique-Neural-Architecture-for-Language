#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import sys
import torch
from torch.utils.data.distributed import DistributedSampler # May be needed for rank determination even without dataloaders
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128" # Keep if relevant
from transformers import (
    AutoModelForCausalLM,
    SchedulerType, # Not needed for saving only
    get_scheduler, # Not needed for saving only
)
import transformers
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam # Not needed for saving only

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# Assume these utils are accessible in the path
# from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
# from utils.ds_utils import get_train_ds_config # Keep for potential ZeRO config
# from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters # Keep convert_lora_to_linear_layer
# from utils.model.model_utils import create_hf_model
# for qwen
from utils.qwen_utils import (
    print_rank_0, to_device, save_hf_format, set_random_seed, 
    get_all_reduce_mean, get_optimizer_grouped_parameters, 
    save_zero_three_model, load_hf_tokenizer, get_train_ds_config,
    create_hf_model, convert_linear_layer_to_lora, convert_lora_to_linear_layer, 
    only_optimize_lora_parameters
)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Perturb and save a transformers model based on region masks.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--region_dir",
        type=str,
        default=None,
        help="Path to parameter region masks (*.pt files).",
        required=False,
    )
    parser.add_argument(
        "--random_ratio",
        type=float,
        default=0.0,
        help="Ratio of parameters to randomly perturb (0.0 to 1.0). If > 0, region_dir is ignored.",
    )
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        required=True,
                        help="Where to store the perturbed model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible operations.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed operations on gpus")
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques (relevant for loading/saving large models).')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage (important for saving).')
    ## LoRA related args - keep if model might have LoRA layers to convert
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, it implies LoRA layers might need conversion.")
    parser.add_argument("--lora_module_name", # Keep if conversion needed
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    # Memory optimization arguments
    parser.add_argument("--use_bf16",
                        action='store_true',
                        help="Use bfloat16 precision to reduce memory usage.")
    parser.add_argument("--cpu_offload_masks",
                        action='store_true', 
                        default=True,
                        help="Keep masks on CPU to reduce GPU memory usage. Default is True.")
    # Add deepspeed config args if needed for ZeRO saving
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Basic validation
    if args.random_ratio == 0 and args.region_dir is None:
        raise ValueError("Either --region_dir or --random_ratio > 0 must be provided.")
    if args.random_ratio == 0 and not os.path.isdir(args.region_dir):
        raise ValueError(f"Region directory not found: {args.region_dir}")
    if args.output_dir is None:
        raise ValueError("--output_dir is required.")

    return args


def main():
    args = parse_args()

    # Setup device and distributed backend
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
             print_rank_0("Running on single GPU.", 0) # Use 0 as default rank
        else:
             print_rank_0("Running on CPU.", 0)
        # For single GPU/CPU, avoid distributed initialization
        args.global_rank = 0
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend THIS IS IMPORTANT FOR ZERO-3 SAVING
        deepspeed.init_distributed()
        args.global_rank = torch.distributed.get_rank()

    # Setup DeepSpeed config, primarily for ZeRO stage
    # The standard config function should work even for saving.
    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage)

    # If passed along, set the seed now.
    set_random_seed(args.seed)

    if args.local_rank != -1:
        torch.distributed.barrier()

    # Load tokenizer
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    # Ensure pad token is set for saving consistency if model uses it
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right' # Consistent with training script

    # Load model - potentially with DeepSpeed init for ZeRO
    # For ZeRO-3, initialization is needed before loading state dict
    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config=ds_config if args.zero_stage > 0 else None, # Enable ZeRO config for memory management
                            disable_dropout=True) # Dropout disabled like in training script
    
    # Apply precision optimization
    if args.use_bf16 and torch.cuda.is_available():
        print_rank_0("Converting model to bfloat16 for memory efficiency...", args.global_rank)
        model = model.to(torch.bfloat16)
    
    print_rank_0(f"Model loaded with dtype: {next(model.parameters()).dtype}", args.global_rank)

    # Initialize DeepSpeed Engine IF ZeRO stage > 0 for model loading/saving management
    engine = None
    if args.zero_stage > 0 and args.local_rank != -1:
        # Use DeepSpeed for memory management
        # Create dummy optimizer for initialization (not used in practice)
        from torch.optim import Adam
        dummy_optimizer = Adam(model.parameters(), lr=1e-4)
        
        model, dummy_optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=dummy_optimizer,
            config_params=ds_config,
            dist_init_required=True)
        engine = model
    elif args.zero_stage > 0 and args.local_rank == -1:
        print_rank_0("Warning: ZeRO stage > 0 requested but running on single GPU. Using standard PyTorch model.", args.global_rank)
        model.to(device)
    else:
         model.to(device)

    print_rank_0("Loading region masks and applying perturbation...", args.global_rank)

    # --- Optimized Perturbation Logic ---
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    model_to_pertub = engine if engine is not None else model

    with torch.no_grad():
        # Process parameters one by one to minimize memory usage
        for name, param in model_to_pertub.named_parameters():
            if 'layers.' not in name or 'norm' in name:
                continue
            
            if args.random_ratio > 0:
                try:
                    num_points = int(args.random_ratio * param.numel())
                    
                    if args.cpu_offload_masks:
                        # Work with CPU mask and GPU parameter
                        param_cpu = param.data.cpu()
                        
                        # Generate random indices on CPU directly to avoid GPU OOM for large tensors
                        # Using randperm on CPU is memory safe
                        perm = torch.randperm(param.numel(), device='cpu')[:num_points]
                        
                        # Apply perturbation on CPU tensor directly using indices
                        # This avoids creating a full-size boolean mask which is memory intensive
                        param_cpu.view(-1)[perm] = 0
                        
                        # Copy back to GPU
                        param.data.copy_(param_cpu)
                        
                        del param_cpu
                        del perm
                    else:
                        # Direct GPU operation (more efficient but uses more memory)
                        # Create mask on same device as param
                        weight_to_freeze = torch.zeros_like(param, dtype=torch.bool)
                        # Select random indices
                        perm = torch.randperm(param.numel(), device=param.device)[:num_points]
                        weight_to_freeze.view(-1)[perm] = True
                        
                        param.data[weight_to_freeze] = 0
                        del weight_to_freeze
                        del perm
                    
                    print_rank_0(f"Applied random perturbation to {name}", args.global_rank)

                except Exception as e:
                    print_rank_0(f"Error generating random mask for {name}: {e}", args.global_rank)
                    continue
                continue
                
            # Construct mask path
            mask_filename = f"{name.replace('module.', '')}.pt"
            save_path = os.path.join(args.region_dir, mask_filename)

            if not os.path.exists(save_path):
                print_rank_0(f"Warning: Mask file not found for {name} at {save_path}. Skipping.", args.global_rank)
                continue

            try:
                # Load mask with memory optimization
                weight_to_freeze = torch.load(save_path, map_location='cpu').bool()
                
                # Determine target device based on CPU offload setting
                if args.cpu_offload_masks:
                    mask_device = 'cpu'
                    print_rank_0(f"Keeping mask for {name} on CPU for memory efficiency", args.global_rank)
                else:
                    mask_device = param.device
                    weight_to_freeze = weight_to_freeze.to(mask_device)
                
                print_rank_0(f"Processing mask for {name} - Shape: {weight_to_freeze.shape}, Frozen: {weight_to_freeze.sum().item()}", args.global_rank)
                
                # MEMORY OPTIMIZED APPROACH - Process in chunks if needed
                if args.cpu_offload_masks:
                    # Work with CPU mask and GPU parameter
                    param_cpu = param.data.cpu()
                    param_cpu[weight_to_freeze] = 0
                    param.data.copy_(param_cpu)
                    del param_cpu
                else:
                    # Direct GPU operation (more efficient but uses more memory)
                    param.data[weight_to_freeze] = 0
                
                print_rank_0(f"Applied perturbation to {name}", args.global_rank)
                
                # Clear mask tensor
                del weight_to_freeze
                
            except Exception as e:
                print_rank_0(f"Error processing {name} from {save_path}: {e}", args.global_rank)
                continue
            
            # Force garbage collection after each parameter to free memory
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

    print_rank_0("Perturbation complete. Starting model save...", args.global_rank)
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    # --- Saving Logic ---
    if args.output_dir is not None:
        print_rank_0(f'Saving perturbed model to {args.output_dir}...', args.global_rank)

        # Use the model instance that holds the parameters (engine for ZeRO, model otherwise)
        model_to_save = engine if engine is not None else model

        # If LoRA was used in the original model, convert back before saving full model
        # Check if the loaded model potentially had LoRA layers based on args or model config
        if args.lora_dim > 0: # Simple check, might need refinement
             print_rank_0("Converting potential LoRA layers back to linear...", args.global_rank)
             model_to_save = convert_lora_to_linear_layer(model_to_save)


        if args.global_rank == 0:
             # Create output directory if it doesn't exist
             os.makedirs(args.output_dir, exist_ok=True)
             # Save in Hugging Face format (works for ZeRO 0/1/2 and single GPU/CPU)
             # For ZeRO-3, this saves config/tokenizer but not weights on rank 0
             save_hf_format(model_to_save, tokenizer, args)

        if args.zero_stage == 3:
            # Special function needed for ZeRO-3 saving consolidation
            print_rank_0("Using ZeRO-3 save function...", args.global_rank)
            # Ensure the save_zero_three_model function handles the model engine correctly
            save_zero_three_model(model_to_save,
                                    args.global_rank,
                                    args.output_dir,
                                    zero_stage=args.zero_stage)
        elif args.global_rank == 0 and args.zero_stage < 3 :
             # For ZeRO 0, 1, 2 or no deepspeed, rank 0 already saved weights with save_hf_format
             print_rank_0("Standard save complete.", args.global_rank)


        print_rank_0("Model saving complete.", args.global_rank)
    # --- End Saving Logic ---

if __name__ == "__main__":
    main() 