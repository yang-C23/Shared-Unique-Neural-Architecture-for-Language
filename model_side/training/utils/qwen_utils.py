# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import torch
import random
import numpy as np
import math
import json
import deepspeed
from transformers import (
    set_seed, 
    AutoTokenizer, 
    AutoConfig,
    AutoModelForCausalLM,
    SchedulerType,
    get_scheduler,
)
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from transformers.integrations import HfDeepSpeedConfig


def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        output[k] = v.to(device)
    return output


def load_hf_tokenizer(model_name_or_path, fast_tokenizer=True):
    """Load tokenizer with forced offline mode"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, 
        fast_tokenizer=fast_tokenizer,
        local_files_only=True,
        trust_remote_code=False
    )
    return tokenizer


def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    rlhf_training=False,
                    disable_dropout=False):
    """Create HF model with forced offline mode"""
    model_config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=True)
    if disable_dropout:
        model_config.dropout = 0.0
    
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
    
    if rlhf_training:
        # the weight loading is handled by create critic model
        model = model_class.from_config(model_config)
    else:
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config,
            local_files_only=True)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(
        8 * math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    return model


def save_hf_format(model, tokenizer, args, sub_folder=""):
    """Save model in HuggingFace format"""
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(args.output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    for key in list(save_dict.keys()):
        if "lora" in key:
            del save_dict[key]
        if "svd" in key:
            del save_dict[key]
        if "sigma" in key:
            del save_dict[key]
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_pretrained(output_dir)


def set_random_seed(seed):
    """Set random seed for reproducibility"""
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_all_reduce_mean(tensor):
    """Get mean across all processes"""
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def get_optimizer_grouped_parameters(model,
                                     weight_decay,
                                     no_decay_name_list=["bias", "LayerNorm.weight"]):
    """Get optimizer parameters with weight decay grouping"""
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def _z3_params_to_fetch(param_list):
    """Helper for ZeRO stage 3 parameter fetching"""
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0):
    """Save model for ZeRO stage 3"""
    zero_stage_3 = (zero_stage == 3)
    os.makedirs(save_dir, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)

    model_to_save = model_ema.module if hasattr(model_ema, 'module') else model_ema
    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():
            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v]),
                                                       enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k and "svd" not in k:
                output_state_dict[k] = v_p
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict


def get_train_ds_config(offload, stage=2):
    """Get DeepSpeed configuration for training"""
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {"device": device},
        "offload_optimizer": {"device": device},
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False
    }
    
    return {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "fp16": {
            "enabled": False
        },
        "bf16": {
            "enabled": True
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }


# LoRA utility functions (stubs for compatibility)
def convert_linear_layer_to_lora(model, part_module_name, lora_dim=0, lora_scaling=1, lora_droppout=0):
    """Convert linear layers to LoRA (stub for compatibility)"""
    return model


def convert_lora_to_linear_layer(model):
    """Convert LoRA back to linear layers (stub for compatibility)"""
    return model


def only_optimize_lora_parameters(model):
    """Only optimize LoRA parameters (stub for compatibility)"""
    return model 