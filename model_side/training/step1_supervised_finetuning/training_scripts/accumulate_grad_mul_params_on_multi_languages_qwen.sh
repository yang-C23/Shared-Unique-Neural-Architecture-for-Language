#!/bin/bash

# Memory optimization environment variables
# export PYTHONPATH=$PYTHONPATH:$(pwd)
# export OMP_NUM_THREADS=1
# export TOKENIZERS_PARALLELISM=false
# # Optimize PyTorch memory allocation for large models
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# # Enable memory mapping for model loading
# export TRANSFORMERS_CACHE=/tmp/transformers_cache

# If CUDA_VISIBLE_DEVICES is not set by the caller, default to 0,1,2,3.
: "${CUDA_VISIBLE_DEVICES:=0,1}"
export CUDA_VISIBLE_DEVICES
total_cards=2

# PRETRAIN_OUT can be set by caller, otherwise use default
if [ "$PRETRAIN_OUT" = "" ]; then
    PRETRAIN_OUT=/leonardo_work/EUHPC_B24_036/yang/qwen2_5_7b/path_to_save_step_English_test
fi

mkdir -p $PRETRAIN_OUT
mkdir -p training_log

echo "Output directory: $PRETRAIN_OUT"

# Run the training script for Qwen2.5-7B
deepspeed Unveiling-Linguistic-Regions-in-LLMs/training/step1_supervised_finetuning/accumulate_grad_mul_params_on_multi_languages_qwen.py \
    --model_name_or_path /leonardo_work/EUHPC_B24_036/yang/qwen2.5_7b/models/Qwen2.5-7B \
    --pretrain_train_data_path /leonardo_work/EUHPC_B24_036/yang/Unveiling-Linguistic-Regions-in-LLMs/dataset/test/english/test \
    --pretrain_test_data_path /leonardo_work/EUHPC_B24_036/yang/Unveiling-Linguistic-Regions-in-LLMs/dataset/test/english/test \
    --max_seq_len 512 \
    --learning_rate 5e-5 \
    --weight_decay 0.001 \
    --total_cards $total_cards \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 8 \
    --zero_stage 3 \
    --offload \
    --seed 1234 \
    --deepspeed \
    --output_dir $PRETRAIN_OUT \
    &> training_log/training_accumu_english_qwen.log 