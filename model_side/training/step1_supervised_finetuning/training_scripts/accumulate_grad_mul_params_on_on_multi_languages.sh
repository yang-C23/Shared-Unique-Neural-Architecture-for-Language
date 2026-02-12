# #!/bin/bash

# # Memory optimization environment variables
# export PYTHONPATH=$PYTHONPATH:$(pwd)
# export OMP_NUM_THREADS=1
# export TOKENIZERS_PARALLELISM=false
# # Optimize PyTorch memory allocation for large models
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# # Enable memory mapping for model loading
# export TRANSFORMERS_CACHE=/tmp/transformers_cache

#export CUDA_VISIBLE_DEVICES=0,1


# If CUDA_VISIBLE_DEVICES is not set by the caller, default to 0,1.
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"
export CUDA_VISIBLE_DEVICES
total_cards=4

# PRETRAIN_OUT=$1
# ZERO_STAGE=$2



if [ "$PRETRAIN_OUT" = "" ]; then
    #PRETRAIN_OUT=./path_to_save_step_English
    PRETRAIN_OUT=/leonardo_work/EUHPC_B24_036/yang/llama2_13b/path_to_save_step_English
fi


mkdir -p $PRETRAIN_OUT
mkdir -p training_log

echo $PRETRAIN_OUT


    # --model_name_or_path meta-llama/Llama-2-7b-hf \
    # Alternative smaller model for testing: meta-llama/Llama-2-7b-hf
deepspeed Unveiling-Linguistic-Regions-in-LLMs/training/step1_supervised_finetuning/accumulate_grad_mul_params_on_multi_languages.py  \
    --model_name_or_path /leonardo_work/EUHPC_B24_036/yang/llama2_13b/models/Llama-2-13b-hf \
    --pretrain_train_data_path /leonardo_work/EUHPC_B24_036/yang/Unveiling-Linguistic-Regions-in-LLMs/dataset/train/english/train \
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
    &> training_log/training_accumu_english.log

#--offload \
