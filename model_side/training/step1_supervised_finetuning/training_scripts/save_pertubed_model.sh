PRETRAIN_OUT=~/Unveiling-Linguistic-Regions-in-LLMs/ragion_freeze_path_to_save/llama_break_top001_perturbed_saved # Choose a distinct output directory
REGION_DIR=~/Unveiling-Linguistic-Regions-in-LLMs/path_to_save_region_selection/three-countries-accumulated-llama2-grad-mul-param/10000/top0.01
MODEL_PATH=llama2_7b/models/llama-2-7b-hf
ZERO_STAGE=0  # Changed from 2 to 0 to avoid DeepSpeed ZeRO and MPI requirements
SEED=1234

mkdir -p $PRETRAIN_OUT

# Note: Ensure CUDA_VISIBLE_DEVICES is set appropriately if using GPU (e.g., export CUDA_VISIBLE_DEVICES=0)
# If using multiple GPUs (matching the training script's total_cards > 1) and ZeRO Stage > 0, 
# you would typically use the deepspeed launcher:
# deepspeed --num_gpus=2 training/step1_supervised_finetuning/save_perturbed_model.py \
# However, for saving with 1 or 0 GPUs as requested, the python command is sufficient.

python training/step1_supervised_finetuning/save_perturbed_model.py \
  --model_name_or_path $MODEL_PATH \
  --region_dir $REGION_DIR \
  --output_dir $PRETRAIN_OUT \
  --zero_stage $ZERO_STAGE \
  --offload \
  --seed $SEED 
  # Add --deepspeed_config path/to/ds_config.json if you have specific non-standard deepspeed settings
  # otherwise, the script uses get_train_ds_config to generate one based on zero_stage and offload.


echo "Perturbed model saved to $PRETRAIN_OUT"





  