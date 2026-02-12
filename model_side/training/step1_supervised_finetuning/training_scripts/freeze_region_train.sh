: "${CUDA_VISIBLE_DEVICES:=0,1}"
export CUDA_VISIBLE_DEVICES
total_cards=2

PRETRAIN_OUT=$1
ZERO_STAGE=$2



if [ "$PRETRAIN_OUT" == "" ]; then
    PRETRAIN_OUT=~/Unveiling-Linguistic-Regions-in-LLMs/ragion_freeze_path_to_save/llama_break_top001
fi
#~/scratch/brain_encoding/model/
#./ragion_freeze_path_to_save/llama_CNBreakonly_top01
mkdir -p $PRETRAIN_OUT

echo $PRETRAIN_OUT

deepspeed training/step1_supervised_finetuning/region_freeze_train.py  \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --pretrain_train_data_path dataset/train/english/train \
    --pretrain_test_data_path dataset/test/chinese/test \
    --english_test_data_path dataset/test/english/test \
    --region_dir path_to_save_region_selection/three-countries-accumulated-llama2-grad-mul-param/10000/top0.01 \
    --learning_rate 5e-5 \
    --weight_decay 0.001 \
    --total_cards $total_cards \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --per_device_eval_batch_size 16 \
    --zero_stage 2 \
    --offload \
    --seed 1234 \
    --deepspeed \
    --output_dir $PRETRAIN_OUT \
    &> ./training_log/training_llama_Breaktop001.log
