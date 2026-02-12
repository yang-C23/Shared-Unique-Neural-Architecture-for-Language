#!/bin/bash
#SBATCH --job-name=ppl_eval_qwen
#SBATCH --output=%j.ppl_eval_qwen.out
#SBATCH --error=%j.ppl_eval_qwen.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=EUHPC_D26_076
#SBATCH --partition=boost_usr_prod

######################## 加载模块 ########################
module purge
module load  gmp/6.2.1 mpfr/4.1.0 mpc/1.2.1
module load gcc/11.3.0
module load cuda/11.8
module load anaconda3/2023.03
module load cmake/3.24.3

cd /leonardo_work/EUHPC_D29_009/yang

######################## PPL Evaluation ########################
# Explicitly use the python executable from the environment to avoid path issues
VENV_ROOT="/leonardo_work/EUHPC_D29_009/yang/unveiling_env"
PYTHON_EXEC="$VENV_ROOT/bin/python"

echo "Checking environment..."
echo "Python executable: $PYTHON_EXEC"
$PYTHON_EXEC -c "import sys; print(sys.executable); import datasets; print('datasets module found')" || exit 1

LOG_FILE="Unveiling-Linguistic-Regions-in-LLMs/training_log/ppl_results_qwen.log"
SCRIPT_PATH="Unveiling-Linguistic-Regions-in-LLMs/training/step1_supervised_finetuning/calc_ppl.py"

echo "Starting PPL Evaluation..."

# Qwen2.5-7B (Base)
echo "Evaluating Qwen2.5-7B on GPU 0..."
CUDA_VISIBLE_DEVICES=0 $PYTHON_EXEC $SCRIPT_PATH \
    --model_path /leonardo_work/EUHPC_D29_009/yang/qwen2.5_7b/models/Qwen2.5-7B \
    --output_log $LOG_FILE &

# qwen2.5-7b_break_top001
# Uncommented this as requested by user in initial prompt, 
# but respecting user's latest file state (commented out). 
# If you want to run both, uncomment below lines.
# echo "Evaluating qwen2.5-7b_break_top001 on GPU 1..."
# CUDA_VISIBLE_DEVICES=1 $PYTHON_EXEC $SCRIPT_PATH \
#     --model_path /leonardo_work/EUHPC_D29_009/yang/qwen2.5_7b/models/qwen2.5-7b_break_top001 \
#     --output_log $LOG_FILE &

wait

echo "Evaluation completed. Results saved to $LOG_FILE"
