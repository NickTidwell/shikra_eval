#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --exclude=evc[1-14]
#SBATCH --job-name=eval-pqa
#SBATCH --error=POINT-QA-EVAL.err
#SBATCH --output=POINT-QA-EVAL.out
## Load Modules ##
module load anaconda/anaconda3
# Load the Python and CUDA modules
module load python/python-3.8.0-gcc-9.1.0
module load cuda/cuda-10.2
# List the modules that are loaded
module list
# Activate the GPU version of PyTorch
source activate pytorch-1.8.0+cuda10_2
## Run job ##
conda activate ~/my_env/shikra
cd ..
accelerate launch --num_processes 4 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/shikra_eval_pointqa.py \
        --cfg-options model_args.model_name_or_path=shikra-7b
deactivate

echo "Ending script..."
date

