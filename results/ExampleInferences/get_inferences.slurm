#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --exclude=evc[1-14]
#SBATCH --job-name=example_inf
#SBATCH --error=INFERENCE.err
#SBATCH --output=INFERENCE.out

## Load Modules ##
module load anaconda/anaconda3

# List the modules that are loaded
module list

conda init bash
# Activate the GPU version of PyTorch
source activate pytorch-1.8.0+cuda10_2

## Run job ##
conda activate ~/my_env/shikra
cd ..

python check.py --model_path shikra-7b --mode 0

deactivate

echo "Ending script..."
date

