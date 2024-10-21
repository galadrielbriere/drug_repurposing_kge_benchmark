#!/bin/bash
#SBATCH -A rnk@v100
#SBATCH --job-name=e1_HolE_noDL
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=e1_HolE_noDL_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread

module load python
conda deactivate
conda activate torch_pyg

cd $WORK/dr_benchmark/experiment1_user_induced_DL/no_DL/models_training/HolE/run1

srun python $WORK/dr_benchmark/dev/run_training.py \
    --config params.yaml 