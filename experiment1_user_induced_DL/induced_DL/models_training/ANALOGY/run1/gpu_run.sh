#!/bin/bash
#SBATCH -A rnk@v100
#SBATCH --job-name=e1_Analogy_inducedDL
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=e1_Analogy_inducedDL_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread

module load python
conda deactivate
conda activate torch_pyg

cd $WORK/dr_benchmark/experiment1_user_induced_DL/induced_DL/models_training/ANALOGY/run1

srun python $WORK/dr_benchmark/dev/run_training.py \
    --config params.yaml 