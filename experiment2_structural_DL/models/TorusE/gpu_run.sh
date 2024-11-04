#!/bin/bash
#SBATCH -A rnk@v100
#SBATCH --job-name=e2r1p_TorusE
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=e2r1p_TorusE_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --hint=nomultithread
#SBATCH -C v100-32g
##SBATCH --qos=qos_gpu-t4

module load python
conda deactivate
conda activate torch_pyg

cd $WORK/dr_benchmark/experiment2_structural_DL/models/TorusE/

srun python $WORK/dr_benchmark/dev/run_training.py \
    --config params.yaml 