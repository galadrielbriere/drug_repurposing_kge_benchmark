#!/bin/bash
#SBATCH -A rnk@v100
#SBATCH --job-name=e2r1p_TransE
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --output=e2r1p_TransE_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread
#SBATCH -C v100-32g
#SBATCH --qos=qos_gpu-t4

module load python
conda deactivate
conda activate torch_pyg

cd $WORK/dr_benchmark/experiment2_structural_DL/permuted/models/TransE/

srun python $WORK/dr_benchmark/dev/run_training.py \
    --config params.yaml 