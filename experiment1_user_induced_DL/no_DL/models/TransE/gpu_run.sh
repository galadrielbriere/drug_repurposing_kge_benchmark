#!/bin/bash
#SBATCH -A rnk@v100
#SBATCH --job-name=e1r1nodl_TransE
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --output=e1r1nodl_TransE_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread
#SBATCH -C v100-32g
#SBATCH --qos=qos_gpu-t4

module load python
conda deactivate
conda activate torch_pyg

cd $WORK/dr_benchmark/experiment1_user_induced_DL/no_DL/models/TransE/

srun python $WORK/dr_benchmark/dev/run_training.py \
    --config params.yaml 