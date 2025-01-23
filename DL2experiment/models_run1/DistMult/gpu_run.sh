#!/bin/bash
#SBATCH -A rnk@v100
#SBATCH --job-name=e2r1p_DistMult
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=e2r1p_DistMult_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread
#SBATCH -C v100-32g
###SBATCH --qos=qos_gpu-t4

module load python
conda deactivate
conda activate torch_pyg

cd $WORK/DL2experiment/models_run1/DistMult/

srun python $WORK/dev/run_training.py \
    --config params.yaml 