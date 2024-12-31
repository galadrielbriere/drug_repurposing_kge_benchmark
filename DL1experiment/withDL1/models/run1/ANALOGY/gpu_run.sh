#!/bin/bash
#SBATCH -A rnk@v100
#SBATCH --job-name=e1r1dl_ANALOGY
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=e1r1dl_ANALOGY_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread
#SBATCH -C v100-32g
###SBATCH --qos=qos_gpu-t4

module load python
conda deactivate
conda activate torch_pyg

cd $WORK/dr_benchmark/DL1experiment/withDL1/models/run1/ANALOGY/

srun python $WORK/dr_benchmark/dev/run_training.py \
    --config params.yaml 