#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1g.5gb:1
#SBATCH --time 3-00:00:00
#SBATCH --job-name=ConvKB
#SBATCH --output=ConvKB_%j.txt
#SBATCH --ntasks=1
#SBATCH --mem=20G

module load conda
source /shared/ifbstor1/software/miniconda/etc/profile.d/conda.sh
conda activate torch_pyg

python /shared/projects/ml_het_bio_nets/drug_repurposing_kge_benchmark/dev/run_training.py \
    --config /shared/projects/ml_het_bio_nets/drug_repurposing_kge_benchmark/models_training/minimal_kg/mixed_test/unpermuted/ConvKB/params.yaml \
    2>&1 | tee /shared/projects/ml_het_bio_nets/drug_repurposing_kge_benchmark/models_training/minimal_kg/mixed_test/unpermuted/ConvKB/out.log
