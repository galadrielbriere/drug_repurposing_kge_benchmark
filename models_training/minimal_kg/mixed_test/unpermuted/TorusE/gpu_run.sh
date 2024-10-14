#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1g.5gb:1
#SBATCH --time 3-00:00:00
#SBATCH --job-name=TorusE
#SBATCH --output=TorusE_%j.txt
#SBATCH --ntasks=1
#SBATCH --mem=20G

module load conda

conda activate drug_repurposing_benchmark

python /shared/projects/ml_het_bio_nets/drug_repurposing_kge_benchmark/dev/run_training.py \
    --config /shared/projects/ml_het_bio_nets/drug_repurposing_kge_benchmark/models_training/minimal_kg/mixed_test/unpermuted/TorusE/params.yaml \
    2>&1 | tee /shared/projects/ml_het_bio_nets/drug_repurposing_kge_benchmark/models_training/minimal_kg/mixed_test/unpermuted/TorusE/out.log
