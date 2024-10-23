#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1g.5gb:1
#SBATCH --time 1-00:00:00
#SBATCH --job-name=install
#SBATCH --output=install_%j.txt
#SBATCH --ntasks=1
#SBATCH --mem=20G

nvidia-smi
module load conda

conda create --name torch_pyg python=3.10  
source /shared/ifbstor1/software/miniconda/etc/profile.d/conda.sh
conda activate torch_pyg
pip install torch torchvision torchaudio
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install torchkge
pip install pandas matplotlib numpy pyyaml tqdm ignite
pip install pytorch-ignite
