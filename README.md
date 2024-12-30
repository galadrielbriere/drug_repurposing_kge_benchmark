# Data Leakage in Biomedical Knowledge Graph Link Prediction: A Benchmark Study

### Authors

- **Galadriel Brière** (Aix Marseille Univ, INSERM, MMG, Marseille, France) 
- **Thomas Stosskopf** (TAGC, TGML, INSERM, UMR1090, Aix-Marseille University) †
- **Benjamin Loire** (Aix Marseille Univ, INSERM, MMG, Marseille, France) †
- **Anaïs Baudot** (Aix Marseille Univ, INSERM, MMG, Marseille, France; Barcelona Supercomputing Center, Barcelona, Spain)

† Equal contribution

---

## Introduction

This repository focuses on evaluating and mitigating data leakage in link prediction tasks for link prediction in biomedical knowledge graphs. It includes a configurable pipeline for preprocessing, training, and evaluating popular KGE models on link prediction tasks. The project includes models from TorchKGE and PyTorch Geometric. Models are trained using PyTorch and PyTorch-Ignite.

---

## Features

- Knowledge graph preprocessing with data leakage control.
- Customizable training and evaluation pipelines using configuration files.
- Supports popular KGE models from TorchKGE and PyTorch Geometric.

---

## Installation

### Prerequisites

- Python >= 3.10
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/galadrielbriere/drug_repurposing_kge_benchmark.git
   cd drug_repurposing_kge_benchmark
   ```

2. Create the environment and install dependencies:
   ```bash
   conda create --name torch_pyg python=3.10  
   conda activate torch_pyg
   pip install torch torchvision torchaudio
   pip install torch_geometric
   pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
   pip install torchkge
   pip install pandas matplotlib numpy pyyaml tqdm ignite pytorch-ignite
   ```

---

## Usage

### Configuration

Modify the `config.yaml` file to set parameters for KG preprocessing, training, and evaluation.

Bellow is an example of expected configuration file.

```yaml
common:  # Global parameters
  seed: 42  # Seed for reproducibility
  input_csv: "/path/to/knowledge_graph.csv"  # Input KG file
  out: "./results"  # Output directory for results
  run_kg_prep: true  # Run KG preprocessing
  run_training: false  # Train embeddings
  run_evaluation: false  # Evaluate the model on the test set

clean_kg:  # Preprocessing settings
  remove_duplicates_triplets: true  # Remove duplicate triples

  make_directed: true  # Make undirected relations directed
  make_directed_params:  # Relations to make directed
    - "disease_disease"
    - "drug_drug"
    - "protein_protein"

  check_synonymous_antisynonymous: true  # Remove redundant or Cartesian product relations
  check_synonymous_antisynonymous_params:
    theta1: 0.8  # Threshold for near-duplicate relations
    theta2: 0.8  # Threshold for near-reverse relations

  permute_kg: false  # Whether to permute a specific relation
  permute_kg_params:  # Relations to permute
    - "indication"  

model:  # Model settings
  name: "TransE"  # Model name (any from TorchKGE)
  emb_dim: 400  # Embedding dimension
  margin: 1  # Margin for translational models 

sampler:  # Sampling strategy
  name: "Mixed"  # Sampler to use (Uniform, Positional, Bernouilli or Mixed)

optimizer:  # Optimizer configuration
  name: "Adam"  # Optimizer name (any from PyTorch)
  params:
    lr: 0.001 # Learning Rate
    weight_decay: 0.001  # Weight decay (L2 regularization)

lr_scheduler:  # Learning rate scheduler
  type: "CosineAnnealingWarmRestarts"  # Scheduler type (any from Pytorch)
  params:
    T_0: 10  # First cycle length
    T_mult: 2  # Multiplicative factor for cycle length

training:  # Training parameters
  max_epochs: 2000  # Maximum number of epochs
  patience: 20  # Early stopping patience on validation MRR
  batch_size: 2048  # Training batch size
  eval_interval: 10  # Evaluation interval during training
  eval_batch_size: 32  # Evaluation batch size

evaluation:  # Evaluation settings
  made_directed_relations:  # Relations made directed
    - "drug_drug"
    - "disease_disease"
    - "protein_protein"
    - "drug_drug_inv"
    - "disease_disease_inv"
    - "protein_protein_inv"
  target_relations:  # Specific relations to evaluate
    - "indication"
  thresholds: [10]  # Thresholds for node frequencies (used for each target relation)
```

---

### Pipeline execution

```bash
python ./dev/run_training.py --config config.yaml 
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
