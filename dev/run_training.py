import logging

# Configure logging
logging.captureWarnings(True)
log_level = logging.INFO# if config["common"]['verbose'] else logging.WARNING
logging.basicConfig(
    level=log_level,  
    format='%(asctime)s - %(levelname)s - %(message)s' 
)

import sys
import os
import argparse
import torch
import torch.optim as optim
import torchkge.models
import torchkge.sampling 
import numpy as np
from torch.optim import lr_scheduler
import warnings
import yaml 
import gc
import json
import csv 
import matplotlib.pyplot as plt
import TransGNN
import DistGNN
import pandas as pd
import networkx as nx

from torchkge.utils.datasets import load_fb15k

from torchkge.evaluation import LinkPredictionEvaluator
from torchkge.utils import MarginLoss, BinaryCrossEntropyLoss, DataLoader

from ignite.metrics import RunningAverage
from ignite.engine import Events, Engine
from ignite.handlers import EarlyStopping, ModelCheckpoint, Checkpoint, DiskSaver

from kg_processing import (
    parse_yaml,
    set_random_seeds,
    prepare_knowledge_graph,
    load_knowledge_graph,
)

from my_knowledge_graph import KnowledgeGraph
from mixed_sampler import MixedNegativeSampler
from positional_sampler import PositionalNegativeSampler

def main(args):

    torch.backends.cudnn.benchmark = True

    # Load configuration
    config = parse_yaml(args.config)
  
    logging.info("Configuration loaded:\n%s", json.dumps(config, indent=4))

    # Create output folder if it doesn't exist
    os.makedirs(config["common"]['out'], exist_ok=True)
    
    # Print common parameters
    num_cores = len(os.sched_getaffinity(0))
    logging.info(f"Setting number of threads to {num_cores}")
    torch.set_num_threads(num_cores)

    logging.info(f'Loading parameters from {args.config}')
    logging.info(f"Output folder: {config['common']['out']}")


    run_kg_prep =  config['common'].get('run_kg_prep', True)
    run_training =  config['common'].get('run_evaluation', True)
    run_plot =  config['common'].get('plot_training_metrics', True)
    run_eval =  config['common'].get('run_evaluation', True)
    run_inference = config['common'].get('run_inference', True)

    if run_kg_prep:
            logging.info(f"Preparing KG...")
            kg_train, kg_val, kg_test = prepare_knowledge_graph(config)
    else:
        logging.info("Loading KG...")
        kg_train, kg_val, kg_test = load_knowledge_graph(config)
        logging.info("Done")

    # kg_train, kg_val, kg_test = load_fb15k()

    if run_training or run_plot or run_eval or run_inference:
        train_model(kg_train, kg_val, kg_test, config)


def plot_learning_curves(training_metrics_file, config):
    df = read_training_metrics(training_metrics_file)
    df['Training Loss'] = pd.to_numeric(df['Training Loss'], errors='coerce')
    df['Validation MRR'] = pd.to_numeric(df['Validation MRR'], errors='coerce')
    
    plt.figure(figsize=(12, 5))

    # Plot pour la perte d'entraînement
    plt.subplot(1, 2, 1)
    plt.plot(df['Epoch'], df['Training Loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.savefig(os.path.join(config['common']['out'], 'training_loss_curve.png'))

    # Plot pour le MRR de validation
    plt.subplot(1, 2, 2)
    plt.plot(df['Epoch'], df['Validation MRR'], label='Validation MRR')
    plt.xlabel('Epoch')
    plt.ylabel('Validation MRR')
    plt.title('Validation MRR over Epochs')
    plt.legend()
    plt.savefig(os.path.join(config['common']['out'], 'validation_mrr_curve.png'))

def initialize_model(config, kg_train, device):
    """Initialize the model based on the configuration."""

    translational_models = ['TransE', 'TransH', 'TransR', 'TransD', 'TorusE', 'TransEModelWithGCN']

    model_name = config['model'].get('name', 'TransE')
    if 'name' not in config['model']:
        warnings.warn("The model name field is missing in the configuration. Defaulting to 'TransE'.")

    emb_dim = config['model'].get('emb_dim', 100)  
    if 'emb_dim' not in config['model']:
        warnings.warn("The model emb_dim field is missing in the configuration. Defaulting to 100.")

    rel_emb_dim = config['model'].get('rel_emb_dim', emb_dim)
    if model_name in ['TransR', 'TransD'] and 'rel_emb_dim' not in config['model']:
        warnings.warn(f"The 'rel_emb_dim' field is missing for model {model_name}. Defaulting to {emb_dim}.")

    dissimilarity = config['model'].get('dissimilarity', 'L2')
    if model_name == 'TransE' and 'dissimilarity' not in config['model']:
        warnings.warn(f"The 'dissimilarity' field is missing for model {model_name}. Defaulting to 'L2'.")
    if model_name == 'TorusE' and 'dissimilarity' not in config['model']:
        dissimilarity = 'torus_L2'
        warnings.warn(f"The 'dissimilarity' field is missing for model {model_name}. Defaulting to 'torus_L2'.")

    margin = config['model'].get('margin', 1.0) 
    if 'margin' not in config['model'] and model_name in translational_models:
        warnings.warn("The model margin field is missing in the configuration. Defaulting to 1.0.")

    # Translational models
    if model_name == 'TransE':
        model = torchkge.models.TransEModel(emb_dim, kg_train.n_ent, kg_train.n_rel,
                                            dissimilarity_type=dissimilarity)
        criterion = MarginLoss(margin)

    elif model_name == 'TransH':
        model = torchkge.models.TransHModel(emb_dim, kg_train.n_ent, kg_train.n_rel)
        criterion = MarginLoss(margin)

    elif model_name == 'TransR':
        model = torchkge.models.TransRModel(emb_dim, rel_emb_dim, kg_train.n_ent, kg_train.n_rel)
        criterion = MarginLoss(margin)

    elif model_name == 'TransD':
        model = torchkge.models.TransDModel(emb_dim, rel_emb_dim, kg_train.n_ent, kg_train.n_rel)
        criterion = MarginLoss(margin)

    elif model_name == 'TorusE':
        model = torchkge.models.TorusEModel(emb_dim, kg_train.n_ent, kg_train.n_rel,
                                            dissimilarity_type=dissimilarity)
        criterion = MarginLoss(margin)

    # Bilinear models
    elif model_name == 'DistMult':
        model = torchkge.models.DistMultModel(emb_dim, kg_train.n_ent, kg_train.n_rel)
        criterion = BinaryCrossEntropyLoss()

    elif model_name == 'HolE':
        model = torchkge.models.HolEModel(emb_dim, kg_train.n_ent, kg_train.n_rel)
        criterion = BinaryCrossEntropyLoss()

    elif model_name == 'ComplEx':
        model = torchkge.models.ComplExModel(emb_dim, kg_train.n_ent, kg_train.n_rel)
        criterion = BinaryCrossEntropyLoss()

    elif model_name == 'RESCAL':
        model = torchkge.models.RESCALModel(emb_dim, kg_train.n_ent, kg_train.n_rel)
        criterion = BinaryCrossEntropyLoss()

    elif model_name == "ANALOGY":
        scalar_share = config['model'].get('scalar_share', 0.5)
        if 'scalar_share' not in config['model']:
            warnings.warn(f"The 'scalar_share' field is missing for model {model_name}. Defaulting to 0.5.")
        model = torchkge.models.AnalogyModel(emb_dim, kg_train.n_ent, kg_train.n_rel, scalar_share)
        criterion = BinaryCrossEntropyLoss()
    
    elif model_name == 'ConvKB':
        n_filters = config['model'].get('n_filters', 32)
        if 'n_filters' not in config['model']:
            warnings.warn("The 'n_filters' field is missing in the configuration. Defaulting to 32.")

        model = torchkge.models.ConvKBModel(emb_dim, n_filters, kg_train.n_ent, kg_train.n_rel)
        criterion = BinaryCrossEntropyLoss()

    elif model_name == "TransEModelWithGCN":
        model = TransGNN.TransEModelWithGCN(emb_dim, kg_train.n_ent, kg_train.n_rel, kg_train, device, num_gcn_layers=1)
        criterion = MarginLoss(margin)

    elif model_name == "DistMultModelWithGCN":
        model = DistGNN.DistMultModelWithGCN(emb_dim, kg_train.n_ent, kg_train.n_rel, kg_train, device, num_gcn_layers=1)
        criterion = BinaryCrossEntropyLoss()

    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.to(device)

    return model, criterion

def initialize_sampler(config, kg_train, kg_val, kg_test):
    """Initialize the sampler."""

    sampler_name = config['sampler'].get('name', 'Uniform')
    if 'name' not in config['sampler']:
        warnings.warn("The sampler name field is missing in the configuration. Defaulting to 'Uniform'.")

    n_neg = config['sampler'].get('n_neg', 1)
    if 'n_neg' not in config['sampler'] and sampler_name in ['Uniform', 'Bernouilli', 'Mixed']:
        warnings.warn("The sampler n_neg field is missing in the configuration. Defaulting to 1.")


    if sampler_name == 'Positional':
        sampler = PositionalNegativeSampler(kg_train, kg_val=kg_val, kg_test=kg_test)
    
    elif sampler_name == 'Uniform':
        sampler = torchkge.sampling.UniformNegativeSampler(kg_train, kg_val=kg_val, kg_test=kg_test, n_neg=n_neg)
    
    elif sampler_name == 'Bernoulli':
        sampler = torchkge.sampling.BernoulliNegativeSampler(kg_train, kg_val=kg_val, kg_test=kg_test, n_neg=n_neg)
    
    elif sampler_name == "Mixed":
        sampler = MixedNegativeSampler(kg_train, kg_val=kg_val, kg_test=kg_test, n_neg=n_neg)
    
    else:
        raise ValueError(f"Sampler type '{sampler_name}' is not supported. Please check the configuration.")
    
    return sampler

def initialize_optimizer(model, config):
    """
    Initialize the optimizer based on the configuration provided.
    
    Args:
    - model: PyTorch model whose parameters will be optimized.
    - config: Configuration dictionary containing optimizer settings.
    
    Returns:
    - optimizer: Initialized optimizer.
    """

    optimizer_name = config['optimizer'].get('name', 'Adam')
    if 'name' not in config['optimizer']:
        warnings.warn("The optimizer name field is missing in the configuration. Defaulting to 'Adam'.")

    # Retrieve optimizer parameters, defaulting to an empty dict if not specified
    optimizer_params = config['optimizer'].get('params', {})
    if 'params' not in config['optimizer']:
        warnings.warn("The optimizer params field is missing in the configuration. Defaulting to empty parameters.")
        optimizer_params = {}

    # Mapping of optimizer names to their corresponding PyTorch classes
    optimizer_mapping = {
        'Adam': optim.Adam,
        'SGD': optim.SGD,
        'RMSprop': optim.RMSprop,
        # Add other optimizers as needed
    }

    # Check if the specified optimizer is supported
    if optimizer_name not in optimizer_mapping:
        raise ValueError(f"Optimizer type '{optimizer_name}' is not supported. Please check the configuration.")

    optimizer_class = optimizer_mapping[optimizer_name]
    
    try:
        # Initialize the optimizer avec les paramètres fournis
        optimizer = optimizer_class(model.parameters(), **optimizer_params)
    except TypeError as e:
        raise ValueError(f"Error initializing optimizer '{optimizer_name}': {e}")
    
    logging.info(f"Optimizer '{optimizer_name}' initialized with parameters: {optimizer_params}")
    return optimizer

def initialize_lr_scheduler(optimizer, config):
    """
    Initializes the learning rate scheduler based on the provided configuration.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which the scheduler is used.
        config (dict): Configuration dictionary containing scheduler settings.
                       Should include a 'lr_scheduler' section if a scheduler is to be used.
    
    Returns:
        torch.optim.lr_scheduler._LRScheduler or None: Instance of the specified scheduler or
                                                         None if no scheduler is configured.
    
    Raises:
        ValueError: If the scheduler type is unsupported or required parameters are missing.
    """
    
    # Check if 'lr_scheduler' is present in the configuration
    if 'lr_scheduler' not in config:
        warnings.warn("No learning rate scheduler specified in the configuration.")
        return None
    
    scheduler_config = config['lr_scheduler']
    
    # Check that the scheduler type is specified
    scheduler_type = scheduler_config.get('type', None)
    if scheduler_type is None:
        warnings.warn("The 'type' field is missing in the scheduler configuration. No scheduler will be used.")
        return None
    
    # Retrieve scheduler parameters, defaulting to an empty dict if not provided
    scheduler_params = scheduler_config.get('params', {})
    
    # Mapping of scheduler names to their corresponding PyTorch classes
    scheduler_mapping = {
        'StepLR': lr_scheduler.StepLR,
        'MultiStepLR': lr_scheduler.MultiStepLR,
        'ExponentialLR': lr_scheduler.ExponentialLR,
        'CosineAnnealingLR': lr_scheduler.CosineAnnealingLR,
        'CosineAnnealingWarmRestarts': lr_scheduler.CosineAnnealingWarmRestarts,
        'ReduceLROnPlateau': lr_scheduler.ReduceLROnPlateau,
        'LambdaLR': lr_scheduler.LambdaLR,
        'OneCycleLR': lr_scheduler.OneCycleLR,
        'CyclicLR': lr_scheduler.CyclicLR,
    }
    
    # Verify that the scheduler type is supported
    if scheduler_type not in scheduler_mapping:
        raise ValueError(f"Scheduler type '{scheduler_type}' is not supported. Please check the configuration.")
    
    scheduler_class = scheduler_mapping[scheduler_type]
    
    # Initialize the scheduler based on its type
    try:
            scheduler = scheduler_class(optimizer, **scheduler_params)
    except TypeError as e:
        raise ValueError(f"Error initializing '{scheduler_type}': {e}")

    
    logging.info(f"Scheduler '{scheduler_type}' initialized with parameters: {scheduler_params}")
    return scheduler

def find_best_model(dir):
    try:
        best = max(
            (f for f in os.listdir(dir) if f.startswith('best_model_checkpoint_val_mrr=') and f.endswith('.pt')),
            key=lambda f: float(f.split('val_mrr=')[1].rstrip('.pt')),
            default=None
        )
        return best
    except ValueError:
        return None

def link_pred(model, kg, batch_size):
    """Link prediction evaluation on test set."""
    # Test MRR measure
    evaluator = LinkPredictionEvaluator(model, kg)
    evaluator.evaluate(b_size=batch_size, verbose=True)
    
    test_mrr = evaluator.mrr()[1]
    return test_mrr


def calculate_mrr_for_relations(kg, model, eval_batch_size, relations):
    # MRR calculé avec pondération par le nombre de faits
    mrr_sum = 0.0
    fact_count = 0
    individual_mrrs = {}  # Dictionnaire pour stocker les MRR par relation

    for relation_name in relations:
        # Récupérer l'indice et les faits associés
        relation_index = kg.rel2ix.get(relation_name)
        indices_to_keep = torch.nonzero(kg.relations == relation_index, as_tuple=False).squeeze()

        if indices_to_keep.numel() == 0:
            continue  # Passer aux relations suivantes si aucun fait associé
            
        print(relation_name)
        
        new_kg = kg.keep_triples(indices_to_keep)
        new_kg.dict_of_rels = kg.dict_of_rels
        new_kg.dict_of_heads = kg.dict_of_heads
        new_kg.dict_of_tails = kg.dict_of_tails
        test_mrr = link_pred(model, new_kg, eval_batch_size)
        
        # Enregistrer le MRR pour chaque relation
        individual_mrrs[relation_name] = test_mrr
        
        # Accumuler le MRR global avec pondération
        mrr_sum += test_mrr * indices_to_keep.numel()
        fact_count += indices_to_keep.numel()
    
    # Calcul du MRR global pour le groupe de relations
    group_mrr = mrr_sum / fact_count if fact_count > 0 else 0
    
    # Retourner le MRR total, le nombre de faits, les MRR individuels par relation, et le MRR global pour le groupe
    return mrr_sum, fact_count, individual_mrrs, group_mrr

def categorize_test_nodes(kg_train, kg_test, relation_name, threshold):
    """
    Categorizes test triples with the specified relation in the test set 
    based on whether their entities have been seen with that relation in the training set,
    and separates them into two groups based on a threshold for occurrences.

    Parameters
    ----------
    kg_train : KnowledgeGraph
        The training knowledge graph.
    kg_test : KnowledgeGraph
        The test knowledge graph.
    relation_name : str
        The name of the relation to check (e.g., 'indication').
    threshold : int
        The minimum number of occurrences of the relation for a node to be considered as "frequent".

    Returns
    -------
    frequent_indices : list
        Indices of triples in the test set with the specified relation where entities have been seen more than `threshold` times with that relation in the training set.
    infrequent_indices : list
        Indices of triples in the test set with the specified relation where entities have been seen fewer than or equal to `threshold` times with that relation in the training set.
    """
    # Get the index of the specified relation in the training graph
    if relation_name not in kg_train.rel2ix:
        raise ValueError(f"The relation '{relation_name}' does not exist in the training knowledge graph.")
    relation_idx = kg_train.rel2ix[relation_name]

    # Count occurrences of nodes with the specified relation in the training set
    train_node_counts = {}
    for i in range(kg_train.n_facts):
        if kg_train.relations[i].item() == relation_idx:
            head = kg_train.head_idx[i].item()
            tail = kg_train.tail_idx[i].item()
            train_node_counts[head] = train_node_counts.get(head, 0) + 1
            train_node_counts[tail] = train_node_counts.get(tail, 0) + 1

    # Separate test triples with the specified relation based on the threshold
    frequent_indices = []
    infrequent_indices = []
    for i in range(kg_test.n_facts):
        if kg_test.relations[i].item() == relation_idx:  # Only consider triples with the specified relation
            head = kg_test.head_idx[i].item()
            tail = kg_test.tail_idx[i].item()
            head_count = train_node_counts.get(head, 0)
            tail_count = train_node_counts.get(tail, 0)

            # Categorize based on threshold
            if head_count > threshold or tail_count > threshold:
                frequent_indices.append(i)
            else:
                infrequent_indices.append(i)

    return frequent_indices, infrequent_indices

def calculate_mrr_for_categories(kg_test, model, eval_batch_size, frequent_indices, infrequent_indices):
    """
    Calculate the MRR for frequent and infrequent categories based on given indices.
    
    Parameters
    ----------
    kg_test : KnowledgeGraph
        The test knowledge graph.
    model : Model
        The model used for link prediction.
    eval_batch_size : int
        The evaluation batch size.
    frequent_indices : list
        Indices of test triples considered as frequent.
    infrequent_indices : list
        Indices of test triples considered as infrequent.

    Returns
    -------
    frequent_mrr : float
        MRR for the frequent category.
    infrequent_mrr : float
        MRR for the infrequent category.
    """

    # Créer des sous-graphes pour les catégories fréquentes et infrequentes
    kg_frequent = kg_test.keep_triples(frequent_indices)
    kg_frequent.dict_of_rels = kg_test.dict_of_rels
    kg_frequent.dict_of_heads = kg_test.dict_of_heads
    kg_frequent.dict_of_tails = kg_test.dict_of_tails
    kg_infrequent = kg_test.keep_triples(infrequent_indices)
    kg_infrequent.dict_of_rels = kg_test.dict_of_rels
    kg_infrequent.dict_of_heads = kg_test.dict_of_heads
    kg_infrequent.dict_of_tails = kg_test.dict_of_tails
    
    # Calculer le MRR pour chaque catégorie
    frequent_mrr = link_pred(model, kg_frequent, eval_batch_size) if frequent_indices else 0
    infrequent_mrr = link_pred(model, kg_infrequent, eval_batch_size) if infrequent_indices else 0

    return frequent_mrr, infrequent_mrr

def read_training_metrics(training_metrics_file):
    df = pd.read_csv(training_metrics_file)

    df = df[~df['Epoch'].astype(str).str.contains('CHECKPOINT RESTART')]

    df['Epoch'] = df['Epoch'].astype(int)
    df = df.sort_values(by='Epoch')

    df = df.drop_duplicates(subset=['Epoch'], keep='last')

    return df

def train_model(kg_train, kg_val, kg_test, config):

    run_training = config['common'].get('run_training', True)
    plot_training_metrics = config['common'].get('plot_training_metrics', True)
    run_eval = config['common'].get('run_evaluation', True)
    run_inference = config['common'].get('run_inference', True)

    #################
    # Initialization
    #################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Detected device: {device}')

    # Set random seeds using common_config
    set_random_seeds(config['common']['seed'])

    logging.info('Initializing model...')
    # Initialize model
    model, criterion = initialize_model(config, kg_train, device)
    optimizer = initialize_optimizer(model, config)

    logging.info('Initializing sampler...')
    # Initialize sampler
    sampler = initialize_sampler(config, kg_train, kg_val, kg_test)

    # Initialize training metrics file
    training_metrics_file = os.path.join(config['common']['out'], 'training_metrics.csv')
    train_losses = []
    val_mrrs = []
    learning_rates = []
    resume_checkpoint = config['training'].get('resume_checkpoint', None)

    if not resume_checkpoint and run_training:
        with open(training_metrics_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Training Loss', 'Validation MRR', 'Learning Rate'])  


    max_epochs = config['training'].get('max_epochs', 1000)
    if 'max_epochs' not in config['training']:
        warnings.warn("The training max_epochs field is missing in the configuration. Defaulting to 1000.")

    batch_size = config['training'].get('batch_size', 528)
    if 'batch_size' not in config['training']:
        warnings.warn("The training batch_size field is missing in the configuration. Defaulting to 528.")

    patience = config['training'].get('patience', max_epochs)  # Default patience is max_epochs if not specified
    if 'patience' not in config['training']:
        warnings.warn(f"The training patience field is missing in the configuration. Defaulting to {max_epochs} (i.e. no early stopping).")

    eval_interval = config['training'].get('eval_interval', 1) 
    if 'eval_interval' not in config['training']:
        warnings.warn("The training eval_interval field is missing in the configuration. Defaulting to 1.")

    eval_batch_size = config['training'].get('eval_batch_size', 32)
    if 'eval_batch_size' not in config['training']:
        warnings.warn("The training eval_batch_size field is missing in the configuration. Defaulting to 32.")

    scheduler = initialize_lr_scheduler(optimizer, config)

    use_cuda = 'all' if device.type == 'cuda' else None
    train_iterator = DataLoader(kg_train, batch_size, use_cuda=use_cuda)
    logging.info(f'Number of training batches: {len(train_iterator)}')
    
    def process_batch(engine, batch):
        h, t, r = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        n_h, n_t = sampler.corrupt_batch(h, t, r)
        n_h, n_t = n_h.to(device), n_t.to(device)  

        optimizer.zero_grad()

        # Calcul de la perte avec les triplets positifs et négatifs
        pos, neg = model(h, t, r, n_h, n_t)
        loss = criterion(pos, neg)
        loss.backward()
        
        # Mise à jour des paramètres de l'optimizer
        optimizer.step()

        # Normalisation des paramètres du modèle
        model.normalize_parameters()

        return loss.item()


    trainer = Engine(process_batch)
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss_ra')


    #################
    # Handlers
    #################
    if config["model"]['name'] in ["TransEModelWithGCN", "DistMultModelWithGCN"]:
        total_batches = len(train_iterator)
        @trainer.on(Events.ITERATION_COMPLETED(every=20))
        def log_batch_progress(engine):
            current_batch = engine.state.iteration
            logging.info(f'Processed {current_batch} batches out of {total_batches}')

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_metrics_to_csv(engine):
        epoch = engine.state.epoch
        train_loss = engine.state.metrics['loss_ra']
        val_mrr = engine.state.metrics.get('val_mrr', 0)
        lr = optimizer.param_groups[0]['lr']

        train_losses.append(train_loss)
        val_mrrs.append(val_mrr)
        learning_rates.append(lr)

        
        with open(training_metrics_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_loss, val_mrr, lr])

        logging.info(f"Epoch {epoch} - Train Loss: {train_loss}, Validation MRR: {val_mrr}, Learning Rate: {lr}")


    ##### Clean memory
    @trainer.on(Events.EPOCH_COMPLETED)
    def clean_memory(engine):
        torch.cuda.empty_cache()
        gc.collect()
        logging.info("Memory cleaned.")

    ##### Evaluate on validation set
    @trainer.on(Events.EPOCH_COMPLETED(every=eval_interval))
    def evaluate(engine):
        logging.info(f"Evaluating on validation set at epoch {engine.state.epoch}...")
        model.eval()  # Met le modèle en mode évaluation
        with torch.no_grad():
            val_mrr = link_pred(model, kg_val, eval_batch_size) 
        engine.state.metrics['val_mrr'] = val_mrr 
        logging.info(f"Validation MRR: {val_mrr}")

        if scheduler and isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_mrr)
            logging.info('Stepping scheduler ReduceLROnPlateau.')

        model.train()  # Remet le modèle en mode entraînement

    ##### Scheduler update
    @trainer.on(Events.EPOCH_COMPLETED)
    def update_scheduler(engine):
        if scheduler and not isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

    ##### Early stopping
    def score_function(engine):
        val_mrr = engine.state.metrics.get('val_mrr', 0)
        return val_mrr
    
    # EarlyStopping handler
    early_stopping = EarlyStopping(
        patience=patience, 
        score_function=score_function, 
        trainer=trainer, 
    )

    # Attach handler to trainer
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=config['training']['eval_interval']), 
        early_stopping
    )

    # def print_gpu_memory(message=""):
    #     allocated = torch.cuda.memory_allocated() / 1024**3
    #     reserved = torch.cuda.memory_reserved() / 1024**3
    #     logging.info(f"{message} - Memory Allocated: {allocated:.2f} GB, Memory Reserved: {reserved:.2f} GB")

    # @trainer.on(Events.EPOCH_COMPLETED)
    # def log_gpu_memory(engine):
    #     print_gpu_memory("After epoch")

    ##### Checkpoint periodic
    if scheduler:
        to_save = {
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'trainer': trainer
        }
    else:
        to_save = {
            'model': model,
            'optimizer': optimizer,
            'trainer': trainer
        }

    checkpoint_dir = os.path.join(config['common']['out'], 'checkpoints')
    
    # Create the checkpoint handler
    checkpoint_handler = Checkpoint(
        to_save,                        # Dictionnaire des objets à sauvegarder
        DiskSaver(dirname=checkpoint_dir, require_empty=False, create_dir=True),  # Gestionnaire de sauvegarde
        n_saved=2,                      # Garder les 2 derniers checkpoints
        global_step_transform=lambda *_: trainer.state.epoch,      # Inclure le numéro d'époque
    )
        
    # Custom save function to move the model to CPU before saving and back to GPU after
    def save_checkpoint_to_cpu(engine):
        # Move model to CPU before saving
        model.to('cpu')

        # Save the checkpoint
        checkpoint_handler(engine)

        # Move model back to GPU
        model.to(device)

    # Attach checkpoint handler to trainer and call save_checkpoint_to_cpu
    trainer.add_event_handler(Events.EPOCH_COMPLETED, save_checkpoint_to_cpu)
    

    # checkpoint_periodic_handler = Checkpoint(
    #     to_save,                        # Dictionnaire des objets à sauvegarder
    #     DiskSaver(dirname=checkpoint_dir, require_empty=False, create_dir=True),  # Gestionnaire de sauvegarde
    #     n_saved=2,                      # Garder les 2 derniers checkpoints
    #     global_step_transform=lambda *_: trainer.state.epoch,      # Inclure le numéro d'époque
    # )

    # # Attach checkpoint handler to trainer
    # trainer.add_event_handler(
    #     Events.EPOCH_COMPLETED, 
    #     checkpoint_periodic_handler)

    ##### Checkpoint best MRR
    def get_val_mrr(engine):
        return engine.state.metrics.get('val_mrr', 0)

    checkpoint_best_handler = ModelCheckpoint(
        dirname=checkpoint_dir,                                     # Répertoire de sauvegarde
        filename_prefix='best_model',                       # Préfixe du nom de fichier
        n_saved=1,                                                 # Garder seulement le meilleur modèle
        score_function=get_val_mrr,                                # Fonction de score basée sur val_mrr
        score_name='val_mrr',                                      # Nom de la métrique utilisée
        require_empty=False,                                       # Le répertoire n'a pas besoin d'être vide
        create_dir=True,                                           # Créer le répertoire s'il n'existe pas
        atomic=True
    )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=config['training']['eval_interval']),
        checkpoint_best_handler,
        to_save
    )

    ##### Late stopping    
    @trainer.on(Events.COMPLETED)
    def on_training_completed(engine):
        logging.info(f"Training completed after {engine.state.epoch} epochs.")

    #################
    # Checkpoint restart 
    #################
    if run_training and resume_checkpoint:
        if os.path.isfile(resume_checkpoint):
            logging.info(f"Resuming from checkpoint: {resume_checkpoint}")
            checkpoint = torch.load(resume_checkpoint)
            # logging.info(f'keys: {checkpoint.keys()}') 
            Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)
            logging.info("Checkpoint loaded successfully.")

            model.to(device)

            with open(training_metrics_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['CHECKPOINT RESTART', 'CHECKPOINT RESTART', 'CHECKPOINT RESTART', 'CHECKPOINT RESTART'])

            if trainer.state.epoch < config['training']['max_epochs']:
                logging.info(f"Starting from epoch {trainer.state.epoch}")
                trainer.run(train_iterator)
            else:
                logging.info(f"Training already completed. Last epoch is {trainer.state.epoch} and max_epochs set to {config['training']['max_epochs']}")

    #################
    # Train
    #################
        else:
            logging.info(f"Checkpoint file {resume_checkpoint} does not exist. Starting training from scratch.")
            trainer.run(train_iterator, max_epochs=config['training']['max_epochs'])
    else:
        if run_training:
            trainer.run(train_iterator, max_epochs=config['training']['max_epochs'])


    #################
    # Report metrics
    #################
    if plot_training_metrics:
        plot_learning_curves(training_metrics_file, config)

    #################
    # Evaluation on test set
    #################

    if run_eval or run_inference:
        model.to("cpu")
        del model
        model = None
        torch.cuda.empty_cache()    
        gc.collect()

        # Charger le meilleur modèle
        new_model, _ = initialize_model(config, kg_train, device)
        logging.info("Loading best model.")
        best_model = find_best_model(checkpoint_dir)
        logging.info(f"Best model is {os.path.join(checkpoint_dir, best_model)}")
        checkpoint = torch.load(os.path.join(checkpoint_dir, best_model), map_location=device)
        new_model.load_state_dict(checkpoint["model"])
        logging.info("Best model successfully loaded.")
        logging.info("Evaluating on the test set with best model...")

        new_model.eval()

        if run_eval:
            list_rel_1 = config.get('evaluation', {}).get('made_directed_relations', [])
            list_rel_2 = config.get('evaluation', {}).get('target_relations', [])
            thresholds = config.get('evaluation', {}).get('thresholds', [])
            mrr_file = os.path.join(config['common']['out'], 'evaluation_metrics.yaml')

            all_relations = set(kg_test.rel2ix.keys())
            remaining_relations = all_relations - set(list_rel_1) - set(list_rel_2)
            remaining_relations = list(remaining_relations)

            total_mrr_sum_list_1, fact_count_list_1, individual_mrrs_list_1, group_mrr_list_1 = calculate_mrr_for_relations(
                kg_test, new_model, eval_batch_size, list_rel_1)
            total_mrr_sum_list_2, fact_count_list_2, individual_mrrs_list_2, group_mrr_list_2 = calculate_mrr_for_relations(
                kg_test, new_model, eval_batch_size, list_rel_2)
            total_mrr_sum_remaining, fact_count_remaining, individual_mrrs_remaining, group_mrr_remaining = calculate_mrr_for_relations(
                kg_test, new_model, eval_batch_size, remaining_relations)

            global_mrr = (total_mrr_sum_list_1 + total_mrr_sum_list_2 + total_mrr_sum_remaining) / (fact_count_list_1 + fact_count_list_2 + fact_count_remaining)

            logging.info(f"Final Test MRR with best model: {global_mrr}")

            results = {
                "Global_MRR": global_mrr,
                "made_directed_relations": {
                    "Global_MRR": group_mrr_list_1,
                    "Individual_MRRs": individual_mrrs_list_1
                },
                "target_relations": {
                    "Global_MRR": group_mrr_list_2,
                    "Individual_MRRs": individual_mrrs_list_2
                },
                "remaining_relations": {
                    "Global_MRR": group_mrr_remaining,
                    "Individual_MRRs": individual_mrrs_remaining
                },
                "target_relations_by_frequency": {}  
            }


            for i in range(len(list_rel_2)):
                relation = list_rel_2[i]
                threshold = thresholds[i]
                frequent_indices, infrequent_indices = categorize_test_nodes(kg_train, kg_test, relation, threshold)
                frequent_mrr, infrequent_mrr = calculate_mrr_for_categories(kg_test, new_model, eval_batch_size, frequent_indices, infrequent_indices)
                logging.info(f"MRR for frequent nodes (threshold={threshold}) in relation {relation}: {frequent_mrr}")
                logging.info(f"MRR for infrequent nodes (threshold={threshold}) in relation {relation}: {infrequent_mrr}")

                results["target_relations_by_frequency"][relation] = {
                    "Frequent_MRR": frequent_mrr,
                    "Infrequent_MRR": infrequent_mrr,
                    "Threshold": threshold
                }
                
            
            with open(mrr_file, "w") as file:
                yaml.dump(results, file, default_flow_style=False, sort_keys=False)

            logging.info(f"Evaluation results stored in {mrr_file}")
        
        if run_inference:
            inference_mrr_file = os.path.join(config['common']['out'], 'inference_metrics.yaml')

            orpha_df= pd.read_csv(run_inference, sep="\t")
            orpha_kg = KnowledgeGraph(df = orpha_df, ent2ix=kg_train.ent2ix, rel2ix=kg_train.rel2ix) 
            
            evaluator = LinkPredictionEvaluator(new_model, orpha_kg)
            evaluator.evaluate(b_size=eval_batch_size, verbose=True)
                
            inference_mrr = evaluator.mrr()[1]
            inference_hit10 = evaluator.hit_at_k(10)[1]

            results = {"Inference MRR": inference_mrr, "Inference hit@10:": inference_hit10}

            logging.info(f"MRR on inference set: {inference_mrr}")

            with open(inference_mrr_file, "w") as file:
                yaml.dump(results, file, default_flow_style=False, sort_keys=False)


            logging.info(f"Evaluation results stored in {inference_mrr_file}")


            


        

    # run_eval_by_degree = config.get('evaluation_by_degree', {})
    # if run_eval and run_eval_by_degree:
    #     relation_interest = config["evaluation_by_degree"]["relation"]
    #     threshold = config["evaluation_by_degree"].get('threshold', 1)
    #     mrr_file_deg = os.path.join(config['common']['out'], f'evaluation_metrics_{relation_interest}_threshold_{threshold}.yaml')
    #     calculate_mrr_by_degree(model, kg_train, kg_test, relation_interest, threshold, eval_batch_size, mrr_file_deg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file.')
    args = parser.parse_args()
    main(args)
