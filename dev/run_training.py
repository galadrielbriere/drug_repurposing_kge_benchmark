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
import gc
import json
import csv 
import matplotlib.pyplot as plt
import TransGNN
import DistGNN


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


    if config["common"]['run_kg_prep']:
            logging.info(f"Preparing KG...")
            kg_train, kg_val, kg_test = prepare_knowledge_graph(config)
    else:
        logging.info("Loading KG...")
        kg_train, kg_val, kg_test = load_knowledge_graph(config)
        logging.info("Done")

    # kg_train, kg_val, kg_test = load_fb15k()

    if config['common']['run_training']:
        train_model(kg_train, kg_val, kg_test, config)

def plot_learning_curves(train_losses, val_mrrs):
    epochs = range(1, len(train_losses) + 1)
    
    # Courbe de la perte d'entraînement
    plt.figure()
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.show()

    # Courbe du MRR sur validation
    plt.figure()
    plt.plot(epochs, val_mrrs, label='Validation MRR')
    plt.xlabel('Epochs')
    plt.ylabel('MRR')
    plt.title('Validation MRR Over Time')
    plt.legend()
    plt.show()

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
    model.normalize_parameters()

    # Test MRR measure
    evaluator = LinkPredictionEvaluator(model, kg)
    evaluator.evaluate(b_size=batch_size, verbose=True)
    
    test_mrr = evaluator.mrr()[1]
    return test_mrr

def train_model(kg_train, kg_val, kg_test, config):

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

    if not resume_checkpoint:
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
        
        # Mise à jour des paramètres
        optimizer.step()

        return loss.item()


    trainer = Engine(process_batch)
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss_ra')


    #################
    # Handlers
    #################
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
        with torch.no_grad():
            val_mrr = link_pred(model, kg_val, eval_batch_size) 
        engine.state.metrics['val_mrr'] = val_mrr 
        logging.info(f"Validation MRR: {val_mrr}")

        if scheduler and isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_mrr)
            logging.info('Stepping scheduler ReduceLROnPlateau.')

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
    checkpoint_periodic_handler = Checkpoint(
        to_save,                        # Dictionnaire des objets à sauvegarder
        DiskSaver(dirname=checkpoint_dir, require_empty=False, create_dir=True),  # Gestionnaire de sauvegarde
        n_saved=2,                      # Garder les 2 derniers checkpoints
        global_step_transform=lambda *_: trainer.state.epoch,      # Inclure le numéro d'époque
    )

    # Attach checkpoint handler to trainer
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, 
        checkpoint_periodic_handler)

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
    if resume_checkpoint:
        if os.path.isfile(resume_checkpoint):
            logging.info(f"Resuming from checkpoint: {resume_checkpoint}")
            checkpoint = torch.load(resume_checkpoint)
            # logging.info(f'keys: {checkpoint.keys()}') 
            Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)
            logging.info("Checkpoint loaded successfully.")

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
        trainer.run(train_iterator, max_epochs=config['training']['max_epochs'])


    #################
    # Report metrics
    #################
    plot_learning_curves(train_losses, val_mrrs)
    plt.savefig(os.path.join(config['common']['out'], 'training_loss_curve.png'))
    plt.savefig(os.path.join(config['common']['out'], 'validation_mrr_curve.png'))

    #################
    # Evaluation on test set
    #################

    # TEST SUR LE DERNIER MODELE
    logging.info("Evaluating on the test set with last model...")
    test_mrr = link_pred(model, kg_test, eval_batch_size)
    logging.info(f"Final Test MRR with last model: {test_mrr}")

    # def print_gpu_memory(message=""):
    #     allocated = torch.cuda.memory_allocated() / 1024**3
    #     reserved = torch.cuda.memory_reserved() / 1024**3
    #     print(f"{message} - Memory Allocated: {allocated:.2f} GB, Memory Reserved: {reserved:.2f} GB")

    # print_gpu_memory("Before clearing variables")
  
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
    checkpoint = torch.load(os.path.join(checkpoint_dir, best_model))
    # print(checkpoint.keys()) 
    new_model.load_state_dict(checkpoint["model"])
    logging.info("Best model successfully loaded.")
    logging.info("Evaluating on the test set with best model...")
    test_mrr = link_pred(new_model, kg_test, eval_batch_size)
    logging.info(f"Final Test MRR with best model: {test_mrr}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file.')
    args = parser.parse_args()
    main(args)
