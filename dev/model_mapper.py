import logging
import warnings

from torch import tensor, long, stack
from torch.nn.functional import normalize

import torchkge
from torchkge.utils import MarginLoss, BinaryCrossEntropyLoss
from torchkge.models import Model

import encoders
import decoders

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  

class ModelMapper(Model):
    def __init__(self, config, kg_train, device):
        emb_dim = config['model'].get('emb_dim', 100)  
        if 'emb_dim' not in config['model']:
            warnings.warn("The model emb_dim field is missing in the configuration. Defaulting to 100.")

        super.__init__(emb_dim, kg_train.n_ent, kg_train.n_rel)
        # Decoder
        translational_models = ['TransE', 'TransH', 'TransR', 'TransD', 'TorusE', 'TransEModelWithGCN']

        decoder_name = config['model'].get('decoder', 'TransE')
        if 'decoder' not in config['model']:
            warnings.warn("The model name field is missing in the configuration. Defaulting to 'TransE'.")

        rel_emb_dim = config['model'].get('rel_emb_dim', emb_dim)
        if decoder_name in ['TransR', 'TransD'] and 'rel_emb_dim' not in config['model']:
            warnings.warn(f"The 'rel_emb_dim' field is missing for model {decoder_name}. Defaulting to {emb_dim}.")

        dissimilarity = config['model'].get('dissimilarity', 'L2')
        if decoder_name == 'TransE' and 'dissimilarity' not in config['model']:
            warnings.warn(f"The 'dissimilarity' field is missing for model {decoder_name}. Defaulting to 'L2'.")
        if decoder_name == 'TorusE' and 'dissimilarity' not in config['model']:
            dissimilarity = 'torus_L2'
            warnings.warn(f"The 'dissimilarity' field is missing for model {decoder_name}. Defaulting to 'torus_L2'.")

        margin = config['model'].get('margin', 1.0) 
        if 'margin' not in config['model'] and decoder_name in translational_models:
            warnings.warn("The model margin field is missing in the configuration. Defaulting to 1.0.")

        match decoder_name:
            # Translational models
            case 'TransE':
                self.decoder = decoders.TransE(emb_dim, self.n_ent, self.n_rel,
                                                    dissimilarity_type=dissimilarity)
                self.criterion = MarginLoss(margin)

            case 'TransH':
                self.decoder = decoders.TransH(emb_dim, self.n_ent, self.n_rel)
                self.criterion = MarginLoss(margin)

            case 'TransR':
                self.decoder = torchkge.models.TransRModel(emb_dim, rel_emb_dim, self.n_ent, self.n_rel)
                self.criterion = MarginLoss(margin)

            case 'TransD':
                self.decoder = torchkge.models.TransDModel(emb_dim, rel_emb_dim, self.n_ent, self.n_rel)
                self.criterion = MarginLoss(margin)

            case 'TorusE':
                self.decoder = torchkge.models.TorusEModel(emb_dim, self.n_ent, self.n_rel,
                                                    dissimilarity_type=dissimilarity)
                self.criterion = MarginLoss(margin)

            # Bilinear models
            case 'DistMult':
                self.decoder = torchkge.models.DistMultModel(emb_dim, self.n_ent, self.n_rel)
                self.criterion = BinaryCrossEntropyLoss()

            case 'HolE':
                self.decoder = torchkge.models.HolEModel(emb_dim, self.n_ent, self.n_rel)
                self.criterion = BinaryCrossEntropyLoss()

            case 'ComplEx':
                self.decoder = torchkge.models.ComplExModel(emb_dim, self.n_ent, self.n_rel)
                self.criterion = BinaryCrossEntropyLoss()

            case 'RESCAL':
                self.decoder = torchkge.models.RESCALModel(emb_dim, self.n_ent, self.n_rel)
                self.criterion = BinaryCrossEntropyLoss()

            case "ANALOGY":
                scalar_share = config['model'].get('scalar_share', 0.5)
                if 'scalar_share' not in config['model']:
                    warnings.warn(f"The 'scalar_share' field is missing for model {decoder_name}. Defaulting to 0.5.")
                self.decoder = torchkge.models.AnalogyModel(emb_dim, self.n_ent, self.n_rel, scalar_share)
                self.criterion = BinaryCrossEntropyLoss()
            
            case 'ConvKB':
                n_filters = config['model'].get('n_filters', 32)
                if 'n_filters' not in config['model']:
                    warnings.warn("The 'n_filters' field is missing in the configuration. Defaulting to 32.")

                self.decoder = torchkge.models.ConvKBModel(emb_dim, n_filters, self.n_ent, self.n_rel)
                self.criterion = BinaryCrossEntropyLoss()

            case _:
                raise ValueError(f"Unknown decoder model: {decoder_name}")

        self.decoder.to(device)

        # Encoder
        encoder_name = config['model'].get('encoder', 'default')
        if 'encoder' not in config['model']:
            warnings.warn("The model name field is missing in the configuration. Defaulting to 'TransE'.")

        mapping_csv = config["common"]["mapping_csv"]

        match encoder_name:
            case "default":
                self.encoder = encoders.DefaultEncoder(self.n_ent, self.n_rel, emb_dim)
            case "GAT":
                self.encoder = encoders.GATEncoder(emb_dim, self.n_ent, self.n_rel, kg_train, device, mapping_csv)
            case "GCN":
                self.encoder = encoders.GCNEncoder(emb_dim, self.n_ent, self.n_rel, kg_train, device, mapping_csv)
            case _:
                raise ValueError(f"Unknown encoder model: {encoder_name}")
            
        self.rel_emb.weight.data = normalize(self.rel_emb.weight.data, p=2, dim=1)


        def scoring_function(self, h_idx, t_idx, r_idx):
            encoder_output = None
            if self.encoder.deep:
                encoder_output = self.encoder.forward()

            # 1. Déterminer les types de nœuds pour les heads et tails à partir de kg_to_node_type
            h_node_types = [self.kg2nodetype[h.item()] for h in h_idx]
            t_node_types = [self.kg2nodetype[t.item()] for t in t_idx]
        
            if encoder_output is not None:
                try:
                    h_het_idx = tensor([
                        self.kg2het[h_type][h.item()] for h, h_type in zip(h_idx, h_node_types)
                    ], dtype=long, device=h_idx.device)
                    t_het_idx = tensor([
                        self.kg2het[t_type][t.item()] for t, t_type in zip(t_idx, t_node_types)
                    ], dtype=long, device=t_idx.device)
                except KeyError as e:
                    logger.error(f"Erreur de mapping pour le nœud ID: {e}")
                    raise
            
                h_embeddings = stack([
                    encoder_output[h_type][h_idx_item] for h_type, h_idx_item in zip(h_node_types, h_het_idx)
                ])
                t_embeddings = stack([
                    encoder_output[t_type][t_idx_item] for t_type, t_idx_item in zip(t_node_types, t_het_idx)
                ])
            else:
                h_embeddings = self.ent_emb(h_idx)
                t_embeddings = self.ent_emb(t_idx)

            r_embeddings = self.rel_emb(r_idx)  # Les relations restent inchangées

            # 4. Normaliser les embeddings des entités (heads et tails)
            h_normalized = normalize(h_embeddings, p=2, dim=1)
            t_normalized = normalize(t_embeddings, p=2, dim=1)

            return self.decoder.score(h_normalized, r_embeddings, t_normalized)
        
    def normalize_parameters(self):
        """
        Normalize the entity embeddings for each node type and the relation embeddings.
        This method should be called at the end of each training epoch and at
        the end of training as well.
        """
        # Some decoders should not normalize parameters or do so in a different way.
        # In this case, they should implement the function themselves and we return it.
        normalize_func = getattr(self.decoder, "normalize_parameters", None)
        if callable(normalize_func):
            return normalize_func(self)
        
        # Normaliser les embeddings des entités pour chaque type de nœud
        for node_type, embedding in self.node_embeddings.items():
            normalized_embedding = normalize(embedding.weight.data, p=2, dim=1)
            embedding.weight.data = normalized_embedding
            logger.debug(f"Normalized embeddings for node type '{node_type}'")

        # Normaliser les embeddings des relations
        self.rel_emb.weight.data = normalize(self.rel_emb.weight.data, p=2, dim=1)
        logger.debug("Normalized relation embeddings")

    def get_embeddings(self):
        """
        Return the embeddings of entities (for each node type) and relations.

        Returns
        -------
        ent_emb: dict of torch.Tensor
            Dictionary where keys are node types and values are entity embeddings tensors.
        rel_emb: torch.Tensor, shape: (n_rel, emb_dim), dtype: torch.float
            Embeddings of relations.
        """
        self.normalize_parameters()
        
        # Récupérer les embeddings des entités pour chaque type de nœud
        if self.encoder.deep:
            ent_emb = {node_type: embedding.weight.data for node_type, embedding in self.node_embeddings.items()}
        else:
            ent_emb = self.ent_emb.weight.data
        
        # Récupérer les embeddings des relations
        rel_emb = self.rel_emb.weight.data
        
        logger.debug("Retrieved embeddings for entities and relations")
        
        return ent_emb, rel_emb