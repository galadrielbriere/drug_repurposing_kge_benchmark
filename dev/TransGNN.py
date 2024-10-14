import torch
import torch.nn as nn
from torch.nn.functional import normalize
from torch_geometric.nn import HeteroConv, SAGEConv
from torchkge.models import TranslationModel, TransEModel
from torch_geometric.data import HeteroData
import pandas as pd 
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  
logger.info("LOADED PACK")

def my_init_embedding(num_embeddings, emb_dim):
    """Initialise une couche d'embedding avec une distribution normale."""
    embedding = nn.Embedding(num_embeddings, emb_dim)
    nn.init.xavier_uniform_(embedding.weight.data)
    return embedding

def extract_node_type(node_name):
    """Extracts the node type from the node name, based on the string before the first underscore."""
    return node_name.split('_')[0]

def create_hetero_data(kg):
    df = kg.get_df()
    
    data = HeteroData()
    
    # Dictionnaires pour stocker les correspondances d'ID
    df_to_hetero_mapping = {}
    hetero_to_df_mapping = {}
    
    df_to_kg_mapping = {}  # Mapping entre les IDs du dataframe et les IDs du KnowledgeGraph
    kg_to_df_mapping = {}  # Mapping inverse
    
    kg_to_hetero_mapping = {}
    hetero_to_kg_mapping = {}

    kg_to_node_type = {}  # Mapping pour chaque ID du KG vers son type de nœud

    # 1. Parser les types de nœuds et les identifiants
    df['from_type'] = df['from'].apply(extract_node_type)
    df['to_type'] = df['to'].apply(extract_node_type)

    # 2. Identifier tous les types de nœuds uniques
    node_types = pd.unique(df[['from_type', 'to_type']].values.ravel('K'))
    logger.info(f"Node types identified: {node_types}")

    # 3. Créer des mappings pour les identifiants de nœuds par type
    node_dict = {}
    for ntype in node_types:
        # Extraire tous les identifiants uniques pour ce type
        nodes = pd.concat([
            df[df['from_type'] == ntype]['from'],
            df[df['to_type'] == ntype]['to']
        ]).unique()

        # Créer un mapping de l'ID du dataframe vers un index entier (ID du HeteroData)
        node_dict[ntype] = {node: i for i, node in enumerate(nodes)}   

        # Créer les correspondances pour ce type de nœud (DataFrame -> HeteroData)
        df_to_hetero_mapping[ntype] = node_dict[ntype]  # Mapping DataFrame -> HeteroData
        hetero_to_df_mapping[ntype] = {v: k for k, v in node_dict[ntype].items()}  # Mapping HeteroData -> DataFrame
        
        # Correspondances entre DataFrame et KnowledgeGraph (utiliser kg_train.ent2ix)
        df_to_kg_mapping[ntype] = {node: kg.ent2ix[node] for node in nodes}  # DataFrame -> KG
        kg_to_df_mapping[ntype] = {v: k for k, v in df_to_kg_mapping[ntype].items()}  # KG -> DataFrame
        
        # Mapping KG -> HeteroData via DataFrame
        kg_to_hetero_mapping[ntype] = {df_to_kg_mapping[ntype][k]: df_to_hetero_mapping[ntype][k] for k in node_dict[ntype].keys()}
        hetero_to_kg_mapping[ntype] = {v: k for k, v in kg_to_hetero_mapping[ntype].items()}  # Inverse (HeteroData -> KG)

        # Ajouter les types de nœuds associés à chaque ID du KG
        for kg_id in df_to_kg_mapping[ntype].values():
            kg_to_node_type[kg_id] = ntype

        # Définir le nombre de nœuds pour ce type dans HeteroData
        data[ntype].num_nodes = len(node_dict[ntype])
        logger.info(f"Initialized node type '{ntype}' with {len(nodes)} nodes.")

    # 4. Construire les edge_index pour chaque type de relation
    for rel, group in df.groupby('rel'):
        # Identifier les types de nœuds sources et cibles dans ce groupe
        src_types = group['from_type'].unique()
        tgt_types = group['to_type'].unique()
        
        for src_type in src_types:
            for tgt_type in tgt_types:
                subset = group[
                    (group['from_type'] == src_type) &
                    (group['to_type'] == tgt_type)
                ]
                
                if subset.empty:
                    continue  # Passer si aucun lien dans ce sous-groupe
                
                # Mapper les identifiants de nœuds aux indices entiers dans HeteroData
                src = subset['from'].map(node_dict[src_type]).values
                tgt = subset['to'].map(node_dict[tgt_type]).values
                
                # Créer le tensor edge_index
                edge_index = torch.tensor(np.array([src, tgt]), dtype=torch.long)

                edge_type = (src_type, rel, tgt_type)
                data[(src_type, rel, tgt_type)].edge_index = edge_index
                logger.info(f"Added edge type '{edge_type}' with {len(src)} edges.")
    
    # Retourner l'objet HeteroData, les mappings et le mapping des types de nœuds
    return data, kg_to_hetero_mapping, hetero_to_kg_mapping, df_to_kg_mapping, kg_to_node_type



class TransEModelWithGCN(TranslationModel):
    def __init__(self, emb_dim, n_entities, n_relations, kg, device, num_gcn_layers=2, aggr='sum', dissimilarity_type='L2'):
        """
        Initialise le modèle TransE modifié avec un Heterogeneous GCN pour les embeddings des entités.

        Parameters
        ----------
        emb_dim : int
            Dimension des embeddings.
        n_entities : int
            Nombre total d'entités.
        n_relations : int
            Nombre total de relations.
        kg : KnowledgeGraph
            Objet contenant les informations du graphe de connaissances.
        num_gcn_layers : int, optional
            Nombre de couches de convolution GCN pour chaque type d'arête, par défaut 2.
        aggr : str, optional
            Type d'agrégation ('sum', 'mean', 'max', 'cat'), par défaut 'sum'.
        dissimilarity_type : str, optional
            Type de dissimilarité ('L1' ou 'L2'), par défaut 'L2'.
        """
        logger.info("MyClass instanciée")
        logger.info(f"n_entities = {n_entities}")
        logger.info(f"n_relations = {n_relations}")
        logger.info(f"dissimilarity_type = {dissimilarity_type}")

        super().__init__(n_entities, n_relations, dissimilarity_type) ###
        logger.info("__init_class super")

        self.emb_dim = emb_dim
        self.hetero_data, self.kg2het, self.het2kg, _, self.kg2nodetype = create_hetero_data(kg)
        self.hetero_data = self.hetero_data.to(device)

        # Initialisation des embeddings des relations
        logger.info('before init')
        logger.info(f"self n_rel = {self.n_rel}")
        logger.info(f"self emb dim = {self.emb_dim}")
        self.rel_emb = my_init_embedding(self.n_rel, self.emb_dim)
        logger.info('after init')

        logger.info(f"self.hetero_data.node_types={self.hetero_data.node_types}")
        # Initialisation des embeddings initiaux pour chaque type de nœud
        self.node_embeddings = nn.ModuleDict()
        for node_type in self.hetero_data.node_types:
            num_nodes = self.hetero_data[node_type].num_nodes
            self.node_embeddings[node_type] = my_init_embedding(num_nodes, self.emb_dim)

        # Définir l'agrégation pour HeteroConv
        self.aggr = aggr
        logger.info("set agrr = OK")

        self.convs = nn.ModuleList()
        # Définition des couches GCN multiples pour chaque type d'arête
        for layer in range(num_gcn_layers):
                    conv = HeteroConv(
                        {edge_type: SAGEConv(self.emb_dim, self.emb_dim, aggr="mean") for edge_type in self.hetero_data.edge_types},
                        aggr=self.aggr
                    )
                    self.convs.append(conv)
                    logger.info(f"Initialized HeteroConv layer {layer+1} with {len(conv.convs)} edge types.")

        logger.info("Def convs = OK")

     
        # self.hetero_conv = HeteroConv(self.convs, aggr=self.aggr)
        # logger.info("set HeteroConv = OK")

        # Normalisation initiale des embeddings des relations
        self.rel_emb.weight.data = normalize(self.rel_emb.weight.data, p=2, dim=1)
  
    def forward_gnn(self):
        """
        Passe les embeddings des nœuds à travers les couches GNN.
        """
        x_dict = {node_type: embedding.weight for node_type, embedding in self.node_embeddings.items()}
        logger.info("Starting forward_gnn")

        # Passer à travers les couches HeteroConv
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, self.hetero_data.edge_index_dict)
            # logger.info(f"Completed HeteroConv layer {i+1}/{len(self.convs)}")

        # logger.info("Completed forward_gnn")
        # logger.info(f'x_dict = {x_dict}')
        return x_dict
    
    def scoring_function(self, h_idx, t_idx, r_idx):
        """
        Compute the scoring function for the triplets given as argument:
        :math:`||h + r - t||_p^p` with p being the `dissimilarity type (either 1 or 2)`.
        Instead of using self.ent_emb, we use the embeddings computed by the GNN.

        Parameters
        ----------
        h_idx : torch.Tensor
            The indices of the head entities (from KG).
        t_idx : torch.Tensor
            The indices of the tail entities (from KG).
        r_idx : torch.Tensor
            The indices of the relations (from KG).

        Returns
        -------
        torch.Tensor
            The computed score for the given triplets.
        """
        # logger.info(f'h_idx={h_idx}')
        
        # Récupérer les embeddings du GNN pour les heads, tails et relations
        gnn_output = self.forward_gnn()
        logger.info(f'GNN FORWARD OK')
        # 1. Déterminer les types de nœuds pour les heads et tails à partir de kg_to_node_type
        h_node_types = [self.kg2nodetype[h.item()] for h in h_idx]
        t_node_types = [self.kg2nodetype[t.item()] for t in t_idx]
        # logger.info(f'h_node_types={h_node_types}')
        # logger.info(f't_node_types={t_node_types}')

        # 2. Mapper les indices du KG vers HeteroData pour les heads et les tails
        try:
            h_het_idx = torch.tensor([
                self.kg2het[h_type][h.item()] for h, h_type in zip(h_idx, h_node_types)
            ], dtype=torch.long, device=h_idx.device)
            t_het_idx = torch.tensor([
                self.kg2het[t_type][t.item()] for t, t_type in zip(t_idx, t_node_types)
            ], dtype=torch.long, device=t_idx.device)
        except KeyError as e:
            logger.error(f"Erreur de mapping pour le nœud ID: {e}")
            raise

        # logger.info(f'h_het_idx={h_het_idx}')
        # logger.info(f't_het_idx={t_het_idx}')

        # 3. Utiliser les indices remappés pour récupérer les embeddings depuis gnn_output
        h_embeddings = torch.stack([
            gnn_output[h_type][h_idx_item] for h_type, h_idx_item in zip(h_node_types, h_het_idx)
        ])
        t_embeddings = torch.stack([
            gnn_output[t_type][t_idx_item] for t_type, t_idx_item in zip(t_node_types, t_het_idx)
        ])
        r_embeddings = self.rel_emb(r_idx)  # Les relations restent inchangées

        # 4. Normaliser les embeddings des entités (heads et tails)
        h_normalized = normalize(h_embeddings, p=2, dim=1)
        t_normalized = normalize(t_embeddings, p=2, dim=1)

        # 5. Appliquer la dissimilarité sur les triplets (h, r, t)
        return -self.dissimilarity(h_normalized + r_embeddings, t_normalized)


    def normalize_parameters(self):
        """
        Normalize the entity embeddings for each node type and the relation embeddings.
        This method should be called at the end of each training epoch and at
        the end of training as well.
        """
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
        ent_emb = {node_type: embedding.weight.data for node_type, embedding in self.node_embeddings.items()}
        
        # Récupérer les embeddings des relations
        rel_emb = self.rel_emb.weight.data
        
        logger.debug("Retrieved embeddings for entities and relations")
        
        return ent_emb, rel_emb

    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, entities=True):
        """
        Link prediction evaluation helper function. Get entities embeddings
        and relations embeddings. The output will be fed to the
        `inference_scoring_function` method.

        Parameters
        ----------
        h_idx : torch.Tensor
            The indices of the head entities (from KG).
        t_idx : torch.Tensor
            The indices of the tail entities (from KG).
        r_idx : torch.Tensor
            The indices of the relations (from KG).
        entities : bool, optional
            If True, prepare candidate entities; otherwise, prepare candidate relations.

        Returns
        -------
        h: torch.Tensor
            Head entity embeddings.
        t: torch.Tensor
            Tail entity embeddings.
        r: torch.Tensor
            Relation embeddings.
        candidates: torch.Tensor
            Candidate embeddings for entities or relations.
        """
        b_size = h_idx.shape[0]

        # Récupérer les embeddings des têtes, des queues et des relations
        h = torch.cat([self.node_embeddings[self.kg2nodetype[h_id.item()]].weight.data[h_id] for h_id in h_idx], dim=0)
        t = torch.cat([self.node_embeddings[self.kg2nodetype[t_id.item()]].weight.data[t_id] for t_id in t_idx], dim=0)
        r = self.rel_emb(r_idx)

        if entities:
            # Préparer les candidats pour les entités (toutes les entités)
            candidates = torch.cat([embedding.weight.data for embedding in self.node_embeddings.values()], dim=0)
            candidates = candidates.view(1, -1, self.emb_dim).expand(b_size, -1, -1)
        else:
            # Préparer les candidats pour les relations (toutes les relations)
            candidates = self.rel_emb.weight.data.unsqueeze(0).expand(b_size, -1, -1)

        logger.debug(f"Prepared {'entity' if entities else 'relation'} candidates for inference")
        
        return h, t, r, candidates
