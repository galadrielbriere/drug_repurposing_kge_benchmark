import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
import pandas as pd 
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  

def my_init_embedding(num_embeddings, emb_dim):
    """Initialise une couche d'embedding avec une distribution normale."""
    embedding = nn.Embedding(num_embeddings, emb_dim)
    nn.init.xavier_uniform_(embedding.weight.data)
    return embedding

def extract_node_type(node_name):
    """Extracts the node type from the node name, based on the string before the first underscore."""
    return node_name.split('_')[0]

def create_hetero_data(kg, mapping_csv):
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

    mapping = pd.read_csv(mapping_csv, sep=",", header=1)
    type_mapping = mapping[["type","id"]] # Keep only type and id column

    # 1. Parser les types de nœuds et les identifiants
    df = pd.merge(df, type_mapping.add_prefix("from_"), how="left", left_on="from", right_on="from_id")
    df = pd.merge(df, type_mapping.add_prefix("to_"), how="left", left_on="to", right_on="to_id", suffixes=(None, "_to"))
    df.drop([i for i in df.columns if "id" in i],axis=1, inplace=True)

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
