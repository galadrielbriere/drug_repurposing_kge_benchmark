# -*- coding: utf-8 -*-
"""
Knowledge Graph Preparation and Cleaning Script
@author: Galadriel Brière <marie-galadriel.briere@univ-amu.fr>

This script is designed to prepare and clean a knowledge graph using various utility functions and configurations. It supports tasks such as parsing a YAML configuration, ensuring entity coverage, setting random seeds for reproducibility, cleaning duplicated triples, and saving/loading the knowledge graph.
"""

import sys
import os
import argparse
import pandas as pd
import pickle
import yaml
import random
import numpy as np
import torch
import logging
from torch import cat

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

sys.path.append('dev')
import my_data_redundancy
import my_knowledge_graph

def parse_yaml(config_path):
    """Load and parse the YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"File {config_path} not found.")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config



def verify_entity_coverage(train_kg, full_kg):
    """
    Verify that all entities in the full knowledge graph are represented in the training set.

    Parameters
    ----------
    train_kg: KnowledgeGraph
        The training knowledge graph.
    full_kg: KnowledgeGraph
        The full knowledge graph.

    Returns
    -------
    tuple
        (bool, list)
        A tuple where the first element is True if all entities in the full knowledge graph are present in the training 
        knowledge graph, and the second element is a list of missing entities (names) if any are missing.
    """
    # Obtenir les identifiants d'entités pour le graphe d'entraînement et le graphe complet
    train_entities = set(cat((train_kg.head_idx, train_kg.tail_idx)).tolist())
    full_entities = set(cat((full_kg.head_idx, full_kg.tail_idx)).tolist())
    
    # Entités manquantes dans le graphe d'entraînement
    missing_entity_ids = full_entities - train_entities
    
    if missing_entity_ids:
        # Inverser le dictionnaire ent2ix pour obtenir idx: entity_name
        ix2ent = {v: k for k, v in full_kg.ent2ix.items()}
        
        # Récupérer les noms des entités manquantes à partir de leurs indices
        missing_entities = [ix2ent[idx] for idx in missing_entity_ids if idx in ix2ent]
        return False, missing_entities
    else:
        return True, []

def set_random_seeds(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_knowledge_graph(config, kg_train, kg_val, kg_test):
    """Save the knowledge graph to files."""
    pickle_filename = os.path.join(config['common']['out'], 'kg.pkl')
    logging.info(f"Saving results to {pickle_filename}...")
    with open(pickle_filename, 'wb') as file:
        pickle.dump(kg_train, file)
        pickle.dump(kg_val, file)
        pickle.dump(kg_test, file)

def prepare_knowledge_graph(config):
    """Prepare and clean the knowledge graph."""
    # Load knowledge graph
    input_file = config["common"]['input_csv']
    kg_df = pd.read_csv(input_file, sep="\t")[["my_x_id", "my_y_id", "relation"]]
    kg_df = kg_df.rename(columns={'my_x_id': 'from', 'my_y_id': 'to', 'relation': 'rel'})

    if config["clean_kg"]["smaller_kg"]:
        logging.info(f"Keeping only relations {config['clean_kg']['keep_relations']}")
        kg_df = kg_df[kg_df['rel'].isin(config["clean_kg"]['keep_relations'])]

    kg = my_knowledge_graph.KnowledgeGraph(df=kg_df)

    # Clean and process knowledge graph
    kg_train, kg_val, kg_test = clean_knowledge_graph(kg, config)

    # Save results
    save_knowledge_graph(config, kg_train, kg_val, kg_test)

    return kg_train, kg_val, kg_test

def load_knowledge_graph(config):
    """Load the knowledge graph from pickle files."""
    pickle_filename = config["common"]['input_pkl']
    logging.info(f'Will not run the preparation step. Using KG stored in: {pickle_filename}')
    with open(pickle_filename, 'rb') as file:
        kg_train = pickle.load(file)
        kg_val = pickle.load(file)
        kg_test = pickle.load(file)
    return kg_train, kg_val, kg_test

def clean_knowledge_graph(kg, config):
    """Clean and prepare the knowledge graph according to the configuration."""

    set_random_seeds(config["common"]["seed"])

    id_to_rel_name = {v: k for k, v in kg.rel2ix.items()}

    if config["clean_kg"]['remove_duplicates_triplets']:
        logging.info("Removing duplicated triplets...")
        kg = my_data_redundancy.remove_duplicates_triplets(kg)

    duplicated_relations_list = []

    if config['clean_kg']['check_synonymous_antisynonymous']:
        logging.info("Checking for synonymous and antisynonymous relations...")
        theta1 = config['clean_kg']['check_synonymous_antisynonymous_params']['theta1']
        theta2 = config['clean_kg']['check_synonymous_antisynonymous_params']['theta2']
        duplicates_relations, rev_duplicates_relations = my_data_redundancy.duplicates(kg, theta1=theta1, theta2=theta2)
        if duplicates_relations:
            logging.info(f'Adding {len(duplicates_relations)} synonymous relations ({[id_to_rel_name[rel] for rel in duplicates_relations]}) to the list of known duplicated relations.')
            duplicated_relations_list.extend(duplicates_relations)
        if rev_duplicates_relations:
            logging.info(f'Adding {len(rev_duplicates_relations)} anti-synonymous relations ({[id_to_rel_name[rel] for rel in rev_duplicates_relations]}) to the list of known duplicated relations.')
            duplicated_relations_list.extend(rev_duplicates_relations)
    
    if config['clean_kg']["permute_kg"]:
        to_permute_relation_names = config['clean_kg']["permute_kg_params"]
        if len(to_permute_relation_names) > 1:
            logging.info(f'Making permutations for relations {", ".join([rel for rel in to_permute_relation_names])}...')
        for rel in to_permute_relation_names:
            logging.info(f'Making permutations for relation {rel} with id {kg.rel2ix[rel]}.')
            kg = my_data_redundancy.permute_tails(kg, kg.rel2ix[rel])

    if config['clean_kg']['make_directed']:
        undirected_relations_names = config['clean_kg']['make_directed_params']
        relation_names = ", ".join([rel for rel in undirected_relations_names])
        logging.info(f'Adding reverse triplets for relations {relation_names}...')
        kg, undirected_relations_list = my_data_redundancy.add_inverse_relations(kg, [kg.rel2ix[key] for key in undirected_relations_names])
            
        if config['clean_kg']['check_synonymous_antisynonymous']:
            logging.info(f'Adding created reverses {[rel for rel in undirected_relations_names]} to the list of known duplicated relations.')
            duplicated_relations_list.extend(undirected_relations_list)

    logging.info("Splitting the dataset into train, validation and test sets...")
    kg_train, kg_val, kg_test = kg.split_kg(validation=True)

    kg_train_ok, _ = verify_entity_coverage(kg_train, kg)
    if not kg_train_ok:
        logging.info("Entity coverage verification failed...")
    else:
        logging.info("Entity coverage verified successfully.")

    if config['clean_kg']['clean_train_set']:
        logging.info("Cleaning the train set to avoid data leakage...")
        logging.info("Step 1: with respect to validation set.")
        kg_train = my_data_redundancy.clean_datasets(kg_train, kg_val, known_reverses=duplicated_relations_list)
        logging.info("Step 2: with respect to test set.")
        kg_train = my_data_redundancy.clean_datasets(kg_train, kg_test, known_reverses=duplicated_relations_list)

    kg_train_ok, _ = verify_entity_coverage(kg_train, kg)
    if not kg_train_ok:
        logging.info("Entity coverage verification failed...")
    else:
        logging.info("Entity coverage verified successfully.")

    if config['clean_kg']['rel_swap']:
        kg_train, kg_val, kg_test = specs_sets(kg_train, kg_val, kg_test, config)

    new_train, new_val, new_test = my_data_redundancy.ensure_entity_coverage(kg_train, kg_val, kg_test)


    kg_train_ok, missing_entities = verify_entity_coverage(new_train, kg)
    if not kg_train_ok:
        logging.info(f"Entity coverage verification failed. {len(missing_entities)} entities are missing.")
        logging.info(f"Missing entities: {missing_entities}")
        raise ValueError('One or more entities are not covered in the training set after ensuring entity coverage...')
    else:
        logging.info("Entity coverage verified successfully.")

    if config['clean_kg']['compute_proportions']:
        logging.info("Computing triplet proportions...")
        logging.info(my_data_redundancy.compute_triplet_proportions(kg_train, kg_test, kg_val))

    return new_train, new_val, new_test

def specs_sets(kg_train, kg_val, kg_test, config):
    """
    Modify the train, validation, and test sets according to specific relation swapping configurations.

    Parameters
    ----------
    kg_train: KnowledgeGraph
        The training knowledge graph.
    kg_val: KnowledgeGraph
        The validation knowledge graph.
    kg_test: KnowledgeGraph
        The test knowledge graph.
    config: dict
        The configuration dictionary containing relation swapping parameters.

    Returns
    -------
    new_kg_train: KnowledgeGraph
        The modified training knowledge graph.
    new_kg_val: KnowledgeGraph
        The modified validation knowledge graph.
    new_kg_test: KnowledgeGraph
        The modified test knowledge graph.
    """
    
    logging.info(f"Changing relation {config['clean_kg']['rel_swap_params'][0]} to {config['clean_kg']['rel_swap_params'][1]}...")
    logging.info(f"Triplets from {config['clean_kg']['rel_swap_params'][0]} will be kept only in test set with relation name {config['clean_kg']['rel_swap_params'][1]}...")
    
    df_train, df_val, df_test = kg_train.get_df(), kg_val.get_df(), kg_test.get_df()
    df_combined = pd.concat([df_train, df_val, df_test])

    # Identify drugs and diseases with the "orpha_treatment" relation
    drugs = df_combined[df_combined["rel"] == config['clean_kg']["rel_swap_params"][0]]["from"].values
    diseases = df_combined[df_combined["rel"] == config['clean_kg']["rel_swap_params"][0]]["to"].values
    kg_df_no_orpha = df_combined[df_combined["rel"] != config['clean_kg']["rel_swap_params"][0]]

    # Determine entities connected by other relations
    set_from = set(kg_df_no_orpha["from"])
    set_to = set(kg_df_no_orpha["to"])
    union_set = set_from.union(set_to)

    # Identify missing drugs and diseases
    missing_drugs = [drug for drug in drugs if drug not in union_set]
    missing_diseases = [disease for disease in diseases if disease not in union_set]

    # Separate relations to test and modify
    to_test = df_train[
        (df_train['rel'] == config['clean_kg']["rel_swap_params"][0]) &
        (~(df_train['from'].isin(missing_drugs)) | (df_train['to'].isin(missing_diseases)))
    ]
    to_modify = df_train[
        (df_train['rel'] == config['clean_kg']["rel_swap_params"][0]) &
        ((df_train['from'].isin(missing_drugs)) | (df_train['to'].isin(missing_diseases)))
    ]

    df_train = df_train.drop(to_test.index)
    to_test['rel'] = config['clean_kg']["rel_swap_params"][1]
    df_test = pd.concat([df_test, to_test])
    df_train.loc[to_modify.index, 'rel'] = config['clean_kg']["rel_swap_params"][1]
    df_val.loc[df_val['rel'] == config['clean_kg']["rel_swap_params"][0], 'rel'] = config['clean_kg']["rel_swap_params"][1]

    if not config['clean_kg']["mixed_test"]:
        rels_in_test = config['clean_kg']["mixed_params"]
        logging.info(f'Keeping only relations {rels_in_test} in the test set and validation set.')
        to_transfer = df_test[~df_test['rel'].isin(rels_in_test)]
        to_transfer_val = df_val[~df_val['rel'].isin(rels_in_test)]

        df_train = pd.concat([df_train, to_transfer, to_transfer_val])
        df_test = df_test[df_test['rel'].isin(rels_in_test)]
        df_val = df_val[df_val['rel'].isin(rels_in_test)]

    # Update entity and relation dictionaries
    df_combined = pd.concat([df_train, df_val, df_test])
    ent2ix = {entity: idx for idx, entity in enumerate(pd.concat([df_combined['from'], df_combined['to']]).unique())}
    rel2ix = {relation: idx for idx, relation in enumerate(df_combined['rel'].unique())}

    # Re-create KnowledgeGraph objects
    new_kg_train = my_knowledge_graph.KnowledgeGraph(df=df_train, ent2ix=ent2ix, rel2ix=rel2ix)
    new_kg_val = my_knowledge_graph.KnowledgeGraph(df=df_val, ent2ix=ent2ix, rel2ix=rel2ix)
    new_kg_test = my_knowledge_graph.KnowledgeGraph(df=df_test, ent2ix=ent2ix, rel2ix=rel2ix)

    return new_kg_train, new_kg_val, new_kg_test
