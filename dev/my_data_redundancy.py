# -*- coding: utf-8 -*-
"""
Original code for the KnowledgeGraph class from TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
This class was initially designed to represent a knowledge graph, with its methods explained in `this paper <https://arxiv.org/pdf/2003.08001.pdf>`__ by Akrami et al.

Modifications and additional functionalities added by Galadriel Brière <marie-galadriel.briere@univ-amu.fr>:
- Fixed a bug in the `duplicate()` function.
- Added new functions:
    - `add_inverse_relation()`: Adds inverse triples for undirected relations.
    - `compute_triplet_proportion()`: Computes the proportion of each relationship type in the training, validation, and test datasets.
    - `permute_tails()`: Randomly permutes a relationship type while maintaining node degrees.
    - `ensure_entity_coverage()`: Verifies whether all entities are represented in the training set.
    - `clean_datasets()`: Prevents data leakage between training and validation/test sets.
    - `remove_duplicates_triplets()`: Keeps only unique triples to avoid redundancy.

These modifications extend the original class, preserving its core functionality while enhancing its flexibility and usability for more advanced applications. All modifications are released under the same license as the original code, respecting the contributions of the original authors.
"""
from itertools import combinations
import torch
from torch import cat
from tqdm.autonotebook import tqdm
import pandas as pd 
from my_knowledge_graph import KnowledgeGraph
from collections import Counter
import random
import logging

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def add_inverse_relations(kg, undirected_relations):
    """
    Adds inverse triples for the specified undirected relations in the knowledge graph.

    Parameters
    ----------
    kg : KnowledgeGraph
        The original knowledge graph.
    undirected_relations: list
        List of undirected relations for which inverse triples should be added.

    Returns
    -------
    KnowledgeGraph, list
        A new instance of KnowledgeGraph with the inverse triples added and a list of pairs
        (old relation ID, new inverse relation ID).
    """

    ix2rel = {v: k for k, v in kg.rel2ix.items()}

    # Copier les indices existants pour les têtes, les queues et les relations
    head_idx, tail_idx, relations = kg.head_idx.clone(), kg.tail_idx.clone(), kg.relations.clone()

    # Liste pour stocker les nouveaux triplets inverses
    new_head_idx, new_tail_idx, new_relations = [], [], []

    # Liste pour stocker les paires (ancienne relation ID, nouvelle relation inverse ID)
    reverse_list = []

    for relation_id in undirected_relations:
        # Créez le nom de la relation inverse
        inverse_relation = f"{ix2rel[relation_id]}_inv"

        # Vérifiez si la relation existe dans le graphe
        if relation_id not in kg.rel2ix.values():
            logging.info(f"Relation {relation_id} non trouvée dans le graphe de connaissances. Skipping...")
            continue

        # Obtenir l'ID de la relation et créer un nouvel ID pour la relation inverse
        inverse_relation_id = len(kg.rel2ix)
        kg.rel2ix[inverse_relation] = inverse_relation_id

        # Ajouter la paire (ancienne relation ID, nouvelle relation inverse ID) à la liste
        reverse_list.append((relation_id, inverse_relation_id))

        # Masque pour les triplets de la relation actuelle
        mask = (relations == relation_id)

        # Ajouter les triplets inverses pour les triplets existants de cette relation
        new_head_idx.append(tail_idx[mask])
        new_tail_idx.append(head_idx[mask])
        new_relations.append(torch.full_like(relations[mask], inverse_relation_id))

    # Concaténer les nouveaux triplets inverses aux triplets existants
    if new_head_idx:
        kg.head_idx = torch.cat((kg.head_idx, *new_head_idx), dim=0)
        kg.tail_idx = torch.cat((kg.tail_idx, *new_tail_idx), dim=0)
        kg.relations = torch.cat((kg.relations, *new_relations), dim=0)

    # Créer une nouvelle instance de KnowledgeGraph avec les données mises à jour
    kg = KnowledgeGraph(
        kg={'heads': kg.head_idx, 'tails': kg.tail_idx, 'relations': kg.relations},
        ent2ix=kg.ent2ix,
        rel2ix=kg.rel2ix
    )

    return kg, reverse_list

def compute_triplet_proportions(kg_train, kg_test, kg_val):
    """
    Computes the proportion of triples for each relation in each of the KnowledgeGraphs
    (train, test, val) relative to the total number of triples for that relation.

    Parameters
    ----------
    kg_train: KnowledgeGraph
        The training KnowledgeGraph instance.
    kg_test: KnowledgeGraph
        The test KnowledgeGraph instance.
    kg_val: KnowledgeGraph
        The validation KnowledgeGraph instance.

    Returns
    -------
    dict
        A dictionary where keys are relation identifiers and values are sub-dictionaries
        with the respective proportions of each relation in kg_train, kg_test, and kg_val.
    """
     
    # Concaténer les relations de tous les KGs
    all_relations = torch.cat((kg_train.relations, kg_test.relations, kg_val.relations))

    # Calculer le nombre total de triplets pour chaque relation
    total_counts = torch.bincount(all_relations)

    # Calculer les occurrences de chaque type de relation dans chaque KG
    train_counts = torch.bincount(kg_train.relations, minlength=len(total_counts))
    test_counts = torch.bincount(kg_test.relations, minlength=len(total_counts))
    val_counts = torch.bincount(kg_val.relations, minlength=len(total_counts))

    # Calculer les proportions pour chaque KG
    proportions = {}
    for rel_id in range(len(total_counts)):
        if total_counts[rel_id] > 0:
            proportions[rel_id] = {
                'train': train_counts[rel_id].item() / total_counts[rel_id].item(),
                'test': test_counts[rel_id].item() / total_counts[rel_id].item(),
                'val': val_counts[rel_id].item() / total_counts[rel_id].item()
            }

    return proportions

def permute_tails(kg, relation_id):
    """
    Randomly permutes the `tails` for a given relation while maintaining the original degree
    of `heads` and `tails`, ensuring there are no triples of the form (a, rel, a) where `head == tail`.

    Parameters
    ----------
    kg: KnowledgeGraph
        The KnowledgeGraph instance on which to perform the permutation.
    relation_id: int
        The ID of the relation for which `tails` should be permuted.

    Returns
    -------
    KnowledgeGraph
        A new instance of KnowledgeGraph with the `tails` permuted.
    """
    
    # Créer une copie des attributs pour la nouvelle instance
    new_head_idx = kg.head_idx.clone()
    new_tail_idx = kg.tail_idx.clone()
    new_relations = kg.relations.clone()

    # Masque pour filtrer les triplets de la relation donnée
    mask = (new_relations == relation_id)

    # Extraire les indices des `heads` et `tails` pour cette relation
    heads_for_relation = new_head_idx[mask].tolist()
    tails_for_relation = new_tail_idx[mask].tolist()

    # Compter le nombre d'occurrences de chaque `tail` dans la relation
    tails_count = Counter(tails_for_relation)

    # Obtenir un dérangement aléatoire des `tails` pour éviter des self-loops tout en préservant les degrés
    permuted_tails = tails_for_relation[:]
    random.shuffle(permuted_tails)

    # Corriger les self-loops tout en préservant le degré des noeuds
    for i in range(len(permuted_tails)):
        if heads_for_relation[i] == permuted_tails[i]:
            # Chercher un autre index pour échanger, en évitant les self-loops et en préservant les degrés
            found = False
            for j in range(i + 1, len(permuted_tails)):
                if heads_for_relation[j] != permuted_tails[i] and heads_for_relation[i] != permuted_tails[j]:
                    # Échanger pour résoudre le self-loop
                    permuted_tails[i], permuted_tails[j] = permuted_tails[j], permuted_tails[i]
                    found = True
                    break
            # Si aucun échange valide n'est trouvé, chercher à partir du début
            if not found:
                for j in range(0, i):
                    if heads_for_relation[j] != permuted_tails[i] and heads_for_relation[i] != permuted_tails[j]:
                        # Échanger pour résoudre le self-loop
                        permuted_tails[i], permuted_tails[j] = permuted_tails[j], permuted_tails[i]
                        break

    # Convertir la liste permutée en un tensor
    permuted_tails = torch.tensor(permuted_tails, dtype=new_tail_idx.dtype)

    # Remplacer les `tails` originaux par les `tails` permutés
    new_tail_idx[mask] = permuted_tails

    # Vérifier si le degré est préservé
    assert Counter(new_tail_idx[mask].tolist()) == tails_count, "Le degré des `tails` n'est pas préservé après permutation."
    assert all(new_head_idx[i] != new_tail_idx[i] for i in range(len(new_head_idx))), "Il y a des triplets avec le même `head` et `tail` après permutation."

    # Retourner une nouvelle instance de KnowledgeGraph avec les `tails` permutés
    return KnowledgeGraph(
        kg={'heads': new_head_idx, 'tails': new_tail_idx, 'relations': new_relations},
        ent2ix=kg.ent2ix,
        rel2ix=kg.rel2ix,
    )

def ensure_entity_coverage(kg_train, kg_val, kg_test):
    """
    Ensure that all entities in kg_train.ent2ix are present in kg_train as head or tail.
    If an entity is missing, move a triplet involving that entity from kg_val or kg_test to kg_train.

    Parameters
    ----------
    kg_train : torchkge.data_structures.KnowledgeGraph
        The training knowledge graph to ensure entity coverage.
    kg_val : torchkge.data_structures.KnowledgeGraph
        The validation knowledge graph from which to move triplets if needed.
    kg_test : torchkge.data_structures.KnowledgeGraph
        The test knowledge graph from which to move triplets if needed.

    Returns
    -------
    kg_train : torchkge.data_structures.KnowledgeGraph
        The updated training knowledge graph with all entities covered.
    kg_val : torchkge.data_structures.KnowledgeGraph
        The updated validation knowledge graph.
    kg_test : torchkge.data_structures.KnowledgeGraph
        The updated test knowledge graph.
    """

    # Obtenir l'ensemble des entités dans kg_train.ent2ix
    train_entities = set(kg_train.ent2ix.values())

    # Obtenir l'ensemble des entités présentes dans kg_train comme head ou tail
    present_heads = set(kg_train.head_idx.tolist())
    present_tails = set(kg_train.tail_idx.tolist())
    present_entities = present_heads.union(present_tails)

    # Identifier les entités manquantes dans kg_train
    missing_entities = train_entities - present_entities

    logging.info(f"Total entities in full kg: {len(train_entities)}")
    logging.info(f"Entities present in kg_train: {len(present_entities)}")
    logging.info(f"Missing entities in kg_train: {len(missing_entities)}")

    def find_and_move_triplets(source_kg, entities):
        nonlocal kg_train, kg_val, kg_test

        # Convert `entities` set to a `Tensor` for compatibility with `torch.isin`
        entities_tensor = torch.tensor(list(entities), dtype=source_kg.head_idx.dtype)

        # Create masks for all triplets where the missing entity is present
        mask_heads = torch.isin(source_kg.head_idx, entities_tensor)
        mask_tails = torch.isin(source_kg.tail_idx, entities_tensor)
        mask = mask_heads | mask_tails

        if mask.any():
            # Extract the indices and corresponding triplets
            indices = torch.nonzero(mask, as_tuple=True)[0]
            triplets = torch.stack([source_kg.head_idx[indices],
                                    source_kg.tail_idx[indices],
                                    source_kg.relations[indices]], dim=1)

            # Add the found triplets to kg_train
            kg_train = kg_train.add_triples(triplets)

            # Remove the triplets from source_kg
            kg_cleaned = source_kg.remove_triples(indices)
            if source_kg == kg_val:
                kg_val = kg_cleaned
            else:
                kg_test = kg_cleaned

            # Update the list of missing entities
            entities_in_triplets = set(triplets[:, 0].tolist() + triplets[:, 1].tolist())
            remaining_entities = entities - set(entities_in_triplets)
            return remaining_entities
        return entities

    # Déplacer les triplets depuis kg_val puis depuis kg_test
    missing_entities = find_and_move_triplets(kg_val, missing_entities)
    if len(missing_entities) > 0:
        missing_entities = find_and_move_triplets(kg_test, missing_entities)

    # Loguer les entités restantes non trouvées
    if len(missing_entities) > 0:
        for entity in missing_entities:
            logging.info(f"Warning: No triplet found involving entity '{entity}' in kg_val or kg_test. Entity remains unconnected in kg_train.")

    return kg_train, kg_val, kg_test


def clean_datasets(kg1, kg2, known_reverses):
    """
    Clean KG1 (training KG) by removing reverse duplicate triples contained in KG2 (test or val KG).

    Parameters
    ----------
    kg1: torchkge.data_structures.KnowledgeGraph
        The first knowledge graph.
    kg2: torchkge.data_structures.KnowledgeGraph
        The second knowledge graph.
    known_reverses: list of tuples
        Each tuple contains two relations (r1, r2) that are known reverse relations.

    Returns
    -------
    kg1: torchkge.data_structures.KnowledgeGraph
        The cleaned first knowledge graph.
    """

    for r1, r2 in known_reverses:

        logging.info(f"Processing relation pair: ({r1}, {r2})")

        # Get (h, t) pairs in kg2 related by r1
        kg2_ht_r1 = get_pairs(kg2, r1, type='ht')
        # Get indices of (h, t) in kg1 that are related by r2
        indices_to_remove_kg1 = [i for i, (h, t) in enumerate(zip(kg1.tail_idx, kg1.head_idx)) if (h.item(), t.item()) in kg2_ht_r1 and kg1.relations[i].item() == r2]
        indices_to_remove_kg1.extend([i for i, (h, t) in enumerate(zip(kg1.head_idx, kg1.tail_idx)) if (h.item(), t.item()) in kg2_ht_r1 and kg1.relations[i].item() == r2])
        
        # Remove those (h, t) pairs from kg1
        kg1 = kg1.remove_triples(torch.tensor(indices_to_remove_kg1, dtype=torch.long))

        logging.info(f"Found {len(indices_to_remove_kg1)} triplets to remove for relation {r2} with reverse {r1}.")

        # Get (h, t) pairs in kg2 related by r2
        kg2_ht_r2 = get_pairs(kg2, r2, type='ht')
        # Get indices of (h, t) in kg1 that are related by r1
        indices_to_remove_kg1_reverse = [i for i, (h, t) in enumerate(zip(kg1.tail_idx, kg1.head_idx)) if (h.item(), t.item()) in kg2_ht_r2 and kg1.relations[i].item() == r1]
        indices_to_remove_kg1_reverse.extend([i for i, (h, t) in enumerate(zip(kg1.head_idx, kg1.tail_idx)) if (h.item(), t.item()) in kg2_ht_r2 and kg1.relations[i].item() == r1])

        # Remove those (h, t) pairs from kg1
        kg1 = kg1.remove_triples(torch.tensor(indices_to_remove_kg1_reverse, dtype=torch.long))

        logging.info(f"Found {len(indices_to_remove_kg1_reverse)} reverse triplets to remove for relation {r1} with reverse {r2}.")
    
    return kg1

def remove_duplicates_triplets(kg):
    """
    Remove duplicate triples from a knowledge graph for each relation and keep only unique triples.

    This function processes each relation separately, identifies unique triples based on head and tail indices,
    and retains only the unique triples by filtering out duplicates.

    Parameters:
    - kg (KnowledgeGraph): An instance of a KnowledgeGraph containing head indices, tail indices, and relations.

    Returns:
    - KnowledgeGraph: A new instance of the KnowledgeGraph containing only unique triples.
    
    The function also updates a dictionary `T` which holds pairs of head and tail indices for each relation
    along with their original indices in the dataset.

    """
    T = {}  # Dictionary to store pairs for each relation
    keep = torch.tensor([], dtype=torch.long)  # Tensor to store indices of triples to keep

    h, t, r = kg.head_idx, kg.tail_idx, kg.relations

    # Process each relation
    for r_ in tqdm(range(kg.n_rel)):
        # Create a mask for the current relation
        mask = (r == r_)

        # Extract pairs of head and tail indices for the current relation
        original_indices = torch.arange(h.size(0))[mask]
        pairs = torch.stack((h[mask], t[mask]), dim=1)
        pairs = torch.sort(pairs, dim=1).values
        pairs = torch.cat([pairs, original_indices.unsqueeze(1)], dim=1)

        # Create a dictionary entry for the relation with pairs
        T[r_] = pairs

        # Identify unique triples and their original indices
        unique, idx, counts = torch.unique(pairs[:, :2], dim=0, sorted=True, return_inverse=True, return_counts=True)
        _, ind_sorted = torch.sort(idx, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))
        first_indices = ind_sorted[cum_sum]

        # Retrieve original indices of first unique entries
        adjusted_indices = pairs[first_indices, 2]

        # Accumulate unique indices globally
        keep = torch.cat((keep, adjusted_indices))

        # Logging duplicate information
        if len(pairs) - len(unique) > 0:
            logging.info(f'{len(pairs) - len(unique)} duplicates found. Keeping {len(unique)} unique triplets for relation {r_}')

    # Return a new KnowledgeGraph instance with only unique triples retained
    return kg.keep_triples(keep)

def concat_kgs(kg_tr, kg_val, kg_te):
    h = cat((kg_tr.head_idx, kg_val.head_idx, kg_te.head_idx))
    t = cat((kg_tr.tail_idx, kg_val.tail_idx, kg_te.tail_idx))
    r = cat((kg_tr.relations, kg_val.relations, kg_te.relations))
    return h, t, r

def get_pairs(kg, r, type='ht'):
    mask = (kg.relations == r)

    if type == 'ht':
        return set((i.item(), j.item()) for i, j in cat(
            (kg.head_idx[mask].view(-1, 1),
             kg.tail_idx[mask].view(-1, 1)), dim=1))
    else:
        assert type == 'th'
        return set((j.item(), i.item()) for i, j in cat(
            (kg.head_idx[mask].view(-1, 1),
             kg.tail_idx[mask].view(-1, 1)), dim=1))

def count_triplets(kg1, kg2, duplicates, rev_duplicates):
    """
    Parameters
    ----------
    kg1: torchkge.data_structures.KnowledgeGraph
    kg2: torchkge.data_structures.KnowledgeGraph
    duplicates: list
        List returned by torchkge.utils.data_redundancy.duplicates.
    rev_duplicates: list
        List returned by torchkge.utils.data_redundancy.duplicates.

    Returns
    -------
    n_duplicates: int
        Number of triplets in kg2 that have their duplicate triplet
        in kg1
    n_rev_duplicates: int
        Number of triplets in kg2 that have their reverse duplicate
        triplet in kg1.
    """
    n_duplicates = 0
    for r1, r2 in duplicates:
        ht_tr = get_pairs(kg1, r2, type='ht')
        ht_te = get_pairs(kg2, r1, type='ht')

        n_duplicates += len(ht_te.intersection(ht_tr))

        ht_tr = get_pairs(kg1, r1, type='ht')
        ht_te = get_pairs(kg2, r2, type='ht')

        n_duplicates += len(ht_te.intersection(ht_tr))

    n_rev_duplicates = 0
    for r1, r2 in rev_duplicates:
        th_tr = get_pairs(kg1, r2, type='th')
        ht_te = get_pairs(kg2, r1, type='ht')

        n_rev_duplicates += len(ht_te.intersection(th_tr))

        th_tr = get_pairs(kg1, r1, type='th')
        ht_te = get_pairs(kg2, r2, type='ht')

        n_rev_duplicates += len(ht_te.intersection(th_tr))

    return n_duplicates, n_rev_duplicates

def duplicates(kg, theta1=0.8, theta2=0.8, counts=False, reverses=None):
    """Return the duplicate and reverse duplicate relations as explained
    in paper by Akrami et al.

    References
    ----------
    * Farahnaz Akrami, Mohammed Samiul Saeef, Quingheng Zhang.
      `Realistic Re-evaluation of Knowledge Graph Completion Methods:
      An Experimental Study. <https://arxiv.org/pdf/2003.08001.pdf>`_
      SIGMOD’20, June 14–19, 2020, Portland, OR, USA

    Parameters
    ----------
    kg_tr: torchkge.data_structures.KnowledgeGraph
        Train set
    kg_val: torchkge.data_structures.KnowledgeGraph
        Validation set
    kg_te: torchkge.data_structures.KnowledgeGraph
        Test set
    theta1: float
        First threshold (see paper).
    theta2: float
        Second threshold (see paper).
    counts: bool
        Should the triplets involving (reverse) duplicate relations be
        counted in all sets.
    reverses: list
        List of known reverse relations.

    Returns
    -------
    duplicates: list
        List of pairs giving duplicate relations.
    rev_duplicates: list
        List of pairs giving reverse duplicate relations.
    """
    # QUESTION : counts not used?
    if reverses is None:
        reverses = []

    T = dict()
    T_inv = dict()
    lengths = dict()

    h, t, r = kg.head_idx, kg.tail_idx, kg.relations

    for r_ in tqdm(range(kg.n_rel)):
        mask = (r == r_)
        lengths[r_] = mask.sum().item()

        pairs = cat((h[mask].view(-1, 1), t[mask].view(-1, 1)), dim=1)

        T[r_] = set([(h_.item(), t_.item()) for h_, t_ in pairs])
        T_inv[r_] = set([(t_.item(), h_.item()) for h_, t_ in pairs])

    logging.info('Finding duplicate relations')

    duplicates = []
    rev_duplicates = []

    iter_ = list(combinations(range(kg.n_rel), 2))

    for r1, r2 in tqdm(iter_):
        a = len(T[r1].intersection(T[r2])) / lengths[r1]
        b = len(T[r1].intersection(T[r2])) / lengths[r2]

        if a > theta1 and b > theta2:
            duplicates.append((r1, r2))

        if (r1, r2) not in reverses:
            a = len(T[r1].intersection(T_inv[r2])) / lengths[r1]
            b = len(T[r1].intersection(T_inv[r2])) / lengths[r2]

            if a > theta1 and b > theta2:
                rev_duplicates.append((r1, r2))

    logging.info('Duplicate relations: {}'.format(len(duplicates)))
    logging.info('Reverse duplicate relations: '
            '{}\n'.format(len(rev_duplicates)))

    return duplicates, rev_duplicates

def cartesian_product_relations(kg, theta=0.8):
    """Return the cartesian product relations as explained in paper by
    Akrami et al.

    References
    ----------
    * Farahnaz Akrami, Mohammed Samiul Saeef, Quingheng Zhang.
      `Realistic Re-evaluation of Knowledge Graph Completion Methods: An
      Experimental Study. <https://arxiv.org/pdf/2003.08001.pdf>`_
      SIGMOD’20, June 14–19, 2020, Portland, OR, USA

    Parameters
    ----------
    kg: torchkge.data_structures.KnowledgeGraph
    theta: float
        Threshold used to compute the cartesian product relations.

    Returns
    -------
    selected_relations: list
        List of relations index that are cartesian product relations
        (see paper for details).

    """
    selected_relations = []

    h, t, r = kg.head_idx, kg.tail_idx, kg.relations

    S = dict()
    O = dict()
    lengths = dict()

    for r_ in tqdm(range(kg.n_rel)):
        mask = (r == r_)
        lengths[r_] = mask.sum().item()

        S[r_] = set(h_.item() for h_ in h[mask])
        O[r_] = set(t_.item() for t_ in t[mask])

        if lengths[r_] / (len(S[r_]) * len(O[r_])) > theta:
            selected_relations.append(r_)

    return selected_relations
