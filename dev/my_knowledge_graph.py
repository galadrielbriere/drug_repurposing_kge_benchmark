# -*- coding: utf-8 -*-
"""
Original code for the KnowledgeGraph class from TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
This class was originally implemented to represent a knowledge graph, with methods explained in `this paper <https://arxiv.org/pdf/2003.08001.pdf>`__ by Akrami et al.

Modifications and additional functionalities added by Galadriel Brière <marie-galadriel.briere@univ-amu.fr>:
- Added new methods:
  - `keep_triples()`, `remove_triples()`, `duplicate_triples()`: Methods for modifying subsets of triples in the knowledge graph.
  - `add_triples()`: Method to add new triples while ensuring entity and relation integrity.
  
These modifications build on the original class, retaining core functionality while extending it for greater flexibility and use in more advanced applications. All modifications are released under the same license as the original code, respecting the contributions of the original authors.
"""

from torch import tensor, cat, eq, zeros_like, int64, long
import torch

from collections import defaultdict

from pandas import DataFrame
from torch import cat, eq, int64, long, randperm, tensor, Tensor, zeros_like
from torch.utils.data import Dataset

from torchkge.exceptions import SizeMismatchError, WrongArgumentsError, SanityError
from torchkge.utils.operations import get_dictionaries


class KnowledgeGraph(Dataset):
    """Knowledge graph representation. At least one of `df` and `kg`
    parameters should be passed.

    Parameters
    ----------
    df: pandas.DataFrame, optional
        Data frame containing three columns [from, to, rel].
    kg: dict, optional
        Dictionary with keys ('heads', 'tails', 'relations') and values
        the corresponding torch long tensors.
    ent2ix: dict, optional
        Dictionary mapping entity labels to their integer key. This is
        computed if not passed as argument.
    rel2ix: dict, optional
        Dictionary mapping relation labels to their integer key. This is
        computed if not passed as argument.
    dict_of_heads: dict, optional
        Dictionary of possible heads :math:`h` so that the triple
        :math:`(h,r,t)` gives a true fact. The keys are tuples (t, r).
        This is computed if not passed as argument.
    dict_of_tails: dict, optional
        Dictionary of possible tails :math:`t` so that the triple
        :math:`(h,r,t)` gives a true fact. The keys are tuples (h, r).
        This is computed if not passed as argument.
    dict_of_rels: dict, optional
        Dictionary of possible relations :math:`r` so that the triple
        :math:`(h,r,t)` gives a true fact. The keys are tuples (h, t).
        This is computed if not passed as argument.


    Attributes
    ----------
    ent2ix: dict
        Dictionary mapping entity labels to their integer key.
    rel2ix: dict
        Dictionary mapping relation labels to their integer key.
    n_ent: int
        Number of distinct entities in the data set.
    n_rel: int
        Number of distinct entities in the data set.
    n_facts: int
        Number of samples in the data set. A sample is a fact: a triplet
        (h, r, l).
    head_idx: torch.Tensor, dtype = torch.long, shape: (n_facts)
        List of the int key of heads for each fact.
    tail_idx: torch.Tensor, dtype = torch.long, shape: (n_facts)
        List of the int key of tails for each fact.
    relations: torch.Tensor, dtype = torch.long, shape: (n_facts)
        List of the int key of relations for each fact.

    """

    def __init__(self, df=None, kg=None, ent2ix=None, rel2ix=None,
                 dict_of_heads=None, dict_of_tails=None, dict_of_rels=None):

        if df is None:
            if kg is None:
                raise WrongArgumentsError("Please provide at least one "
                                          "argument of `df` and kg`")
            else:
                try:
                    assert (type(kg) == dict) & ('heads' in kg.keys()) & \
                           ('tails' in kg.keys()) & \
                           ('relations' in kg.keys())
                except AssertionError:
                    raise WrongArgumentsError("Keys in the `kg` dict should "
                                              "contain `heads`, `tails`, "
                                              "`relations`.")
                try:
                    assert (rel2ix is not None) & (ent2ix is not None)
                except AssertionError:
                    raise WrongArgumentsError("Please provide the two "
                                              "dictionaries ent2ix and rel2ix "
                                              "if building from `kg`.")
        else:
            if kg is not None:
                raise WrongArgumentsError("`df` and kg` arguments should not "
                                          "both be provided.")

        if ent2ix is None:
            self.ent2ix = get_dictionaries(df, ent=True)
        else:
            self.ent2ix = ent2ix

        if rel2ix is None:
            self.rel2ix = get_dictionaries(df, ent=False)
        else:
            self.rel2ix = rel2ix

        self.n_ent = max(self.ent2ix.values()) + 1
        self.n_rel = max(self.rel2ix.values()) + 1

        if df is not None:
            # build kg from a pandas dataframe
            self.n_facts = len(df)
            self.head_idx = tensor(df['from'].map(self.ent2ix).values).long()
            self.tail_idx = tensor(df['to'].map(self.ent2ix).values).long()
            self.relations = tensor(df['rel'].map(self.rel2ix).values).long()
        else:
            # build kg from another kg
            self.n_facts = kg['heads'].shape[0]
            self.head_idx = kg['heads']
            self.tail_idx = kg['tails']
            self.relations = kg['relations']

        if dict_of_heads is None or dict_of_tails is None or dict_of_rels is None:
            self.dict_of_heads = defaultdict(set)
            self.dict_of_tails = defaultdict(set)
            self.dict_of_rels = defaultdict(set)
            self.evaluate_dicts()

        else:
            self.dict_of_heads = dict_of_heads
            self.dict_of_tails = dict_of_tails
            self.dict_of_rels = dict_of_rels
        try:
            self.sanity_check()
        except AssertionError:
            raise SanityError("Please check the sanity of arguments.")

    def __len__(self):
        return self.n_facts

    def __getitem__(self, item):
        return (self.head_idx[item].item(),
                self.tail_idx[item].item(),
                self.relations[item].item())

    def sanity_check(self):
        assert (type(self.dict_of_heads) == defaultdict) & \
               (type(self.dict_of_tails) == defaultdict) & \
               (type(self.dict_of_rels) == defaultdict)
        assert (type(self.ent2ix) == dict) & (type(self.rel2ix) == dict)
        assert (len(self.ent2ix) == self.n_ent) & \
               (len(self.rel2ix) == self.n_rel)
        assert (type(self.head_idx) == Tensor) & \
               (type(self.tail_idx) == Tensor) & \
               (type(self.relations) == Tensor)
        assert (self.head_idx.dtype == int64) & \
               (self.tail_idx.dtype == int64) & (self.relations.dtype == int64)
        assert (len(self.head_idx) == len(self.tail_idx) == len(self.relations))


    def split_kg(self, share=0.8, sizes=None, validation=False):
        """Split the knowledge graph into train and test. If `sizes` is
        provided then it is used to split the samples as explained below. If
        only `share` is provided, the split is done at random but it assures
        to keep at least one fact involving each type of entity and relation
        in the training subset.
        Does not update the dictionary of facts.

        Parameters
        ----------
        share: float
            Percentage to allocate to train set.
        sizes: tuple
            Tuple of ints of length 2 or 3.

            * If len(sizes) == 2, then the first sizes[0] values of the
              knowledge graph will be used as training set and the rest as
              test set.

            * If len(sizes) == 3, then the first sizes[0] values of the
              knowledge graph will be used as training set, the following
              sizes[1] as validation set and the last sizes[2] as testing set.
        validation: bool
            Indicate if a validation set should be produced along with train
            and test sets.

        Returns
        -------
        train_kg: torchkge.data_structures.KnowledgeGraph
        val_kg: torchkge.data_structures.KnowledgeGraph, optional
        test_kg: torchkge.data_structures.KnowledgeGraph

        """
        if sizes is not None:
            try:
                if len(sizes) == 3:
                    try:
                        assert (sizes[0] + sizes[1] + sizes[2] == self.n_facts)
                    except AssertionError:
                        raise WrongArgumentsError('Sizes should sum to the '
                                                  'number of facts.')
                elif len(sizes) == 2:
                    try:
                        assert (sizes[0] + sizes[1] == self.n_facts)
                    except AssertionError:
                        raise WrongArgumentsError('Sizes should sum to the '
                                                  'number of facts.')
                else:
                    raise SizeMismatchError('Tuple `sizes` should be of '
                                            'length 2 or 3.')
            except AssertionError:
                raise SizeMismatchError('Tuple `sizes` should sum up to the '
                                        'number of facts in the knowledge '
                                        'graph.')
        else:
            assert share < 1

        if ((sizes is not None) and (len(sizes) == 3)) or \
                ((sizes is None) and validation):
            # return training, validation and a testing graphs

            if (sizes is None) and validation:
                mask_tr, mask_val, mask_te = self.get_mask(share,
                                                           validation=True)
            else:
                mask_tr = cat([tensor([1 for _ in range(sizes[0])]),
                               tensor([0 for _ in range(sizes[1] + sizes[2])])]).bool()
                mask_val = cat([tensor([0 for _ in range(sizes[0])]),
                                tensor([1 for _ in range(sizes[1])]),
                                tensor([0 for _ in range(sizes[2])])]).bool()
                mask_te = ~(mask_tr | mask_val)

            return (KnowledgeGraph(
                        kg={'heads': self.head_idx[mask_tr],
                            'tails': self.tail_idx[mask_tr],
                            'relations': self.relations[mask_tr]},
                        ent2ix=self.ent2ix, rel2ix=self.rel2ix,
                        dict_of_heads=self.dict_of_heads,
                        dict_of_tails=self.dict_of_tails,
                        dict_of_rels=self.dict_of_rels),
                    KnowledgeGraph(
                        kg={'heads': self.head_idx[mask_val],
                            'tails': self.tail_idx[mask_val],
                            'relations': self.relations[mask_val]},
                        ent2ix=self.ent2ix, rel2ix=self.rel2ix,
                        dict_of_heads=self.dict_of_heads,
                        dict_of_tails=self.dict_of_tails,
                        dict_of_rels=self.dict_of_rels),
                    KnowledgeGraph(
                        kg={'heads': self.head_idx[mask_te],
                            'tails': self.tail_idx[mask_te],
                            'relations': self.relations[mask_te]},
                        ent2ix=self.ent2ix, rel2ix=self.rel2ix,
                        dict_of_heads=self.dict_of_heads,
                        dict_of_tails=self.dict_of_tails,
                        dict_of_rels=self.dict_of_rels))
        else:
            # return training and testing graphs

            assert (((sizes is not None) and len(sizes) == 2) or
                    ((sizes is None) and not validation))
            if sizes is None:
                mask_tr, mask_te = self.get_mask(share, validation=False)
            else:
                mask_tr = cat([tensor([1 for _ in range(sizes[0])]),
                               tensor([0 for _ in range(sizes[1])])]).bool()
                mask_te = ~mask_tr
            return (KnowledgeGraph(
                        kg={'heads': self.head_idx[mask_tr],
                            'tails': self.tail_idx[mask_tr],
                            'relations': self.relations[mask_tr]},
                        ent2ix=self.ent2ix, rel2ix=self.rel2ix,
                        dict_of_heads=self.dict_of_heads,
                        dict_of_tails=self.dict_of_tails,
                        dict_of_rels=self.dict_of_rels),
                    KnowledgeGraph(
                        kg={'heads': self.head_idx[mask_te],
                            'tails': self.tail_idx[mask_te],
                            'relations': self.relations[mask_te]},
                        ent2ix=self.ent2ix, rel2ix=self.rel2ix,
                        dict_of_heads=self.dict_of_heads,
                        dict_of_tails=self.dict_of_tails,
                        dict_of_rels=self.dict_of_rels))



    def get_mask(self, share, validation=False):
        """Returns masks to split knowledge graph into train, test and
        optionally validation sets. The mask is first created by dividing
        samples between subsets based on relation equilibrium. Then if any
        entity is not present in the training subset it is manually added by
        assigning a share of the sample involving the missing entity either
        as head or tail.

        Parameters
        ----------
        share: float
        validation: bool

        Returns
        -------
        mask: torch.Tensor, shape: (n), dtype: torch.bool
        mask_val: torch.Tensor, shape: (n), dtype: torch.bool (optional)
        mask_te: torch.Tensor, shape: (n), dtype: torch.bool
        """

        uniques_r, counts_r = self.relations.unique(return_counts=True)
        uniques_e, _ = cat((self.head_idx,
                            self.tail_idx)).unique(return_counts=True)

        mask = zeros_like(self.relations).bool()
        if validation:
            mask_val = zeros_like(self.relations).bool()

        # splitting relations among subsets
        for i, r in enumerate(uniques_r):
            rand = randperm(counts_r[i].item())

            # list of indices k such that relations[k] == r
            sub_mask = eq(self.relations, r).nonzero(as_tuple=False)[:, 0]

            assert len(sub_mask) == counts_r[i].item()

            if validation:
                train_size, val_size, test_size = self.get_sizes(counts_r[i].item(),
                                                                 share=share,
                                                                 validation=True)
                mask[sub_mask[rand[:train_size]]] = True
                mask_val[sub_mask[rand[train_size:train_size + val_size]]] = True

            else:
                train_size, test_size = self.get_sizes(counts_r[i].item(),
                                                       share=share,
                                                       validation=False)
                mask[sub_mask[rand[:train_size]]] = True

        # adding missing entities to the train set
        u = cat((self.head_idx[mask], self.tail_idx[mask])).unique()
        if len(u) < self.n_ent:
            missing_entities = tensor(list(set(uniques_e.tolist()) -
                                           set(u.tolist())), dtype=long)
            for e in missing_entities:
                sub_mask = ((self.head_idx == e) |
                            (self.tail_idx == e)).nonzero(as_tuple=False)[:, 0]
                rand = randperm(len(sub_mask))
                sizes = self.get_sizes(mask.shape[0],
                                       share=share,
                                       validation=validation)
                mask[sub_mask[rand[:sizes[0]]]] = True
                if validation:
                    mask_val[sub_mask[rand[:sizes[0]]]] = False

        if validation:
            assert not (mask & mask_val).any().item()
            return mask, mask_val, ~(mask | mask_val)
        else:
            return mask, ~mask

    @staticmethod
    def get_sizes(count, share, validation=False):
        """With `count` samples, returns how many should go to train and test

        """
        if count == 1:
            if validation:
                return 1, 0, 0
            else:
                return 1, 0
        if count == 2:
            if validation:
                return 1, 1, 0
            else:
                return 1, 1

        n_train = int(count * share)
        assert n_train < count
        if n_train == 0:
            n_train += 1

        if not validation:
            return n_train, count - n_train
        else:
            if count - n_train == 1:
                n_train -= 1
                return n_train, 1, 1
            else:
                n_val = int(int(count - n_train) / 2)
                return n_train, n_val, count - n_train - n_val



    def evaluate_dicts(self):
        """Evaluates dicts of possible alternatives to an entity in a fact
        that still gives a true fact in the entire knowledge graph.

        """
        for i in range(self.n_facts):
            self.dict_of_heads[(self.tail_idx[i].item(),
                                self.relations[i].item())].add(self.head_idx[i].item())
            self.dict_of_tails[(self.head_idx[i].item(),
                                self.relations[i].item())].add(self.tail_idx[i].item())
            self.dict_of_rels[(self.head_idx[i].item(),
                               self.tail_idx[i].item())].add(self.relations[i].item())



    def get_df(self):
        """
        Returns a Pandas DataFrame with columns ['from', 'to', 'rel'].
        """
        ix2ent = {v: k for k, v in self.ent2ix.items()}
        ix2rel = {v: k for k, v in self.rel2ix.items()}

        df = DataFrame(cat((self.head_idx.view(1, -1),
                            self.tail_idx.view(1, -1),
                            self.relations.view(1, -1))).transpose(0, 1).numpy(),
                       columns=['from', 'to', 'rel'])

        df['from'] = df['from'].apply(lambda x: ix2ent[x])
        df['to'] = df['to'].apply(lambda x: ix2ent[x])
        df['rel'] = df['rel'].apply(lambda x: ix2rel[x])

        return df
    

    def keep_triples(self, indices_to_keep):
        """
        Keeps only the specified triples in the knowledge graph and returns a new
        KnowledgeGraph instance with these triples. Updates the dictionnary of facts.

        Parameters
        ----------
        indices_to_keep : list or torch.Tensor
            Indices of triples to keep in the knowledge graph.

        Returns
        -------
        KnowledgeGraph
            A new instance of KnowledgeGraph with only the specified triples.
        """
        # Create masks for indices to keep
        mask = torch.zeros(self.n_facts, dtype=torch.bool)
        mask[indices_to_keep] = True
        
        # Use the mask to filter the triples to keep
        new_heads = self.head_idx[mask]
        new_tails = self.tail_idx[mask]
        new_relations = self.relations[mask]


        # Create a new KnowledgeGraph instance
        return KnowledgeGraph(
            kg={'heads': new_heads, 'tails': new_tails, 'relations': new_relations},
            ent2ix=self.ent2ix,
            rel2ix=self.rel2ix
        )

    def remove_triples(self, indices_to_remove):
        """
        Removes specified triples from the knowledge graph and returns a new
        KnowledgeGraph instance without these triples.

        Parameters
        ----------
        indices_to_remove : list or torch.Tensor
            Indices of triples to remove from the knowledge graph.

        Returns
        -------
        KnowledgeGraph
            A new instance of KnowledgeGraph without the specified triples.
        """
        # Create masks for indices not to remove
        mask = torch.ones(self.n_facts, dtype=torch.bool)
        mask[indices_to_remove] = False
        
        # Use the mask to filter out the triples
        new_heads = self.head_idx[mask]
        new_tails = self.tail_idx[mask]
        new_relations = self.relations[mask]


        # Create a new KnowledgeGraph instance
        return KnowledgeGraph(
            kg={'heads': new_heads, 'tails': new_tails, 'relations': new_relations},
            ent2ix=self.ent2ix,
            rel2ix=self.rel2ix, 
            dict_of_heads=self.dict_of_heads,
            dict_of_tails=self.dict_of_tails,
            dict_of_rels=self.dict_of_rels
        )
    
    # def duplicate_triples(self, indices_to_duplicate):
    #     """
    #     Duplicates specified triples in the knowledge graph and returns a new
    #     KnowledgeGraph instance with these duplicates.

    #     Parameters
    #     ----------
    #     indices_to_duplicate : list or torch.Tensor
    #         Indices of triples to duplicate in the knowledge graph.

    #     Returns
    #     -------
    #     KnowledgeGraph
    #         A new instance of KnowledgeGraph with the specified triples duplicated.
    #     """
    #     # Créer des tensors additionnels pour les indices à dupliquer
    #     duplicated_heads = torch.cat((self.head_idx, self.head_idx[indices_to_duplicate]))
    #     duplicated_tails = torch.cat((self.tail_idx, self.tail_idx[indices_to_duplicate]))
    #     duplicated_relations = torch.cat((self.relations, self.relations[indices_to_duplicate]))

    #     # Créer une nouvelle instance de KnowledgeGraph avec les triplets dupliqués
    #     return KnowledgeGraph(
    #         kg={'heads': duplicated_heads, 'tails': duplicated_tails, 'relations': duplicated_relations},
    #         ent2ix=self.ent2ix,
    #         rel2ix=self.rel2ix
    #     )
    
    # def add_inverse_relations(self, undirected_relations):
    #     """
    #     Ajoute des triplets inverses pour les relations non dirigées spécifiées dans le graphe de connaissances.

    #     Parameters
    #     ----------
    #     undirected_relations: list
    #         Liste des relations non dirigées pour lesquelles on veut ajouter les triplets inverses.

    #     Returns
    #     -------
    #     KnowledgeGraph
    #         Une nouvelle instance de KnowledgeGraph avec les triplets inverses ajoutés.
    #     """

    #     ix2rel = {v: k for k, v in self.rel2ix.items()}

    #     # Copier les indices existants pour les têtes, les queues et les relations
    #     head_idx, tail_idx, relations = self.head_idx.clone(), self.tail_idx.clone(), self.relations.clone()

    #     # Liste pour stocker les nouveaux triplets inverses
    #     new_head_idx, new_tail_idx, new_relations = [], [], []

    #     for relation_id in undirected_relations:
    #         # Créez le nom de la relation inverse
    #         inverse_relation = f"{ix2rel[relation_id]}_inv"

    #         # Vérifiez si la relation existe dans le graphe
    #         if relation_id not in self.rel2ix.values():
    #             print(f"Relation {relation_id} non trouvée dans le graphe de connaissances. Skipping...")
    #             continue

    #         # Obtenir l'ID de la relation et créer un nouvel ID pour la relation inverse
    #         inverse_relation_id = len(self.rel2ix)
    #         self.rel2ix[inverse_relation] = inverse_relation_id

    #         # Masque pour les triplets de la relation actuelle
    #         mask = (relations == relation_id)

    #         # Ajouter les triplets inverses pour les triplets existants de cette relation
    #         new_head_idx.append(tail_idx[mask])
    #         new_tail_idx.append(head_idx[mask])
    #         new_relations.append(torch.full_like(relations[mask], inverse_relation_id))

    #     # Concaténer les nouveaux triplets inverses aux triplets existants
    #     if new_head_idx:

    #         self.head_idx = torch.cat((self.head_idx, *new_head_idx), dim=0)
    #         self.tail_idx = torch.cat((self.tail_idx, *new_tail_idx), dim=0)
    #         self.relations = torch.cat((self.relations, *new_relations), dim=0)

    #     # Retourner une nouvelle instance de KnowledgeGraph avec les triplets inverses ajoutés
    #     return KnowledgeGraph(
    #         kg={'heads': self.head_idx, 'tails': self.tail_idx, 'relations': self.relations},
    #         ent2ix=self.ent2ix,
    #         rel2ix=self.rel2ix
    #     )
    
    def add_triples(self, new_triples):
        """
        Ajoute de nouveaux triplets au graphe de connaissances.

        Parameters
        ----------
        new_triples : torch.Tensor
            Tensor de forme (n, 3) où chaque ligne représente un triplet (head_idx, tail_idx, rel_idx).

        Returns
        -------
        KnowledgeGraph
            Une nouvelle instance de KnowledgeGraph avec les nouveaux triplets ajoutés.
        """
        if not isinstance(new_triples, torch.Tensor):
            raise TypeError("new_triples doit être un torch.Tensor.")
        if new_triples.dim() != 2 or new_triples.size(1) != 3:
            raise ValueError("new_triples doit avoir la forme (n, 3).")

        # Vérifier que les entités et relations existent dans ent2ix et rel2ix
        max_ent_idx = max(new_triples[:, 0].max().item(), new_triples[:, 1].max().item())
        max_rel_idx = new_triples[:, 2].max().item()

        if max_ent_idx >= self.n_ent:
            raise ValueError(f"L'indice d'entité maximal ({max_ent_idx}) dépasse le nombre d'entités ({self.n_ent}).")
        if max_rel_idx >= self.n_rel:
            raise ValueError(f"L'indice de relation maximal ({max_rel_idx}) dépasse le nombre de relations ({self.n_rel}).")

        # Concaténer les nouveaux triplets aux triplets existants
        updated_head_idx = torch.cat((self.head_idx, new_triples[:, 0]), dim=0)
        updated_tail_idx = torch.cat((self.tail_idx, new_triples[:, 1]), dim=0)
        updated_relations = torch.cat((self.relations, new_triples[:, 2]), dim=0)

        # Mettre à jour dict_of_heads, dict_of_tails, dict_of_rels
        for h, t, r in new_triples.tolist():
            self.dict_of_heads[(t, r)].add(h)
            self.dict_of_tails[(h, r)].add(t)
            self.dict_of_rels[(h, t)].add(r)

        # Créer une nouvelle instance de KnowledgeGraph avec les triplets mis à jour
        return KnowledgeGraph(
            kg={'heads': updated_head_idx, 'tails': updated_tail_idx, 'relations': updated_relations},
            ent2ix=self.ent2ix,
            rel2ix=self.rel2ix,
            dict_of_heads=self.dict_of_heads,
            dict_of_tails=self.dict_of_tails,
            dict_of_rels=self.dict_of_rels
        )
