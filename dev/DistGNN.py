import torch
import torch.nn as nn
from torch.nn.functional import normalize
from torch_geometric.nn import HeteroConv, SAGEConv
from torchkge.models import BilinearModel
import logging
from utils import my_init_embedding, create_hetero_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  

class DistMultModelWithGCN(BilinearModel):
    def __init__(self, emb_dim, n_entities, n_relations, kg, device, num_gcn_layers=2, aggr='sum'):
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
        """
        super().__init__(emb_dim, n_entities, n_relations) ###

        self.hetero_data, self.kg2het, self.het2kg, _, self.kg2nodetype = create_hetero_data(kg)
        self.hetero_data = self.hetero_data.to(device)

        # Initialisation des embeddings des relations
        self.rel_emb = my_init_embedding(self.n_rel, self.emb_dim)

        logger.info(f"self.hetero_data.node_types={self.hetero_data.node_types}")
        # Initialisation des embeddings initiaux pour chaque type de nœud
        self.node_embeddings = nn.ModuleDict()
        for node_type in self.hetero_data.node_types:
            num_nodes = self.hetero_data[node_type].num_nodes
            self.node_embeddings[node_type] = my_init_embedding(num_nodes, self.emb_dim)

        # Définir l'agrégation pour HeteroConv
        self.aggr = aggr

        self.convs = nn.ModuleList()
        # Définition des couches GCN multiples pour chaque type d'arête
        for layer in range(num_gcn_layers):
                    conv = HeteroConv(
                        {edge_type: SAGEConv(self.emb_dim, self.emb_dim, aggr="mean") for edge_type in self.hetero_data.edge_types},
                        aggr=self.aggr
                    )
                    self.convs.append(conv)
                    logger.info(f"Initialized HeteroConv layer {layer+1} with {len(conv.convs)} edge types.")

     
        # self.hetero_conv = HeteroConv(self.convs, aggr=self.aggr)
        # logger.info("set HeteroConv = OK")

        # Normalisation initiale des embeddings des relations
        self.rel_emb.weight.data = normalize(self.rel_emb.weight.data, p=2, dim=1)
  
    def forward_gnn(self):
        """
        Passe les embeddings des nœuds à travers les couches GNN.
        """
        x_dict = {node_type: embedding.weight for node_type, embedding in self.node_embeddings.items()}

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

        # 5. Calculer la similarité pour le triplet (h, r, t)
        return (h_normalized * r_embeddings * t_normalized).sum(dim=1)


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
