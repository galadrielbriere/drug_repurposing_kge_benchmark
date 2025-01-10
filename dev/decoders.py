from torch import empty, matmul, tensor
from torch.nn import Parameter
from torch.nn.functional import normalize
from torchkge.models import TranslationModel, BilinearModel
import torch
import logging

from model_mapper import ModelMapper
from utils import my_init_embedding

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  

# Translational Models
# kwargs are used when models can accept arguments other than the base n_entities and n_relations.
class TransE(TranslationModel):
    def __init__(self, n_entities, n_relations, **kwargs):
        super().__init__(n_entities, n_relations, dissimilarity_type=kwargs.get("dissimilarity_type", "L2"))

    def score(self, h_norm, r_emb, t_norm):
        return -self.dissimilarity(h_norm + r_emb, t_norm)
    
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

class TransH(TranslationModel, ModelMapper):
    def __init__(self, n_entities, n_relations, **kwargs):
        super().__init__(n_entities, n_relations, dissimilarity_type="L2")
        
        emb_dim = kwargs.get("emb_dim",100)
        self.norm_vect = my_init_embedding(n_relations, emb_dim)

        self.evaluated_projections = False
        self.projected_entities = Parameter(empty(size=(n_relations,
                                                        n_entities,
                                                        emb_dim,)),
                                                        requires_grad=False)
        
    def score(self, h_norm, r_emb, t_norm):
        norm_vect = normalize(self.norm_vect)

    @staticmethod
    def project(ent, norm_vect):
        return ent - (ent * norm_vect).sum(dim=1).view(-1, 1) * norm_vect
    
    def normalize_parameters(self, mapper):
        mapper.ent_emb.weight.data = normalize(mapper.ent_emb.weight.data,
                                             p=2, dim=1)
        self.norm_vect.weight.data = normalize(self.norm_vect.weight.data,
                                               p=2, dim=1)
        mapper.rel_emb.weight.data = self.project(mapper.rel_emb.weight.data,
                                                self.norm_vect.weight.data)
