# -*- coding: utf-8 -*-
"""
Original code from Galadriel Brière <marie-galadriel.briere@univ-amu.fr>, based on existing sampler implementations from Armand Boschin <aboschin@enst.fr> and TorchKGE developpers.
"""

from torchkge.sampling import NegativeSampler, UniformNegativeSampler, BernoulliNegativeSampler
from positional_sampler import PositionalNegativeSampler
from torch import cat

class MixedNegativeSampler(NegativeSampler):
    """
    A custom negative sampler that combines the BernoulliNegativeSampler
    and the PositionalNegativeSampler. For each triplet, it samples `n_neg` negative samples
    using both samplers.
    
    Parameters
    ----------
    kg: torchkge.data_structures.KnowledgeGraph
        Main knowledge graph (usually training one).
    kg_val: torchkge.data_structures.KnowledgeGraph (optional)
        Validation knowledge graph.
    kg_test: torchkge.data_structures.KnowledgeGraph (optional)
        Test knowledge graph.
    n_neg: int
        Number of negative sample to create from each fact.
    """
    
    def __init__(self, kg, kg_val=None, kg_test=None, n_neg=1):
        super().__init__(kg, kg_val, kg_test, n_neg)
        # Initialize both Bernoulli and Positional samplers
        self.uniform_sampler = UniformNegativeSampler(kg, kg_val, kg_test, n_neg)
        self.bernoulli_sampler = BernoulliNegativeSampler(kg, kg_val, kg_test, n_neg)
        self.positional_sampler = PositionalNegativeSampler(kg, kg_val, kg_test)
        
    def corrupt_batch(self, heads, tails, relations, n_neg=None):
        """For each true triplet, produce `n_neg` corrupted ones from the
        Unniform sampler, the Bernoulli sampler and the Positional sampler. If `heads` and `tails` are
        cuda objects, then the returned tensors are on the GPU.

        Parameters
        ----------
        heads: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of heads of the relations in the
            current batch.
        tails: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of tails of the relations in the
            current batch.
        relations: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of relations in the current
            batch.
        n_neg: int (optional)
            Number of negative samples to create from each fact. If None, the class-level
            `n_neg` value is used.

        Returns
        -------
        combined_neg_heads: torch.Tensor, dtype: torch.long
            Tensor containing the integer key of negatively sampled heads from both samplers.
        combined_neg_tails: torch.Tensor, dtype: torch.long
            Tensor containing the integer key of negatively sampled tails from both samplers.
        """

        if heads.device != tails.device or heads.device != relations.device:
            raise ValueError(f"Tensors are on different devices: h is on {heads.device}, t is on {tails.device}, r is on {relations.device}")

        if n_neg is None:
            n_neg = self.n_neg

        # Get negative samples from Uniform sampler
        uniform_neg_heads, uniform_neg_tails = self.uniform_sampler.corrupt_batch(
            heads, tails, relations, n_neg=n_neg
        )
        
        # Get negative samples from Bernoulli sampler
        bernoulli_neg_heads, bernoulli_neg_tails = self.bernoulli_sampler.corrupt_batch(
            heads, tails, relations, n_neg=n_neg
        )
        
        # Get negative samples from Positional sampler
        positional_neg_heads, positional_neg_tails = self.positional_sampler.corrupt_batch(
            heads, tails, relations
        )
        
        # Combine results from all samplers
        combined_neg_heads = cat([uniform_neg_heads, bernoulli_neg_heads, positional_neg_heads])
        combined_neg_tails = cat([uniform_neg_tails,bernoulli_neg_tails, positional_neg_tails])
        
        return combined_neg_heads, combined_neg_tails
