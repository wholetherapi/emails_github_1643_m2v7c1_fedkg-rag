import torch
import torch.nn.functional as F


class DenseRetriever:
    """Dense passage retrieval over locally held knowledge graphs."""

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, query, passages):
        q = F.normalize(query, dim=-1)
        p = F.normalize(passages, dim=-1)
        return torch.einsum("...d,...kd->...k", q, p)
