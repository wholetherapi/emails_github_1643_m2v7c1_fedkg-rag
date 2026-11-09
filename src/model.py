import torch
import torch.nn as nn

from retriever import DenseRetriever


class FederatedRAG(nn.Module):
    """Federated retrieval-augmented generator with a cross-graph re-ranker."""

    def __init__(self, encoder_dim=768, hidden_dim=1024, num_graphs=3):
        super().__init__()
        self.retriever = DenseRetriever(encoder_dim)
        self.re_ranker = nn.Sequential(
            nn.Linear(encoder_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.graph_embedding = nn.Embedding(num_graphs, encoder_dim)

    def forward(self, query, passages, graph_ids):
        scores = self.retriever(query, passages)
        graph_bias = self.graph_embedding(graph_ids).sum(dim=0)
        joint = torch.cat([query, graph_bias], dim=-1)
        return scores + self.re_ranker(joint)
