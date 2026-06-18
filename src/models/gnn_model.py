"""
Phase 2 GraphSAGE model.

A 2-layer GraphSAGE classifier for node-level fraud-ring detection. Per the
blueprint (Phase 2 + Phase 3) the model must expose the **penultimate-layer
embeddings** so they can be cached in Redis and looked up at inference time.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    """Two-layer GraphSAGE.

    forward()  -> per-node logits (shape [N, num_classes]) for classification.
    embed()    -> penultimate-layer activations (shape [N, hidden_dim]); these
                  are the learned risk embeddings cached to Redis in Phase 3.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_classes: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return the second-to-last layer output (node embeddings)."""
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)   # penultimate layer activations
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.embed(x, edge_index)
        return self.classifier(h)


def build_model(
    in_channels: int,
    hidden_channels: int = 64,
    num_classes: int = 2,
    dropout: float = 0.2,
) -> GraphSAGE:
    """Factory used by train_gnn.py and any downstream loader."""
    return GraphSAGE(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_classes=num_classes,
        dropout=dropout,
    )
