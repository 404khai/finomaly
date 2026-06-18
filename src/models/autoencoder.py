"""
Phase 3 Autoencoder model.

A standard asymmetric PyTorch Autoencoder for point-anomaly detection.

Architecture
------------
The encoder compresses the 11-dim scaled feature vector through progressively
narrower hidden layers; the decoder mirrors (approximately) to reconstruct
the input. Because we train **only on normal transactions**, any pattern the
model cannot reconstruct well (high MSE) is a candidate anomaly.

  Input (11) -> 64 -> 32 -> bottleneck (8) -> 32 -> 64 -> Output (11)

The 95th-percentile reconstruction error on the validation set (also normal)
serves as the anomaly threshold: transactions scoring above it in Phase 4 are
flagged as potential fraud by the autoencoder branch.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(
        self,
        in_features: int = 11,
        hidden_dims: list[int] | None = None,
        bottleneck_dim: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """
        Parameters
        ----------
        in_features : int
            Number of scaled numeric features (from Phase 1 feature_columns.json).
        hidden_dims : list[int]
            Encoder hidden layer sizes (default [64, 32]).
        bottleneck_dim : int
            Size of the latent bottleneck.
        dropout : float
            Dropout applied after each hidden layer.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        # Encoder: in -> hidden1 -> hidden2 -> bottleneck
        enc_layers: list[nn.Module] = []
        prev = in_features
        for h in hidden_dims:
            enc_layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        enc_layers.append(nn.Linear(prev, bottleneck_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder: bottleneck -> hidden2 -> hidden1 -> out
        dec_layers: list[nn.Module] = [
            nn.Linear(bottleneck_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        prev = hidden_dims[-1]
        for h in reversed(hidden_dims[:-1]):
            dec_layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        dec_layers.append(nn.Linear(prev, in_features))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input. Output shape == input shape."""
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the bottleneck representation (optional utility)."""
        return self.encoder(x)


def build_autoencoder(
    in_features: int = 11,
    hidden_dims: list[int] | None = None,
    bottleneck_dim: int = 8,
    dropout: float = 0.1,
) -> Autoencoder:
    """Factory — keeps construction params in one place for train_autoencoder.py."""
    return Autoencoder(
        in_features=in_features,
        hidden_dims=hidden_dims,
        bottleneck_dim=bottleneck_dim,
        dropout=dropout,
    )
