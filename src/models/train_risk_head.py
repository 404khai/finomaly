"""
Train the small EmbeddingRiskHead used by the Phase 4 consumer's GNN branch.

Consumes the Phase 2 node embeddings + graph labels and learns to map a
``concat(sender_emb, receiver_emb)`` pair to a fraud logit, using the node-level
labels as a proxy edge label (positive if EITHER endpoint is a fraud-ring node).

Run once after Phase 2, before launching the consumer::

    python -m src.models.train_risk_head
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.streaming.consumer import EmbeddingRiskHead

RANDOM_STATE = 42


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sample_edges(
    emb: torch.Tensor,
    y: torch.Tensor,
    edge_index: torch.Tensor,
    n_negative: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (sender,receiver,label) tensors.

    Positive edges = graph edges whose endpoints contain at least one positive
    node. Negative edges = random pairs of non-fraudulent nodes.
    """
    y_np = y.numpy()
    pos_idx = np.where(y_np == 1)[0]
    neg_idx = np.where(y_np == 0)[0]
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    # Edge positive if either endpoint is a positive node.
    edge_pos = (y_np[src] == 1) | (y_np[dst] == 1)

    pos_src = src[edge_pos][:min(len(edge_pos), 100000)]
    pos_dst = dst[edge_pos][:len(pos_src)]
    if len(pos_src) == 0:
        # Fall back to a few random positives if none qualify.
        pos_src = rng.choice(pos_idx, size=5000, replace=True)
        pos_dst = rng.choice(pos_idx, size=5000, replace=True)

    # Negative samples: random pairs of negative-labeled nodes.
    neg_src = rng.choice(neg_idx, size=min(n_negative, len(pos_src)), replace=True)
    neg_dst = rng.choice(neg_idx, size=len(neg_src), replace=True)

    all_src = np.concatenate([pos_src, neg_src])
    all_dst = np.concatenate([pos_dst, neg_dst])
    labels = np.concatenate([
        np.ones(len(pos_src), dtype=np.float32),
        np.zeros(len(neg_src), dtype=np.float32),
    ])
    feats = np.concatenate(
        [emb[all_src].numpy(), emb[all_dst].numpy()], axis=1
    ).astype(np.float32)
    return torch.from_numpy(feats), torch.from_numpy(labels)


def train(
    embeddings_path: Path,
    graph_path: Path,
    out_path: Path,
    emb_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] embeddings <- {embeddings_path.name}")
    emb = torch.load(embeddings_path, weights_only=True).float()
    assert emb.shape[1] == emb_dim, f"expected emb_dim={emb_dim}, got {emb.shape[1]}"
    print(f"[load] graph     <- {graph_path.name}")
    data = torch.load(graph_path, weights_only=False)
    y = data.y
    edge_index = data.edge_index
    print(f"[load] nodes={emb.shape[0]:,}  edges={edge_index.shape[1]:,}  "
          f"pos_nodes={int(y.sum()):,}")

    print("[sample] building edge dataset ...")
    X, lbl = _sample_edges(emb, y, edge_index, n_negative=50000, rng=rng)
    print(f"[sample] {len(lbl):,} edges (pos={int(lbl.sum()):,})")

    ds = TensorDataset(X, lbl)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = EmbeddingRiskHead(emb_dim=emb_dim)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    pos = float(lbl.sum())
    neg = float(len(lbl) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)])
    print(f"[train] pos_weight={pos_weight.item():.2f}")

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        tot, nb = 0.0, 0
        for xb, yb in loader:
            logit = model(xb)
            loss = F.binary_cross_entropy_with_logits(
                logit, yb, pos_weight=pos_weight)
            optim.zero_grad()
            loss.backward()
            optim.step()
            tot += float(loss.item()); nb += 1
        with torch.no_grad():
            probs = torch.sigmoid(model(X)).numpy()
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(lbl.numpy(), probs) if 0 < lbl.sum() < len(lbl) else float("nan")
        print(f"[epoch {epoch:2d}] loss={tot/nb:.4f}  roc_auc={auc:.4f}  "
              f"({time.time()-t0:.1f}s)")

    torch.save({"state_dict": model.state_dict(), "emb_dim": emb_dim}, out_path)
    print(f"[done] risk head -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train the GNN embedding risk head.")
    p.add_argument("--embeddings", type=Path,
                   default=_project_root() / "data" / "processed" / "gnn" / "node_embeddings.pt")
    p.add_argument("--graph", type=Path,
                   default=_project_root() / "data" / "processed" / "graph.pt")
    p.add_argument("--out", type=Path,
                   default=_project_root() / "data" / "processed" / "gnn" / "risk_head.pt")
    p.add_argument("--emb-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = p.parse_args()
    train(args.embeddings, args.graph, args.out, args.emb_dim,
          args.epochs, args.batch_size, args.lr, args.seed)
