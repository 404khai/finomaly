"""
Phase 2 GraphSAGE training.

Trains the 2-layer GraphSAGE from gnn_model.py to classify nodes as normal vs.
fraud-ring. Uses **full-graph** message-passing (the graph fits in memory at
~9M nodes × 12 features ≈ 430 MB) with loss computed only on the training
mask nodes. This avoids the NeighborSampler / pyg-lib / torch-sparse
requirement that has no cp314 wheels yet.

After training, the **penultimate-layer embeddings for every node** are dumped
to ``data/processed/gnn/node_embeddings.pt`` — these are the per-user risk
vectors that Phase 3 pushes into Redis for real-time lookup.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from src.models.gnn_model import build_model

RANDOM_STATE = 42


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _metrics(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> dict:
    """ROC-AUC, PR-AUC, and recall @ ~5 % FPR on the positive class."""
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    mask_np = mask.numpy()
    truth = y[mask_np].numpy()
    pred = probs[mask_np]
    n_pos = int(truth.sum())
    n_neg = int((1 - truth).sum())
    if n_pos == 0 or n_neg == 0:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"), "recall": float("nan")}
    roc = roc_auc_score(truth, pred)
    pr = average_precision_score(truth, pred)
    neg = pred[truth == 0]
    fpr_thresh = np.quantile(neg, 0.95) if len(neg) else 0.5
    recall = float(((pred[truth == 1] >= fpr_thresh)).mean())
    return {"roc_auc": float(roc), "pr_auc": float(pr), "recall": float(recall)}


def train(
    graph_path: Path,
    out_dir: Path,
    epochs: int,
    hidden: int,
    lr: float,
    seed: int,
    device_str: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _resolve_device(device_str)
    print(f"[setup] device={device}")

    print(f"[load] graph <- {graph_path}")
    data = torch.load(graph_path, weights_only=False)
    n_feat = data.x.shape[1]
    n_train = int(data.train_mask.sum())
    n_val = int(data.val_mask.sum())
    n_pos = int(data.y[data.train_mask].sum())
    print(
        f"[load] nodes={data.num_nodes:,}  edges={data.edge_index.shape[1]:,}  "
        f"feat={n_feat}  train_nodes={n_train:,} (pos={n_pos})"
    )

    # Move to device once — full-graph forward.
    data = data.to(device)
    model = build_model(in_channels=n_feat, hidden_channels=hidden).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # pos_weight for the class imbalance (~0.18 % positive in training).
    pos = int(data.y[data.train_mask].sum())
    neg = n_train - pos
    pos_w = torch.tensor(
        [1.0, max(neg / max(pos, 1), 1.0)], device=device, dtype=torch.float32,
    )
    print(f"[setup] pos_weight={pos_w[1].item():.2f}  (pos={pos} neg={neg})")

    best_val_auc = -1.0
    best_state: dict | None = None
    patience, bad_epochs = 5, 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()

        # Full-graph message passing — loss only on training nodes.
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask],
                                weight=pos_w)
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss = float(loss.item())

        # Eval every epoch (full-graph forward is ~1-2s on CPU for this size).
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)

        tr = _metrics(logits, data.y, data.train_mask)
        va = _metrics(logits, data.y, data.val_mask)
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_roc_auc": va["roc_auc"], "val_pr_auc": va["pr_auc"],
        })
        print(
            f"[epoch {epoch:3d}] loss={train_loss:.4f}  "
            f"train_auc={tr['roc_auc']:.4f}  val_auc={va['roc_auc']:.4f}  "
            f"val_pr={va['pr_auc']:.4f}  val_rec@5%fpr={va['recall']:.4f}  "
            f"({time.time() - t0:.1f}s)"
        )

        if not np.isnan(va["roc_auc"]) and va["roc_auc"] > best_val_auc:
            best_val_auc = va["roc_auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience and best_state is not None:
                print(f"[early-stop] no val AUC improvement for {patience} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[restore] best val ROC-AUC = {best_val_auc:.4f}")

    # --- Full-graph embeddings + final metrics ---
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        emb = model.embed(data.x, data.edge_index)

    emb = emb.cpu()
    logits = logits.cpu()
    te = _metrics(logits, data.y, data.test_mask)

    emb_path = out_dir / "node_embeddings.pt"
    model_path = out_dir / "gnn_model.pt"
    torch.save(emb, emb_path)
    torch.save(
        {"state_dict": model.state_dict(),
         "in_channels": n_feat, "hidden_channels": hidden},
        model_path,
    )

    metrics = {
        "best_val_roc_auc": float(best_val_auc) if best_val_auc > 0 else None,
        "test": te,
        "config": {
            "epochs": epochs, "hidden": hidden, "lr": lr,
            "seed": seed, "device": str(device),
        },
        "embeddings_path": str(emb_path),
        "embeddings_shape": list(emb.shape),
    }
    metrics_path = out_dir / "gnn_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\n[done] embeddings -> {emb_path}  shape={tuple(emb.shape)}")
    print(f"       model      -> {model_path}")
    print(f"       metrics    -> {metrics_path}")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train GraphSAGE on the fraud graph (Phase 2).")
    p.add_argument("--graph-path", type=Path,
                   default=_project_root() / "data" / "processed" / "graph.pt")
    p.add_argument("--out-dir", type=Path,
                   default=_project_root() / "data" / "processed" / "gnn")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    args = p.parse_args()
    train(
        graph_path=args.graph_path, out_dir=args.out_dir, epochs=args.epochs,
        hidden=args.hidden, lr=args.lr, seed=args.seed, device_str=args.device,
    )
