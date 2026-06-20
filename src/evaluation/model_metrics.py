"""
Phase 6 — Model metrics.

Measures the headline metrics for each branch:

  * **Autoencoder** — Recall (and Precision, F1, ROC-AUC, PR-AUC) on the full
    PaySim test set (X_test / y_test from Phase 1). In fraud, recall is the
    primary metric: missing a fraud is more costly than a false positive.

  * **GNN** — ROC-AUC and PR-AUC on the synthetic fraud-ring node-classification
    task (graph.pt from Phase 2). The blueprint specifically asks for ROC-AUC
    on identifying the injected fraud rings.

Run::

    python -m src.evaluation.model_metrics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.autoencoder import build_autoencoder
from src.models.gnn_model import build_model

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Autoencoder metrics — Recall on PaySim test set
# ---------------------------------------------------------------------------
def evaluate_autoencoder(out_path: Path | None = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_test = torch.load(DATA_DIR / "X_test.pt", weights_only=True)
    y_test = torch.load(DATA_DIR / "y_test.pt", weights_only=True).numpy()
    threshold = float(json.loads(
        (DATA_DIR / "ae" / "anomaly_threshold.json").read_text()
    )["anomaly_threshold"])

    ckpt = torch.load(DATA_DIR / "ae" / "ae_model.pt",
                      weights_only=True, map_location=device)
    ae = build_autoencoder(
        in_features=ckpt["in_features"],
        hidden_dims=ckpt["hidden_dims"],
        bottleneck_dim=ckpt["bottleneck_dim"],
        dropout=0.0,
    ).to(device).eval()
    ae.load_state_dict(ckpt["state_dict"])

    print(f"[ae] scoring {len(X_test):,} test transactions ...")
    BATCH = 8192
    errors = np.empty(len(X_test), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X_test), BATCH):
            xb = X_test[i:i + BATCH].to(device)
            recon = ae(xb)
            errors[i:i + BATCH] = ((recon - xb) ** 2).mean(dim=1).cpu().numpy()

    scores = errors / threshold  # >=1 means flagged by AE
    preds = (scores >= 1.0).astype(int)
    y = y_test.astype(int)
    n_fraud = int(y.sum())
    n_normal = int((1 - y).sum())

    # The 95th-pct threshold is a fixed decision rule from training.
    recall = float(recall_score(y, preds, zero_division=0))
    precision = float(precision_score(y, preds, zero_division=0))
    f1 = float(f1_score(y, preds, zero_division=0))

    # Rank metrics — do not depend on the threshold choice.
    roc = float(roc_auc_score(y, scores))
    pr = float(average_precision_score(y, scores))

    # Also compute recall at the threshold that maximizes F1 (best-case),
    # so we can show both the deployed rule and the best achievable.
    prec_curve, rec_curve, thr_curve = precision_recall_curve(y, scores)
    f1_curve = 2 * prec_curve * rec_curve / (prec_curve + rec_curve + 1e-12)
    best_idx = int(np.nanargmax(f1_curve))
    best_thr = float(thr_curve[best_idx]) if best_idx < len(thr_curve) else 1.0
    best_recall = float(rec_curve[best_idx])
    best_precision = float(prec_curve[best_idx])
    best_f1 = float(f1_curve[best_idx])

    metrics = {
        "model": "autoencoder",
        "test_set": {
            "n_transactions": int(len(y)),
            "n_fraud": n_fraud,
            "n_normal": n_normal,
            "fraud_rate_pct": round(100 * n_fraud / len(y), 4),
        },
        "deployed_rule": {  # 95th-pct threshold from Phase 3
            "threshold_ratio": 1.0,
            "raw_mse_threshold": threshold,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "flagged_pct": round(100 * preds.mean(), 2),
        },
        "rank_metrics": {
            "roc_auc": roc,
            "pr_auc": pr,
        },
        "best_f1_rule": {
            "threshold_ratio": best_thr,
            "recall": best_recall,
            "precision": best_precision,
            "f1": best_f1,
        },
        "interpretation": (
            "Recall is the headline metric for fraud. The deployed 95th-pct "
            "rule trades precision for recall; the best-F1 row shows the "
            "optimal operating point if a higher threshold were chosen."
        ),
    }

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2))
    return metrics


# ---------------------------------------------------------------------------
# GNN metrics — ROC-AUC on synthetic fraud-ring detection
# ---------------------------------------------------------------------------
def evaluate_gnn(out_path: Path | None = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.load(DATA_DIR / "graph.pt", weights_only=False).to(device)
    ckpt = torch.load(DATA_DIR / "gnn" / "gnn_model.pt",
                      weights_only=True, map_location=device)
    model = build_model(
        in_channels=ckpt["in_channels"],
        hidden_channels=ckpt["hidden_channels"],
    ).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])

    print(f"[gnn] full-graph inference on {data.num_nodes:,} nodes ...")
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    y = data.y.cpu().numpy()

    metrics_by_split = {}
    for split_name, mask_attr in [("train", "train_mask"),
                                   ("val", "val_mask"),
                                   ("test", "test_mask")]:
        mask = getattr(data, mask_attr).cpu().numpy()
        y_s = y[mask]
        p_s = probs[mask]
        n_pos = int(y_s.sum())
        if n_pos == 0 or n_pos == len(y_s):
            metrics_by_split[split_name] = {"roc_auc": float("nan"),
                                            "pr_auc": float("nan")}
            continue
        metrics_by_split[split_name] = {
            "roc_auc": float(roc_auc_score(y_s, p_s)),
            "pr_auc": float(average_precision_score(y_s, p_s)),
            "n_nodes": int(mask.sum()),
            "n_positive": n_pos,
        }

    metrics = {
        "model": "graphSAGE",
        "task": "node classification (synthetic fraud-ring detection)",
        "graph": {
            "n_nodes": int(data.num_nodes),
            "n_edges": int(data.edge_index.shape[1]),
            "n_positive": int(y.sum()),
            "positive_rate_pct": round(100 * y.mean(), 4),
        },
        "splits": metrics_by_split,
        "test_roc_auc": metrics_by_split["test"]["roc_auc"],
        "interpretation": (
            "ROC-AUC on the test split is the headline metric per the Phase 6 "
            "blueprint. A score near 0.5 means the GNN is no better than "
            "chance at distinguishing fraud-ring nodes from normal nodes."
        ),
    }

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 6 model metrics.")
    p.add_argument("--out-dir", type=Path,
                   default=_project_root() / "data" / "processed" / "evaluation")
    args = p.parse_args()

    print("=" * 60)
    print("[1/2] Autoencoder — Recall on PaySim test set")
    print("=" * 60)
    ae_metrics = evaluate_autoencoder(args.out_dir / "autoencoder_metrics.json")
    print(json.dumps(ae_metrics, indent=2))

    print("\n" + "=" * 60)
    print("[2/2] GraphSAGE — ROC-AUC on synthetic fraud rings")
    print("=" * 60)
    gnn_metrics = evaluate_gnn(args.out_dir / "gnn_metrics.json")
    print(json.dumps(gnn_metrics, indent=2))

    print(f"\n[done] metrics written to {args.out_dir}/")


if __name__ == "__main__":
    main()
