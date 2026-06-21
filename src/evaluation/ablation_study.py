"""
Phase 6 — Ablation study.

Quantifies what each model branch contributes to the final fraud-detection
decision by scoring the PaySim test set three ways and comparing recall and
precision at a fixed alert rate:

  * **AE-only**     — flag if reconstruction error >= 95th-pct threshold.
  * **GNN-only**    — flag if the GNN risk-head score >= 0.5 (the structural
                      signal from the sender/receiver embeddings).
  * **Combined**    — the deployed blend: 0.5 * ae + 0.5 * gnn >= 0.5.

The business claim this validates: the Autoencoder is good at *point*
anomalies (a stolen card making an unusually large transfer) while the GNN
is good at *structural* anomalies (a money-muling ring). Running both
together catches strictly more fraud than either alone.

Run::

    python -m src.evaluation.ablation_study
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

from src.models.autoencoder import build_autoencoder
from src.streaming.consumer import EmbeddingRiskHead, _extract_features

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
EMB_DIM = 64


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_test_data():
    """Return (test_meta DataFrame, y array). test_meta has raw tx fields."""
    import pandas as pd
    meta = pd.read_parquet(DATA_DIR / "test_meta.parquet")
    y = meta["isFraud"].to_numpy().astype(int)
    return meta, y


def _score_all(meta, ae, scaler, threshold, risk_head, cache, device,
               batch_size: int = 2048):
    """Score every test transaction with both branches. Returns ae_scores,
    gnn_scores arrays."""
    n = len(meta)
    ae_scores = np.empty(n, dtype=np.float32)
    gnn_scores = np.empty(n, dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = meta.iloc[start:end]

        # AE branch (vectorized over the batch).
        feats = np.stack([
            _extract_features(chunk.iloc[i].to_dict())
            for i in range(len(chunk))
        ]).astype(np.float32)
        x_scaled = scaler.transform(feats).astype(np.float32)
        with torch.no_grad():
            xt = torch.from_numpy(x_scaled).to(device)
            recon = ae(xt)
            mse = ((recon - xt) ** 2).mean(dim=1).cpu().numpy()
        ae_scores[start:end] = mse / threshold if threshold > 0 else mse

        # GNN branch (Redis lookups — one pair per tx).
        emb_pairs = np.stack([
            cache.pair(int(chunk.iloc[i]["orig_node_id"]),
                       int(chunk.iloc[i]["dest_node_id"]))
            for i in range(len(chunk))
        ]).astype(np.float32)
        with torch.no_grad():
            logits = risk_head(torch.from_numpy(emb_pairs).to(device))
            gnn_scores[start:end] = torch.sigmoid(logits).cpu().numpy()

        if (end // batch_size) % 5 == 0 or end >= n:
            print(f"  [scored {end:,} / {n:,}]")

    return ae_scores, gnn_scores


def _evaluate_branch(scores: np.ndarray, y: np.ndarray, threshold: float,
                     name: str) -> dict:
    """Score a branch at a fixed decision threshold."""
    preds = (scores >= threshold).astype(int)
    return {
        "branch": name,
        "decision_threshold": float(threshold),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "flagged_pct": round(100 * preds.mean(), 2),
        "n_flagged": int(preds.sum()),
        "n_fraud": int(y.sum()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 6 ablation study.")
    p.add_argument("--max-rows", type=int, default=20000,
                   help="Cap test rows for speed (full set is 6.4M).")
    p.add_argument("--out", type=Path,
                   default=_project_root() / "data" / "processed" /
                            "evaluation" / "ablation.json")
    p.add_argument("--ae-threshold-ratio", type=float, default=1.0,
                   help="AE decision cutoff as a multiple of the 95th-pct.")
    p.add_argument("--gnn-threshold", type=float, default=0.5)
    p.add_argument("--combined-threshold", type=float, default=0.5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models.
    ae_ckpt = torch.load(DATA_DIR / "ae" / "ae_model.pt",
                         weights_only=True, map_location=device)
    ae = build_autoencoder(
        in_features=ae_ckpt["in_features"],
        hidden_dims=ae_ckpt["hidden_dims"],
        bottleneck_dim=ae_ckpt["bottleneck_dim"],
        dropout=0.0,
    ).to(device).eval()
    ae.load_state_dict(ae_ckpt["state_dict"])
    scaler = joblib.load(DATA_DIR / "scaler.joblib")
    threshold = float(json.loads(
        (DATA_DIR / "ae" / "anomaly_threshold.json").read_text()
    )["anomaly_threshold"]) * args.ae_threshold_ratio

    risk_head = EmbeddingRiskHead(emb_dim=EMB_DIM).to(device).eval()
    rh_path = DATA_DIR / "gnn" / "risk_head.pt"
    if rh_path.exists():
        risk_head.load_state_dict(
            torch.load(rh_path, weights_only=True, map_location=device)["state_dict"]
        )

    from src.streaming.consumer import EmbeddingCache
    cache = EmbeddingCache("localhost", 6379, 0, emb_dim=EMB_DIM)
    cache.client.ping()

    # Load test data.
    meta, y = _load_test_data()
    if args.max_rows and len(meta) > args.max_rows:
        # Stratified sample so we keep fraud rows in the subset.
        rng = np.random.default_rng(42)
        fraud_idx = np.where(y == 1)[0]
        normal_idx = np.where(y == 0)[0]
        rng.shuffle(normal_idx)
        n_keep = args.max_rows - len(fraud_idx)
        if n_keep > 0:
            keep = np.concatenate([fraud_idx, normal_idx[:n_keep]])
            keep.sort()
            meta = meta.iloc[keep].reset_index(drop=True)
            y = y[keep]
    print(f"[ablation] scoring {len(meta):,} test tx "
          f"({int(y.sum())} fraud)")

    print("[ablation] scoring both branches ...")
    ae_scores, gnn_scores = _score_all(
        meta, ae, scaler, threshold, risk_head, cache, device,
    )

    # Combined score = weighted blend, same as the consumer/API.
    combined_scores = 0.5 * np.minimum(ae_scores, 1.0) + 0.5 * gnn_scores

    results = {
        "n_transactions": int(len(y)),
        "n_fraud": int(y.sum()),
        "branches": {
            "ae_only": _evaluate_branch(ae_scores, y, args.ae_threshold_ratio,
                                         "Autoencoder (point anomalies)"),
            "gnn_only": _evaluate_branch(gnn_scores, y, args.gnn_threshold,
                                          "GraphSAGE GNN (structural anomalies)"),
            "combined": _evaluate_branch(combined_scores, y,
                                          args.combined_threshold,
                                          "Combined AE + GNN (deployed)"),
        },
    }

    # Headline comparison table.
    ae_r = results["branches"]["ae_only"]["recall"]
    gnn_r = results["branches"]["gnn_only"]["recall"]
    comb_r = results["branches"]["combined"]["recall"]
    print("\n" + "=" * 60)
    print(f"{'Branch':<35} {'Recall':>8} {'Precision':>10} {'F1':>6}")
    print("-" * 60)
    for k, v in results["branches"].items():
        print(f"{v['branch']:<35} {v['recall']*100:>7.2f}% "
              f"{v['precision']*100:>9.2f}% {v['f1']:>6.3f}")
    print("=" * 60)
    print(f"\n[finding] AE recall={ae_r*100:.2f}%  GNN recall={gnn_r*100:.2f}%  "
          f"Combined recall={comb_r*100:.2f}%")
    if comb_r > max(ae_r, gnn_r):
        print("[finding] Combined strictly improves recall — both branches "
              "contribute complementary signal.")
    else:
        print("[finding] One branch dominates; the other adds little at this "
              "threshold. Consider re-tuning per-branch weights.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n[done] ablation results -> {args.out}")


if __name__ == "__main__":
    main()
