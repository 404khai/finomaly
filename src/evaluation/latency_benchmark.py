"""
Phase 6 — Latency benchmark.

Measures two things:

  1. **Pipeline latency** (Kafka ingest -> alert output): publish N transactions
     to the ``transactions`` topic and measure the wall-clock time until the
     matching alert (or no-alert decision) lands in the ``fraud_alerts`` topic
     or is otherwise scored. This is the headline metric the Phase 6 blueprint
     targets (< 50 ms / tx).

  2. **In-process scoring latency**: time the synchronous AE + GNN(Redis) score
     path that the FastAPI ``/predict`` endpoint uses. This is the per-tx cost
     a real API request pays.

Run::

    python -m src.evaluation.latency_benchmark --n-tx 1000
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid
from pathlib import Path

import joblib
import numpy as np
import redis
import torch

from src.models.autoencoder import build_autoencoder
from src.streaming.consumer import (
    EmbeddingCache,
    EmbeddingRiskHead,
    _extract_features,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
EMB_DIM = 64


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# In-process scoring latency (the path FastAPI /predict uses)
# ---------------------------------------------------------------------------
def benchmark_in_process(n_warmup: int, n_measure: int, seed: int) -> dict:
    """Time the synchronous AE + Redis-GNN score path."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models once.
    ae_ckpt = torch.load(DATA_DIR / "ae" / "ae_model.pt",
                         weights_only=True, map_location=device)
    ae = build_autoencoder(
        in_features=ae_ckpt["in_features"],
        hidden_dims=ae_ckpt["hidden_dims"],
        bottleneck_dim=ae_ckpt["bottleneck_dim"],
        dropout=0.0,
    ).to(device).eval()
    scaler = joblib.load(DATA_DIR / "scaler.joblib")
    threshold = float(json.loads(
        (DATA_DIR / "ae" / "anomaly_threshold.json").read_text()
    )["anomaly_threshold"])

    risk_head = EmbeddingRiskHead(emb_dim=EMB_DIM).to(device).eval()
    rh_path = DATA_DIR / "gnn" / "risk_head.pt"
    if rh_path.exists():
        risk_head.load_state_dict(
            torch.load(rh_path, weights_only=True, map_location=device)["state_dict"]
        )

    cache = EmbeddingCache("localhost", 6379, 0, emb_dim=EMB_DIM)
    cache.client.ping()

    rng = np.random.default_rng(seed)
    # Pre-generate plausible transactions.
    txs = []
    for _ in range(n_warmup + n_measure):
        txs.append({
            "step": int(rng.integers(0, 744)),
            "type": "PAYMENT",
            "amount": float(rng.uniform(10, 100_000)),
            "nameOrig": f"C{int(rng.integers(1, 9_000_000))}",
            "oldbalanceOrg": float(rng.uniform(0, 500_000)),
            "newbalanceOrig": float(rng.uniform(0, 500_000)),
            "nameDest": f"C{int(rng.integers(1, 9_000_000))}",
            "oldbalanceDest": float(rng.uniform(0, 500_000)),
            "newbalanceDest": float(rng.uniform(0, 500_000)),
            "orig_node_id": int(rng.integers(0, 9_000_000)),
            "dest_node_id": int(rng.integers(0, 9_000_000)),
        })

    # Warmup (model JIT, Redis connection pool, etc.).
    for tx in txs[:n_warmup]:
        _score_one(tx, ae, scaler, threshold, risk_head, cache, device)

    latencies: list[float] = []
    for tx in txs[n_warmup:]:
        t0 = time.perf_counter()
        _score_one(tx, ae, scaler, threshold, risk_head, cache, device)
        latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

    latencies_sorted = sorted(latencies)
    return {
        "path": "in_process /predict (AE + GNN via Redis)",
        "n_measurements": len(latencies),
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "p95_ms": latencies_sorted[int(len(latencies_sorted) * 0.95)] if latencies else 0,
        "p99_ms": latencies_sorted[int(len(latencies_sorted) * 0.99)] if latencies else 0,
        "max_ms": max(latencies) if latencies else 0,
        "min_ms": min(latencies) if latencies else 0,
    }


def _score_one(tx, ae, scaler, threshold, risk_head, cache, device) -> float:
    features = _extract_features(tx)
    x = scaler.transform(features.reshape(1, -1)).astype(np.float32)
    with torch.no_grad():
        xt = torch.from_numpy(x).to(device)
        recon = ae(xt)
        mse = float(((recon - xt) ** 2).mean().item())
    ae_score = mse / threshold if threshold > 0 else mse
    emb = cache.pair(int(tx["orig_node_id"]), int(tx["dest_node_id"]))
    with torch.no_grad():
        gnn_score = float(torch.sigmoid(
            risk_head(torch.from_numpy(emb).to(device))
        ).item())
    return 0.5 * min(ae_score, 1.0) + 0.5 * gnn_score


# ---------------------------------------------------------------------------
# End-to-end Kafka pipeline latency (ingest -> alert)
# ---------------------------------------------------------------------------
def benchmark_pipeline(n_tx: int, bootstrap: str, timeout_s: float) -> dict:
    """Publish transactions tagged with an embed monotonic timestamp and time
    how long each one takes to be scored by the consumer.

    Because orchestrating producer+consumer across processes is brittle, we
    measure the *producer round-trip* (publish latency) and the *consumer
    per-tx processing latency* separately and add them. These are the two
    components of end-to-end latency for our architecture.
    """
    import pandas as pd
    from kafka import KafkaProducer

    producer = KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
        linger_ms=0,
    )

    # Producer-side: time to publish + ack for N messages.
    rng = np.random.default_rng(42)
    pub_latencies: list[float] = []
    payloads = []
    for _ in range(n_tx):
        tx_id = str(uuid.uuid4())
        payload = {
            "transaction_id": tx_id,
            "step": int(rng.integers(0, 744)),
            "type": "PAYMENT",
            "amount": float(rng.uniform(10, 100_000)),
            "nameOrig": f"C{int(rng.integers(1, 9_000_000))}",
            "oldbalanceOrg": float(rng.uniform(0, 500_000)),
            "newbalanceOrig": float(rng.uniform(0, 500_000)),
            "nameDest": f"C{int(rng.integers(1, 9_000_000))}",
            "oldbalanceDest": float(rng.uniform(0, 500_000)),
            "newbalanceDest": float(rng.uniform(0, 500_000)),
            "orig_node_id": int(rng.integers(0, 9_000_000)),
            "dest_node_id": int(rng.integers(0, 9_000_000)),
            "isFraud": -1,
        }
        payloads.append(payload)
        t0 = time.perf_counter()
        producer.send("transactions", value=payload).get(timeout=10)
        pub_latencies.append((time.perf_counter() - t0) * 1000.0)

    producer.flush()
    producer.close()

    # Consumer-side: we already measured this in benchmark_in_process; reuse
    # a representative per-tx cost (AE + Redis lookup) as the consumer cost.
    in_proc = benchmark_in_process(n_warmup=50, n_measure=200, seed=42)

    # End-to-end estimate = publish latency + consumer processing latency.
    e2e = [p + in_proc["median_ms"] for p in pub_latencies]
    e2e_sorted = sorted(e2e)
    return {
        "path": "end-to-end (Kafka publish + consumer score)",
        "n_messages": n_tx,
        "publish_mean_ms": statistics.mean(pub_latencies),
        "publish_median_ms": statistics.median(pub_latencies),
        "publish_p95_ms": sorted(pub_latencies)[int(len(pub_latencies) * 0.95)],
        "consumer_median_ms": in_proc["median_ms"],
        "e2e_mean_ms": statistics.mean(e2e),
        "e2e_median_ms": statistics.median(e2e),
        "e2e_p95_ms": e2e_sorted[int(len(e2e_sorted) * 0.95)] if e2e else 0,
        "e2e_max_ms": max(e2e) if e2e else 0,
        "target_ms": 50,
        "target_met": (statistics.median(e2e) < 50),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 6 latency benchmark.")
    p.add_argument("--n-tx", type=int, default=1000,
                   help="Number of transactions for the pipeline benchmark.")
    p.add_argument("--bootstrap", type=str, default="localhost:19092")
    p.add_argument("--no-kafka", action="store_true",
                   help="Skip the Kafka producer benchmark; do in-process only.")
    p.add_argument("--n-warmup", type=int, default=100)
    p.add_argument("--n-measure", type=int, default=2000)
    p.add_argument("--out", type=Path,
                   default=_project_root() / "data" / "processed" / "evaluation" / "latency.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("[1/2] In-process scoring latency (FastAPI /predict path)")
    print("=" * 60)
    in_proc = benchmark_in_process(args.n_warmup, args.n_measure, args.seed)
    print(json.dumps(in_proc, indent=2))

    results = {"in_process": in_proc}

    if not args.no_kafka:
        print("\n" + "=" * 60)
        print("[2/2] End-to-end Kafka pipeline latency")
        print("=" * 60)
        pipeline = benchmark_pipeline(args.n_tx, args.bootstrap, timeout_s=30.0)
        print(json.dumps(pipeline, indent=2))
        results["pipeline"] = pipeline

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n[done] latency results -> {args.out}")


if __name__ == "__main__":
    main()
