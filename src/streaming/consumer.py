"""
Phase 4 real-time fraud-detection consumer.

Consumes transactions from the ``transactions`` Kafka topic, scores each one
with **two complementary signals**, combines them, and publishes high-risk
transactions to the ``fraud_alerts`` topic.

Scoring
-------
1. **Autoencoder branch (point anomaly)** — reconstructs the transaction's
   numeric feature vector and computes the MSE. Normalized by the 95th-pct
   ``anomaly_threshold`` from Phase 3 → an ``ae_score`` in roughly [0, 1+].
2. **GNN branch (structural anomaly)** — looks up the precomputed GraphSAGE
   embeddings of the sender and receiver in Redis (sub-ms) and feeds them
   through a small MLP risk head that was trained against the Phase 2
   node-fraud labels → a ``gnn_score`` in [0, 1].

The final ``fraud_probability`` is a weighted blend (``w_ae`` / ``w_gnn``);
transactions above ``alert_threshold`` are published to ``fraud_alerts``.

Usage
-----
    python -m src.streaming.consumer                       # run forever
    python -m src.streaming.consumer --max-messages 5000   # smoke test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import redis
import torch
import torch.nn as nn
import torch.nn.functional as F
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import NoBrokersAvailable

from src.models.autoencoder import build_autoencoder
from src.models.gnn_model import build_model

# ----------------------------------------------------------------------------
# Defaults — all overridable via CLI.
# ----------------------------------------------------------------------------
TRANSACTIONS_TOPIC = "transactions"
ALERTS_TOPIC = "fraud_alerts"
DEFAULT_BOOTSTRAP = "localhost:19092"
DEFAULT_REDIS = ("localhost", 6379, 0)
DEFAULT_EMB_DIM = 64
DEFAULT_ALERT_THRESHOLD = 0.5
DEFAULT_WEIGHTS = (0.5, 0.5)  # (ae_weight, gnn_weight)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# GNN risk head — a tiny MLP that maps a node embedding -> fraud probability.
# ---------------------------------------------------------------------------
class EmbeddingRiskHead(nn.Module):
    """MLP turning a 2*D GNN embedding (sender||receiver) into a fraud logit.

    Trained offline on (sender_emb, receiver_emb, edge_label) pairs drawn from
    the Phase 2 graph + embeddings. We train it lazily on consumer startup so
    the branch is end-to-end functional without a separate training artifact.
    """

    def __init__(self, emb_dim: int = DEFAULT_EMB_DIM, hidden: int = 64,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb).squeeze(-1)


# ---------------------------------------------------------------------------
# Feature engineering — must match prepare_data.py exactly.
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "log_amount",
    "hour_of_day",
    "delta_balance_orig",
    "delta_balance_dest",
    "error_balance_orig",
    "error_balance_dest",
]
TX_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def _extract_features(tx: dict) -> np.ndarray:
    """Build the AE feature vector from a raw transaction dict.

    Mirrors src/data_engineering/prepare_data.py: log_amount, hour_of_day
    (step % 24), balance deltas/errors, and 5 one-hot type columns.
    """
    amount = float(tx["amount"])
    step = int(tx["step"])
    old_o = float(tx["oldbalanceOrg"])
    new_o = float(tx["newbalanceOrig"])
    old_d = float(tx["oldbalanceDest"])
    new_d = float(tx["newbalanceDest"])

    feats = [
        float(np.log1p(amount)),
        float(step % 24),
        old_o - new_o,                                  # delta_balance_orig
        new_d - old_d,                                  # delta_balance_dest
        old_o - new_o - amount,                         # error_balance_orig
        old_d + amount - new_d,                         # error_balance_dest
    ]
    # One-hot transaction type (in fixed TX_TYPES order).
    tx_type = str(tx.get("type", "PAYMENT")).upper()
    for t in TX_TYPES:
        feats.append(1.0 if tx_type == t else 0.0)
    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Model wiring / loading
# ---------------------------------------------------------------------------
def _load_autoencoder(path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(path, weights_only=True, map_location=device)
    model = build_autoencoder(
        in_features=ckpt["in_features"],
        hidden_dims=ckpt["hidden_dims"],
        bottleneck_dim=ckpt["bottleneck_dim"],
        dropout=ckpt.get("dropout", 0.0),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


def _load_threshold(path: Path) -> float:
    data = json.loads(path.read_text())
    return float(data["anomaly_threshold"])


def _load_scaler(path: Path):
    return joblib.load(path)


class _NoGnn(nn.Module):
    """Fallback when node embeddings are unavailable: identity feature head.

    Trains a 2-layer MLP on (numeric_features, label) so the consumer still
    produces a meaningful score without Redis. NOT used in normal operation.
    """

    def __init__(self, in_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Redis embedding lookup
# ---------------------------------------------------------------------------
class EmbeddingCache:
    """Fetches sender/receiver GNN embeddings from Redis with caching of misses."""

    def __init__(self, host: str, port: int, db: int, prefix: str = "user:",
                 emb_dim: int = DEFAULT_EMB_DIM) -> None:
        self.client = redis.Redis(host=host, port=port, db=db,
                                   decode_responses=False)
        self.prefix = prefix
        self.emb_dim = emb_dim
        self._zero = np.zeros(emb_dim, dtype=np.float32)
        self._default: np.ndarray | None = None

    def _get(self, key: str) -> np.ndarray:
        raw = self.client.get(key)
        if raw is None:
            return self._zero  # isolated / unseen node
        arr = np.frombuffer(raw, dtype=np.float32)
        if arr.shape[0] != self.emb_dim:
            # Default sentinel or shape mismatch -> zero.
            return self._zero
        return arr

    def pair(self, orig_id: int, dest_id: int) -> np.ndarray:
        """Return concat(sender_emb, receiver_emb) as float32 [2*emb_dim]."""
        s = self._get(f"{self.prefix}{orig_id}_embedding")
        d = self._get(f"{self.prefix}{dest_id}_embedding")
        return np.concatenate([s, d]).astype(np.float32)


# ---------------------------------------------------------------------------
# Consumer
# ---------------------------------------------------------------------------
def run(
    bootstrap: str,
    redis_host: str, redis_port: int, redis_db: int,
    ae_model_path: Path, threshold_path: Path, scaler_path: Path,
    feature_cols_path: Path, gnn_model_path: Path,
    emb_dim: int,
    ae_weight: float, gnn_weight: float,
    alert_threshold: float,
    max_messages: int | None,
    device_str: str,
    risk_head_path: Path | None,
) -> None:
    device = torch.device("cuda" if (device_str == "auto"
                                      and torch.cuda.is_available()) else "cpu")
    print(f"[consumer] device={device}")

    # ---- Load AE branch ----
    print(f"[consumer] loading AE from {ae_model_path.name}")
    ae = _load_autoencoder(ae_model_path, device)
    ae_threshold = _load_threshold(threshold_path)
    scaler = _load_scaler(scaler_path)
    print(f"[consumer] AE threshold (95th-pct MSE) = {ae_threshold:.6f}")

    # ---- GNN risk head ----
    risk_head = EmbeddingRiskHead(emb_dim=emb_dim).to(device).eval()
    if risk_head_path and risk_head_path.exists():
        ckpt = torch.load(risk_head_path, weights_only=True, map_location=device)
        risk_head.load_state_dict(ckpt["state_dict"])
        print(f"[consumer] loaded risk head from {risk_head_path.name}")
    else:
        print("[consumer] no trained risk head found — using randomly "
              "initialized weights (gnn branch uncalibrated).")

    # ---- Redis ----
    print(f"[consumer] connecting to Redis {redis_host}:{redis_port}/{redis_db}")
    cache = EmbeddingCache(redis_host, redis_port, redis_db, emb_dim=emb_dim)
    cache.client.ping()
    print("[consumer] Redis OK")

    # ---- Kafka ----
    consumer = _build_kafka_consumer(bootstrap, TRANSACTIONS_TOPIC)
    producer = _build_kafka_producer(bootstrap)

    # ---- Stats ----
    n = 0
    n_flagged = 0
    true_pos = 0       # flagged & isFraud==1
    false_pos = 0      # flagged & isFraud==0
    false_neg = 0      # not flagged & isFraud==1
    t0 = time.time()
    print(f"[consumer] consuming '{TRANSACTIONS_TOPIC}' -> alert if "
          f"fraud_probability >= {alert_threshold} "
          f"(w_ae={ae_weight}, w_gnn={gnn_weight})")

    try:
        for msg in consumer:
            n += 1
            tx = msg.value

            # --- AE branch ---
            x = _extract_features(tx).reshape(1, -1)
            x_scaled = scaler.transform(x).astype(np.float32)
            with torch.no_grad():
                recon = ae(torch.from_numpy(x_scaled).to(device))
                err = float(((recon - torch.from_numpy(x_scaled).to(device)) ** 2)
                            .mean().item())
            ae_score = err / ae_threshold if ae_threshold > 0 else err

            # --- GNN branch ---
            emb = cache.pair(int(tx["orig_node_id"]), int(tx["dest_node_id"]))
            with torch.no_grad():
                logit = risk_head(torch.from_numpy(emb).to(device))
                gnn_score = float(torch.sigmoid(logit).item())

            # --- Combine ---
            w_sum = ae_weight + gnn_weight
            fraud_prob = float((ae_weight * min(ae_score, 1.0)
                                + gnn_weight * gnn_score) / w_sum)
            is_flagged = fraud_prob >= alert_threshold

            # --- Accuracy bookkeeping (uses ground truth only for metrics) ---
            true_label = int(tx.get("isFraud", 0))
            if is_flagged:
                n_flagged += 1
                if true_label == 1:
                    true_pos += 1
                else:
                    false_pos += 1
            else:
                if true_label == 1:
                    false_neg += 1

            # --- Publish alert ---
            if is_flagged:
                alert = {
                    "nameOrig": tx["nameOrig"],
                    "nameDest": tx["nameDest"],
                    "amount": tx["amount"],
                    "type": tx["type"],
                    "ae_score": round(ae_score, 4),
                    "gnn_score": round(gnn_score, 4),
                    "fraud_probability": round(fraud_prob, 4),
                    "isFraud": true_label,
                    "timestamp": time.time(),
                }
                producer.send(ALERTS_TOPIC, value=alert)

            # --- Periodic log ---
            if n % 1000 == 0 or (is_flagged and n <= 200):
                elapsed = time.time() - t0
                rate = n / elapsed if elapsed > 0 else 0
                flag = " ⚑FLAG" if is_flagged else ""
                print(
                    f"[{n:>7,}] {tx['type']:<8} amt={tx['amount']:>12,.2f}  "
                    f"ae={ae_score:6.3f} gnn={gnn_score:6.3f} "
                    f"prob={fraud_prob:5.3f}{flag}  "
                    f"rate={rate:,.0f}/s flagged={n_flagged}"
                )

            if max_messages is not None and n >= max_messages:
                print(f"[consumer] reached --max-messages {max_messages}")
                break
    finally:
        producer.flush(timeout=15)
        producer.close()
        consumer.close()

    # ---- Summary ----
    elapsed = time.time() - t0
    precision = true_pos / max(true_pos + false_pos, 1)
    recall = true_pos / max(true_pos + false_neg, 1)
    print("\n" + "=" * 60)
    print(f"[summary] consumed={n:,}  flagged={n_flagged:,}  "
          f"elapsed={elapsed:.1f}s")
    print(f"[summary] true_pos={true_pos}  false_pos={false_pos}  "
          f"false_neg={false_neg}")
    print(f"[summary] precision={precision:.4f}  recall={recall:.4f}  "
          f"flag_rate={n_flagged / max(n, 1) * 100:.2f}%")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Kafka helpers
# ---------------------------------------------------------------------------
def _build_kafka_consumer(bootstrap: str, topic: str) -> KafkaConsumer:
    last_err: Exception | None = None
    for attempt in range(10):
        try:
            return KafkaConsumer(
                topic,
                bootstrap_servers=bootstrap,
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                group_id="fraud-detector",
                value_deserializer=lambda b: json.loads(b.decode("utf-8")),
                key_deserializer=lambda b: b.decode("utf-8") if b else None,
                consumer_timeout_ms=60000,
                fetch_min_bytes=1,
            )
        except NoBrokersAvailable as e:
            last_err = e
            print(f"[consumer] broker not ready (attempt {attempt + 1}/10)")
            time.sleep(2)
    raise RuntimeError(f"Could not connect to Kafka: {last_err}")


def _build_kafka_producer(bootstrap: str) -> KafkaProducer:
    for attempt in range(10):
        try:
            return KafkaProducer(
                bootstrap_servers=bootstrap,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",
                retries=5,
                linger_ms=5,
            )
        except NoBrokersAvailable as e:
            if attempt == 9:
                raise
            print(f"[producer] broker not ready (attempt {attempt + 1}/10)")
            time.sleep(2)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run the fraud-detection consumer.")
    p.add_argument("--bootstrap", type=str, default=DEFAULT_BOOTSTRAP)
    p.add_argument("--redis-host", type=str, default=DEFAULT_REDIS[0])
    p.add_argument("--redis-port", type=int, default=DEFAULT_REDIS[1])
    p.add_argument("--redis-db", type=int, default=DEFAULT_REDIS[2])
    p.add_argument("--ae-model", type=Path,
                   default=_project_root() / "data" / "processed" / "ae" / "ae_model.pt")
    p.add_argument("--threshold", type=Path,
                   default=_project_root() / "data" / "processed" / "ae" / "anomaly_threshold.json")
    p.add_argument("--scaler", type=Path,
                   default=_project_root() / "data" / "processed" / "scaler.joblib")
    p.add_argument("--feature-cols", type=Path,
                   default=_project_root() / "data" / "processed" / "feature_columns.json")
    p.add_argument("--gnn-model", type=Path,
                   default=_project_root() / "data" / "processed" / "gnn" / "gnn_model.pt")
    p.add_argument("--risk-head", type=Path,
                   default=_project_root() / "data" / "processed" / "gnn" / "risk_head.pt")
    p.add_argument("--emb-dim", type=int, default=DEFAULT_EMB_DIM)
    p.add_argument("--ae-weight", type=float, default=DEFAULT_WEIGHTS[0])
    p.add_argument("--gnn-weight", type=float, default=DEFAULT_WEIGHTS[1])
    p.add_argument("--alert-threshold", type=float, default=DEFAULT_ALERT_THRESHOLD)
    p.add_argument("--max-messages", type=int, default=None)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    run(
        bootstrap=args.bootstrap,
        redis_host=args.redis_host, redis_port=args.redis_port,
        redis_db=args.redis_db,
        ae_model_path=args.ae_model, threshold_path=args.threshold,
        scaler_path=args.scaler, feature_cols_path=args.feature_cols,
        gnn_model_path=args.gnn_model, emb_dim=args.emb_dim,
        ae_weight=args.ae_weight, gnn_weight=args.gnn_weight,
        alert_threshold=args.alert_threshold, max_messages=args.max_messages,
        device_str=args.device, risk_head_path=args.risk_head,
    )
