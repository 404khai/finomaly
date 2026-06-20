"""
Phase 5 — FastAPI fraud-detection API.

Exposes three endpoints:

  POST /predict
      Accept a single transaction as JSON, score it in real-time with the
      Autoencoder + GNN (via Redis), and return the combined fraud probability
      with a flag decision. Also publishes the result to the ``transactions``
      Kafka topic for the streaming pipeline.

  GET  /user/{user_id}/risk
      Return a user's GNN risk profile: their embedding norm, whether they sit
      in a high-risk cluster, and a qualitative risk band.

  GET  /explain/{transaction_id}
      Re-score a stored transaction and return a **human-readable explanation**
      of which model branch triggered the alert and why (e.g. "Amount is 5x
      the standard deviation of the normal population", "Receiver belongs to a
      known fraud ring cluster").

Start with::

    uvicorn src.api.main:app --reload --port 8000
    # or:
    python -m uvicorn src.api.main:app --reload --port 8000
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import redis
import torch
from fastapi import FastAPI, HTTPException
from kafka import KafkaProducer
from pydantic import BaseModel, Field

from src.models.autoencoder import build_autoencoder
from src.streaming.consumer import EmbeddingRiskHead, EmbeddingCache, _extract_features

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parents[2]
DATA_DIR = _project_root / "data" / "processed"

KAFKA_BOOTSTRAP = "localhost:19092"
TRANSACTIONS_TOPIC = "transactions"
REDIS_HOST, REDIS_PORT, REDIS_DB = "localhost", 6379, 0
ALERT_THRESHOLD = 0.5
AE_WEIGHT, GNN_WEIGHT = 0.5, 0.5
EMB_DIM = 64

# AE threshold stats from Phase 3 training (used for explainers).
AE_THRESHOLD_FILE = DATA_DIR / "ae" / "anomaly_threshold.json"

# In-memory alert store for the /explain endpoint (keyed by tx_id).
# In production this would be a database; for the demo we keep a bounded dict.
_alerts_store: dict[str, dict] = {}
_ALERT_STORE_MAX = 50_000

# ---------------------------------------------------------------------------
# Shared state (set during lifespan)
# ---------------------------------------------------------------------------
ae_model: Any = None
ae_threshold: float = 0.0
ae_val_stats: dict = {}
scaler: Any = None
risk_head: Any = None
cache: EmbeddingCache | None = None
kafka_producer: KafkaProducer | None = None
device: torch.device = torch.device("cpu")

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class TransactionIn(BaseModel):
    """Raw transaction — mirrors the PaySim schema expected by the producer."""
    step: int = Field(..., ge=0, description="1-hour timestep in the simulation")
    type: str = Field(..., description="Transaction type: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER")
    amount: float = Field(..., gt=0, description="Transaction amount")
    nameOrig: str = Field(..., description="Sender account name (e.g. C123456789)")
    oldbalanceOrg: float = Field(..., ge=0, description="Sender balance before tx")
    newbalanceOrig: float = Field(..., ge=0, description="Sender balance after tx")
    nameDest: str = Field(..., description="Receiver account name (e.g. M123456789)")
    oldbalanceDest: float = Field(..., ge=0, description="Receiver balance before tx")
    newbalanceDest: float = Field(..., ge=0, description="Receiver balance after tx")


class PredictResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_flagged: bool
    ae_score: float
    gnn_score: float
    alert_threshold: float
    timestamp: float


class UserRiskProfile(BaseModel):
    user_id: str
    node_id: int | None
    risk_band: str
    embedding_norm: float | None
    risk_score: float | None
    cluster_risk: str
    message: str


class ExplainResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_flagged: bool
    ae_score: float
    gnn_score: float
    triggered_by: str
    explanation: str
    details: dict


# ---------------------------------------------------------------------------
# Scoring helpers (shared by /predict and /explain)
# ---------------------------------------------------------------------------

def _score_transaction(tx: TransactionIn) -> dict:
    """Run both AE + GNN branches on a single transaction.

    Returns a dict with ae_score, gnn_score, fraud_probability, is_flagged,
    and raw intermediate values for the explainers.
    """
    # --- Feature engineering (mirrors prepare_data.py) ---
    features = _extract_features(tx.model_dump())
    x_scaled = scaler.transform(features.reshape(1, -1)).astype(np.float32)

    # --- AE branch ---
    with torch.no_grad():
        x_t = torch.from_numpy(x_scaled).to(device)
        recon = ae_model(x_t)
        mse = float(((recon - x_t) ** 2).mean().item())
    ae_score = mse / ae_threshold if ae_threshold > 0 else mse

    # --- GNN branch via Redis ---
    # Resolve node ids from the node_id_map.
    orig_node_id = _resolve_node_id(tx.nameOrig)
    dest_node_id = _resolve_node_id(tx.nameDest)
    emb_pair = cache.pair(orig_node_id, dest_node_id)
    with torch.no_grad():
        logit = risk_head(torch.from_numpy(emb_pair).to(device))
        gnn_score = float(torch.sigmoid(logit).item())

    # --- Combined ---
    w_sum = AE_WEIGHT + GNN_WEIGHT
    fraud_prob = float((AE_WEIGHT * min(ae_score, 1.0)
                         + GNN_WEIGHT * gnn_score) / w_sum)
    is_flagged = fraud_prob >= ALERT_THRESHOLD

    return {
        "ae_score": ae_score,
        "gnn_score": gnn_score,
        "fraud_probability": fraud_prob,
        "is_flagged": is_flagged,
        "mse": mse,
        "ae_threshold": ae_threshold,
        "orig_node_id": orig_node_id,
        "dest_node_id": dest_node_id,
        "features_raw": tx.model_dump(),
        "features_scaled": x_scaled.flatten().tolist(),
    }


def _resolve_node_id(name: str) -> int:
    """Look up a node id by account name. Falls back to hashing for unseen names."""
    if _node_id_map is not None and name in _node_id_map:
        return _node_id_map[name]
    # Deterministic hash fallback for unseen accounts.
    return hash(name) % 10_000_000


def _explain_ae(tx: TransactionIn, ae_score: float, mse: float) -> str:
    """Generate a human-readable explanation for the AE branch."""
    explanations: list[str] = []
    mean_mse = ae_val_stats.get("mean_mse", 0.001)
    std_mse = ae_val_stats.get("std_mse", 0.01)
    median_mse = ae_val_stats.get("median_mse", 0.001)

    # How many standard deviations above the normal mean?
    z_score = (mse - mean_mse) / max(std_mse, 1e-9)
    if ae_score > 1.0:
        if z_score > 10:
            explanations.append(
                f"Reconstruction error is {z_score:.1f}x standard deviations "
                f"above the normal mean — highly anomalous amount/balance pattern."
            )
        else:
            explanations.append(
                f"Reconstruction error ({mse:.4f}) exceeds the 95th-percentile "
                f"threshold ({ae_threshold:.4f}) by {ae_score:.1f}x."
            )

    # Check for suspicious amount patterns.
    amount = tx.amount
    if amount > 100_000:
        explanations.append(f"Large transaction amount: ${amount:,.2f}")
    elif amount > 10_000:
        explanations.append(f"Moderate-high amount: ${amount:,.2f}")

    # Check for balance inconsistencies (error_balance_orig != 0 for fraud).
    error_orig = abs(tx.oldbalanceOrg - tx.newbalanceOrig - tx.amount)
    error_dest = abs(tx.oldbalanceDest + tx.amount - tx.newbalanceDest)
    if error_orig > 1.0:
        explanations.append(
            f"Sender balance inconsistency: expected delta of {tx.amount:,.2f} "
            f"but balances differ by an additional {error_orig:,.2f}."
        )
    if error_dest > 1.0:
        explanations.append(
            f"Receiver balance inconsistency: received {tx.amount:,.2f} but "
            f"balance only changed by {tx.newbalanceDest - tx.oldbalanceDest:,.2f}."
        )

    # Hour of day check.
    hour = tx.step % 24
    if hour < 4 or hour > 23:
        explanations.append(
            f"Transaction at unusual hour: {hour}:00 (step={tx.step % 24})."
        )

    if not explanations:
        explanations.append(
            f"AE reconstruction error is {ae_score:.2f}x the normal threshold "
            f"(MSE={mse:.4f}, threshold={ae_threshold:.4f})."
        )
    return " | ".join(explanations)


def _explain_gnn(tx: TransactionIn, gnn_score: float,
                 orig_node_id: int, dest_node_id: int) -> str:
    """Generate a human-readable explanation for the GNN branch."""
    explanations: list[str] = []

    if gnn_score > 0.9:
        explanations.append("GNN identifies this transaction as very high-risk.")
    elif gnn_score > 0.7:
        explanations.append("GNN identifies elevated structural risk.")
    elif gnn_score > 0.5:
        explanations.append("GNN signals moderate structural risk.")

    # Check embedding norms (high norm = outlier in embedding space).
    orig_key = f"user:{orig_node_id}_embedding"
    dest_key = f"user:{dest_node_id}_embedding"
    orig_emb = cache._get(orig_key)
    dest_emb = cache._get(dest_key)
    orig_norm = float(np.linalg.norm(orig_emb))
    dest_norm = float(np.linalg.norm(dest_emb))

    if orig_norm > 50:
        explanations.append(
            f"Sender embedding norm ({orig_norm:.1f}) is very high — "
            f"suggests an anomalous position in the transaction network."
        )
    elif orig_norm < 1.0 and np.count_nonzero(orig_emb) == 0:
        explanations.append("Sender has no graph history (isolated node).")

    if dest_norm > 50:
        explanations.append(
            f"Receiver embedding norm ({dest_norm:.1f}) is very high — "
            f"possible fraud-ring cluster member."
        )
    elif dest_norm < 1.0 and np.count_nonzero(dest_emb) == 0:
        explanations.append("Receiver has no graph history (isolated node).")

    # Self-loop / same-account transfers are suspicious.
    if tx.nameOrig == tx.nameDest:
        explanations.append("Sender and receiver are the same account — self-transfer.")

    if not explanations:
        explanations.append(
            f"GNN structural risk score is {gnn_score:.4f} "
            f"(sender norm={orig_norm:.1f}, receiver norm={dest_norm:.1f})."
        )
    return " | ".join(explanations)


# ---------------------------------------------------------------------------
# Lifespan — load models once at startup
# ---------------------------------------------------------------------------
_node_id_map: dict[str, int] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ae_model, ae_threshold, ae_val_stats, scaler, risk_head
    global cache, kafka_producer, device, _node_id_map

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[api] device={device}")

    # Load AE.
    print("[api] loading autoencoder ...")
    ae_ckpt = torch.load(DATA_DIR / "ae" / "ae_model.pt",
                         weights_only=True, map_location=device)
    ae_model = build_autoencoder(
        in_features=ae_ckpt["in_features"],
        hidden_dims=ae_ckpt["hidden_dims"],
        bottleneck_dim=ae_ckpt["bottleneck_dim"],
        dropout=0.0,
    ).to(device).eval()
    scaler = joblib.load(DATA_DIR / "scaler.joblib")

    thresh_data = json.loads(AE_THRESHOLD_FILE.read_text())
    ae_threshold = float(thresh_data["anomaly_threshold"])
    ae_val_stats = thresh_data.get("val_stats", {})
    print(f"[api] AE loaded (threshold={ae_threshold:.6f})")

    # Load GNN risk head.
    print("[api] loading risk head ...")
    risk_head = EmbeddingRiskHead(emb_dim=EMB_DIM).to(device).eval()
    rh_path = DATA_DIR / "gnn" / "risk_head.pt"
    if rh_path.exists():
        rh_ckpt = torch.load(rh_path, weights_only=True, map_location=device)
        risk_head.load_state_dict(rh_ckpt["state_dict"])
        print("[api] risk head loaded")
    else:
        print("[api] WARNING: no trained risk head — GNN branch uncalibrated")

    # Load node_id_map for name -> node_id resolution.
    map_path = DATA_DIR / "node_id_map.parquet"
    if map_path.exists():
        import pandas as pd
        df = pd.read_parquet(map_path, columns=["name", "node_id"])
        _node_id_map = dict(zip(df["name"].astype(str), df["node_id"].astype(int)))
        print(f"[api] node_id_map: {len(_node_id_map):,} entries")

    # Redis embedding cache.
    print(f"[api] connecting to Redis {REDIS_HOST}:{REDIS_PORT} ...")
    cache = EmbeddingCache(REDIS_HOST, REDIS_PORT, REDIS_DB, emb_dim=EMB_DIM)
    cache.client.ping()

    # Kafka producer (optional — still produces to streaming topic).
    print(f"[api] connecting to Kafka {KAFKA_BOOTSTRAP} ...")
    try:
        kafka_producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            retries=5,
            linger_ms=5,
        )
        print("[api] Kafka producer ready")
    except Exception as e:
        print(f"[api] WARNING: Kafka not available ({e}) — /predict will still score but won't publish")
        kafka_producer = None

    print("[api] ready — docs at http://localhost:8000/docs")
    yield

    if kafka_producer is not None:
        kafka_producer.close()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Finomaly Fraud Detection API",
    description=(
        "Real-time fraud detection combining a PyTorch Autoencoder (point anomalies) "
        "with a GraphSAGE GNN (structural anomalies). Exposes scoring, user risk "
        "profiles, and model explainability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictResponse)
def predict(tx: TransactionIn):
    """Score a single transaction in real-time.

    Runs the Autoencoder (reconstruction error vs 95th-pct threshold) and the
    GNN branch (sender/receiver embeddings from Redis → risk head) and returns
    the blended fraud probability. Also publishes to the ``transactions`` Kafka
    topic for the streaming consumer pipeline.
    """
    tx_id = str(uuid.uuid4())
    result = _score_transaction(tx)

    # Publish to Kafka for the streaming pipeline.
    if kafka_producer is not None:
        payload = tx.model_dump()
        payload["orig_node_id"] = result["orig_node_id"]
        payload["dest_node_id"] = result["dest_node_id"]
        payload["isFraud"] = -1  # unknown — no ground truth in API path
        payload["transaction_id"] = tx_id
        kafka_producer.send(TRANSACTIONS_TOPIC, value=payload)

    # Store for /explain.
    stored = {
        **result,
        "transaction_id": tx_id,
        "nameOrig": tx.nameOrig,
        "nameDest": tx.nameDest,
        "timestamp": time.time(),
    }
    _store_alert(tx_id, stored)

    return PredictResponse(
        transaction_id=tx_id,
        fraud_probability=round(result["fraud_probability"], 4),
        is_flagged=result["is_flagged"],
        ae_score=round(result["ae_score"], 4),
        gnn_score=round(result["gnn_score"], 4),
        alert_threshold=ALERT_THRESHOLD,
        timestamp=stored["timestamp"],
    )


@app.get("/user/{user_id}/risk", response_model=UserRiskProfile)
def user_risk(user_id: str):
    """Return a user's GNN-based risk profile.

    Looks up the user's precomputed GraphSAGE embedding in Redis and reports
    their risk band (LOW / MEDIUM / HIGH / CRITICAL) based on embedding norm
    and a qualitative assessment of their position in the transaction graph.
    """
    node_id = _resolve_node_id(user_id)
    emb = cache._get(f"user:{node_id}_embedding")
    norm = float(np.linalg.norm(emb))

    # Risk score via the risk head with a synthetic neutral partner.
    zero = np.zeros(EMB_DIM, dtype=np.float32)
    pair = np.concatenate([emb, zero]).astype(np.float32)
    with torch.no_grad():
        self_risk = float(torch.sigmoid(
            risk_head(torch.from_numpy(pair).to(device))
        ).item())

    # Risk bands based on embedding norm (proxy for structural abnormality).
    if norm < 1.0 and np.count_nonzero(emb) == 0:
        risk_band = "UNKNOWN"
        cluster_msg = "No graph history — isolated or unseen node."
    elif norm > 60:
        risk_band = "CRITICAL"
        cluster_msg = "Embedding norm is extreme — likely part of a high-risk cluster or fraud ring."
    elif norm > 30:
        risk_band = "HIGH"
        cluster_msg = "Elevated embedding norm — associated with suspicious transaction patterns."
    elif norm > 10:
        risk_band = "MEDIUM"
        cluster_msg = "Moderate structural risk — some proximity to anomalous nodes."
    else:
        risk_band = "LOW"
        cluster_msg = "Normal graph position — no structural risk indicators."

    return UserRiskProfile(
        user_id=user_id,
        node_id=node_id,
        risk_band=risk_band,
        embedding_norm=round(norm, 2) if norm > 0 else None,
        risk_score=round(self_risk, 4) if self_risk > 0 else None,
        cluster_risk=cluster_msg,
        message=f"Node {node_id} | norm={norm:.2f} | risk_score={self_risk:.4f}",
    )


@app.get("/explain/{transaction_id}", response_model=ExplainResponse)
def explain(transaction_id: str):
    """Explain why a transaction was flagged (or why it wasn't).

    Looks up the stored score and produces a human-readable explanation of
    which model branch contributed most to the decision and why.
    """
    stored = _alerts_store.get(transaction_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=(
            f"Transaction '{transaction_id}' not found. Submit it via "
            f"POST /predict first, or check the transaction_id."
        ))

    tx_data = stored["features_raw"]
    ae_score = stored["ae_score"]
    gnn_score = stored["gnn_score"]
    fraud_prob = stored["fraud_probability"]
    is_flagged = stored["is_flagged"]

    # Determine which branch triggered.
    ae_triggered = ae_score >= 1.0
    gnn_triggered = gnn_score >= ALERT_THRESHOLD
    if ae_triggered and gnn_triggered:
        triggered_by = "BOTH (Autoencoder + GNN)"
    elif ae_triggered:
        triggered_by = "Autoencoder"
    elif gnn_triggered:
        triggered_by = "GNN"
    elif is_flagged:
        triggered_by = "COMBINED (individually below threshold but blend exceeds cutoff)"
    else:
        triggered_by = "NONE"

    # Generate branch-specific explanations.
    tx_obj = TransactionIn(**tx_data)
    ae_explanation = _explain_ae(tx_obj, ae_score, stored["mse"])
    gnn_explanation = _explain_gnn(
        tx_obj, gnn_score, stored["orig_node_id"], stored["dest_node_id"],
    )

    explanation = f"Flagged: {is_flagged} | Triggered by: {triggered_by}\n\n"
    explanation += f"[Autoencoder] {ae_explanation}\n\n"
    explanation += f"[GNN] {gnn_explanation}"

    return ExplainResponse(
        transaction_id=transaction_id,
        fraud_probability=round(fraud_prob, 4),
        is_flagged=is_flagged,
        ae_score=round(ae_score, 4),
        gnn_score=round(gnn_score, 4),
        triggered_by=triggered_by,
        explanation=explanation,
        details={
            "ae_threshold": stored["ae_threshold"],
            "mse": round(stored["mse"], 6),
            "sender_node_id": stored["orig_node_id"],
            "receiver_node_id": stored["dest_node_id"],
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store_alert(tx_id: str, data: dict) -> None:
    """Store a scored transaction for later explainability (bounded dict)."""
    if len(_alerts_store) >= _ALERT_STORE_MAX:
        # Evict oldest entries (first 10%).
        keys = list(_alerts_store.keys())
        for k in keys[: _ALERT_STORE_MAX // 10]:
            _alerts_store.pop(k, None)
    _alerts_store[tx_id] = data


@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok", "timestamp": time.time()}
