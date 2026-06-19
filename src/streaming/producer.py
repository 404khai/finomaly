"""
Phase 4 Kafka producer.

Streams the PaySim test transactions one-by-one to the ``transactions`` Kafka
topic, adding a small random delay to simulate real-time arrival. Each message
is a JSON payload mirroring a raw transaction plus the integer node ids from
Phase 1 (so the consumer can look up GNN embeddings without recomputing them).

Run with::

    python -m src.streaming.producer                 # full stream
    python -m src.streaming.producer --max-rows 5000 # quick smoke test
    python -m src.streaming.producer --no-delay      # drain as fast as possible
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

TRANSACTIONS_TOPIC = "transactions"
DEFAULT_BOOTSTRAP = "localhost:19092"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _build_producer(bootstrap: str) -> KafkaProducer:
    """Connect to Redpanda/Kafka, retrying briefly while the broker comes up."""
    last_err: Exception | None = None
    for attempt in range(10):
        try:
            return KafkaProducer(
                bootstrap_servers=bootstrap,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: str(k).encode("utf-8") if k is not None else None,
                acks="all",
                retries=5,
                linger_ms=10,
                buffer_memory=32 * 1024 * 1024,
                max_block_ms=10000,
            )
        except NoBrokersAvailable as e:
            last_err = e
            print(f"[producer] broker not ready (attempt {attempt + 1}/10), retrying...")
            time.sleep(2)
    raise RuntimeError(f"Could not connect to Kafka at {bootstrap}: {last_err}")


def stream(
    meta_path: Path,
    bootstrap: str,
    topic: str,
    max_rows: int | None,
    delay: float,
    no_delay: bool,
    seed: int,
) -> int:
    print(f"[producer] loading {meta_path.name} ...")
    df = pd.read_parquet(meta_path)
    if max_rows is not None:
        df = df.head(max_rows)
    print(f"[producer] {len(df):,} rows queued -> topic '{topic}' @ {bootstrap}")

    producer = _build_producer(bootstrap)
    rng = random.Random(seed)

    sent = 0
    fraud_in_stream = 0
    t0 = time.time()
    try:
        for _, row in df.iterrows():
            # Payload: raw transaction fields + node ids (Redis lookup keys).
            payload = {
                "step": int(row["step"]),
                "type": str(row["type"]),
                "amount": float(row["amount"]),
                "nameOrig": str(row["nameOrig"]),
                "oldbalanceOrg": float(row["oldbalanceOrg"]),
                "newbalanceOrig": float(row["newbalanceOrig"]),
                "nameDest": str(row["nameDest"]),
                "oldbalanceDest": float(row["oldbalanceDest"]),
                "newbalanceDest": float(row["newbalanceDest"]),
                "isFraud": int(row["isFraud"]),  # ground truth, NOT used by model
                "orig_node_id": int(row["orig_node_id"]),
                "dest_node_id": int(row["dest_node_id"]),
            }
            # Key on sender so same-user transactions hash to one partition.
            producer.send(
                topic, key=payload["nameOrig"], value=payload,
            )
            sent += 1
            fraud_in_stream += payload["isFraud"]

            if sent % 1000 == 0:
                elapsed = time.time() - t0
                rate = sent / elapsed if elapsed > 0 else 0
                print(
                    f"[producer] sent={sent:>8,}  fraud_in_stream={fraud_in_stream}  "
                    f"rate={rate:,.0f}/s"
                )

            if not no_delay and delay > 0:
                # Small jitter simulates real-time arrival.
                time.sleep(delay * rng.uniform(0.5, 1.5))
    finally:
        producer.flush(timeout=30)
        producer.close()

    elapsed = time.time() - t0
    print(
        f"[producer] DONE  sent={sent:,}  fraud={fraud_in_stream}  "
        f"elapsed={elapsed:.1f}s  avg_rate={sent / max(elapsed, 1e-9):,.0f}/s"
    )
    return sent


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stream PaySim transactions to Kafka.")
    p.add_argument("--meta-path", type=Path,
                   default=_project_root() / "data" / "processed" / "test_meta.parquet")
    p.add_argument("--bootstrap", type=str, default=DEFAULT_BOOTSTRAP)
    p.add_argument("--topic", type=str, default=TRANSACTIONS_TOPIC)
    p.add_argument("--max-rows", type=int, default=None,
                   help="Stream only the first N rows (smoke testing).")
    p.add_argument("--delay", type=float, default=0.0,
                   help="Mean seconds of jitter between rows (0 = no sleep).")
    p.add_argument("--no-delay", action="store_true",
                   help="Disable inter-row delay entirely.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    stream(
        meta_path=args.meta_path, bootstrap=args.bootstrap, topic=args.topic,
        max_rows=args.max_rows, delay=args.delay, no_delay=args.no_delay,
        seed=args.seed,
    )
