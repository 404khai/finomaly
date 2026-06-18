"""
Phase 3 Redis embedding cache.

Loads the GNN node embeddings produced by ``train_gnn.py`` (Phase 2) and pushes
every node's embedding vector into a local Redis instance as:

    Key  : ``user:<node_id>_embedding``
    Value: ``numpy.float32`` bytes (raw 4-byte floats, suitable for fast reads)

For nodes that have no trained embedding (isolated or unseen), a zero-vector
of the correct dimension is stored so the Phase 4 consumer never needs to
handle a cache miss at runtime.

Prerequisites
--------------
  * ``data/processed/gnn/node_embeddings.pt`` exists (run Phase 2 first).
  * Redis is running (``scripts/start_services.sh`` or ``docker compose up -d``).

Usage
-----
    python scripts/cache_embeddings.py                          # defaults
    python scripts/cache_embeddings.py --host localhost --db 0
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import redis
import torch


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_embeddings(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(
            f"GNN embeddings not found at {path}. Run Phase 2 "
            "(src/models/train_gnn.py) first."
        )
    emb = torch.load(path, weights_only=True)
    print(f"[load] {path.name}: {tuple(emb.shape)} {emb.dtype}")
    return emb


def _node_id_map(path: Path) -> dict[str, int] | None:
    """Optional: load name -> node_id mapping so we can store by name too."""
    if not path.exists():
        return None
    import pandas as pd
    df = pd.read_parquet(path, columns=["name", "node_id"])
    print(f"[load] node_id_map: {len(df):,} entries")
    return dict(zip(df["name"].astype(str), df["node_id"].astype(int)))


def push_to_redis(
    embeddings: torch.Tensor,
    node_map: dict[str, int] | None,
    host: str,
    port: int,
    db: int,
    password: str | None,
    ttl: int | None,
    key_prefix: str,
    pipeline_size: int,
) -> int:
    """Push every node embedding into Redis. Returns the count stored."""
    r = redis.Redis(host=host, port=port, db=db, password=password,
                     decode_responses=False)
    r.ping()  # raise early if Redis is down
    print(f"[redis] connected to {host}:{port}/{db}")

    n_nodes = embeddings.shape[1]
    zero_emb = np.zeros(n_nodes, dtype=np.float32).tobytes()

    stored = 0
    t0 = time.time()
    pipe = r.pipeline(transaction=False)

    for node_id in range(embeddings.shape[0]):
        # Embedding as raw float32 bytes — compact and fast to deserialize.
        emb_bytes = embeddings[node_id].numpy().astype(np.float32).tobytes()
        key = f"{key_prefix}{node_id}_embedding"
        if ttl:
            pipe.setex(key, ttl, emb_bytes)
        else:
            pipe.set(key, emb_bytes)
        stored += 1

        if stored % pipeline_size == 0:
            pipe.execute()
            elapsed = time.time() - t0
            print(f"  [{stored:>9,} / {embeddings.shape[0]:,}] "
                  f"({stored / embeddings.shape[0] * 100:.1f}%) "
                  f"{stored / elapsed:,.0f} nodes/s")
            pipe = r.pipeline(transaction=False)

    # Flush remaining.
    pipe.execute()

    # Store a default zero-vector under a sentinel key for unseen nodes.
    r.set(f"{key_prefix}default_embedding", zero_emb)
    r.set(f"{key_prefix}embedding_dim", str(n_nodes))

    elapsed = time.time() - t0
    print(f"[done] {stored:,} embeddings cached in {elapsed:.1f}s "
          f"({stored / elapsed:,.0f} nodes/s)")
    print(f"       key prefix: {key_prefix}*")
    print(f"       embedding dim: {n_nodes}")
    print(f"       default (zero) key: {key_prefix}default_embedding")
    return stored


def main() -> None:
    p = argparse.ArgumentParser(
        description="Cache GNN node embeddings in Redis (Phase 3)."
    )
    p.add_argument("--emb-path", type=Path,
                   default=_project_root() / "data" / "processed" / "gnn" / "node_embeddings.pt")
    p.add_argument("--node-map-path", type=Path,
                   default=_project_root() / "data" / "processed" / "node_id_map.parquet")
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--port", type=int, default=6379)
    p.add_argument("--db", type=int, default=0)
    p.add_argument("--password", type=str, default=None)
    p.add_argument("--ttl", type=int, default=None,
                   help="TTL in seconds (default: no expiry).")
    p.add_argument("--key-prefix", type=str, default="user:",
                   help="Redis key prefix (e.g. 'user:' -> 'user:123_embedding').")
    p.add_argument("--pipeline-size", type=int, default=5000,
                   help="Pipeline batch size for Redis writes.")
    args = p.parse_args()

    emb = _load_embeddings(args.emb_path)
    node_map = _node_id_map(args.node_map_path)
    push_to_redis(
        embeddings=emb,
        node_map=node_map,
        host=args.host, port=args.port, db=args.db,
        password=args.password, ttl=args.ttl,
        key_prefix=args.key_prefix, pipeline_size=args.pipeline_size,
    )


if __name__ == "__main__":
    main()
