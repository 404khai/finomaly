"""
Phase 2 graph construction.

Turns the PaySim transactions produced in Phase 1 into a single undirected
PyG ``Data`` object suitable for node classification with GraphSAGE:

  * Nodes  = accounts, indexed by the contiguous int ``node_id`` built in
             prepare_data.py (0 .. N-1). Shared sender/receiver id space.
  * Edges  = transactions, made undirected so circular money flows (the
             signature of a fraud ring) show up as dense cliques.
  * Node features = per-node aggregates of its transaction history
             (volume, degree, partner count, fraud-edge fraction, tx-type mix).
  * Labels = a node is positive (``y=1``) if it touches a real PaySim fraud
             edge OR belongs to one of the synthetically injected fraud rings.

Synthetic fraud-ring injection (AGENTS.md §3)
---------------------------------------------
Real fintech fraud-ring labels are confidential, so we *inject* them: for each
ring we pick 5-10 nodes and wire them into a dense cluster of high-velocity
circular transfers (A->B->C->...->A plus cross edges) with large amounts. These
structures are exactly what a GNN should detect that a tabular autoencoder
cannot — they are structural, not point, anomalies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

RANDOM_STATE = 42
TX_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_transactions(meta_path: Path) -> pd.DataFrame:
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Processed metadata not found at {meta_path}. Run "
            "src/data_engineering/prepare_data.py first (Phase 1)."
        )
    cols = [
        "orig_node_id", "dest_node_id", "type", "amount",
        "step", "isFraud",
    ]
    df = pd.read_parquet(meta_path, columns=cols)
    # Defensive: ensure integer node ids.
    df["orig_node_id"] = df["orig_node_id"].astype("int64")
    df["dest_node_id"] = df["dest_node_id"].astype("int64")
    print(f"[load] {len(df):,} transactions from {meta_path.name}")
    return df


def _num_nodes(df: pd.DataFrame, node_map_path: Path) -> int:
    """Total node count = size of the Phase 1 node-id space (safer than max+1)."""
    if node_map_path.exists():
        n = pd.read_parquet(node_map_path, columns=["node_id"])["node_id"].nunique()
        print(f"[nodes] {n:,} accounts from node_id_map.parquet")
        return int(n)
    n = int(max(df["orig_node_id"].max(), df["dest_node_id"].max())) + 1
    print(f"[nodes] {n:,} accounts (inferred from edge max; node map missing)")
    return n


def _build_node_features(df: pd.DataFrame, n_nodes: int) -> torch.Tensor:
    """Aggregate each node's transaction history into a compact feature vector.

    These are *structural/behavioral* summaries — what the node looks like in
    the graph — which is exactly the signal a GNN consumes. Kept small (~14
    dims) so the full 9M-node feature matrix fits comfortably in memory.
    """
    print("[features] aggregating per-node statistics ...")
    log_amt = np.log1p(df["amount"].to_numpy(dtype=np.float64))

    orig = df["orig_node_id"].to_numpy()
    dest = df["dest_node_id"].to_numpy()
    fraud = df["isFraud"].to_numpy(dtype=np.float32)

    feats = np.zeros((n_nodes, 6), dtype=np.float32)
    # volume, count, sum_log_amt, sum_fraud, in_degree, out_degree
    np.add.at(feats[:, 0], orig, df["amount"].to_numpy(dtype=np.float64))  # sent vol
    np.add.at(feats[:, 1], orig, 1)                                        # out-deg
    np.add.at(feats[:, 2], orig, log_amt)                                  # sum log amt (out)
    np.add.at(feats[:, 3], orig, fraud)                                    # fraud out
    np.add.at(feats[:, 4], dest, 1)                                        # in-deg
    np.add.at(feats[:, 5], dest, fraud)                                    # fraud in

    # Distinct partners per node (sender side + receiver side).
    partners = pd.Series(np.concatenate([orig, dest]))
    node_part = pd.Series(np.concatenate([orig, dest]))
    uniq_partners = (
        pd.DataFrame({"n": node_part, "p": partners})
        .drop_duplicates()
        .groupby("n").size()
    )

    # One-hot tx-type mean per node (sender side): mean over the node's txs.
    type_oh = np.zeros((len(df), len(TX_TYPES)), dtype=np.float32)
    type_codes = pd.Categorical(df["type"], categories=TX_TYPES).codes
    for j in range(len(TX_TYPES)):
        type_oh[:, j] = (type_codes == j).astype(np.float32)
    type_by_node = np.zeros((n_nodes, len(TX_TYPES)), dtype=np.float32)
    for j in range(len(TX_TYPES)):
        np.add.at(type_by_node[:, j], orig, type_oh[:, j])

    # Assemble + log-transform the heavy-tailed magnitude features.
    log_volume = np.log1p(feats[:, 0])
    log_count = np.log1p(feats[:, 1])
    avg_log_amt = np.where(feats[:, 1] > 0, feats[:, 2] / np.maximum(feats[:, 1], 1), 0.0)
    total_deg = feats[:, 1] + feats[:, 4]
    fraud_count = feats[:, 3] + feats[:, 5]
    fraud_frac = np.where(total_deg > 0, fraud_count / np.maximum(total_deg, 1), 0.0)
    n_uniq = np.zeros(n_nodes, dtype=np.float32)
    n_uniq[uniq_partners.index.to_numpy()] = uniq_partners.to_numpy(dtype=np.float32)
    log_partners = np.log1p(n_uniq)

    x = np.stack([
        log_volume, log_count, avg_log_amt, fraud_frac,
        feats[:, 4].astype(np.float32),   # in-degree
        feats[:, 1].astype(np.float32),   # out-degree
        log_partners,
        *type_by_node.T,                  # 5 type-share dims
    ], axis=1).astype(np.float32)

    print(f"[features] node feature matrix: {x.shape}")
    return torch.from_numpy(x)


def _edge_index_from(df: pd.DataFrame) -> torch.Tensor:
    """Coalesce duplicate edges and symmetrize (undirected) for ring detection."""
    print("[edges] coalescing + symmetrizing ...")
    # Dedup (orig,dest) pairs to keep the edge list small.
    pairs = df[["orig_node_id", "dest_node_id"]].drop_duplicates()
    ei = torch.from_numpy(pairs.to_numpy().T.astype(np.int64))
    ei = to_undirected(ei, num_nodes=None)
    print(f"[edges] {ei.shape[1]:,} undirected edges")
    return ei


def _inject_fraud_rings(
    n_nodes: int,
    real_fraud_nodes: np.ndarray,
    num_rings: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Inject dense circular fraud rings and return (edges, ring_node_ids).

    Each ring is a clique-ish cycle of 5-10 nodes wired with high-velocity
    circular transfers (A->B->...->A) plus extra cross edges.

    Node selection: 1-2 nodes per ring are "anchors" drawn from nodes already
    seen in real fraud edges (overlap makes the structure plausible), and the
    rest are FRESH random nodes with no individual fraud history. Those fresh
    ring members are only identifiable as fraudulent via their graph
    *structure*, which is exactly what a GNN should learn that a tabular
    autoencoder cannot.
    """
    print(f"[rings] injecting {num_rings} synthetic fraud rings ...")
    srcs: list[int] = []
    dsts: list[int] = []
    ring_nodes_all: set[int] = set()
    n_fresh = 0

    anchors = real_fraud_nodes if len(real_fraud_nodes) > 0 else np.arange(n_nodes)

    for _ in range(num_rings):
        size = int(rng.integers(5, 11))  # 5..10 nodes per ring
        n_anchors = int(rng.integers(1, 3))  # 1 or 2 anchors
        n_anchors = min(n_anchors, size - 1)
        anchor_members = rng.choice(anchors, size=n_anchors, replace=False)
        fresh_members = rng.choice(n_nodes, size=size - n_anchors, replace=False)
        members = np.unique(np.concatenate([anchor_members, fresh_members]))
        n_fresh += len(members) - len(anchor_members)
        if len(members) < 3:
            continue

        # Circular backbone: m0->m1->...->mK->m0
        for i in range(len(members)):
            a, b = members[i], members[(i + 1) % len(members)]
            srcs.append(int(a)); dsts.append(int(b))
        # Dense cross edges (each node connects to ~2 extra ring members).
        for u in members:
            extras = rng.choice(
                [m for m in members if m != u],
                size=min(2, len(members) - 1), replace=False,
            )
            for v in extras:
                srcs.append(int(u)); dsts.append(int(v))

        ring_nodes_all.update(int(m) for m in members)

    ring_edges = np.stack([np.array(srcs, dtype=np.int64),
                           np.array(dsts, dtype=np.int64)], axis=1)
    print(
        f"[rings] {len(ring_nodes_all)} ring nodes "
        f"(~{n_fresh} fresh / structure-only positives)"
    )
    return ring_edges, np.array(sorted(ring_nodes_all), dtype=np.int64), len(ring_nodes_all)


def _build_labels(
    df: pd.DataFrame, n_nodes: int, ring_nodes: np.ndarray,
) -> torch.Tensor:
    """y=1 for nodes in a real fraud edge or an injected ring; else 0."""
    fraud_mask = df["isFraud"].to_numpy() == 1
    real_fraud_nodes = np.unique(np.concatenate([
        df.loc[fraud_mask, "orig_node_id"].to_numpy(),
        df.loc[fraud_mask, "dest_node_id"].to_numpy(),
    ]))
    y = np.zeros(n_nodes, dtype=np.int64)
    y[real_fraud_nodes] = 1
    y[ring_nodes] = 1
    print(
        f"[labels] positive nodes: {int(y.sum()):,} "
        f"(real-fraud nodes={len(real_fraud_nodes):,}, "
        f"ring nodes={len(ring_nodes):,}) | "
        f"positive rate={y.mean()*100:.4f}%"
    )
    return torch.from_numpy(y), real_fraud_nodes


def _stratified_masks(
    y: torch.Tensor, train_frac=0.70, val_frac=0.15, seed=RANDOM_STATE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stratified train/val/test split at the node level, balanced on label."""
    y_np = y.numpy()
    rng = np.random.default_rng(seed)
    train_mask = np.zeros(len(y_np), dtype=bool)
    val_mask = np.zeros(len(y_np), dtype=bool)
    test_mask = np.zeros(len(y_np), dtype=bool)
    for label in (0, 1):
        idx = np.where(y_np == label)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(n * train_frac)
        n_va = int(n * val_frac)
        train_mask[idx[:n_tr]] = True
        val_mask[idx[n_tr:n_tr + n_va]] = True
        test_mask[idx[n_tr + n_va:]] = True
    print(
        f"[split] train={train_mask.sum():,} val={val_mask.sum():,} "
        f"test={test_mask.sum():,} "
        f"(pos: tr={int(y_np[train_mask].sum())} "
        f"va={int(y_np[val_mask].sum())} te={int(y_np[test_mask].sum())})"
    )
    return (torch.from_numpy(train_mask), torch.from_numpy(val_mask),
            torch.from_numpy(test_mask))


def main(
    meta_path: Path,
    node_map_path: Path,
    out_path: Path,
    num_rings: int,
    seed: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    df = _load_transactions(meta_path)
    n_nodes = _num_nodes(df, node_map_path)

    x = _build_node_features(df, n_nodes)
    base_ei = _edge_index_from(df)

    # Labels (need real-fraud node set for ring seeding).
    y, real_fraud_nodes = _build_labels(df, n_nodes, ring_nodes=np.array([], dtype=np.int64))
    # Rebuild labels with rings actually injected now.
    ring_edges, ring_nodes, n_ring_nodes = _inject_fraud_rings(
        n_nodes, real_fraud_nodes, num_rings, rng,
    )
    y, _ = _build_labels(df, n_nodes, ring_nodes)

    # Merge ring edges into the edge index and re-symmetrize.
    extra_ei = torch.from_numpy(ring_edges.T.astype(np.int64))
    edge_index = torch.cat([base_ei, extra_ei], dim=1)
    edge_index = to_undirected(edge_index, num_nodes=n_nodes)
    print(f"[edges] final undirected edges (with rings): {edge_index.shape[1]:,}")

    train_mask, val_mask, test_mask = _stratified_masks(y, seed=seed)

    data = Data(
        x=x,
        edge_index=edge_index.contiguous(),
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    data.num_nodes = n_nodes
    torch.save(data, out_path)

    stats = {
        "num_nodes": n_nodes,
        "num_edges": int(edge_index.shape[1]),
        "num_features": int(x.shape[1]),
        "num_positive_nodes": int(y.sum().item()),
        "positive_rate": float(y.float().mean().item()),
        "num_rings": num_rings,
        "num_ring_nodes": n_ring_nodes,
        "train_nodes": int(train_mask.sum()),
        "val_nodes": int(val_mask.sum()),
        "test_nodes": int(test_mask.sum()),
    }
    stats_path = out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"[done] graph -> {out_path}")
    print(f"        stats -> {stats_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build the PaySim transaction graph (Phase 2).")
    p.add_argument("--meta-path", type=Path,
                   default=_project_root() / "data" / "processed" / "test_meta.parquet")
    p.add_argument("--node-map-path", type=Path,
                   default=_project_root() / "data" / "processed" / "node_id_map.parquet")
    p.add_argument("--out-path", type=Path,
                   default=_project_root() / "data" / "processed" / "graph.pt")
    p.add_argument("--num-rings", type=int, default=50,
                   help="Number of synthetic fraud rings to inject.")
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = p.parse_args()
    main(args.meta_path, args.node_map_path, args.out_path,
         args.num_rings, args.seed)
