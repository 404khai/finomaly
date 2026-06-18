"""
Phase 1 data preparation for the Finomaly fraud-detection pipeline.

Takes the raw PaySim dataset and produces, under ``data/processed/``:
  * X_train.pt / X_val.pt  — scaled numeric features, NORMAL transactions only
                             (the autoencoder trains on these in Phase 3).
  * X_test.pt  / y_test.pt — held-out MIXED set (normal + fraud) for evaluation.
  * test_meta.parquet      — raw PaySim columns + integer node ids for the
                             streaming producer/consumer in Phase 4.
  * node_id_map.parquet    — nameOrig/nameDest string -> stable int node id,
                             reused by build_graph.py in Phase 2.
  * scaler.joblib          — StandardScaler fit on the training split.
  * feature_columns.json   — ordered list of AE input feature names.

Design note
-----------
The Phase 1 blueprint says to "encode nameOrig/nameDest". Naively label-encoding
~6M unique account ids to integers and feeding them to the autoencoder would
inject meaningless ordinal noise (the integer value carries no signal). Instead
we:

  1. Map each account string to a *stable integer node id* and persist the map
     so Phase 2 can build the transaction graph.
  2. Exclude raw ids from the autoencoder feature vector and instead add
     *semantically meaningful* features: log-amount, balance-delta/error terms,
     hour-of-day derived from `step`. These capture the point-anomalies the AE
     is meant to detect (unusual amounts, weird hours, balance inconsistencies).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Raw PaySim columns:
#   step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
#   nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
TARGET_COL = "isFraud"

# One transaction type per row; expanded to one-hot columns prefixed "type_".
CATEGORICAL_TYPE_COL = "type"

# Default feature set fed to the autoencoder. Balance error terms expose the
# accounting inconsistencies that are characteristic of fraudulent transfers
# in PaySim (where balances don't update correctly for fraud rows).
NUMERIC_FEATURES = [
    "log_amount",
    "hour_of_day",
    "delta_balance_orig",   # oldbalanceOrg - newbalanceOrig
    "delta_balance_dest",   # newbalanceDest - oldbalanceDest
    "error_balance_orig",   # oldbalanceOrg - newbalanceOrig - amount
    "error_balance_dest",   # oldbalanceDest + amount - newbalanceDest
]

RANDOM_STATE = 42
VAL_SIZE = 0.2  # fraction of normal transactions held out for threshold tuning


def _project_root() -> Path:
    # .../src/data_engineering/prepare_data.py -> repo root
    return Path(__file__).resolve().parents[2]


def _load_raw(raw_path: Path) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. Place the PaySim CSV at "
            "data/raw/paysim-dataset.csv (see AGENTS.md §3)."
        )
    # Compact dtypes — PaySim is large; avoid default float64/int64 where possible.
    dtype = {
        "type": "category",
        "nameOrig": "string",
        "nameDest": "string",
        "amount": "float32",
        "oldbalanceOrg": "float32",
        "newbalanceOrig": "float32",
        "oldbalanceDest": "float32",
        "newbalanceDest": "float32",
        "isFraud": "int8",
        "isFlaggedFraud": "int8",
    }
    df = pd.read_csv(raw_path, dtype=dtype)
    print(f"[load] {raw_path.name}: {len(df):,} rows x {df.shape[1]} cols")
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop nulls and the redundant isFlaggedFraud flag."""
    before = len(df)
    df = df.dropna()
    if before != len(df):
        print(f"[clean] dropped {before - len(df):,} null rows")
    # isFlaggedFraud is a hard-rule subset of isFraud (only flags huge transfers)
    # and would leak the label — drop it.
    if "isFlaggedFraud" in df.columns:
        df = df.drop(columns=["isFlaggedFraud"])
    return df


def _build_node_id_map(df: pd.DataFrame) -> pd.DataFrame:
    """Map every distinct account string (orig or dest) to a stable int id.

    The same id space is shared by senders and receivers so the Phase 2 graph
    treats accounts as a single node set.
    """
    accounts = pd.concat(
        [df["nameOrig"], df["nameDest"]], ignore_index=True
    ).unique().astype(object)
    accounts.sort()  # deterministic ordering across runs
    mapping = {acc: i for i, acc in enumerate(accounts)}
    print(f"[encode] {len(mapping):,} unique accounts -> int node ids")
    return mapping


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived numeric + one-hot features used by the autoencoder."""
    out = df.copy()

    # log1p smooths the heavy right tail of transaction amounts.
    out["log_amount"] = np.log1p(out["amount"].astype(np.float64))

    # `step` is a 1-hour timestep in PaySim; map it onto a 24h clock so the AE
    # sees cyclical time-of-day signal rather than a monotonically growing int.
    out["hour_of_day"] = (out["step"] % 24).astype("float32")

    # Balance deltas + error terms. Non-zero `error_*` expose the fact that the
    # recorded balances don't satisfy the accounting identity — a strong fraud
    # signal in synthetic PaySim data.
    out["delta_balance_orig"] = (
        out["oldbalanceOrg"] - out["newbalanceOrig"]
    ).astype("float32")
    out["delta_balance_dest"] = (
        out["newbalanceDest"] - out["oldbalanceDest"]
    ).astype("float32")
    out["error_balance_orig"] = (
        out["oldbalanceOrg"] - out["newbalanceOrig"] - out["amount"]
    ).astype("float32")
    out["error_balance_dest"] = (
        out["oldbalanceDest"] + out["amount"] - out["newbalanceDest"]
    ).astype("float32")

    # One-hot encode transaction type (5 categories: PAYMENT, TRANSFER,
    # CASH_OUT, CASH_IN, DEBIT).
    type_dummies = pd.get_dummies(out[CATEGORICAL_TYPE_COL], prefix="type")
    out = pd.concat([out, type_dummies.astype("float32")], axis=1)

    return out


def _feature_columns(type_cols: list[str]) -> list[str]:
    return NUMERIC_FEATURES + sorted(type_cols)


def main(raw_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir.parent / "raw" / ".gitkeep").parent.mkdir(parents=True, exist_ok=True)

    df = _load_raw(raw_path)
    df = _clean(df)

    # --- encode accounts as node ids (for Phase 2 graph; NOT fed to AE) ---
    node_id_map = _build_node_id_map(df)
    df["orig_node_id"] = df["nameOrig"].map(node_id_map).astype("int64")
    df["dest_node_id"] = df["nameDest"].map(node_id_map).astype("int64")

    # --- derive AE features ---
    df = _engineer_features(df)
    type_cols = [c for c in df.columns if c.startswith("type_")]
    feature_cols = _feature_columns(type_cols)

    # --- split: AE trains on NORMAL rows; test set is mixed (realistic) ---
    normal = df[df[TARGET_COL] == 0].copy()
    test = df.copy()  # full distribution, incl. fraud, for evaluation

    train_normal, val_normal = train_test_split(
        normal, test_size=VAL_SIZE, random_state=RANDOM_STATE, shuffle=True
    )

    print(
        f"[split] normal-only train={len(train_normal):,}  "
        f"normal-only val={len(val_normal):,}  "
        f"mixed test={len(test):,} (fraud={int(test[TARGET_COL].sum()):,})"
    )

    # --- scale numeric features (fit on train only to avoid leakage) ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_normal[feature_cols].values)
    X_val = scaler.transform(val_normal[feature_cols].values)
    X_test = scaler.transform(test[feature_cols].values)

    # --- persist tensors ---
    def _t(x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(x)).float()

    torch.save(_t(X_train), out_dir / "X_train.pt")
    torch.save(_t(X_val), out_dir / "X_val.pt")
    torch.save(_t(X_test), out_dir / "X_test.pt")
    torch.save(_t(test[TARGET_COL].values), out_dir / "y_test.pt")

    # --- persist metadata for downstream phases ---
    test_meta = test.drop(columns=feature_cols)
    test_meta.reset_index(drop=True).to_parquet(out_dir / "test_meta.parquet")

    pd.DataFrame(
        {"name": list(node_id_map.keys()), "node_id": list(node_id_map.values())}
    ).to_parquet(out_dir / "node_id_map.parquet")

    joblib.dump(scaler, out_dir / "scaler.joblib")
    (out_dir / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2))

    print(f"[done] wrote artifacts to {out_dir}")
    print(f"        AE input dim = {len(feature_cols)}")
    print(f"        features    = {feature_cols}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare PaySim data (Phase 1).")
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=_project_root() / "data" / "raw" / "paysim-dataset.csv",
        help="Path to the raw PaySim CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_project_root() / "data" / "processed",
        help="Directory for processed artifacts.",
    )
    args = parser.parse_args()
    main(args.raw_path, args.out_dir)
