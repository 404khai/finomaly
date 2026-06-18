"""
Phase 3 Autoencoder training.

Trains the autoencoder from autoencoder.py **only on normal transactions**
(X_train / X_val from Phase 1). After training the 95th-percentile
reconstruction error on the validation set is computed and saved as the
``anomaly_threshold`` used by the real-time consumer in Phase 4.

Outputs (``data/processed/ae/``)
-------------------------------
  * ae_model.pt       — full model state_dict + config
  * anomaly_threshold.json  — the 95th-pct MSE value + training summary
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.autoencoder import build_autoencoder

RANDOM_STATE = 42


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train(
    x_train_path: Path,
    x_val_path: Path,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    bottleneck: int,
    hidden_dims: list[int] | None,
    dropout: float,
    seed: int,
    device_str: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _resolve_device(device_str)
    print(f"[setup] device={device}")

    # --- Load Phase 1 tensors (normal transactions only) ---
    X_train = torch.load(x_train_path, weights_only=True)
    X_val = torch.load(x_val_path, weights_only=True)
    in_features = X_train.shape[1]
    print(
        f"[load] X_train {tuple(X_train.shape)}  X_val {tuple(X_val.shape)}  "
        f"in_features={in_features}"
    )

    train_loader = DataLoader(
        TensorDataset(X_train), batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(X_val), batch_size=batch_size, shuffle=False,
        num_workers=0,
    )

    model = build_autoencoder(
        in_features=in_features,
        hidden_dims=hidden_dims,
        bottleneck_dim=bottleneck,
        dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params:,}  arch={in_features}→{hidden_dims}→{bottleneck}")

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=3,
    )

    best_val_loss = float("inf")
    best_state: dict | None = None
    patience, bad_epochs = 7, 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        t0 = time.time()
        train_loss_sum = 0.0
        n_batches = 0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            recon = model(batch_x)
            loss = torch.nn.functional.mse_loss(recon, batch_x)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss_sum += float(loss.item())
            n_batches += 1
        train_loss = train_loss_sum / max(n_batches, 1)

        # ---- val (MSE) ----
        model.eval()
        val_loss_sum = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)
                recon = model(batch_x)
                val_loss_sum += float(torch.nn.functional.mse_loss(recon, batch_x).item())
                n_val_batches += 1
        val_loss = val_loss_sum / max(n_val_batches, 1)
        scheduler.step(val_loss)

        current_lr = optim.param_groups[0]["lr"]
        history.append({
            "epoch": epoch, "train_mse": train_loss,
            "val_mse": val_loss, "lr": current_lr,
        })
        print(
            f"[epoch {epoch:3d}] train_mse={train_loss:.6f}  "
            f"val_mse={val_loss:.6f}  lr={current_lr:.2e}  "
            f"({time.time() - t0:.1f}s)"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience and best_state is not None:
                print(f"[early-stop] no val MSE improvement for {patience} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[restore] best val MSE = {best_val_loss:.6f}")

    # --- 95th-percentile reconstruction error on validation set ---
    print("[threshold] computing per-sample MSE on validation set ...")
    model.eval()
    all_errors: list[float] = []
    with torch.no_grad():
        for (batch_x,) in val_loader:
            batch_x = batch_x.to(device)
            recon = model(batch_x)
            # MSE per sample (averaged over features).
            errors = ((recon - batch_x) ** 2).mean(dim=1)
            all_errors.extend(errors.cpu().tolist())

    all_errors_arr = np.array(all_errors, dtype=np.float64)
    threshold = float(np.percentile(all_errors_arr, 95))
    print(
        f"[threshold] 95th-pct reconstruction error = {threshold:.6f}  "
        f"(mean={all_errors_arr.mean():.6f}, "
        f"median={np.median(all_errors_arr):.6f}, "
        f"max={all_errors_arr.max():.6f})"
    )

    # --- persist ---
    model_path = out_dir / "ae_model.pt"
    threshold_path = out_dir / "anomaly_threshold.json"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_features": in_features,
            "hidden_dims": hidden_dims,
            "bottleneck_dim": bottleneck,
            "dropout": dropout,
        },
        model_path,
    )

    threshold_data = {
        "anomaly_threshold": threshold,
        "percentile": 95,
        "metric": "mse",
        "description": (
            "Transactions with per-sample MSE above this value are flagged as "
            "anomalous by the autoencoder branch."
        ),
        "val_stats": {
            "mean_mse": float(all_errors_arr.mean()),
            "median_mse": float(np.median(all_errors_arr)),
            "max_mse": float(all_errors_arr.max()),
            "std_mse": float(all_errors_arr.std()),
        },
        "training": {
            "best_val_mse": float(best_val_loss),
            "epochs_trained": len(history),
            "config": {
                "lr": lr, "batch_size": batch_size,
                "hidden_dims": hidden_dims, "bottleneck_dim": bottleneck,
                "dropout": dropout, "seed": seed,
                "device": str(device),
            },
        },
    }
    threshold_path.write_text(json.dumps(threshold_data, indent=2))

    print(f"\n[done] model     -> {model_path}")
    print(f"       threshold -> {threshold_path}  (95th-pct MSE = {threshold:.6f})")
    print(json.dumps(threshold_data, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train the autoencoder (Phase 3).")
    p.add_argument("--x-train", type=Path,
                   default=_project_root() / "data" / "processed" / "X_train.pt")
    p.add_argument("--x-val", type=Path,
                   default=_project_root() / "data" / "processed" / "X_val.pt")
    p.add_argument("--out-dir", type=Path,
                   default=_project_root() / "data" / "processed" / "ae")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bottleneck", type=int, default=8)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 32])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    args = p.parse_args()
    train(
        x_train_path=args.x_train, x_val_path=args.x_val,
        out_dir=args.out_dir, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr,
        bottleneck=args.bottleneck, hidden_dims=args.hidden_dims,
        dropout=args.dropout, seed=args.seed, device_str=args.device,
    )
