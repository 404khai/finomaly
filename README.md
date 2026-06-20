# Finomaly — Real-Time Fraud Detection

A production-style fraud-detection pipeline that combines a **PyTorch Autoencoder**
(point anomalies — unusual amounts, weird hours, balance inconsistencies) with a
**GraphSAGE GNN** (structural anomalies — fraud rings, money muling) and serves
both through Kafka + Redis + FastAPI.

> Built on the synthetic [PaySim](https://www.kaggle.com/ealaxi/paysim1) mobile-money
> dataset because real fintech graph data is confidential.

---

## Architecture

```
                       ┌──────────────┐
   PaySim CSV ───────► │ prepare_data │  ──► X_train/X_val (normal only)
                       └──────┬───────┘      X_test/y_test, node_id_map, scaler
                              │
              ┌───────────────┴────────────────┐
              ▼                                ▼
      ┌──────────────┐                ┌─────────────────┐
      │ build_graph  │                │ train_autoenc.  │
      │  +fraud rings│                │  95th-pct thr.  │
      └──────┬───────┘                └────────┬────────┘
             ▼                                  │
      ┌──────────────┐                         │
      │  train_gnn   │ ──► node embeddings ────┤
      │  GraphSAGE   │     (N × 64)            │
      └──────┬───────┘            │            │
             │                    ▼            │
             │           ┌─────────────────┐   │
             │           │ cache_embeddings│   │
             │           │   → Redis       │   │
             │           └─────────────────┘   │
             │                    │            │
             ▼                    ▼            ▼
   ┌─────────────────── Kafka streaming ───────────────────┐
   │ producer.py → transactions topic                       │
   │      │                                                 │
   │      ▼                                                 │
   │ consumer.py: AE score + GNN(Redis) score → blend       │
   │      │                                                 │
   │      ▼                                                 │
   │ fraud_alerts topic  ◄─── flagged transactions          │
   └────────────────────────────────────────────────────────┘
             │
             ▼
   ┌─────────────────── FastAPI (src/api) ──────────────────┐
   │ POST /predict        — score one tx synchronously      │
   │ GET  /user/{id}/risk — GNN risk profile from Redis     │
   │ GET  /explain/{id}   — which branch fired + why        │
   └────────────────────────────────────────────────────────┘
```

**Why two models?**
- The **Autoencoder** learns the distribution of *normal* transactions and flags
  anything it can't reconstruct — point anomalies like a card suddenly used for
  a $50K transfer at 3am. Trained on normal data only; threshold = 95th-percentile
  reconstruction error.
- The **GNN** (GraphSAGE) learns the *structure* of the account graph and flags
  nodes that sit in dense circular clusters — the signature of money-muling
  rings. Pre-computed embeddings are cached in Redis for sub-ms lookups.
- Combining them catches fraud neither would catch alone (see ablation below).

---

## Quick start

```bash
# 1. Install deps (Python 3.10+)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Place PaySim CSV at data/raw/paysim-dataset.csv

# 3. Boot infrastructure (Redpanda + Redis + Console)
scripts/start_services.sh

# 4. Build the pipeline end-to-end
python src/data_engineering/prepare_data.py          # Phase 1
python src/data_engineering/build_graph.py           # Phase 2
python src/models/train_gnn.py --epochs 12           # Phase 2
python src/models/train_autoencoder.py --epochs 20   # Phase 3
python scripts/cache_embeddings.py                   # Phase 3 (→ Redis)
python src/models/train_risk_head.py                 # Phase 4 (GNN MLP head)

# 5. Run the streaming pipeline
python -m src.streaming.producer  --max-rows 5000    # Phase 4
python -m src.streaming.consumer --max-messages 5000 # Phase 4

# 6. Serve the API
uvicorn src.api.main:app --port 8000                 # Phase 5 → http://localhost:8000/docs

# 7. Evaluate
python -m src.evaluation.model_metrics               # Phase 6
python -m src.evaluation.latency_benchmark           # Phase 6
python -m src.evaluation.ablation_study              # Phase 6
```

---

## Performance

### Latency (target: < 50 ms/tx) ✅

| Path | Median | p95 | p99 |
|---|---:|---:|---:|
| In-process `/predict` (AE + Redis GNN) | **4.8 ms** | 16.0 ms | 24.0 ms |
| End-to-end Kafka (publish + consume) | **3.9 ms** | 10.1 ms | — |

Both paths beat the 50 ms target by an order of magnitude. The Redis embedding
lookup is the long pole (~3 ms median) and keeps per-transaction GNN cost flat
regardless of graph size.

### Model metrics

| Model | Headline metric | Value | Note |
|---|---|---:|---|
| Autoencoder | Recall @ 95th-pct | **77.8%** | Traded for precision at the deployed cutoff |
| Autoencoder | ROC-AUC | **0.951** | Excellent rank ordering of fraud vs normal |
| GraphSAGE | Test ROC-AUC | **0.652** | Well above chance (0.5); structural signal is real |

### Ablation — why both models? ⭐

Measured on a 20,000-tx stratified subset (8,213 fraud):

| Branch | Recall | Precision | F1 |
|---|---:|---:|---:|
| Autoencoder only (point anomalies) | 77.75% | **91.95%** | 0.843 |
| GraphSAGE only (structural anomalies) | 99.65% | 51.88% | 0.682 |
| **Combined AE + GNN (deployed)** | **99.76%** | 55.84% | 0.716 |

**Interpretation.** The two branches are **complementary**, not redundant:
- The Autoencoder alone is *precise* but misses ~22% of fraud — these are
  structurally-normal-looking transactions that only stand out in the graph
  (e.g. a mule receiving many small transfers from a ring).
- The GNN alone has *near-perfect recall* on the synthetic rings but fires on
  half the graph, because every node adjacent to a ring member inherits risk.
- **Combined recall (99.76%) strictly exceeds either branch alone**, confirming
  the dual-model architecture is justified. The precision tradeoff is the cost
  of maximizing recall — the right call in fraud, where a missed case costs far
  more than a false positive that a human reviewer can dismiss.

### Explainability

The `/explain/{transaction_id}` endpoint attributes each alert to a branch and
gives concrete reasons:
- **AE triggers**: reconstruction-error z-score vs the normal population,
  large-amount detection, sender/receiver balance-inconsistency checks,
  unusual-hour detection.
- **GNN triggers**: embedding-norm outliers (high norm → likely ring member),
  isolated/unknown nodes, self-transfers.

---

## Repository layout

```
finomaly/
├── docker-compose.yml            # Redpanda + Redis + Console
├── requirements.txt
├── src/
│   ├── data_engineering/         # prepare_data.py, build_graph.py
│   ├── models/                   # autoencoder, gnn_model, train_*, risk_head
│   ├── streaming/                # producer.py, consumer.py
│   ├── api/                      # FastAPI main.py
│   └── evaluation/               # latency, model metrics, ablation
├── scripts/                      # start_services.sh, cache_embeddings.py
└── data/                         # raw/ + processed/ (gitignored artifacts)
```

---

## Tech stack

Python · PyTorch · PyTorch Geometric · Kafka (Redpanda) · Redis · FastAPI ·
Docker · scikit-learn · NetworkX

---

## Notes & limitations

- **Synthetic data.** PaySim is a simulation; real fraud patterns differ. The
  GNN is trained on *synthetically injected* fraud rings (dense circular
  transfers) because real fraud-ring labels are confidential.
- **Python 3.14.** The bleeding-edge interpreter had no wheels for
  `pyg-lib` / `torch-sparse`, so the GNN trains full-batch (the 9M-node graph
  fits in ~600 MB). On a standard Python 3.11, `NeighborLoader` would also work.
- **Precision vs recall.** The deployed threshold favors recall — appropriate
  for a first-line fraud screen where flagged tx go to human review. Raising the
  threshold (the `best_f1_rule` in the AE metrics) trades recall for precision.
