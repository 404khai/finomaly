# AGENTS.md: Real-Time Fraud Detection System (Fintech)

## 1. Project Overview
**Goal:** Build a real-time transaction fraud detection pipeline tailored for a high-volume mobile money/banking platform (like Moniepoint).
**Core Logic:** 
1. **Autoencoder:** Detects point-anomalies (e.g., unusually large transaction amounts or weird times of day) in real-time.
2. **Graph Neural Network (GNN):** Detects structural anomalies (e.g., fraud rings, money muling) by analyzing the network of users and merchants.
3. **Streaming:** Transactions flow through Kafka. The Autoencoder scores them instantly. The GNN provides pre-computed risk embeddings for the users involved via a fast Redis cache.
**Tech Stack:** Python, PyTorch, PyTorch Geometric (PyG), Kafka, FastAPI, Redis, Docker.

## 2. System Architecture
- **Data Source:** Synthetic/Kaggle mobile money dataset (PaySim).
- **Ingestion:** Python Producer pushes JSON transactions to a Kafka topic (`tx-stream`).
- **Stream Processor (Consumer):** Consumes from Kafka. Extracts features.
- **Inference Engine:** 
  - Passes transaction features to the **Autoencoder** (calculates reconstruction error).
  - Looks up `sender_id` and `receiver_id` in **Redis** to fetch their **GNN Node Embeddings** (risk scores).
  - Combines scores to output a final `fraud_probability`.
- **Output:** Writes the flagged transaction to an `alerts` Kafka topic or a PostgreSQL DB.
- **API Layer:** FastAPI app to simulate transactions, query user risk scores, and serve a basic dashboard.

## 3. Data Sources
Since real-world fintech graph data is highly confidential, we will use and augment public datasets:
1. **PaySim (Kaggle):** Mobile money simulation. (Use for tabular Autoencoder training).
2. **Synthetic Graph Generation:** We will write a script using `NetworkX` to inject "fraud rings" (dense clusters of high-velocity circular transactions) into the PaySim data to train the GNN.

## 4. Folder Structure
```text
finomaly/
├── AGENTS.md               # This file
├── docker-compose.yml      # Kafka, Zookeeper, Redis setup
├── requirements.txt        # Python dependencies
├── data/
│   ├── raw/                # Original CSVs
│   └── processed/          # PyTorch tensors, Graph .pt files
├── src/
│   ├── data_engineering/   # Scripts to generate graphs and preprocess
│   ├── models/             # PyTorch Autoencoder and PyG GraphSAGE models
│   ├── streaming/          # Kafka producers and consumers
│   └── api/                # FastAPI application
├── scripts/                # Bash scripts for starting services
└── notebooks/              # Jupyter notebooks for EDA and training