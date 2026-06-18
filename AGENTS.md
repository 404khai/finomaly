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

## 2.1 Crucial Design Decisions (DO NOT IGNORE)
1. **High-Cardinality ID Handling:** We will **NOT** label-encode `nameOrig` and `nameDest` for the Autoencoder. Feeding 6M unique IDs as integers into a dense neural network introduces arbitrary ordinality and noise. 
   - *Autoencoder Input:* Strictly tabular/behavioral features (e.g., `log(amount)`, `oldbalanceErr`, `newbalanceErr`, `hour_of_day`, `tx_type_encoded`).
   - *GNN Input:* `nameOrig` and `nameDest` are mapped to contiguous integer node indices (0 to N) strictly for building the PyTorch Geometric edge index.
2. **Feature Engineering:** Financial amounts are highly skewed. We will apply `np.log1p()` to transaction amounts and balance differences before feeding them to the Autoencoder to ensure stable gradient descent.
3. **Streaming Broker:** We use **Redpanda** instead of Confluent Kafka + Zookeeper. It is a drop-in Kafka replacement that is lighter, faster, and does not require Zookeeper. Python code should connect to `localhost:19092`.

## 3. Data Sources
Since real-world fintech graph data is highly confidential, we will use and augment public datasets:
1. **PaySim (Kaggle):** Mobile money simulation. (Use for tabular Autoencoder training).
2. **Synthetic Graph Generation:** We will write a script using `NetworkX` to inject "fraud rings" (dense clusters of high-velocity circular transactions) into the PaySim data to train the GNN.

## 4. Folder Structure
```text
finomaly/
├── AGENTS.md               # This file
├── .gitignore               # ignore files not meant for github
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
├── blueprint/                # phase-by-phase prompts
└── notebooks/              # Jupyter notebooks for EDA and training