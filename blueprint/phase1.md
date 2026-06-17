## Phase 1: Environment, Docker & Data Prep
**Objective:** Set up the local infrastructure and prepare the dataset.

### Action Items:
1. Create a docker-compose.yml that spins up:
- Confluent Kafka & Zookeeper (or Redpanda for lighter weight).
- Redis (for caching GNN embeddings).

2. Write a Python script src/data_engineering/prepare_data.py that:
- Downloads or loads the PaySim dataset (df_all.csv).
- Cleans the data, drops nulls, and encodes categorical variables (step, type, nameOrig, nameDest).
- Splits data into train (normal transactions only, for Autoencoder) and test (mixed).

3. Create requirements.txt with torch, torch-geometric, kafka-python, fastapi, uvicorn, redis, pandas, scikit-learn, networkx.

``` bash
AI Agent Prompt 1: "Let's start Phase 1. Create the docker-compose.yml for Kafka and Redis. Then write src/data_engineering/prepare_data.py to download and preprocess the PaySim dataset. Ensure we isolate the 'normal' transactions for autoencoder training."