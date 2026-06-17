Phase 3: Autoencoder Training & Redis Caching (Offline)
Objective: Train the Autoencoder for point-anomalies and push GNN embeddings to Redis for real-time lookup.
Action Items:
Write src/models/autoencoder.py:
Create a standard asymmetric PyTorch Autoencoder.
Input: Transaction tabular features. Output: Reconstructed features.
Loss function: Mean Squared Error (MSE).
Write src/models/train_autoencoder.py:
Train only on normal transactions.
Calculate the 95th percentile of reconstruction error on the validation set. This becomes our anomaly_threshold.
Write scripts/cache_embeddings.py:
Load the GNN node embeddings from Phase 2.
Connect to Redis.
Iterate through nodes and save their embeddings as Redis keys (e.g., user:123_embedding). If a node is new/unseen, set a default "zero" embedding.
AI Agent Prompt 3: "Phase 3. Implement the PyTorch Autoencoder and its training script, calculating the 95th percentile reconstruction error threshold. Next, write the Redis caching script to push the GNN node embeddings into our local Redis instance."
