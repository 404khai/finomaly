6. Evaluation & Testing (For the Developer/Agent)
To ensure this project is interview-ready, we must evaluate it:
Latency Test: Measure the time from Kafka Ingest -> Output. Target: < 50ms per transaction.
Model Metrics:
Autoencoder: Measure Recall on the test set. (In fraud, missing a fraud case is worse than a false positive).
GNN: Measure ROC-AUC on identifying the synthetic fraud rings.
Ablation Study: In the README, document what happens if we turn off the GNN. Show that the Autoencoder catches stolen credit cards (point anomalies), while the GNN catches organized syndicate rings (structural anomalies). This proves business understanding.