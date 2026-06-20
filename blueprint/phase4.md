## Phase 4: Real-Time Kafka Streaming Pipeline
**Objective:** Build the real-time inference engine.

Action Items:
1. Write src/streaming/producer.py:
- Reads rows from the test dataset one by one.
- Adds a slight random delay (simulating real-time).
- Serializes to JSON and pushes to Kafka topic transactions.

2. Write src/streaming/consumer.py:
- Consumes from transactions.
- Step A: Extracts tabular features and passes them through the loaded Autoencoder to get reconstruction_error.
- Step B: Extracts nameOrig and nameDest. Queries Redis for their GNN embeddings. Passes embeddings through a simple similarity check or a small Multi-Layer Perceptron (MLP) to get a graph_risk_score.
- Step C: Combines both scores. If reconstruction_error > threshold OR graph_risk_score > threshold, flag as FRAUD.
- Pushes the result to a Kafka topic fraud_alerts.


### AI Agent Prompt 4: "Phase 4. Start Docker Compose. Write the Kafka producer.py to stream the test dataset. Then write the consumer.py that loads the Autoencoder and GNN models, queries Redis for embeddings, calculates the combined fraud score, and writes to the fraud_alerts topic. Let's test it via terminal logs."
