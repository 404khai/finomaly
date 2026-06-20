## Phase 5: FastAPI Integration & "Explainability"
**Objective:** Expose the system to the web and add Fintech-grade explainability.
Action Items:
1. Write src/api/main.py (FastAPI):
- POST /predict: Accepts a raw JSON transaction, pushes it to Kafka, and synchronously waits for the consumer to return the fraud score.
- GET /user/{user_id}/risk: Returns the user's GNN risk score and their recent transaction velocity.
- GET /explain/{transaction_id}: Returns why it was flagged (e.g., "Flagged by Autoencoder: Amount is 5x standard deviation", "Flagged by GNN: Receiver is part of a known high-risk cluster").

2. Add Swagger UI documentation via FastAPI.

### AI Agent Prompt 5: "Phase 5. Wrap our streaming logic in a FastAPI application in src/api/main.py. Create endpoints to simulate a transaction, query a user's risk profile, and provide an 'explainability' endpoint that tells us exactly which model (Autoencoder vs GNN) triggered the alert and why."