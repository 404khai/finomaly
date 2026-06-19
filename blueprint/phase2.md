## Phase 2: Graph Generation & GNN Training (Offline) 
**Objective:** Create a transaction graph and train a GraphSAGE model to identify fraud rings. 

### Action Items: 
1. Write src/data_engineering/build_graph.py: 
- Convert transaction data into a directed graph where Nodes = Users/Merchants, Edges = Transactions. 
- Use NetworkX to inject synthetic "fraud rings" into the graph (e.g., create 50 hubs where money cycles rapidly between 5-10 nodes). 
- Convert the NetworkX graph to a PyTorch Geometric Data object. 

2. Write src/models/gnn_model.py: 
- Implement a 2-layer GraphSAGE model using PyTorch Geometric. 
- The model should perform node classification (Normal vs. Fraud Ring Node). 

3. Write src/models/train_gnn.py to train the model and save the learned Node Embeddings (the output of the second-to-last layer) as a PyTorch tensor. 

### AI Agent Prompt 2: "Phase 2. First, write build_graph.py to convert the transaction data into a PyG graph, ensuring we synthetically inject fraud rings. Then, implement the GraphSAGE model in gnn_model.py and the training loop in train_gnn.py. Save the final node embeddings."

