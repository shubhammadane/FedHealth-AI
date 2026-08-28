# Architecture

FedHealth-AI separates raw-data processing, local training, federated aggregation, evaluation, and presentation.

1. UCI dataset is fetched and cached.
2. Identifiers are removed and categorical/numerical variables are processed.
3. A stratified 80/20 train/test split is created before client partitioning.
4. Only the training set is partitioned into five simulated clients.
5. Clients train local PyTorch networks.
6. Model parameters are aggregated with weighted FedAvg; FedProx adds a proximal term.
7. The held-out test set evaluates the global model.
8. Streamlit reads the saved artifacts.

No raw client CSV is sent to the aggregation function.
