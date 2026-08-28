from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

try:
    import flwr as fl
except ImportError:  # allows preprocessing/UI work without Flower installed
    fl = None

from .config import BATCH_SIZE, LEARNING_RATE, LOCAL_EPOCHS
from .metrics import classification_metrics
from .model import ReadmissionNet, numpy_to_state, predict_proba, state_to_numpy, train_epoch


class HospitalClient:
    """Flower-compatible NumPyClient implementation for a simulated hospital."""

    def __init__(self, data_path: Path, input_dim: int, algorithm="fedavg", prox_mu=0.01):
        self.data_path = Path(data_path)
        self.input_dim = input_dim
        self.algorithm = algorithm
        self.prox_mu = prox_mu
        df = pd.read_csv(self.data_path)
        self.X = df.drop(columns=["_target"]).values.astype("float32")
        self.y = df["_target"].values.astype("float32")
        self.model = ReadmissionNet(input_dim)

    def get_parameters(self):
        return state_to_numpy(self.model)

    def set_parameters(self, parameters):
        numpy_to_state(self.model, parameters)

    def fit(self, parameters):
        self.set_parameters(parameters)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        reference = [p.detach().clone() for p in self.model.parameters()] if self.algorithm == "fedprox" else None
        loss = 0.0
        for _ in range(LOCAL_EPOCHS):
            loss = train_epoch(
                self.model, self.X, self.y, optimizer, criterion,
                batch_size=BATCH_SIZE, prox_state=reference,
                mu=self.prox_mu if reference is not None else 0.0
            )
        return self.get_parameters(), len(self.X), {"loss": float(loss)}

    def evaluate(self, parameters):
        self.set_parameters(parameters)
        probs = predict_proba(self.model, self.X)
        m = classification_metrics(self.y, probs)
        return float(1.0 - (m["accuracy"] or 0.0)), len(self.X), m


def make_flower_client(data_path, input_dim, algorithm="fedavg"):
    if fl is None:
        raise ImportError("Flower is required for Flower client construction. Install flwr.")
    base = HospitalClient(data_path, input_dim, algorithm)
    # Adapter to Flower's NumPyClient API.
    class _Client(fl.client.NumPyClient):
        def get_parameters(self, config):
            return base.get_parameters()
        def fit(self, parameters, config):
            return base.fit(parameters)
        def evaluate(self, parameters, config):
            return base.evaluate(parameters)
    return _Client()
