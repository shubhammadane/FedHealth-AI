import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import (
    BATCH_SIZE, CLIENTS_DIR, CLIENT_METRICS_PATH, LEARNING_RATE, LOCAL_EPOCHS,
    RANDOM_SEED
)
from .metrics import classification_metrics
from .model import ReadmissionNet, predict_proba, train_epoch
from .utils import LOGGER, save_json, set_seed


def train_one_client(path: Path, input_dim: int, epochs=LOCAL_EPOCHS):
    df = pd.read_csv(path)
    X = df.drop(columns=["_target"]).values
    y = df["_target"].values
    model = ReadmissionNet(input_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    losses = []
    for _ in range(epochs):
        losses.append(train_epoch(model, X, y, optimizer, criterion, BATCH_SIZE))
    probs = predict_proba(model, X)
    metrics = classification_metrics(y, probs)
    return model, {"loss": float(losses[-1]), **metrics, "samples": int(len(y))}


def run_local_training():
    set_seed(RANDOM_SEED)
    first = sorted(CLIENTS_DIR.glob("hospital_*.csv"))
    if not first:
        raise FileNotFoundError("No client files found. Run `python -m src.partition` first.")
    input_dim = len(pd.read_csv(first[0]).columns) - 1
    results = {}
    for path in first:
        LOGGER.info("Local training: %s", path.name)
        _, metrics = train_one_client(path, input_dim)
        results[path.stem] = metrics
    save_json(results, CLIENT_METRICS_PATH)
    return results


if __name__ == "__main__":
    print(run_local_training())
