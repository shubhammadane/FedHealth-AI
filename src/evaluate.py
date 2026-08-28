import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_curve

from .config import (
    CLIENT_METRICS_PATH, FEDERATED_HISTORY_PATH, GLOBAL_MODEL_PATH,
    METRICS_PATH, RESULTS_DIR, X_TEST_PATH, Y_TEST_PATH
)
from .metrics import classification_metrics, get_confusion_matrix
from .model import ReadmissionNet, predict_proba
from .utils import LOGGER, save_json


def load_global_model(input_dim: int):
    model = ReadmissionNet(input_dim)
    checkpoint = torch.load(GLOBAL_MODEL_PATH, map_location="cpu")
    model.load_state_dict(checkpoint)
    return model


def evaluate_global_model():
    X = pd.read_csv(X_TEST_PATH).values.astype("float32")
    y = pd.read_csv(Y_TEST_PATH)["readmitted"].values
    model = load_global_model(X.shape[1])
    probs = predict_proba(model, X)
    metrics = classification_metrics(y, probs)
    cm = get_confusion_matrix(y, probs)
    fpr, tpr, thresholds = roc_curve(y, probs)

    payload = {
        "global_model": metrics,
        "confusion_matrix": cm.tolist(),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()},
    }
    save_json(payload, RESULTS_DIR / "global_evaluation.json")
    return payload


if __name__ == "__main__":
    print(evaluate_global_model())
