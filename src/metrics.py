import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def classification_metrics(y_true, probabilities, threshold=0.5):
    pred = (np.asarray(probabilities) >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, probabilities)
    except ValueError:
        out["roc_auc"] = None
    return out


def get_confusion_matrix(y_true, probabilities, threshold=0.5):
    pred = (np.asarray(probabilities) >= threshold).astype(int)
    return confusion_matrix(y_true, pred, labels=[0, 1])
