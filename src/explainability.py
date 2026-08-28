import json
import pandas as pd
import numpy as np

from .config import FEATURE_COLUMNS_PATH, X_TEST_PATH, GLOBAL_MODEL_PATH, RESULTS_DIR
from .model import ReadmissionNet, predict_proba
import torch


def permutation_importance_global(n_repeats=2):
    """Model-agnostic permutation importance on the held-out test set."""
    X = pd.read_csv(X_TEST_PATH)
    y = pd.read_csv(X_TEST_PATH.parent / "y_test.csv")["readmitted"].values
    feature_names = json.loads(FEATURE_COLUMNS_PATH.read_text(encoding="utf-8"))["columns"]
    model = ReadmissionNet(X.shape[1])
    model.load_state_dict(torch.load(GLOBAL_MODEL_PATH, map_location="cpu"))

    baseline = np.mean(((predict_proba(model, X.values) >= 0.5).astype(int) == y))
    rng = np.random.default_rng(42)
    rows = []
    for idx, feature in enumerate(feature_names):
        scores = []
        for _ in range(n_repeats):
            Xp = X.values.copy()
            Xp[:, idx] = rng.permutation(Xp[:, idx])
            score = np.mean(((predict_proba(model, Xp) >= 0.5).astype(int) == y))
            scores.append(baseline - score)
        rows.append({"feature": feature, "importance": float(np.mean(scores))})
    out = pd.DataFrame(rows).sort_values("importance", ascending=False)
    out.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    return out


if __name__ == "__main__":
    print(permutation_importance_global().head(20).to_string(index=False))
