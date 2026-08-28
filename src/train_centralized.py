import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .config import (
    CENTRALIZED_MODEL_PATH, METRICS_PATH, X_TEST_PATH, X_TRAIN_PATH,
    Y_TEST_PATH, Y_TRAIN_PATH
)
from .metrics import classification_metrics
from .utils import LOGGER, save_json


def train_centralized():
    X_train = pd.read_csv(X_TRAIN_PATH).values
    X_test = pd.read_csv(X_TEST_PATH).values
    y_train = pd.read_csv(Y_TRAIN_PATH)["readmitted"].values
    y_test = pd.read_csv(Y_TEST_PATH)["readmitted"].values

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=250, random_state=42, n_jobs=-1, class_weight="balanced_subsample"
        ),
    }

    all_metrics = {}
    fitted = {}
    for name, model in models.items():
        LOGGER.info("Training centralized %s", name)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        all_metrics[name] = classification_metrics(y_test, probs)
        fitted[name] = model

    # Random Forest is the required baseline artifact.
    joblib.dump(fitted["random_forest"], CENTRALIZED_MODEL_PATH)
    save_json({"centralized": all_metrics}, METRICS_PATH)
    LOGGER.info("Centralized model saved to %s", CENTRALIZED_MODEL_PATH)
    print(all_metrics)
    return fitted, all_metrics


if __name__ == "__main__":
    train_centralized()
