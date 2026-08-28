import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

from .config import (
    CLEANED_DATA_PATH, FEATURE_COLUMNS_PATH, PREPROCESSOR_PATH,
    X_TEST_PATH, X_TRAIN_PATH, Y_TEST_PATH, Y_TRAIN_PATH, RANDOM_SEED, TEST_SIZE
)
from .data_loader import load_cached_raw
from .feature_engineering import get_categorical_columns, get_numeric_columns, prepare_target, select_features
from .utils import LOGGER, save_json, set_seed


def build_preprocessor(X: pd.DataFrame):
    cat_cols = get_categorical_columns(X)
    num_cols = get_numeric_columns(X)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ], remainder="drop")


def preprocess_and_save():
    set_seed()
    raw = load_cached_raw()
    raw = raw.drop_duplicates().reset_index(drop=True)
    y = prepare_target(raw[["readmitted"]])
    X = select_features(raw.drop(columns=["readmitted"]))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out().tolist()

    pd.DataFrame(X_train_t, columns=feature_names).to_csv(X_TRAIN_PATH, index=False)
    pd.DataFrame(X_test_t, columns=feature_names).to_csv(X_TEST_PATH, index=False)
    pd.Series(y_train, name="readmitted").to_csv(Y_TRAIN_PATH, index=False)
    pd.Series(y_test, name="readmitted").to_csv(Y_TEST_PATH, index=False)
    raw.to_csv(CLEANED_DATA_PATH, index=False)

    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    # Backward-compatible scaler artifact name; the complete preprocessor is preferred.
    joblib.dump(preprocessor, PREPROCESSOR_PATH.parent / "scaler.pkl")
    save_json({"columns": feature_names}, FEATURE_COLUMNS_PATH)

    LOGGER.info("Preprocessing completed")
    LOGGER.info("Processed train shape: %s", X_train_t.shape)
    LOGGER.info("Processed test shape: %s", X_test_t.shape)
    print("\nTarget distribution:")
    print(y.value_counts().sort_index())
    return X_train_t, X_test_t, y_train.to_numpy(), y_test.to_numpy(), feature_names


if __name__ == "__main__":
    preprocess_and_save()
