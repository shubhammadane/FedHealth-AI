from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CLIENTS_DIR = DATA_DIR / "clients"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

NUM_CLIENTS = 5
NUM_ROUNDS = 10
LOCAL_EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
TEST_SIZE = 0.20
PARTITION_TYPE = "non_iid"
ALGORITHM = "fedavg"

TARGET_COLUMN = "readmitted"
ID_COLUMNS = ["encounter_id", "patient_nbr"]

RAW_DATA_PATH = RAW_DIR / "diabetic_data.csv"
CLEANED_DATA_PATH = PROCESSED_DIR / "cleaned_data.csv"
X_TRAIN_PATH = PROCESSED_DIR / "X_train.csv"
X_TEST_PATH = PROCESSED_DIR / "X_test.csv"
Y_TRAIN_PATH = PROCESSED_DIR / "y_train.csv"
Y_TEST_PATH = PROCESSED_DIR / "y_test.csv"

SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
CENTRALIZED_MODEL_PATH = MODELS_DIR / "centralized_model.pkl"
GLOBAL_MODEL_PATH = MODELS_DIR / "global_model.pth"

METRICS_PATH = RESULTS_DIR / "metrics.json"
FEDERATED_HISTORY_PATH = RESULTS_DIR / "federated_history.json"
CLIENT_METRICS_PATH = RESULTS_DIR / "client_metrics.json"

for _p in [RAW_DIR, PROCESSED_DIR, CLIENTS_DIR, MODELS_DIR, RESULTS_DIR]:
    _p.mkdir(parents=True, exist_ok=True)
