# FedHealth-AI
## Privacy-Preserving Federated Learning Framework for Diabetes Hospital Readmission Prediction

FedHealth-AI is a presentation-ready research prototype for predicting diabetes hospital readmission risk with simulated multi-hospital Federated Learning.

> **Important:** The project uses a public, de-identified UCI dataset and **simulates** five hospital clients by partitioning the training data. It does not connect to 130 real hospitals, and it does not provide medical diagnosis.

## 1. Project Overview

The system compares:
- Centralized Logistic Regression and Random Forest baselines
- Local-only client models
- Federated FedAvg
- Federated FedProx under controlled Non-IID simulation

It includes preprocessing, model training, evaluation, explainability, and a Streamlit dashboard.

## 2. Dataset

UCI Diabetes 130-US Hospitals for Years 1999–2008, dataset ID 296.

Official source:
https://archive.ics.uci.edu/dataset/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008

Programmatic loading uses:

```python
from ucimlrepo import fetch_ucirepo
dataset = fetch_ucirepo(id=296)
X = dataset.data.features
y = dataset.data.targets
```

The raw cache is written to `data/raw/diabetic_data.csv`.

If UCI cannot be reached, download the dataset manually from the official source and place a compatible CSV at `data/raw/diabetic_data.csv`.

## 3. Target

Original:
- `NO`
- `>30`
- `<30`

Binary:
- `NO -> 0` (No readmission)
- `>30 -> 1`
- `<30 -> 1` (Readmission)

## 4. Architecture

```text
                    FEDERATED SERVER
                           │
                    Global Model
                           │
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
   Hospital 1         Hospital 2         Hospital 3
        ↓                  ↓                  ↓
   Local Dataset       Local Dataset       Local Dataset
        ↓                  ↓                  ↓
   Local Training      Local Training      Local Training
        ↓                  ↓                  ↓
   Model Update        Model Update        Model Update
        └──────────────────┼──────────────────┘
                           ↓
                         FedAvg
                           ↓
                    Updated Global Model
```

The current local simulation performs weighted FedAvg directly with PyTorch model parameters and includes a Flower-compatible `NumPyClient` implementation. This makes the experiment runnable without depending on a distributed cluster.

## 5. Installation — Windows

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 6. Run the pipeline

```bat
python -m src.data_loader
python -m src.preprocessing
python -m src.partition
python -m src.train_centralized
python -m src.train_local
python -m src.federated_simulation
python -m src.evaluate
python -m src.explainability
streamlit run app.py
```

## 7. Validation

```bat
python -m compileall src
pytest -q
```

The data-dependent commands require internet access for the first UCI fetch.

## 8. Reproducibility

Configuration is centralized in `src/config.py`:
- 5 simulated clients
- 10 rounds
- 2 local epochs
- batch size 32
- learning rate 0.001
- random seed 42
- Non-IID default partition

All charts and displayed experimental metrics are generated from saved experiment artifacts. No performance numbers are hardcoded.

## 9. Results

Generated artifacts include:
- `results/metrics.json`
- `results/federated_history.json`
- `results/client_metrics.json`
- `results/global_evaluation.json`
- `results/feature_importance.csv`

The repository intentionally does not ship fabricated result numbers.

## 10. Privacy

Federated Learning reduces the need to centralize raw training records, but it is not a complete privacy guarantee. Future research can add secure aggregation, differential privacy, encryption, and robust aggregation.

## 11. Limitations

1. Hospital clients are simulated.
2. The dataset is public and de-identified.
3. There is no real hospital deployment.
4. There is no clinical validation.
5. Federated Learning alone does not guarantee privacy.
6. The historical dataset may contain bias.
7. Class imbalance can affect evaluation.
8. External validation is required.

## 12. Future Work

- Adaptive client selection
- Differential privacy
- Secure aggregation
- Robust federated learning
- Explainable federated AI
- Real multi-institution deployment
- Multimodal federated healthcare AI combining tabular data, medical images, and clinical text

## 13. Demo flow

1. Dashboard
2. Dataset
3. Simulated hospital clients
4. Start federated training
5. Review round-by-round curves
6. Compare model metrics
7. Patient prediction
8. Explainability
9. Privacy architecture
10. Research questions and future work

## 14. Troubleshooting

**`ModuleNotFoundError`**  
Activate the virtual environment and rerun `pip install -r requirements.txt`.

**UCI download fails**  
Check internet access. Alternatively place the manually obtained compatible CSV in `data/raw/diabetic_data.csv`.

**Missing preprocessor/model**  
Run preprocessing and federated training before using the prediction page.

**Out-of-memory errors**  
Reduce `BATCH_SIZE`, `LOCAL_EPOCHS`, or the number of rounds in `src/config.py`.

**Streamlit shows missing results**  
Run the training/evaluation commands first; the UI intentionally displays “Not run” or a helpful message rather than fake numbers.
