from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from src.config import (
    CLIENTS_DIR, FEDERATED_HISTORY_PATH, FEATURE_COLUMNS_PATH, GLOBAL_MODEL_PATH,
    METRICS_PATH, PROCESSED_DIR, RESULTS_DIR, RANDOM_SEED
)
from src.model import ReadmissionNet, predict_proba
from src.partition import partition_data
from src.federated_simulation import run_federated_simulation

st.set_page_config(page_title="FedHealth-AI", page_icon="🏥", layout="wide")

st.title("🏥 FedHealth-AI")
st.caption("Privacy-Preserving Federated Learning for Healthcare")

DISCLAIMER = (
    "This application is an academic research prototype. Predictions are not medical "
    "diagnoses and should not replace professional clinical judgment."
)

with st.sidebar:
    st.markdown("## FedHealth-AI")
    page = st.radio("Navigation", [
        "🏠 Dashboard", "📊 Dataset", "🏥 Hospital Clients", "🧠 Federated Training",
        "🔮 Patient Prediction", "📈 Model Performance", "🔍 Explainability",
        "🔐 Privacy", "📚 Research", "ℹ️ About"
    ])


def load_json_if_exists(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def latest_metrics():
    history = load_json_if_exists(FEDERATED_HISTORY_PATH)
    return history[-1] if history else {}


if page == "🏠 Dashboard":
    st.header("Research Dashboard")
    raw_path = Path("data/raw/diabetic_data.csv")
    patients = pd.read_csv(raw_path).shape[0] if raw_path.exists() else 0
    features = len(json.loads(FEATURE_COLUMNS_PATH.read_text())["columns"]) if FEATURE_COLUMNS_PATH.exists() else 0
    latest = latest_metrics()
    cols = st.columns(6)
    values = [
        ("Total Patients", patients or "Not loaded"),
        ("Features", features or "Not processed"),
        ("Simulated Clients", 5),
        ("Federated Rounds", latest.get("round", "Not run")),
        ("Global Accuracy", f"{latest['accuracy']:.3f}" if latest.get("accuracy") is not None else "Not run"),
        ("Global F1", f"{latest['f1']:.3f}" if latest.get("f1") is not None else "Not run"),
    ]
    for c, (label, value) in zip(cols, values):
        c.metric(label, value)
    st.info("Raw patient records are not uploaded to the federated server in this research simulation.")
    st.markdown("### What this prototype demonstrates")
    st.markdown("- UCI diabetes readmission data\n- Five simulated hospital clients\n- IID / Non-IID partitioning\n- PyTorch local training\n- Weighted FedAvg and optional FedProx\n- Interactive prediction, metrics, and explainability")

elif page == "📊 Dataset":
    st.header("Dataset")
    raw_path = Path("data/raw/diabetic_data.csv")
    if not raw_path.exists():
        st.warning("Dataset not loaded yet. Run `python -m src.data_loader`.")
    else:
        df = pd.read_csv(raw_path)
        c1, c2, c3 = st.columns(3)
        c1.metric("Patients", len(df))
        c2.metric("Columns", len(df.columns))
        c3.metric("Missing cells", int(df.isna().sum().sum()))
        if "readmitted" in df:
            dist = df["readmitted"].value_counts(dropna=False).rename_axis("class").reset_index(name="count")
            st.plotly_chart(px.bar(dist, x="class", y="count", title="Original Target Distribution"), use_container_width=True)
        st.dataframe(df.head(20), use_container_width=True)

elif page == "🏥 Hospital Clients":
    st.header("Simulated Hospital Clients")
    paths = sorted(CLIENTS_DIR.glob("hospital_*.csv"))
    if not paths:
        st.warning("Clients not created yet. Run `python -m src.partition`.")
    else:
        rows = []
        for p in paths:
            d = pd.read_csv(p)
            rows.append({"Client": p.stem, "Patients": len(d), "Readmission Rate": d["_target"].mean()})
        stats = pd.DataFrame(rows)
        st.dataframe(stats, use_container_width=True)
        st.plotly_chart(px.bar(stats, x="Client", y="Patients", title="Client Dataset Sizes"), use_container_width=True)
        st.caption("These are simulated partitions of the public UCI dataset, not actual hospital systems.")

elif page == "🧠 Federated Training":
    st.header("Federated Training")
    c1, c2, c3, c4 = st.columns(4)
    clients = c1.number_input("Number of Clients", 2, 20, 5)
    rounds = c2.number_input("Federated Rounds", 1, 100, 10)
    epochs = c3.number_input("Local Epochs", 1, 20, 2)
    partition = c4.selectbox("Partition Type", ["non_iid", "iid"])
    algorithm = st.selectbox("Algorithm", ["fedavg", "fedprox"])
    if st.button("🚀 Start Federated Training", type="primary"):
        try:
            with st.status("Running real federated training...", expanded=True):
                history = run_federated_simulation(
                    num_clients=int(clients), rounds=int(rounds), local_epochs=int(epochs),
                    partition_type=partition, algorithm=algorithm
                )
                st.write(f"Completed {len(history)} rounds.")
            st.success("Training completed. Global model saved.")
            hist = pd.DataFrame(history)
            st.plotly_chart(px.line(hist, x="round", y="training_loss", title="Training Loss"), use_container_width=True)
            st.plotly_chart(px.line(hist, x="round", y=["accuracy", "f1", "roc_auc"], title="Federated Metrics"), use_container_width=True)
        except Exception as exc:
            st.error(f"Training failed: {exc}")

elif page == "🔮 Patient Prediction":
    st.header("Patient Readmission Risk")
    st.warning(DISCLAIMER)
    preprocessor_path = Path("models/preprocessor.pkl")
    if not GLOBAL_MODEL_PATH.exists() or not preprocessor_path.exists():
        st.info("Train/preprocess the project first so the global model and preprocessor exist.")
    else:
        preprocessor = joblib.load(preprocessor_path)
        raw_columns = list(getattr(preprocessor, "feature_names_in_", []))
        form = {}
        # The UI focuses on common clinically meaningful variables; other required
        # model fields are filled with their training-time imputation defaults.
        common = [
            ("age", "Age", "[0-10)"), ("gender", "Gender", "Male"),
            ("time_in_hospital", "Time in Hospital (days)", 4),
            ("num_lab_procedures", "Lab Procedures", 40),
            ("num_procedures", "Procedures", 1),
            ("num_medications", "Medications", 10),
            ("number_outpatient", "Outpatient Visits", 0),
            ("number_emergency", "Emergency Visits", 0),
            ("number_inpatient", "Inpatient Visits", 0),
            ("number_diagnoses", "Diagnoses", 6),
            ("A1Cresult", "A1C Result", "None"),
            ("insulin", "Insulin", "No"),
            ("diabetesMed", "Diabetes Medication", "Yes"),
        ]
        cols = st.columns(2)
        for i, (key, label, default) in enumerate(common):
            with cols[i % 2]:
                if isinstance(default, str):
                    form[key] = st.text_input(label, value=default)
                else:
                    form[key] = st.number_input(label, value=float(default))
        if st.button("🔮 Predict Readmission Risk", type="primary"):
            row = pd.DataFrame([{c: form.get(c, np.nan) for c in raw_columns}])
            X = preprocessor.transform(row)
            model = ReadmissionNet(X.shape[1])
            model.load_state_dict(torch.load(GLOBAL_MODEL_PATH, map_location="cpu"))
            probability = float(predict_proba(model, X)[0])
            risk = "HIGH" if probability >= 0.5 else "LOW"
            st.metric("Readmission Risk", risk)
            st.metric("Probability", f"{probability:.1%}")
            st.caption("Model: Federated Global Model")
            st.info(DISCLAIMER)

elif page == "📈 Model Performance":
    st.header("Model Performance")
    metrics = load_json_if_exists(METRICS_PATH)
    if metrics:
        st.json(metrics)
    history = load_json_if_exists(FEDERATED_HISTORY_PATH)
    if history:
        hist = pd.DataFrame(history)
        st.dataframe(hist, use_container_width=True)
        st.plotly_chart(px.line(hist, x="round", y="accuracy", title="Accuracy vs Federated Round"), use_container_width=True)
        st.plotly_chart(px.line(hist, x="round", y="f1", title="F1 vs Federated Round"), use_container_width=True)
    eval_path = RESULTS_DIR / "global_evaluation.json"
    if eval_path.exists():
        ev = load_json_if_exists(eval_path)
        st.subheader("Confusion Matrix")
        cm = np.array(ev["confusion_matrix"])
        st.plotly_chart(px.imshow(cm, text_auto=True, x=["No Readmission", "Readmission"], y=["No Readmission", "Readmission"]), use_container_width=True)
        st.subheader("ROC Curve")
        roc = ev["roc_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="Global Model"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance"))
        st.plotly_chart(fig, use_container_width=True)

elif page == "🔍 Explainability":
    st.header("Explainability")
    imp = RESULTS_DIR / "feature_importance.csv"
    if imp.exists():
        df = pd.read_csv(imp).head(20)
        st.plotly_chart(px.bar(df.sort_values("importance"), x="importance", y="feature", orientation="h",
                               title="Permutation Feature Importance"), use_container_width=True)
        st.caption("Importance is model-specific association evidence, not causal medical evidence.")
    else:
        st.info("Run `python -m src.explainability` after federated training to generate feature importance.")

elif page == "🔐 Privacy":
    st.header("Privacy Architecture")
    st.code("""                 CENTRAL SERVER
                       ↑
                Model Updates
                       ↑
       ┌───────────────┼───────────────┐
       │               │               │
  Hospital 1      Hospital 2      Hospital 3
       │               │               │
   Local Data      Local Data      Local Data""")
    st.markdown("""
**Raw Patient Data → Remains Local**

Federated Learning allows participating clients to train models locally and share model
updates instead of directly centralizing raw training data. Additional privacy and
security mechanisms may still be required.

Potential future mechanisms:
- Secure Aggregation
- Differential Privacy
- Encryption
- Robust Aggregation
""")

elif page == "📚 Research":
    st.header("Research Framing")
    st.subheader("Problem")
    st.write("Traditional centralized healthcare AI can require combining data from multiple institutions.")
    st.subheader("Proposed Solution")
    st.write("Federated Learning enables collaborative model training while keeping raw training data at the client side.")
    st.subheader("Research Questions")
    for q in [
        "How does Federated Learning compare with centralized learning?",
        "How does Non-IID data affect performance?",
        "Does FedProx improve stability under heterogeneous clients?",
        "What is the communication-round effect on model convergence?",
        "How can additional privacy mechanisms improve the framework?",
    ]:
        st.write("• " + q)

elif page == "ℹ️ About":
    st.header("About")
    st.markdown("""
**Project:** FedHealth-AI  
**Domain:** Artificial Intelligence · Machine Learning · Federated Learning · Healthcare AI  
**Dataset:** UCI Diabetes 130-US Hospitals for Years 1999–2008 (ID 296)  
**Framework:** PyTorch + Flower-compatible client architecture  
**UI:** Streamlit

This is an academic research prototype. The five hospital clients are simulated partitions
of a public de-identified dataset; they are not real hospital deployments.
""")
