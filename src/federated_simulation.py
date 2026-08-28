import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import (
    ALGORITHM,
    BATCH_SIZE,
    FEDERATED_HISTORY_PATH,
    GLOBAL_MODEL_PATH,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    NUM_CLIENTS,
    NUM_ROUNDS,
    RANDOM_SEED,
    X_TEST_PATH,
    Y_TEST_PATH,
)
from .metrics import classification_metrics
from .model import (
    ReadmissionNet,
    predict_proba,
    state_to_numpy,
    numpy_to_state,
    train_epoch,
)
from .partition import partition_data
from .utils import LOGGER, save_json, set_seed


def _weighted_average(client_params):
    """
    Weighted FedAvg aggregation.

    Formula:
        w_global = sum((n_k / n) * w_k)

    Each client provides a list of NumPy arrays,
    one array for every model parameter tensor.
    """

    if not client_params:
        raise ValueError("No client parameters were provided.")

    total_samples = sum(
        num_samples
        for _, _, num_samples in client_params
    )

    if total_samples <= 0:
        raise ValueError(
            "Total client sample count must be positive."
        )

    averaged_parameters = []

    # Process each model parameter layer separately.
    number_of_layers = len(client_params[0][0])

    for layer_index in range(number_of_layers):

        weighted_layer = None

        for parameters, _, num_samples in client_params:

            layer = np.asarray(
                parameters[layer_index],
                dtype=np.float32,
            )

            weight = (
                float(num_samples)
                / float(total_samples)
            )

            contribution = layer * weight

            if weighted_layer is None:
                weighted_layer = contribution.copy()
            else:
                weighted_layer += contribution

        averaged_parameters.append(
            weighted_layer.astype(np.float32)
        )

    return averaged_parameters


def run_federated_simulation(
    num_clients=NUM_CLIENTS,
    rounds=NUM_ROUNDS,
    local_epochs=LOCAL_EPOCHS,
    partition_type="non_iid",
    algorithm=ALGORITHM,
):
    """
    Run the complete local Federated Learning simulation.

    Supports:
        - FedAvg
        - FedProx
        - IID partitioning
        - Non-IID partitioning
    """

    set_seed(RANDOM_SEED)

    LOGGER.info(
        "Starting federated simulation"
    )

    LOGGER.info(
        "Clients: %d | Rounds: %d | Local epochs: %d",
        num_clients,
        rounds,
        local_epochs,
    )

    LOGGER.info(
        "Partition: %s | Algorithm: %s",
        partition_type,
        algorithm,
    )

    # ---------------------------------------------------------
    # 1. Create simulated hospital clients
    # ---------------------------------------------------------

    paths = partition_data(
        partition_type=partition_type,
        num_clients=num_clients,
    )

    if not paths:
        raise RuntimeError(
            "No hospital client datasets were created."
        )

    # ---------------------------------------------------------
    # 2. Determine input dimension
    # ---------------------------------------------------------

    first_client = pd.read_csv(paths[0])

    if "_target" not in first_client.columns:
        raise ValueError(
            "Client dataset does not contain '_target'."
        )

    input_dim = len(first_client.columns) - 1

    LOGGER.info(
        "Model input dimension: %d",
        input_dim,
    )

    # ---------------------------------------------------------
    # 3. Initialize global model
    # ---------------------------------------------------------

    global_model = ReadmissionNet(
        input_dim
    )

    global_params = state_to_numpy(
        global_model
    )

    # ---------------------------------------------------------
    # 4. Load held-out test dataset
    # ---------------------------------------------------------

    if not X_TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test features not found: {X_TEST_PATH}"
        )

    if not Y_TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test targets not found: {Y_TEST_PATH}"
        )

    X_test = (
        pd.read_csv(X_TEST_PATH)
        .values
        .astype("float32")
    )

    y_test = (
        pd.read_csv(Y_TEST_PATH)["readmitted"]
        .values
    )

    LOGGER.info(
        "Test dataset: %d samples, %d features",
        X_test.shape[0],
        X_test.shape[1],
    )

    # ---------------------------------------------------------
    # 5. Federated training
    # ---------------------------------------------------------

    history = []

    for rnd in range(1, rounds + 1):

        LOGGER.info(
            "========== Round %d/%d ==========",
            rnd,
            rounds,
        )

        client_results = []
        round_losses = []

        # -----------------------------------------------------
        # Local client training
        # -----------------------------------------------------

        for client_number, path in enumerate(
            paths,
            start=1,
        ):

            LOGGER.info(
                "Training Hospital Client %02d/%02d",
                client_number,
                len(paths),
            )

            df = pd.read_csv(path)

            if "_target" not in df.columns:
                raise ValueError(
                    f"_target column missing in {path}"
                )

            X_local = (
                df.drop(columns=["_target"])
                .values
                .astype("float32")
            )

            y_local = (
                df["_target"]
                .values
                .astype("float32")
            )

            if len(X_local) == 0:
                raise ValueError(
                    f"Client {path.name} is empty."
                )

            # Create local model.
            local_model = ReadmissionNet(
                input_dim
            )

            # Start from current global parameters.
            numpy_to_state(
                local_model,
                global_params,
            )

            criterion = nn.BCEWithLogitsLoss()

            optimizer = torch.optim.Adam(
                local_model.parameters(),
                lr=LEARNING_RATE,
            )

            # -------------------------------------------------
            # FedProx reference model
            # -------------------------------------------------

            reference_parameters = None

            if algorithm.lower() == "fedprox":

                reference_parameters = [
                    parameter.detach().clone()
                    for parameter
                    in local_model.parameters()
                ]

            # -------------------------------------------------
            # Local epochs
            # -------------------------------------------------

            local_loss = 0.0

            for epoch in range(
                1,
                local_epochs + 1,
            ):

                local_loss = train_epoch(
                    local_model,
                    X_local,
                    y_local,
                    optimizer,
                    criterion,
                    BATCH_SIZE,
                    prox_state=reference_parameters,
                    mu=(
                        0.01
                        if algorithm.lower()
                        == "fedprox"
                        else 0.0
                    ),
                )

                LOGGER.info(
                    "Client %02d | Epoch %d/%d | Loss %.4f",
                    client_number,
                    epoch,
                    local_epochs,
                    local_loss,
                )

            round_losses.append(
                float(local_loss)
            )

            # Get trained parameters.
            local_parameters = state_to_numpy(
                local_model
            )

            # Store:
            # parameters
            # loss
            # number of samples
            client_results.append(
                (
                    local_parameters,
                    local_loss,
                    len(X_local),
                )
            )

        # -----------------------------------------------------
        # 6. FedAvg / FedProx aggregation
        # -----------------------------------------------------

        LOGGER.info(
            "Aggregating %d client models...",
            len(client_results),
        )

        global_params = _weighted_average(
            client_results
        )

        # Update global model.
        numpy_to_state(
            global_model,
            global_params,
        )

        LOGGER.info(
            "Global aggregation completed."
        )

        # -----------------------------------------------------
        # 7. Evaluate global model
        # -----------------------------------------------------

        probabilities = predict_proba(
            global_model,
            X_test,
        )

        metrics = classification_metrics(
            y_test,
            probabilities,
        )

        entry = {
            "round": int(rnd),
            "training_loss": float(
                np.mean(round_losses)
            ),
            "accuracy": float(
                metrics["accuracy"]
            ),
            "precision": float(
                metrics["precision"]
            ),
            "recall": float(
                metrics["recall"]
            ),
            "f1": float(
                metrics["f1"]
            ),
            "roc_auc": (
                float(metrics["roc_auc"])
                if metrics["roc_auc"] is not None
                else None
            ),
            "algorithm": algorithm,
            "partition_type": partition_type,
            "num_clients": int(num_clients),
            "local_epochs": int(local_epochs),
        }

        history.append(entry)

        LOGGER.info(
            "Round %d/%d | Loss=%.4f | Accuracy=%.4f | "
            "Precision=%.4f | Recall=%.4f | F1=%.4f | ROC-AUC=%s",
            rnd,
            rounds,
            entry["training_loss"],
            entry["accuracy"],
            entry["precision"],
            entry["recall"],
            entry["f1"],
            (
                f"{entry['roc_auc']:.4f}"
                if entry["roc_auc"] is not None
                else "N/A"
            ),
        )

    # ---------------------------------------------------------
    # 8. Save final global model
    # ---------------------------------------------------------

    GLOBAL_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        global_model.state_dict(),
        GLOBAL_MODEL_PATH,
    )

    LOGGER.info(
        "Global model saved to: %s",
        GLOBAL_MODEL_PATH,
    )

    # ---------------------------------------------------------
    # 9. Save federated history
    # ---------------------------------------------------------

    save_json(
        history,
        FEDERATED_HISTORY_PATH,
    )

    LOGGER.info(
        "Federated history saved to: %s",
        FEDERATED_HISTORY_PATH,
    )

    LOGGER.info(
        "========== Federated Training Completed =========="
    )

    return history


if __name__ == "__main__":

    try:

        history = run_federated_simulation()

        if history:
            print("\nFinal Federated Result:")
            print(history[-1])

    except Exception as exc:

        LOGGER.error(
            "Federated training failed: %s",
            exc,
        )

        raise