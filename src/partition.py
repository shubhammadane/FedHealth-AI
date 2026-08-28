from pathlib import Path

import numpy as np
import pandas as pd

from .config import CLIENTS_DIR, NUM_CLIENTS, RANDOM_SEED, X_TRAIN_PATH, Y_TRAIN_PATH
from .utils import LOGGER, set_seed


def partition_data(partition_type="non_iid", num_clients=NUM_CLIENTS):
    """
    Create reproducible simulated hospital clients.

    IID:
        Randomly distributes the training set approximately equally.

    Non-IID:
        Creates controlled statistical heterogeneity while ensuring every
        client receives examples from both target classes.
    """
    set_seed(RANDOM_SEED)

    X = pd.read_csv(X_TRAIN_PATH)
    y = pd.read_csv(Y_TRAIN_PATH)["readmitted"]

    if len(X) != len(y):
        raise ValueError("Feature and target row counts do not match.")

    if num_clients < 2:
        raise ValueError("num_clients must be >= 2.")

    df = X.copy()
    df["_target"] = y.to_numpy()

    if partition_type.lower() == "iid":
        # Random approximately equal-size partitions.
        shuffled = df.sample(
            frac=1.0,
            random_state=RANDOM_SEED
        ).reset_index(drop=True)

        parts = np.array_split(shuffled, num_clients)

    elif partition_type.lower() == "non_iid":
        # ---------------------------------------------------------
        # Controlled Non-IID partition
        # ---------------------------------------------------------
        #
        # Each client gets BOTH classes, but class proportions differ.
        # This simulates statistical heterogeneity without creating
        # unrealistic 100% single-class clients.
        #
        # The proportions below are sampling weights, not fabricated
        # medical outcomes. Actual records are sampled from the
        # processed training data.
        # ---------------------------------------------------------

        class_0 = df[df["_target"] == 0].copy()
        class_1 = df[df["_target"] == 1].copy()

        if len(class_0) < num_clients or len(class_1) < num_clients:
            raise ValueError(
                "Not enough samples from both classes to create clients."
            )

        rng = np.random.default_rng(RANDOM_SEED)

        # Target readmission proportions for controlled heterogeneity.
        #
        # These are intentionally moderate so every client contains
        # both classes.
        target_rates = np.linspace(0.25, 0.65, num_clients)

        total_samples = len(df)

        # Equal client sizes.
        client_sizes = [
            total_samples // num_clients
            for _ in range(num_clients)
        ]

        for i in range(total_samples % num_clients):
            client_sizes[i] += 1

        # Calculate desired class counts.
        desired_ones = [
            int(round(size * rate))
            for size, rate in zip(client_sizes, target_rates)
        ]

        desired_zeros = [
            size - ones
            for size, ones in zip(client_sizes, desired_ones)
        ]

        # Correct rounding so every available record is assigned.
        desired_ones = np.array(desired_ones)
        desired_zeros = np.array(desired_zeros)

        desired_ones = np.floor(
            desired_ones * len(class_1) / desired_ones.sum()
        ).astype(int)

        desired_zeros = np.floor(
            desired_zeros * len(class_0) / desired_zeros.sum()
        ).astype(int)

        # Rebalance totals to match available samples.
        def distribute_remaining(counts, total_available):
            counts = counts.copy()

            while counts.sum() < total_available:
                idx = int(np.argmin(counts))
                counts[idx] += 1

            while counts.sum() > total_available:
                idx = int(np.argmax(counts))

                # Never allow a client to become empty.
                if counts[idx] > 1:
                    counts[idx] -= 1
                else:
                    break

            return counts

        desired_ones = distribute_remaining(
            desired_ones,
            len(class_1)
        )

        desired_zeros = distribute_remaining(
            desired_zeros,
            len(class_0)
        )

        # Shuffle each class independently.
        class_0 = class_0.sample(
            frac=1.0,
            random_state=RANDOM_SEED
        ).reset_index(drop=True)

        class_1 = class_1.sample(
            frac=1.0,
            random_state=RANDOM_SEED + 1
        ).reset_index(drop=True)

        parts = []

        start_0 = 0
        start_1 = 0

        for i in range(num_clients):
            end_0 = start_0 + desired_zeros[i]
            end_1 = start_1 + desired_ones[i]

            client_0 = class_0.iloc[start_0:end_0]
            client_1 = class_1.iloc[start_1:end_1]

            client_df = pd.concat(
                [client_0, client_1],
                axis=0
            )

            client_df = client_df.sample(
                frac=1.0,
                random_state=RANDOM_SEED + i + 10
            ).reset_index(drop=True)

            parts.append(client_df)

            start_0 = end_0
            start_1 = end_1

    else:
        raise ValueError(
            "partition_type must be either 'iid' or 'non_iid'."
        )

    # ---------------------------------------------------------
    # Save client datasets
    # ---------------------------------------------------------

    paths = []

    for idx, part in enumerate(parts, start=1):

        path = CLIENTS_DIR / f"hospital_{idx:02d}.csv"

        part.to_csv(path, index=False)

        paths.append(path)

        readmission_rate = part["_target"].mean()

        LOGGER.info(
            "Client %02d: %d samples | readmission rate: %.3f",
            idx,
            len(part),
            readmission_rate
        )

    return paths


def client_statistics(paths=None):
    """
    Calculate statistics for generated simulated hospital clients.
    """

    if paths is None:
        paths = sorted(
            CLIENTS_DIR.glob("hospital_*.csv")
        )

    rows = []

    for path in paths:

        df = pd.read_csv(path)

        if "_target" not in df.columns:
            raise ValueError(
                f"_target column missing in {path}"
            )

        rows.append({
            "client": path.stem,
            "samples": len(df),
            "class_0": int(
                (df["_target"] == 0).sum()
            ),
            "class_1": int(
                (df["_target"] == 1).sum()
            ),
            "readmission_rate": float(
                df["_target"].mean()
            )
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":

    paths = partition_data(
        partition_type="non_iid",
        num_clients=NUM_CLIENTS
    )

    stats = client_statistics(paths)

    print()
    print(stats.to_string(index=False))