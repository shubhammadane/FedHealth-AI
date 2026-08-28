import numpy as np


def fedavg(weighted_parameters):
    """Weighted FedAvg: sum_k (n_k / n) * w_k."""
    if not weighted_parameters:
        raise ValueError("No client parameters supplied.")
    total = sum(n for _, n in weighted_parameters)
    averaged = []
    for layer_values in zip(*(params for params, _ in weighted_parameters)):
        layer = sum(values * (n / total) for values, (_, n) in zip(layer_values, weighted_parameters))
        averaged.append(layer.astype(np.float32))
    return averaged


def make_flower_strategy():
    """Return Flower's FedAvg strategy when Flower is installed."""
    import flwr as fl
    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
    )
