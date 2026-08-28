import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ReadmissionNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        hidden = max(32, min(128, input_dim // 2))
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden, max(16, hidden // 2)),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(max(16, hidden // 2), 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def make_loader(X, y=None, batch_size=32, shuffle=False):
    X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
    if y is None:
        return DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=shuffle)
    y_t = torch.tensor(np.asarray(y), dtype=torch.float32)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


def train_epoch(model, X, y, optimizer, criterion, batch_size=32, device="cpu", prox_state=None, mu=0.0):
    model.train()
    loader = make_loader(X, y, batch_size=batch_size, shuffle=True)
    total_loss = 0.0
    total_n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        if prox_state is not None and mu > 0:
            prox = sum(torch.sum((p - ref) ** 2) for p, ref in zip(model.parameters(), prox_state))
            loss = loss + (mu / 2.0) * prox
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
        total_n += len(yb)
    return total_loss / max(1, total_n)


@torch.no_grad()
def predict_proba(model, X, batch_size=256, device="cpu"):
    model.eval()
    loader = make_loader(X, batch_size=batch_size)
    probs = []
    for (xb,) in loader:
        logits = model(xb.to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def state_to_numpy(model):
    return [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]


def numpy_to_state(model, parameters):
    keys = list(model.state_dict().keys())
    state = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    model.load_state_dict(state)
