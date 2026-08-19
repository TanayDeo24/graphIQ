"""Small feedforward autoencoder anomaly detector (PyTorch), per build
spec: input -> 16 -> 4 -> 16 -> input, ReLU, MSE reconstruction loss,
Adam. Reconstruction error is the anomaly score.
"""

import numpy as np
import torch
from torch import nn

from src.models.spend.features import FEATURE_COLUMNS, get_feature_matrix

INPUT_DIM = len(FEATURE_COLUMNS)


class SpendAutoencoder(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 4), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def _standardize(X: np.ndarray) -> tuple:
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def fit_autoencoder(df, epochs: int = 30, batch_size: int = 512, lr: float = 1e-3,
                     random_state: int = 42) -> tuple:
    torch.manual_seed(random_state)
    X = get_feature_matrix(df)
    X_std, mean, std = _standardize(X)

    model = SpendAutoencoder(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_tensor = torch.tensor(X_std, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = loss_fn(reconstructed, batch)
            loss.backward()
            optimizer.step()

    return model, mean, std


def score(model: SpendAutoencoder, df, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Anomaly score = per-sample MSE reconstruction error. Higher = more anomalous."""
    X = get_feature_matrix(df)
    X_std = (X - mean) / std
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_std, dtype=torch.float32)
        reconstructed = model(X_tensor)
        errors = ((reconstructed - X_tensor) ** 2).mean(dim=1).numpy()
    return errors


def per_feature_reconstruction_error(model: SpendAutoencoder, df, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Per-sample, per-feature squared error — used for explainability decomposition."""
    X = get_feature_matrix(df)
    X_std = (X - mean) / std
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_std, dtype=torch.float32)
        reconstructed = model(X_tensor)
        errors = ((reconstructed - X_tensor) ** 2).numpy()
    return errors
