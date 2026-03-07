"""
PyTorch AutoEncoder for unsupervised fraud anomaly scoring.

Architecture:
  Encoder: input_dim → 128 → 64 → 32 → bottleneck (16)
  Decoder: 16 → 32 → 64 → 128 → input_dim

Trained on legitimate transactions only. High reconstruction error
= anomalous (potentially fraudulent) transaction.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ─── Model definition ────────────────────────────────────────────────────────


class FraudAutoEncoder(nn.Module):
    """
    Deep autoencoder with batch normalisation and dropout for regularisation.
    Bottleneck dimension of 16 forces learning of compact fraud-discriminative
    representations from 40+ engineered features.
    """

    def __init__(self, input_dim: int, bottleneck: int = 16, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck = bottleneck

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, bottleneck),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error (anomaly score)."""
        with torch.no_grad():
            x_hat = self.forward(x)
            return torch.mean((x - x_hat) ** 2, dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return bottleneck embeddings (used as XGBoost features)."""
        with torch.no_grad():
            return self.encoder(x)


# ─── Trainer ─────────────────────────────────────────────────────────────────


class AutoEncoderTrainer:
    """
    Trains FraudAutoEncoder on legitimate transactions only.
    Supports early stopping and threshold calibration.
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck: int = 16,
        dropout: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 512,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FraudAutoEncoder(input_dim, bottleneck, dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.batch_size = batch_size
        self.threshold: Optional[float] = None
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def fit(
        self,
        X_legit: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        patience: int = 10,
        verbose: bool = True,
    ) -> "AutoEncoderTrainer":
        """
        Train on legitimate transactions only.
        Args:
            X_legit: Feature matrix of non-fraud transactions (normalised).
            X_val:   Optional validation set (all transactions) for loss tracking.
            epochs:  Max training epochs.
            patience: Early stopping patience.
        """
        X_tensor = torch.FloatTensor(X_legit).to(self.device)
        loader = DataLoader(
            TensorDataset(X_tensor), batch_size=self.batch_size, shuffle=True
        )

        best_val = float("inf")
        no_improve = 0
        best_state = None

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            for (batch,) in loader:
                self.optimizer.zero_grad()
                out = self.model(batch)
                loss = self.criterion(out, batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            train_loss = epoch_loss / len(loader)
            self.train_losses.append(train_loss)

            # Validation
            val_loss = train_loss
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    xv = torch.FloatTensor(X_val).to(self.device)
                    val_loss = self.criterion(self.model(xv), xv).item()
                self.val_losses.append(val_loss)

            if verbose and epoch % 10 == 0:
                print(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
                )

            # Early stopping
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                no_improve = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        return self

    def calibrate_threshold(
        self, X_legit: np.ndarray, percentile: float = 95.0
    ) -> float:
        """
        Set anomaly threshold as the `percentile` of reconstruction errors
        on the legitimate training set. Transactions above this threshold
        are flagged as anomalous.
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X_legit).to(self.device)
        errors = self.model.reconstruction_error(X_tensor).cpu().numpy()
        self.threshold = float(np.percentile(errors, percentile))
        return self.threshold

    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Return reconstruction error scores for each sample."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        return self.model.reconstruction_error(X_tensor).cpu().numpy()

    def predict_binary(self, X: np.ndarray) -> np.ndarray:
        """Return binary anomaly flags using calibrated threshold."""
        if self.threshold is None:
            raise ValueError("Call calibrate_threshold() first.")
        scores = self.predict_anomaly_score(X)
        return (scores > self.threshold).astype(int)

    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Return bottleneck embeddings for stacking with XGBoost."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        return self.model.encode(X_tensor).cpu().numpy()

    def save(self, path: str) -> None:
        """Save model weights and threshold."""
        os.makedirs(Path(path).parent, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "input_dim": self.model.input_dim,
                "bottleneck": self.model.bottleneck,
                "threshold": self.threshold,
                "train_losses": self.train_losses,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "AutoEncoderTrainer":
        """Load saved model."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location=device)
        trainer = cls(
            input_dim=ckpt["input_dim"],
            bottleneck=ckpt["bottleneck"],
            device=device,
        )
        trainer.model.load_state_dict(ckpt["model_state"])
        trainer.threshold = ckpt.get("threshold")
        trainer.train_losses = ckpt.get("train_losses", [])
        return trainer
