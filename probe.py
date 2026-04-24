"""probe.py  MLP probe with validation accuracy threshold calibration"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


class _MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class HallucinationProbe:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.threshold = 0.5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        torch.manual_seed(42)
        np.random.seed(42)

        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train).astype(np.float32)

        X_train_scaled = self.scaler.fit_transform(X_train).astype(np.float32)

        X_t = torch.tensor(X_train_scaled, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32, device=self.device)

        input_dim = X_train_scaled.shape[1]
        self.model = _MLP(input_dim).to(self.device)

        # Class imbalance correction: more hallucinated than truthful.
        n_pos = float((y_train == 1).sum())
        n_neg = float((y_train == 0).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-3,
        )

        self.model.train()
        for _ in range(250):
            optimizer.zero_grad()
            logits = self.model(X_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

        # threshold calibration on validation accuracy
        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val).astype(int)

            probs = self.predict_proba(X_val)[:, 1]

            best_thr = 0.5
            best_acc = -1.0
            best_f1 = -1.0

            for thr in np.linspace(0.05, 0.95, 181):
                pred = (probs >= thr).astype(int)
                acc = accuracy_score(y_val, pred)
                f1 = f1_score(y_val, pred, zero_division=0)

                if acc > best_acc or (acc == best_acc and f1 > best_f1):
                    best_acc = acc
                    best_f1 = f1
                    best_thr = float(thr)

            self.threshold = best_thr

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler.transform(X).astype(np.float32)

        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
            probs_pos = torch.sigmoid(logits).detach().cpu().numpy()

        probs_pos = probs_pos.astype(np.float32)
        probs_neg = 1.0 - probs_pos

        return np.stack([probs_neg, probs_pos], axis=1)

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(np.float32)
