"""feature selection + logistic probe."""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class HallucinationProbe:
    def __init__(self):
        self.pipeline = None
        self.threshold = 0.5

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train).astype(int)

        # Conservative feature selection: reduce 1792 coordinates to top 512.
        k = min(512, X_train.shape[1])

        self.pipeline = Pipeline([
            ("selector", SelectKBest(score_func=f_classif, k=k)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                C=0.5,
                max_iter=4000,
                solver="liblinear",
                random_state=42,
            )),
        ])

        self.pipeline.fit(X_train, y_train)

        # Tune threshold on validation for accuracy, because competition metric is accuracy.
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
        return self.pipeline.predict_proba(X).astype(np.float32)

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(np.float32)
