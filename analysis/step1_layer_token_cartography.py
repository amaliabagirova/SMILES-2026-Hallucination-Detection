from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from splitting import split_data

TAIL_PATH = "./artifacts/tail_hidden_states.pt"
OUT_CSV = "./artifacts/layer_token_scores.csv"


def evaluate_one_cell(X, y, splits):
    accs, f1s, aucs = [], [], []

    for idx_train, idx_val, idx_test in splits:
        X_train = X[idx_train]
        y_train = y[idx_train]
        X_test = X[idx_test]
        y_test = y[idx_test]

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                solver="liblinear",
                random_state=42,
            )),
        ])

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        aucs.append(roc_auc_score(y_test, y_prob))

    return {
        "mean_test_accuracy": float(np.mean(accs)),
        "std_test_accuracy": float(np.std(accs)),
        "mean_test_f1": float(np.mean(f1s)),
        "std_test_f1": float(np.std(f1s)),
        "mean_test_auroc": float(np.mean(aucs)),
        "std_test_auroc": float(np.std(aucs)),
    }


def main():
    os.makedirs("./artifacts", exist_ok=True)

    payload = torch.load(TAIL_PATH)
    tails = payload["tail_hidden_states"].float().numpy()   # (689, 24, 8, 896)
    y = payload["labels"].cpu().numpy().astype(int)

    print("Loaded cached tail states")
    print("tails shape:", tails.shape)
    print("labels shape:", y.shape)

    dummy_df = pd.DataFrame({"label": y})
    splits = split_data(y, dummy_df)

    results = []

    n_layers = tails.shape[1]
    n_tail_tokens = tails.shape[2]

    for layer_idx in tqdm(range(n_layers), desc="Layers"):
        for token_pos in range(n_tail_tokens):
            
            token_offset = token_pos - n_tail_tokens  

            X = tails[:, layer_idx, token_pos, :]  # (n_samples, 896)

            metrics = evaluate_one_cell(X, y, splits)

            results.append({
                "layer": int(layer_idx),
                "token_pos": int(token_pos),
                "token_offset": int(token_offset),
                "feature_dim": int(X.shape[1]),
                **metrics,
            })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(
        by=["mean_test_auroc", "mean_test_accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    df_results.to_csv(OUT_CSV, index=False)

    print(f"Saved results to: {OUT_CSV}")
    print("\nTop 10 cells by mean_test_auroc:\n")
    print(df_results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
