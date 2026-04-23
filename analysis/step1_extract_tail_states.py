from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import torch
import pandas as pd
from tqdm.auto import tqdm

from model import get_model_and_tokenizer

DATA_PATH = "./data/dataset.csv"
OUT_PATH = "./artifacts/tail_hidden_states.pt"
TAIL_TOKENS = 8
MAX_LENGTH = 512


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_tail_hidden_states(model, tokenizer, text: str, device: torch.device):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    # embeddings + transformer layers
    hidden_states = outputs.hidden_states

    # drop embeddings layer, keep transformer layers only
    layer_states = torch.stack(hidden_states[1:], dim=0)  # (n_layers, 1, seq_len, hidden_dim)
    layer_states = layer_states[:, 0]  # (n_layers, seq_len, hidden_dim)

    real_idx = torch.nonzero(attention_mask[0].bool(), as_tuple=False).squeeze(-1)
    if real_idx.numel() == 0:
        real_idx = torch.arange(input_ids.shape[1], device=device)

    tail_idx = real_idx[-TAIL_TOKENS:]
    tail = layer_states[:, tail_idx, :]  # (n_layers, tail_len, hidden_dim)

    tail_len = tail.shape[1]
    if tail_len < TAIL_TOKENS:
        pad = torch.zeros(
            (tail.shape[0], TAIL_TOKENS - tail_len, tail.shape[2]),
            dtype=tail.dtype,
            device=tail.device,
        )
        tail = torch.cat([pad, tail], dim=1)

    return tail.detach().cpu()


def main():
    os.makedirs("./artifacts", exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows")

    model, tokenizer = get_model_and_tokenizer()
    model = model.to(device)
    model.eval()

    all_tails = []
    labels = []

    start = time.time()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting tail states"):
        prompt = str(row["prompt"])
        response = str(row["response"])
        label = int(float(row["label"]))

        text = prompt + response
        tail = extract_tail_hidden_states(model, tokenizer, text, device)

        all_tails.append(tail)
        labels.append(label)

    tails_tensor = torch.stack(all_tails, dim=0)  # (n_samples, n_layers, tail_tokens, hidden_dim)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    payload = {
        "tail_hidden_states": tails_tensor,
        "labels": labels_tensor,
        "tail_tokens": TAIL_TOKENS,
        "max_length": MAX_LENGTH,
    }

    torch.save(payload, OUT_PATH)

    elapsed = time.time() - start
    print(f"Saved to: {OUT_PATH}")
    print(f"tail_hidden_states shape: {tails_tensor.shape}")
    print(f"labels shape: {labels_tensor.shape}")
    print(f"Done in {elapsed:.1f} s")


if __name__ == "__main__":
    main()
