"""2A sparse cells with per-cell L2 normalization.

Uses:
- transformer layer 6, token -1
- transformer layer 23, token -1
"""

from __future__ import annotations
import torch
import torch.nn.functional as F

SELECTED_CELLS = ((6, -1), (23, -1))


def _get_transformer_layers(hidden_states: torch.Tensor) -> torch.Tensor:
    if hidden_states.shape[0] == 25:
        return hidden_states[1:]
    if hidden_states.shape[0] == 24:
        return hidden_states
    return hidden_states[-24:]


def _get_real_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    real_idx = torch.nonzero(attention_mask.bool(), as_tuple=False).squeeze(-1)
    if real_idx.numel() == 0:
        real_idx = torch.arange(attention_mask.shape[0], device=attention_mask.device)
    return real_idx


def _resolve_token_index(real_idx: torch.Tensor, token_offset: int) -> int:
    k = real_idx.numel()
    pos = k + token_offset
    pos = max(0, min(pos, k - 1))
    return int(real_idx[pos].item())


def aggregate(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    layers = _get_transformer_layers(hidden_states)
    real_idx = _get_real_token_indices(attention_mask)

    vecs = []
    for layer_idx, token_offset in SELECTED_CELLS:
        tok_idx = _resolve_token_index(real_idx, token_offset)
        v = layers[layer_idx, tok_idx, :]
        v = F.normalize(v, p=2, dim=0)
        vecs.append(v)

    return torch.cat(vecs, dim=0)


def extract_geometric_features(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    layers = _get_transformer_layers(hidden_states)
    real_idx = _get_real_token_indices(attention_mask)

    raw_vecs = []
    normed_vecs = []

    for layer_idx, token_offset in SELECTED_CELLS:
        tok_idx = _resolve_token_index(real_idx, token_offset)
        v = layers[layer_idx, tok_idx, :]
        raw_vecs.append(v)
        normed_vecs.append(F.normalize(v, p=2, dim=0))

    norm1 = torch.norm(raw_vecs[0], p=2).unsqueeze(0)
    norm2 = torch.norm(raw_vecs[1], p=2).unsqueeze(0)
    cosine = F.cosine_similarity(
        normed_vecs[0].unsqueeze(0),
        normed_vecs[1].unsqueeze(0),
        dim=1,
    )
    seq_len = attention_mask.bool().sum().to(dtype=hidden_states.dtype).unsqueeze(0)

    return torch.cat([norm1, norm2, cosine, seq_len], dim=0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
