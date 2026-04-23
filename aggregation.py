"""aggregation.py sparse two-cell aggregation.

Both layer indices are zero-based and correspond to the 24 transformer
layers used in step-1 cartography.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


SELECTED_LAYERS = (6, 23)


def _get_transformer_layers(hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Accept either:
    - (24, seq_len, hidden_dim): transformer layers only
    - (25, seq_len, hidden_dim): embeddings + 24 transformer layers

    Return transformer layers only, shape (24, seq_len, hidden_dim).
    """
    n_layers = hidden_states.shape[0]

    if n_layers == 24:
        return hidden_states
    if n_layers == 25:
        return hidden_states[1:]

    return hidden_states[-24:]


def _get_last_real_token_index(attention_mask: torch.Tensor) -> int:
    real_idx = torch.nonzero(attention_mask.bool(), as_tuple=False).squeeze(-1)
    if real_idx.numel() == 0:
        return int(attention_mask.shape[0] - 1)
    return int(real_idx[-1].item())


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Build a sparse representation by concatenating the final real-token vectors
    from two selected layers.

    Output shape:
        2 * 896 = 1792
    """
    layers = _get_transformer_layers(hidden_states)  # (24, seq_len, hidden_dim)
    last_idx = _get_last_real_token_index(attention_mask)

    vecs = [layers[layer_idx, last_idx, :] for layer_idx in SELECTED_LAYERS]
    feature = torch.cat(vecs, dim=0)
    return feature


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Small auxiliary features for the selected layer pair.
    """
    layers = _get_transformer_layers(hidden_states)
    last_idx = _get_last_real_token_index(attention_mask)

    v1 = layers[SELECTED_LAYERS[0], last_idx, :]
    v2 = layers[SELECTED_LAYERS[1], last_idx, :]

    norm1 = torch.norm(v1, p=2).unsqueeze(0)
    norm2 = torch.norm(v2, p=2).unsqueeze(0)
    cosine = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0), dim=1)
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
