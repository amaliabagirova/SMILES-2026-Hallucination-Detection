"""L2-normalized 2A + compact geometry.

Uses:
- transformer layer 6, token -1
- transformer layer 23, token -1

Features:
- L2-normalized v6
- L2-normalized v23
- cosine(v6, v23)
- ||v6||
- ||v23||
- ||v23 - v6||
- sequence length
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


def _get_selected_vectors(hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    layers = _get_transformer_layers(hidden_states)
    real_idx = _get_real_token_indices(attention_mask)

    vectors = []
    for layer_idx, token_offset in SELECTED_CELLS:
        tok_idx = _resolve_token_index(real_idx, token_offset)
        vectors.append(layers[layer_idx, tok_idx, :])

    return vectors[0], vectors[1]


def aggregate(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    v6, v23 = _get_selected_vectors(hidden_states, attention_mask)

    v6_normed = F.normalize(v6, p=2, dim=0)
    v23_normed = F.normalize(v23, p=2, dim=0)

    cosine = F.cosine_similarity(
        v6_normed.unsqueeze(0),
        v23_normed.unsqueeze(0),
        dim=1,
    )

    norm6 = torch.norm(v6, p=2).unsqueeze(0)
    norm23 = torch.norm(v23, p=2).unsqueeze(0)
    norm_diff = torch.norm(v23 - v6, p=2).unsqueeze(0)

    seq_len = attention_mask.bool().sum().to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(0)

    geometry = torch.cat([cosine, norm6, norm23, norm_diff, seq_len], dim=0)

    return torch.cat([v6_normed, v23_normed, geometry], dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    # Geometry is already included
    return torch.empty(0, device=hidden_states.device, dtype=hidden_states.dtype)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    return aggregate(hidden_states, attention_mask)
