# Copyright (c) Anlu authors
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple, Union
import torch
from torch import nn

class RMSNorm(nn.Module):
    """Root Mean Square Normalization layer."""

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.to(torch.float32).pow(2).mean(dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return (self.scale * x_normed).type_as(x)

class FlamingoPerceiverResampler(nn.Module):
    """Flamingo-aligned Perceiver Resampler: latents attend to [reference + latents]."""

    def __init__(
        self,
        d_proj: int,
        num_heads: int,
        num_memory_tokens: int,
        num_layers: int,
        dropout: float,
        norm_type: str = "layernorm",
    ) -> None:
        super().__init__()
        if norm_type not in {"layernorm", "rmsnorm"}:
            raise ValueError("norm_type must be either 'layernorm' or 'rmsnorm'.")
        norm_cls = RMSNorm if norm_type == "rmsnorm" else nn.LayerNorm
        self.memory_queries = nn.Parameter(torch.randn(1, num_memory_tokens, d_proj) * 0.02)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "query_norm": norm_cls(d_proj),
                        "kv_norm": norm_cls(d_proj),
                        "cross_attn": nn.MultiheadAttention(
                            embed_dim=d_proj,
                            num_heads=num_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "ffn_norm": norm_cls(d_proj),
                        "ffn": nn.Sequential(
                            nn.Linear(d_proj, d_proj * 4),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(d_proj * 4, d_proj),
                        ),
                    }
                )
                for _ in range(max(1, num_layers))
            ]
        )
        self.final_norm = norm_cls(d_proj)

    def forward(
        self,
        reference_tokens: torch.Tensor,
        reference_mask: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list[torch.Tensor]]]:
        batch_size = reference_tokens.size(0)
        memory = self.memory_queries.expand(batch_size, -1, -1)
        reference_mask = reference_mask.bool()
        memory_mask = torch.ones(batch_size, memory.size(1), dtype=torch.bool, device=reference_mask.device)
        attention_weights = []
        for layer in self.layers:
            kv_tokens = torch.cat([reference_tokens, memory], dim=1)
            kv_mask = torch.cat([reference_mask, memory_mask], dim=1)
            key_padding_mask = ~kv_mask
            attention_kwargs = {"need_weights": return_attention_weights}
            if return_attention_weights:
                attention_kwargs["average_attn_weights"] = False
            attended, weights = layer["cross_attn"](
                query=layer["query_norm"](memory),
                key=layer["kv_norm"](kv_tokens),
                value=layer["kv_norm"](kv_tokens),
                key_padding_mask=key_padding_mask,
                **attention_kwargs,
            )
            memory = memory + attended
            memory = memory + layer["ffn"](layer["ffn_norm"](memory))
            if return_attention_weights:
                attention_weights.append(weights)
        memory = self.final_norm(memory)
        if return_attention_weights:
            return memory, attention_weights
        return memory

def _parse_layer_indices(layer_spec: str, num_layers: int) -> Tuple[int, ...]:
    if layer_spec.strip().lower() in {"all", "*"}:
        return tuple(range(num_layers))
    layers = []
    for part in layer_spec.split(","):
        part = part.strip()
        if not part:
            continue
        layer_idx = int(part)
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"Layer index {layer_idx} is out of range for num_layers={num_layers}.")
        layers.append(layer_idx)
    if not layers:
        raise ValueError("mgm_delta_layers must contain at least one layer index.")
    return tuple(dict.fromkeys(layers))
