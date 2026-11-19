# Copyright (c) Anlu authors
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import List, Optional

def default_variate_layer_indices(num_layers: int) -> List[int]:
    """Place variate-axis attention only on the final layer (Toto 2.0 sweep best)."""
    if num_layers <= 0:
        return []
    return [num_layers - 1]

@dataclass
class TotoEncoderConfig:
    """Configuration for the Toto-style factorized time/variate encoder."""

    d_model: int = 512
    d_proj: int = 256
    patch_size: int = 32
    num_layers: int = 8
    num_heads: int = 8
    d_ff: int = 2048
    d_ff_dropout: float = 0.0
    use_rope: bool = True
    activation: str = "gelu"
    num_features: int = 1
    variate_layer_indices: Optional[List[int]] = None
    residual_attn_ratio: float = 1.0
    residual_mlp_ratio: float = 1.0
    attn_scale: str = "dim"  # 1/d_head; matches 2_1 checkpoint time-axis training
    # Last/variate layer mixing: "mean" is out(mean(V)); "attention" keeps QKV.
    variate_mix: str = "mean"
    # Residual SiLU patch MLP width. None means d_model, so output projection
    # does not expand to patch_size * d_proj (8192) and balloon parameters.
    proj_hidden_dim: Optional[int] = None
    init_mode: str = "timercd"

    def __post_init__(self) -> None:
        if self.variate_layer_indices is None:
            self.variate_layer_indices = default_variate_layer_indices(self.num_layers)
        if self.d_ff <= 0:
            self.d_ff = self.d_model * 4

    @property
    def head_dim(self) -> int:
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            )
        return self.d_model // self.num_heads
