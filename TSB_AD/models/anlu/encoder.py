# Copyright (c) Anlu authors
# SPDX-License-Identifier: Apache-2.0
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return (self.scale * x_normed).type_as(x)


class ResidualSiLUMLP(nn.Module):
    """Two-layer SiLU MLP with residual connection (Toto 2.0 patch projections)."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=True)
        self.res_proj = (
            nn.Identity()
            if in_dim == out_dim
            else nn.Linear(in_dim, out_dim, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_proj(x) + self.fc2(F.silu(self.fc1(x)))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return freqs.to(dtype=dtype)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to (B, seq_len, embed_dim)."""
    batch_size, seq_len, embed_dim = x.shape
    x_pairs = x.view(batch_size, seq_len, embed_dim // 2, 2)
    cos = freqs[:seq_len].cos().unsqueeze(0)
    sin = freqs[:seq_len].sin().unsqueeze(0)
    x0 = x_pairs[..., 0]
    x1 = x_pairs[..., 1]
    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos
    return torch.stack([y0, y1], dim=-1).reshape(batch_size, seq_len, embed_dim)


class PerDimScale(nn.Module):
    """Learned per-dimension query scaling used in Toto 2.0 / TimesFM 2.5."""

    def __init__(self, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return q * self.scale.unsqueeze(0).unsqueeze(0)


class FactorizedAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attn_scale: str = "dim",
        use_rope: bool = True,
        use_per_dim_scale: bool = True,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attn_scale = attn_scale
        self.use_rope = use_rope
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.per_dim_scale = PerDimScale(num_heads, self.head_dim) if use_per_dim_scale else None
        self.rope = RotaryEmbedding(self.head_dim) if use_rope else None

    def _scale(self) -> float:
        if self.attn_scale == "dim":
            return 1.0 / float(self.head_dim)
        return 1.0 / math.sqrt(float(self.head_dim))

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        rope_freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.use_rope and rope_freqs is not None:
            q = apply_rope(q, rope_freqs)
            k = apply_rope(k, rope_freqs)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.per_dim_scale is not None:
            q = self.per_dim_scale(q.transpose(1, 2)).transpose(1, 2)

        sdpa_mask = attn_mask.unsqueeze(1) if attn_mask is not None else None
        query_has_keys = None
        if sdpa_mask is not None:
            # cuDNN SDPA backward can return NaNs for fully masked queries.
            # Give those rows one dummy key, then restore their zero output.
            query_has_keys = sdpa_mask.any(dim=-1, keepdim=True)
            dummy_key = torch.arange(k.shape[-2], device=k.device) == 0
            sdpa_mask = sdpa_mask | (~query_has_keys & dummy_key)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self._scale(),
        )
        if query_has_keys is not None:
            y = y.masked_fill(~query_has_keys, 0.0)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(y)


class TotoMLP(nn.Module):
    """Feed-forward block with no bias on linear layers (Toto 2.0)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(hidden))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attn_scale: str,
        use_rope: bool,
        dropout: float,
        residual_attn_ratio: float,
        residual_mlp_ratio: float,
        mix_mode: str = "attention",
    ) -> None:
        super().__init__()
        if mix_mode not in {"attention", "mean"}:
            raise ValueError(f"Unsupported mix_mode: {mix_mode}")
        self.mix_mode = mix_mode
        self.residual_attn_ratio = residual_attn_ratio
        self.residual_mlp_ratio = residual_mlp_ratio
        self.norm1 = RMSNorm(d_model)
        self.attn = FactorizedAttention(
            d_model=d_model,
            num_heads=num_heads,
            attn_scale=attn_scale,
            use_rope=use_rope,
            use_per_dim_scale=True,
        )
        if self.mix_mode == "mean":
            for name, param in self.attn.named_parameters():
                if "v_proj" not in name and "out_proj" not in name:
                    param.requires_grad = False
        self.norm2 = RMSNorm(d_model)
        self.mlp = TotoMLP(d_model, d_ff, dropout=dropout)

    @staticmethod
    def _mean_mix(x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attn_mask is None:
            return x.mean(dim=1, keepdim=True).expand_as(x)
        valid = attn_mask.diagonal(dim1=-2, dim2=-1).to(dtype=x.dtype)
        denom = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
        weights = (valid / denom).unsqueeze(-1)
        return (x * weights).sum(dim=1, keepdim=True).expand_as(x)

    def _mean_mix_v_out(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # Match C>1 attention with uniform softmax: out(mean(V)).
        mixed_v = self._mean_mix(self.attn.v_proj(x), attn_mask)
        return self.attn.out_proj(mixed_v)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        rope_freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        if self.mix_mode == "mean":
            x = self._mean_mix_v_out(x, attn_mask)
        else:
            x = self.attn(x, attn_mask=attn_mask, rope_freqs=rope_freqs)
        x = residual + self.residual_attn_ratio * x
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        return residual + self.residual_mlp_ratio * x


class TotoFactorizedEncoder(nn.Module):
    """
    Toto 2.0-style encoder with alternating time-axis and variate-axis attention.

    Internal tensor layout after patch embedding: (B, C, S, D)
    Output layout: (B, T, C, d_proj)
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.d_model = config.d_model
        self.d_proj = config.d_proj
        self.num_layers = config.num_layers
        self.use_rope = config.use_rope
        self.variate_layer_indices = set(config.variate_layer_indices or [])

        proj_hidden_dim = config.proj_hidden_dim or config.d_model
        self.input_projection = ResidualSiLUMLP(
            2 * config.patch_size,
            config.d_model,
            hidden_dim=proj_hidden_dim,
        )
        self.output_projection = ResidualSiLUMLP(
            config.d_model,
            config.patch_size * config.d_proj,
            hidden_dim=proj_hidden_dim,
        )

        variate_mix = getattr(config, "variate_mix", "mean")
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    d_ff=config.d_ff,
                    attn_scale=config.attn_scale,
                    use_rope=True,
                    dropout=config.d_ff_dropout,
                    residual_attn_ratio=config.residual_attn_ratio,
                    residual_mlp_ratio=config.residual_mlp_ratio,
                    mix_mode="mean" if (layer_idx in self.variate_layer_indices and variate_mix == "mean") else "attention",
                )
                for layer_idx in range(config.num_layers)
            ]
        )
        if self.use_rope:
            self.rope_embedder = RotaryEmbedding(config.d_model)

        if config.init_mode == "toto":
            self.apply(self._init_linear)
            residual_scale = 1.0 / math.sqrt(2.0 * max(1, config.num_layers))
            with torch.no_grad():
                self.input_projection.fc2.weight.mul_(residual_scale)
                self.output_projection.fc2.weight.mul_(residual_scale)
                for layer in self.layers:
                    layer.attn.out_proj.weight.mul_(residual_scale)
                    layer.mlp.down_proj.weight.mul_(residual_scale)
        elif config.init_mode == "timercd":
            # Match TimeRCD ts_encoder_bi_bias: _init_parameters is a no-op for
            # current parameter names, so encoder linears keep PyTorch defaults.
            pass
        else:
            raise ValueError(f"Unsupported init_mode: {config.init_mode}")

    @staticmethod
    def _init_linear(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _patchify(
        self,
        time_series: torch.Tensor,
        attention_mask: torch.Tensor,
        missing_channel: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        batch_size, seq_len, num_channels = time_series.shape
        patch_size = self.patch_size
        padded_length = math.ceil(seq_len / patch_size) * patch_size
        if padded_length > seq_len:
            pad_amount = padded_length - seq_len
            time_series = F.pad(time_series, (0, 0, 0, pad_amount), value=0.0)
            attention_mask = F.pad(attention_mask, (0, pad_amount), value=False)
            missing_channel = F.pad(missing_channel, (0, 0, 0, pad_amount), value=1.0)

        num_patches = padded_length // patch_size
        value_patches = time_series.view(batch_size, num_patches, patch_size, num_channels)
        value_patches = value_patches.permute(0, 3, 1, 2).contiguous()
        missing_patches = missing_channel.view(batch_size, num_patches, patch_size, num_channels)
        missing_patches = missing_patches.permute(0, 3, 1, 2).contiguous()
        patches = torch.cat([value_patches, missing_patches], dim=-1)

        patch_mask = attention_mask.view(batch_size, num_patches, patch_size).any(dim=-1)
        return patches, patch_mask, seq_len

    def _build_time_mask(self, patch_mask: torch.Tensor, num_channels: int) -> torch.Tensor:
        # (B*C, S, S) pairwise mask: padding patches are invalid as query and key.
        # CPM-masked patches stay valid so they can read visible context.
        batch_size, num_patches = patch_mask.shape
        valid = patch_mask.unsqueeze(1).expand(-1, num_channels, -1).reshape(
            batch_size * num_channels, num_patches
        )
        return valid.unsqueeze(-1) & valid.unsqueeze(-2)

    def _build_variate_mask(self, patch_mask: torch.Tensor, num_channels: int) -> torch.Tensor:
        batch_size, num_patches = patch_mask.shape
        valid = patch_mask.unsqueeze(-1).expand(-1, -1, num_channels).reshape(
            batch_size * num_patches, num_channels
        )
        return valid.unsqueeze(-1) & valid.unsqueeze(-2)

    def forward(
        self,
        time_series: torch.Tensor,
        mask: torch.Tensor,
        masked_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if time_series.dim() == 2:
            time_series = time_series.unsqueeze(-1)
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if masked_positions is None:
            masked_positions = torch.zeros_like(mask)
        elif masked_positions.dtype != torch.bool:
            masked_positions = masked_positions.bool()

        device = time_series.device
        batch_size, _, num_channels = time_series.shape
        missing_channel = ((~mask) | masked_positions).to(time_series.dtype).unsqueeze(-1)
        missing_channel = missing_channel.expand(-1, -1, num_channels)
        patches, patch_mask, seq_len = self._patchify(time_series, mask, missing_channel)
        num_patches = patches.shape[2]

        embedded = self.input_projection(patches)
        hidden = embedded  # (B, C, S, D)

        rope_freqs = None
        if self.use_rope:
            rope_freqs = self.rope_embedder(num_patches, device=device, dtype=hidden.dtype)

        for layer_idx, layer in enumerate(self.layers):
            if layer_idx in self.variate_layer_indices:
                hidden_var = hidden.permute(0, 2, 1, 3).contiguous()
                flat = hidden_var.reshape(batch_size * num_patches, num_channels, self.d_model)
                variate_mask = self._build_variate_mask(patch_mask, num_channels)
                flat = layer(flat, attn_mask=variate_mask, rope_freqs=None)
                hidden = flat.view(batch_size, num_patches, num_channels, self.d_model)
                hidden = hidden.permute(0, 2, 1, 3).contiguous()
            else:
                flat = hidden.reshape(batch_size * num_channels, num_patches, self.d_model)
                time_mask = self._build_time_mask(patch_mask, num_channels)
                flat = layer(flat, attn_mask=time_mask, rope_freqs=rope_freqs)
                hidden = flat.view(batch_size, num_channels, num_patches, self.d_model)

        projected = self.output_projection(hidden)
        patch_proj = projected.view(
            batch_size,
            num_channels,
            num_patches,
            self.patch_size,
            self.d_proj,
        )
        patch_proj = patch_proj.permute(0, 2, 3, 1, 4).contiguous()
        local_embeddings = patch_proj.view(batch_size, -1, num_channels, self.d_proj)[:, :seq_len, :, :]
        return local_embeddings
