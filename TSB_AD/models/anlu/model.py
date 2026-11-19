# Copyright (c) Anlu authors
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    FlamingoPerceiverResampler,
    RMSNorm,
    _parse_layer_indices,
)
from types import SimpleNamespace as TimeReferenceConfig
from .config import TotoEncoderConfig
from .encoder import TotoFactorizedEncoder


def build_toto_encoder_config(encoder_cfg) -> TotoEncoderConfig:
    """Build a Toto encoder config that matches the v4c50 TotoRCD checkpoint."""
    num_layers = encoder_cfg.num_layers
    return TotoEncoderConfig(
        d_model=encoder_cfg.d_model,
        d_proj=encoder_cfg.d_proj,
        patch_size=encoder_cfg.patch_size,
        num_layers=num_layers,
        num_heads=encoder_cfg.num_heads,
        d_ff=getattr(encoder_cfg, "d_ff", 2048),
        d_ff_dropout=0.0,
        use_rope=encoder_cfg.use_rope,
        activation=getattr(encoder_cfg, "activation", "gelu"),
        num_features=encoder_cfg.num_features,
        variate_layer_indices=list(
            getattr(encoder_cfg, "variate_layer_indices", None) or [num_layers - 1]
        ),
        residual_attn_ratio=1.0,
        residual_mlp_ratio=1.0,
        attn_scale=getattr(encoder_cfg, "attn_scale", "dim"),
        variate_mix=getattr(encoder_cfg, "variate_mix", "mean"),
        proj_hidden_dim=None,
        init_mode="timercd",
    )


class TotoFlamingoSharedResamplerModel(nn.Module):
    """Flamingo-shared adapters on a frozen TotoRCD factorized encoder.

    Time-axis adapters (layers not in ``variate_layer_indices``) read a
    per-channel memory ``(B, C, M, D)`` so batch and channel stay out of QKV.
    The last / variate-axis adapter reads a per-patch memory ``(B, P, M, D)``
    so the patch axis stays out of QKV. One Perceiver is applied twice, once
    per axis, on the same reference last hidden state.
    """

    def __init__(self, config: TimeReferenceConfig) -> None:
        super().__init__()
        self.config = config
        encoder_config = config.encoder
        self.selected_layers = _parse_layer_indices(config.flamingo_layers, encoder_config.num_layers)
        self.reference_last_layer = encoder_config.num_layers - 1
        toto_config = build_toto_encoder_config(encoder_config)
        self.encoder1 = TotoFactorizedEncoder(toto_config)
        self.encoder2 = TotoFactorizedEncoder(toto_config)
        self.memory_resampler = FlamingoPerceiverResampler(
            d_proj=encoder_config.d_model,
            num_heads=encoder_config.num_heads,
            num_memory_tokens=config.mgm_memory_tokens,
            num_layers=config.mgm_resampler_layers,
            dropout=config.dropout,
            norm_type="rmsnorm",
        )
        self.query_norms = nn.ModuleList([RMSNorm(encoder_config.d_model) for _ in self.selected_layers])
        self.memory_norms = nn.ModuleList([RMSNorm(encoder_config.d_model) for _ in self.selected_layers])
        self.cross_attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=encoder_config.d_model,
                    num_heads=encoder_config.num_heads,
                    dropout=config.dropout,
                    batch_first=True,
                )
                for _ in self.selected_layers
            ]
        )
        self.adapter_output_projections = nn.ModuleList(
            [nn.Linear(encoder_config.d_model, encoder_config.d_model) for _ in self.selected_layers]
        )
        self.adapter_ffn_norms = nn.ModuleList([RMSNorm(encoder_config.d_model) for _ in self.selected_layers])
        self.adapter_ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(encoder_config.d_model, encoder_config.d_model * 4),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(encoder_config.d_model * 4, encoder_config.d_model),
                )
                for _ in self.selected_layers
            ]
        )
        self.anomaly_head = nn.Sequential(
            nn.Linear(encoder_config.d_proj, encoder_config.d_proj // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(encoder_config.d_proj // 2, 2),
        )
        self.query_reconstruction_head = nn.Sequential(
            nn.Linear(encoder_config.d_proj, encoder_config.d_proj * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(encoder_config.d_proj * 4, encoder_config.d_proj * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(encoder_config.d_proj * 4, 1),
        )
        self.attn_gate_alphas = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in self.selected_layers]
        )
        self.ffn_gate_alphas = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in self.selected_layers]
        )
        self._foundation_frozen = False
        self._query_last_layer_unfrozen = False

    def freeze_foundation_encoders(self) -> None:
        for module in (self.encoder1, self.encoder2):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False
        self._foundation_frozen = True

    def unfreeze_query_last_transformer_layer(self) -> None:
        if not self._foundation_frozen:
            raise RuntimeError("Freeze the foundation encoders before unfreezing a query layer.")
        layer = self.encoder1.layers[-1]
        for parameter in layer.parameters():
            parameter.requires_grad = True
        self._query_last_layer_unfrozen = True

    def train(self, mode: bool = True) -> "TotoFlamingoSharedResamplerModel":
        super().train(mode)
        if getattr(self, "_foundation_frozen", False):
            self.encoder1.eval()
            self.encoder2.eval()
            if self._query_last_layer_unfrozen:
                self.encoder1.layers[-1].train(mode)
        if not any(parameter.requires_grad for parameter in self.anomaly_head.parameters()):
            self.anomaly_head.eval()
        if not any(parameter.requires_grad for parameter in self.query_reconstruction_head.parameters()):
            self.query_reconstruction_head.eval()
        return self

    def _embed_patches(
        self,
        encoder: TotoFactorizedEncoder,
        time_series: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
        if time_series.dim() == 2:
            time_series = time_series.unsqueeze(-1)
        mask = mask.bool()
        batch_size, _, num_channels = time_series.shape
        masked_positions = torch.zeros_like(mask)
        missing_channel = ((~mask) | masked_positions).to(time_series.dtype).unsqueeze(-1)
        missing_channel = missing_channel.expand(-1, -1, num_channels)
        patches, patch_mask, seq_len = encoder._patchify(time_series, mask, missing_channel)
        hidden = encoder.input_projection(patches)
        num_patches = hidden.size(2)
        return hidden, patch_mask, batch_size, seq_len, num_channels, num_patches

    def _apply_encoder_layer(
        self,
        encoder: TotoFactorizedEncoder,
        hidden: torch.Tensor,
        patch_mask: torch.Tensor,
        layer_idx: int,
        batch_size: int,
        num_channels: int,
        num_patches: int,
        rope_freqs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        layer = encoder.layers[layer_idx]
        if layer_idx in encoder.variate_layer_indices:
            hidden_var = hidden.permute(0, 2, 1, 3).contiguous()
            flat = hidden_var.reshape(batch_size * num_patches, num_channels, encoder.d_model)
            variate_mask = encoder._build_variate_mask(patch_mask, num_channels)
            flat = layer(flat, attn_mask=variate_mask, rope_freqs=None)
            hidden = flat.view(batch_size, num_patches, num_channels, encoder.d_model)
            return hidden.permute(0, 2, 1, 3).contiguous()
        flat = hidden.reshape(batch_size * num_channels, num_patches, encoder.d_model)
        time_mask = encoder._build_time_mask(patch_mask, num_channels)
        flat = layer(flat, attn_mask=time_mask, rope_freqs=rope_freqs)
        return flat.view(batch_size, num_channels, num_patches, encoder.d_model)

    def _project_hidden_to_local(
        self,
        encoder: TotoFactorizedEncoder,
        hidden: torch.Tensor,
        batch_size: int,
        seq_len: int,
        num_channels: int,
        num_patches: int,
    ) -> torch.Tensor:
        projected = encoder.output_projection(hidden)
        patch_proj = projected.view(
            batch_size,
            num_channels,
            num_patches,
            encoder.patch_size,
            encoder.d_proj,
        )
        patch_proj = patch_proj.permute(0, 2, 3, 1, 4).contiguous()
        return patch_proj.view(batch_size, -1, num_channels, encoder.d_proj)[:, :seq_len, :, :]

    def _flatten_tokens(
        self,
        hidden: torch.Tensor,
        patch_mask: torch.Tensor,
        batch_size: int,
        num_channels: int,
        num_patches: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = hidden.reshape(batch_size, num_channels * num_patches, hidden.size(-1))
        token_mask = patch_mask.unsqueeze(1).expand(-1, num_channels, -1).reshape(
            batch_size, num_channels * num_patches
        )
        return tokens, token_mask

    def _adapter_axis(self, layer_idx: int) -> str:
        if layer_idx in self.encoder1.variate_layer_indices:
            return "variate"
        return "time"

    def _resample_axis_memory(
        self,
        hidden: torch.Tensor,
        patch_mask: torch.Tensor,
        axis: str,
        return_attention_weights: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, num_channels, num_patches, width = hidden.shape
        memory_tokens = int(self.memory_resampler.memory_queries.size(1))
        if axis == "time":
            tokens = hidden.reshape(batch_size * num_channels, num_patches, width)
            token_mask = patch_mask.unsqueeze(1).expand(-1, num_channels, -1).reshape(
                batch_size * num_channels, num_patches
            )
        elif axis == "variate":
            tokens = hidden.permute(0, 2, 1, 3).contiguous().reshape(
                batch_size * num_patches, num_channels, width
            )
            token_mask = patch_mask.unsqueeze(-1).expand(-1, -1, num_channels).reshape(
                batch_size * num_patches, num_channels
            )
        else:
            raise ValueError(f"Unsupported adapter axis: {axis}")
        resampled = self.memory_resampler(
            tokens,
            token_mask,
            return_attention_weights=return_attention_weights,
        )
        if return_attention_weights:
            memory, weights = resampled
        else:
            memory, weights = resampled, None
        if axis == "time":
            memory = memory.view(batch_size, num_channels, memory_tokens, width)
        else:
            memory = memory.view(batch_size, num_patches, memory_tokens, width)
        return memory, weights

    def _align_variate_memory(self, memory: torch.Tensor, target_patches: int) -> torch.Tensor:
        source_patches = int(memory.size(1))
        if source_patches == target_patches:
            return memory
        batch_size, _, memory_tokens, width = memory.shape
        flat = memory.permute(0, 2, 3, 1).reshape(batch_size, memory_tokens * width, source_patches)
        aligned = F.interpolate(flat, size=target_patches, mode="linear", align_corners=False)
        return aligned.view(batch_size, memory_tokens, width, target_patches).permute(0, 3, 1, 2).contiguous()

    def _apply_adapter(
        self,
        hidden: torch.Tensor,
        adapter_idx: int,
        memory: torch.Tensor,
        axis: str,
        return_cross_attention_weights: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch_size, num_channels, num_patches, width = hidden.shape
        memory_tokens = int(memory.size(-2))
        if axis == "time":
            if memory.shape[:2] != (batch_size, num_channels):
                raise ValueError(
                    "Time-axis adapter memory must be (B, C, M, D), "
                    f"got {tuple(memory.shape)} for hidden {tuple(hidden.shape)}."
                )
            query_tokens = hidden.reshape(batch_size * num_channels, num_patches, width)
            kv_tokens = memory.reshape(batch_size * num_channels, memory_tokens, width)
        elif axis == "variate":
            if memory.shape[:2] != (batch_size, num_patches):
                raise ValueError(
                    "Variate-axis adapter memory must be (B, P, M, D), "
                    f"got {tuple(memory.shape)} for hidden {tuple(hidden.shape)}."
                )
            query_tokens = hidden.permute(0, 2, 1, 3).contiguous().reshape(
                batch_size * num_patches, num_channels, width
            )
            kv_tokens = memory.reshape(batch_size * num_patches, memory_tokens, width)
        else:
            raise ValueError(f"Unsupported adapter axis: {axis}")
        attention_kwargs = {"need_weights": return_cross_attention_weights}
        if return_cross_attention_weights:
            attention_kwargs["average_attn_weights"] = False
        context, weights = self.cross_attentions[adapter_idx](
            query=self.query_norms[adapter_idx](query_tokens),
            key=self.memory_norms[adapter_idx](kv_tokens),
            value=self.memory_norms[adapter_idx](kv_tokens),
            **attention_kwargs,
        )
        attn_gate = torch.tanh(self.attn_gate_alphas[adapter_idx])
        query_tokens = query_tokens + attn_gate * self.adapter_output_projections[adapter_idx](context)
        ffn_gate = torch.tanh(self.ffn_gate_alphas[adapter_idx])
        query_tokens = query_tokens + ffn_gate * self.adapter_ffns[adapter_idx](
            self.adapter_ffn_norms[adapter_idx](query_tokens)
        )
        if axis == "time":
            hidden = query_tokens.view(batch_size, num_channels, num_patches, width)
        else:
            hidden = query_tokens.view(batch_size, num_patches, num_channels, width)
            hidden = hidden.permute(0, 2, 1, 3).contiguous()
        return hidden, attn_gate, weights if return_cross_attention_weights else None

    def forward(
        self,
        win1: torch.Tensor,
        win1_mask: torch.Tensor,
        win2: torch.Tensor,
        win2_mask: torch.Tensor,
        return_dict: bool = False,
        return_cross_attention_weights: bool = False,
        return_resampler_attention_weights: bool = False,
        query_attention_mask: Optional[torch.Tensor] = None,
        reference_attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if query_attention_mask is None:
            query_attention_mask = win1_mask
        elif query_attention_mask.shape != win1_mask.shape:
            raise ValueError(
                f"Query attention mask shape {tuple(query_attention_mask.shape)} "
                f"does not match valid mask shape {tuple(win1_mask.shape)}."
            )
        query_attention_mask = query_attention_mask.bool() & win1_mask.bool()

        if reference_attention_mask is None:
            reference_attention_mask = win2_mask
        elif reference_attention_mask.shape != win2_mask.shape:
            raise ValueError(
                f"Reference attention mask shape {tuple(reference_attention_mask.shape)} "
                f"does not match valid mask shape {tuple(win2_mask.shape)}."
            )
        reference_attention_mask = reference_attention_mask.bool() & win2_mask.bool()

        ref_hidden, ref_patch_mask, ref_batch, _, ref_channels, ref_patches = self._embed_patches(
            self.encoder2, win2, reference_attention_mask
        )
        ref_rope = None
        if self.encoder2.use_rope:
            ref_rope = self.encoder2.rope_embedder(
                ref_patches, device=ref_hidden.device, dtype=ref_hidden.dtype
            )
        for layer_idx in range(self.encoder2.num_layers):
            ref_hidden = self._apply_encoder_layer(
                self.encoder2,
                ref_hidden,
                ref_patch_mask,
                layer_idx,
                ref_batch,
                ref_channels,
                ref_patches,
                ref_rope,
            )
        reference_last_hidden, _ = self._flatten_tokens(
            ref_hidden, ref_patch_mask, ref_batch, ref_channels, ref_patches
        )
        time_memory = None
        variate_memory = None
        time_resampler_weights = None
        variate_resampler_weights = None
        selected_axes = {self._adapter_axis(layer_idx) for layer_idx in self.selected_layers}
        if "time" in selected_axes:
            time_memory, time_resampler_weights = self._resample_axis_memory(
                ref_hidden, ref_patch_mask, "time", return_resampler_attention_weights
            )
        if "variate" in selected_axes:
            variate_memory, variate_resampler_weights = self._resample_axis_memory(
                ref_hidden, ref_patch_mask, "variate", return_resampler_attention_weights
            )

        hidden, patch_mask, batch_size, seq_len, num_channels, num_patches = self._embed_patches(
            self.encoder1, win1, query_attention_mask
        )
        rope_freqs = None
        if self.encoder1.use_rope:
            rope_freqs = self.encoder1.rope_embedder(
                num_patches, device=hidden.device, dtype=hidden.dtype
            )
        layer_index_to_adapter = {layer_idx: adapter_idx for adapter_idx, layer_idx in enumerate(self.selected_layers)}
        adapter_gates = []
        cross_attention_weights = []

        for layer_idx in range(self.encoder1.num_layers):
            hidden = self._apply_encoder_layer(
                self.encoder1,
                hidden,
                patch_mask,
                layer_idx,
                batch_size,
                num_channels,
                num_patches,
                rope_freqs,
            )
            adapter_idx = layer_index_to_adapter.get(layer_idx)
            if adapter_idx is None:
                continue
            axis = self._adapter_axis(layer_idx)
            if axis == "time":
                if time_memory is None:
                    raise RuntimeError("Time-axis adapter requested but time memory was not built.")
                if time_memory.size(1) != hidden.size(1):
                    raise ValueError(
                        "Time-axis adapter requires matching query/reference channel counts, "
                        f"got query C={hidden.size(1)} and memory C={time_memory.size(1)}."
                    )
                adapter_memory = time_memory
            else:
                if variate_memory is None:
                    raise RuntimeError("Variate-axis adapter requested but variate memory was not built.")
                adapter_memory = self._align_variate_memory(variate_memory, hidden.size(2))
            hidden, attn_gate, weights = self._apply_adapter(
                hidden,
                adapter_idx,
                adapter_memory,
                axis,
                return_cross_attention_weights,
            )
            adapter_gates.append(attn_gate)
            if return_cross_attention_weights:
                cross_attention_weights.append(weights)

        local_embeddings = self._project_hidden_to_local(
            self.encoder1, hidden, batch_size, seq_len, num_channels, num_patches
        )
        class_logits = self.anomaly_head(local_embeddings).mean(dim=-2)
        logits = class_logits[..., 1] - class_logits[..., 0]
        logits = logits.masked_fill(~win1_mask.bool(), 0.0)
        query_reconstruction = self.query_reconstruction_head(local_embeddings).squeeze(-1)
        if return_dict:
            return {
                "anomaly_logits": logits,
                "query_reconstruction": query_reconstruction,
                "query_local_embeddings": local_embeddings,
                "memory_tokens": {
                    "time": time_memory,
                    "variate": variate_memory,
                },
                "reference_last_hidden": reference_last_hidden,
                "gate": adapter_gates,
                "selected_layers": self.selected_layers,
                **(
                    {"cross_attention_weights": cross_attention_weights}
                    if return_cross_attention_weights
                    else {}
                ),
                **(
                    {
                        "resampler_attention_weights": {
                            "time": time_resampler_weights,
                            "variate": variate_resampler_weights,
                        }
                    }
                    if return_resampler_attention_weights
                    else {}
                ),
            }
        return logits
