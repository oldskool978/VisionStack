# ShadowBoxer/models/qwen35/text.py
from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .cache import Qwen3_5HybridCache
from .config import (
    Qwen3_5ModelOutput,
    Qwen3_5RMSNorm,
    Qwen3_5TextConfig,
    _EYE_MATRIX_CACHE,
    _STRICTLY_UPPER_MASK_CACHE,
    _act_fn,
    _linear,
    _rotate_half,
)
from .deltanet import Qwen3_5GatedDeltaNet


class Qwen3_5TextRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.config = config
        rope_params = config.rope_parameters or {}
        self.base = float(rope_params.get("rope_theta", 10000000.0))
        partial_factor = float(rope_params.get("partial_rotary_factor", 0.25))
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.dim = int(head_dim * partial_factor)
        self.register_buffer(
            "inv_freq", self._compute_inv_freq(torch.device("cpu")), persistent=False
        )
        self.mrope_section = list(rope_params.get("mrope_section", [11, 11, 10]))

    def _compute_inv_freq(self, device: torch.device) -> torch.Tensor:
        return (
            1.0
            / (
                self.base
                ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
            )
        ).to(device)

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.inv_freq = self._compute_inv_freq(self.inv_freq.device)
        return self

    def _recompose(self, freq: torch.Tensor) -> torch.Tensor:
        freqs_thw = freq[0].clone()
        for dim_idx, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim_idx] * 3
            idx = slice(offset, length, 3)
            freqs_thw[..., idx] = freq[dim_idx, ..., idx]
        return torch.cat((freqs_thw, freqs_thw), dim=-1)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.inv_freq.dtype != torch.float32 or self.inv_freq.device != x.device:
            self.inv_freq = self._compute_inv_freq(x.device)
        pos_3d = position_ids.expand(3, -1, -1)
        freqs = pos_3d[..., None].float() * self.inv_freq
        cos = self._recompose(freqs.cos())
        sin = self._recompose(freqs.sin())
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _apply_rotary_pos_emb_partial(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


class Qwen3_5Attention(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim * 2,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[Qwen3_5HybridCache] = None,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            _linear(self.q_proj, hidden_states).view(
                *input_shape, -1, self.head_dim * 2
            ),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(
            _linear(self.k_proj, hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = (
            _linear(self.v_proj, hidden_states).view(hidden_shape).transpose(1, 2)
        )

        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb_partial(
            query_states, key_states, cos, sin
        )

        has_past = False
        if cache is not None:
            past_k = cache.key_cache.get(self.layer_idx)
            past_v = cache.value_cache.get(self.layer_idx)
            if past_k is not None and past_v is not None:
                has_past = True
                key_states = torch.cat([past_k, key_states], dim=2)
                value_states = torch.cat([past_v, value_states], dim=2)
            cache.key_cache[self.layer_idx] = key_states
            cache.value_cache[self.layer_idx] = value_states

        enable_gqa = self.num_key_value_groups > 1
        q_len = query_states.shape[2]
        kv_len = key_states.shape[2]

        if attention_mask is not None and attention_mask.ndim == 4:
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                is_causal=False,
                scale=self.scaling,
                enable_gqa=enable_gqa,
            )
        elif has_past and q_len > 1:
            past_len = kv_len - q_len
            causal_mask = torch.ones(
                (q_len, kv_len), dtype=torch.bool, device=query_states.device
            ).tril(diagonal=past_len)
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask[None, None, :, :],
                is_causal=False,
                scale=self.scaling,
                enable_gqa=enable_gqa,
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                is_causal=(q_len > 1 and not has_past),
                scale=self.scaling,
                enable_gqa=enable_gqa,
            )
        del query_states, key_states, value_states

        attn_output = (
            attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        )
        attn_output = attn_output * torch.sigmoid(gate)
        del gate
        return _linear(self.o_proj, attn_output)


class Qwen3_5MLP(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, intermediate_size: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_name = config.hidden_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _linear(
            self.down_proj,
            _act_fn(_linear(self.gate_proj, x), self.act_name)
            * _linear(self.up_proj, x),
        )


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.block_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        linear_mask: Optional[torch.Tensor] = None,
        full_attn_mask: Optional[torch.Tensor] = None,
        cache: Optional[Qwen3_5HybridCache] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.block_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states, attention_mask=linear_mask, cache=cache
            )
        else:
            hidden_states = self.self_attn(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=full_attn_mask,
                cache=cache,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3_5MTPHead(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.config = config
        num_mtp_layers = max(1, int(getattr(config, "mtp_num_hidden_layers", 1)))
        mtp_cfg = dataclasses.replace(
            config,
            num_hidden_layers=num_mtp_layers,
            layer_types=["full_attention"] * num_mtp_layers,
        )
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(mtp_cfg, idx) for idx in range(num_mtp_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        next_token_embeds: torch.Tensor,
        rotary_emb: Qwen3_5TextRotaryEmbedding,
        position_ids: Optional[torch.LongTensor] = None,
        full_attn_mask: Optional[torch.Tensor] = None,
        cache: Optional[Qwen3_5HybridCache] = None,
    ) -> torch.Tensor:
        norm_emb = self.pre_fc_norm_embedding(next_token_embeds)
        norm_hid = self.pre_fc_norm_hidden(hidden_states)
        fused = _linear(self.fc, torch.cat([norm_emb, norm_hid], dim=-1))

        batch_size, seq_len, _ = fused.shape
        past_seen = cache.past_seen_tokens if cache is not None else 0
        if position_ids is None:
            pos = torch.arange(
                past_seen + 1, past_seen + seq_len + 1, device=fused.device
            ).view(1, 1, -1)
            if cache is not None and cache.rope_deltas is not None and past_seen > 0:
                pos = pos + cache.rope_deltas.to(device=fused.device).unsqueeze(0)
            position_ids = pos.expand(3, batch_size, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, batch_size, -1)
        elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]

        position_embeddings = rotary_emb(fused, position_ids)
        for layer in self.layers:
            fused = layer(
                fused,
                position_embeddings=position_embeddings,
                full_attn_mask=full_attn_mask,
                cache=cache,
            )
        if cache is not None:
            cache.past_seen_tokens = past_seen + seq_len
        return self.norm(fused)


class Qwen3_5TextModel(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.config = config
        pad_idx = (
            int(config.pad_token_id)
            if (
                config.pad_token_id is not None
                and 0 <= int(config.pad_token_id) < int(config.vocab_size)
            )
            else None
        )
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, pad_idx
        )
        self.layers = nn.ModuleList(
            [
                Qwen3_5DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        _STRICTLY_UPPER_MASK_CACHE.clear()
        _EYE_MATRIX_CACHE.clear()
        return self

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Qwen3_5HybridCache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Qwen3_5ModelOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        active_cache: Optional[Qwen3_5HybridCache] = (
            past_key_values
            if past_key_values is not None
            else (Qwen3_5HybridCache() if use_cache else None)
        )
        past_seen = active_cache.past_seen_tokens if active_cache is not None else 0
        batch_size, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device

        if active_cache is not None and active_cache.record_checkpoints:
            active_cache.resolve_active_checkpoints(past_seen, seq_len)

        if position_ids is None:
            pos = (
                torch.arange(past_seen, past_seen + seq_len, device=device).view(
                    1, 1, -1
                )
            )
            if (
                active_cache is not None
                and active_cache.rope_deltas is not None
                and past_seen > 0
            ):
                pos = pos + active_cache.rope_deltas.to(device=device).unsqueeze(0)
            position_ids = pos.expand(3, batch_size, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, batch_size, -1)
        elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]

        linear_mask: Optional[torch.Tensor] = None
        full_attn_mask: Optional[torch.Tensor] = None
        if attention_mask is not None and attention_mask.ndim == 2:
            if not attention_mask.bool().all():
                linear_mask = attention_mask[:, -seq_len:].bool()
                total_len = past_seen + seq_len
                causal = torch.ones(
                    (seq_len, total_len), dtype=torch.bool, device=device
                ).tril(diagonal=past_seen)
                full_mask_2d = (
                    attention_mask[:, :total_len].bool()
                    if attention_mask.shape[1] >= total_len
                    else F.pad(
                        attention_mask.bool(),
                        (total_len - attention_mask.shape[1], 0),
                        value=True,
                    )
                )
                full_attn_mask = (
                    causal[None, None, :, :] & full_mask_2d[:, None, None, :]
                )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                linear_mask=linear_mask,
                full_attn_mask=full_attn_mask,
                cache=active_cache,
            )

        if active_cache is not None:
            active_cache.past_seen_tokens = past_seen + seq_len
        del position_embeddings, linear_mask, full_attn_mask
        hidden_states = self.norm(hidden_states)
        return Qwen3_5ModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=active_cache,
        )


__all__ = [
    "Qwen3_5TextRotaryEmbedding",
    "_apply_rotary_pos_emb_partial",
    "Qwen3_5Attention",
    "Qwen3_5MLP",
    "Qwen3_5DecoderLayer",
    "Qwen3_5MTPHead",
    "Qwen3_5TextModel",
]