# ShadowBoxer/models/qwen35/vision.py
from __future__ import annotations

from collections import OrderedDict
from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .config import (
    Qwen3_5ModelOutput,
    Qwen3_5VisionConfig,
    _FP8_DTYPES,
    _act_fn,
    _linear,
    _resolve_compute_dtype,
    _rotate_half,
)


class Qwen3_5VisionRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        rope_params = config.rope_parameters or {}
        self.base = float(rope_params.get("rope_theta", 10000.0))
        dim = config.hidden_size // config.num_heads
        self.spatial_dim = dim // 2
        self.register_buffer(
            "inv_freq", self._compute_inv_freq(torch.device("cpu")), persistent=False
        )

    def _compute_inv_freq(self, device: torch.device) -> torch.Tensor:
        return (
            1.0
            / (
                self.base
                ** (
                    torch.arange(0, self.spatial_dim, 2, dtype=torch.float32)
                    / self.spatial_dim
                )
            )
        ).to(device)

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.inv_freq = self._compute_inv_freq(self.inv_freq.device)
        return self

    @staticmethod
    def _recompose(freq: torch.Tensor) -> torch.Tensor:
        freq_h, freq_w = freq[0], freq[1]
        freq_hw = torch.cat([freq_h, freq_w], dim=-1)
        return torch.cat([freq_hw, freq_hw], dim=-1)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.inv_freq.dtype != torch.float32 or self.inv_freq.device != x.device:
            self.inv_freq = self._compute_inv_freq(x.device)
        freqs = position_ids[..., None].float() * self.inv_freq
        cos = self._recompose(freqs.cos())
        sin = self._recompose(freqs.sin())
        return cos, sin


class Qwen3_5VisionMLP(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_name = config.hidden_act

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return _linear(
            self.linear_fc2,
            _act_fn(_linear(self.linear_fc1, hidden_state), self.act_name),
        )


class Qwen3_5VisionPatchEmbed(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel_size = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = _resolve_compute_dtype(self.proj.weight, hidden_states.dtype)
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        if self.proj.weight.dtype in _FP8_DTYPES:
            b = self.proj.bias.to(target_dtype) if self.proj.bias is not None else None
            out = F.conv3d(
                hidden_states.to(target_dtype),
                self.proj.weight.to(target_dtype),
                b,
                stride=self.proj.stride,
            )
            return out.view(-1, self.embed_dim)
        return self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)


class Qwen3_5VisionPatchMerger(nn.Module):
    def __init__(
        self, config: Qwen3_5VisionConfig, use_postshuffle_norm: bool = False
    ):
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6
        )
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(
            x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x
        ).view(-1, self.hidden_size)
        return _linear(self.linear_fc2, self.act_fn(_linear(self.linear_fc1, normed)))


def _apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q_f32, k_f32 = q.float(), k.float()
    cos_f32, sin_f32 = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q_f32 * cos_f32) + (_rotate_half(q_f32) * sin_f32)
    k_embed = (k_f32 * cos_f32) + (_rotate_half(k_f32) * sin_f32)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class Qwen3_5VisionAttention(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            _linear(self.qkv, hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb_vision(
            query_states, key_states, cos, sin
        )

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        if cu_seqlens.numel() == 2:
            attn_output = (
                F.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    is_causal=False,
                    scale=self.scaling,
                )
                .transpose(1, 2)
                .reshape(seq_length, -1)
                .contiguous()
            )
        else:
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            q_splits = torch.split(query_states, lengths, dim=2)
            k_splits = torch.split(key_states, lengths, dim=2)
            v_splits = torch.split(value_states, lengths, dim=2)
            attn_outputs = [
                F.scaled_dot_product_attention(
                    q, k, v, is_causal=False, scale=self.scaling
                ).transpose(1, 2)
                for q, k, v in zip(q_splits, k_splits, v_splits)
            ]
            attn_output = (
                torch.cat(attn_outputs, dim=1).reshape(seq_length, -1).contiguous()
            )
        del query_states, key_states, value_states
        return _linear(self.proj, attn_output)


class Qwen3_5VisionBlock(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3_5VisionAttention(config)
        self.mlp = Qwen3_5VisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3_5VisionModel(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3_5VisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(
            config.num_position_embeddings, config.hidden_size
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        self.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(config)

        self.blocks = nn.ModuleList(
            [Qwen3_5VisionBlock(config) for _ in range(config.depth)]
        )
        self.merger = Qwen3_5VisionPatchMerger(config, use_postshuffle_norm=False)
        self._geom_cache: OrderedDict[
            Tuple[Tuple[Tuple[int, int, int], ...], str, torch.dtype, int, int],
            Tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                Tuple[torch.Tensor, torch.Tensor],
            ],
        ] = OrderedDict()

    @property
    def dtype(self) -> torch.dtype:
        return _resolve_compute_dtype(self.patch_embed.proj.weight)

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def clear_caches(self) -> None:
        self._geom_cache.clear()

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.clear_caches()
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.clear_caches()

    def _compute_geometry(
        self, grid_thw: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        merge = self.spatial_merge_size
        side = self.num_grid_per_side
        pos_ids_list: List[torch.Tensor] = []
        interp_embeds_list: List[torch.Tensor] = []
        seqlens: List[int] = [0]

        for row in grid_thw.tolist():
            t, h, w = int(row[0]), int(row[1]), int(row[2])
            h_ids = torch.arange(h, device=device)
            w_ids = torch.arange(w, device=device)
            gh, gw = torch.meshgrid(h_ids, w_ids, indexing="ij")
            gh = (
                gh.reshape(h // merge, merge, w // merge, merge)
                .permute(0, 2, 1, 3)
                .flatten()
            )
            gw = (
                gw.reshape(h // merge, merge, w // merge, merge)
                .permute(0, 2, 1, 3)
                .flatten()
            )
            hw_ids = torch.stack([gh, gw], dim=0).repeat(1, t)
            pos_ids_list.append(hw_ids)

            if h == 1:
                y_coords = torch.zeros(1, device=device, dtype=torch.float32)
            else:
                y_coords = torch.linspace(
                    0, side - 1, h, device=device, dtype=torch.float32
                )
            if w == 1:
                x_coords = torch.zeros(1, device=device, dtype=torch.float32)
            else:
                x_coords = torch.linspace(
                    0, side - 1, w, device=device, dtype=torch.float32
                )

            y0 = y_coords.floor().long().clamp(0, side - 1)
            x0 = x_coords.floor().long().clamp(0, side - 1)
            y1 = (y0 + 1).clamp(0, side - 1)
            x1 = (x0 + 1).clamp(0, side - 1)

            wy = y_coords - y0.float()
            wx = x_coords - x0.float()

            idx_00 = (y0[:, None] * side + x0[None, :]).flatten()
            idx_01 = (y0[:, None] * side + x1[None, :]).flatten()
            idx_10 = (y1[:, None] * side + x0[None, :]).flatten()
            idx_11 = (y1[:, None] * side + x1[None, :]).flatten()

            w_00 = ((1.0 - wy)[:, None] * (1.0 - wx)[None, :]).flatten().unsqueeze(-1)
            w_01 = ((1.0 - wy)[:, None] * wx[None, :]).flatten().unsqueeze(-1)
            w_10 = (wy[:, None] * (1.0 - wx)[None, :]).flatten().unsqueeze(-1)
            w_11 = (wy[:, None] * wx[None, :]).flatten().unsqueeze(-1)

            emb_2d = (
                self.pos_embed(idx_00) * w_00
                + self.pos_embed(idx_01) * w_01
                + self.pos_embed(idx_10) * w_10
                + self.pos_embed(idx_11) * w_11
            )
            emb_merged = (
                emb_2d.view(h // merge, merge, w // merge, merge, -1)
                .permute(0, 2, 1, 3, 4)
                .reshape(h * w, -1)
                .repeat(t, 1)
            )
            interp_embeds_list.append(emb_merged)

            for _ in range(t):
                seqlens.append(seqlens[-1] + h * w)

        position_ids = torch.cat(pos_ids_list, dim=1)
        pos_embeds = torch.cat(interp_embeds_list, dim=0)
        cu_seqlens = torch.tensor(seqlens, dtype=torch.int32, device=device)
        return position_ids, pos_embeds, cu_seqlens

    def _get_cached_geometry_and_rotary(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        device = hidden_states.device
        grid_key = tuple(
            (int(r[0]), int(r[1]), int(r[2])) for r in grid_thw.tolist()
        )
        cache_key = (
            grid_key,
            str(device),
            self.pos_embed.weight.dtype,
            id(self.pos_embed.weight),
            int(self.pos_embed.weight.data_ptr()),
        )
        cached = self._geom_cache.get(cache_key)
        if cached is not None:
            self._geom_cache.move_to_end(cache_key)
            return cached

        position_ids, pos_embeds, cu_seqlens = self._compute_geometry(grid_thw, device)
        position_embeddings = self.rotary_pos_emb(hidden_states, position_ids)
        entry = (position_ids, pos_embeds, cu_seqlens, position_embeddings)
        self._geom_cache[cache_key] = entry
        while len(self._geom_cache) > 8:
            self._geom_cache.popitem(last=False)
        return entry

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> Qwen3_5ModelOutput:
        _, pos_embeds, cu_seqlens, position_embeddings = (
            self._get_cached_geometry_and_rotary(hidden_states, grid_thw)
        )
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )

        merged_hidden_states = self.merger(hidden_states)
        return Qwen3_5ModelOutput(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )


__all__ = [
    "Qwen3_5VisionRotaryEmbedding",
    "Qwen3_5VisionMLP",
    "Qwen3_5VisionPatchEmbed",
    "Qwen3_5VisionPatchMerger",
    "_apply_rotary_pos_emb_vision",
    "Qwen3_5VisionAttention",
    "Qwen3_5VisionBlock",
    "Qwen3_5VisionModel",
]