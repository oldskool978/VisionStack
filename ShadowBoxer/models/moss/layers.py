# ShadowBoxer/models/moss/layers.py
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from .config import (
    _FFN_TILE_TOKENS,
    IsoKVSpec,
    ModTuple6,
    PiecewiseModSpec,
    _RMSNorm,
    _linear,
    _resolve_compute_dtype,
    _situ_inplace,
)
from .rope import _rotate


def _slice_blhd_to_bhld(
    t: torch.Tensor, start: int, end: int
) -> torch.Tensor:
    s = t[:, start:end, :, :]
    if t.shape[0] > 1 and (end - start) < t.shape[1]:
        s = s.contiguous()
    return s.transpose(1, 2)


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rope: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    heads: int,
    attn_mask: Optional[torch.Tensor] = None,
    iso_kv_spec: Optional[IsoKVSpec] = None,
    target_attn_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if q.ndim == 3:
        q = q.unflatten(-1, (heads, -1))
    if k.ndim == 3:
        k = k.unflatten(-1, (heads, -1))
    if v.ndim == 3:
        v = v.unflatten(-1, (heads, -1))
    q_rot = _rotate(q, rope)
    k_rot = _rotate(k, rope)
    if q_rot.dtype != v.dtype:
        q_rot = q_rot.to(v.dtype)
    if k_rot.dtype != v.dtype:
        k_rot = k_rot.to(v.dtype)

    if iso_kv_spec is not None and attn_mask is None:
        if len(iso_kv_spec) == 4:
            target_tokens, ref_token_lengths, _, collab_groups = iso_kv_spec
        else:
            target_tokens, ref_token_lengths, _ = iso_kv_spec
            collab_groups = tuple((i,) for i in range(len(ref_token_lengths)))
        total_ref = sum(ref_token_lengths)
        if total_ref > 0:
            _, total_seq, _, _ = q_rot.shape
            text_start = target_tokens + total_ref
            text_len = total_seq - text_start
            if not v.is_contiguous():
                v = v.contiguous()
            out_blhd = torch.empty_like(
                q_rot, memory_format=torch.contiguous_format
            )

            k_full = k_rot.transpose(1, 2)
            v_full = v.transpose(1, 2)
            q_target = _slice_blhd_to_bhld(q_rot, 0, target_tokens)
            eff_target_bias = (
                target_attn_bias.to(dtype=q_target.dtype)
                if (
                    target_attn_bias is not None
                    and target_attn_bias.dtype != q_target.dtype
                )
                else target_attn_bias
            )
            out_target = F.scaled_dot_product_attention(
                q_target, k_full, v_full, attn_mask=eff_target_bias
            )
            out_blhd[:, :target_tokens, :, :].copy_(out_target.transpose(1, 2))
            del out_target

            if text_len > 0:
                q_text = _slice_blhd_to_bhld(q_rot, text_start, total_seq)
                out_text = F.scaled_dot_product_attention(
                    q_text, k_full, v_full, attn_mask=None
                )
                out_blhd[:, text_start:, :, :].copy_(out_text.transpose(1, 2))
                del out_text

            ref_spans: List[Tuple[int, int]] = []
            span_cursor = target_tokens
            for r_len in ref_token_lengths:
                ref_spans.append((span_cursor, span_cursor + r_len))
                span_cursor += r_len

            k_text_blhd = k_rot[:, text_start:, :, :]
            v_text_blhd = v[:, text_start:, :, :]
            num_ref_spans = len(ref_spans)
            for r_idx, (c_start, c_end) in enumerate(ref_spans):
                r_len = c_end - c_start
                if r_len <= 0:
                    continue
                q_ref = _slice_blhd_to_bhld(q_rot, c_start, c_end)
                peers = (
                    collab_groups[r_idx]
                    if r_idx < len(collab_groups)
                    else (r_idx,)
                )
                is_suffix_contiguous = (
                    len(peers) > 0
                    and peers[-1] == num_ref_spans - 1
                    and len(peers) == (peers[-1] - peers[0] + 1)
                )
                if is_suffix_contiguous:
                    span_start = ref_spans[peers[0]][0]
                    k_ref = _slice_blhd_to_bhld(k_rot, span_start, total_seq)
                    v_ref = _slice_blhd_to_bhld(v, span_start, total_seq)
                    out_ref = F.scaled_dot_product_attention(
                        q_ref, k_ref, v_ref, attn_mask=None
                    )
                else:
                    k_pieces = [
                        k_rot[:, ref_spans[p][0] : ref_spans[p][1], :, :]
                        for p in peers
                    ]
                    v_pieces = [
                        v[:, ref_spans[p][0] : ref_spans[p][1], :, :]
                        for p in peers
                    ]
                    k_pieces.append(k_text_blhd)
                    v_pieces.append(v_text_blhd)
                    k_cat = torch.cat(k_pieces, dim=1)
                    v_cat = torch.cat(v_pieces, dim=1)
                    out_ref = F.scaled_dot_product_attention(
                        q_ref,
                        k_cat.transpose(1, 2),
                        v_cat.transpose(1, 2),
                        attn_mask=None,
                    )
                    del k_cat, v_cat, k_pieces, v_pieces
                out_blhd[:, c_start:c_end, :, :].copy_(out_ref.transpose(1, 2))
                del out_ref

            del q_rot, k_rot
            return out_blhd.flatten(2)

    if target_attn_bias is not None and attn_mask is None:
        _, total_seq, _, _ = q_rot.shape
        target_tokens = int(target_attn_bias.shape[2])
        if 0 < target_tokens < total_seq:
            if not v.is_contiguous():
                v = v.contiguous()
            out_blhd = torch.empty_like(
                q_rot, memory_format=torch.contiguous_format
            )
            k_full = k_rot.transpose(1, 2)
            v_full = v.transpose(1, 2)
            q_target = _slice_blhd_to_bhld(q_rot, 0, target_tokens)
            eff_target_bias = (
                target_attn_bias.to(dtype=q_target.dtype)
                if target_attn_bias.dtype != q_target.dtype
                else target_attn_bias
            )
            out_target = F.scaled_dot_product_attention(
                q_target, k_full, v_full, attn_mask=eff_target_bias
            )
            out_blhd[:, :target_tokens, :, :].copy_(out_target.transpose(1, 2))
            del out_target
            q_rest = _slice_blhd_to_bhld(q_rot, target_tokens, total_seq)
            out_rest = F.scaled_dot_product_attention(
                q_rest, k_full, v_full, attn_mask=None
            )
            out_blhd[:, target_tokens:, :, :].copy_(out_rest.transpose(1, 2))
            del out_rest, q_rot, k_rot
            return out_blhd.flatten(2)

    return (
        F.scaled_dot_product_attention(
            q_rot.transpose(1, 2),
            k_rot.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_mask,
        )
        .transpose(1, 2)
        .flatten(2)
    )


def _apply_mod_inplace(
    y: torch.Tensor,
    modulation: Union[torch.Tensor, ModTuple6, PiecewiseModSpec],
    is_ffn: bool,
) -> None:
    if isinstance(modulation, tuple) and len(modulation) == 4:
        mod_t, mod_z, target_tokens, total_img_tokens = modulation
        t_shift, t_scale_p1 = (mod_t[3], mod_t[4]) if is_ffn else (mod_t[0], mod_t[1])
        z_shift, z_scale_p1 = (mod_z[3], mod_z[4]) if is_ffn else (mod_z[0], mod_z[1])
        y[:, :target_tokens, :].mul_(t_scale_p1).add_(t_shift)
        y[:, target_tokens:total_img_tokens, :].mul_(z_scale_p1).add_(z_shift)
        y[:, total_img_tokens:, :].mul_(t_scale_p1).add_(t_shift)
        return

    if isinstance(modulation, tuple):
        shift, scale_p1 = (
            (modulation[3], modulation[4])
            if is_ffn
            else (modulation[0], modulation[1])
        )
    else:
        if modulation.ndim == 3:
            shift_a, scale_a, _, shift_f, scale_f, _ = (
                modulation[:, :, None, :].unbind(1)
            )
        else:
            shift_a, scale_a, _, shift_f, scale_f, _ = modulation.unbind(1)
        shift = shift_f if is_ffn else shift_a
        scale_p1 = 1 + (scale_f if is_ffn else scale_a)
    y.mul_(scale_p1).add_(shift)


def _apply_gate_residual_inplace(
    x: torch.Tensor,
    delta: torch.Tensor,
    modulation: Union[torch.Tensor, ModTuple6, PiecewiseModSpec],
    is_ffn: bool,
) -> torch.Tensor:
    if isinstance(modulation, tuple) and len(modulation) == 4:
        mod_t, mod_z, target_tokens, total_img_tokens = modulation
        t_gate = mod_t[5] if is_ffn else mod_t[2]
        z_gate = mod_z[5] if is_ffn else mod_z[2]
        delta[:, :target_tokens, :].mul_(t_gate)
        delta[:, target_tokens:total_img_tokens, :].mul_(z_gate)
        delta[:, total_img_tokens:, :].mul_(t_gate)
        return x.add_(delta)

    if isinstance(modulation, tuple):
        gate = modulation[5] if is_ffn else modulation[2]
    else:
        if modulation.ndim == 3:
            _, _, gate_a, _, _, gate_f = modulation[:, :, None, :].unbind(1)
        else:
            _, _, gate_a, _, _, gate_f = modulation.unbind(1)
        gate = gate_f if is_ffn else gate_a

    if gate.shape[1] == 1:
        return x.addcmul_(delta, gate.to(x.dtype))
    delta.mul_(gate)
    return x.add_(delta)


class _TextBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1 = _RMSNorm(dim)
        self.norm2 = _RMSNorm(dim)
        self.q_norm = _RMSNorm(dim // heads)
        self.k_norm = _RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate = nn.Linear(dim, ffn_dim)
        self.ffn_up = nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        rope: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        q, k, v = _linear(self.qkv, self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1)))
        k = self.k_norm(k.unflatten(-1, (self.heads, -1)))
        v = v.unflatten(-1, (self.heads, -1))
        x = x + _linear(self.proj, _attention(q, k, v, rope, self.heads))
        y = self.norm2(x)
        return x + _linear(
            self.ffn_down,
            _situ_inplace(_linear(self.ffn_up, y), _linear(self.ffn_gate, y)),
        )


class _TextProjection(nn.Module):
    def __init__(self, text_dim: int, dim: int, ffn_dim: int, heads: int):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([_TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(
        self,
        x: torch.Tensor,
        rope: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        x = _linear(self.proj, x)
        for block in self.blocks:
            x = block(x, rope)
        return x


class _Block(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1 = _RMSNorm(dim)
        self.norm2 = _RMSNorm(dim)
        self.q_norm = _RMSNorm(dim // heads)
        self.k_norm = _RMSNorm(dim // heads)
        self.proj = nn.ModuleDict(
            {name: nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")}
        )
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        modulation: Union[torch.Tensor, ModTuple6, PiecewiseModSpec],
        attn_mask: Optional[torch.Tensor] = None,
        iso_kv_spec: Optional[IsoKVSpec] = None,
        target_attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dtype = _resolve_compute_dtype(self.proj["q"].weight, x.dtype)
        y = self.norm1(x)
        _apply_mod_inplace(y, modulation, is_ffn=False)
        if y.dtype != dtype:
            y = y.to(dtype)

        q = self.q_norm(_linear(self.proj["q"], y).unflatten(-1, (self.heads, -1)))
        k = self.k_norm(_linear(self.proj["k"], y).unflatten(-1, (self.heads, -1)))
        v = _linear(self.proj["v"], y).unflatten(-1, (self.heads, -1))
        del y

        attn_out = _attention(
            q,
            k,
            v,
            rope,
            self.heads,
            attn_mask=attn_mask,
            iso_kv_spec=iso_kv_spec,
            target_attn_bias=target_attn_bias,
        )
        del q, k, v
        proj_o = _linear(self.proj["o"], attn_out)
        del attn_out
        if x.is_inference() and not x.is_contiguous():
            x = x.clone()
        x = _apply_gate_residual_inplace(x, proj_o, modulation, is_ffn=False)
        del proj_o

        y = self.norm2(x)
        _apply_mod_inplace(y, modulation, is_ffn=True)
        if y.dtype != dtype:
            y = y.to(dtype)
        seq_len = y.shape[1]
        if seq_len <= _FFN_TILE_TOKENS:
            hidden, gate = _linear(self.proj["ffn_up_gate"], y).chunk(2, -1)
            del y
            act = _situ_inplace(hidden, gate)
            del gate
            ffn_out = _linear(self.proj["ffn_down"], act)
            del act, hidden
        else:
            ffn_out = torch.empty_like(x)
            for start in range(0, seq_len, _FFN_TILE_TOKENS):
                end = min(start + _FFN_TILE_TOKENS, seq_len)
                y_tile = y[:, start:end, :]
                if y.shape[0] > 1:
                    y_tile = y_tile.contiguous()
                hidden, gate = _linear(
                    self.proj["ffn_up_gate"], y_tile
                ).chunk(2, -1)
                act = _situ_inplace(hidden, gate)
                del gate
                ffn_out[:, start:end, :] = _linear(self.proj["ffn_down"], act)
                del act, hidden
            del y

        return _apply_gate_residual_inplace(x, ffn_out, modulation, is_ffn=True)


__all__ = [
    "_slice_blhd_to_bhld",
    "_attention",
    "_apply_mod_inplace",
    "_apply_gate_residual_inplace",
    "_TextBlock",
    "_TextProjection",
    "_Block",
]