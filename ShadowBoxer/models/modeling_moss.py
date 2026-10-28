# ShadowBoxer/models/modeling_moss.py
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

_FP8_DTYPES: Set[torch.dtype] = {
    dtype
    for name in (
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
    )
    if (dtype := getattr(torch, name, None)) is not None
}

_FFN_TILE_TOKENS = 2048
_BOX_ISOLATION_LOGIT_SCALE = -20.0

CollabGroups = Tuple[Tuple[int, ...], ...]
IsoKVSpec = Union[
    Tuple[int, Tuple[int, ...], int],
    Tuple[int, Tuple[int, ...], int, CollabGroups],
]
NormBBox = Tuple[float, float, float, float]
ModTuple6 = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
PiecewiseModSpec = Tuple[ModTuple6, ModTuple6, int, int]


def _maybe_pin_cpu_tensor(t: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode(False):
        cpu_t = t.detach().to(device="cpu").clone().contiguous()
        if torch.cuda.is_available():
            try:
                return cpu_t.pin_memory()
            except Exception:
                return cpu_t
        return cpu_t


def _resolve_compute_dtype(
    weight: torch.Tensor, fallback: torch.dtype = torch.bfloat16
) -> torch.dtype:
    if weight.dtype in _FP8_DTYPES:
        return torch.bfloat16 if fallback in _FP8_DTYPES else fallback
    return weight.dtype


def _linear(layer: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    w = layer.weight
    if w.dtype in _FP8_DTYPES:
        compute_dtype = (
            x.dtype
            if x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            else torch.bfloat16
        )
        x_in = x if x.dtype == compute_dtype else x.to(compute_dtype)
        b = layer.bias.to(compute_dtype) if layer.bias is not None else None
        return F.linear(x_in, w.to(compute_dtype), b)
    if x.dtype != w.dtype:
        x = x.to(w.dtype)
    return layer(x)


@dataclass
class TransformerOutput:
    sample: torch.Tensor


@dataclass
class MOSSImageConfig:
    dim: int = 3072
    ffn_dim: int = 8192
    num_heads: int = 24
    num_layers: int = 42
    text_dim: int = 4096
    freq_dim: int = 256
    latent_channels: int = 128
    rope_base: float = 10000.0

    @classmethod
    def from_json_file(cls, path: Union[str, Path]) -> MOSSImageConfig:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in raw.items() if k in valid}
        return cls(**filtered)


class _RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.normalized_shape = (dim,)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight if self.weight.dtype == x.dtype else self.weight.to(x.dtype)
        return F.rms_norm(x, self.normalized_shape, w, 1e-6)


def _situ(hidden: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (
        25.0 * torch.tanh(hidden / 25.0)
    )


def _situ_inplace(hidden: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    sig_g = torch.sigmoid(gate)
    gate.mul_(0.25).tanh_().mul_(4.0).mul_(sig_g)
    del sig_g
    hidden.mul_(0.04).tanh_().mul_(25.0).mul_(gate)
    return hidden


def validate_skip_layers(
    layers: Optional[Iterable[int]], depth: int, name: str = "skip_layers"
) -> Set[int]:
    if layers is None:
        return set()
    parsed = tuple(int(x) for x in layers)
    if any(not (0 <= idx < depth) for idx in parsed):
        raise ValueError(f"{name} must contain integer block indices in [0, {depth}).")
    return set(parsed)


def _normalize_ref_bboxes(
    ref_shapes: Optional[List[Tuple[int, int]]],
    ref_bboxes: Optional[Sequence[Optional[Sequence[float]]]],
) -> Tuple[Optional[NormBBox], ...]:
    if not ref_shapes:
        return ()
    if not ref_bboxes:
        return tuple(None for _ in ref_shapes)
    out: List[Optional[NormBBox]] = []
    for idx in range(len(ref_shapes)):
        b = ref_bboxes[idx] if idx < len(ref_bboxes) else None
        if b is None or len(b) != 4:
            out.append(None)
        else:
            y0 = max(0.0, min(1.0, float(b[0])))
            x0 = max(0.0, min(1.0, float(b[1])))
            y1 = max(y0, min(1.0, float(b[2])))
            x1 = max(x0, min(1.0, float(b[3])))
            if y0 <= 1e-5 and x0 <= 1e-5 and y1 >= 1.0 - 1e-5 and x1 >= 1.0 - 1e-5:
                out.append(None)
            else:
                out.append((round(y0, 6), round(x0, 6), round(y1, 6), round(x1, 6)))
    return tuple(out)


def _boxes_intersect(box_a: Optional[NormBBox], box_b: Optional[NormBBox]) -> bool:
    if box_a is None or box_b is None:
        return False
    ay0, ax0, ay1, ax1 = box_a
    by0, bx0, by1, bx1 = box_b
    inter_h = min(ay1, by1) - max(ay0, by0)
    inter_w = min(ax1, bx1) - max(ax0, bx0)
    return (inter_h > 1e-5) and (inter_w > 1e-5)


def _compute_collaborating_ref_groups(
    num_refs: int,
    norm_boxes: Tuple[Optional[NormBBox], ...],
) -> CollabGroups:
    if num_refs <= 0:
        return ()
    groups: List[Tuple[int, ...]] = []
    for i in range(num_refs):
        box_i = norm_boxes[i] if i < len(norm_boxes) else None
        if box_i is None:
            groups.append((i,))
            continue
        peers: List[int] = []
        for j in range(num_refs):
            if i == j:
                peers.append(j)
            else:
                box_j = norm_boxes[j] if j < len(norm_boxes) else None
                if _boxes_intersect(box_i, box_j):
                    peers.append(j)
        groups.append(tuple(peers))
    return tuple(groups)


def _normalize_box_isolation(
    norm_boxes: Tuple[Optional[NormBBox], ...],
    reference_box_isolation: Optional[Union[float, Sequence[Optional[float]]]],
) -> Tuple[float, ...]:
    if not norm_boxes or reference_box_isolation is None:
        return tuple(0.0 for _ in norm_boxes)
    if isinstance(reference_box_isolation, (int, float)):
        global_val = max(0.0, min(1.0, float(reference_box_isolation)))
        return tuple(
            round(global_val, 6) if b is not None else 0.0 for b in norm_boxes
        )
    out: List[float] = []
    for idx, b in enumerate(norm_boxes):
        if b is None or idx >= len(reference_box_isolation):
            out.append(0.0)
        else:
            raw = reference_box_isolation[idx]
            val = max(0.0, min(1.0, float(raw))) if raw is not None else 0.0
            out.append(round(val, 6))
    return tuple(out)


def _rotate(
    x: torch.Tensor,
    rope: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    if isinstance(rope, tuple):
        sin_signed, cos = rope
    else:
        sin, cos = rope.chunk(2, -1)
        sin_signed = torch.cat((-sin, sin), -1).unsqueeze(-2)
        cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return torch.addcmul(x * cos, torch.cat((b, a), -1), sin_signed)


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
            target_tokens, ref_token_lengths, _, collab_groups = iso_kv_spec  # type: ignore[misc]
        else:
            target_tokens, ref_token_lengths, _ = iso_kv_spec  # type: ignore[misc]
            collab_groups = tuple((i,) for i in range(len(ref_token_lengths)))
        total_ref = sum(ref_token_lengths)
        if total_ref > 0:
            b_sz, total_seq, _, head_dim = q_rot.shape
            text_start = target_tokens + total_ref
            text_len = total_seq - text_start
            if not v.is_contiguous():
                v = v.contiguous()
            out_blhd = torch.empty_like(q_rot, memory_format=torch.contiguous_format)
            k_full = k_rot.transpose(1, 2)
            v_full = v.transpose(1, 2)

            q_target = (
                q_rot[:, :target_tokens, :, :]
                .view(b_sz, target_tokens, heads, head_dim)
                .transpose(1, 2)
            )
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
                q_text = (
                    q_rot[:, text_start:, :, :]
                    .view(b_sz, text_len, heads, head_dim)
                    .transpose(1, 2)
                )
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

            for r_idx, (c_start, c_end) in enumerate(ref_spans):
                r_len = c_end - c_start
                if r_len <= 0:
                    continue
                q_ref = (
                    q_rot[:, c_start:c_end, :, :]
                    .view(b_sz, r_len, heads, head_dim)
                    .transpose(1, 2)
                )
                peers = (
                    collab_groups[r_idx]
                    if r_idx < len(collab_groups)
                    else (r_idx,)
                )
                if len(peers) == 1 and peers[0] == r_idx and c_end == text_start:
                    rem_len = total_seq - c_start
                    k_ref = (
                        k_rot[:, c_start:, :, :]
                        .view(b_sz, rem_len, heads, head_dim)
                        .transpose(1, 2)
                    )
                    v_ref = (
                        v[:, c_start:, :, :]
                        .view(b_sz, rem_len, heads, head_dim)
                        .transpose(1, 2)
                    )
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
        b_sz, total_seq, _, head_dim = q_rot.shape
        target_tokens = int(target_attn_bias.shape[2])
        if 0 < target_tokens < total_seq:
            if not v.is_contiguous():
                v = v.contiguous()
            out_blhd = torch.empty_like(q_rot, memory_format=torch.contiguous_format)
            k_full = k_rot.transpose(1, 2)
            v_full = v.transpose(1, 2)
            q_target = (
                q_rot[:, :target_tokens, :, :]
                .view(b_sz, target_tokens, heads, head_dim)
                .transpose(1, 2)
            )
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
            rem_tokens = total_seq - target_tokens
            q_rest = (
                q_rot[:, target_tokens:, :, :]
                .view(b_sz, rem_tokens, heads, head_dim)
                .transpose(1, 2)
            )
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
        x = _apply_gate_residual_inplace(x.clone(), proj_o, modulation, is_ffn=False)
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
                hidden, gate = _linear(
                    self.proj["ffn_up_gate"], y[:, start:end, :]
                ).chunk(2, -1)
                act = _situ_inplace(hidden, gate)
                del gate
                ffn_out[:, start:end, :] = _linear(self.proj["ffn_down"], act)
                del act, hidden
            del y

        return _apply_gate_residual_inplace(x, ffn_out, modulation, is_ffn=True)


class _RoPE(nn.Module):
    def __init__(self, head_dim: int, base: float):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        self.register_buffer("bands", self._default_bands(torch.device("cpu")))
        self._phase_cache: OrderedDict[
            Tuple[Any, ...],
            torch.Tensor,
        ] = OrderedDict()
        self._expanded_cache: OrderedDict[
            Tuple[Any, ...],
            Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        ] = OrderedDict()

    def _default_bands(self, device: torch.device) -> torch.Tensor:
        exponent = (
            torch.arange(self.num_bands, device="cpu", dtype=torch.float32)
            / self.num_bands
        )
        return (1.0 / (self.base**exponent)).to(device)

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.bands = self._default_bands(self.bands.device)
        self.clear_cache()
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.bands = self._default_bands(self.bands.device)
        self.clear_cache()

    def clear_cache(self) -> None:
        self._phase_cache.clear()
        self._expanded_cache.clear()

    def _compute_phase(
        self,
        height: int,
        width: int,
        text_length: int,
        device: torch.device,
        ref_shapes: Optional[List[Tuple[int, int]]] = None,
        ref_bboxes: Optional[Sequence[Optional[Sequence[float]]]] = None,
    ) -> torch.Tensor:
        if self.bands.dtype != torch.float32 or self.bands.device != device:
            self.bands = self._default_bands(device)
        cy = (height - 1) / 2.0
        cx = (width - 1) / 2.0
        h, w = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        image = (
            torch.stack(
                (torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1
            )
            .flatten(0, 1)
            .float()
        )
        image[:, 2] -= cy
        image[:, 3] -= cx
        coord_parts = [image]

        if ref_shapes:
            norm_boxes = _normalize_ref_bboxes(ref_shapes, ref_bboxes)
            h_f = float(height)
            w_f = float(width)
            for r_idx, ((rh, rw), r_box) in enumerate(
                zip(ref_shapes, norm_boxes), start=1
            ):
                if r_box is not None:
                    uy0, ux0, uy1, ux1 = r_box
                    row_centers = (
                        torch.arange(rh, device=device, dtype=torch.float32) + 0.5
                    ) / float(max(1, rh))
                    col_centers = (
                        torch.arange(rw, device=device, dtype=torch.float32) + 0.5
                    ) / float(max(1, rw))
                    u_y = uy0 + row_centers * (uy1 - uy0)
                    u_x = ux0 + col_centers * (ux1 - ux0)
                    y_axis = (u_y - 0.5) * h_f
                    x_axis = (u_x - 0.5) * w_f
                    gh, gw = torch.meshgrid(y_axis, x_axis, indexing="ij")
                    ref_coord = torch.stack(
                        (
                            torch.full_like(gh, float(r_idx)),
                            torch.full_like(gh, float(text_length)),
                            gh,
                            gw,
                        ),
                        -1,
                    ).flatten(0, 1)
                else:
                    gh, gw = torch.meshgrid(
                        torch.arange(rh, device=device),
                        torch.arange(rw, device=device),
                        indexing="ij",
                    )
                    ref_coord = (
                        torch.stack(
                            (
                                torch.full_like(gh, r_idx),
                                torch.full_like(gh, text_length),
                                gh,
                                gw,
                            ),
                            -1,
                        )
                        .flatten(0, 1)
                        .float()
                    )
                    ref_coord[:, 2] -= (rh - 1) / 2.0
                    ref_coord[:, 3] -= (rw - 1) / 2.0
                coord_parts.append(ref_coord)

        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        coord_parts.append(text)
        return torch.cat(coord_parts, 0).unsqueeze(-1) * self.bands

    def get_expanded_rope(
        self,
        height: int,
        width: int,
        text_length: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        ref_shapes: Optional[List[Tuple[int, int]]] = None,
        ref_bboxes: Optional[Sequence[Optional[Sequence[float]]]] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        ref_key = tuple(ref_shapes) if ref_shapes else ()
        bbox_key = _normalize_ref_bboxes(ref_shapes, ref_bboxes)
        cache_key = (height, width, text_length, ref_key, bbox_key, str(device), dtype)
        cached = self._expanded_cache.get(cache_key)
        if cached is not None:
            self._expanded_cache.move_to_end(cache_key)
            return cached

        phase = self._compute_phase(
            height, width, text_length, device, ref_shapes, bbox_key
        )
        sin_half = phase.sin().flatten(-2).to(dtype=dtype)
        cos_half = phase.cos().flatten(-2).to(dtype=dtype)
        sin_signed = (
            torch.cat((-sin_half, sin_half), dim=-1)[None, :, None, :].contiguous()
        )
        cos_full = torch.cat((cos_half, cos_half), dim=-1)[None, :, None, :].contiguous()
        total_img_tokens = sin_signed.shape[1] - text_length
        sin_text = sin_signed[:, total_img_tokens:, :, :]
        cos_text = cos_full[:, total_img_tokens:, :, :]
        pair = ((sin_signed, cos_full), (sin_text, cos_text))
        self._expanded_cache[cache_key] = pair
        while len(self._expanded_cache) > 16:
            self._expanded_cache.popitem(last=False)
        return pair

    def forward(
        self,
        height: int,
        width: int,
        text_length: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        ref_shapes: Optional[List[Tuple[int, int]]] = None,
        ref_bboxes: Optional[Sequence[Optional[Sequence[float]]]] = None,
    ) -> torch.Tensor:
        ref_key = tuple(ref_shapes) if ref_shapes else ()
        bbox_key = _normalize_ref_bboxes(ref_shapes, ref_bboxes)
        cache_key = (height, width, text_length, ref_key, bbox_key, str(device), dtype)
        cached = self._phase_cache.get(cache_key)
        if cached is not None:
            self._phase_cache.move_to_end(cache_key)
            return cached
        phase = self._compute_phase(
            height, width, text_length, device, ref_shapes, bbox_key
        )
        rope_out = (
            torch.cat((phase.sin(), phase.cos()), -2)
            .flatten(-2)
            .unsqueeze(0)
            .to(dtype=dtype)
        )
        self._phase_cache[cache_key] = rope_out
        while len(self._phase_cache) > 16:
            self._phase_cache.popitem(last=False)
        return rope_out


class MOSSImageTransformer(nn.Module):
    def __init__(
        self,
        config: Optional[MOSSImageConfig] = None,
        dim: int = 3072,
        ffn_dim: int = 8192,
        num_heads: int = 24,
        num_layers: int = 42,
        text_dim: int = 4096,
        freq_dim: int = 256,
        latent_channels: int = 128,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        if config is not None:
            dim = config.dim
            ffn_dim = config.ffn_dim
            num_heads = config.num_heads
            num_layers = config.num_layers
            text_dim = config.text_dim
            freq_dim = config.freq_dim
            latent_channels = config.latent_channels
            rope_base = config.rope_base
        if dim % num_heads or (dim // num_heads) % 8 or freq_dim % 2:
            raise ValueError(
                "Require dim divisible by heads, head_dim divisible by 8, and even freq_dim."
            )
        self.config = MOSSImageConfig(
            dim=dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            text_dim=text_dim,
            freq_dim=freq_dim,
            latent_channels=latent_channels,
            rope_base=rope_base,
        )
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = _TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.rope = _RoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList(
            [_Block(dim, ffn_dim, num_heads) for _ in range(num_layers)]
        )
        self.final_norm_video = _RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

        self._lora_backups: Dict[str, torch.Tensor] = {}
        self._active_lora_tag: Optional[str] = None
        self._isolated_mask_cache: OrderedDict[
            Tuple[Any, ...], torch.Tensor
        ] = OrderedDict()
        self._box_aperture_bias_cache: OrderedDict[
            Tuple[Any, ...], torch.Tensor
        ] = OrderedDict()
        self._mod_zero_cache: OrderedDict[
            Tuple[str, torch.dtype, Optional[str]], torch.Tensor
        ] = OrderedDict()
        self._time_mod_cache: OrderedDict[
            Tuple[Any, ...],
            Tuple[Any, torch.Tensor, torch.Tensor],
        ] = OrderedDict()
        self._text_proj_cache: OrderedDict[
            Tuple[Any, ...],
            Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor],
        ] = OrderedDict()
        self._ref_proj_cache: OrderedDict[
            Tuple[Any, ...],
            Tuple[torch.Tensor, torch.Tensor],
        ] = OrderedDict()
        self._mod_seq_buf: Optional[torch.Tensor] = None

    @classmethod
    def from_config_dir(cls, model_dir: Union[str, Path]) -> MOSSImageTransformer:
        cfg_file = Path(model_dir) / "config.json"
        if cfg_file.is_file():
            return cls(config=MOSSImageConfig.from_json_file(cfg_file))
        return cls()

    @staticmethod
    def format_lora_tag(lora_tag: str, scale: float) -> str:
        return f"{lora_tag}:{float(scale):.4f}"

    def is_lora_active(self, lora_tag: str, scale: float) -> bool:
        if abs(float(scale)) < 1e-7:
            return self._active_lora_tag is None
        return (
            self._active_lora_tag == self.format_lora_tag(lora_tag, scale)
            and bool(self._lora_backups)
        )

    def clear_transient_caches(self) -> None:
        self._time_mod_cache.clear()
        self._text_proj_cache.clear()
        self._ref_proj_cache.clear()
        self._mod_seq_buf = None

    def clear_caches(self) -> None:
        self.rope.clear_cache()
        self._isolated_mask_cache.clear()
        self._box_aperture_bias_cache.clear()
        self._mod_zero_cache.clear()
        self.clear_transient_caches()

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.clear_caches()
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.clear_caches()

    @torch.no_grad()
    def unfuse_lora(self) -> None:
        if not self._lora_backups:
            self._active_lora_tag = None
            return
        param_dict = dict(self.named_parameters())
        touched_device: Optional[torch.device] = None
        for name, orig_cpu in self._lora_backups.items():
            if name in param_dict:
                p = param_dict[name]
                touched_device = p.device
                if p.dtype == orig_cpu.dtype:
                    p.data.copy_(orig_cpu, non_blocking=True)
                else:
                    p.data.copy_(
                        orig_cpu.to(device=p.device, dtype=p.dtype, non_blocking=True)
                    )
        if touched_device is not None and touched_device.type == "cuda":
            torch.cuda.synchronize(touched_device)
        self._lora_backups.clear()
        self._active_lora_tag = None
        self._mod_zero_cache.clear()
        self.clear_transient_caches()

    @torch.no_grad()
    def fuse_lora_state_dict(
        self,
        lora_state: Dict[str, torch.Tensor],
        scale: float = 1.0,
        lora_tag: str = "custom",
    ) -> int:
        target_tag = self.format_lora_tag(lora_tag, scale)
        if self._active_lora_tag == target_tag:
            return len(self._lora_backups)
        if self._lora_backups:
            self.unfuse_lora()
        if abs(scale) < 1e-7:
            return 0

        param_dict = dict(self.named_parameters())
        down_keys = [
            k
            for k in lora_state.keys()
            if k.endswith(".lora_A.weight") or k.endswith(".lora_down.weight")
        ]
        fused_count = 0
        for d_key in down_keys:
            if d_key.endswith(".lora_A.weight"):
                prefix = d_key[: -len(".lora_A.weight")]
                u_key = prefix + ".lora_B.weight"
            else:
                prefix = d_key[: -len(".lora_down.weight")]
                u_key = prefix + ".lora_up.weight"
            if u_key not in lora_state:
                continue

            clean_prefix = prefix
            for strip_tok in (
                "base_model.model.",
                "diffusion_model.",
                "transformer.",
                "model.",
            ):
                if clean_prefix.startswith(strip_tok):
                    clean_prefix = clean_prefix[len(strip_tok) :]

            target_param_name = f"{clean_prefix}.weight"
            if target_param_name not in param_dict:
                continue

            param = param_dict[target_param_name]
            if target_param_name not in self._lora_backups:
                self._lora_backups[target_param_name] = _maybe_pin_cpu_tensor(param.data)

            lora_a = lora_state[d_key].to(device=param.device, dtype=torch.float32)
            lora_b = lora_state[u_key].to(device=param.device, dtype=torch.float32)
            alpha_key = prefix + ".alpha"
            rank = float(lora_a.shape[0])
            alpha = (
                float(lora_state[alpha_key].item())
                if alpha_key in lora_state
                else rank
            )
            eff_scale = float(scale) * (alpha / max(rank, 1.0))

            delta = (lora_b @ lora_a) * eff_scale
            if delta.shape == param.shape:
                if param.dtype in _FP8_DTYPES:
                    finfo = torch.finfo(param.dtype)
                    updated = (param.data.float() + delta).clamp(
                        float(finfo.min), float(finfo.max)
                    )
                    param.data.copy_(updated.to(param.dtype))
                else:
                    param.data.copy_((param.data.float() + delta).to(param.dtype))
                fused_count += 1

        self._active_lora_tag = target_tag
        self._mod_zero_cache.clear()
        self.clear_transient_caches()
        return fused_count

    def _compute_time_modulation(
        self,
        timestep: Union[float, int, torch.Tensor],
        batch: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        w_dtype = _resolve_compute_dtype(self.time_embedding[0].weight)
        if isinstance(timestep, (int, float)):
            cache_key: Tuple[Any, ...] = (
                "scalar",
                float(timestep),
                batch,
                str(device),
                w_dtype,
                self._active_lora_tag,
            )
        elif isinstance(timestep, torch.Tensor) and timestep.device.type == "cpu":
            t_vals = tuple(float(v) for v in timestep.detach().reshape(-1).tolist())
            cache_key = (
                "cpu_tensor",
                t_vals,
                batch,
                str(device),
                w_dtype,
                self._active_lora_tag,
            )
        elif isinstance(timestep, torch.Tensor):
            cache_key = (
                "dev_tensor",
                id(timestep),
                int(timestep.data_ptr()),
                tuple(timestep.shape),
                timestep.dtype,
                batch,
                str(device),
                w_dtype,
                self._active_lora_tag,
            )
        else:
            cache_key = (
                "other",
                id(timestep),
                batch,
                str(device),
                w_dtype,
                self._active_lora_tag,
            )
        cached = self._time_mod_cache.get(cache_key)
        if cached is not None:
            self._time_mod_cache.move_to_end(cache_key)
            return cached[1], cached[2]

        times = torch.as_tensor(
            timestep, device=device, dtype=torch.float32
        ).reshape(-1)
        times = times.expand(batch)
        half = self.config.freq_dim // 2
        freq = 10000.0 ** (
            -torch.arange(half, device=device, dtype=torch.float64) / half
        )
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).to(w_dtype)
        h_time = F.silu(_linear(self.time_embedding[0], embedding))
        time_vec = _linear(self.time_embedding[2], h_time).float()
        mod_in = time_vec.to(
            _resolve_compute_dtype(self.shared_block_modulation.weight, w_dtype)
        )
        head_in = time_vec.to(
            _resolve_compute_dtype(self.video_head_modulation.weight, w_dtype)
        )
        mod = _linear(self.shared_block_modulation, mod_in).view(batch, 6, -1)
        head = _linear(self.video_head_modulation, head_in).view(batch, 2, -1)
        self._time_mod_cache[cache_key] = (timestep, mod, head)
        while len(self._time_mod_cache) > 4:
            self._time_mod_cache.popitem(last=False)
        return mod, head

    def _get_cached_zero_modulation(self, device: torch.device) -> torch.Tensor:
        w_dtype = _resolve_compute_dtype(self.time_embedding[0].weight)
        key = (str(device), w_dtype, self._active_lora_tag)
        cached = self._mod_zero_cache.get(key)
        if cached is not None:
            self._mod_zero_cache.move_to_end(key)
            return cached
        mod_zero, _ = self._compute_time_modulation(0.0, 1, device)
        self._mod_zero_cache[key] = mod_zero
        while len(self._mod_zero_cache) > 4:
            self._mod_zero_cache.popitem(last=False)
        return mod_zero

    def _get_or_compute_text_projection(
        self,
        encoder_hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor],
        index: int,
        act_dtype: torch.dtype,
        height: int,
        width: int,
        ref_shapes: Optional[List[Tuple[int, int]]],
        ref_bboxes: Optional[Sequence[Optional[Sequence[float]]]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], int]:
        mask_id = id(mask) if mask is not None else 0
        mask_ptr = int(mask.data_ptr()) if mask is not None else 0
        mask_shape = tuple(mask.shape) if mask is not None else ()
        key = (
            id(encoder_hidden_states),
            int(encoder_hidden_states.data_ptr()),
            tuple(encoder_hidden_states.shape),
            index,
            mask_id,
            mask_ptr,
            mask_shape,
            act_dtype,
            str(encoder_hidden_states.device),
            self._active_lora_tag,
        )
        entry = self._text_proj_cache.get(key)
        if entry is not None:
            cached_ehs, cached_mask, cached_text = entry
            if cached_ehs is encoder_hidden_states and cached_mask is mask:
                self._text_proj_cache.move_to_end(key)
                text_len = int(cached_text.shape[1])
                full_rope_4d, _ = self.rope.get_expanded_rope(
                    height,
                    width,
                    text_len,
                    encoder_hidden_states.device,
                    dtype=act_dtype,
                    ref_shapes=ref_shapes,
                    ref_bboxes=ref_bboxes,
                )
                return cached_text, full_rope_4d, text_len

        if mask is not None:
            context = encoder_hidden_states[index : index + 1, mask[index].bool()]
        else:
            context = encoder_hidden_states[index : index + 1]
        text_len = int(context.shape[1])
        if text_len == 0:
            raise ValueError("Each sample needs at least one valid text token.")

        full_rope_4d, text_rope_4d = self.rope.get_expanded_rope(
            height,
            width,
            text_len,
            encoder_hidden_states.device,
            dtype=act_dtype,
            ref_shapes=ref_shapes,
            ref_bboxes=ref_bboxes,
        )
        text = self.text_embedder(
            context if context.dtype == act_dtype else context.to(act_dtype),
            text_rope_4d,
        )
        self._text_proj_cache[key] = (encoder_hidden_states, mask, text)
        while len(self._text_proj_cache) > 8:
            self._text_proj_cache.popitem(last=False)
        return text, full_rope_4d, text_len

    def _get_or_compute_ref_projection(
        self,
        ref_lat: torch.Tensor,
        index: int,
        act_dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        eff_idx = index if ref_lat.shape[0] > index else 0
        key = (
            id(ref_lat),
            int(ref_lat.data_ptr()),
            tuple(ref_lat.shape),
            eff_idx,
            act_dtype,
            str(device),
            self._active_lora_tag,
        )
        entry = self._ref_proj_cache.get(key)
        if entry is not None:
            cached_ref, cached_emb = entry
            if cached_ref is ref_lat:
                self._ref_proj_cache.move_to_end(key)
                return cached_emb

        r_slice = ref_lat[eff_idx : eff_idx + 1]
        r_flat = (
            r_slice.flatten(2)
            .transpose(1, 2)
            .to(device=device, dtype=act_dtype)
            .contiguous()
        )
        r_emb = _linear(self.video_embedder, r_flat)
        self._ref_proj_cache[key] = (ref_lat, r_emb)
        while len(self._ref_proj_cache) > 8:
            self._ref_proj_cache.popitem(last=False)
        return r_emb

    def _acquire_mod_seq_buf(
        self, num_regimes: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        dim = self.config.dim
        alloc_regimes = max(2, int(num_regimes))
        if (
            self._mod_seq_buf is None
            or self._mod_seq_buf.device != device
            or self._mod_seq_buf.dtype != dtype
            or self._mod_seq_buf.shape[2] < alloc_regimes
            or self._mod_seq_buf.shape[3] != dim
        ):
            self._mod_seq_buf = torch.empty(
                (1, 6, alloc_regimes, dim), device=device, dtype=dtype
            )
        return self._mod_seq_buf[:, :, :alloc_regimes, :]

    @staticmethod
    def _build_isolated_reference_mask(
        target_tokens: int,
        ref_token_lengths: List[int],
        text_tokens: int,
        device: torch.device,
        collab_groups: Optional[CollabGroups] = None,
    ) -> torch.Tensor:
        total_ref = sum(ref_token_lengths)
        total_seq = target_tokens + total_ref + text_tokens
        mask = torch.zeros(
            (1, 1, total_seq, total_seq), dtype=torch.bool, device=device
        )
        mask[:, :, :target_tokens, :] = True
        text_start = target_tokens + total_ref
        mask[:, :, text_start:, :] = True

        ref_spans: List[Tuple[int, int]] = []
        cursor = target_tokens
        for r_len in ref_token_lengths:
            ref_spans.append((cursor, cursor + r_len))
            cursor += r_len

        for r_idx, (c_start, c_end) in enumerate(ref_spans):
            mask[:, :, c_start:c_end, c_start:c_end] = True
            mask[:, :, c_start:c_end, text_start:] = True
            if collab_groups and r_idx < len(collab_groups):
                for peer_idx in collab_groups[r_idx]:
                    if 0 <= peer_idx < len(ref_spans):
                        p_start, p_end = ref_spans[peer_idx]
                        mask[:, :, c_start:c_end, p_start:p_end] = True
        return mask

    def _get_isolated_reference_mask(
        self,
        target_tokens: int,
        ref_token_lengths: List[int],
        text_tokens: int,
        device: torch.device,
        ref_bboxes: Optional[Sequence[Optional[Sequence[float]]]] = None,
    ) -> torch.Tensor:
        dummy_shapes = [(1, 1) for _ in ref_token_lengths]
        norm_boxes = _normalize_ref_bboxes(dummy_shapes, ref_bboxes)
        collab_groups = _compute_collaborating_ref_groups(
            len(ref_token_lengths), norm_boxes
        )
        key = (
            target_tokens,
            tuple(ref_token_lengths),
            text_tokens,
            collab_groups,
            str(device),
        )
        cached = self._isolated_mask_cache.get(key)
        if cached is not None:
            self._isolated_mask_cache.move_to_end(key)
            return cached
        mask = self._build_isolated_reference_mask(
            target_tokens=target_tokens,
            ref_token_lengths=ref_token_lengths,
            text_tokens=text_tokens,
            device=device,
            collab_groups=collab_groups,
        )
        self._isolated_mask_cache[key] = mask
        while len(self._isolated_mask_cache) > 8:
            self._isolated_mask_cache.popitem(last=False)
        return mask

    @staticmethod
    def _build_box_aperture_bias(
        height: int,
        width: int,
        ref_token_lengths: Sequence[int],
        text_tokens: int,
        norm_boxes: Tuple[Optional[NormBBox], ...],
        iso_tuple: Tuple[float, ...],
        feather_radius_lat: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        target_tokens = height * width
        total_ref = sum(ref_token_lengths)
        total_seq = target_tokens + total_ref + text_tokens
        bias = torch.zeros(
            (1, 1, target_tokens, total_seq), dtype=dtype, device=device
        )

        yy = torch.arange(height, device=device, dtype=torch.float32).unsqueeze(1) + 0.5
        xx = torch.arange(width, device=device, dtype=torch.float32).unsqueeze(0) + 0.5
        feather = max(0.25, float(feather_radius_lat))

        cursor = target_tokens
        for r_idx, r_len in enumerate(ref_token_lengths):
            r_box = norm_boxes[r_idx] if r_idx < len(norm_boxes) else None
            iso_val = iso_tuple[r_idx] if r_idx < len(iso_tuple) else 0.0
            if r_box is not None and iso_val > 1e-6 and r_len > 0:
                uy0, ux0, uy1, ux1 = r_box
                ly0 = uy0 * height
                ly1 = uy1 * height
                lx0 = ux0 * width
                lx1 = ux1 * width
                dy = torch.maximum(
                    torch.maximum(ly0 - yy, yy - ly1), torch.zeros_like(yy)
                )
                dx = torch.maximum(
                    torch.maximum(lx0 - xx, xx - lx1), torch.zeros_like(xx)
                )
                dist = torch.sqrt(dy * dy + dx * dx)
                w_inside = 0.5 * (
                    1.0 + torch.cos(math.pi * (dist / feather).clamp(0.0, 1.0))
                )
                attenuation = (
                    (_BOX_ISOLATION_LOGIT_SCALE * float(iso_val))
                    * (1.0 - w_inside)
                ).reshape(1, 1, target_tokens, 1).to(dtype=dtype)
                bias[:, :, :, cursor : cursor + r_len] = attenuation.expand(
                    1, 1, target_tokens, r_len
                )
            cursor += r_len
        return bias

    def _get_box_aperture_bias(
        self,
        height: int,
        width: int,
        ref_token_lengths: Sequence[int],
        text_tokens: int,
        ref_shapes: Optional[List[Tuple[int, int]]],
        ref_bboxes: Optional[Sequence[Optional[Sequence[float]]]],
        reference_box_isolation: Optional[Union[float, Sequence[Optional[float]]]],
        feather_radius_lat: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        norm_boxes = _normalize_ref_bboxes(ref_shapes, ref_bboxes)
        iso_tuple = _normalize_box_isolation(norm_boxes, reference_box_isolation)
        if not any(v > 1e-6 for v in iso_tuple):
            return None

        key = (
            int(height),
            int(width),
            tuple(int(x) for x in ref_token_lengths),
            int(text_tokens),
            norm_boxes,
            iso_tuple,
            round(float(feather_radius_lat), 4),
            str(device),
            dtype,
        )
        cached = self._box_aperture_bias_cache.get(key)
        if cached is not None:
            self._box_aperture_bias_cache.move_to_end(key)
            return cached

        bias = self._build_box_aperture_bias(
            height=height,
            width=width,
            ref_token_lengths=ref_token_lengths,
            text_tokens=text_tokens,
            norm_boxes=norm_boxes,
            iso_tuple=iso_tuple,
            feather_radius_lat=feather_radius_lat,
            device=device,
            dtype=dtype,
        )
        self._box_aperture_bias_cache[key] = bias
        while len(self._box_aperture_bias_cache) > 8:
            self._box_aperture_bias_cache.popitem(last=False)
        return bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Union[float, int, torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        skip_layers: Optional[Iterable[int]] = None,
        reference_latents: Optional[List[torch.Tensor]] = None,
        reference_zero_timestep: bool = False,
        reference_isolated_kv: bool = False,
        reference_bboxes: Optional[Sequence[Optional[Sequence[float]]]] = None,
        reference_box_isolation: Optional[Union[float, Sequence[Optional[float]]]] = None,
        box_feather_radius: float = 1.25,
    ) -> Union[TransformerOutput, Tuple[torch.Tensor, ...]]:
        skip_set = validate_skip_layers(skip_layers, len(self.layers))
        if (
            hidden_states.ndim != 4
            or hidden_states.shape[1] != self.config.latent_channels
        ):
            raise ValueError("hidden_states must be [B, latent_channels, H, W].")
        batch, _, height, width = hidden_states.shape
        if (
            encoder_hidden_states.ndim != 3
            or encoder_hidden_states.shape[0] != batch
        ):
            raise ValueError("encoder_hidden_states must be [B, S, text_dim].")
        mask = encoder_attention_mask
        mod, head = self._compute_time_modulation(
            timestep, batch, hidden_states.device
        )
        mod_zero: Optional[torch.Tensor] = None
        if reference_latents and reference_zero_timestep:
            mod_zero = self._get_cached_zero_modulation(hidden_states.device)

        ref_shapes: Optional[List[Tuple[int, int]]] = None
        if reference_latents:
            ref_shapes = [
                (int(r.shape[-2]), int(r.shape[-1])) for r in reference_latents
            ]

        norm_boxes = _normalize_ref_bboxes(ref_shapes, reference_bboxes)
        collab_groups = _compute_collaborating_ref_groups(
            len(ref_shapes) if ref_shapes else 0, norm_boxes
        )

        outputs: List[torch.Tensor] = []
        target_tokens = height * width
        act_dtype = _resolve_compute_dtype(self.video_embedder.weight)

        for index in range(batch):
            video = hidden_states[index : index + 1].flatten(2).transpose(1, 2)
            video = _linear(self.video_embedder, video.to(act_dtype).contiguous())
            seq_parts = [video]
            total_img_tokens = target_tokens
            ref_token_lengths: List[int] = []
            if reference_latents:
                for ref_lat in reference_latents:
                    r_emb = self._get_or_compute_ref_projection(
                        ref_lat, index, act_dtype, hidden_states.device
                    )
                    seq_parts.append(r_emb)
                    r_len = int(r_emb.shape[1])
                    ref_token_lengths.append(r_len)
                    total_img_tokens += r_len

            text, full_rope_4d, text_len = self._get_or_compute_text_projection(
                encoder_hidden_states=encoder_hidden_states,
                mask=mask,
                index=index,
                act_dtype=video.dtype,
                height=height,
                width=width,
                ref_shapes=ref_shapes,
                ref_bboxes=reference_bboxes,
            )
            seq_parts.append(text)
            x = torch.cat(seq_parts, 1)
            del seq_parts, video

            mod_i = mod[index : index + 1]
            if mod_zero is not None and total_img_tokens > target_tokens:
                mod_pair = self._acquire_mod_seq_buf(
                    2, mod_i.device, mod_i.dtype
                )
                mod_pair[:, :, 0, :].copy_(mod_i)
                mod_pair[:, 1, 0, :].add_(1.0)
                mod_pair[:, 4, 0, :].add_(1.0)
                mod_pair[:, :, 1, :].copy_(mod_zero)
                mod_pair[:, 1, 1, :].add_(1.0)
                mod_pair[:, 4, 1, :].add_(1.0)
                mod_t_tuple: ModTuple6 = tuple(  # type: ignore[assignment]
                    mod_pair[:, :, 0:1, :].unbind(1)
                )
                mod_z_tuple: ModTuple6 = tuple(  # type: ignore[assignment]
                    mod_pair[:, :, 1:2, :].unbind(1)
                )
                active_mod: Union[ModTuple6, PiecewiseModSpec] = (
                    mod_t_tuple,
                    mod_z_tuple,
                    target_tokens,
                    total_img_tokens,
                )
            else:
                shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (
                    mod_i[:, :, None, :].unbind(1)
                )
                active_mod = (
                    shift_a,
                    1 + scale_a,
                    gate_a,
                    shift_f,
                    1 + scale_f,
                    gate_f,
                )

            iso_kv_spec: Optional[IsoKVSpec] = None
            if reference_latents and reference_isolated_kv and ref_token_lengths:
                iso_kv_spec = (
                    target_tokens,
                    tuple(ref_token_lengths),
                    text_len,
                    collab_groups,
                )

            target_attn_bias: Optional[torch.Tensor] = None
            if (
                reference_latents
                and ref_token_lengths
                and reference_bboxes is not None
                and reference_box_isolation is not None
            ):
                target_attn_bias = self._get_box_aperture_bias(
                    height=height,
                    width=width,
                    ref_token_lengths=ref_token_lengths,
                    text_tokens=text_len,
                    ref_shapes=ref_shapes,
                    ref_bboxes=reference_bboxes,
                    reference_box_isolation=reference_box_isolation,
                    feather_radius_lat=box_feather_radius,
                    device=hidden_states.device,
                    dtype=x.dtype,
                )

            for layer_index, layer in enumerate(self.layers):
                if layer_index not in skip_set:
                    x = layer(
                        x,
                        full_rope_4d,
                        active_mod,
                        attn_mask=None,
                        iso_kv_spec=iso_kv_spec,
                        target_attn_bias=target_attn_bias,
                    )
            x = self.final_norm_video(
                x[:, :target_tokens].to(self.final_norm_video.weight.dtype)
            )
            x = (
                x * (1 + head[index : index + 1, 1, None])
                + head[index : index + 1, 0, None]
            )
            x = _linear(self.final_linear_video, x.to(act_dtype))
            outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
        sample = torch.cat(outputs, dim=0)
        return TransformerOutput(sample=sample) if return_dict else (sample,)