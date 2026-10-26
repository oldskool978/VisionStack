# ShadowBoxer/models/modeling_moss.py
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
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

IsoKVSpec = Tuple[int, Tuple[int, ...], int]
NormBBox = Tuple[float, float, float, float]


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
    return x * cos + torch.cat((b, a), -1) * sin_signed


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rope: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    heads: int,
    attn_mask: Optional[torch.Tensor] = None,
    iso_kv_spec: Optional[IsoKVSpec] = None,
) -> torch.Tensor:
    if q.ndim == 3:
        q = q.unflatten(-1, (heads, -1))
    if k.ndim == 3:
        k = k.unflatten(-1, (heads, -1))
    if v.ndim == 3:
        v = v.unflatten(-1, (heads, -1))
    q = _rotate(q, rope).transpose(1, 2)
    k = _rotate(k, rope).transpose(1, 2)
    v = v.transpose(1, 2)
    q_in = q if q.dtype == v.dtype else q.to(v.dtype)
    k_in = k if k.dtype == v.dtype else k.to(v.dtype)
    v_in = v

    if iso_kv_spec is not None and attn_mask is None:
        target_tokens, ref_token_lengths, _ = iso_kv_spec
        total_ref = sum(ref_token_lengths)
        if total_ref > 0:
            text_start = target_tokens + total_ref
            q_c = q_in.contiguous()
            k_c = k_in.contiguous()
            v_c = v_in.contiguous()
            del q_in, k_in, v_in, q, k, v
            out_buf = torch.empty_like(q_c, memory_format=torch.contiguous_format)

            q_main = torch.cat(
                (q_c[:, :, :target_tokens, :], q_c[:, :, text_start:, :]), dim=2
            )
            out_main = F.scaled_dot_product_attention(
                q_main, k_c, v_c, attn_mask=None
            )
            out_buf[:, :, :target_tokens, :].copy_(out_main[:, :, :target_tokens, :])
            out_buf[:, :, text_start:, :].copy_(out_main[:, :, target_tokens:, :])
            del q_main, out_main

            k_text = k_c[:, :, text_start:, :]
            v_text = v_c[:, :, text_start:, :]
            cursor = target_tokens
            for r_len in ref_token_lengths:
                next_cursor = cursor + r_len
                q_ref = q_c[:, :, cursor:next_cursor, :]
                if next_cursor == text_start:
                    k_ref = k_c[:, :, cursor:, :]
                    v_ref = v_c[:, :, cursor:, :]
                    out_ref = F.scaled_dot_product_attention(
                        q_ref, k_ref, v_ref, attn_mask=None
                    )
                else:
                    k_ref = torch.cat(
                        (k_c[:, :, cursor:next_cursor, :], k_text), dim=2
                    )
                    v_ref = torch.cat(
                        (v_c[:, :, cursor:next_cursor, :], v_text), dim=2
                    )
                    out_ref = F.scaled_dot_product_attention(
                        q_ref, k_ref, v_ref, attn_mask=None
                    )
                    del k_ref, v_ref
                out_buf[:, :, cursor:next_cursor, :].copy_(out_ref)
                del out_ref
                cursor = next_cursor

            del q_c, k_c, v_c
            return out_buf.transpose(1, 2).flatten(2)

    return (
        F.scaled_dot_product_attention(
            q_in,
            k_in,
            v_in,
            attn_mask=attn_mask,
        )
        .transpose(1, 2)
        .flatten(2)
    )


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
            self.ffn_down, _situ(_linear(self.ffn_up, y), _linear(self.ffn_gate, y))
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
        modulation: Union[
            torch.Tensor,
            Tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ],
        attn_mask: Optional[torch.Tensor] = None,
        iso_kv_spec: Optional[IsoKVSpec] = None,
    ) -> torch.Tensor:
        if isinstance(modulation, tuple):
            shift_a, scale_a_p1, gate_a, shift_f, scale_f_p1, gate_f = modulation
        else:
            if modulation.ndim == 3:
                shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (
                    modulation[:, :, None, :].unbind(1)
                )
            else:
                shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
            scale_a_p1 = 1 + scale_a
            scale_f_p1 = 1 + scale_f

        y = self.norm1(x) * scale_a_p1 + shift_a
        dtype = _resolve_compute_dtype(self.proj["q"].weight, x.dtype)
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
        )
        del q, k, v
        x = x + _linear(self.proj["o"], attn_out) * gate_a
        del attn_out

        y = self.norm2(x) * scale_f_p1 + shift_f
        if y.dtype != dtype:
            y = y.to(dtype)
        hidden, gate = _linear(self.proj["ffn_up_gate"], y).chunk(2, -1)
        del y
        act = _situ(hidden, gate)
        del hidden, gate
        return x + _linear(self.proj["ffn_down"], act) * gate_f


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
            for r_idx, ((rh, rw), r_box) in enumerate(
                zip(ref_shapes, norm_boxes), start=1
            ):
                if r_box is not None:
                    uy0, ux0, uy1, ux1 = r_box
                    y_start = uy0 * (height - 1) - cy
                    y_end = uy1 * (height - 1) - cy
                    x_start = ux0 * (width - 1) - cx
                    x_end = ux1 * (width - 1) - cx
                    if rh > 1:
                        y_axis = torch.linspace(
                            y_start, y_end, rh, device=device, dtype=torch.float32
                        )
                    else:
                        y_axis = torch.tensor(
                            [0.5 * (y_start + y_end)],
                            device=device,
                            dtype=torch.float32,
                        )
                    if rw > 1:
                        x_axis = torch.linspace(
                            x_start, x_end, rw, device=device, dtype=torch.float32
                        )
                    else:
                        x_axis = torch.tensor(
                            [0.5 * (x_start + x_end)],
                            device=device,
                            dtype=torch.float32,
                        )
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
            Tuple[int, Tuple[int, ...], int, str], torch.Tensor
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
        self, total_seq: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        dim = self.config.dim
        alloc_seq = ((total_seq + 255) // 256) * 256
        if (
            self._mod_seq_buf is None
            or self._mod_seq_buf.device != device
            or self._mod_seq_buf.dtype != dtype
            or self._mod_seq_buf.shape[2] < total_seq
            or self._mod_seq_buf.shape[2] > alloc_seq * 2
            or self._mod_seq_buf.shape[3] != dim
        ):
            self._mod_seq_buf = torch.empty(
                (1, 6, alloc_seq, dim), device=device, dtype=dtype
            )
        return self._mod_seq_buf[:, :, :total_seq, :]

    @staticmethod
    def _build_isolated_reference_mask(
        target_tokens: int,
        ref_token_lengths: List[int],
        text_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_ref = sum(ref_token_lengths)
        total_seq = target_tokens + total_ref + text_tokens
        mask = torch.zeros((1, 1, total_seq, total_seq), dtype=torch.bool, device=device)
        mask[:, :, :target_tokens, :] = True
        text_start = target_tokens + total_ref
        mask[:, :, text_start:, :] = True
        cursor = target_tokens
        for r_len in ref_token_lengths:
            mask[:, :, cursor : cursor + r_len, cursor : cursor + r_len] = True
            mask[:, :, cursor : cursor + r_len, text_start:] = True
            cursor += r_len
        return mask

    def _get_isolated_reference_mask(
        self,
        target_tokens: int,
        ref_token_lengths: List[int],
        text_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        key = (target_tokens, tuple(ref_token_lengths), text_tokens, str(device))
        cached = self._isolated_mask_cache.get(key)
        if cached is not None:
            self._isolated_mask_cache.move_to_end(key)
            return cached
        mask = self._build_isolated_reference_mask(
            target_tokens=target_tokens,
            ref_token_lengths=ref_token_lengths,
            text_tokens=text_tokens,
            device=device,
        )
        self._isolated_mask_cache[key] = mask
        while len(self._isolated_mask_cache) > 8:
            self._isolated_mask_cache.popitem(last=False)
        return mask

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
                total_seq = total_img_tokens + text_len
                mod_i_p1 = mod_i.clone()
                mod_i_p1[:, 1, :].add_(1.0)
                mod_i_p1[:, 4, :].add_(1.0)
                mod_z_p1 = mod_zero.clone()
                mod_z_p1[:, 1, :].add_(1.0)
                mod_z_p1[:, 4, :].add_(1.0)

                mod_seq = self._acquire_mod_seq_buf(
                    total_seq, mod_i.device, mod_i.dtype
                )
                mod_seq[:, :, :target_tokens, :].copy_(mod_i_p1[:, :, None, :])
                mod_seq[:, :, target_tokens:total_img_tokens, :].copy_(
                    mod_z_p1[:, :, None, :]
                )
                mod_seq[:, :, total_img_tokens:, :].copy_(mod_i_p1[:, :, None, :])
                del mod_i_p1, mod_z_p1
                shift_a, scale_a_p1, gate_a, shift_f, scale_f_p1, gate_f = (
                    mod_seq.unbind(1)
                )
            else:
                shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (
                    mod_i[:, :, None, :].unbind(1)
                )
                scale_a_p1 = 1 + scale_a
                scale_f_p1 = 1 + scale_f

            active_mod = (
                shift_a,
                scale_a_p1,
                gate_a,
                shift_f,
                scale_f_p1,
                gate_f,
            )

            iso_kv_spec: Optional[IsoKVSpec] = None
            if reference_latents and reference_isolated_kv and ref_token_lengths:
                iso_kv_spec = (target_tokens, tuple(ref_token_lengths), text_len)

            for layer_index, layer in enumerate(self.layers):
                if layer_index not in skip_set:
                    x = layer(
                        x,
                        full_rope_4d,
                        active_mod,
                        attn_mask=None,
                        iso_kv_spec=iso_kv_spec,
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