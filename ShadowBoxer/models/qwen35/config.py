# ShadowBoxer/models/qwen35/config.py
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from .cache import Qwen3_5HybridCache

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

_INV_LN2: float = 1.0 / math.log(2.0)

_STRICTLY_UPPER_MASK_CACHE: OrderedDict[Tuple[int, str], torch.Tensor] = OrderedDict()
_EYE_MATRIX_CACHE: OrderedDict[Tuple[int, torch.dtype, str], torch.Tensor] = OrderedDict()


def _shannon_entropy_bits(logits: torch.Tensor) -> float:
    logits_f32 = logits.float()
    log_z = torch.logsumexp(logits_f32, dim=-1, keepdim=True)
    delta = logits_f32 - log_z
    probs = torch.exp(delta)
    nats = -torch.where(probs > 0.0, probs * delta, torch.zeros_like(probs)).sum(
        dim=-1
    )
    return float((nats * _INV_LN2).mean().item())


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


def _get_strictly_upper_mask(chunk_size: int, device: torch.device) -> torch.Tensor:
    key = (chunk_size, str(device))
    cached = _STRICTLY_UPPER_MASK_CACHE.get(key)
    if cached is not None:
        _STRICTLY_UPPER_MASK_CACHE.move_to_end(key)
        return cached
    mask = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=device).triu(1)
    _STRICTLY_UPPER_MASK_CACHE[key] = mask
    while len(_STRICTLY_UPPER_MASK_CACHE) > 8:
        _STRICTLY_UPPER_MASK_CACHE.popitem(last=False)
    return mask


def _get_eye_matrix(
    size: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    key = (size, dtype, str(device))
    cached = _EYE_MATRIX_CACHE.get(key)
    if cached is not None:
        _EYE_MATRIX_CACHE.move_to_end(key)
        return cached
    eye = torch.eye(size, dtype=dtype, device=device)
    _EYE_MATRIX_CACHE[key] = eye
    while len(_EYE_MATRIX_CACHE) > 8:
        _EYE_MATRIX_CACHE.popitem(last=False)
    return eye


def _act_fn(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "silu":
        return F.silu(x)
    if name == "gelu_pytorch_tanh":
        return F.gelu(x, approximate="tanh")
    if name == "gelu":
        return F.gelu(x)
    return F.silu(x)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _l2norm_fp32(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


class Qwen3_5RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        normed = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * (1.0 + self.weight.float())).to(dtype=x.dtype)


class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        in_dtype = hidden_states.dtype
        h_f32 = hidden_states.float()
        var = h_f32.pow(2).mean(-1, keepdim=True)
        normed = h_f32 * torch.rsqrt(var + self.variance_epsilon)
        scaled = self.weight * normed.to(in_dtype)
        gated = scaled.float() * F.silu(gate.float())
        return gated.to(in_dtype)


@dataclass
class Qwen3_5ModelOutput:
    last_hidden_state: torch.Tensor
    pooler_output: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None
    rope_deltas: Optional[torch.Tensor] = None
    past_key_values: Optional[Qwen3_5HybridCache] = None


@dataclass
class Qwen3_5VisionConfig:
    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 4096
    num_position_embeddings: int = 2304
    initializer_range: float = 0.02
    rope_parameters: Dict[str, Any] = field(
        default_factory=lambda: {"rope_type": "axial", "rope_theta": 10000.0}
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Qwen3_5VisionConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        if "rope_parameters" not in filtered or not filtered["rope_parameters"]:
            filtered["rope_parameters"] = {
                "rope_type": "axial",
                "rope_theta": float(data.get("rope_theta", 10000.0)),
            }
        return cls(**filtered)


@dataclass
class Qwen3_5TextConfig:
    vocab_size: int = 248320
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0
    head_dim: int = 256
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    mtp_num_hidden_layers: int = 1
    layer_types: Optional[List[str]] = None
    pad_token_id: Optional[int] = 248044
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = 248044
    rope_parameters: Dict[str, Any] = field(
        default_factory=lambda: {
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
        }
    )

    def __post_init__(self) -> None:
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention" if bool((i + 1) % 4) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self.pad_token_id is not None and not (
            0 <= int(self.pad_token_id) < int(self.vocab_size)
        ):
            self.pad_token_id = None
        if self.eos_token_id is not None and not (
            0 <= int(self.eos_token_id) < int(self.vocab_size)
        ):
            self.eos_token_id = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Qwen3_5TextConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        if "rope_parameters" not in filtered or not filtered["rope_parameters"]:
            rope_scaling = data.get("rope_scaling") or {}
            filtered["rope_parameters"] = {
                "rope_type": rope_scaling.get("rope_type", "default"),
                "rope_theta": float(data.get("rope_theta", 10000000.0)),
                "partial_rotary_factor": float(
                    data.get(
                        "partial_rotary_factor",
                        rope_scaling.get("partial_rotary_factor", 0.25),
                    )
                ),
                "mrope_section": list(
                    rope_scaling.get("mrope_section", [11, 11, 10])
                ),
            }
        return cls(**filtered)


@dataclass
class Qwen3_5Config:
    text_config: Qwen3_5TextConfig = field(default_factory=Qwen3_5TextConfig)
    vision_config: Qwen3_5VisionConfig = field(default_factory=Qwen3_5VisionConfig)
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    im_start_token_id: int = 248045
    im_end_token_id: int = 248046
    think_start_token_id: int = 248068
    think_end_token_id: int = 248069
    tie_word_embeddings: bool = False
    include_lm_head: bool = False
    include_mtp: bool = False

    @classmethod
    def from_json_file(cls, config_path: Union[str, Path]) -> Qwen3_5Config:
        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
        t_dict = raw.get("text_config", raw)
        v_dict = raw.get("vision_config", {})
        return cls(
            text_config=Qwen3_5TextConfig.from_dict(t_dict),
            vision_config=Qwen3_5VisionConfig.from_dict(v_dict),
            image_token_id=int(raw.get("image_token_id", 248056)),
            video_token_id=int(raw.get("video_token_id", 248057)),
            vision_start_token_id=int(raw.get("vision_start_token_id", 248053)),
            vision_end_token_id=int(raw.get("vision_end_token_id", 248054)),
            im_start_token_id=int(raw.get("im_start_token_id", 248045)),
            im_end_token_id=int(raw.get("im_end_token_id", 248046)),
            think_start_token_id=int(raw.get("think_start_token_id", 248068)),
            think_end_token_id=int(raw.get("think_end_token_id", 248069)),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
            include_lm_head=bool(raw.get("include_lm_head", False)),
            include_mtp=bool(raw.get("include_mtp", False)),
        )


Qwen35Config = Qwen3_5Config
Qwen35TextConfig = Qwen3_5TextConfig
Qwen35VisionConfig = Qwen3_5VisionConfig

__all__ = [
    "_FP8_DTYPES",
    "_INV_LN2",
    "_STRICTLY_UPPER_MASK_CACHE",
    "_EYE_MATRIX_CACHE",
    "_shannon_entropy_bits",
    "_resolve_compute_dtype",
    "_linear",
    "_get_strictly_upper_mask",
    "_get_eye_matrix",
    "_act_fn",
    "_rotate_half",
    "_l2norm_fp32",
    "Qwen3_5RMSNorm",
    "Qwen3_5RMSNormGated",
    "Qwen3_5ModelOutput",
    "Qwen3_5VisionConfig",
    "Qwen3_5TextConfig",
    "Qwen3_5Config",
    "Qwen35Config",
    "Qwen35TextConfig",
    "Qwen35VisionConfig",
]