# ShadowBoxer/models/moss/config.py
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple, Union

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
_HIERARCHICAL_AREA_GAMMA = 4.0
_HIERARCHICAL_Z_GAMMA = 2.5
_ISOKV_COLLAB_OVERLAP_MIN = 0.25

CollabGroups = Tuple[Tuple[int, ...], ...]
IsoKVSpec = Union[
    Tuple[int, Tuple[int, ...], int],
    Tuple[int, Tuple[int, ...], int, CollabGroups],
]
NormBBox = Tuple[float, float, float, float]
CarveoutSpec = Tuple[NormBBox, float, Optional[float]]

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
        src = t.detach()
        if torch.cuda.is_available():
            try:
                out = torch.empty(
                    src.shape, dtype=src.dtype, device="cpu", pin_memory=True
                )
                out.copy_(src, non_blocking=False)
                return out
            except Exception:
                pass
        return src.to(device="cpu").clone().contiguous()


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


__all__ = [
    "_FP8_DTYPES",
    "_FFN_TILE_TOKENS",
    "_BOX_ISOLATION_LOGIT_SCALE",
    "_HIERARCHICAL_AREA_GAMMA",
    "_HIERARCHICAL_Z_GAMMA",
    "_ISOKV_COLLAB_OVERLAP_MIN",
    "CollabGroups",
    "IsoKVSpec",
    "NormBBox",
    "CarveoutSpec",
    "ModTuple6",
    "PiecewiseModSpec",
    "_maybe_pin_cpu_tensor",
    "_resolve_compute_dtype",
    "_linear",
    "TransformerOutput",
    "MOSSImageConfig",
    "_RMSNorm",
    "_situ",
    "_situ_inplace",
    "validate_skip_layers",
]