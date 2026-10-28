# ShadowBoxer/core/prng.py
from __future__ import annotations

from typing import Dict, Tuple, Union

import torch

from .env import _canonicalize_device

_SPLITMIX64_C1 = 0x9E3779B97F4A7C15
_SPLITMIX64_C2 = 0xBF58476D1CE4E5B9
_SPLITMIX64_C3 = 0x94D049BB133111EB
_MASK64 = 0xFFFFFFFFFFFFFFFF
_MASK63 = 0x7FFFFFFFFFFFFFFF

_GENERATOR_CACHE: Dict[str, torch.Generator] = {}


def splitmix64(seed: int) -> int:
    z = (seed + _SPLITMIX64_C1) & _MASK64
    z = ((z ^ (z >> 30)) * _SPLITMIX64_C2) & _MASK64
    z = ((z ^ (z >> 27)) * _SPLITMIX64_C3) & _MASK64
    return z ^ (z >> 31)


def derive_stage_keys(master_seed: int) -> Tuple[int, int, int]:
    base = master_seed & _MASK64
    k_init = splitmix64(base ^ 0x5A5A5A5A5A5A5A5A)
    k_sde = splitmix64(base ^ 0xA5A5A5A5A5A5A5A5)
    k_aux = splitmix64(base ^ 0x3C3C3C3C3C3C3C3C)
    return k_init, k_sde, k_aux


def derive_effective_seed(seed: int, stream_id: int = 0, offset: int = 0) -> int:
    h = splitmix64((seed ^ (stream_id * _SPLITMIX64_C1)) & _MASK64)
    h = splitmix64((h + offset + _SPLITMIX64_C2) & _MASK64)
    return int(h & _MASK63)


def philox_randn(
    shape: Union[Tuple[int, ...], torch.Size],
    seed: int,
    stream_id: int = 0,
    offset: int = 0,
    device: Union[str, torch.device] = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    target_device = _canonicalize_device(device)
    eff_seed = derive_effective_seed(seed, stream_id, offset)
    dev_key = f"{target_device.type}:{target_device.index if target_device.index is not None else 0}"
    gen = _GENERATOR_CACHE.get(dev_key)
    if gen is None:
        gen = torch.Generator(device=target_device)
        _GENERATOR_CACHE[dev_key] = gen
    gen.manual_seed(eff_seed)
    return torch.randn(shape, generator=gen, device=target_device, dtype=dtype)


def subphase_radial_clamp(
    packed_latents: torch.Tensor, max_sigma: float = 4.5
) -> torch.Tensor:
    if packed_latents.ndim != 4 or packed_latents.shape[1] != 128:
        return packed_latents
    orig_dtype = packed_latents.dtype
    b, _, h, w = packed_latents.shape
    z = packed_latents.float().view(b, 32, 4, h, w)
    rms = torch.sqrt(torch.mean(z * z, dim=2, keepdim=True) + 1e-8)
    excess = torch.clamp(rms - max_sigma, min=0.0)
    clamped_rms = max_sigma + excess * torch.rsqrt(1.0 + excess * excess)
    scale = torch.where(rms > max_sigma, clamped_rms / rms, torch.ones_like(rms))
    return (z * scale).view(b, 128, h, w).to(orig_dtype)


__all__ = [
    "_SPLITMIX64_C1",
    "_SPLITMIX64_C2",
    "_SPLITMIX64_C3",
    "_MASK64",
    "_MASK63",
    "_GENERATOR_CACHE",
    "splitmix64",
    "derive_stage_keys",
    "derive_effective_seed",
    "philox_randn",
    "subphase_radial_clamp",
]