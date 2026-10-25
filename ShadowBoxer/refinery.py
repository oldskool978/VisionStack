from __future__ import annotations

from collections import OrderedDict
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

DTYPE_MAP: Dict[str, Optional[torch.dtype]] = {
    "native": None,
    "none": None,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "e4m3fn": getattr(torch, "float8_e4m3fn", None),
    "e5m2": getattr(torch, "float8_e5m2", None),
}

DTYPE_LOOKUP: Dict[str, Optional[torch.dtype]] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "e4m3fn": getattr(torch, "float8_e4m3fn", None),
    "e5m2": getattr(torch, "float8_e5m2", None),
    "e8m0fnu": getattr(torch, "float8_e8m0fnu", None),
}

FP8_DTYPES: List[torch.dtype] = [
    v for k, v in DTYPE_LOOKUP.items() if "e" in k and v is not None
]

_ENGINE_CACHE: Dict[str, Any] = {}


def clear_refinery_cache() -> None:
    global _ENGINE_CACHE
    _ENGINE_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class PhysicsLab:
    PI_BY_2 = math.pi / 2.0
    SIGMA_SCALE = 1.4826
    SQRT_2 = 1.4142135623730951

    MANTISSA_MAP: Dict[str, int] = {
        "float32": 23,
        "float16": 10,
        "bfloat16": 7,
        "float8_e4m3fn": 3,
        "float8_e4m3fnuz": 3,
        "float8_e5m2": 2,
        "float8_e5m2fnuz": 2,
        "float8_e8m0fnu": 0,
        "int8": 7,
        "uint8": 0,
    }

    EXPONENT_BIAS_MAP: Dict[str, Tuple[int, int]] = {
        "float8_e4m3fn": (-6, 7),
        "float8_e5m2": (-14, 15),
        "float16": (-14, 15),
        "bfloat16": (-126, 127),
        "float32": (-126, 127),
    }

    @staticmethod
    def get_mantissa_width(dtype: torch.dtype) -> int:
        key = str(dtype).split(".")[-1]
        return PhysicsLab.MANTISSA_MAP.get(key, 10)

    @staticmethod
    def get_machine_epsilon(dtype: torch.dtype) -> float:
        try:
            return float(torch.finfo(dtype).eps)
        except Exception:
            key = str(dtype).split(".")[-1]
            if key == "bfloat16":
                return 2.0**-7
            if key == "float16":
                return 2.0**-10
            if key == "float8_e4m3fn":
                return 2.0**-3
            if key == "float8_e5m2":
                return 2.0**-2
            return 0.125

    @staticmethod
    def get_conductivity_scaler(
        src_dtype: torch.dtype, tgt_dtype: torch.dtype, dither_enabled: bool
    ) -> float:
        m_src = PhysicsLab.get_mantissa_width(src_dtype)
        m_tgt = PhysicsLab.get_mantissa_width(tgt_dtype)
        if m_src < 5 and m_tgt <= 2 and dither_enabled:
            return 1.0
        if not dither_enabled:
            return 1.0
        if m_src < 5:
            return 0.5
        return 1.0

    @staticmethod
    def get_hybrid_grace_strength(error_ratio: torch.Tensor) -> torch.Tensor:
        strength_quad = error_ratio.pow(2.0)
        strength_cos = 1.0 - torch.cos(error_ratio * PhysicsLab.PI_BY_2)
        return (strength_quad + strength_cos) * 0.5

    @staticmethod
    def estimate_robust_sigma(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.numel() < 2:
            return torch.tensor(1.0, device=tensor.device, dtype=torch.float32)
        sample = tensor
        if tensor.numel() > 524288:
            stride = tensor.numel() // 524288
            sample = tensor.reshape(-1)[::stride]
        median = sample.median()
        mad = (sample - median).abs().median()
        return torch.clamp(mad * PhysicsLab.SIGMA_SCALE, min=1e-9)

    @staticmethod
    def perona_malik_conductivity(
        gradient: torch.Tensor, k: torch.Tensor
    ) -> torch.Tensor:
        term = gradient.abs() / (k + 1e-12)
        return torch.exp(-(term * term))


class TopologyEngine:
    def __init__(self, device: torch.device, max_cache_entries: int = 8):
        self.device = device
        self.max_cache_entries = max_cache_entries
        self.cache_fwd: OrderedDict[Tuple[int, ...], torch.Tensor] = OrderedDict()

    def _spread_bits_3d(self, v: torch.Tensor) -> torch.Tensor:
        v = v.long() & 0x1FFFFF
        v = (v | (v << 32)) & 0x1F00000000FFFF
        v = (v | (v << 16)) & 0x1F0000FF0000FF
        v = (v | (v << 8)) & 0x100F00F00F00F00F
        v = (v | (v << 4)) & 0x10C30C30C30C30C3
        v = (v | (v << 2)) & 0x1249249249249249
        return v

    def _compute_indices(self, shape: Tuple[int, ...]) -> torch.Tensor:
        C, H, W = shape
        cs = torch.arange(C, device=self.device, dtype=torch.int64)
        hs = torch.arange(H, device=self.device, dtype=torch.int64)
        ws = torch.arange(W, device=self.device, dtype=torch.int64)
        grid_c, grid_h, grid_w = torch.meshgrid(cs, hs, ws, indexing="ij")
        mc_c = self._spread_bits_3d(grid_c.flatten())
        mc_h = self._spread_bits_3d(grid_h.flatten())
        mc_w = self._spread_bits_3d(grid_w.flatten())
        z_code = (mc_c << 2) | (mc_h << 1) | mc_w
        gray_code = z_code ^ (z_code >> 1)
        return torch.argsort(gray_code)

    def get_indices(self, shape: Tuple[int, ...]) -> torch.Tensor:
        key = tuple(shape)
        if key in self.cache_fwd:
            self.cache_fwd.move_to_end(key)
            return self.cache_fwd[key]
        indices = self._compute_indices(key)
        self.cache_fwd[key] = indices
        while len(self.cache_fwd) > self.max_cache_entries:
            self.cache_fwd.popitem(last=False)
        return indices

    def linearize(self, tensor: torch.Tensor) -> torch.Tensor:
        B, C, H, W = tensor.shape
        fwd_idx = self.get_indices((C, H, W))
        flat = tensor.reshape(B, -1)
        idx_expanded = fwd_idx.unsqueeze(0).expand(B, -1)
        return torch.gather(flat, 1, idx_expanded)

    def restore(
        self, z_tensor: torch.Tensor, orig_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        B, C, H, W = orig_shape
        fwd_idx = self.get_indices((C, H, W))
        idx_expanded = fwd_idx.unsqueeze(0).expand(B, -1)
        restored = torch.empty_like(z_tensor)
        restored.scatter_(1, idx_expanded, z_tensor)
        return restored.reshape(orig_shape)


class TheContender:
    def __init__(self, device: torch.device):
        self.device = device
        self.topology = TopologyEngine(device)
        self.bayer_cache: Optional[torch.Tensor] = None

    def _get_full_rank_bayer(self, length: int) -> torch.Tensor:
        if self.bayer_cache is not None and self.bayer_cache.shape[0] >= length:
            return self.bayer_cache[:length]
        bayer_8x8 = torch.tensor(
            [
                [0, 48, 12, 60, 3, 51, 15, 63],
                [32, 16, 44, 28, 35, 19, 47, 31],
                [8, 56, 4, 52, 11, 59, 7, 55],
                [40, 24, 36, 20, 43, 27, 39, 23],
                [2, 50, 14, 62, 1, 49, 13, 61],
                [34, 18, 46, 30, 33, 17, 45, 29],
                [10, 58, 6, 54, 9, 57, 5, 53],
                [42, 26, 38, 22, 41, 25, 37, 21],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        bayer_norm = (bayer_8x8.reshape(-1) - 31.5) / 64.0
        repeats = (length + 63) // 64
        full_pattern = bayer_norm.repeat(repeats)[:length]
        self.bayer_cache = full_pattern
        return full_pattern

    @torch.no_grad()
    def quantize(
        self, x_input: torch.Tensor, target_dtype: torch.dtype
    ) -> torch.Tensor:
        orig_shape = x_input.shape
        if x_input.dim() >= 5:
            t_in = x_input.reshape(orig_shape[0], orig_shape[1], orig_shape[2], -1)
        elif x_input.dim() == 4:
            t_in = x_input
        elif x_input.dim() == 3:
            t_in = x_input.unsqueeze(1)
        elif x_input.dim() == 2:
            t_in = x_input.unsqueeze(0).unsqueeze(0)
        else:
            t_in = x_input.reshape(1, 1, 1, -1)

        B, C, H, W = t_in.shape
        dims = list(range(t_in.dim()))
        dims.pop(1)
        means = t_in.mean(dim=dims, keepdim=True)
        t_centered = t_in - means

        x_flat = self.topology.linearize(t_centered)
        eps_factor = PhysicsLab.get_machine_epsilon(target_dtype)
        try:
            max_val = float(torch.finfo(target_dtype).max)
        except Exception:
            max_val = 65504.0

        total_len = x_flat.numel()
        dither_pat = self._get_full_rank_bayer(total_len).reshape(B, -1)
        t_nearest = x_flat.to(target_dtype).float()
        current_error = (x_flat - t_nearest).abs()
        step_size_est = torch.abs(x_flat) * eps_factor
        max_error_est = step_size_est * 0.5
        error_ratio = torch.nan_to_num(
            current_error / (max_error_est + 1e-12), nan=0.0
        ).clamp(0.0, 1.0)
        grace_strength = PhysicsLab.get_hybrid_grace_strength(error_ratio)
        dither_amp = max_error_est * grace_strength

        rail_threshold = max_val * 0.9
        rail_mask = torch.clamp(
            (max_val - torch.abs(x_flat)) / (max_val - rail_threshold + 1e-9), 0.0, 1.0
        )
        x_dithered = x_flat + (dither_pat * dither_amp * rail_mask)
        t_out_flat = x_dithered.to(target_dtype).float()
        t_restored = self.topology.restore(t_out_flat, (B, C, H, W))
        return (t_restored + means).reshape(orig_shape).to(target_dtype)


class TheRefinery:
    DTYPE_BOUNDARIES: Dict[torch.dtype, Tuple[float, float]] = {
        torch.float32: (
            float(torch.finfo(torch.float32).min),
            float(torch.finfo(torch.float32).max),
        ),
        torch.float16: (
            float(torch.finfo(torch.float16).min),
            float(torch.finfo(torch.float16).max),
        ),
        torch.bfloat16: (
            float(torch.finfo(torch.bfloat16).min),
            float(torch.finfo(torch.bfloat16).max),
        ),
        torch.int8: (-128.0, 127.0),
    }

    def __init__(self, device: torch.device):
        self.device = device
        self.last_bayer_shape: Optional[Tuple[int, ...]] = None
        self.last_bayer_tensor: Optional[torch.Tensor] = None
        self._inf_bounds: Dict[torch.dtype, Tuple[torch.Tensor, torch.Tensor]] = {}

        def recurse_bayer(sz: int) -> torch.Tensor:
            if sz == 1:
                return torch.tensor([[0.0]], device=self.device)
            q = recurse_bayer(sz // 2)
            m = torch.empty((sz, sz), device=self.device)
            m[: sz // 2, : sz // 2] = 4 * q + 0
            m[: sz // 2, sz // 2 :] = 4 * q + 2
            m[sz // 2 :, : sz // 2] = 4 * q + 3
            m[sz // 2 :, sz // 2 :] = 4 * q + 1
            return m

        self.bayer_8x8 = (recurse_bayer(8) - 31.5) / 64.0

    def _get_inf_pair(self, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        pair = self._inf_bounds.get(dtype)
        if pair is None or pair[0].device != self.device:
            pair = (
                torch.tensor(-float("inf"), device=self.device, dtype=dtype),
                torch.tensor(float("inf"), device=self.device, dtype=dtype),
            )
            self._inf_bounds[dtype] = pair
        return pair

    def _get_bayer_block(self, shape: Tuple[int, ...]) -> torch.Tensor:
        if self.last_bayer_shape == shape and self.last_bayer_tensor is not None:
            return self.last_bayer_tensor
        C, N, B_sz = shape
        total_len = C * N * B_sz
        flat = self.bayer_8x8.reshape(-1).repeat((total_len // 64) + 1)[:total_len]
        result = flat.reshape(C, N, B_sz)
        self.last_bayer_shape = shape
        self.last_bayer_tensor = result
        return result

    def _get_quant_bounds(
        self, tensor: torch.Tensor, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tensor_q = tensor.to(dtype)
        tensor_f32 = tensor_q.float()
        if dtype in (torch.bfloat16, torch.float16, torch.float32):
            t_neg_inf, t_pos_inf = self._get_inf_pair(dtype)
            low = torch.nextafter(tensor_q, t_neg_inf).float()
            high = torch.nextafter(tensor_q, t_pos_inf).float()
            return tensor_f32, low, high

        key = str(dtype).split(".")[-1]
        mant_bits = PhysicsLab.MANTISSA_MAP.get(key, 3)
        e_min, e_max = PhysicsLab.EXPONENT_BIAS_MAP.get(key, (-6, 7))
        abs_val = tensor_f32.abs()
        log2_val = torch.floor(torch.log2(torch.clamp(abs_val, min=1e-8)))
        exp_clamped = torch.clamp(log2_val, min=float(e_min), max=float(e_max))
        step_size = torch.pow(2.0, exp_clamped - mant_bits)
        sign = torch.sign(tensor_f32)
        sign = torch.where(sign == 0.0, torch.ones_like(sign), sign)
        abs_quant = torch.floor(abs_val / step_size) * step_size
        s_low = sign * abs_quant
        s_high = s_low + (sign * step_size)
        low = torch.minimum(s_low, s_high)
        high = torch.maximum(s_low, s_high)
        return tensor_f32, low, high

    def _smart_seamstress(
        self, blocks: torch.Tensor, conductivity_threshold: torch.Tensor
    ) -> torch.Tensor:
        if blocks.dim() != 3 or blocks.shape[1] < 2:
            return blocks
        seam_diff = blocks[:, 1:, 0] - blocks[:, :-1, -1]
        c_seam = PhysicsLab.perona_malik_conductivity(seam_diff, conductivity_threshold)
        w_seam = c_seam * 0.25
        left = blocks[:, :-1, -1].clone()
        right = blocks[:, 1:, 0].clone()
        blocks[:, :-1, -1] = left * (1.0 - w_seam) + right * w_seam
        blocks[:, 1:, 0] = right * (1.0 - w_seam) + left * w_seam
        return blocks

    def _quantize_slice(
        self,
        t_slice: torch.Tensor,
        target_dtype: torch.dtype,
        k_threshold: torch.Tensor,
        block_size: int,
        dither_enabled: bool,
    ) -> torch.Tensor:
        C, N_elements = t_slice.shape
        pad = (block_size - (N_elements % block_size)) % block_size
        if pad > 0:
            t_slice = F.pad(t_slice, (0, pad), "constant", 0)
        num_blocks = t_slice.shape[1] // block_size
        t_blocked = t_slice.reshape(C, num_blocks, block_size)
        min_v, max_v = TheRefinery.DTYPE_BOUNDARIES.get(target_dtype, (-448.0, 448.0))
        out_blocked = torch.empty_like(t_blocked)

        if dither_enabled:
            bayer = self._get_bayer_block((C, num_blocks, block_size))
            error_state = torch.zeros((C, num_blocks), device=self.device)
            for i in range(block_size):
                y = t_blocked[:, :, i]
                z = torch.clamp(y + error_state, min_v, max_v)
                z_near, z_low, z_high = self._get_quant_bounds(z, target_dtype)
                is_above = y > z_near
                s_low = torch.where(is_above, z_near, z_low)
                s_high = torch.where(is_above, z_high, z_near)
                step = torch.clamp(s_high - s_low, min=1e-12)
                max_err = step * 0.5
                curr_err = (y - z_near).abs()
                strength = PhysicsLab.get_hybrid_grace_strength(
                    torch.clamp(curr_err / max_err, 0.0, 1.0)
                )
                nudge = bayer[:, :, i] * (max_err * strength)
                z_final = z + nudge
                q = torch.clamp(z_final, s_low, s_high).to(target_dtype).float()
                out_blocked[:, :, i] = q
                error_state = torch.clamp(z_final - q, -1.0, 1.0)
        else:
            z_clamped = torch.clamp(t_blocked, min_v, max_v)
            out_blocked = z_clamped.to(target_dtype).float()

        out_smoothed = self._smart_seamstress(out_blocked, k_threshold)
        return out_smoothed.reshape(C, -1)[:, :N_elements]

    @torch.no_grad()
    def quantize(
        self,
        tensor_fp32: torch.Tensor,
        target_dtype: torch.dtype,
        src_dtype: torch.dtype,
        block_size: int = 16,
        dither_enabled: bool = True,
        max_chunk_channels: int = 2048,
    ) -> torch.Tensor:
        orig_shape = tensor_fp32.shape
        if tensor_fp32.dim() == 1:
            t_2d = tensor_fp32.unsqueeze(0)
        elif tensor_fp32.dim() >= 2:
            t_2d = tensor_fp32.reshape(orig_shape[0], -1)
        else:
            return tensor_fp32.to(target_dtype)

        C, _ = t_2d.shape
        means = torch.nan_to_num(t_2d.mean(dim=1, keepdim=True), nan=0.0)
        t_centered = t_2d - means
        sigma_robust = PhysicsLab.estimate_robust_sigma(t_centered)
        k_scaler = PhysicsLab.get_conductivity_scaler(
            src_dtype, target_dtype, dither_enabled
        )
        k_threshold = (PhysicsLab.SQRT_2 * sigma_robust) * k_scaler

        out_2d = torch.empty_like(t_centered)
        for c_start in range(0, C, max_chunk_channels):
            c_end = min(c_start + max_chunk_channels, C)
            t_slice = t_centered[c_start:c_end]
            out_2d[c_start:c_end] = self._quantize_slice(
                t_slice, target_dtype, k_threshold, block_size, dither_enabled
            )

        t_restored = out_2d + means
        return t_restored.reshape(orig_shape).to(target_dtype)


def _get_engine_instances(device: torch.device) -> Tuple[TheContender, TheRefinery]:
    key = str(device)
    if key not in _ENGINE_CACHE:
        _ENGINE_CACHE[key] = (TheContender(device), TheRefinery(device))
    return _ENGINE_CACHE[key]


def refineTensor(
    origin_tensor: torch.Tensor,
    target_dtype: torch.dtype,
    fp8_dtypes: Optional[List[torch.dtype]] = None,
    dtype_map: Optional[Dict[str, Optional[torch.dtype]]] = None,
    original_on_disk_dtype: Optional[torch.dtype] = None,
    target_device: Optional[torch.device] = None,
) -> torch.Tensor:
    with torch.inference_mode(False), torch.no_grad():
        src_dtype = (
            original_on_disk_dtype
            if original_on_disk_dtype is not None
            else origin_tensor.dtype
        )
        if not origin_tensor.is_floating_point() or src_dtype == target_dtype:
            return origin_tensor

        if target_device is None:
            target_device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        fp32_t = origin_tensor.to(target_device, non_blocking=True).float()
        active_map = dtype_map if dtype_map is not None else DTYPE_LOOKUP

        if target_dtype not in TheRefinery.DTYPE_BOUNDARIES:
            try:
                if target_dtype.is_floating_point:
                    finfo = torch.finfo(target_dtype)
                    TheRefinery.DTYPE_BOUNDARIES[target_dtype] = (
                        float(finfo.min),
                        float(finfo.max),
                    )
            except Exception:
                e4m3 = active_map.get("e4m3fn")
                e5m2 = active_map.get("e5m2")
                e8m0 = active_map.get("e8m0fnu")
                if target_dtype == e4m3:
                    TheRefinery.DTYPE_BOUNDARIES[target_dtype] = (-448.0, 448.0)
                elif target_dtype == e5m2:
                    TheRefinery.DTYPE_BOUNDARIES[target_dtype] = (-57344.0, 57344.0)
                elif target_dtype == e8m0:
                    TheRefinery.DTYPE_BOUNDARIES[target_dtype] = (
                        -(2.0**127),
                        (2.0**127),
                    )
                else:
                    TheRefinery.DTYPE_BOUNDARIES[target_dtype] = (
                        -float("inf"),
                        float("inf"),
                    )

        contender, refinery = _get_engine_instances(target_device)
        m_tgt = PhysicsLab.get_mantissa_width(target_dtype)
        m_src = PhysicsLab.get_mantissa_width(src_dtype)

        if target_dtype == torch.bfloat16:
            result = contender.quantize(fp32_t, target_dtype)
        elif m_tgt <= 2:
            if m_src >= 5:
                result = refinery.quantize(
                    fp32_t, target_dtype, src_dtype, dither_enabled=False
                )
            else:
                result = refinery.quantize(
                    fp32_t, target_dtype, src_dtype, dither_enabled=True
                )
        else:
            result = refinery.quantize(
                fp32_t, target_dtype, src_dtype, dither_enabled=True
            )

        return result.to(origin_tensor.device, non_blocking=True)


__all__ = [
    "DTYPE_MAP",
    "DTYPE_LOOKUP",
    "FP8_DTYPES",
    "PhysicsLab",
    "TopologyEngine",
    "TheContender",
    "TheRefinery",
    "clear_refinery_cache",
    "refineTensor",
]