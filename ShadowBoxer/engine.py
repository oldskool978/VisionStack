# ShadowBoxer/engine.py
from __future__ import annotations

from collections import OrderedDict
import ctypes
from dataclasses import dataclass
import gc
import json
import logging
import math
import os
from pathlib import Path
import random
import shutil
import struct
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

sys.modules["torchvision"] = None
sys.modules["torchvision.transforms"] = None
sys.modules["torchvision.io"] = None


def _resolve_cache_dir() -> Path:
    env_hf = os.environ.get("HF_HOME")
    if env_hf and env_hf.strip():
        return Path(env_hf).resolve()
    parent_cache = (ROOT_DIR.parent / ".hf_cache").resolve()
    if parent_cache.exists():
        return parent_cache
    return (ROOT_DIR / ".hf_cache").resolve()


CACHE_DIR = _resolve_cache_dir()
MIOPEN_CACHE_DIR = CACHE_DIR / "miopen"
MIOPEN_DB_DIR = MIOPEN_CACHE_DIR / "db"
MIOPEN_KERNELS_DIR = MIOPEN_CACHE_DIR / "kernels"
COMGR_CACHE_DIR = MIOPEN_CACHE_DIR / "comgr"
TMP_DIR = Path(os.environ.get("TMP") or (ROOT_DIR / "artifacts" / "tmp")).resolve()

for _d in (
    CACHE_DIR,
    MIOPEN_DB_DIR,
    MIOPEN_KERNELS_DIR,
    COMGR_CACHE_DIR,
    TMP_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)


def _configure_rocm_runtime_env() -> None:
    os.environ.setdefault("HF_HOME", str(CACHE_DIR))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(CACHE_DIR / "hub"))
    os.environ.setdefault("MIOPEN_USER_DB_PATH", str(MIOPEN_DB_DIR))
    os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", str(MIOPEN_KERNELS_DIR))
    os.environ.setdefault("AMD_COMGR_CACHE", "1")
    os.environ.setdefault("AMD_COMGR_CACHE_DIR", str(COMGR_CACHE_DIR))
    os.environ.pop("MIOPEN_FIND_ENFORCE", None)
    if os.environ.get("MIOPEN_FIND_MODE", "").strip().lower() in (
        "",
        "1",
        "2",
        "normal",
        "fast",
    ):
        os.environ["MIOPEN_FIND_MODE"] = "3"
    os.environ.setdefault("MIOPEN_LOG_LEVEL", "0")
    os.environ.setdefault("MIOPEN_ENABLE_LOGGING", "0")
    os.environ.setdefault("AMD_LOG_LEVEL", "0")
    os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")
    os.environ.setdefault("HSA_ENABLE_SDMA", "1")

    cur_sys_db = os.environ.get("MIOPEN_SYSTEM_DB_PATH", "").strip()
    if cur_sys_db:
        p_db = Path(cur_sys_db)
        if p_db.is_dir() and any(p_db.iterdir()):
            return
    for sp_cand in sys.path:
        if not sp_cand:
            continue
        sp_path = Path(sp_cand)
        if not sp_path.is_dir():
            continue
        for db_cand in (
            list(sp_path.glob("_rocm_sdk_*/share/miopen/db"))
            + [sp_path / "torch" / "share" / "miopen" / "db"]
        ):
            if db_cand.is_dir() and any(db_cand.iterdir()):
                os.environ["MIOPEN_SYSTEM_DB_PATH"] = str(db_cand.resolve())
                return


_configure_rocm_runtime_env()

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

import numpy as np
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors
import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
    torch.backends.cuda.matmul.allow_tf32 = False
if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("highest")

from models.modeling_moss import (
    MOSSImageConfig,
    MOSSImageTransformer,
    validate_skip_layers,
)
from models.modeling_qwen35 import Qwen3_5Model
from models.modeling_vae import Flux2AutoencoderKL
from models.tokenizer import MOSSImageTokenizer
from schema import (
    GenerationRequest,
    GenerationResponse,
    SUPPORTED_MODELS,
    bbox_to_normalized,
)

try:
    from vizinjector import (
        ConditionedVisionBundle,
        VIZJ_TIER_DUAL_LATENT,
        VIZJ_TIER_VAE_LATENT,
        VIZJ_TIER_WIRE_WEBP,
        VizInjectorClient,
        VizInjectorRequest,
        subphase_radial_clamp as _viz_subphase_radial_clamp,
    )

    HAS_VIZINJECTOR = True
except Exception:
    ConditionedVisionBundle = Any
    VIZJ_TIER_WIRE_WEBP = 0
    VIZJ_TIER_VAE_LATENT = 1
    VIZJ_TIER_DUAL_LATENT = 2
    VizInjectorClient = None
    VizInjectorRequest = None
    _viz_subphase_radial_clamp = None
    HAS_VIZINJECTOR = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

_SPLITMIX64_C1 = 0x9E3779B97F4A7C15
_SPLITMIX64_C2 = 0xBF58476D1CE4E5B9
_SPLITMIX64_C3 = 0x94D049BB133111EB
_MASK64 = 0xFFFFFFFFFFFFFFFF
_MASK63 = 0x7FFFFFFFFFFFFFFF

_VRAM_TRIM_THRESHOLD_BYTES = int(28.5 * (1024**3))
_MIN_RECLAIMABLE_CACHE_BYTES = 64 * (1024**2)
_MAX_INACTIVE_SPLIT_BYTES = 512 * (1024**2)
_AI_TOOLKIT_MAX_REF_DIM = 512

_GENERATOR_CACHE: Dict[str, torch.Generator] = {}


def _canonicalize_device(device: Union[str, torch.device]) -> torch.device:
    dev = torch.device(device) if isinstance(device, str) else device
    if dev.type == "cuda" and dev.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return dev


def _same_device(dev_a: torch.device, dev_b: torch.device) -> bool:
    return _canonicalize_device(dev_a) == _canonicalize_device(dev_b)


class _DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", ctypes.c_uint),
        ("DeviceId", ctypes.c_uint),
        ("SubSysId", ctypes.c_uint),
        ("Revision", ctypes.c_uint),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", ctypes.c_longlong),
        ("Flags", ctypes.c_uint),
    ]


class _DXGI_QUERY_VIDEO_MEMORY_INFO(ctypes.Structure):
    _fields_ = [
        ("Budget", ctypes.c_ulonglong),
        ("CurrentUsage", ctypes.c_ulonglong),
        ("AvailableForReservation", ctypes.c_ulonglong),
        ("CurrentReservation", ctypes.c_ulonglong),
    ]


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_ulong),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    ]


@dataclass
class WDDMMemorySnapshot:
    dedicated_total_gb: float
    dedicated_used_gb: float
    dedicated_avail_gb: float
    shared_budget_gb: float
    shared_used_gb: float
    shared_delta_gb: float
    pt_allocated_gb: float
    pt_reserved_gb: float
    pt_reclaimable_gb: float
    is_spilling: bool
    pt_inactive_split_gb: float = 0.0
    contiguous_avail_gb: float = 0.0
    host_total_gb: float = 0.0
    host_avail_gb: float = 0.0


class WDDMMemoryGuard:
    def __init__(self, device: torch.device):
        self.device = _canonicalize_device(device)
        self.active = False
        self.adapter_name = "Generic GPU"
        self.luid_key = ""
        self.dedicated_total_gb = 0.0
        self.shared_budget_gb = 0.0
        self.shared_baseline_gb = 0.0
        self._adapter3 = ctypes.c_void_p()

        if sys.platform == "win32" and self.device.type == "cuda":
            self._init_win32_telemetry()

    @staticmethod
    def _call_com(
        ptr: ctypes.c_void_p, index: int, argtypes: List[Any], *args: Any
    ) -> int:
        vtable = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_void_p))[0]
        func_addr = ctypes.cast(vtable, ctypes.POINTER(ctypes.c_void_p))[index]
        func = ctypes.WINFUNCTYPE(ctypes.c_long, *argtypes)(func_addr)
        return int(func(ptr, *args))

    @staticmethod
    def _query_host_ram_gb() -> Tuple[float, float]:
        if sys.platform != "win32":
            return 0.0, 0.0
        try:
            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)) != 0:
                return (
                    float(stat.ullTotalPhys) / (1024**3),
                    float(stat.ullAvailPhys) / (1024**3),
                )
        except Exception:
            pass
        return 0.0, 0.0

    def _init_win32_telemetry(self) -> None:
        try:
            iid_factory1 = _GUID(
                0x770AAE78,
                0xF26F,
                0x4DBA,
                (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87),
            )
            iid_adapter3 = _GUID(
                0x645967A4,
                0x1392,
                0x4310,
                (ctypes.c_ubyte * 8)(0xA7, 0x98, 0x80, 0x53, 0xCE, 0x3E, 0x93, 0xFD),
            )
            factory = ctypes.c_void_p()
            if (
                ctypes.windll.dxgi.CreateDXGIFactory1(
                    ctypes.byref(iid_factory1), ctypes.byref(factory)
                )
                != 0
            ):
                return

            target_hw_idx = self.device.index if self.device.index is not None else 0
            hw_seen = 0
            adapter_idx = 0
            while True:
                adapter = ctypes.c_void_p()
                hr = self._call_com(
                    factory,
                    12,
                    [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)],
                    adapter_idx,
                    ctypes.byref(adapter),
                )
                if hr < 0:
                    break
                desc = _DXGI_ADAPTER_DESC1()
                self._call_com(
                    adapter,
                    10,
                    [ctypes.c_void_p, ctypes.POINTER(_DXGI_ADAPTER_DESC1)],
                    ctypes.byref(desc),
                )
                if not (desc.Flags & 2) and desc.DedicatedVideoMemory > 0:
                    if hw_seen == target_hw_idx:
                        high = (desc.AdapterLuid >> 32) & 0xFFFFFFFF
                        low = desc.AdapterLuid & 0xFFFFFFFF
                        self.luid_key = f"0x{high:08x}_0x{low:08x}".lower()
                        self.adapter_name = str(desc.Description)
                        self.dedicated_total_gb = desc.DedicatedVideoMemory / (1024**3)
                        self.shared_budget_gb = desc.SharedSystemMemory / (1024**3)
                        adapter3 = ctypes.c_void_p()
                        hr_qi = self._call_com(
                            adapter,
                            0,
                            [
                                ctypes.c_void_p,
                                ctypes.POINTER(_GUID),
                                ctypes.POINTER(ctypes.c_void_p),
                            ],
                            ctypes.byref(iid_adapter3),
                            ctypes.byref(adapter3),
                        )
                        if hr_qi >= 0 and adapter3.value:
                            self._adapter3 = adapter3
                            self.active = True
                        self._call_com(adapter, 2, [ctypes.c_void_p])
                        break
                    hw_seen += 1
                self._call_com(adapter, 2, [ctypes.c_void_p])
                adapter_idx += 1

            self._call_com(factory, 2, [ctypes.c_void_p])
            if self.active:
                _, shr_used = self._query_video_memory_info()
                self.shared_baseline_gb = shr_used
        except Exception:
            self.active = False

    def _query_video_memory_info(self) -> Tuple[float, float]:
        if not self.active or not self._adapter3.value:
            return 0.0, 0.0
        local_info = _DXGI_QUERY_VIDEO_MEMORY_INFO()
        non_local_info = _DXGI_QUERY_VIDEO_MEMORY_INFO()
        ded_gb = 0.0
        shr_gb = 0.0
        hr_local = self._call_com(
            self._adapter3,
            14,
            [
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.c_int,
                ctypes.POINTER(_DXGI_QUERY_VIDEO_MEMORY_INFO),
            ],
            0,
            0,
            ctypes.byref(local_info),
        )
        if hr_local >= 0:
            ded_gb = float(local_info.CurrentUsage) / (1024**3)
        hr_non_local = self._call_com(
            self._adapter3,
            14,
            [
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.c_int,
                ctypes.POINTER(_DXGI_QUERY_VIDEO_MEMORY_INFO),
            ],
            0,
            1,
            ctypes.byref(non_local_info),
        )
        if hr_non_local >= 0:
            shr_gb = float(non_local_info.CurrentUsage) / (1024**3)
            if non_local_info.Budget > 0:
                self.shared_budget_gb = float(non_local_info.Budget) / (1024**3)
        return ded_gb, shr_gb

    def recalibrate_baseline(self) -> None:
        if self.active:
            _, shr_used = self._query_video_memory_info()
            if self.shared_baseline_gb <= 0.0 or shr_used < self.shared_baseline_gb:
                self.shared_baseline_gb = shr_used

    def poll(self) -> WDDMMemorySnapshot:
        pt_alloc_bytes = 0
        pt_res_bytes = 0
        pt_split_bytes = 0
        if self.device.type == "cuda" and torch.cuda.is_available():
            pt_alloc_bytes = torch.cuda.memory_allocated(self.device)
            pt_res_bytes = torch.cuda.memory_reserved(self.device)
            try:
                stats = torch.cuda.memory_stats(self.device)
                pt_split_bytes = int(
                    stats.get("inactive_split_bytes.all.current", 0)
                )
            except Exception:
                pt_split_bytes = 0

        pt_alloc_gb = pt_alloc_bytes / (1024**3)
        pt_res_gb = pt_res_bytes / (1024**3)
        pt_reclaim_gb = max(0.0, pt_res_gb - pt_alloc_gb)
        pt_split_gb = pt_split_bytes / (1024**3)
        host_total_gb, host_avail_gb = self._query_host_ram_gb()

        if self.active:
            ded_used, shr_used = self._query_video_memory_info()
            ded_avail = max(0.0, self.dedicated_total_gb - ded_used)
            shr_delta = max(0.0, shr_used - self.shared_baseline_gb)
            contig_avail = ded_avail + max(0.0, pt_reclaim_gb - pt_split_gb)
            is_spilling = shr_delta > 0.25 or ded_avail < 1.25
            return WDDMMemorySnapshot(
                dedicated_total_gb=self.dedicated_total_gb,
                dedicated_used_gb=ded_used,
                dedicated_avail_gb=ded_avail,
                shared_budget_gb=self.shared_budget_gb,
                shared_used_gb=shr_used,
                shared_delta_gb=shr_delta,
                pt_allocated_gb=pt_alloc_gb,
                pt_reserved_gb=pt_res_gb,
                pt_reclaimable_gb=pt_reclaim_gb,
                is_spilling=is_spilling,
                pt_inactive_split_gb=pt_split_gb,
                contiguous_avail_gb=contig_avail,
                host_total_gb=host_total_gb,
                host_avail_gb=host_avail_gb,
            )

        free_b, total_b = (0, 0)
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                free_b, total_b = torch.cuda.mem_get_info(self.device)
            except Exception:
                total_b = int(32 * (1024**3))
                free_b = max(0, total_b - pt_res_bytes)
        ded_total = total_b / (1024**3)
        ded_avail = free_b / (1024**3)
        ded_used = max(0.0, ded_total - ded_avail)
        contig_avail = ded_avail + max(0.0, pt_reclaim_gb - pt_split_gb)
        return WDDMMemorySnapshot(
            dedicated_total_gb=ded_total,
            dedicated_used_gb=ded_used,
            dedicated_avail_gb=ded_avail,
            shared_budget_gb=0.0,
            shared_used_gb=0.0,
            shared_delta_gb=0.0,
            pt_allocated_gb=pt_alloc_gb,
            pt_reserved_gb=pt_res_gb,
            pt_reclaimable_gb=pt_reclaim_gb,
            is_spilling=False,
            pt_inactive_split_gb=pt_split_gb,
            contiguous_avail_gb=contig_avail,
            host_total_gb=host_total_gb,
            host_avail_gb=host_avail_gb,
        )

    def enforce_stage_budget(
        self,
        required_headroom_gb: float = 2.00,
        force: bool = False,
    ) -> WDDMMemorySnapshot:
        snap = self.poll()
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return snap
        reserved_bytes = int(snap.pt_reserved_gb * (1024**3))
        split_bytes = int(snap.pt_inactive_split_gb * (1024**3))
        adaptive_trim_bytes = (
            max(
                _VRAM_TRIM_THRESHOLD_BYTES,
                int(snap.dedicated_total_gb * 0.96 * (1024**3)),
            )
            if snap.dedicated_total_gb > 0.0
            else _VRAM_TRIM_THRESHOLD_BYTES
        )
        low_dedicated = snap.dedicated_avail_gb < 1.00
        needs_eviction = (
            force
            or snap.is_spilling
            or (snap.contiguous_avail_gb < required_headroom_gb)
            or (low_dedicated and split_bytes > _MAX_INACTIVE_SPLIT_BYTES)
            or (low_dedicated and reserved_bytes > adaptive_trim_bytes)
        )
        if needs_eviction:
            gc.collect()
            reclaimable_bytes = max(
                0,
                torch.cuda.memory_reserved(self.device)
                - torch.cuda.memory_allocated(self.device),
            )
            if (
                force
                or snap.is_spilling
                or reclaimable_bytes >= _MIN_RECLAIMABLE_CACHE_BYTES
            ):
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()
                snap = self.poll()
        return snap

    def close(self) -> None:
        if self.active and self._adapter3.value:
            try:
                self._call_com(self._adapter3, 2, [ctypes.c_void_p])
            except Exception:
                pass
        self._adapter3 = ctypes.c_void_p()
        self.active = False

    def __del__(self) -> None:
        self.close()


class _MetaWeightInitMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs) if kwargs is not None else {}
        if (
            func in (torch.empty, torch.zeros, torch.ones)
            and kwargs.get("device") is None
        ):
            kwargs["device"] = torch.device("meta")
        return func(*args, **kwargs)


def _pin_cpu_tensor(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None:
        return None
    with torch.inference_mode(False):
        cpu_t = t.detach().to(device="cpu").clone().contiguous()
        if torch.cuda.is_available():
            try:
                return cpu_t.pin_memory()
            except Exception:
                return cpu_t
        return cpu_t


def _evacuate_to_cpu(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None:
        return None
    if t.device.type == "cpu":
        return t
    return t.detach().to(device="cpu").contiguous()


def _probe_image_dimensions(path: Path) -> Optional[Tuple[int, int]]:
    try:
        with open(path, "rb") as f:
            hdr = f.read(64)
        if len(hdr) >= 24 and hdr[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", hdr[16:24])
            if w > 0 and h > 0:
                return int(w), int(h)
        if len(hdr) >= 48 and hdr[:16] == b"VIZJ_CONTAINER01":
            orig_w, orig_h, vae_w, vae_h = struct.unpack("<IIII", hdr[32:48])
            if vae_w > 0 and vae_h > 0:
                return int(vae_w), int(vae_h)
            if orig_w > 0 and orig_h > 0:
                return int(orig_w), int(orig_h)
        with Image.open(path) as img:
            return int(img.width), int(img.height)
    except Exception:
        return None


def _resolve_unboxed_reference_dims(
    path: Path,
    req_w: int,
    req_h: int,
    edit_format: str,
) -> Tuple[int, int]:
    fmt = str(edit_format or "standard").strip().lower()
    cap_w = (
        min(req_w, _AI_TOOLKIT_MAX_REF_DIM)
        if fmt in ("ai_toolkit_isolated", "ai_toolkit_full")
        else req_w
    )
    cap_h = (
        min(req_h, _AI_TOOLKIT_MAX_REF_DIM)
        if fmt in ("ai_toolkit_isolated", "ai_toolkit_full")
        else req_h
    )
    max_area = float(max(64 * 64, cap_w * cap_h))
    dims = _probe_image_dimensions(path)
    if dims is not None and dims[0] > 0 and dims[1] > 0:
        src_w, src_h = dims
        src_area = float(src_w * src_h)
        scale = min(1.0, math.sqrt(max_area / src_area))
        out_w = max(64, int(round((src_w * scale) / 16.0)) * 16)
        out_h = max(64, int(round((src_h * scale) / 16.0)) * 16)
        return out_w, out_h
    return max(64, (cap_w // 16) * 16), max(64, (cap_h // 16) * 16)


def _normalize_vizinjector_framing(framing_mode: Optional[str]) -> str:
    fm = str(framing_mode or "inscribed").strip().lower()
    if fm in ("inscribed", "cover_crop"):
        return "inscribed"
    if fm in ("contain", "stretch", "never_crop"):
        return fm
    return "inscribed"


def build_box_aperture_mask(
    layout_items: Sequence[Dict[str, Any]],
    height: int,
    width: int,
    coords_are_pixels: bool,
    feather_radius_lat: float,
    device: torch.device,
    only_flagged_boxes: bool = False,
) -> Optional[torch.Tensor]:
    h_lat = max(1, height // 16)
    w_lat = max(1, width // 16)
    active_boxes: List[Tuple[float, float, float, float]] = []
    for item in layout_items:
        if not isinstance(item, dict) or item.get("enabled", True) is False:
            continue
        if only_flagged_boxes and not (
            bool(item.get("inpaint_box", False))
            or bool(item.get("paste_to_canvas", False))
        ):
            continue
        bbox = item.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            uy0, ux0, uy1, ux1 = bbox_to_normalized(
                bbox, height=height, width=width, coords_are_pixels=coords_are_pixels
            )
            if (uy1 - uy0) > 1e-4 and (ux1 - ux0) > 1e-4:
                active_boxes.append((uy0, ux0, uy1, ux1))

    if not active_boxes:
        return None

    yy = torch.arange(h_lat, device=device, dtype=torch.float32).unsqueeze(1) + 0.5
    xx = torch.arange(w_lat, device=device, dtype=torch.float32).unsqueeze(0) + 0.5
    feather = max(0.25, float(feather_radius_lat))
    mask = torch.zeros((h_lat, w_lat), device=device, dtype=torch.float32)

    for uy0, ux0, uy1, ux1 in active_boxes:
        ly0 = uy0 * h_lat
        ly1 = uy1 * h_lat
        lx0 = ux0 * w_lat
        lx1 = ux1 * w_lat
        dy = torch.maximum(torch.maximum(ly0 - yy, yy - ly1), torch.zeros_like(yy))
        dx = torch.maximum(torch.maximum(lx0 - xx, xx - lx1), torch.zeros_like(xx))
        dist = torch.sqrt(dy * dy + dx * dx)
        box_m = 0.5 * (1.0 + torch.cos(math.pi * (dist / feather).clamp(0.0, 1.0)))
        mask = torch.maximum(mask, box_m)

    return mask.unsqueeze(0).unsqueeze(0).clamp(0.0, 1.0)


def splat_box_latents_to_canvas(
    base_canvas: Optional[torch.Tensor],
    splat_entries: Sequence[
        Tuple[
            torch.Tensor,
            Tuple[float, float, float, float],
            Optional[torch.Tensor],
            Optional[float],
        ]
    ],
    h_lat: int,
    w_lat: int,
    latent_channels: int,
    feather_radius_lat: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if base_canvas is not None:
        canvas = base_canvas.to(device=device, dtype=torch.float32).clone()
        if canvas.shape[-2:] != (h_lat, w_lat):
            canvas = F.interpolate(
                canvas, size=(h_lat, w_lat), mode="bilinear", align_corners=False
            )
    else:
        canvas = torch.zeros(
            (1, latent_channels, h_lat, w_lat), device=device, dtype=torch.float32
        )

    union_alpha = torch.zeros((1, 1, h_lat, w_lat), device=device, dtype=torch.float32)
    feather = max(0.25, float(feather_radius_lat))

    for box_lat_cpu, norm_bbox, box_mask_cpu, _ in splat_entries:
        uy0, ux0, uy1, ux1 = norm_bbox
        y0 = max(0, min(h_lat - 1, int(round(uy0 * h_lat))))
        x0 = max(0, min(w_lat - 1, int(round(ux0 * w_lat))))
        y1 = max(y0 + 1, min(h_lat, int(round(uy1 * h_lat))))
        x1 = max(x0 + 1, min(w_lat, int(round(ux1 * w_lat))))
        bh = y1 - y0
        bw = x1 - x0

        b_lat = box_lat_cpu.to(device=device, dtype=torch.float32)
        if b_lat.shape[-2:] != (bh, bw):
            b_lat = F.interpolate(
                b_lat, size=(bh, bw), mode="bilinear", align_corners=False
            )

        yy = torch.arange(bh, device=device, dtype=torch.float32).unsqueeze(1) + 0.5
        xx = torch.arange(bw, device=device, dtype=torch.float32).unsqueeze(0) + 0.5
        d_edge = torch.minimum(
            torch.minimum(yy, float(bh) - yy),
            torch.minimum(xx, float(bw) - xx),
        )
        edge_ramp = 0.5 * (
            1.0 - torch.cos(math.pi * (d_edge / feather).clamp(0.0, 1.0))
        )
        local_alpha = edge_ramp.unsqueeze(0).unsqueeze(0)

        if box_mask_cpu is not None:
            b_mask = box_mask_cpu.to(device=device, dtype=torch.float32)
            if b_mask.shape[-2:] != (bh, bw):
                b_mask = F.interpolate(
                    b_mask, size=(bh, bw), mode="bilinear", align_corners=False
                )
            local_alpha = local_alpha * b_mask.clamp(0.0, 1.0)

        region = canvas[:, :, y0:y1, x0:x1]
        canvas[:, :, y0:y1, x0:x1] = (
            local_alpha * b_lat + (1.0 - local_alpha) * region
        )
        union_alpha[:, :, y0:y1, x0:x1] = torch.maximum(
            union_alpha[:, :, y0:y1, x0:x1], local_alpha
        )

    return canvas, union_alpha.clamp(0.0, 1.0)


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


def compute_moss_v2_sigmas(
    num_inference_steps: int,
    shift: float = 5.0,
    training_shift: float = 1.0,
    num_train_timesteps: int = 1000,
) -> torch.Tensor:
    if num_inference_steps < 1 or shift <= 0 or training_shift <= 0:
        raise ValueError("Steps and shifts must be positive.")
    total = int(num_train_timesteps)

    def _grid(count: int, s: float) -> torch.Tensor:
        lower = s / (total + s - 1.0)
        values = torch.linspace(1.0, lower, count, dtype=torch.float32)
        return s * values / (1.0 + (s - 1.0) * values)

    sigmas = _grid(num_inference_steps, float(shift))
    training = _grid(total, float(training_shift)).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where(
        (sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(),
        lower,
        upper,
    )
    return torch.unique_consecutive(training[nearest])


class MOSSImageScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        training_shift: float = 1.0,
    ):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.training_shift = float(training_shift)
        self.timesteps: Optional[torch.Tensor] = None
        self.sigmas: Optional[torch.Tensor] = None
        self._step_index: Optional[int] = None

    def reset(self) -> None:
        self._step_index = 0

    def set_timesteps(
        self,
        num_inference_steps: int = 50,
        device: Optional[Union[str, torch.device]] = None,
        shift: Optional[float] = None,
    ) -> None:
        if shift is not None:
            self.shift = float(shift)
        target_device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        snapped = compute_moss_v2_sigmas(
            num_inference_steps=num_inference_steps,
            shift=self.shift,
            training_shift=self.training_shift,
            num_train_timesteps=self.num_train_timesteps,
        ).to(device=target_device)
        self.timesteps = snapped * float(self.num_train_timesteps)
        self.sigmas = torch.cat(
            [snapped, torch.zeros(1, device=target_device, dtype=torch.float32)]
        )
        self._step_index = 0

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        num_steps = len(self.timesteps)
        if self._step_index is None or self._step_index >= num_steps:
            self._step_index = 0
        idx = self._step_index
        dt = self.sigmas[idx + 1] - self.sigmas[idx]
        prev_sample = sample.float() + dt * model_output.float()
        self._step_index += 1
        if self._step_index >= num_steps:
            self._step_index = None
        return (prev_sample,) if return_dict else prev_sample


class BifurcatedFlowMatchScheduler:
    def __init__(
        self,
        early_solver: str = "euler",
        late_solver: str = "euler",
        handoff_threshold: float = 0.35,
        shift: float = 5.0,
        training_shift: float = 1.0,
        num_train_timesteps: int = 1000,
        ipndm_max_order: int = 4,
        eta: float = 0.0,
        s_noise: float = 1.0,
        use_v2_grid: bool = True,
    ):
        self.early_solver = str(early_solver).lower()
        self.late_solver = str(late_solver).lower()
        self.handoff_threshold = max(0.0, min(1.0, float(handoff_threshold)))
        self.shift = float(shift)
        self.training_shift = float(training_shift)
        self.num_train_timesteps = int(num_train_timesteps)
        self.ipndm_max_order = max(1, min(4, int(ipndm_max_order)))
        self.eta = max(0.0, min(1.0, float(eta)))
        self.s_noise = max(0.0, float(s_noise))
        self.use_v2_grid = bool(use_v2_grid)

        self.base_sigmas: Optional[torch.Tensor] = None
        self.sigmas: Optional[torch.Tensor] = None
        self.timesteps: Optional[torch.Tensor] = None
        self._step_configs: List[Dict[str, Any]] = []
        self._step_index: Optional[int] = None
        self._k_star: int = 0
        self._sample_anchor: Optional[torch.Tensor] = None
        self._v0: Optional[torch.Tensor] = None
        self._h: Optional[float] = None
        self._history: List[torch.Tensor] = []
        self._boundary_flushed: bool = False
        self.has_corrector: bool = False
        self._early_steps: int = 0
        self._late_steps: int = 0

    def reset(self) -> None:
        self._step_index = 0
        self._sample_anchor = None
        self._v0 = None
        self._h = None
        self._history.clear()
        self._boundary_flushed = False

    @property
    def current_is_early(self) -> bool:
        if self._step_index is None or self._step_index >= len(self._step_configs):
            return True
        return bool(self._step_configs[self._step_index]["is_early"])

    @property
    def current_progress(self) -> float:
        if self._step_index is None or not self._step_configs:
            return 0.0
        cfg = self._step_configs[min(self._step_index, len(self._step_configs) - 1)]
        return float(max(0.0, min(1.0, 1.0 - float(cfg["s_curr"]))))

    @staticmethod
    def _lagrange_ab_weights_for_order(
        h_n: float,
        nodes: Sequence[float],
        m: int,
    ) -> Tuple[float, float, float, float]:
        if m <= 1:
            return (h_n, 0.0, 0.0, 0.0)
        active_nodes = nodes[:m]
        for a in range(m):
            for b in range(a + 1, m):
                if abs(active_nodes[a] - active_nodes[b]) < 1e-12:
                    return (h_n, 0.0, 0.0, 0.0)

        def _eval_basis(k: int, x: float) -> float:
            xk = active_nodes[k]
            num = 1.0
            den = 1.0
            for j in range(m):
                if j != k:
                    xj = active_nodes[j]
                    num *= x - xj
                    den *= xk - xj
            return num / den

        mid = 0.5 * h_n
        scale = h_n / 6.0
        w = [0.0, 0.0, 0.0, 0.0]
        for k in range(m):
            f0 = 1.0 if k == 0 else 0.0
            f_mid = _eval_basis(k, mid)
            f_end = _eval_basis(k, h_n)
            w[k] = scale * (f0 + 4.0 * f_mid + f_end)
        return (w[0], w[1], w[2], w[3])

    @classmethod
    def _compute_ab_weights(
        cls,
        h_n: float,
        h_n_1: float,
        h_n_2: float,
        h_n_3: float,
        order: int,
    ) -> Tuple[float, float, float, float]:
        if order <= 1 or abs(h_n) < 1e-12 or abs(h_n_1) < 1e-12:
            return (h_n, 0.0, 0.0, 0.0)
        max_m = min(int(order), 4)
        if max_m >= 2 and abs(h_n_1) < 1e-12:
            max_m = 1
        if max_m >= 3 and abs(h_n_2) < 1e-12:
            max_m = 2
        if max_m >= 4 and abs(h_n_3) < 1e-12:
            max_m = 3

        x0 = 0.0
        x1 = -h_n_1
        x2 = -(h_n_1 + h_n_2)
        x3 = -(h_n_1 + h_n_2 + h_n_3)
        nodes = (x0, x1, x2, x3)
        abs_h = abs(h_n)
        for m in range(max_m, 1, -1):
            cand = cls._lagrange_ab_weights_for_order(h_n, nodes, m)
            l1_sum = abs(cand[0]) + abs(cand[1]) + abs(cand[2]) + abs(cand[3])
            if l1_sum <= 2.15 * abs_h + 1e-12:
                return cand
        return (h_n, 0.0, 0.0, 0.0)

    def set_timesteps(
        self,
        num_inference_steps: int = 50,
        device: Optional[Union[str, torch.device]] = None,
        shift: Optional[float] = None,
        handoff_threshold: Optional[float] = None,
        start_step: int = 0,
    ) -> None:
        if shift is not None:
            self.shift = float(shift)
        if handoff_threshold is not None:
            self.handoff_threshold = max(0.0, min(1.0, float(handoff_threshold)))
        target_device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )

        if self.use_v2_grid:
            snapped = compute_moss_v2_sigmas(
                num_inference_steps=num_inference_steps,
                shift=self.shift,
                training_shift=self.training_shift,
                num_train_timesteps=self.num_train_timesteps,
            )
            sigmas_np = snapped.numpy().astype(np.float64)
        else:
            raw = np.linspace(
                1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=np.float64
            )
            sigmas_np = self.shift * raw / (1.0 + (self.shift - 1.0) * raw)

        if 0 < start_step < len(sigmas_np):
            sigmas_np = sigmas_np[start_step:]

        base_sigmas_np = np.append(sigmas_np, 0.0)
        self.base_sigmas = torch.from_numpy(base_sigmas_np.astype(np.float32)).to(
            device=target_device
        )

        active_steps = len(sigmas_np)
        if self.handoff_threshold >= 1.0:
            k_star = active_steps
        elif self.handoff_threshold <= 0.0:
            k_star = 0
        else:
            cand = [
                i
                for i in range(active_steps)
                if (1.0 - float(base_sigmas_np[i])) >= self.handoff_threshold - 1e-7
            ]
            k_star = cand[0] if cand else active_steps

        self._k_star = k_star
        self._early_steps = k_star
        self._late_steps = active_steps - k_star

        corrector_types = ("heun", "sde_gpu_pp", "multitree")
        early_has_corrector = self.early_solver in corrector_types
        late_has_corrector = self.late_solver in corrector_types

        step_configs: List[Dict[str, Any]] = []
        eval_timesteps: List[float] = []

        for i in range(active_steps):
            is_early = i < k_star
            solver = self.early_solver if is_early else self.late_solver
            s_curr = float(base_sigmas_np[i])
            s_next = float(base_sigmas_np[i + 1])
            dt = s_next - s_curr
            interval_corrector = (
                (early_has_corrector if is_early else late_has_corrector)
                and (s_next > 0.0)
            )
            if interval_corrector:
                t_pred = s_curr * self.num_train_timesteps
                t_corr = s_next * self.num_train_timesteps
                eval_timesteps.extend([t_pred, t_corr])
                step_configs.append(
                    {
                        "mode": "predictor",
                        "interval_idx": i,
                        "is_early": is_early,
                        "s_curr": s_curr,
                        "s_next": s_next,
                        "dt": dt,
                        "timestep": t_pred,
                        "solver": solver,
                    }
                )
                step_configs.append(
                    {
                        "mode": "corrector",
                        "interval_idx": i,
                        "is_early": is_early,
                        "s_curr": s_curr,
                        "s_next": s_next,
                        "dt": dt,
                        "timestep": t_corr,
                        "solver": solver,
                    }
                )
            else:
                t_step = s_curr * self.num_train_timesteps
                eval_timesteps.append(t_step)
                j = i if is_early else (i - k_star)
                order = min(self.ipndm_max_order, j + 1)
                h_n_1 = (
                    float(base_sigmas_np[i] - base_sigmas_np[i - 1]) if j >= 1 else 0.0
                )
                h_n_2 = (
                    float(base_sigmas_np[i - 1] - base_sigmas_np[i - 2])
                    if j >= 2
                    else 0.0
                )
                h_n_3 = (
                    float(base_sigmas_np[i - 2] - base_sigmas_np[i - 3])
                    if j >= 3
                    else 0.0
                )
                weights = self._compute_ab_weights(dt, h_n_1, h_n_2, h_n_3, order)
                step_configs.append(
                    {
                        "mode": "multistep",
                        "interval_idx": i,
                        "is_early": is_early,
                        "s_curr": s_curr,
                        "s_next": s_next,
                        "dt": dt,
                        "timestep": t_step,
                        "solver": solver,
                        "order": order,
                        "weights": weights,
                    }
                )

        self._step_configs = step_configs
        self.timesteps = torch.tensor(
            eval_timesteps, dtype=torch.float32, device=target_device
        )
        self.sigmas = self.base_sigmas
        self.has_corrector = any(cfg["mode"] == "corrector" for cfg in step_configs)
        self.reset()

    def _push_history(self, v: torch.Tensor) -> None:
        if len(self._history) >= self.ipndm_max_order - 1:
            self._history.pop(0)
        self._history.append(v.detach())

    def _calc_sde_dispersion(
        self,
        ref_tensor: torch.Tensor,
        v_tot: torch.Tensor,
        s_curr: float,
        s_next: float,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if self.eta <= 0.0 or self.s_noise <= 0.0 or self.handoff_threshold <= 0.0:
            return torch.zeros_like(ref_tensor)
        progress = self.current_progress
        if progress >= self.handoff_threshold or s_next <= 0.0:
            return torch.zeros_like(ref_tensor)
        envelope = max(
            0.0,
            min(1.0, (self.handoff_threshold - progress) / self.handoff_threshold),
        )
        dt_val = abs(s_next - s_curr)
        max_rho = 0.5 * s_next
        renoise = min(
            max_rho,
            self.eta * s_next * math.sqrt(max(dt_val, 1e-8)) * envelope * self.s_noise,
        )
        if renoise <= 0.0:
            return torch.zeros_like(ref_tensor)
        gen_dev = generator.device.type if generator is not None else None
        if generator is not None and gen_dev == ref_tensor.device.type:
            noise = torch.randn(
                ref_tensor.shape,
                generator=generator,
                device=ref_tensor.device,
                dtype=torch.float32,
            )
        else:
            noise = torch.randn(
                ref_tensor.shape, device=ref_tensor.device, dtype=torch.float32
            )
        eps_hat = ref_tensor + (1.0 - s_curr) * v_tot
        attenuation = (
            math.sqrt(max(0.0, s_next * s_next - renoise * renoise)) - s_next
        )
        return (attenuation * eps_hat) + (noise * renoise)

    def step(
        self,
        model_output: Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]],
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        num_passes = len(self._step_configs)
        if self._step_index is None or self._step_index >= num_passes:
            self.reset()

        v = (
            model_output[0].float()
            if isinstance(model_output, (tuple, list))
            else model_output.float()
        )
        sample_f32 = sample.float()
        cfg = self._step_configs[self._step_index]
        mode = cfg["mode"]
        s_curr = cfg["s_curr"]
        s_next = cfg["s_next"]
        dt = cfg["dt"]
        is_early = cfg["is_early"]
        solver = cfg["solver"]

        if not is_early and not self._boundary_flushed:
            self._history.clear()
            self._boundary_flushed = True

        if mode == "predictor":
            self._sample_anchor = sample_f32.clone()
            self._v0 = v.detach()
            self._h = dt
            prev_sample = sample_f32 + dt * v
        elif mode == "corrector":
            sample_0 = (
                self._sample_anchor if self._sample_anchor is not None else sample_f32
            )
            v0 = self._v0 if self._v0 is not None else v
            h_dt = self._h if self._h is not None else dt
            if solver in ("heun", "sde_gpu_pp", "multitree"):
                dx = (h_dt / 2.0) * (v0 + v)
            else:
                dx = h_dt * v0
            if is_early and self.eta > 0.0 and s_next > 0.0:
                v_tot = 0.5 * (v0 + v)
                disp = self._calc_sde_dispersion(
                    sample_0, v_tot, s_curr, s_next, generator=generator
                )
                prev_sample = sample_0 + dx + disp
            else:
                prev_sample = sample_0 + dx
            self._sample_anchor = None
            self._v0 = None
            self._h = None
        else:
            c0, c1, c2, c3 = cfg["weights"]
            if solver in ("ipndm", "multitree"):
                dx = c0 * v
                if len(self._history) >= 1 and c1 != 0.0:
                    dx = dx + c1 * self._history[-1]
                if len(self._history) >= 2 and c2 != 0.0:
                    dx = dx + c2 * self._history[-2]
                if len(self._history) >= 3 and c3 != 0.0:
                    dx = dx + c3 * self._history[-3]
            else:
                dx = dt * v

            if is_early and self.eta > 0.0 and s_next > 0.0:
                disp = self._calc_sde_dispersion(
                    sample_f32, v, s_curr, s_next, generator=generator
                )
                prev_sample = sample_f32 + dx + disp
            else:
                prev_sample = sample_f32 + dx

            if solver in ("ipndm", "multitree") and self.ipndm_max_order > 1:
                self._push_history(v)

        self._step_index += 1
        if self._step_index >= num_passes:
            self.reset()
        return (prev_sample,) if return_dict else prev_sample


class PromptEmbedCache:
    def __init__(self, max_entries: int = 32):
        self.max_entries = max_entries
        self._cache: OrderedDict[Tuple[Any, ...], torch.Tensor] = OrderedDict()
        self._last_tokens: Optional[List[int]] = None
        self.hits: int = 0
        self.pos_hits: int = 0
        self.misses: int = 0
        self.last_common_prefix_len: int = 0

    def clear(self) -> None:
        self._cache.clear()
        self._last_tokens = None

    def lookup(
        self,
        token_ids: Sequence[int],
        device: torch.device,
        is_negative: bool = False,
        vision_signature: Tuple[Any, ...] = (),
    ) -> Optional[torch.Tensor]:
        tok_list = [int(x) for x in token_ids]
        if not is_negative:
            if self._last_tokens is not None:
                self.last_common_prefix_len = MOSSImageTokenizer.find_common_prefix(
                    self._last_tokens, tok_list
                )
            else:
                self.last_common_prefix_len = 0
            self._last_tokens = tok_list

        key: Tuple[Any, ...] = (
            (tuple(tok_list), vision_signature)
            if vision_signature
            else tuple(tok_list)
        )
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            if not is_negative:
                self.pos_hits += 1
            return self._cache[key].to(device=device, non_blocking=True)
        self.misses += 1
        return None

    def insert(
        self,
        token_ids: Sequence[int],
        hidden: torch.Tensor,
        vision_signature: Tuple[Any, ...] = (),
    ) -> None:
        key: Tuple[Any, ...] = (
            (tuple(int(x) for x in token_ids), vision_signature)
            if vision_signature
            else tuple(int(x) for x in token_ids)
        )
        pinned = _pin_cpu_tensor(hidden)
        if pinned is not None:
            self._cache[key] = pinned
            self._cache.move_to_end(key)
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)


@dataclass
class CachedReferenceRecord:
    packed_latents: Optional[torch.Tensor]
    vit_embeds: Optional[torch.Tensor]
    vit_patches: Optional[torch.Tensor]
    grid_thw: Optional[torch.Tensor]
    llm_tokens: int
    latent_mask: Optional[torch.Tensor]
    viz_meta: Optional[Any] = None
    source_signature: Tuple[Any, ...] = ()


class ReferenceVisionCache:
    def __init__(self, max_entries: int = 32):
        self.max_entries = max_entries
        self._cache: OrderedDict[Tuple[Any, ...], CachedReferenceRecord] = OrderedDict()
        self._vae_cache: OrderedDict[
            Tuple[Any, ...],
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Any]],
        ] = OrderedDict()
        self.hits: int = 0
        self.misses: int = 0

    def clear(self) -> None:
        self._cache.clear()
        self._vae_cache.clear()

    @staticmethod
    def canonicalize_framing(
        path: Path, target_w: int, target_h: int, framing_mode: str
    ) -> str:
        fm = str(framing_mode).lower()
        if fm == "cover_crop":
            fm = "inscribed"
        if fm == "never_crop":
            dims = _probe_image_dimensions(path)
            if dims is not None:
                src_w, src_h = dims
                if src_w * target_h == src_h * target_w:
                    return "inscribed"
        return fm

    @classmethod
    def make_vae_key(
        cls,
        path: Path,
        target_w: int,
        target_h: int,
        framing_mode: str,
        input_role: str,
        clamp_sigma: float,
        mtf_boost: float = 0.08,
        mask_feather_radius: float = 1.25,
    ) -> Tuple[Any, ...]:
        st = path.stat()
        norm_fm = cls.canonicalize_framing(path, target_w, target_h, framing_mode)
        eff_mtf = round(float(mtf_boost), 4) if input_role == "photo_ref" else 0.0
        return (
            str(path.resolve()),
            int(st.st_mtime_ns),
            int(st.st_size),
            int(target_w),
            int(target_h),
            norm_fm,
            str(input_role),
            round(float(clamp_sigma), 4),
            eff_mtf,
            round(float(mask_feather_radius), 4),
        )

    @classmethod
    def make_key(
        cls,
        path: Path,
        target_w: int,
        target_h: int,
        framing_mode: str,
        input_role: str,
        edit_format: str,
        max_vit_patches: int,
        clamp_sigma: float,
    ) -> Tuple[Any, ...]:
        st = path.stat()
        norm_fm = cls.canonicalize_framing(path, target_w, target_h, framing_mode)
        return (
            str(path.resolve()),
            int(st.st_mtime_ns),
            int(st.st_size),
            int(target_w),
            int(target_h),
            norm_fm,
            str(input_role),
            str(edit_format),
            int(max_vit_patches),
            round(float(clamp_sigma), 4),
        )

    def lookup(self, key: Tuple[Any, ...]) -> Optional[CachedReferenceRecord]:
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def lookup_vae(
        self, vae_key: Tuple[Any, ...]
    ) -> Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Any]]]:
        if vae_key in self._vae_cache:
            self._vae_cache.move_to_end(vae_key)
            self.hits += 1
            return self._vae_cache[vae_key]
        return None

    def insert(
        self,
        key: Tuple[Any, ...],
        record: CachedReferenceRecord,
        vae_key: Optional[Tuple[Any, ...]] = None,
    ) -> CachedReferenceRecord:
        pinned_rec = CachedReferenceRecord(
            packed_latents=_pin_cpu_tensor(record.packed_latents),
            vit_embeds=_pin_cpu_tensor(record.vit_embeds),
            vit_patches=_pin_cpu_tensor(record.vit_patches),
            grid_thw=_pin_cpu_tensor(record.grid_thw),
            llm_tokens=int(record.llm_tokens),
            latent_mask=_pin_cpu_tensor(record.latent_mask),
            viz_meta=record.viz_meta,
            source_signature=record.source_signature or key[:6],
        )
        self._cache[key] = pinned_rec
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

        if vae_key is not None and pinned_rec.packed_latents is not None:
            self._vae_cache[vae_key] = (
                pinned_rec.packed_latents,
                pinned_rec.latent_mask,
                pinned_rec.viz_meta,
            )
            self._vae_cache.move_to_end(vae_key)
            while len(self._vae_cache) > self.max_entries:
                self._vae_cache.popitem(last=False)
        return pinned_rec


def _format_seconds(seconds: float) -> str:
    s = max(0, int(seconds))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class EngineProgressReporter:
    def __init__(self) -> None:
        self.current_stage: Optional[str] = None
        self.stage_start: float = time.perf_counter()
        self.last_render: float = 0.0
        self.render_interval: float = 0.08
        self.last_count: int = 0
        self.stage_durations: Dict[str, float] = {}

    def start_stage(self, stage: str, total: int) -> None:
        now = time.perf_counter()
        if self.current_stage is not None and self.current_stage != stage:
            self._finish_stage(now)
        self.current_stage = stage
        self.stage_start = now
        self.last_render = 0.0
        self.last_count = 0
        self._render_bar(stage, 0, total, now)

    def __call__(self, stage: str, current: int, total: int) -> None:
        now = time.perf_counter()
        if stage != self.current_stage:
            if self.current_stage is not None:
                self._finish_stage(now)
            self.current_stage = stage
            self.stage_start = now
            self.last_render = 0.0
            self.last_count = 0

        self.last_count = current
        is_final = current >= total
        if not is_final and (now - self.last_render < self.render_interval):
            return

        self.last_render = now
        self._render_bar(stage, current, total, now)
        if is_final:
            self._finish_stage(now)

    def _render_bar(self, stage: str, current: int, total: int, now: float) -> None:
        elapsed = max(now - self.stage_start, 1e-4)
        pct = (current / max(total, 1)) * 100.0
        rate = (current / elapsed) if (elapsed >= 0.05 and current >= 1) else 0.0
        remaining_units = max(0, total - current)
        eta = (remaining_units / rate) if rate > 0.0 else 0.0
        time_str = f"[{_format_seconds(elapsed)}<{_format_seconds(eta)}"

        if stage == "context":
            prefix = "[1/3 Qwen3.5 Context]"
            rate_str = f"{rate:5.1f} seq/s" if rate > 0 else "  --.- seq/s"
            suffix = f" {pct:5.1f}% | {current:2d}/{total:2d} seq {time_str}, {rate_str}]"
        elif stage == "dit":
            prefix = "[2/3 MOSS-DiT Flow]"
            rate_str = f"{rate:5.2f} stp/s" if rate > 0 else "  --.-- stp/s"
            suffix = f" {pct:5.1f}% | {current:3d}/{total:3d} stp {time_str}, {rate_str}]"
        elif stage == "vae":
            prefix = "[3/3 Flux2-VAE FP32]"
            rate_str = f"{rate:5.2f} img/s" if rate > 0 else "  --.-- img/s"
            suffix = f" {pct:5.1f}% | {current:2d}/{total:2d} img {time_str}, {rate_str}]"
        else:
            prefix = f"[{stage.upper()}]"
            suffix = f" {pct:5.1f}% | {current}/{total} {time_str}]"

        cols = max(60, min(shutil.get_terminal_size((80, 20)).columns, 140))
        avail = cols - len(prefix) - len(suffix) - 3
        bar_len = max(8, min(avail, 30))
        filled = min(bar_len, int(bar_len * current / max(total, 1)))
        bar = "=" * filled + ">" if (filled < bar_len and current > 0) else "=" * filled
        bar = bar.ljust(bar_len, " ")
        line = f"\r{prefix} [{bar}]{suffix}"
        sys.stdout.write(line.ljust(cols - 1)[: cols - 1])
        sys.stdout.flush()

    def _finish_stage(self, now: float) -> None:
        elapsed = max(now - self.stage_start, 1e-4)
        if self.current_stage is not None:
            self.stage_durations[self.current_stage] = elapsed
        stage_names = {
            "context": "Qwen3.5 Hybrid Context",
            "dit": "MOSS-DiT Flow Matching",
            "vae": "Flux2-VAE FP32 Decode",
        }
        name = stage_names.get(
            self.current_stage,
            self.current_stage.upper() if self.current_stage else "Stage",
        )
        cols = max(60, min(shutil.get_terminal_size((80, 20)).columns, 140))
        summary = f"\r[+] {name:<26} finalized in {elapsed:6.2f}s ({self.last_count} units)."
        sys.stdout.write(summary.ljust(cols - 1)[: cols - 1] + "\n")
        sys.stdout.flush()
        self.current_stage = None

    def close(self) -> None:
        if self.current_stage is not None:
            self._finish_stage(time.perf_counter())


def resolve_model_path(repo_or_path: Union[str, Path]) -> Path:
    direct_path = Path(repo_or_path)
    if direct_path.exists():
        return direct_path.resolve()

    repo_str = str(repo_or_path).replace("\\", "/")
    hub_candidates = [
        CACHE_DIR / "hub",
        CACHE_DIR,
        ROOT_DIR.parent / ".hf_cache" / "hub",
        ROOT_DIR / ".hf_cache" / "hub",
    ]
    repo_folder_name = "models--" + repo_str.replace("/", "--")
    seen_dirs = set()
    for hub_dir in hub_candidates:
        resolved_hub = hub_dir.resolve()
        if resolved_hub in seen_dirs or not resolved_hub.exists():
            continue
        seen_dirs.add(resolved_hub)
        cand = resolved_hub / repo_folder_name / "snapshots"
        if cand.exists():
            snapshots = sorted(
                [p for p in cand.iterdir() if p.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if snapshots:
                return snapshots[0].resolve()
    raise FileNotFoundError(
        f"Cached snapshot not found for '{repo_or_path}' in {CACHE_DIR / 'hub'}. Run download_model.py first."
    )


def _resolve_target_tensor_dtype(
    clean_key: str,
    src_tensor: torch.Tensor,
    dtype: Optional[torch.dtype],
) -> torch.dtype:
    if not src_tensor.is_floating_point():
        return src_tensor.dtype
    if (
        clean_key.endswith("bands")
        or clean_key.endswith("inv_freq")
        or clean_key.startswith("bn.")
    ):
        return torch.float32
    if clean_key.endswith("A_log") or dtype is None:
        return src_tensor.dtype
    if src_tensor.dtype == torch.float32 and dtype != torch.float32:
        return dtype
    return src_tensor.dtype


def stream_safetensors_inplace(
    model_dir: Path,
    target_module: torch.nn.Module,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> bool:
    with torch.inference_mode(False), torch.no_grad():
        target_dev = _canonicalize_device(device)
        sf_files = sorted(set(model_dir.glob("*.safetensors")))
        if not sf_files:
            raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

        param_map: Dict[str, torch.nn.Parameter] = dict(
            target_module.named_parameters()
        )
        buf_map: Dict[str, torch.Tensor] = dict(target_module.named_buffers())

        for p in param_map.values():
            if p.is_meta or not _same_device(p.device, target_dev):
                return False

        for sf_path in sf_files:
            resolved_sf = sf_path.resolve(strict=True)
            with safe_open(str(resolved_sf), framework="pt", device="cpu") as f:
                for k in f.keys():
                    clean_key = k[6:] if k.startswith("model.") else k
                    target_t: Optional[torch.Tensor] = param_map.get(clean_key)
                    if target_t is None:
                        target_t = buf_map.get(clean_key)
                    if target_t is None:
                        continue

                    v = f.get_tensor(k)
                    t_dtype = _resolve_target_tensor_dtype(clean_key, v, dtype)
                    if (
                        not target_t.is_meta
                        and _same_device(target_t.device, target_dev)
                        and target_t.shape == v.shape
                        and target_t.dtype == t_dtype
                        and not target_t.is_inference()
                    ):
                        if v.dtype != t_dtype:
                            v_cast = v.to(dtype=t_dtype)
                            target_t.data.copy_(v_cast, non_blocking=True)
                            del v_cast
                        else:
                            target_t.data.copy_(v, non_blocking=True)
                    else:
                        target_t.data = v.to(device=target_dev, dtype=t_dtype)
                    del v

        clear_fn = getattr(target_module, "clear_caches", None)
        if callable(clear_fn):
            clear_fn()
        if target_dev.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(target_dev)
        return True


def load_sharded_safetensors_into(
    model_dir: Path,
    target_module: torch.nn.Module,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> None:
    with torch.inference_mode(False), torch.no_grad():
        target_dev = _canonicalize_device(device)
        first_param = next(target_module.parameters(), None)
        if (
            first_param is not None
            and not first_param.is_meta
            and _same_device(first_param.device, target_dev)
        ):
            if stream_safetensors_inplace(
                model_dir, target_module, device=target_dev, dtype=dtype
            ):
                return

        sf_files = sorted(set(model_dir.glob("*.safetensors")))
        if not sf_files:
            raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

        state_dict: Dict[str, torch.Tensor] = {}
        for sf_path in sf_files:
            resolved_sf = sf_path.resolve(strict=True)
            with safe_open(str(resolved_sf), framework="pt", device="cpu") as f:
                for k in f.keys():
                    clean_key = k[6:] if k.startswith("model.") else k
                    v = f.get_tensor(k)
                    t_dtype = _resolve_target_tensor_dtype(clean_key, v, dtype)
                    if v.dtype != t_dtype:
                        state_dict[clean_key] = v.to(device=target_dev, dtype=t_dtype)
                    else:
                        state_dict[clean_key] = v.to(device=target_dev)
                    del v

        target_module.load_state_dict(state_dict, strict=True, assign=True)
        del state_dict
        gc.collect()

        for name, param in target_module.named_parameters():
            if not _same_device(param.device, target_dev):
                t_dtype = _resolve_target_tensor_dtype(name, param, dtype)
                param.data = param.data.to(device=target_dev, dtype=t_dtype)

        for name, buf in target_module.named_buffers():
            if buf.is_floating_point():
                if (
                    name.endswith("bands")
                    or name.endswith("inv_freq")
                    or dtype is None
                ):
                    if (
                        not _same_device(buf.device, target_dev)
                        or buf.dtype != torch.float32
                    ):
                        buf.data = buf.data.to(device=target_dev, dtype=torch.float32)
                elif not _same_device(buf.device, target_dev):
                    buf.data = buf.data.to(device=target_dev)
            elif not _same_device(buf.device, target_dev):
                buf.data = buf.data.to(device=target_dev)


def compute_image_telemetry(pil_img: Image.Image) -> Tuple[float, float, float, float]:
    rgb = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    mean_luma = float(np.mean(luma))
    rms_contrast = float(np.std(luma))

    luma_u8 = np.clip(np.round(luma * 255.0), 0, 255).astype(np.uint8)
    hist = np.bincount(luma_u8.ravel(), minlength=256).astype(np.float64)
    probs = hist[hist > 0] / float(luma_u8.size)
    entropy = float(-np.sum(probs * np.log2(probs)))

    if luma.shape[0] >= 3 and luma.shape[1] >= 3:
        lap = (
            luma[:-2, 1:-1]
            + luma[2:, 1:-1]
            + luma[1:-1, :-2]
            + luma[1:-1, 2:]
            - 4.0 * luma[1:-1, 1:-1]
        )
        sharpness = float(np.var(lap) * 1000.0)
    else:
        sharpness = 0.0
    return mean_luma, rms_contrast, entropy, sharpness


def load_image_tensor_16px(
    image_path: Union[str, Path],
    target_h: Optional[int] = None,
    target_w: Optional[int] = None,
) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    w_in, h_in = img.size
    if target_h is not None and target_w is not None:
        h_out = max(64, (target_h // 16) * 16)
        w_out = max(64, (target_w // 16) * 16)
    else:
        scale = min(1.0, math.sqrt((1024.0 * 1024.0) / float(max(1, w_in * h_in))))
        h_out = max(64, (int(round(h_in * scale)) // 16) * 16)
        w_out = max(64, (int(round(w_in * scale)) // 16) * 16)
    if (w_in, h_in) != (w_out, h_out):
        img = img.resize((w_out, h_out), resample=Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def pack_vision_patches_32px(
    image_path: Union[str, Path], max_patches: int = 2304
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    img = Image.open(image_path).convert("RGB")
    w_in, h_in = img.size
    max_pixels = max_patches * 16 * 16
    scale = min(1.0, math.sqrt(float(max_pixels) / float(max(1, w_in * h_in))))
    h_vis = max(32, (int(round(h_in * scale)) // 32) * 32)
    w_vis = max(32, (int(round(w_in * scale)) // 32) * 32)
    if (w_in, h_in) != (w_vis, h_vis):
        img = img.resize((w_vis, h_vis), resample=Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    normed = (arr - mean) / std
    chw = torch.from_numpy(normed).permute(2, 0, 1).contiguous()
    grid_h = h_vis // 16
    grid_w = w_vis // 16
    merge = 2
    patches = chw.unsqueeze(1).expand(3, 2, h_vis, w_vis)
    patches = patches.view(
        3, 2, grid_h // merge, merge, 16, grid_w // merge, merge, 16
    )
    patches = (
        patches.permute(2, 5, 3, 6, 0, 1, 4, 7)
        .contiguous()
        .view(grid_h * grid_w, 3, 2, 16, 16)
    )
    grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)
    llm_tokens = (grid_h * grid_w) // (merge * merge)
    return patches, grid_thw, llm_tokens


class MOSSImageEngine:
    def __init__(
        self,
        model_variant: str = "sft",
        repo_id: Optional[str] = None,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        vae_dtype: torch.dtype = torch.float32,
        prefer_vulkan: bool = True,
    ):
        self.model_variant = model_variant.lower()
        self.repo_id = repo_id or SUPPORTED_MODELS.get(
            self.model_variant, SUPPORTED_MODELS["sft"]
        )
        self.device = _canonicalize_device(device)
        self.dtype = dtype
        self.vae_dtype = vae_dtype

        self.tokenizer: Optional[MOSSImageTokenizer] = None
        self.text_encoder: Optional[Qwen3_5Model] = None
        self.transformer: Optional[MOSSImageTransformer] = None
        self.vae: Optional[Flux2AutoencoderKL] = None

        self.embed_cache = PromptEmbedCache(max_entries=32)
        self.ref_cache = ReferenceVisionCache(max_entries=32)
        self.memory_guard = WDDMMemoryGuard(self.device)

        self.vizinjector: Optional[Any] = None
        if HAS_VIZINJECTOR and VizInjectorClient is not None:
            try:
                local_viz_dll = ROOT_DIR / "vizinjector" / (
                    "vizinjector.dll"
                    if sys.platform == "win32"
                    else "libvizinjector.so"
                )
                self.vizinjector = VizInjectorClient(
                    lib_path=local_viz_dll if local_viz_dll.is_file() else None,
                    prefer_vulkan=prefer_vulkan,
                )
            except Exception:
                self.vizinjector = None

        self._loaded_repo_id: Optional[str] = None
        self._current_offload_state: Optional[bool] = None
        self._warmed_up: bool = False

    def __enter__(self) -> MOSSImageEngine:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()

    @property
    def vizinjector_active(self) -> bool:
        return self.vizinjector is not None

    @property
    def vizinjector_backend_name(self) -> str:
        if self.vizinjector is None:
            return "PyTorch Fallback"
        return (
            f"Vulkan ({self.vizinjector.backend_name})"
            if self.vizinjector.vulkan_active
            else "VizInjector C11 AVX2"
        )

    def _sync_device(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _trim_reserved_vram(
        self, force: bool = False, required_headroom_gb: float = 2.00
    ) -> WDDMMemorySnapshot:
        return self.memory_guard.enforce_stage_budget(
            required_headroom_gb=required_headroom_gb, force=force
        )

    def clear_runtime_caches(self) -> None:
        self.embed_cache.clear()
        self.ref_cache.clear()
        if self.transformer is not None:
            self.transformer.clear_caches()
        if self.text_encoder is not None:
            self.text_encoder.clear_caches()
        if torch.cuda.is_available():
            gc.collect()
            self._sync_device()
            torch.cuda.empty_cache()

    def prepare_for_exit(self) -> None:
        try:
            self._sync_device()
        except Exception:
            pass
        try:
            self.memory_guard.close()
        except Exception:
            pass

    def shutdown(self) -> None:
        with torch.inference_mode(False), torch.no_grad():
            try:
                self._sync_device()
            except Exception:
                pass
            try:
                self.memory_guard.close()
            except Exception:
                pass

            if self.vizinjector is not None:
                try:
                    self.vizinjector.close()
                except Exception:
                    pass
                self.vizinjector = None

            if self.tokenizer is not None:
                close_fn = getattr(self.tokenizer, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception:
                        pass
                self.tokenizer = None

            self.embed_cache.clear()
            self.ref_cache.clear()
            _GENERATOR_CACHE.clear()

            if self.transformer is not None:
                try:
                    self.transformer.unfuse_lora()
                    self.transformer.clear_caches()
                except Exception:
                    pass
                self.transformer = None

            if self.text_encoder is not None:
                try:
                    self.text_encoder.clear_caches()
                except Exception:
                    pass
                self.text_encoder = None

            self.vae = None
            self._loaded_repo_id = None
            self._current_offload_state = None
            self._warmed_up = False

            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    self._sync_device()
                except Exception:
                    pass

    def list_available_models(self) -> List[Dict[str, object]]:
        models_info = []
        for variant, repo in SUPPORTED_MODELS.items():
            try:
                snap = resolve_model_path(repo)
                manifest_file = snap / "conditioning_manifest.json"
                cond_meta = (
                    json.loads(manifest_file.read_text(encoding="utf-8"))
                    if manifest_file.is_file()
                    else {"precision": "native"}
                )
                models_info.append(
                    {
                        "variant": variant,
                        "repo_id": repo,
                        "cached": True,
                        "snapshot_path": str(snap),
                        "active": (self._loaded_repo_id == repo),
                        "conditioning": cond_meta,
                    }
                )
            except FileNotFoundError:
                models_info.append(
                    {
                        "variant": variant,
                        "repo_id": repo,
                        "cached": False,
                        "snapshot_path": None,
                        "active": False,
                        "conditioning": None,
                    }
                )
        return models_info

    def _init_or_swap_components(self, target_repo_id: str, cpu_offload: bool) -> None:
        with torch.inference_mode(False), torch.no_grad():
            if self.device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA execution requested but no compatible GPU detected."
                )

            target_device = torch.device("cpu") if cpu_offload else self.device
            if (
                self.transformer is not None
                and self.text_encoder is not None
                and self.vae is not None
            ):
                if self._current_offload_state != cpu_offload:
                    self.text_encoder.to(target_device)
                    self.vae.to(target_device)
                    if self._loaded_repo_id == target_repo_id:
                        self.transformer.to(target_device)
                        self.transformer.clear_caches()
                        self._current_offload_state = cpu_offload
                        self._trim_reserved_vram(force=True, required_headroom_gb=1.50)
                        self.memory_guard.recalibrate_baseline()
                        return
                    self._current_offload_state = cpu_offload
                elif self._loaded_repo_id == target_repo_id:
                    return

            root_path = resolve_model_path(target_repo_id)

            if (
                self.transformer is not None
                and self.text_encoder is not None
                and self.vae is not None
                and self._loaded_repo_id != target_repo_id
                and self._current_offload_state == cpu_offload
            ):
                t0 = time.perf_counter()
                logging.info(
                    "Executing DiT-only hot-swap: %s -> %s",
                    self._loaded_repo_id,
                    target_repo_id,
                )
                transformer_dir = root_path / "transformer"
                self.transformer.unfuse_lora()
                self.transformer.clear_caches()

                cfg_file = transformer_dir / "config.json"
                same_arch = True
                if cfg_file.is_file():
                    incoming_cfg = MOSSImageConfig.from_json_file(cfg_file)
                    same_arch = incoming_cfg == self.transformer.config

                streamed_inplace = False
                if same_arch:
                    streamed_inplace = stream_safetensors_inplace(
                        transformer_dir,
                        self.transformer,
                        device=target_device,
                        dtype=self.dtype,
                    )

                if not streamed_inplace:
                    del self.transformer
                    self.transformer = None
                    self._trim_reserved_vram(force=True, required_headroom_gb=11.0)
                    with _MetaWeightInitMode():
                        self.transformer = MOSSImageTransformer.from_config_dir(
                            transformer_dir
                        )
                    load_sharded_safetensors_into(
                        transformer_dir,
                        self.transformer,
                        device=target_device,
                        dtype=self.dtype,
                    )
                    self._trim_reserved_vram(force=False, required_headroom_gb=1.50)

                self.transformer.clear_caches()
                self.transformer.eval().requires_grad_(False)
                self._loaded_repo_id = target_repo_id
                self.memory_guard.recalibrate_baseline()
                logging.info(
                    "Hot-swap completed in %.2fs (Shared Qwen3.5 & Flux2-VAE retained).",
                    time.perf_counter() - t0,
                )
                return

            if self.transformer is not None:
                del self.transformer, self.text_encoder, self.vae
                self.transformer = None
                self.text_encoder = None
                self.vae = None
                self._trim_reserved_vram(force=True, required_headroom_gb=26.5)
            else:
                self._trim_reserved_vram(force=False, required_headroom_gb=26.5)
                self.memory_guard.recalibrate_baseline()

            t_start = time.perf_counter()
            tokenizer_dir = root_path / "tokenizer"
            te_dir = root_path / "text_encoder"
            transformer_dir = root_path / "transformer"
            vae_dir = root_path / "vae"

            logging.info(
                "Loading Sovereign VizToken Qwen3.5 (SWISS5 AVX2) tokenizer from %s",
                tokenizer_dir,
            )
            self.tokenizer = MOSSImageTokenizer(tokenizer_dir)

            logging.info(
                "Loading Sovereign Qwen3_5Model (24 Linear-Attn + 8 Full-Attn + 27-Layer ViT) from %s",
                te_dir,
            )
            with _MetaWeightInitMode():
                self.text_encoder = Qwen3_5Model.from_config_dir(te_dir)
            load_sharded_safetensors_into(
                te_dir, self.text_encoder, device=target_device, dtype=self.dtype
            )
            self.text_encoder.eval().requires_grad_(False)
            self._trim_reserved_vram(force=True, required_headroom_gb=11.0)

            logging.info(
                "Loading Sovereign MOSSImageTransformer (42-Layer DiT) from %s",
                transformer_dir,
            )
            with _MetaWeightInitMode():
                self.transformer = MOSSImageTransformer.from_config_dir(
                    transformer_dir
                )
            load_sharded_safetensors_into(
                transformer_dir,
                self.transformer,
                device=target_device,
                dtype=self.dtype,
            )
            self.transformer.eval().requires_grad_(False)
            self._trim_reserved_vram(force=True, required_headroom_gb=2.50)

            logging.info(
                "Loading Sovereign Flux2AutoencoderKL (FP32) from %s", vae_dir
            )
            with _MetaWeightInitMode():
                self.vae = Flux2AutoencoderKL.from_config_dir(vae_dir)
            load_sharded_safetensors_into(
                vae_dir, self.vae, device=target_device, dtype=self.vae_dtype
            )
            self.vae.eval().requires_grad_(False)

            self._loaded_repo_id = target_repo_id
            self._current_offload_state = cpu_offload
            self._trim_reserved_vram(force=True, required_headroom_gb=1.50)
            self.memory_guard.recalibrate_baseline()
            logging.info(
                "MOSSImageEngine initialized (%s) in %.2fs.",
                target_repo_id,
                time.perf_counter() - t_start,
            )

    def _apply_or_revert_edit_lora(
        self, edit_lora_path: Optional[str], edit_lora_scale: float
    ) -> bool:
        with torch.inference_mode(False), torch.no_grad():
            if self.transformer is None:
                return False
            if (
                not edit_lora_path
                or not edit_lora_path.strip()
                or abs(edit_lora_scale) < 1e-6
            ):
                if self.transformer._lora_backups:
                    self.transformer.unfuse_lora()
                    self._trim_reserved_vram(force=False, required_headroom_gb=1.25)
                return False

            raw_spec = edit_lora_path.strip()
            resolved_lora: Optional[Path] = None
            if raw_spec.lower() in ("auto", "moss_edit", "chinoll/moss-image-edit"):
                try:
                    resolved_lora = resolve_model_path("chinoll/Moss-Image-Edit")
                except FileNotFoundError:
                    resolved_lora = None
            else:
                p = Path(raw_spec)
                if not p.is_absolute():
                    p = (ROOT_DIR / p).resolve()
                if p.exists():
                    resolved_lora = p.resolve()
                else:
                    try:
                        resolved_lora = resolve_model_path(raw_spec)
                    except FileNotFoundError:
                        resolved_lora = None

            if resolved_lora is None:
                logging.warning(
                    "Requested Edit LoRA '%s' not found; skipping.", raw_spec
                )
                self.transformer.unfuse_lora()
                return False

            lora_tag_str = str(resolved_lora)
            if self.transformer.is_lora_active(lora_tag_str, edit_lora_scale):
                return True

            sf_files = (
                [resolved_lora]
                if resolved_lora.is_file()
                else sorted(resolved_lora.rglob("*.safetensors"))
            )
            if not sf_files:
                self.transformer.unfuse_lora()
                return False

            lora_sd: Dict[str, torch.Tensor] = {}
            for sf in sf_files:
                lora_sd.update(load_safetensors(str(sf), device="cpu"))
            fused = self.transformer.fuse_lora_state_dict(
                lora_sd, scale=edit_lora_scale, lora_tag=lora_tag_str
            )
            del lora_sd
            self._trim_reserved_vram(force=False, required_headroom_gb=1.25)
            return fused > 0

    def _condition_single_reference(
        self,
        image_path: Path,
        request: GenerationRequest,
        role_override: Optional[str] = None,
        framing_override: Optional[str] = None,
        target_w_override: Optional[int] = None,
        target_h_override: Optional[int] = None,
        force_vit: bool = False,
    ) -> Tuple[CachedReferenceRecord, bool]:
        eff_role = role_override or request.input_role
        eff_framing = framing_override or request.framing_mode
        eff_w = target_w_override if target_w_override is not None else request.width
        eff_h = target_h_override if target_h_override is not None else request.height
        eff_w = max(64, (int(eff_w) // 16) * 16)
        eff_h = max(64, (int(eff_h) // 16) * 16)

        cache_key = ReferenceVisionCache.make_key(
            path=image_path,
            target_w=eff_w,
            target_h=eff_h,
            framing_mode=eff_framing,
            input_role=eff_role,
            edit_format=request.edit_format,
            max_vit_patches=request.max_vit_patches,
            clamp_sigma=request.latent_clamp_sigma,
        )
        vae_key = ReferenceVisionCache.make_vae_key(
            path=image_path,
            target_w=eff_w,
            target_h=eff_h,
            framing_mode=eff_framing,
            input_role=eff_role,
            clamp_sigma=request.latent_clamp_sigma,
            mtf_boost=request.mtf_boost,
            mask_feather_radius=request.mask_feather_radius,
        )

        cached_vae_tuple: Optional[
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Any]]
        ] = None
        if request.use_reference_cache:
            cached = self.ref_cache.lookup(cache_key)
            if cached is not None:
                return cached, True
            cached_vae_tuple = self.ref_cache.lookup_vae(vae_key)

        max_patches = (
            min(request.max_vit_patches, 576)
            if request.edit_format in ("ai_toolkit_isolated", "ai_toolkit_full")
            else request.max_vit_patches
        )
        want_tier2_export = bool(
            request.export_vizj_path
            and int(request.export_vizj_tier) == VIZJ_TIER_DUAL_LATENT
        )
        need_vit = eff_role == "photo_ref" and (
            force_vit
            or request.edit_format != "moss_image_edit"
            or want_tier2_export
        )

        if (
            cached_vae_tuple is not None
            and not need_vit
            and cached_vae_tuple[0] is not None
        ):
            rec = CachedReferenceRecord(
                packed_latents=cached_vae_tuple[0],
                vit_embeds=None,
                vit_patches=None,
                grid_thw=None,
                llm_tokens=0,
                latent_mask=cached_vae_tuple[1],
                viz_meta=cached_vae_tuple[2],
                source_signature=vae_key[:6],
            )
            if request.use_reference_cache:
                rec = self.ref_cache.insert(cache_key, rec, vae_key=vae_key)
            return rec, True

        if self.vizinjector is not None and VizInjectorRequest is not None:
            have_cached_vae = (
                cached_vae_tuple is not None and cached_vae_tuple[0] is not None
            )
            viz_req = VizInjectorRequest(
                target_width=eff_w,
                target_height=eff_h,
                max_vit_patches=max_patches,
                framing_mode=_normalize_vizinjector_framing(eff_framing),
                input_role=eff_role,
                edit_format=request.edit_format,
                emit_vae_branch=False if have_cached_vae else None,
                emit_mask_branch=False if have_cached_vae else None,
                emit_vit_branch=True if need_vit else None,
                deband_8bit=request.deband_8bit,
                prefer_vulkan=request.prefer_vulkan,
                seed=int(request.seed or 42),
                mtf_boost=request.mtf_boost,
                soft_knee_tau=request.soft_knee_tau,
                mask_feather_radius=request.mask_feather_radius,
                latent_clamp_sigma=request.latent_clamp_sigma,
            )
            bundle = self.vizinjector.condition_image(image_path, viz_req)

            if have_cached_vae:
                packed_lat = cached_vae_tuple[0]
                latent_mask = cached_vae_tuple[1]
                cached_meta = cached_vae_tuple[2]
                if bundle.meta is not None:
                    if packed_lat is not None and packed_lat.ndim == 4:
                        bundle.meta.vae_size = (
                            int(packed_lat.shape[3]) * 16,
                            int(packed_lat.shape[2]) * 16,
                        )
                    if (
                        cached_meta is not None
                        and getattr(cached_meta, "telemetry", None) is not None
                    ):
                        bundle.meta.telemetry = cached_meta.telemetry
            else:
                packed_lat = _evacuate_to_cpu(bundle.packed_latents)
                if packed_lat is None and bundle.vae_tensor is not None:
                    raw_lat = self.vae.encode_to_packed_latents(bundle.vae_tensor)
                    clamped_lat = subphase_radial_clamp(
                        raw_lat, max_sigma=request.latent_clamp_sigma
                    )
                    packed_lat = _evacuate_to_cpu(clamped_lat)
                    del raw_lat, clamped_lat
                    bundle.vae_tensor = None
                    self._trim_reserved_vram(force=False, required_headroom_gb=1.25)
                latent_mask = _evacuate_to_cpu(bundle.latent_mask)

            vit_embeds = _evacuate_to_cpu(bundle.vit_embeds)
            if (
                vit_embeds is None
                and need_vit
                and bundle.vit_patches is not None
                and bundle.grid_thw is not None
            ):
                img_splits = self.text_encoder.get_image_features(
                    bundle.vit_patches,
                    bundle.grid_thw.to(
                        device=self.text_encoder.visual.device, non_blocking=True
                    ),
                )
                vit_embeds = _evacuate_to_cpu(torch.cat(img_splits, dim=0))
                del img_splits
                self.text_encoder.visual.clear_caches()
                self._trim_reserved_vram(force=False, required_headroom_gb=1.25)

            rec = CachedReferenceRecord(
                packed_latents=packed_lat,
                vit_embeds=vit_embeds,
                vit_patches=_evacuate_to_cpu(bundle.vit_patches),
                grid_thw=_evacuate_to_cpu(bundle.grid_thw),
                llm_tokens=int(bundle.llm_tokens) if need_vit else 0,
                latent_mask=latent_mask,
                viz_meta=bundle.meta,
                source_signature=vae_key[:6],
            )
            was_hit = have_cached_vae
        else:
            if cached_vae_tuple is not None and cached_vae_tuple[0] is not None:
                packed_lat = cached_vae_tuple[0]
                was_hit = True
            else:
                img_t = load_image_tensor_16px(image_path, eff_h, eff_w)
                raw_lat = self.vae.encode_to_packed_latents(img_t)
                clamped_lat = subphase_radial_clamp(
                    raw_lat, max_sigma=request.latent_clamp_sigma
                )
                packed_lat = _evacuate_to_cpu(clamped_lat)
                del img_t, raw_lat, clamped_lat
                self._trim_reserved_vram(force=False, required_headroom_gb=1.25)
                was_hit = False

            vit_embeds = None
            vit_patches = None
            grid_thw = None
            llm_tokens = 0
            if need_vit:
                vit_patches, grid_thw, llm_tokens = pack_vision_patches_32px(
                    image_path, max_patches=max_patches
                )
                img_splits = self.text_encoder.get_image_features(
                    vit_patches,
                    grid_thw.to(
                        device=self.text_encoder.visual.device, non_blocking=True
                    ),
                )
                vit_embeds = _evacuate_to_cpu(torch.cat(img_splits, dim=0))
                del img_splits
                self.text_encoder.visual.clear_caches()
                self._trim_reserved_vram(force=False, required_headroom_gb=1.25)
            rec = CachedReferenceRecord(
                packed_latents=packed_lat,
                vit_embeds=vit_embeds,
                vit_patches=vit_patches,
                grid_thw=grid_thw,
                llm_tokens=llm_tokens,
                latent_mask=None,
                viz_meta=None,
                source_signature=vae_key[:6],
            )

        if request.use_reference_cache:
            rec = self.ref_cache.insert(cache_key, rec, vae_key=vae_key)
        return rec, was_hit

    @torch.inference_mode()
    def encode_prompt(
        self,
        prompt: str,
        max_sequence_length: int = 4096,
        enable_thinking: bool = True,
        device: Optional[torch.device] = None,
        use_cache: bool = True,
        is_negative: bool = False,
        vision_reference_paths: Optional[Sequence[str]] = None,
        prebuilt_vision_records: Optional[Sequence[CachedReferenceRecord]] = None,
        system_prompt: Optional[str] = None,
        thinking_trace: Optional[str] = None,
        add_vision_id: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        te_device = next(self.text_encoder.parameters()).device
        out_device = _canonicalize_device(device) if device is not None else te_device

        pixel_values: Optional[torch.Tensor] = None
        image_grid_thw: Optional[torch.Tensor] = None
        patch_counts: Optional[List[int]] = None
        pre_vit_embeds: Optional[torch.Tensor] = None
        vision_sig_list: List[Any] = []

        if prebuilt_vision_records:
            valid_recs = [
                r
                for r in prebuilt_vision_records
                if r.llm_tokens > 0 and r.grid_thw is not None
            ]
            if valid_recs:
                patch_counts = [int(r.llm_tokens) for r in valid_recs]
                for r in valid_recs:
                    vision_sig_list.append(
                        (r.source_signature, int(r.llm_tokens))
                    )
                image_grid_thw = torch.cat(
                    [r.grid_thw for r in valid_recs], dim=0
                ).to(device=te_device, non_blocking=True)
                if all(r.vit_embeds is not None for r in valid_recs):
                    pre_vit_embeds = torch.cat(
                        [r.vit_embeds for r in valid_recs], dim=0
                    ).to(device=te_device, non_blocking=True)
                elif all(r.vit_patches is not None for r in valid_recs):
                    pixel_values = torch.cat(
                        [r.vit_patches for r in valid_recs], dim=0
                    ).to(device=te_device, non_blocking=True)
                else:
                    resolved_embeds: List[torch.Tensor] = []
                    for r in valid_recs:
                        if r.vit_embeds is not None:
                            resolved_embeds.append(
                                r.vit_embeds.to(device=te_device, non_blocking=True)
                            )
                        elif r.vit_patches is not None and r.grid_thw is not None:
                            splits = self.text_encoder.get_image_features(
                                r.vit_patches,
                                r.grid_thw.to(
                                    device=self.text_encoder.visual.device,
                                    non_blocking=True,
                                ),
                            )
                            resolved_embeds.append(
                                torch.cat(splits, dim=0).to(
                                    device=te_device, non_blocking=True
                                )
                            )
                            del splits
                    if resolved_embeds:
                        pre_vit_embeds = torch.cat(resolved_embeds, dim=0)
                        del resolved_embeds
        elif vision_reference_paths:
            patch_tensors: List[torch.Tensor] = []
            grid_tensors: List[torch.Tensor] = []
            ordered_items: List[
                Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]
            ] = []
            patch_counts = []
            for p_str in vision_reference_paths:
                p = Path(p_str)
                if p.is_file():
                    st = p.stat()
                    if self.vizinjector is not None and VizInjectorRequest is not None:
                        v_req = VizInjectorRequest(
                            emit_vae_branch=False,
                            emit_mask_branch=False,
                            emit_vit_branch=True,
                            prefer_vulkan=True,
                        )
                        v_bundle = self.vizinjector.condition_image(p, v_req)
                        if v_bundle.grid_thw is not None and v_bundle.llm_tokens > 0:
                            n_llm = int(v_bundle.llm_tokens)
                            grid_tensors.append(v_bundle.grid_thw)
                            patch_counts.append(n_llm)
                            ordered_items.append(
                                (
                                    v_bundle.vit_embeds,
                                    v_bundle.vit_patches,
                                    v_bundle.grid_thw,
                                )
                            )
                            if v_bundle.vit_patches is not None:
                                patch_tensors.append(v_bundle.vit_patches)
                            vision_sig_list.append(
                                (str(p.resolve()), int(st.st_mtime_ns), int(n_llm))
                            )
                            continue
                    patches, g_thw, n_llm = pack_vision_patches_32px(p)
                    patch_tensors.append(patches)
                    grid_tensors.append(g_thw)
                    patch_counts.append(n_llm)
                    ordered_items.append((None, patches, g_thw))
                    vision_sig_list.append(
                        (str(p.resolve()), int(st.st_mtime_ns), int(n_llm))
                    )
            if grid_tensors:
                image_grid_thw = torch.cat(grid_tensors, dim=0).to(
                    device=te_device, non_blocking=True
                )
                if all(it[0] is not None for it in ordered_items):
                    pre_vit_embeds = torch.cat(
                        [it[0] for it in ordered_items if it[0] is not None], dim=0
                    ).to(device=te_device, non_blocking=True)
                elif len(patch_tensors) == len(grid_tensors):
                    pixel_values = torch.cat(patch_tensors, dim=0).to(
                        device=te_device, non_blocking=True
                    )
                else:
                    resolved_embeds = []
                    for emb_t, pat_t, g_t in ordered_items:
                        if emb_t is not None:
                            resolved_embeds.append(
                                emb_t.to(device=te_device, non_blocking=True)
                            )
                        elif pat_t is not None:
                            splits = self.text_encoder.get_image_features(
                                pat_t,
                                g_t.to(
                                    device=self.text_encoder.visual.device,
                                    non_blocking=True,
                                ),
                            )
                            resolved_embeds.append(
                                torch.cat(splits, dim=0).to(
                                    device=te_device, non_blocking=True
                                )
                            )
                            del splits
                    if resolved_embeds:
                        pre_vit_embeds = torch.cat(resolved_embeds, dim=0)
                        del resolved_embeds

        tok_list = self.tokenizer.encode_chat(
            prompt=prompt,
            enable_thinking=enable_thinking,
            max_sequence_length=max_sequence_length,
            image_patch_counts=patch_counts,
            system_prompt=system_prompt if not is_negative else None,
            thinking_trace=thinking_trace if not is_negative else None,
            add_vision_id=add_vision_id,
        )
        vision_signature = tuple(vision_sig_list)

        if use_cache:
            cached = self.embed_cache.lookup(
                tok_list,
                out_device,
                is_negative=is_negative,
                vision_signature=vision_signature,
            )
            if cached is not None:
                del pixel_values, image_grid_thw, pre_vit_embeds
                mask = torch.ones(
                    (1, cached.shape[1]), dtype=torch.bool, device=out_device
                )
                return cached, mask

        input_ids = torch.tensor([tok_list], dtype=torch.long, device=te_device)
        attention_mask = torch.ones_like(input_ids, device=te_device)
        inputs_embeds: Optional[torch.Tensor] = None

        if pre_vit_embeds is not None and image_grid_thw is not None:
            base_embeds = self.text_encoder.get_input_embeddings()(input_ids)
            image_mask = (
                input_ids == self.text_encoder.config.image_token_id
            ).unsqueeze(-1)
            inputs_embeds = base_embeds.masked_scatter(
                image_mask,
                pre_vit_embeds.to(
                    device=base_embeds.device,
                    dtype=base_embeds.dtype,
                    non_blocking=True,
                ),
            )
            del base_embeds, image_mask, pre_vit_embeds

        output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        hidden_cpu = _evacuate_to_cpu(output.last_hidden_state)
        del (
            output,
            input_ids,
            attention_mask,
            inputs_embeds,
            pixel_values,
            image_grid_thw,
        )

        if use_cache and hidden_cpu is not None:
            self.embed_cache.insert(
                tok_list, hidden_cpu, vision_signature=vision_signature
            )

        hidden_out = hidden_cpu.to(device=out_device, non_blocking=True)
        mask_out = torch.ones(
            (1, hidden_out.shape[1]), dtype=torch.bool, device=out_device
        )
        return hidden_out, mask_out

    @staticmethod
    def postprocess_to_pil(images: torch.Tensor) -> List[Image.Image]:
        imgs = (images.float() * 0.5 + 0.5).clamp(0.0, 1.0)
        imgs = (
            imgs.permute(0, 2, 3, 1)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .detach()
            .cpu()
            .numpy()
        )
        return [Image.fromarray(arr) for arr in imgs]

    @torch.inference_mode()
    def warmup(self, height: int = 1024, width: int = 1024) -> float:
        self._init_or_swap_components(self.repo_id, cpu_offload=False)
        t0 = time.perf_counter()
        exec_dev = self.device

        sys.stdout.write(
            "  [1/3] Warmup: Qwen3.5 Hybrid Gated-DeltaNet & Full-Attn Prefill...\n"
        )
        sys.stdout.flush()
        dummy_ids = torch.tensor(
            [[248045, 872, 198, 100, 248044, 198, 248045, 77091, 198]],
            dtype=torch.long,
            device=exec_dev,
        )
        dummy_out = self.text_encoder(
            input_ids=dummy_ids,
            attention_mask=torch.ones_like(dummy_ids, device=exec_dev),
        )
        dummy_ctx_cpu = _evacuate_to_cpu(dummy_out.last_hidden_state)
        del dummy_ids, dummy_out
        self.text_encoder.clear_caches()
        self._sync_device()
        self._trim_reserved_vram(force=False, required_headroom_gb=1.50)

        sys.stdout.write(
            f"  [2/3] Warmup: 42-Layer MOSS-DiT 4D-RoPE Velocity Field ({width}x{height})...\n"
        )
        sys.stdout.flush()
        h_lat, w_lat = height // 16, width // 16
        dummy_lat = torch.zeros(
            (1, self.transformer.config.latent_channels, h_lat, w_lat),
            device=exec_dev,
            dtype=torch.float32,
        )
        dummy_ctx = dummy_ctx_cpu.to(device=exec_dev, non_blocking=True)
        dummy_mask = torch.ones(
            (1, dummy_ctx.shape[1]),
            device=exec_dev,
            dtype=torch.bool,
        )
        warmup_skip = set(range(1, len(self.transformer.layers)))
        _ = self.transformer(
            dummy_lat,
            torch.tensor([500.0], device=exec_dev),
            dummy_ctx,
            dummy_mask,
            skip_layers=warmup_skip,
        )
        del dummy_ctx_cpu, dummy_ctx, dummy_mask, dummy_lat
        self.transformer.clear_transient_caches()
        self._sync_device()
        self._trim_reserved_vram(force=False, required_headroom_gb=1.50)

        sys.stdout.write(
            f"  [3/3] Warmup: Flux2-VAE FP32 Unpatch & Decoder ({width}x{height})...\n"
        )
        sys.stdout.flush()
        dummy_lat_vae = torch.zeros(
            (1, self.transformer.config.latent_channels, h_lat, w_lat),
            device=exec_dev,
            dtype=torch.float32,
        )
        dummy_dec = self.vae.decode_packed_latents(dummy_lat_vae)
        del dummy_lat_vae, dummy_dec
        self._sync_device()
        self._trim_reserved_vram(force=True, required_headroom_gb=1.50)
        self.memory_guard.recalibrate_baseline()

        elapsed = time.perf_counter() - t0
        self._warmed_up = True
        sys.stdout.write(
            f"  [+] MOSSImageEngine Warmup Completed in {elapsed:.2f}s.\n"
        )
        sys.stdout.flush()
        return elapsed

    @torch.inference_mode()
    def synthesize(
        self,
        request: GenerationRequest,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> GenerationResponse:
        request.validate()
        target_repo = request.resolve_repo_id()
        cpu_offload = bool(request.cpu_offload)
        self._init_or_swap_components(target_repo, cpu_offload=cpu_offload)

        if request.prepass_warmup and not self._warmed_up and not cpu_offload:
            self.warmup(height=request.height, width=request.width)

        lora_active = self._apply_or_revert_edit_lora(
            request.edit_lora_path, request.edit_lora_scale
        )
        if not cpu_offload:
            self._trim_reserved_vram(force=False, required_headroom_gb=1.25)

        reporter: Optional[EngineProgressReporter] = None
        if progress_callback is None:
            reporter = EngineProgressReporter()
            effective_cb = reporter
        else:
            effective_cb = progress_callback

        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        start_time = time.perf_counter()
        explicit_seed = (
            request.seed
            if (request.seed is not None and request.seed >= 0)
            else random.randint(100000, 99999999)
        )
        random.seed(explicit_seed)
        np.random.seed(explicit_seed % (2**32))
        torch.manual_seed(explicit_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(explicit_seed)

        init_latents_cpu: Optional[torch.Tensor] = None
        inpaint_mask_cpu: Optional[torch.Tensor] = None
        ref_latents_cpu_list: List[torch.Tensor] = []
        ref_bboxes_list: List[Optional[Tuple[float, float, float, float]]] = []
        ref_box_iso_list: List[float] = []
        splat_entries: List[
            Tuple[
                torch.Tensor,
                Tuple[float, float, float, float],
                Optional[torch.Tensor],
                Optional[float],
            ]
        ] = []
        ref_records_for_prompt: List[CachedReferenceRecord] = []
        any_ref_cache_hit = False
        exported_vizj_str: Optional[str] = None

        has_init = bool(
            request.init_image_path and Path(request.init_image_path).is_file()
        )
        has_control = bool(
            request.control_image_path and Path(request.control_image_path).is_file()
        )
        visual_ref_specs, _ = request.resolve_visual_references()
        valid_ref_specs = [
            spec for spec in visual_ref_specs if Path(spec["path"]).is_file()
        ]

        if has_init or has_control or valid_ref_specs:
            if cpu_offload:
                self.vae.to(self.device)
                self.text_encoder.to(self.device)

            if has_init:
                init_p = Path(request.init_image_path)
                init_rec, hit = self._condition_single_reference(
                    init_p,
                    request,
                    role_override="photo_ref",
                    target_w_override=request.width,
                    target_h_override=request.height,
                )
                any_ref_cache_hit = any_ref_cache_hit or hit
                init_latents_cpu = _evacuate_to_cpu(init_rec.packed_latents)

                if (
                    request.inpaint_mask_path
                    and Path(request.inpaint_mask_path).is_file()
                ):
                    mask_p = Path(request.inpaint_mask_path)
                    if self.vizinjector is not None and VizInjectorRequest is not None:
                        m_req = VizInjectorRequest(
                            target_width=request.width,
                            target_height=request.height,
                            framing_mode=_normalize_vizinjector_framing(
                                request.framing_mode
                            ),
                            input_role="inpaint_mask",
                            mask_feather_radius=request.mask_feather_radius,
                            prefer_vulkan=request.prefer_vulkan,
                        )
                        m_bundle = self.vizinjector.condition_image(mask_p, m_req)
                        if m_bundle.latent_mask is not None:
                            inpaint_mask_cpu = _evacuate_to_cpu(m_bundle.latent_mask)
                    if inpaint_mask_cpu is None:
                        m_img = Image.open(mask_p).convert("L")
                        m_img = m_img.resize(
                            (request.width // 16, request.height // 16),
                            resample=Image.Resampling.BILINEAR,
                        )
                        m_arr = np.asarray(m_img, dtype=np.float32) / 255.0
                        inpaint_mask_cpu = (
                            torch.from_numpy(m_arr).unsqueeze(0).unsqueeze(0).contiguous()
                        )
                elif request.layout_items and (
                    bool(request.preserve_canvas_outside_boxes)
                    or any(
                        isinstance(it, dict)
                        and (
                            bool(it.get("inpaint_box", False))
                            or bool(it.get("paste_to_canvas", False))
                        )
                        for it in request.layout_items
                    )
                ):
                    only_flagged = not bool(request.preserve_canvas_outside_boxes)
                    inpaint_mask_cpu = build_box_aperture_mask(
                        layout_items=request.layout_items,
                        height=request.height,
                        width=request.width,
                        coords_are_pixels=request.layout_coords_are_pixels,
                        feather_radius_lat=float(request.mask_feather_radius or 1.25),
                        device=torch.device("cpu"),
                        only_flagged_boxes=only_flagged,
                    )
                elif init_rec.latent_mask is not None:
                    if float(torch.min(init_rec.latent_mask).item()) < 0.99:
                        inpaint_mask_cpu = _evacuate_to_cpu(init_rec.latent_mask)

            if has_control:
                ctrl_p = Path(request.control_image_path)
                ctrl_role = (
                    request.input_role
                    if request.input_role in ("color_map", "depth_map")
                    else "color_map"
                )
                ctrl_w, ctrl_h = _resolve_unboxed_reference_dims(
                    ctrl_p,
                    request.width,
                    request.height,
                    request.edit_format,
                )
                ctrl_rec, hit = self._condition_single_reference(
                    ctrl_p,
                    request,
                    role_override=ctrl_role,
                    target_w_override=ctrl_w,
                    target_h_override=ctrl_h,
                )
                any_ref_cache_hit = any_ref_cache_hit or hit
                if ctrl_rec.packed_latents is not None:
                    ref_latents_cpu_list.append(
                        _evacuate_to_cpu(ctrl_rec.packed_latents)
                    )
                    ref_bboxes_list.append(None)
                    ref_box_iso_list.append(0.0)

            for spec in valid_ref_specs:
                r_path = Path(spec["path"])
                px_box = spec.get("pixel_bbox")
                norm_box = spec.get("norm_bbox")
                is_box_ref = norm_box is not None
                emits_vit = bool(
                    spec.get(
                        "emits_vit",
                        spec.get("input_role", request.input_role) == "photo_ref"
                        and (is_box_ref or request.edit_format != "moss_image_edit"),
                    )
                )

                if px_box is not None:
                    ymin, xmin, ymax, xmax = px_box
                    ref_w = max(64, ((xmax - xmin + 8) // 16) * 16)
                    ref_h = max(64, ((ymax - ymin + 8) // 16) * 16)
                else:
                    ref_w, ref_h = _resolve_unboxed_reference_dims(
                        r_path,
                        request.width,
                        request.height,
                        request.edit_format,
                    )

                r_rec, hit = self._condition_single_reference(
                    r_path,
                    request,
                    role_override=spec.get("input_role", request.input_role),
                    framing_override=spec.get("framing_mode", request.framing_mode),
                    target_w_override=ref_w,
                    target_h_override=ref_h,
                    force_vit=emits_vit,
                )
                any_ref_cache_hit = any_ref_cache_hit or hit
                if r_rec.packed_latents is not None:
                    lat_cpu = _evacuate_to_cpu(r_rec.packed_latents)
                    ref_latents_cpu_list.append(lat_cpu)
                    ref_bboxes_list.append(norm_box)
                    ref_box_iso_list.append(float(spec.get("isolate_box_attn") or 0.0))
                    if bool(spec.get("paste_to_canvas", False)) and norm_box is not None:
                        splat_entries.append(
                            (
                                lat_cpu,
                                norm_box,
                                _evacuate_to_cpu(r_rec.latent_mask),
                                spec.get("denoise_strength"),
                            )
                        )
                if r_rec.llm_tokens > 0 and emits_vit:
                    ref_records_for_prompt.append(r_rec)

                if (
                    request.export_vizj_path
                    and self.vizinjector is not None
                    and r_rec.viz_meta is not None
                ):
                    b_exp = ConditionedVisionBundle(
                        vae_tensor=None,
                        vit_patches=r_rec.vit_patches,
                        grid_thw=r_rec.grid_thw,
                        llm_tokens=int(r_rec.llm_tokens),
                        latent_mask=r_rec.latent_mask,
                        packed_latents=r_rec.packed_latents,
                        vit_embeds=r_rec.vit_embeds,
                        wire_vizj_bytes=None,
                        meta=r_rec.viz_meta,
                    )
                    exp_dest = Path(request.export_vizj_path)
                    if not exp_dest.is_absolute():
                        exp_dest = ROOT_DIR / exp_dest
                    self.vizinjector.save_vizj(
                        b_exp,
                        dest_path=exp_dest,
                        tier=int(request.export_vizj_tier),
                        use_bf16=True,
                    )
                    exported_vizj_str = str(exp_dest)

            if cpu_offload:
                self.vae.to("cpu")
                self._trim_reserved_vram(force=True, required_headroom_gb=4.20)
            else:
                self._trim_reserved_vram(force=False, required_headroom_gb=1.25)

        effective_prompt = request.compile_effective_prompt()
        pos_hits_before = self.embed_cache.pos_hits
        early_cfg = float(request.early_cfg)
        late_cfg = float(request.late_cfg)
        any_cfg = (abs(early_cfg - 1.0) > 1e-5) or (abs(late_cfg - 1.0) > 1e-5)
        ctx_total = 2 if any_cfg else 1

        if cpu_offload:
            self.text_encoder.to(self.device)
        if reporter is not None:
            reporter.start_stage("context", ctx_total)

        pos_embeds, pos_mask = self.encode_prompt(
            effective_prompt,
            max_sequence_length=request.max_sequence_length,
            enable_thinking=request.enable_thinking,
            device=torch.device("cpu"),
            is_negative=False,
            prebuilt_vision_records=ref_records_for_prompt,
            system_prompt=request.system_prompt,
            thinking_trace=request.thinking_trace,
            add_vision_id=bool(
                request.add_vision_id if request.add_vision_id is not None else True
            ),
        )
        cache_hit = self.embed_cache.pos_hits > pos_hits_before
        common_prefix_len = self.embed_cache.last_common_prefix_len
        self._sync_device()
        effective_cb("context", 1, ctx_total)

        neg_embeds, neg_mask = None, None
        if any_cfg:
            neg_embeds, neg_mask = self.encode_prompt(
                request.negative_prompt or "",
                max_sequence_length=request.max_sequence_length,
                enable_thinking=request.enable_thinking,
                device=torch.device("cpu"),
                is_negative=True,
                add_vision_id=bool(
                    request.add_vision_id if request.add_vision_id is not None else True
                ),
            )
            self._sync_device()
            effective_cb("context", 2, ctx_total)

        del ref_records_for_prompt
        self.text_encoder.clear_caches()
        if cpu_offload:
            self.text_encoder.to("cpu")
            self._trim_reserved_vram(force=True, required_headroom_gb=11.0)
        else:
            self._trim_reserved_vram(force=False, required_headroom_gb=1.25)

        if cpu_offload:
            self.transformer.to(self.device)

        depth = len(self.transformer.layers)
        uncond_skip_set = validate_skip_layers(
            request.uncond_skip_layers, depth, "uncond_skip_layers"
        )
        slg_skip_set = validate_skip_layers(request.slg_layers, depth, "slg_layers")

        num_imgs = request.num_images_per_prompt
        pos_dev = pos_embeds.to(device=self.device, non_blocking=True)
        pos_mask_dev = pos_mask.to(device=self.device, non_blocking=True)
        neg_dev = (
            neg_embeds.to(device=self.device, non_blocking=True)
            if neg_embeds is not None
            else None
        )
        neg_mask_dev = (
            neg_mask.to(device=self.device, non_blocking=True)
            if neg_mask is not None
            else None
        )
        del pos_embeds, pos_mask, neg_embeds, neg_mask

        if num_imgs == 1:
            positive = pos_dev
            positive_mask = pos_mask_dev
            negative = neg_dev
            negative_mask = neg_mask_dev
        else:
            positive = pos_dev.repeat_interleave(num_imgs, dim=0)
            positive_mask = pos_mask_dev.repeat_interleave(num_imgs, dim=0)
            negative = (
                neg_dev.repeat_interleave(num_imgs, dim=0)
                if neg_dev is not None
                else None
            )
            negative_mask = (
                neg_mask_dev.repeat_interleave(num_imgs, dim=0)
                if neg_mask_dev is not None
                else None
            )
            del pos_dev, pos_mask_dev, neg_dev, neg_mask_dev

        ref_latents_list: List[torch.Tensor] = [
            r_cpu.to(device=self.device, dtype=torch.float32, non_blocking=True)
            for r_cpu in ref_latents_cpu_list
            if r_cpu is not None
        ]
        del ref_latents_cpu_list

        h_lat = request.height // 16
        w_lat = request.width // 16
        shape = (
            num_imgs,
            self.transformer.config.latent_channels,
            h_lat,
            w_lat,
        )
        k_init, k_sde, _ = derive_stage_keys(explicit_seed)
        noise = philox_randn(
            shape=shape,
            seed=k_init,
            stream_id=0x01,
            offset=0,
            device=self.device,
            dtype=torch.float32,
        )

        clean_init_latents: Optional[torch.Tensor] = None
        if init_latents_cpu is not None:
            clean_init_latents = init_latents_cpu.to(
                device=self.device, dtype=torch.float32, non_blocking=True
            )
            if clean_init_latents.shape[-2:] != (h_lat, w_lat):
                clean_init_latents = F.interpolate(
                    clean_init_latents,
                    size=(h_lat, w_lat),
                    mode="bilinear",
                    align_corners=False,
                )
        del init_latents_cpu

        composite_used = False
        splat_union_alpha: Optional[torch.Tensor] = None
        eff_denoise_strength = float(request.denoise_strength)
        if splat_entries:
            composite_used = True
            clean_init_latents, splat_union_alpha = splat_box_latents_to_canvas(
                base_canvas=clean_init_latents,
                splat_entries=splat_entries,
                h_lat=h_lat,
                w_lat=w_lat,
                latent_channels=self.transformer.config.latent_channels,
                feather_radius_lat=float(request.mask_feather_radius or 1.25),
                device=self.device,
            )
            box_dns = [
                float(e[3]) for e in splat_entries if e[3] is not None
            ]
            if box_dns and eff_denoise_strength >= 0.999:
                eff_denoise_strength = max(0.05, min(1.0, sum(box_dns) / len(box_dns)))
            elif not has_init and eff_denoise_strength >= 0.999:
                eff_denoise_strength = 0.78
        del splat_entries

        inpaint_mask_tensor: Optional[torch.Tensor] = None
        if inpaint_mask_cpu is not None:
            inpaint_mask_tensor = inpaint_mask_cpu.to(
                device=self.device, dtype=torch.float32, non_blocking=True
            )
            if inpaint_mask_tensor.shape[-2:] != (h_lat, w_lat):
                inpaint_mask_tensor = F.interpolate(
                    inpaint_mask_tensor,
                    size=(h_lat, w_lat),
                    mode="bilinear",
                    align_corners=False,
                )
        del inpaint_mask_cpu

        start_step = 0
        if clean_init_latents is not None and 0.0 < eff_denoise_strength < 1.0:
            start_step = max(
                0,
                min(
                    request.num_inference_steps - 1,
                    int(
                        round(
                            request.num_inference_steps
                            * (1.0 - eff_denoise_strength)
                        )
                    ),
                ),
            )

        use_bifurcated = (
            request.early_solver != "euler"
            or request.late_solver != "euler"
            or abs(early_cfg - late_cfg) > 1e-5
            or request.eta > 0.0
            or not request.use_v2_grid
            or start_step > 0
        )

        if use_bifurcated:
            active_scheduler: Union[
                MOSSImageScheduler, BifurcatedFlowMatchScheduler
            ] = BifurcatedFlowMatchScheduler(
                early_solver=request.early_solver,
                late_solver=request.late_solver,
                handoff_threshold=request.handoff_threshold,
                shift=request.shift,
                eta=request.eta,
                s_noise=request.s_noise,
                use_v2_grid=request.use_v2_grid,
            )
            active_scheduler.set_timesteps(
                num_inference_steps=request.num_inference_steps,
                device=self.device,
                start_step=start_step,
            )
        else:
            active_scheduler = MOSSImageScheduler(shift=request.shift)
            active_scheduler.set_timesteps(
                num_inference_steps=request.num_inference_steps,
                device=self.device,
            )

        if clean_init_latents is not None and start_step > 0:
            sigma_0 = float(active_scheduler.sigmas[0].item())
            noised_init = (1.0 - sigma_0) * clean_init_latents + sigma_0 * noise
            if not has_init and splat_union_alpha is not None:
                latents = (
                    splat_union_alpha * noised_init
                    + (1.0 - splat_union_alpha) * noise
                )
            else:
                latents = noised_init
            del noised_init
        else:
            latents = noise
        del splat_union_alpha

        sde_gen = torch.Generator(device=self.device).manual_seed(k_sde & _MASK63)
        timesteps = active_scheduler.timesteps
        num_passes = len(timesteps)
        total_nfe = 0

        active_ref_latents = ref_latents_list if ref_latents_list else None
        active_ref_bboxes = (
            ref_bboxes_list
            if (ref_bboxes_list and any(b is not None for b in ref_bboxes_list))
            else None
        )
        active_ref_box_iso = (
            ref_box_iso_list
            if (
                active_ref_bboxes is not None
                and any(v > 1e-6 for v in ref_box_iso_list)
            )
            else None
        )
        uncond_uses_refs = request.effective_uncond_uses_references
        uncond_ref_latents = active_ref_latents if uncond_uses_refs else None
        uncond_ref_bboxes = active_ref_bboxes if uncond_uses_refs else None
        uncond_ref_box_iso = active_ref_box_iso if uncond_uses_refs else None
        ref_zero_t = request.effective_reference_zero_timestep
        ref_iso_kv = request.effective_reference_isolated_kv
        box_feather_lat = float(request.mask_feather_radius or 1.25)

        use_cuda_events = self.device.type == "cuda" and torch.cuda.is_available()
        step_events = (
            [torch.cuda.Event(), torch.cuda.Event()] if use_cuda_events else []
        )

        if reporter is not None:
            reporter.start_stage("dit", num_passes)

        dit_t0 = time.perf_counter()
        for pass_index, timestep in enumerate(timesteps):
            if isinstance(active_scheduler, BifurcatedFlowMatchScheduler):
                is_early = active_scheduler.current_is_early
                step_frac = active_scheduler.current_progress
            else:
                is_early = True
                step_frac = 1.0 - float(
                    active_scheduler.sigmas[min(pass_index, num_passes - 1)].item()
                )

            current_cfg = early_cfg if is_early else late_cfg
            do_step_cfg = abs(current_cfg - 1.0) > 1e-5
            skip_active = request.slg_start <= step_frac < request.slg_stop
            do_slg = bool(slg_skip_set and request.slg_scale > 0.0 and skip_active)

            prediction = self.transformer(
                latents,
                timestep,
                positive,
                positive_mask,
                reference_latents=active_ref_latents,
                reference_zero_timestep=ref_zero_t,
                reference_isolated_kv=ref_iso_kv,
                reference_bboxes=active_ref_bboxes,
                reference_box_isolation=active_ref_box_iso,
                box_feather_radius=box_feather_lat,
            ).sample.float()
            conditional = prediction if do_slg else None
            total_nfe += 1

            if do_step_cfg and negative is not None:
                skip_kwargs = (
                    {"skip_layers": uncond_skip_set}
                    if (uncond_skip_set and skip_active)
                    else {}
                )
                uncond = self.transformer(
                    latents,
                    timestep,
                    negative,
                    negative_mask,
                    reference_latents=uncond_ref_latents,
                    reference_zero_timestep=ref_zero_t and uncond_uses_refs,
                    reference_isolated_kv=ref_iso_kv and uncond_uses_refs,
                    reference_bboxes=uncond_ref_bboxes,
                    reference_box_isolation=uncond_ref_box_iso,
                    box_feather_radius=box_feather_lat,
                    **skip_kwargs,
                ).sample.float()
                prediction = uncond + current_cfg * (prediction - uncond)
                del uncond
                total_nfe += 1

            if do_slg and conditional is not None:
                skipped = self.transformer(
                    latents,
                    timestep,
                    positive,
                    positive_mask,
                    skip_layers=slg_skip_set,
                    reference_latents=active_ref_latents,
                    reference_zero_timestep=ref_zero_t,
                    reference_isolated_kv=ref_iso_kv,
                    reference_bboxes=active_ref_bboxes,
                    reference_box_isolation=active_ref_box_iso,
                    box_feather_radius=box_feather_lat,
                ).sample.float()
                if not do_step_cfg:
                    prediction = skipped
                prediction = prediction + request.slg_scale * (conditional - skipped)
                del skipped
                total_nfe += 1
            del conditional

            latents = active_scheduler.step(
                prediction, timestep, latents, generator=sde_gen, return_dict=False
            )
            del prediction

            if inpaint_mask_tensor is not None and clean_init_latents is not None:
                next_sigma = (
                    float(
                        active_scheduler.sigmas[
                            min(pass_index + 1, len(active_scheduler.sigmas) - 1)
                        ].item()
                    )
                    if not isinstance(active_scheduler, BifurcatedFlowMatchScheduler)
                    else float(
                        active_scheduler._step_configs[
                            min(pass_index, num_passes - 1)
                        ]["s_next"]
                    )
                )
                noised_bg = (1.0 - next_sigma) * clean_init_latents + next_sigma * noise
                latents = (
                    inpaint_mask_tensor * latents
                    + (1.0 - inpaint_mask_tensor) * noised_bg
                )
                del noised_bg

            if use_cuda_events:
                curr_ev = step_events[pass_index & 1]
                curr_ev.record()
                if pass_index > 0:
                    step_events[(pass_index - 1) & 1].synchronize()
                    effective_cb("dit", pass_index, num_passes)
                if pass_index == num_passes - 1:
                    curr_ev.synchronize()
                    effective_cb("dit", num_passes, num_passes)
            else:
                effective_cb("dit", pass_index + 1, num_passes)

        dit_elapsed = time.perf_counter() - dit_t0
        if isinstance(active_scheduler, BifurcatedFlowMatchScheduler):
            early_steps = active_scheduler._early_steps
            late_steps = active_scheduler._late_steps
        else:
            early_steps = len(active_scheduler.timesteps)
            late_steps = 0

        max_box_iso_used = (
            max(ref_box_iso_list)
            if ref_box_iso_list
            else float(request.box_attn_isolation or 0.0)
        )

        latents_cpu = _evacuate_to_cpu(latents)
        self.transformer.clear_transient_caches()
        del (
            latents,
            timesteps,
            active_scheduler,
            step_events,
            positive,
            positive_mask,
            negative,
            negative_mask,
            active_ref_latents,
            active_ref_bboxes,
            active_ref_box_iso,
            uncond_ref_latents,
            uncond_ref_bboxes,
            uncond_ref_box_iso,
            ref_latents_list,
            ref_bboxes_list,
            ref_box_iso_list,
            clean_init_latents,
            inpaint_mask_tensor,
            noise,
        )

        if cpu_offload:
            self.transformer.to("cpu")
            self._trim_reserved_vram(force=True, required_headroom_gb=4.40)
            self.vae.to(self.device)
        else:
            self._trim_reserved_vram(force=False, required_headroom_gb=1.25)

        if reporter is not None:
            reporter.start_stage("vae", num_imgs)

        latents_vae = latents_cpu.to(device=self.device, non_blocking=True)
        del latents_cpu
        decoded_gpu = self.vae.decode_packed_latents(latents_vae)
        del latents_vae
        decoded_cpu = _evacuate_to_cpu(decoded_gpu)
        del decoded_gpu
        self._sync_device()

        if cpu_offload:
            self.vae.to("cpu")
            self._trim_reserved_vram(force=True, required_headroom_gb=4.20)
        else:
            self._trim_reserved_vram(force=False, required_headroom_gb=1.25)

        if self.vizinjector is not None:
            pil_images, viz_telems = self.vizinjector.postprocess_to_pil(
                decoded_cpu, apply_bayer_dither=request.apply_bayer_dither
            )
            mean_luma = viz_telems[0].mean_luminance
            rms_contrast = viz_telems[0].rms_contrast
            entropy = viz_telems[0].shannon_entropy_bits
            sharpness = viz_telems[0].laplacian_sharpness
        else:
            pil_images = self.postprocess_to_pil(decoded_cpu)
            mean_luma, rms_contrast, entropy, sharpness = compute_image_telemetry(
                pil_images[0]
            )
        del decoded_cpu

        effective_cb("vae", len(pil_images), len(pil_images))
        if reporter is not None:
            reporter.close()

        elapsed = time.perf_counter() - start_time
        peak_vram_gb = 0.0
        if self.device.type == "cuda" and torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)

        out_path = Path(request.output_path)
        if not out_path.is_absolute():
            out_path = ROOT_DIR / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pil_images[0].save(str(out_path))

        extra_paths: List[str] = []
        for idx, extra_img in enumerate(pil_images[1:], start=2):
            ep = out_path.with_name(f"{out_path.stem}_{idx}{out_path.suffix}")
            extra_img.save(str(ep))
            extra_paths.append(str(ep))

        step_ms = (dit_elapsed * 1000.0) / float(max(1, early_steps + late_steps))
        return GenerationResponse(
            output_path=str(out_path),
            additional_paths=extra_paths,
            exported_vizj_path=exported_vizj_str,
            model_variant_used=request.model_variant,
            repo_id_used=target_repo,
            width=request.width,
            height=request.height,
            seed_used=explicit_seed,
            generation_time_seconds=elapsed,
            step_latency_ms=step_ms,
            peak_vram_gb=peak_vram_gb,
            cpu_offload_active=cpu_offload,
            num_inference_steps_used=request.num_inference_steps,
            total_nfe=total_nfe,
            shift_used=request.shift,
            use_v2_grid_used=request.use_v2_grid,
            early_solver_used=request.early_solver,
            late_solver_used=request.late_solver,
            handoff_threshold_used=request.handoff_threshold,
            early_steps=early_steps,
            late_steps=late_steps,
            early_cfg_used=early_cfg,
            late_cfg_used=late_cfg,
            uncond_skip_layers_used=list(request.uncond_skip_layers or []),
            slg_layers_used=list(request.slg_layers or []),
            slg_scale_used=request.slg_scale,
            eta_used=request.eta,
            s_noise_used=request.s_noise,
            enable_thinking_used=request.enable_thinking,
            viztoken_active=True,
            tiktalkin_active=True,
            vizinjector_active=self.vizinjector_active,
            vizinjector_backend=self.vizinjector_backend_name,
            prompt_cache_hit=cache_hit,
            reference_cache_hit=any_ref_cache_hit,
            common_prefix_tokens=common_prefix_len,
            edit_format_used=request.edit_format,
            framing_mode_used=request.framing_mode,
            input_role_used=request.input_role,
            reference_zero_timestep_used=ref_zero_t,
            reference_isolated_kv_used=ref_iso_kv,
            box_attn_isolation_used=max_box_iso_used,
            uncond_uses_references_used=bool(uncond_uses_refs),
            composite_box_images_used=composite_used,
            edit_lora_active=lora_active,
            effective_prompt=effective_prompt,
            mean_luminance=mean_luma,
            rms_contrast=rms_contrast,
            shannon_entropy_bits=entropy,
            laplacian_sharpness=sharpness,
        )