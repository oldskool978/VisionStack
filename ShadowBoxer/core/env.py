# ShadowBoxer/core/env.py
from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
from typing import Any, Optional, Union
import warnings

ROOT_DIR = Path(__file__).resolve().parent.parent
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

import torch

if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
    torch.backends.cuda.matmul.allow_tf32 = False
if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("highest")

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


def _canonicalize_device(device: Union[str, torch.device]) -> torch.device:
    dev = torch.device(device) if isinstance(device, str) else device
    if dev.type == "cuda" and dev.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return dev


def _same_device(dev_a: torch.device, dev_b: torch.device) -> bool:
    return _canonicalize_device(dev_a) == _canonicalize_device(dev_b)


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


__all__ = [
    "ROOT_DIR",
    "CACHE_DIR",
    "MIOPEN_CACHE_DIR",
    "MIOPEN_DB_DIR",
    "MIOPEN_KERNELS_DIR",
    "COMGR_CACHE_DIR",
    "TMP_DIR",
    "HAS_VIZINJECTOR",
    "ConditionedVisionBundle",
    "VIZJ_TIER_WIRE_WEBP",
    "VIZJ_TIER_VAE_LATENT",
    "VIZJ_TIER_DUAL_LATENT",
    "VizInjectorClient",
    "VizInjectorRequest",
    "_viz_subphase_radial_clamp",
    "_resolve_cache_dir",
    "_configure_rocm_runtime_env",
    "_canonicalize_device",
    "_same_device",
    "_pin_cpu_tensor",
    "_evacuate_to_cpu",
]