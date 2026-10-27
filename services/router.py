#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.modules["torchvision"] = None
sys.modules["torchvision.transforms"] = None
sys.modules["torchvision.io"] = None

import argparse
import asyncio
import base64
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import gc
import hashlib
import hmac
import io
import json
import logging
import mimetypes
from pathlib import Path
import shutil
import struct
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

import numpy as np
from PIL import Image
import torch
import uvicorn
from fastapi import APIRouter, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("image/png", ".png")
mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("application/octet-stream", ".vizj")
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

SERVICES_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SERVICES_DIR.parent
SHADOWBOXER_DIR = BACKEND_ROOT / "ShadowBoxer"

PATH_ANCHORS = [
    BACKEND_ROOT,
    SHADOWBOXER_DIR,
    SERVICES_DIR,
]
for p in reversed(PATH_ANCHORS):
    p_str = str(p)
    if p.exists() and p_str not in sys.path:
        sys.path.insert(0, p_str)

for pkg_dir in [
    SHADOWBOXER_DIR,
    SHADOWBOXER_DIR / "models",
    SHADOWBOXER_DIR / "viztoken",
    SHADOWBOXER_DIR / "vizinjector",
    SERVICES_DIR,
]:
    if pkg_dir.exists():
        init_file = pkg_dir / "__init__.py"
        if not init_file.exists():
            try:
                init_file.touch()
            except OSError:
                pass

try:
    import ShadowBoxer.schema as shadowboxer_schema_mod

    sys.modules["schema"] = shadowboxer_schema_mod
except Exception:
    import schema as shadowboxer_schema_mod

    sys.modules["ShadowBoxer.schema"] = shadowboxer_schema_mod

from ShadowBoxer.schema import (
    BASELINE_ENGINE_DEFAULTS,
    DEFAULT_LAYOUT_ITEMS,
    DEFAULT_PRESET_FILENAME,
    GenerationRequest,
    SUPPORTED_EDIT_FORMATS,
    SUPPORTED_FRAMING_MODES,
    SUPPORTED_INPUT_ROLES,
    SUPPORTED_MODELS,
    SUPPORTED_PROMPT_MODES,
    SUPPORTED_SOLVERS,
    get_active_engine_defaults,
    has_custom_default_preset,
    parse_layer_list,
)

SUPPORTED_MASK_BLEND_MODES: List[str] = list(
    getattr(
        shadowboxer_schema_mod,
        "SUPPORTED_MASK_BLEND_MODES",
        ["auto", "union", "intersect", "paint_only", "boxes_only", "alpha_only"],
    )
)

ARTIFACTS_DIR = BACKEND_ROOT / "artifacts"
STORAGE_ROOT = BACKEND_ROOT / "storage" / "users"
CONFIG_DIR = BACKEND_ROOT / "config"
for _d in (ARTIFACTS_DIR, STORAGE_ROOT, CONFIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

POW_SECRET = os.environ.get(
    "VISIONSTACK_POW_SECRET", "vs_pow_entropy_matrix_2026_secure"
)
SESSION_SECRET = os.environ.get(
    "VISIONSTACK_SESSION_SECRET", "vs_session_key_2026_master"
)
POW_DIFFICULTY = int(os.environ.get("VISIONSTACK_POW_DIFFICULTY", "4"))
POW_EXPIRATION_SECONDS = 300
ARTIFACT_TTL_SECONDS = 86400
_REPLAY_CACHE: Dict[str, float] = {}

_IS_STANDALONE: bool = os.environ.get("VISIONSTACK_STANDALONE", "0").lower() in (
    "1",
    "true",
    "yes",
)
_CLI_SITE_ROOT: Optional[Path] = None
_CLI_CONFIG_FILE: Optional[Path] = None

_FORBIDDEN_SPA_SEGMENTS = {
    "..",
    "services",
    "shadowboxer",
    "vizinjector",
    "viztoken",
    "storage",
    "config",
    "artifacts",
    "tools",
}


def _bootstrap_cli_environment() -> None:
    global _IS_STANDALONE, _CLI_SITE_ROOT, _CLI_CONFIG_FILE
    for i, arg in enumerate(sys.argv):
        if arg == "--site-root" and i + 1 < len(sys.argv):
            _CLI_SITE_ROOT = Path(sys.argv[i + 1]).resolve()
        elif arg.startswith("--site-root="):
            _CLI_SITE_ROOT = Path(arg.split("=", 1)[1]).resolve()
        elif arg == "--config" and i + 1 < len(sys.argv):
            _CLI_CONFIG_FILE = Path(sys.argv[i + 1]).resolve()
        elif arg.startswith("--config="):
            _CLI_CONFIG_FILE = Path(arg.split("=", 1)[1]).resolve()
        elif arg == "--standalone":
            _IS_STANDALONE = True
            os.environ["VISIONSTACK_STANDALONE"] = "1"
        elif arg == "--headless":
            _IS_STANDALONE = False
            os.environ["VISIONSTACK_STANDALONE"] = "0"


_bootstrap_cli_environment()


def set_cli_overrides(
    standalone: Optional[bool] = None,
    site_root: Optional[str] = None,
    config_file: Optional[str] = None,
) -> None:
    global _IS_STANDALONE, _CLI_SITE_ROOT, _CLI_CONFIG_FILE
    if standalone is not None:
        _IS_STANDALONE = standalone
        os.environ["VISIONSTACK_STANDALONE"] = "1" if standalone else "0"
    if site_root:
        _CLI_SITE_ROOT = Path(site_root).resolve()
    if config_file:
        _CLI_CONFIG_FILE = Path(config_file).resolve()


def is_standalone_mode() -> bool:
    return _IS_STANDALONE or os.environ.get(
        "VISIONSTACK_STANDALONE", "0"
    ).lower() in ("1", "true", "yes")


def resolve_site_root() -> Path:
    if is_standalone_mode():
        candidates = [
            BACKEND_ROOT / "webui",
            Path.cwd() / "webui",
        ]
        for c in candidates:
            if (c / "index.html").exists():
                return c.resolve()
        return (BACKEND_ROOT / "webui").resolve()

    if _CLI_SITE_ROOT and _CLI_SITE_ROOT.exists():
        return _CLI_SITE_ROOT.resolve()

    env_site = os.environ.get("VISIONSTACK_SITE_DIR")
    if env_site and Path(env_site).exists():
        return Path(env_site).resolve()

    candidates = [
        BACKEND_ROOT / "webui",
        Path.cwd() / "webui",
        BACKEND_ROOT / "site",
        Path.cwd() / "site",
    ]
    for c in candidates:
        if (c / "index.html").exists():
            return c.resolve()
    return (BACKEND_ROOT / "webui").resolve()


def get_themes_root() -> Path:
    return resolve_site_root() / "themes"


class PollingLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if '"GET ' in msg:
            if "/jobs/" in msg or "/health" in msg:
                if (
                    " 200 " in msg
                    or " 304 " in msg
                    or " 200 OK" in msg
                    or " 304 Not Modified" in msg
                ):
                    return False
        return True


def slugify(username: str) -> str:
    slug = username.strip().lower()
    slug = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in slug])
    return "_".join(filter(None, slug.split("_"))) or "user"


def _probe_vizj_dimensions(binary: bytes) -> Tuple[int, int]:
    if len(binary) >= 48 and binary[:16] == b"VIZJ_CONTAINER01":
        try:
            orig_w, orig_h, vae_w, vae_h = struct.unpack("<IIII", binary[32:48])
            if vae_w > 0 and vae_h > 0:
                return int(vae_w), int(vae_h)
            if orig_w > 0 and orig_h > 0:
                return int(orig_w), int(orig_h)
        except Exception:
            pass
    return 1024, 1024


def _inspect_reference_file(file_path: Path, user_slug: str) -> Dict[str, Any]:
    st = file_path.stat()
    is_vizj = file_path.suffix.lower() == ".vizj"
    width, height = 1024, 1024
    has_alpha = False

    try:
        if is_vizj:
            with open(file_path, "rb") as f:
                hdr = f.read(64)
            width, height = _probe_vizj_dimensions(hdr)
        else:
            with Image.open(file_path) as img:
                width, height = int(img.width), int(img.height)
                if img.mode in ("RGBA", "LA") or (
                    img.mode == "P" and "transparency" in img.info
                ):
                    alpha_band = img.convert("RGBA").getchannel("A")
                    extrema = alpha_band.getextrema()
                    if extrema and extrema[0] < 250:
                        has_alpha = True
    except Exception:
        pass

    aspect = round(float(width) / float(max(1, height)), 4)
    return {
        "ref_id": file_path.name,
        "filename": file_path.name,
        "width": width,
        "height": height,
        "aspect_ratio": aspect,
        "has_alpha": has_alpha,
        "is_vizj": is_vizj,
        "size_bytes": st.st_size,
        "created_at": datetime.fromtimestamp(
            st.st_mtime, tz=timezone.utc
        ).isoformat(),
        "url": f"api/v1/references/stream/{user_slug}/{file_path.name}",
    }


def get_user_registry_candidates() -> List[Path]:
    if _CLI_CONFIG_FILE and _CLI_CONFIG_FILE.exists():
        return [_CLI_CONFIG_FILE.resolve()]
    custom_file = os.environ.get("VISIONSTACK_USERS_FILE")
    if custom_file and Path(custom_file).exists():
        return [Path(custom_file).resolve()]
    custom_dir = os.environ.get("VISIONSTACK_CONFIG_DIR")
    if custom_dir and (Path(custom_dir) / "users.json").exists():
        return [(Path(custom_dir) / "users.json").resolve()]

    resolved_site = resolve_site_root()
    return [
        resolved_site / "config" / "users.json",
        BACKEND_ROOT / "config" / "users.json",
        BACKEND_ROOT / "webui" / "config" / "users.json",
        SERVICES_DIR / "config" / "users.json",
    ]


def load_user_registry() -> Tuple[Dict[str, Any], Path]:
    candidates = get_user_registry_candidates()
    seen_paths = set()
    for p in candidates:
        resolved = p.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        if resolved.exists() and resolved.is_file():
            try:
                with open(resolved, "r", encoding="utf-8-sig") as f:
                    raw_data = json.load(f)
                    data = (
                        raw_data.get("users", raw_data)
                        if isinstance(raw_data, dict)
                        else {}
                    )
                    if isinstance(data, dict) and data:
                        return data, resolved
            except Exception as e:
                logging.getLogger("uvicorn.error").warning(
                    "Failed parsing user registry at %s: %s", resolved, e
                )

    fallback = {
        "administrator": {
            "display_name": "Administrator",
            "daily_quota": 999,
            "assigned_theme": "obsidian_gold",
            "custom_permissions": ["admin", "unlimited_quota"],
        },
        "admin": {
            "display_name": "Administrator",
            "daily_quota": 999,
            "assigned_theme": "obsidian_gold",
            "custom_permissions": ["admin", "unlimited_quota"],
        },
        "studio": {
            "display_name": "Studio Creator",
            "daily_quota": 50,
            "assigned_theme": "cyber_neon",
            "custom_permissions": [],
        },
    }
    return fallback, Path("EMBEDDED_MEMORY_FALLBACK")


def create_session_token(slug: str) -> str:
    ts = str(int(time.time()))
    payload = f"{slug}:{ts}"
    signature = hmac.new(
        SESSION_SECRET.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"{payload}:{signature}"


def verify_session_token(token: Optional[str]) -> str:
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )
    try:
        parts = token.split(":")
        if len(parts) != 3:
            raise ValueError
        slug, ts, sig = parts[0], parts[1], parts[2]
        payload = f"{slug}:{ts}"
        expected = hmac.new(
            SESSION_SECRET.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(sig, expected):
            raise ValueError
        if time.time() - int(ts) > 604800:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session expired.",
            )
        return slug
    except (ValueError, IndexError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session credentials.",
        )


def generate_pow_challenge() -> Dict[str, Any]:
    now = int(time.time())
    nonce = uuid.uuid4().hex
    raw = f"{now}:{nonce}:{POW_DIFFICULTY}"
    sig = hmac.new(
        POW_SECRET.encode("utf-8"), raw.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return {
        "timestamp": now,
        "nonce": nonce,
        "difficulty": POW_DIFFICULTY,
        "challenge": raw,
        "signature": sig,
    }


def verify_pow_solution(
    challenge: str, signature: str, solution_nonce: str
) -> bool:
    global _REPLAY_CACHE
    now = time.time()
    _REPLAY_CACHE = {
        k: v
        for k, v in _REPLAY_CACHE.items()
        if now - v < POW_EXPIRATION_SECONDS
    }

    cache_key = f"{challenge}:{solution_nonce}"
    if cache_key in _REPLAY_CACHE:
        return False

    expected_sig = hmac.new(
        POW_SECRET.encode("utf-8"), challenge.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(signature, expected_sig):
        return False

    try:
        ts_str, _, diff_str = challenge.split(":")
        ts = int(ts_str)
        diff = int(diff_str)
        if now - ts > POW_EXPIRATION_SECONDS or ts > now + 30:
            return False
    except Exception:
        return False

    attempt = f"{challenge}:{solution_nonce}".encode("utf-8")
    h = hashlib.sha256(attempt).hexdigest()
    if not h.startswith("0" * diff):
        return False

    _REPLAY_CACHE[cache_key] = now
    return True


class PowSubmission(BaseModel):
    challenge: str
    signature: str
    solution_nonce: str


class AuthPayload(BaseModel):
    username: str


class ReferenceUploadPayload(BaseModel):
    filename: str = Field(default="reference.png", max_length=160)
    role: str = Field(default="photo_ref", max_length=40)
    data_url: str = Field(..., min_length=16)


class SynthesisPayload(BaseModel):
    title: str = Field(default="Untitled Vision Master", max_length=120)
    model_variant: Optional[str] = Field(default="sft")
    prompt: str = Field(default="", max_length=8000)
    negative_prompt: str = Field(default="", max_length=4000)
    system_prompt: Optional[str] = Field(default=None, max_length=4000)
    thinking_trace: Optional[str] = Field(default=None, max_length=4000)
    add_vision_id: Optional[bool] = Field(default=None)
    prompt_mode: str = Field(default="natural")
    layout_summary: Optional[str] = Field(default=None, max_length=4000)
    layout_items: List[Dict[str, Any]] = Field(default_factory=list)
    layout_coords_are_pixels: bool = Field(default=True)
    preserve_canvas_outside_boxes: Optional[bool] = Field(default=None)
    box_attn_isolation: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    uncond_uses_references: Optional[bool] = Field(default=None)
    composite_box_images_to_init: Optional[bool] = Field(default=None)

    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)

    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=200)
    shift: Optional[float] = Field(default=None, ge=0.1, le=20.0)
    use_v2_grid: Optional[bool] = Field(default=None)

    early_solver: Optional[str] = Field(default=None)
    late_solver: Optional[str] = Field(default=None)
    handoff_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    guidance_scale: Optional[float] = Field(default=None, ge=0.0, le=30.0)
    early_cfg: Optional[float] = Field(default=None, ge=0.0, le=30.0)
    late_cfg: Optional[float] = Field(default=None, ge=0.0, le=30.0)

    uncond_skip_layers: Optional[List[int]] = Field(default=None)
    slg_layers: Optional[List[int]] = Field(default=None)
    slg_scale: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    slg_start: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    slg_stop: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    eta: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    s_noise: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    max_sequence_length: Optional[int] = Field(default=None, ge=64, le=4096)
    enable_thinking: Optional[bool] = Field(default=None)

    init_image_path: Optional[str] = Field(default=None)
    denoise_strength: Optional[float] = Field(default=None, ge=0.01, le=1.0)
    inpaint_mask_path: Optional[str] = Field(default=None)
    mask_blend_mode: Optional[str] = Field(default=None)
    invert_inpaint_mask: Optional[bool] = Field(default=None)
    control_image_path: Optional[str] = Field(default=None)
    reference_image_paths: List[str] = Field(default_factory=list)

    framing_mode: Optional[str] = Field(default=None)
    input_role: Optional[str] = Field(default=None)
    edit_format: Optional[str] = Field(default=None)
    max_vit_patches: Optional[int] = Field(default=None, ge=64, le=9216)
    deband_8bit: Optional[bool] = Field(default=None)
    prefer_vulkan: Optional[bool] = Field(default=None)
    mtf_boost: Optional[float] = Field(default=None, ge=0.0, le=0.5)
    soft_knee_tau: Optional[float] = Field(default=None, ge=0.001, le=0.25)
    mask_feather_radius: Optional[float] = Field(default=None, ge=0.0, le=6.0)
    latent_clamp_sigma: Optional[float] = Field(default=None, ge=2.0, le=12.0)
    apply_bayer_dither: Optional[bool] = Field(default=None)
    use_reference_cache: Optional[bool] = Field(default=None)

    reference_zero_timestep: Optional[bool] = Field(default=None)
    reference_isolated_kv: Optional[bool] = Field(default=None)
    edit_lora_path: Optional[str] = Field(default=None)
    edit_lora_scale: Optional[float] = Field(default=None, ge=0.0, le=4.0)

    export_vizj: bool = Field(default=False)
    export_vizj_tier: int = Field(default=2, ge=0, le=2)
    num_images_per_prompt: int = Field(default=1, ge=1, le=8)

    seed: Optional[int] = Field(default=None, ge=0)
    cpu_offload: Optional[bool] = Field(default=None)
    pow: PowSubmission

    @field_validator("seed", mode="before")
    @classmethod
    def _coerce_seed(cls, v: Any) -> Optional[int]:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        try:
            val = int(v)
            return val if val >= 0 else None
        except (ValueError, TypeError):
            return None

    @field_validator("width", "height", mode="before")
    @classmethod
    def _coerce_dims(cls, v: Any) -> int:
        try:
            val = max(256, min(2048, int(v)))
            return (val // 16) * 16
        except (ValueError, TypeError):
            return 1024

    @field_validator("uncond_skip_layers", "slg_layers", mode="before")
    @classmethod
    def _coerce_layer_lists(cls, v: Any) -> Optional[List[int]]:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return parse_layer_list(v)

    @field_validator("system_prompt", "thinking_trace", mode="before")
    @classmethod
    def _coerce_opt_strings(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    @field_validator(
        "shift",
        "handoff_threshold",
        "guidance_scale",
        "early_cfg",
        "late_cfg",
        "slg_scale",
        "slg_start",
        "slg_stop",
        "eta",
        "s_noise",
        "denoise_strength",
        "mtf_boost",
        "soft_knee_tau",
        "mask_feather_radius",
        "latent_clamp_sigma",
        "edit_lora_scale",
        "box_attn_isolation",
        mode="before",
    )
    @classmethod
    def _coerce_floats(cls, v: Any) -> Any:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return v


class PresetUpdatePayload(BaseModel):
    engine_defaults: Dict[str, Any] = Field(default_factory=dict)


class SynthesisJob:
    def __init__(self, user_slug: str, request_data: Dict[str, Any]):
        self.job_id = f"vs_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        self.user_slug = user_slug
        self.request_data = request_data
        self.status = "QUEUED"
        self.progress_pct = 0
        self.stage_description = "Studio Queue Position Registered"
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.output_file: Optional[Path] = None
        self.additional_files: List[Path] = []
        self.exported_vizj_file: Optional[Path] = None
        self.telemetry: Optional[Dict[str, Any]] = None
        self.recipe: Optional[Dict[str, Any]] = None
        self.working_draft: Optional[Dict[str, Any]] = None
        self.error_message: Optional[str] = None


def _resolve_user_upload_path(
    user_slug: str, raw_ref: Optional[str]
) -> Optional[str]:
    if not raw_ref or not str(raw_ref).strip():
        return None
    ref_str = str(raw_ref).strip()
    user_uploads_dir = (STORAGE_ROOT / user_slug / "uploads").resolve()
    user_images_dir = (STORAGE_ROOT / user_slug / "images").resolve()
    test_inputs_dir = (SHADOWBOXER_DIR / "artifacts" / "test_inputs").resolve()

    cand_name = Path(ref_str).name
    for base_dir in (user_uploads_dir, user_images_dir, test_inputs_dir):
        cand_path = (base_dir / cand_name).resolve()
        if cand_path.is_file() and cand_path.is_relative_to(base_dir):
            return str(cand_path)

    direct = Path(ref_str)
    if direct.is_file():
        resolved = direct.resolve()
        for allowed_root in (
            user_uploads_dir,
            user_images_dir,
            test_inputs_dir,
            ARTIFACTS_DIR.resolve(),
        ):
            if resolved.is_relative_to(allowed_root):
                return str(resolved)
    return None


class EnginePipeline:
    def __init__(self) -> None:
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_str)
        self._engine: Optional[Any] = None
        self._lock = threading.Lock()
        self._memory_guard: Optional[Any] = None
        try:
            from ShadowBoxer.engine import WDDMMemoryGuard

            self._memory_guard = WDDMMemoryGuard(self.device)
        except Exception:
            try:
                from engine import WDDMMemoryGuard

                self._memory_guard = WDDMMemoryGuard(self.device)
            except Exception:
                self._memory_guard = None

    def _get_or_create_engine(self, model_variant: str = "sft") -> Any:
        if self._engine is None:
            from ShadowBoxer.engine import MOSSImageEngine

            self._engine = MOSSImageEngine(
                model_variant=model_variant,
                device=self.device_str,
                prefer_vulkan=True,
            )
        return self._engine

    def _flush_hardware_memory(self) -> None:
        if self._engine is not None:
            try:
                self._engine._trim_reserved_vram(
                    force=False, required_headroom_gb=1.25
                )
                return
            except Exception:
                pass
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def poll_hardware_telemetry(self) -> Dict[str, Any]:
        active_guard = (
            self._engine.memory_guard
            if (
                self._engine is not None
                and getattr(self._engine, "memory_guard", None) is not None
            )
            else self._memory_guard
        )
        if active_guard is not None:
            snap = active_guard.poll()
            viz_name = (
                self._engine.vizinjector_backend_name
                if self._engine is not None
                else "Standby (Vulkan Ready)"
            )
            loaded_repo = (
                self._engine._loaded_repo_id
                if self._engine is not None
                else None
            )
            return {
                "adapter_name": active_guard.adapter_name,
                "wddm_active": bool(active_guard.active),
                "dedicated_total_gb": round(snap.dedicated_total_gb, 3),
                "dedicated_used_gb": round(snap.dedicated_used_gb, 3),
                "dedicated_avail_gb": round(snap.dedicated_avail_gb, 3),
                "contiguous_avail_gb": round(snap.contiguous_avail_gb, 3),
                "shared_delta_gb": round(snap.shared_delta_gb, 3),
                "pt_allocated_gb": round(snap.pt_allocated_gb, 3),
                "pt_reserved_gb": round(snap.pt_reserved_gb, 3),
                "pt_inactive_split_gb": round(snap.pt_inactive_split_gb, 4),
                "host_avail_gb": round(snap.host_avail_gb, 2),
                "vizinjector_backend": viz_name,
                "loaded_repo_id": loaded_repo,
            }
        return {
            "adapter_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "CPU"
            ),
            "wddm_active": False,
            "dedicated_total_gb": 0.0,
            "dedicated_used_gb": 0.0,
            "dedicated_avail_gb": 0.0,
            "contiguous_avail_gb": 0.0,
            "shared_delta_gb": 0.0,
            "pt_allocated_gb": 0.0,
            "pt_reserved_gb": 0.0,
            "pt_inactive_split_gb": 0.0,
            "host_avail_gb": 0.0,
            "vizinjector_backend": "Standby (Lazy-Init on First Job)",
            "loaded_repo_id": None,
        }

    def run(
        self,
        job: SynthesisJob,
        progress_cb: Callable[[int, str], None],
    ) -> Tuple[
        Path,
        List[Path],
        Optional[Path],
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, Any],
    ]:
        with self._lock:
            req_data = dict(job.request_data)
            raw_seed = req_data.get("seed")
            seed = (
                int(raw_seed)
                if (raw_seed is not None and str(raw_seed).strip() != "")
                else int(np.random.randint(100000, 99999999))
            )

            output_png_path = ARTIFACTS_DIR / f"{job.job_id}.png"
            export_vizj_path: Optional[Path] = (
                (ARTIFACTS_DIR / f"{job.job_id}.vizj")
                if bool(req_data.get("export_vizj"))
                else None
            )

            resolved_init = _resolve_user_upload_path(
                job.user_slug, req_data.get("init_image_path")
            )
            resolved_mask = _resolve_user_upload_path(
                job.user_slug, req_data.get("inpaint_mask_path")
            )
            resolved_ctrl = _resolve_user_upload_path(
                job.user_slug, req_data.get("control_image_path")
            )
            resolved_refs: List[str] = []
            for r_item in req_data.get("reference_image_paths") or []:
                rp = _resolve_user_upload_path(job.user_slug, r_item)
                if rp:
                    resolved_refs.append(rp)

            sanitized_boxes: List[Dict[str, Any]] = []
            for box in req_data.get("layout_items") or []:
                if not isinstance(box, dict):
                    continue
                b_copy = dict(box)
                raw_box_img = (
                    b_copy.get("image_path")
                    or b_copy.get("image")
                    or b_copy.get("vizj_path")
                )
                if raw_box_img:
                    resolved_box_img = _resolve_user_upload_path(
                        job.user_slug, str(raw_box_img)
                    )
                    if resolved_box_img:
                        b_copy["image_path"] = resolved_box_img
                    else:
                        b_copy.pop("image_path", None)
                        b_copy.pop("image", None)
                        b_copy.pop("vizj_path", None)
                sanitized_boxes.append(b_copy)

            valid_fields = set(GenerationRequest.model_fields.keys())
            req_kwargs: Dict[str, Any] = {
                k: v
                for k, v in req_data.items()
                if v is not None and k in valid_fields
            }
            req_kwargs["seed"] = seed
            req_kwargs["output_path"] = str(output_png_path)
            req_kwargs["init_image_path"] = resolved_init
            req_kwargs["inpaint_mask_path"] = resolved_mask
            req_kwargs["control_image_path"] = resolved_ctrl
            req_kwargs["reference_image_paths"] = resolved_refs
            req_kwargs["layout_items"] = sanitized_boxes
            if export_vizj_path is not None:
                req_kwargs["export_vizj_path"] = str(export_vizj_path)

            gen_req = GenerationRequest(**req_kwargs)
            progress_cb(
                6,
                f"Initializing ShadowBoxer ({gen_req.model_variant.upper()}) Pipeline...",
            )
            eng = self._get_or_create_engine(model_variant=gen_req.model_variant)

            def on_engine_step(stage: str, cur: int, tot: int) -> None:
                if stage == "context":
                    pct = 8 + int((cur / max(1, tot)) * 10)
                    progress_cb(
                        pct,
                        f"Encoding Qwen3.5 Hybrid Context ({cur}/{tot} sequences)...",
                    )
                elif stage == "dit":
                    pct = 18 + int((cur / max(1, tot)) * 70)
                    progress_cb(
                        pct,
                        f"Integrating 4D-RoPE Velocity Field ({cur}/{tot} passes)...",
                    )
                elif stage == "vae":
                    pct = 88 + int((cur / max(1, tot)) * 10)
                    progress_cb(
                        pct,
                        "Decoding Flux2-VAE FP32 Latents & Radiometric Mastering...",
                    )

            try:
                with torch.inference_mode():
                    resp = eng.synthesize(gen_req, progress_callback=on_engine_step)

                hw_snap = eng.memory_guard.poll()
                telemetry = {
                    "seed": resp.seed_used,
                    "model_variant": resp.model_variant_used,
                    "repo_id": resp.repo_id_used,
                    "width": resp.width,
                    "height": resp.height,
                    "generation_time_seconds": round(
                        resp.generation_time_seconds, 4
                    ),
                    "step_latency_ms": round(resp.step_latency_ms, 2),
                    "peak_vram_gb": round(resp.peak_vram_gb, 3),
                    "wddm_dedicated_used_gb": round(hw_snap.dedicated_used_gb, 3),
                    "wddm_dedicated_total_gb": round(hw_snap.dedicated_total_gb, 3),
                    "wddm_dedicated_avail_gb": round(hw_snap.dedicated_avail_gb, 3),
                    "wddm_contiguous_avail_gb": round(
                        hw_snap.contiguous_avail_gb, 3
                    ),
                    "wddm_shared_spill_delta_gb": round(hw_snap.shared_delta_gb, 3),
                    "pt_inactive_split_gb": round(hw_snap.pt_inactive_split_gb, 4),
                    "host_avail_gb": round(hw_snap.host_avail_gb, 2),
                    "num_inference_steps": resp.num_inference_steps_used,
                    "total_nfe": resp.total_nfe,
                    "shift": resp.shift_used,
                    "use_v2_grid": resp.use_v2_grid_used,
                    "early_solver": resp.early_solver_used,
                    "late_solver": resp.late_solver_used,
                    "handoff_threshold": resp.handoff_threshold_used,
                    "early_steps": resp.early_steps,
                    "late_steps": resp.late_steps,
                    "early_cfg": resp.early_cfg_used,
                    "late_cfg": resp.late_cfg_used,
                    "uncond_skip_layers": resp.uncond_skip_layers_used,
                    "slg_layers": resp.slg_layers_used,
                    "slg_scale": resp.slg_scale_used,
                    "eta": resp.eta_used,
                    "s_noise": resp.s_noise_used,
                    "enable_thinking": resp.enable_thinking_used,
                    "add_vision_id": bool(
                        gen_req.add_vision_id
                        if gen_req.add_vision_id is not None
                        else True
                    ),
                    "viztoken_active": resp.viztoken_active,
                    "vizinjector_backend": resp.vizinjector_backend,
                    "prompt_cache_hit": resp.prompt_cache_hit,
                    "reference_cache_hit": resp.reference_cache_hit,
                    "common_prefix_tokens": resp.common_prefix_tokens,
                    "edit_format": resp.edit_format_used,
                    "framing_mode": resp.framing_mode_used,
                    "input_role": resp.input_role_used,
                    "mask_blend_mode": getattr(
                        resp, "mask_blend_mode_used", str(gen_req.mask_blend_mode or "auto")
                    ),
                    "invert_inpaint_mask": getattr(
                        resp, "invert_inpaint_mask_used", bool(gen_req.invert_inpaint_mask)
                    ),
                    "reference_zero_timestep": resp.reference_zero_timestep_used,
                    "reference_isolated_kv": resp.reference_isolated_kv_used,
                    "box_attn_isolation": round(resp.box_attn_isolation_used, 3),
                    "uncond_uses_references": resp.uncond_uses_references_used,
                    "composite_box_images": resp.composite_box_images_used,
                    "edit_lora_active": resp.edit_lora_active,
                    "effective_prompt": resp.effective_prompt,
                    "mean_luminance": round(resp.mean_luminance, 4),
                    "rms_contrast": round(resp.rms_contrast, 4),
                    "shannon_entropy_bits": round(resp.shannon_entropy_bits, 4),
                    "laplacian_sharpness": round(resp.laplacian_sharpness, 4),
                }

                recipe = {
                    "model_variant": gen_req.model_variant,
                    "prompt": gen_req.prompt,
                    "negative_prompt": gen_req.negative_prompt,
                    "system_prompt": gen_req.system_prompt,
                    "thinking_trace": gen_req.thinking_trace,
                    "add_vision_id": gen_req.add_vision_id,
                    "prompt_mode": gen_req.prompt_mode,
                    "layout_summary": gen_req.layout_summary,
                    "layout_items": req_data.get("layout_items") or [],
                    "box_attn_isolation": gen_req.box_attn_isolation,
                    "uncond_uses_references": gen_req.uncond_uses_references,
                    "composite_box_images_to_init": gen_req.composite_box_images_to_init,
                    "mask_blend_mode": gen_req.mask_blend_mode,
                    "invert_inpaint_mask": gen_req.invert_inpaint_mask,
                    "width": gen_req.width,
                    "height": gen_req.height,
                    "early_solver": gen_req.early_solver,
                    "late_solver": gen_req.late_solver,
                    "handoff_threshold": gen_req.handoff_threshold,
                    "num_inference_steps": gen_req.num_inference_steps,
                    "shift": gen_req.shift,
                    "early_cfg": gen_req.early_cfg,
                    "late_cfg": gen_req.late_cfg,
                    "uncond_skip_layers": gen_req.uncond_skip_layers,
                    "edit_format": gen_req.edit_format,
                    "framing_mode": gen_req.framing_mode,
                    "input_role": gen_req.input_role,
                }

                working_draft = {
                    "title": req_data.get("title", "Untitled Vision Master"),
                    "model_variant": gen_req.model_variant,
                    "prompt": gen_req.prompt,
                    "negative_prompt": gen_req.negative_prompt,
                    "system_prompt": gen_req.system_prompt,
                    "thinking_trace": gen_req.thinking_trace,
                    "add_vision_id": gen_req.add_vision_id,
                    "prompt_mode": gen_req.prompt_mode,
                    "layout_summary": gen_req.layout_summary,
                    "layout_items": req_data.get("layout_items") or [],
                    "layout_coords_are_pixels": gen_req.layout_coords_are_pixels,
                    "preserve_canvas_outside_boxes": gen_req.preserve_canvas_outside_boxes,
                    "box_attn_isolation": gen_req.box_attn_isolation,
                    "uncond_uses_references": gen_req.uncond_uses_references,
                    "composite_box_images_to_init": gen_req.composite_box_images_to_init,
                    "mask_blend_mode": gen_req.mask_blend_mode,
                    "invert_inpaint_mask": gen_req.invert_inpaint_mask,
                    "width": gen_req.width,
                    "height": gen_req.height,
                    "num_inference_steps": gen_req.num_inference_steps,
                    "shift": gen_req.shift,
                    "use_v2_grid": gen_req.use_v2_grid,
                    "early_solver": gen_req.early_solver,
                    "late_solver": gen_req.late_solver,
                    "handoff_threshold": gen_req.handoff_threshold,
                    "early_cfg": gen_req.early_cfg,
                    "late_cfg": gen_req.late_cfg,
                    "uncond_skip_layers": gen_req.uncond_skip_layers,
                    "slg_layers": gen_req.slg_layers,
                    "slg_scale": gen_req.slg_scale,
                    "slg_start": gen_req.slg_start,
                    "slg_stop": gen_req.slg_stop,
                    "eta": gen_req.eta,
                    "s_noise": gen_req.s_noise,
                    "enable_thinking": gen_req.enable_thinking,
                    "init_image_path": req_data.get("init_image_path"),
                    "denoise_strength": gen_req.denoise_strength,
                    "inpaint_mask_path": req_data.get("inpaint_mask_path"),
                    "control_image_path": req_data.get("control_image_path"),
                    "reference_image_paths": req_data.get("reference_image_paths")
                    or [],
                    "edit_format": gen_req.edit_format,
                    "framing_mode": gen_req.framing_mode,
                    "input_role": gen_req.input_role,
                    "mtf_boost": gen_req.mtf_boost,
                    "mask_feather_radius": gen_req.mask_feather_radius,
                    "latent_clamp_sigma": gen_req.latent_clamp_sigma,
                    "reference_zero_timestep": gen_req.reference_zero_timestep,
                    "reference_isolated_kv": gen_req.reference_isolated_kv,
                    "edit_lora_path": gen_req.edit_lora_path,
                    "edit_lora_scale": gen_req.edit_lora_scale,
                    "num_images_per_prompt": gen_req.num_images_per_prompt,
                    "seed": resp.seed_used,
                }

                extra_paths = [
                    Path(p) for p in resp.additional_paths if Path(p).is_file()
                ]
                return (
                    output_png_path,
                    extra_paths,
                    export_vizj_path
                    if (export_vizj_path and export_vizj_path.is_file())
                    else None,
                    telemetry,
                    recipe,
                    working_draft,
                )
            finally:
                self._flush_hardware_memory()


class ComputeQueue:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[SynthesisJob] = asyncio.Queue()
        self.jobs: Dict[str, SynthesisJob] = {}
        self.events: Dict[str, asyncio.Event] = {}
        self.pipeline = EnginePipeline()
        self.worker_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.active_job: Optional[SynthesisJob] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._worker_loop())
        if self.cleanup_task is None or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())

    def get_job_event(self, job_id: str) -> asyncio.Event:
        if job_id not in self.events:
            self.events[job_id] = asyncio.Event()
        return self.events[job_id]

    def notify_job(self, job_id: str) -> None:
        if job_id in self.events:
            ev = self.events[job_id]
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(ev.set)
            else:
                ev.set()

    def enqueue(self, user_slug: str, request_data: Dict[str, Any]) -> SynthesisJob:
        job = SynthesisJob(user_slug, request_data)
        self.jobs[job.job_id] = job
        self.get_job_event(job.job_id)
        self.queue.put_nowait(job)
        return job

    def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self.jobs.get(job_id)
        if not job:
            return None

        ahead_count = 0
        if job.status == "QUEUED":
            queue_items = list(self.queue._queue)
            try:
                ahead_count = queue_items.index(job)
                if self.active_job is not None:
                    ahead_count += 1
            except ValueError:
                ahead_count = 1 if self.active_job is not None else 0

        est_seconds = 0
        if job.status == "QUEUED":
            est_seconds = ahead_count * 45 + 45
        elif job.status == "PROCESSING":
            est_seconds = max(2, int(45 * (1.0 - job.progress_pct / 100.0)))

        image_url = (
            f"api/v1/images/stream/{job.user_slug}/{job.job_id}_master.png"
            if job.status == "COMPLETED"
            else None
        )
        additional_urls = (
            [
                f"api/v1/images/stream/{job.user_slug}/{p.name}"
                for p in job.additional_files
            ]
            if job.status == "COMPLETED"
            else []
        )
        vizj_url = (
            f"api/v1/images/stream/{job.user_slug}/{job.job_id}.vizj"
            if (job.status == "COMPLETED" and job.exported_vizj_file is not None)
            else None
        )
        return {
            "job_id": job.job_id,
            "status": job.status,
            "stage": job.stage_description,
            "progress_pct": job.progress_pct,
            "users_ahead": ahead_count,
            "estimated_wait_seconds": est_seconds,
            "created_at": job.created_at,
            "image_url": image_url,
            "additional_image_urls": additional_urls,
            "vizj_url": vizj_url,
            "telemetry": job.telemetry,
            "recipe": job.recipe,
            "working_draft": job.working_draft,
            "error": job.error_message,
        }

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(3600)
            now = time.time()
            for pat in ("*.png", "*.vizj"):
                for artifact in ARTIFACTS_DIR.glob(pat):
                    try:
                        if (
                            artifact.is_file()
                            and (now - artifact.stat().st_mtime)
                            > ARTIFACT_TTL_SECONDS
                        ):
                            artifact.unlink(missing_ok=True)
                    except Exception:
                        pass

            dead_jobs = [
                jid
                for jid, j in self.jobs.items()
                if j.status in ("COMPLETED", "FAILED")
                and (now - datetime.fromisoformat(j.created_at).timestamp())
                > ARTIFACT_TTL_SECONDS
            ]
            for jid in dead_jobs:
                self.jobs.pop(jid, None)
                self.events.pop(jid, None)

    async def _worker_loop(self) -> None:
        while True:
            job = await self.queue.get()
            self.active_job = job
            job.status = "PROCESSING"
            job.progress_pct = 5
            job.stage_description = "Initializing Sovereign Vision Engine..."
            self.notify_job(job.job_id)

            loop = asyncio.get_running_loop()

            def progress_hook(pct: int, desc: str) -> None:
                job.progress_pct = pct
                job.stage_description = desc
                self.notify_job(job.job_id)

            try:
                (
                    output_file,
                    extra_files,
                    vizj_file,
                    telemetry,
                    recipe,
                    working_draft,
                ) = await loop.run_in_executor(
                    None,
                    self.pipeline.run,
                    job,
                    progress_hook,
                )

                user_dir = STORAGE_ROOT / job.user_slug
                user_images_dir = user_dir / "images"
                user_images_dir.mkdir(parents=True, exist_ok=True)

                stored_png = user_images_dir / f"{job.job_id}_master.png"
                if output_file.exists():
                    stored_png.write_bytes(output_file.read_bytes())

                stored_extras: List[Path] = []
                for idx, ef in enumerate(extra_files, start=2):
                    if ef.exists():
                        dest_extra = user_images_dir / f"{job.job_id}_master_{idx}.png"
                        dest_extra.write_bytes(ef.read_bytes())
                        stored_extras.append(dest_extra)

                stored_vizj: Optional[Path] = None
                if vizj_file is not None and vizj_file.exists():
                    stored_vizj = user_images_dir / f"{job.job_id}.vizj"
                    stored_vizj.write_bytes(vizj_file.read_bytes())

                history_file = user_dir / "history.json"
                history_data: Dict[str, Any] = {
                    "user_slug": job.user_slug,
                    "images": [],
                }
                if history_file.exists():
                    try:
                        with open(history_file, "r", encoding="utf-8-sig") as f:
                            loaded = json.load(f)
                            if isinstance(loaded, dict):
                                history_data = loaded
                    except Exception:
                        pass

                image_entry = {
                    "image_id": job.job_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "title": job.request_data.get(
                        "title", "Untitled Vision Master"
                    ),
                    "model_variant": telemetry.get("model_variant", "sft"),
                    "width": telemetry.get("width", 1024),
                    "height": telemetry.get("height", 1024),
                    "seed": telemetry.get("seed", 42),
                    "image_url": f"api/v1/images/stream/{job.user_slug}/{job.job_id}_master.png",
                    "additional_image_urls": [
                        f"api/v1/images/stream/{job.user_slug}/{p.name}"
                        for p in stored_extras
                    ],
                    "vizj_url": (
                        f"api/v1/images/stream/{job.user_slug}/{job.job_id}.vizj"
                        if stored_vizj is not None
                        else None
                    ),
                    "status": "COMPLETED",
                    "recipe": recipe,
                    "working_draft": working_draft,
                    "telemetry": telemetry,
                }
                history_data.setdefault("images", []).insert(0, image_entry)

                temp_hist = history_file.with_suffix(".tmp")
                with open(temp_hist, "w", encoding="utf-8") as f:
                    json.dump(history_data, f, indent=2, ensure_ascii=False)
                temp_hist.replace(history_file)

                job.output_file = output_file
                job.additional_files = stored_extras
                job.exported_vizj_file = stored_vizj
                job.telemetry = telemetry
                job.recipe = recipe
                job.working_draft = working_draft
                job.progress_pct = 100
                job.stage_description = "Studio Vision Master Complete"
                job.status = "COMPLETED"
            except Exception as e:
                logging.getLogger("uvicorn.error").error(
                    "Synthesis job %s failed: %s", job.job_id, e, exc_info=True
                )
                job.status = "FAILED"
                job.error_message = str(e)
            finally:
                self.notify_job(job.job_id)
                self.active_job = None
                self.queue.task_done()


compute_queue = ComputeQueue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    compute_queue.start()
    users, src = load_user_registry()
    defaults = get_active_engine_defaults()
    print("=" * 84)
    print("  VISIONSTACK UNIFIED SPATIO-VISUAL ENGINE // SERVICE ROUTER ACTIVE")
    print(
        f"  Mode            : {'STANDALONE DESKTOP' if is_standalone_mode() else 'HEADLESS / PROXY DAEMON'}"
    )
    print(f"  Site Root       : {resolve_site_root()}")
    print(f"  User Registry   : {src}")
    print(
        f"  Engine Preset   : {'Active default.json Override' if has_custom_default_preset() else 'Optimal Baseline (V03 Heun->IPNDM)'}"
    )
    print(
        f"  Default Solvers : Early={defaults['early_solver'].upper()} (CFG {defaults['early_cfg']}) -> Late={defaults['late_solver'].upper()} (CFG {defaults['late_cfg']}) @ t*={defaults['handoff_threshold']}"
    )
    print(f"  Accounts ({len(users)})    : {', '.join(users.keys())}")
    print("=" * 84)
    yield


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        return response


app = FastAPI(
    title="VisionStack Unified Spatio-Visual Daemon",
    version="2.0.0",
    root_path="",
    lifespan=lifespan,
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router = APIRouter()


@api_router.get("/auth/challenge")
async def get_challenge():
    return generate_pow_challenge()


@api_router.post("/auth/login")
async def login(payload: AuthPayload):
    raw_input = payload.username.strip()
    input_slug = slugify(raw_input)
    users_map, _ = load_user_registry()

    matched_slug = None
    if input_slug in users_map:
        matched_slug = input_slug
    else:
        for key, meta in users_map.items():
            if slugify(key) == input_slug:
                matched_slug = key
                break
            display_name = str(meta.get("display_name", "")).strip()
            if display_name and (
                display_name.lower() == raw_input.lower()
                or slugify(display_name) == input_slug
            ):
                matched_slug = key
                break

    if not matched_slug:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid creator credentials.",
        )

    raw_slug = slugify(matched_slug)
    user_meta = users_map.get(matched_slug, {})
    token = create_session_token(raw_slug)

    user_dir = STORAGE_ROOT / raw_slug
    (user_dir / "images").mkdir(parents=True, exist_ok=True)
    (user_dir / "uploads").mkdir(parents=True, exist_ok=True)
    history_file = user_dir / "history.json"

    images: List[Dict[str, Any]] = []
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8-sig") as f:
                images = json.load(f).get("images", [])
                for img in images:
                    if img.get("image_url") or img.get("recipe"):
                        img.setdefault("status", "COMPLETED")
        except Exception:
            pass

    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    tokens_used_today = sum(
        1
        for img in images
        if img.get("created_at", "").startswith(today_utc)
        and not img.get("is_default", False)
    )
    daily_quota = int(user_meta.get("daily_quota", 25))
    perms = user_meta.get("custom_permissions", [])
    is_admin = "admin" in perms or "unlimited_quota" in perms
    tokens_remaining = (
        999 if is_admin else max(0, daily_quota - tokens_used_today)
    )

    registry_path = get_themes_root() / "registry.json"
    assigned_theme = user_meta.get("assigned_theme") or "obsidian_gold"
    theme_manifest = {}
    if registry_path.exists():
        try:
            with open(registry_path, "r", encoding="utf-8-sig") as f:
                theme_manifest = (
                    json.load(f).get("themes", {}).get(assigned_theme, {})
                )
        except Exception:
            pass

    return {
        "status": "authenticated",
        "token": token,
        "user": {
            "slug": raw_slug,
            "display_name": user_meta.get("display_name", payload.username),
            "daily_quota": daily_quota,
            "tokens_remaining": tokens_remaining,
            "assigned_theme": assigned_theme,
            "is_admin": is_admin,
        },
        "theme": {
            "theme_id": assigned_theme,
            "manifest": theme_manifest,
        },
        "images": images,
        "engine_defaults": get_active_engine_defaults(),
        "default_layout_items": DEFAULT_LAYOUT_ITEMS,
    }


@api_router.get("/themes/registry")
async def get_themes():
    registry_path = get_themes_root() / "registry.json"
    if not registry_path.exists():
        return {"revolving_pool": [], "themes": {}}
    try:
        with open(registry_path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return {"revolving_pool": [], "themes": {}}


@api_router.get("/health")
async def health():
    defaults = get_active_engine_defaults()
    hw = compute_queue.pipeline.poll_hardware_telemetry()
    return {
        "status": "online",
        "engine_device": compute_queue.pipeline.device_str,
        "active_job": (
            compute_queue.active_job.job_id if compute_queue.active_job else None
        ),
        "queue_depth": compute_queue.queue.qsize(),
        "engine_preset_active": has_custom_default_preset(),
        "baseline_parameters": dict(defaults),
        "supported_models": SUPPORTED_MODELS,
        "supported_solvers": SUPPORTED_SOLVERS,
        "supported_prompt_modes": SUPPORTED_PROMPT_MODES,
        "supported_edit_formats": SUPPORTED_EDIT_FORMATS,
        "supported_framing_modes": SUPPORTED_FRAMING_MODES,
        "supported_input_roles": SUPPORTED_INPUT_ROLES,
        "supported_mask_blend_modes": SUPPORTED_MASK_BLEND_MODES,
        "hardware": hw,
    }


@api_router.post("/presets/default")
async def update_default_preset(
    payload: PresetUpdatePayload,
    authorization: Optional[str] = Header(None),
):
    token = authorization.replace("Bearer ", "").strip() if authorization else None
    slug = verify_session_token(token)
    users_map, _ = load_user_registry()
    user_meta = users_map.get(slug, {})
    perms = user_meta.get("custom_permissions", [])
    if "admin" not in perms and "unlimited_quota" not in perms:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator permissions required to update authoritative default.json.",
        )

    merged = dict(get_active_engine_defaults())
    merged.update(payload.engine_defaults)
    merged.setdefault("prompt", "Authoritative ShadowBoxer Studio Preset")
    req = GenerationRequest.from_dict(merged)
    target_path = SHADOWBOXER_DIR / DEFAULT_PRESET_FILENAME
    req.save_preset(target_path)
    return {
        "status": "updated",
        "preset_path": str(target_path),
        "engine_preset_active": has_custom_default_preset(),
        "baseline_parameters": get_active_engine_defaults(),
    }


@api_router.post("/references/upload")
async def upload_reference_asset(
    payload: ReferenceUploadPayload,
    authorization: Optional[str] = Header(None),
):
    token = authorization.replace("Bearer ", "").strip() if authorization else None
    slug = verify_session_token(token)

    raw_data = payload.data_url.strip()
    if "," in raw_data and raw_data.startswith("data:"):
        raw_data = raw_data.split(",", 1)[1]
    try:
        binary = base64.b64decode(raw_data)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid base64 image payload.",
        )

    if len(binary) > 40 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Reference asset exceeds 40 MB limit.",
        )

    uploads_dir = STORAGE_ROOT / slug / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    ref_id = f"ref_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"

    if len(binary) >= 16 and binary[:16] == b"VIZJ_CONTAINER01":
        dest_file = uploads_dir / f"{ref_id}.vizj"
        dest_file.write_bytes(binary)
        meta = _inspect_reference_file(dest_file, slug)
        meta["filename"] = payload.filename
        meta["role"] = payload.role
        return meta

    try:
        with Image.open(io.BytesIO(binary)) as img:
            rgba = img.convert("RGBA")
            dest_file = uploads_dir / f"{ref_id}.png"
            rgba.save(dest_file, format="PNG")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported or corrupted image file: {e}",
        )

    meta = _inspect_reference_file(dest_file, slug)
    meta["filename"] = payload.filename
    meta["role"] = payload.role
    return meta


@api_router.post("/references/stage-defaults")
async def stage_default_test_assets(authorization: Optional[str] = Header(None)):
    token = authorization.replace("Bearer ", "").strip() if authorization else None
    slug = verify_session_token(token)

    try:
        from ShadowBoxer.harness import stage_test_input_files
    except Exception:
        from harness import stage_test_input_files

    test_dir = SHADOWBOXER_DIR / "artifacts" / "test_inputs"
    staged_map = stage_test_input_files(test_dir)
    uploads_dir = STORAGE_ROOT / slug / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    staged_names: List[str] = []
    seen_files = set()
    for src_path in staged_map.values():
        p = Path(src_path)
        if p.is_file() and p.name not in seen_files:
            seen_files.add(p.name)
            dst_path = uploads_dir / p.name
            shutil.copy2(p, dst_path)
            staged_names.append(dst_path.name)

    return {
        "status": "staged",
        "staged_files": staged_names,
    }


@api_router.get("/references")
async def list_reference_assets(authorization: Optional[str] = Header(None)):
    token = authorization.replace("Bearer ", "").strip() if authorization else None
    slug = verify_session_token(token)
    uploads_dir = STORAGE_ROOT / slug / "uploads"
    if not uploads_dir.exists():
        return {"references": []}
    items = []
    for f in sorted(
        uploads_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        if f.is_file() and f.suffix.lower() in (
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".vizj",
        ):
            items.append(_inspect_reference_file(f, slug))
    return {"references": items}


@api_router.get("/references/stream/{user_slug}/{filename}")
async def stream_reference_asset(user_slug: str, filename: str):
    safe_slug = slugify(user_slug)
    safe_name = Path(filename).name
    target = (STORAGE_ROOT / safe_slug / "uploads" / safe_name).resolve()
    if not target.is_file() or not target.is_relative_to(
        (STORAGE_ROOT / safe_slug / "uploads").resolve()
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Reference asset not found.",
        )
    media = (
        "application/octet-stream"
        if target.suffix.lower() == ".vizj"
        else "image/png"
    )
    return FileResponse(
        str(target),
        media_type=media,
        headers={
            "Cross-Origin-Resource-Policy": "cross-origin",
            "Cache-Control": "public, max-age=3600",
        },
    )


@api_router.delete("/references/{ref_id}")
async def delete_reference_asset(
    ref_id: str,
    authorization: Optional[str] = Header(None),
):
    token = authorization.replace("Bearer ", "").strip() if authorization else None
    slug = verify_session_token(token)
    safe_name = Path(ref_id).name
    target = (STORAGE_ROOT / slug / "uploads" / safe_name).resolve()
    if target.is_file() and target.is_relative_to(
        (STORAGE_ROOT / slug / "uploads").resolve()
    ):
        target.unlink(missing_ok=True)
    return {"status": "deleted", "ref_id": safe_name}


@api_router.post("/synthesize")
async def synthesize(
    payload: SynthesisPayload,
    authorization: Optional[str] = Header(None),
):
    token = authorization.replace("Bearer ", "").strip() if authorization else None
    slug = verify_session_token(token)

    if not payload.prompt.strip() and not payload.layout_items:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Prompt or spatial layout boxes required.",
        )

    users_map, _ = load_user_registry()
    user_meta = users_map.get(slug, {})
    perms = user_meta.get("custom_permissions", [])
    is_unlimited = "admin" in perms or "unlimited_quota" in perms

    if not is_unlimited:
        history_file = STORAGE_ROOT / slug / "history.json"
        tokens_used_today = 0
        if history_file.exists():
            try:
                with open(history_file, "r", encoding="utf-8-sig") as f:
                    raw_images = json.load(f).get("images", [])
                    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    tokens_used_today = sum(
                        1
                        for img in raw_images
                        if img.get("created_at", "").startswith(today_utc)
                        and not img.get("is_default", False)
                    )
            except Exception:
                pass
        daily_quota = int(user_meta.get("daily_quota", 25))
        if tokens_used_today >= daily_quota:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Daily synthesis token quota reached. Quotas reset at 00:00 UTC.",
            )

    if not verify_pow_solution(
        payload.pow.challenge,
        payload.pow.signature,
        payload.pow.solution_nonce,
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bot defense proof-of-work verification failed.",
        )

    job = compute_queue.enqueue(slug, payload.model_dump())
    return compute_queue.get_status(job.job_id)


@api_router.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    request: Request,
    authorization: Optional[str] = Header(None),
):
    token = authorization.replace("Bearer ", "").strip() if authorization else None
    verify_session_token(token)
    info = compute_queue.get_status(job_id)
    if not info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Master job record not found.",
        )

    etag = f'W/"{job_id}-{info.get("status")}-{info.get("progress_pct")}-{info.get("users_ahead")}"'
    if_none_match = request.headers.get("if-none-match")
    if if_none_match and if_none_match == etag:
        return Response(
            status_code=status.HTTP_304_NOT_MODIFIED, headers={"ETag": etag}
        )
    return JSONResponse(content=info, headers={"ETag": etag})


@api_router.get("/jobs/{job_id}/stream")
async def stream_job_status(
    job_id: str,
    request: Request,
    authorization: Optional[str] = Header(None),
    token: Optional[str] = None,
):
    auth_token = token or (
        authorization.replace("Bearer ", "").strip() if authorization else None
    )
    verify_session_token(auth_token)
    job = compute_queue.jobs.get(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Master job record not found.",
        )

    async def event_generator():
        last_state_str = None
        event = compute_queue.get_job_event(job_id)
        while True:
            if await request.is_disconnected():
                break
            current_status = compute_queue.get_status(job_id)
            if not current_status:
                break
            state_str = json.dumps(current_status)
            if state_str != last_state_str:
                last_state_str = state_str
                yield f"data: {state_str}\n\n"
            if current_status.get("status") in ("COMPLETED", "FAILED"):
                break
            try:
                await asyncio.wait_for(event.wait(), timeout=1.0)
                event.clear()
            except asyncio.TimeoutError:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@api_router.get("/images/{job_id}")
async def get_image_direct(job_id: str):
    job = compute_queue.jobs.get(job_id)
    if (
        not job
        or job.status != "COMPLETED"
        or not job.output_file
        or not job.output_file.exists()
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Master image artifact not ready.",
        )
    return FileResponse(
        str(job.output_file),
        media_type="image/png",
        headers={
            "Cross-Origin-Resource-Policy": "cross-origin",
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
        },
    )


@api_router.get("/images/stream/{user_slug}/{filename}")
async def get_image_stream_user(user_slug: str, filename: str):
    safe_slug = slugify(user_slug)
    safe_name = Path(filename).name
    target_file = STORAGE_ROOT / safe_slug / "images" / safe_name

    media_type = (
        "application/octet-stream"
        if safe_name.lower().endswith(".vizj")
        else "image/png"
    )
    headers = {
        "Cross-Origin-Resource-Policy": "cross-origin",
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-cache",
    }
    if target_file.exists() and target_file.is_file():
        return FileResponse(
            str(target_file), media_type=media_type, headers=headers
        )

    clean_stem = (
        safe_name.replace("_master.png", "")
        .replace(".png", "")
        .replace(".vizj", "")
    )
    ext = ".vizj" if safe_name.lower().endswith(".vizj") else ".png"
    artifact_file = ARTIFACTS_DIR / f"{clean_stem}{ext}"
    if artifact_file.exists() and artifact_file.is_file():
        return FileResponse(
            str(artifact_file), media_type=media_type, headers=headers
        )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Master image artifact unavailable.",
    )


@api_router.delete("/images/{image_id}")
async def delete_user_image(
    image_id: str,
    authorization: Optional[str] = Header(None),
):
    token = authorization.replace("Bearer ", "").strip() if authorization else None
    slug = verify_session_token(token)

    user_dir = STORAGE_ROOT / slug
    history_file = user_dir / "history.json"
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8-sig") as f:
                history_data = json.load(f)
            original_images = history_data.get("images", [])
            history_data["images"] = [
                img
                for img in original_images
                if img.get("image_id") != image_id
            ]
            temp_hist = history_file.with_suffix(".tmp")
            with open(temp_hist, "w", encoding="utf-8") as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            temp_hist.replace(history_file)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed updating vault storage: {e}"
            )

    images_dir = user_dir / "images"
    for suffix in (f"{image_id}_master.png", f"{image_id}.vizj"):
        f_path = images_dir / suffix
        if f_path.exists():
            f_path.unlink(missing_ok=True)
    if images_dir.is_dir():
        for extra_p in images_dir.glob(f"{image_id}_master_*.png"):
            extra_p.unlink(missing_ok=True)
    return {"status": "deleted", "image_id": image_id}


for _prefix in (
    "/api/v1",
    "/v1",
    "/VisionStack/api/v1",
    "/visionstack/api/v1",
    "/ShadowBoxer/api/v1",
    "/shadowboxer/api/v1",
):
    app.include_router(api_router, prefix=_prefix)


def mount_static_and_spa() -> None:
    site_dir = resolve_site_root()
    if site_dir.exists() and (site_dir / "index.html").exists():
        for sub in ("assets", "themes", "public", "wasm"):
            sub_dir = site_dir / sub
            if sub_dir.exists():
                app.mount(
                    f"/{sub}", StaticFiles(directory=str(sub_dir)), name=sub
                )

    @app.get("/")
    async def serve_index():
        idx_file = resolve_site_root() / "index.html"
        if idx_file.is_file():
            return FileResponse(str(idx_file))
        return JSONResponse(
            content={
                "service": "VisionStack Unified Spatio-Visual Daemon",
                "status": "online",
            }
        )

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        norm_path = full_path.replace("\\", "/").strip("/")
        path_lower = norm_path.lower()
        parts_lower = [p for p in path_lower.split("/") if p]
        if any(
            p in _FORBIDDEN_SPA_SEGMENTS or p.startswith(".")
            for p in parts_lower
        ):
            raise HTTPException(status_code=403, detail="Forbidden")

        if (
            path_lower.startswith("auth/")
            or path_lower.startswith("synthesize")
            or path_lower.startswith("jobs/")
            or path_lower.startswith("images/")
            or path_lower.startswith("references")
            or path_lower.startswith("presets/")
            or path_lower.startswith("health")
            or path_lower.startswith("themes/")
            or path_lower.startswith("api/")
            or path_lower.startswith("v1/")
            or path_lower.startswith("visionstack/")
            or path_lower.startswith("shadowboxer/")
        ):
            return JSONResponse(
                status_code=404,
                content={"detail": f"Route not found: /{full_path}"},
            )

        current_site = resolve_site_root().resolve()
        try:
            target = (current_site / norm_path).resolve()
            if not target.is_relative_to(current_site):
                raise HTTPException(status_code=403, detail="Forbidden")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=404, detail="Not Found")

        if target.exists() and target.is_file():
            if any(
                part.lower() in _FORBIDDEN_SPA_SEGMENTS
                for part in target.parts
            ):
                raise HTTPException(status_code=403, detail="Forbidden")
            return FileResponse(str(target))

        suffix = Path(norm_path).suffix.lower()
        if suffix and suffix not in (".html", ".htm"):
            raise HTTPException(status_code=404, detail="Not Found")

        idx_fallback = current_site / "index.html"
        if idx_fallback.is_file():
            return FileResponse(str(idx_fallback))
        raise HTTPException(status_code=404, detail="Not Found")


mount_static_and_spa()


def run_server(host: str, port: int) -> None:
    uv_access = logging.getLogger("uvicorn.access")
    uv_access.addFilter(PollingLogFilter())
    uvicorn.run(app, host=host, port=port, log_level="info")


def launch_standalone(host: str, port: int) -> None:
    try:
        import webview
    except ImportError:
        import webbrowser

        threading.Thread(
            target=run_server, args=(host, port), daemon=True
        ).start()
        time.sleep(1.0)
        webbrowser.open(f"http://{host}:{port}/")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            sys.exit(0)

    server_thread = threading.Thread(
        target=run_server, args=(host, port), daemon=True
    )
    server_thread.start()
    time.sleep(1.0)
    webview.create_window(
        title="VisionStack — Sovereign Spatio-Visual Studio",
        url=f"http://{host}:{port}/",
        width=1440,
        height=920,
        min_size=(1080, 720),
        background_color="#07090e",
    )
    webview.start(debug=False)
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VisionStack Unified Services & Compute Daemon Router"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host interface to bind"
    )
    parser.add_argument(
        "--port", type=int, default=8766, help="Port to bind"
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Launch native desktop window",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run strictly as headless daemon",
    )
    parser.add_argument(
        "--site-root",
        type=str,
        default=None,
        help="Explicit path to site/webui assets directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Explicit path to users.json file",
    )
    args = parser.parse_args()

    set_cli_overrides(
        standalone=(
            args.standalone
            if args.standalone
            else (False if args.headless else None)
        ),
        site_root=args.site_root,
        config_file=args.config,
    )

    if is_standalone_mode():
        launch_standalone(args.host, args.port)
    else:
        run_server(args.host, args.port)