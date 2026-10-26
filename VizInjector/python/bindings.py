# VizInjector/python/bindings.py
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch

from .modeling_vae import Flux2AutoencoderKL
from .schema import (
    VizInjectorRequest,
    VizInjectorResponseMeta,
    VizInjectorTelemetry,
)

HERE_DIR = Path(__file__).resolve().parent
ROOT_DIR = HERE_DIR.parent
IS_WIN = sys.platform == "win32"
_DLL_NAME = "vizinjector.dll" if IS_WIN else "libvizinjector.so"
_VIZJ_MAGIC = b"VIZJ_CONTAINER01"
_VIZJ_HEADER_SIZE = 144

VIZJ_TIER_WIRE_WEBP = 0
VIZJ_TIER_VAE_LATENT = 1
VIZJ_TIER_DUAL_LATENT = 2

VIZJ_DTYPE_FP32 = 0
VIZJ_DTYPE_BF16 = 1
VIZJ_DTYPE_FP16 = 2
VIZJ_DTYPE_UINT8_WEBP = 3


class CVizPerceptualTelemetry(ctypes.Structure):
    _fields_ = [
        ("mean_luminance", ctypes.c_float),
        ("rms_contrast", ctypes.c_float),
        ("shannon_entropy_bits", ctypes.c_float),
        ("laplacian_sharpness", ctypes.c_float),
    ]


class CVizInjectorConfig(ctypes.Structure):
    _fields_ = [
        ("target_width", ctypes.c_uint32),
        ("target_height", ctypes.c_uint32),
        ("max_vit_patches", ctypes.c_uint32),
        ("framing_mode", ctypes.c_int32),
        ("input_role", ctypes.c_int32),
        ("edit_mode_384_vit", ctypes.c_int32),
        ("emit_vae_branch", ctypes.c_int32),
        ("emit_vit_branch", ctypes.c_int32),
        ("emit_mask_branch", ctypes.c_int32),
        ("emit_wire_vizj", ctypes.c_int32),
        ("deband_8bit", ctypes.c_int32),
        ("prefer_vulkan", ctypes.c_int32),
        ("seed", ctypes.c_uint32),
        ("mtf_boost", ctypes.c_float),
        ("soft_knee_tau", ctypes.c_float),
        ("mask_feather_radius", ctypes.c_float),
        ("webp_quality", ctypes.c_float),
        ("webp_lossless", ctypes.c_int32),
    ]


class CVizInjectorOutput(ctypes.Structure):
    _fields_ = [
        ("vae_planar_f32", ctypes.POINTER(ctypes.c_float)),
        ("vae_width", ctypes.c_uint32),
        ("vae_height", ctypes.c_uint32),
        ("vae_Floats", ctypes.c_uint32),
        ("vit_patches_f32", ctypes.POINTER(ctypes.c_float)),
        ("vit_width", ctypes.c_uint32),
        ("vit_height", ctypes.c_uint32),
        ("vit_num_patches", ctypes.c_uint32),
        ("vit_Floats", ctypes.c_uint32),
        ("grid_thw", ctypes.c_uint32 * 3),
        ("llm_tokens", ctypes.c_uint32),
        ("latent_mask_f32", ctypes.POINTER(ctypes.c_float)),
        ("mask_width", ctypes.c_uint32),
        ("mask_height", ctypes.c_uint32),
        ("mask_Floats", ctypes.c_uint32),
        ("wire_vizj_bytes", ctypes.POINTER(ctypes.c_uint8)),
        ("wire_vizj_len", ctypes.c_uint32),
        ("orig_width", ctypes.c_uint32),
        ("orig_height", ctypes.c_uint32),
        ("mean_luminance", ctypes.c_float),
        ("rms_contrast", ctypes.c_float),
        ("shannon_entropy_bits", ctypes.c_float),
        ("laplacian_sharpness", ctypes.c_float),
        ("backend_used", ctypes.c_int32),
        ("status_code", ctypes.c_int32),
    ]


class CVizContainerHeader(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("magic", ctypes.c_char * 16),
        ("version", ctypes.c_uint32),
        ("tier", ctypes.c_uint32),
        ("vae_dtype", ctypes.c_uint32),
        ("vit_dtype", ctypes.c_uint32),
        ("orig_width", ctypes.c_uint32),
        ("orig_height", ctypes.c_uint32),
        ("vae_width", ctypes.c_uint32),
        ("vae_height", ctypes.c_uint32),
        ("vit_width", ctypes.c_uint32),
        ("vit_height", ctypes.c_uint32),
        ("grid_thw", ctypes.c_uint32 * 3),
        ("llm_tokens", ctypes.c_uint32),
        ("vae_offset", ctypes.c_uint64),
        ("vae_bytes", ctypes.c_uint64),
        ("vit_offset", ctypes.c_uint64),
        ("vit_bytes", ctypes.c_uint64),
        ("mask_offset", ctypes.c_uint64),
        ("mask_bytes", ctypes.c_uint64),
        ("mean_luminance", ctypes.c_float),
        ("rms_contrast", ctypes.c_float),
        ("shannon_entropy_bits", ctypes.c_float),
        ("laplacian_sharpness", ctypes.c_float),
        ("payload_crc32c", ctypes.c_uint32),
        ("reserved", ctypes.c_uint8 * 4),
    ]


assert (
    ctypes.sizeof(CVizContainerHeader) == _VIZJ_HEADER_SIZE
), f"CVizContainerHeader size mismatch: {ctypes.sizeof(CVizContainerHeader)}"


@dataclass
class ConditionedVisionBundle:
    vae_tensor: Optional[torch.Tensor]
    vit_patches: Optional[torch.Tensor]
    grid_thw: Optional[torch.Tensor]
    llm_tokens: int
    latent_mask: Optional[torch.Tensor]
    packed_latents: Optional[torch.Tensor]
    vit_embeds: Optional[torch.Tensor]
    wire_vizj_bytes: Optional[bytes]
    meta: VizInjectorResponseMeta


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


def _locate_native_library(explicit_path: Optional[Union[str, Path]] = None) -> Path:
    candidates: List[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path).resolve())
    env_override = os.environ.get("VIZINJECTOR_DLL_PATH")
    if env_override:
        candidates.append(Path(env_override).resolve())
    candidates.extend(
        [
            HERE_DIR / _DLL_NAME,
            ROOT_DIR / "dist" / "bin" / _DLL_NAME,
            ROOT_DIR / _DLL_NAME,
        ]
    )
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(
        f"Unable to locate sovereign binary {_DLL_NAME}. Run python tools/forge_dag.py first."
    )


def _locate_ffmpeg_binary() -> Optional[Path]:
    ff_name = "ffmpeg.exe" if IS_WIN else "ffmpeg"
    candidates = [
        HERE_DIR / ff_name,
        ROOT_DIR / "dist" / "bin" / ff_name,
        ROOT_DIR / "bin" / "ffmpeg" / ff_name,
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def resolve_local_vae_dir() -> Optional[Path]:
    hub_candidates: List[Path] = []
    env_hub = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env_hub and env_hub.strip():
        hub_candidates.append(Path(env_hub).resolve())
    env_hf = os.environ.get("HF_HOME")
    if env_hf and env_hf.strip():
        hub_candidates.append((Path(env_hf) / "hub").resolve())
    hub_candidates.extend(
        [
            (ROOT_DIR.parent / ".hf_cache" / "hub").resolve(),
            (ROOT_DIR / ".hf_cache" / "hub").resolve(),
            (HERE_DIR.parent / ".hf_cache" / "hub").resolve(),
        ]
    )
    seen = set()
    for hub_dir in hub_candidates:
        if hub_dir in seen or not hub_dir.is_dir():
            continue
        seen.add(hub_dir)
        for repo_dir in sorted(hub_dir.glob("models--*")):
            snap_root = repo_dir / "snapshots"
            if not snap_root.is_dir():
                continue
            snaps = sorted(
                [p for p in snap_root.iterdir() if p.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for s in snaps:
                vae_dir = s / "vae"
                if vae_dir.is_dir():
                    sf_files = [
                        sf for sf in vae_dir.glob("*.safetensors") if sf.exists()
                    ]
                    if sf_files:
                        return vae_dir.resolve()
    return None


class VizInjectorClient:
    def __init__(
        self,
        lib_path: Optional[Union[str, Path]] = None,
        prefer_vulkan: bool = True,
    ):
        self.dll_path = _locate_native_library(lib_path)
        if IS_WIN:
            for d in [
                self.dll_path.parent,
                ROOT_DIR / "bin" / "sysroot" / "bin",
                ROOT_DIR / "bin" / "ffmpeg",
            ]:
                if d.is_dir() and hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(str(d.resolve()))
                    except Exception:
                        pass
        self.lib = ctypes.CDLL(str(self.dll_path))
        self._bind_signatures()
        self.vulkan_active = bool(
            self.lib.vizinjector_init_backend(1 if prefer_vulkan else 0) == 1
        )
        self._local_vae: Optional[Flux2AutoencoderKL] = None
        self._closed: bool = False

    def __enter__(self) -> VizInjectorClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._local_vae = None
            try:
                self.lib.vizinjector_shutdown_backend()
            except Exception:
                pass
            self._closed = True

    def _bind_signatures(self) -> None:
        self.lib.vizinjector_init_backend.argtypes = [ctypes.c_int32]
        self.lib.vizinjector_init_backend.restype = ctypes.c_int32

        self.lib.vizinjector_get_backend_name.argtypes = []
        self.lib.vizinjector_get_backend_name.restype = ctypes.c_char_p

        self.lib.vizinjector_shutdown_backend.argtypes = []
        self.lib.vizinjector_shutdown_backend.restype = None

        self.lib.vizinjector_get_default_config.argtypes = [
            ctypes.POINTER(CVizInjectorConfig)
        ]
        self.lib.vizinjector_get_default_config.restype = None

        self.lib.vizinjector_process_image_bytes.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_uint32,
            ctypes.POINTER(CVizInjectorConfig),
            ctypes.POINTER(CVizInjectorOutput),
        ]
        self.lib.vizinjector_process_image_bytes.restype = ctypes.c_int32

        self.lib.vizinjector_process_rgba8.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(CVizInjectorConfig),
            ctypes.POINTER(CVizInjectorOutput),
        ]
        self.lib.vizinjector_process_rgba8.restype = ctypes.c_int32

        self.lib.vizinjector_postprocess_vae_to_rgb8.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(CVizPerceptualTelemetry),
        ]
        self.lib.vizinjector_postprocess_vae_to_rgb8.restype = ctypes.c_int32

        self.lib.vizinjector_pack_vizj_buffer.argtypes = [
            ctypes.POINTER(CVizContainerHeader),
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self.lib.vizinjector_pack_vizj_buffer.restype = ctypes.c_int32

        self.lib.vizinjector_unpack_vizj_buffer.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_uint64,
            ctypes.POINTER(CVizContainerHeader),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.vizinjector_unpack_vizj_buffer.restype = ctypes.c_int32

        self.lib.vizinjector_free_output.argtypes = [
            ctypes.POINTER(CVizInjectorOutput)
        ]
        self.lib.vizinjector_free_output.restype = None

        self.lib.vizinjector_free_buffer.argtypes = [ctypes.c_void_p]
        self.lib.vizinjector_free_buffer.restype = None

    @property
    def backend_name(self) -> str:
        raw = self.lib.vizinjector_get_backend_name()
        return raw.decode("utf-8", errors="replace") if raw else "C11 AVX2"

    def get_or_load_local_vae(
        self, device: Union[str, torch.device] = "cuda"
    ) -> Flux2AutoencoderKL:
        if self._local_vae is not None:
            return self._local_vae
        vae_dir = resolve_local_vae_dir()
        if vae_dir is None:
            raise FileNotFoundError(
                "Standalone Flux2AutoencoderKL not found in HF_HOME. Run python tools/download_vae.py first."
            )
        target_dev = (
            device if torch.cuda.is_available() else torch.device("cpu")
        )
        self._local_vae = Flux2AutoencoderKL.from_pretrained_local(
            vae_dir, device=target_dev, dtype=torch.float32
        )
        return self._local_vae

    def _build_c_config(self, req: VizInjectorRequest) -> CVizInjectorConfig:
        c_cfg = CVizInjectorConfig()
        self.lib.vizinjector_get_default_config(ctypes.byref(c_cfg))
        c_cfg.target_width = int(req.target_width)
        c_cfg.target_height = int(req.target_height)
        c_cfg.max_vit_patches = int(req.max_vit_patches)
        c_cfg.framing_mode = int(req.framing_mode_id)
        c_cfg.input_role = int(req.input_role_id)
        c_cfg.edit_mode_384_vit = 1 if req.edit_mode_384_vit else 0
        c_cfg.emit_vae_branch = 1 if req.emit_vae_branch else 0
        c_cfg.emit_vit_branch = 1 if req.emit_vit_branch else 0
        c_cfg.emit_mask_branch = 1 if req.emit_mask_branch else 0
        c_cfg.emit_wire_vizj = 1 if req.emit_wire_vizj else 0
        c_cfg.deband_8bit = 1 if req.deband_8bit else 0
        c_cfg.prefer_vulkan = 1 if req.prefer_vulkan else 0
        c_cfg.seed = int(req.seed)
        c_cfg.mtf_boost = float(req.mtf_boost)
        c_cfg.soft_knee_tau = float(req.soft_knee_tau)
        c_cfg.mask_feather_radius = float(req.mask_feather_radius)
        c_cfg.webp_quality = float(req.webp_quality)
        c_cfg.webp_lossless = 1 if req.webp_lossless else 0
        return c_cfg

    def _harvest_c_output(
        self, out: CVizInjectorOutput, req: VizInjectorRequest
    ) -> ConditionedVisionBundle:
        try:
            vae_tensor: Optional[torch.Tensor] = None
            if out.vae_planar_f32 and out.vae_Floats > 0:
                vae_tensor = torch.empty(
                    (1, 3, int(out.vae_height), int(out.vae_width)),
                    dtype=torch.float32,
                )
                ctypes.memmove(
                    vae_tensor.data_ptr(),
                    out.vae_planar_f32,
                    int(out.vae_Floats) * 4,
                )

            vit_patches: Optional[torch.Tensor] = None
            grid_thw: Optional[torch.Tensor] = None
            if out.vit_patches_f32 and out.vit_Floats > 0:
                n_p = int(out.vit_num_patches)
                vit_patches = torch.empty(
                    (n_p, 3, 2, 16, 16), dtype=torch.float32
                )
                ctypes.memmove(
                    vit_patches.data_ptr(),
                    out.vit_patches_f32,
                    int(out.vit_Floats) * 4,
                )
                grid_thw = torch.tensor(
                    [
                        [
                            int(out.grid_thw[0]),
                            int(out.grid_thw[1]),
                            int(out.grid_thw[2]),
                        ]
                    ],
                    dtype=torch.long,
                )

            latent_mask: Optional[torch.Tensor] = None
            if out.latent_mask_f32 and out.mask_Floats > 0:
                latent_mask = torch.empty(
                    (1, 1, int(out.mask_height), int(out.mask_width)),
                    dtype=torch.float32,
                )
                ctypes.memmove(
                    latent_mask.data_ptr(),
                    out.latent_mask_f32,
                    int(out.mask_Floats) * 4,
                )

            wire_bytes: Optional[bytes] = None
            if out.wire_vizj_bytes and out.wire_vizj_len > 0:
                wire_bytes = ctypes.string_at(
                    out.wire_vizj_bytes, int(out.wire_vizj_len)
                )

            telemetry = VizInjectorTelemetry(
                mean_luminance=float(out.mean_luminance),
                rms_contrast=float(out.rms_contrast),
                shannon_entropy_bits=float(out.shannon_entropy_bits),
                laplacian_sharpness=float(out.laplacian_sharpness),
            )
            backend_str = (
                f"Vulkan ({self.backend_name})"
                if out.backend_used == 1
                else "C11 AVX2"
            )
            meta = VizInjectorResponseMeta(
                orig_size=(int(out.orig_width), int(out.orig_height)),
                vae_size=(int(out.vae_width), int(out.vae_height)),
                vit_size=(int(out.vit_width), int(out.vit_height)),
                grid_thw=(
                    int(out.grid_thw[0]),
                    int(out.grid_thw[1]),
                    int(out.grid_thw[2]),
                ),
                llm_tokens=int(out.llm_tokens),
                backend_used=backend_str,
                edit_format=req.edit_format,
                input_role=req.input_role,
                reference_zero_timestep=req.reference_zero_timestep,
                reference_isolated_kv=req.reference_uses_isolated_kv_cache,
                telemetry=telemetry,
            )
            return ConditionedVisionBundle(
                vae_tensor=vae_tensor,
                vit_patches=vit_patches,
                grid_thw=grid_thw,
                llm_tokens=int(out.llm_tokens),
                latent_mask=latent_mask,
                packed_latents=None,
                vit_embeds=None,
                wire_vizj_bytes=wire_bytes,
                meta=meta,
            )
        finally:
            self.lib.vizinjector_free_output(ctypes.byref(out))

    def _ffmpeg_decode_fallback(self, file_bytes: bytes) -> np.ndarray:
        ff_bin = _locate_ffmpeg_binary()
        if ff_bin is None:
            raise RuntimeError(
                "Exotic image container encountered and hermetic ffmpeg binary is not available."
            )
        ffprobe_bin = ff_bin.with_name(
            "ffprobe.exe" if IS_WIN else "ffprobe"
        )
        proc_probe = subprocess.run(
            [
                str(ffprobe_bin),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0:s=x",
                "-",
            ],
            input=file_bytes,
            capture_output=True,
            check=True,
        )
        dims = proc_probe.stdout.decode("utf-8").strip().splitlines()[0].split("x")
        w, h = int(dims[0]), int(dims[1])
        proc_dec = subprocess.run(
            [
                str(ff_bin),
                "-v",
                "error",
                "-i",
                "pipe:0",
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgba",
                "pipe:1",
            ],
            input=file_bytes,
            capture_output=True,
            check=True,
        )
        arr = np.frombuffer(proc_dec.stdout, dtype=np.uint8)
        return arr.reshape((h, w, 4)).copy()

    def condition_rgba8(
        self,
        rgba_array: np.ndarray,
        request: Optional[VizInjectorRequest] = None,
    ) -> ConditionedVisionBundle:
        req = request if request is not None else VizInjectorRequest()
        if rgba_array.ndim != 3 or rgba_array.shape[2] not in (3, 4):
            raise ValueError(
                f"Expected HxWx3 or HxWx4 uint8 array, got {rgba_array.shape}"
            )
        h, w, ch = rgba_array.shape
        if ch == 3:
            alpha = np.full((h, w, 1), 255, dtype=np.uint8)
            rgba_array = np.concatenate([rgba_array, alpha], axis=2)
        contig = np.ascontiguousarray(rgba_array, dtype=np.uint8)
        c_ptr = contig.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        c_cfg = self._build_c_config(req)
        c_out = CVizInjectorOutput()
        rc = self.lib.vizinjector_process_rgba8(
            c_ptr, w, h, ctypes.byref(c_cfg), ctypes.byref(c_out)
        )
        if rc != 0:
            raise RuntimeError(
                f"VizInjector native RGBA8 conditioning failed with code {rc}"
            )
        return self._harvest_c_output(c_out, req)

    def condition_image(
        self,
        source: Union[str, Path, bytes, np.ndarray, Image.Image],
        request: Optional[VizInjectorRequest] = None,
        **kwargs: Any,
    ) -> ConditionedVisionBundle:
        req = (
            request.model_copy(update=kwargs)
            if request is not None
            else VizInjectorRequest(**kwargs)
        )
        if isinstance(source, np.ndarray):
            return self.condition_rgba8(source, req)
        if isinstance(source, Image.Image):
            return self.condition_rgba8(
                np.asarray(source.convert("RGBA"), dtype=np.uint8), req
            )
        if isinstance(source, (str, Path)):
            p = Path(source)
            if not p.is_file():
                raise FileNotFoundError(f"Input image not found: {p}")
            raw_bytes = p.read_bytes()
        elif isinstance(source, (bytes, bytearray, memoryview)):
            raw_bytes = bytes(source) if not isinstance(source, bytes) else source
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        if len(raw_bytes) >= _VIZJ_HEADER_SIZE and raw_bytes[:16] == _VIZJ_MAGIC:
            hdr = CVizContainerHeader.from_buffer_copy(raw_bytes[:_VIZJ_HEADER_SIZE])
            if hdr.tier in (VIZJ_TIER_VAE_LATENT, VIZJ_TIER_DUAL_LATENT):
                return self.load_vizj_bytes(raw_bytes, req)

        in_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        in_ptr = in_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        c_cfg = self._build_c_config(req)
        c_out = CVizInjectorOutput()
        rc = self.lib.vizinjector_process_image_bytes(
            in_ptr, len(raw_bytes), ctypes.byref(c_cfg), ctypes.byref(c_out)
        )
        if rc != 0:
            rgba_fallback = self._ffmpeg_decode_fallback(raw_bytes)
            return self.condition_rgba8(rgba_fallback, req)
        return self._harvest_c_output(c_out, req)

    @torch.inference_mode()
    def latentize(
        self,
        source: Union[
            str, Path, bytes, np.ndarray, Image.Image, ConditionedVisionBundle
        ],
        vae_module: Optional[Any] = None,
        vision_module: Optional[Any] = None,
        request: Optional[VizInjectorRequest] = None,
        **kwargs: Any,
    ) -> ConditionedVisionBundle:
        req = (
            request.model_copy(update=kwargs)
            if request is not None
            else VizInjectorRequest(**kwargs)
        )
        if isinstance(source, ConditionedVisionBundle):
            bundle = source
        else:
            bundle = self.condition_image(source, req)

        active_vae = (
            vae_module if vae_module is not None else self.get_or_load_local_vae()
        )
        if bundle.vae_tensor is not None and active_vae is not None:
            vae_dev = next(active_vae.parameters()).device
            vae_dtype = next(active_vae.parameters()).dtype
            inp = bundle.vae_tensor.to(device=vae_dev, dtype=vae_dtype)
            raw_lat = active_vae.encode_to_packed_latents(inp)
            bundle.packed_latents = subphase_radial_clamp(
                raw_lat, max_sigma=req.latent_clamp_sigma
            )

        if (
            bundle.vit_patches is not None
            and bundle.grid_thw is not None
            and vision_module is not None
        ):
            vis_dev = next(vision_module.parameters()).device
            vis_dtype = next(vision_module.parameters()).dtype
            p_in = bundle.vit_patches.to(device=vis_dev, dtype=vis_dtype)
            g_in = bundle.grid_thw.to(device=vis_dev)
            vis_out = vision_module(p_in, g_in)
            bundle.vit_embeds = vis_out.pooler_output.detach()

        return bundle

    def save_vizj(
        self,
        bundle: ConditionedVisionBundle,
        dest_path: Optional[Union[str, Path]] = None,
        tier: int = VIZJ_TIER_VAE_LATENT,
        use_bf16: bool = True,
    ) -> bytes:
        hdr = CVizContainerHeader()
        hdr.magic = _VIZJ_MAGIC
        hdr.version = 1
        hdr.tier = int(tier)
        hdr.orig_width = int(bundle.meta.orig_size[0])
        hdr.orig_height = int(bundle.meta.orig_size[1])

        vae_w = int(bundle.meta.vae_size[0])
        vae_h = int(bundle.meta.vae_size[1])
        if bundle.packed_latents is not None and bundle.packed_latents.ndim == 4:
            vae_h = int(bundle.packed_latents.shape[2]) * 16
            vae_w = int(bundle.packed_latents.shape[3]) * 16
        elif bundle.vae_tensor is not None and bundle.vae_tensor.ndim == 4:
            vae_h = int(bundle.vae_tensor.shape[2])
            vae_w = int(bundle.vae_tensor.shape[3])
        hdr.vae_width = vae_w
        hdr.vae_height = vae_h

        vit_w = int(bundle.meta.vit_size[0])
        vit_h = int(bundle.meta.vit_size[1])
        if bundle.grid_thw is not None and bundle.grid_thw.numel() >= 3:
            g_flat = bundle.grid_thw.detach().cpu().view(-1).tolist()
            hdr.grid_thw[0] = int(g_flat[0])
            hdr.grid_thw[1] = int(g_flat[1])
            hdr.grid_thw[2] = int(g_flat[2])
            if vit_w == 0 or vit_h == 0:
                vit_h = int(g_flat[1]) * 16
                vit_w = int(g_flat[2]) * 16
        else:
            hdr.grid_thw[0] = int(bundle.meta.grid_thw[0])
            hdr.grid_thw[1] = int(bundle.meta.grid_thw[1])
            hdr.grid_thw[2] = int(bundle.meta.grid_thw[2])
        hdr.vit_width = vit_w
        hdr.vit_height = vit_h

        if bundle.vit_embeds is not None and bundle.vit_embeds.ndim == 2:
            hdr.llm_tokens = int(bundle.vit_embeds.shape[0])
        else:
            hdr.llm_tokens = int(bundle.llm_tokens)

        hdr.mean_luminance = float(bundle.meta.telemetry.mean_luminance)
        hdr.rms_contrast = float(bundle.meta.telemetry.rms_contrast)
        hdr.shannon_entropy_bits = float(
            bundle.meta.telemetry.shannon_entropy_bits
        )
        hdr.laplacian_sharpness = float(
            bundle.meta.telemetry.laplacian_sharpness
        )

        t_vae_cpu: Optional[torch.Tensor] = None
        vae_ptr_arg: Optional[ctypes.c_void_p] = None
        vae_nbytes = 0
        if bundle.packed_latents is not None:
            if use_bf16:
                hdr.vae_dtype = VIZJ_DTYPE_BF16
                t_vae_cpu = (
                    bundle.packed_latents.detach()
                    .to(device="cpu", dtype=torch.bfloat16)
                    .contiguous()
                )
            else:
                hdr.vae_dtype = VIZJ_DTYPE_FP32
                t_vae_cpu = (
                    bundle.packed_latents.detach()
                    .to(device="cpu", dtype=torch.float32)
                    .contiguous()
                )
            vae_ptr_arg = ctypes.c_void_p(t_vae_cpu.data_ptr())
            vae_nbytes = t_vae_cpu.numel() * t_vae_cpu.element_size()

        t_vit_cpu: Optional[torch.Tensor] = None
        vit_ptr_arg: Optional[ctypes.c_void_p] = None
        vit_nbytes = 0
        if tier == VIZJ_TIER_DUAL_LATENT and bundle.vit_embeds is not None:
            hdr.vit_dtype = VIZJ_DTYPE_BF16
            t_vit_cpu = (
                bundle.vit_embeds.detach()
                .to(device="cpu", dtype=torch.bfloat16)
                .contiguous()
            )
            vit_ptr_arg = ctypes.c_void_p(t_vit_cpu.data_ptr())
            vit_nbytes = t_vit_cpu.numel() * t_vit_cpu.element_size()

        t_mask_cpu: Optional[torch.Tensor] = None
        mask_ptr_arg: Optional[ctypes.c_void_p] = None
        mask_nbytes = 0
        if bundle.latent_mask is not None:
            t_mask_cpu = (
                bundle.latent_mask.detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
            )
            mask_ptr_arg = ctypes.c_void_p(t_mask_cpu.data_ptr())
            mask_nbytes = t_mask_cpu.numel() * t_mask_cpu.element_size()

        out_ptr = ctypes.POINTER(ctypes.c_uint8)()
        out_len = ctypes.c_uint64(0)
        rc = self.lib.vizinjector_pack_vizj_buffer(
            ctypes.byref(hdr),
            vae_ptr_arg,
            vae_nbytes,
            vit_ptr_arg,
            vit_nbytes,
            mask_ptr_arg,
            mask_nbytes,
            ctypes.byref(out_ptr),
            ctypes.byref(out_len),
        )
        if rc != 0 or not out_ptr:
            raise RuntimeError(f"Failed packing .vizj container (code {rc})")
        try:
            blob = ctypes.string_at(out_ptr, int(out_len.value))
        finally:
            self.lib.vizinjector_free_buffer(out_ptr)

        if dest_path is not None:
            p = Path(dest_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(blob)
        return blob

    def load_vizj_bytes(
        self,
        container_bytes: bytes,
        request: Optional[VizInjectorRequest] = None,
    ) -> ConditionedVisionBundle:
        req = request if request is not None else VizInjectorRequest()
        in_arr = np.frombuffer(container_bytes, dtype=np.uint8)
        in_ptr = in_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        hdr = CVizContainerHeader()
        vae_ptr = ctypes.c_void_p()
        vit_ptr = ctypes.c_void_p()
        mask_ptr = ctypes.c_void_p()
        rc = self.lib.vizinjector_unpack_vizj_buffer(
            in_ptr,
            len(container_bytes),
            ctypes.byref(hdr),
            ctypes.byref(vae_ptr),
            ctypes.byref(vit_ptr),
            ctypes.byref(mask_ptr),
        )
        if rc != 0:
            raise ValueError(
                f"Corrupted or invalid .vizj container (error code {rc})"
            )

        if hdr.tier == VIZJ_TIER_WIRE_WEBP:
            wire_req = req.model_copy(
                update={
                    "target_width": (
                        int(hdr.vae_width) if hdr.vae_width > 0 else req.target_width
                    ),
                    "target_height": (
                        int(hdr.vae_height) if hdr.vae_height > 0 else req.target_height
                    ),
                    "deband_8bit": False,
                    "mtf_boost": 0.0,
                }
            )
            return self.condition_image(container_bytes, wire_req)

        lat_h = int(hdr.vae_height) // 16
        lat_w = int(hdr.vae_width) // 16
        if (lat_h <= 0 or lat_w <= 0) and hdr.vae_bytes > 0:
            elem_sz = 2 if hdr.vae_dtype in (VIZJ_DTYPE_BF16, VIZJ_DTYPE_FP16) else 4
            spatial_cells = int(hdr.vae_bytes) // (128 * elem_sz)
            side = int(math.isqrt(max(1, spatial_cells)))
            if side * side == spatial_cells:
                lat_h = side
                lat_w = side
                hdr.vae_height = lat_h * 16
                hdr.vae_width = lat_w * 16

        packed_latents: Optional[torch.Tensor] = None
        if vae_ptr.value and hdr.vae_bytes > 0 and lat_h > 0 and lat_w > 0:
            if hdr.vae_dtype == VIZJ_DTYPE_BF16:
                t_bf16 = torch.empty(
                    (1, 128, lat_h, lat_w), dtype=torch.bfloat16
                )
                if t_bf16.numel() * t_bf16.element_size() == int(hdr.vae_bytes):
                    ctypes.memmove(
                        t_bf16.data_ptr(), vae_ptr.value, int(hdr.vae_bytes)
                    )
                    packed_latents = t_bf16.float()
            else:
                packed_latents = torch.empty(
                    (1, 128, lat_h, lat_w), dtype=torch.float32
                )
                if packed_latents.numel() * packed_latents.element_size() == int(
                    hdr.vae_bytes
                ):
                    ctypes.memmove(
                        packed_latents.data_ptr(), vae_ptr.value, int(hdr.vae_bytes)
                    )

        vit_embeds: Optional[torch.Tensor] = None
        if vit_ptr.value and hdr.vit_bytes > 0:
            n_tok = max(1, int(hdr.llm_tokens))
            embed_dim = int(hdr.vit_bytes) // (2 * n_tok)
            if embed_dim > 0:
                vit_embeds = torch.empty((n_tok, embed_dim), dtype=torch.bfloat16)
                ctypes.memmove(
                    vit_embeds.data_ptr(), vit_ptr.value, int(hdr.vit_bytes)
                )

        latent_mask: Optional[torch.Tensor] = None
        if mask_ptr.value and hdr.mask_bytes > 0 and lat_h > 0 and lat_w > 0:
            latent_mask = torch.empty(
                (1, 1, lat_h, lat_w), dtype=torch.float32
            )
            if latent_mask.numel() * latent_mask.element_size() == int(hdr.mask_bytes):
                ctypes.memmove(
                    latent_mask.data_ptr(), mask_ptr.value, int(hdr.mask_bytes)
                )

        vit_w = int(hdr.vit_width)
        vit_h = int(hdr.vit_height)
        if (
            (vit_w == 0 or vit_h == 0)
            and int(hdr.grid_thw[1]) > 0
            and int(hdr.grid_thw[2]) > 0
        ):
            vit_h = int(hdr.grid_thw[1]) * 16
            vit_w = int(hdr.grid_thw[2]) * 16

        grid_thw = torch.tensor(
            [[int(hdr.grid_thw[0]), int(hdr.grid_thw[1]), int(hdr.grid_thw[2])]],
            dtype=torch.long,
        )
        telem = VizInjectorTelemetry(
            mean_luminance=float(hdr.mean_luminance),
            rms_contrast=float(hdr.rms_contrast),
            shannon_entropy_bits=float(hdr.shannon_entropy_bits),
            laplacian_sharpness=float(hdr.laplacian_sharpness),
        )
        meta = VizInjectorResponseMeta(
            orig_size=(int(hdr.orig_width), int(hdr.orig_height)),
            vae_size=(int(hdr.vae_width), int(hdr.vae_height)),
            vit_size=(vit_w, vit_h),
            grid_thw=(
                int(hdr.grid_thw[0]),
                int(hdr.grid_thw[1]),
                int(hdr.grid_thw[2]),
            ),
            llm_tokens=int(hdr.llm_tokens),
            backend_used="VIZJ Zero-Copy Cache",
            edit_format=req.edit_format,
            input_role=req.input_role,
            reference_zero_timestep=req.reference_uses_zero_timestep,
            reference_isolated_kv=req.reference_uses_isolated_kv_cache,
            telemetry=telem,
        )
        return ConditionedVisionBundle(
            vae_tensor=None,
            vit_patches=None,
            grid_thw=grid_thw,
            llm_tokens=int(hdr.llm_tokens),
            latent_mask=latent_mask,
            packed_latents=packed_latents,
            vit_embeds=vit_embeds,
            wire_vizj_bytes=None,
            meta=meta,
        )

    def postprocess_to_pil(
        self,
        decoded_tensor: torch.Tensor,
        apply_bayer_dither: bool = True,
    ) -> Tuple[List[Image.Image], List[VizInjectorTelemetry]]:
        if decoded_tensor.ndim == 3:
            decoded_tensor = decoded_tensor.unsqueeze(0)
        b, c, h, w = decoded_tensor.shape
        if c != 3:
            raise ValueError(
                f"Expected 3-channel RGB tensor, got shape {decoded_tensor.shape}"
            )
        cpu_f32 = decoded_tensor.detach().cpu().float().contiguous().numpy()
        pil_list: List[Image.Image] = []
        telem_list: List[VizInjectorTelemetry] = []
        for i in range(b):
            sample = np.ascontiguousarray(cpu_f32[i], dtype=np.float32)
            out_rgb8 = np.empty((h, w, 3), dtype=np.uint8)
            c_telem = CVizPerceptualTelemetry()
            rc = self.lib.vizinjector_postprocess_vae_to_rgb8(
                sample.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                w,
                h,
                1 if apply_bayer_dither else 0,
                out_rgb8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.byref(c_telem),
            )
            if rc != 0:
                raise RuntimeError(f"Native postprocessing failed with code {rc}")
            pil_list.append(Image.fromarray(out_rgb8, mode="RGB"))
            telem_list.append(
                VizInjectorTelemetry(
                    mean_luminance=float(c_telem.mean_luminance),
                    rms_contrast=float(c_telem.rms_contrast),
                    shannon_entropy_bits=float(c_telem.shannon_entropy_bits),
                    laplacian_sharpness=float(c_telem.laplacian_sharpness),
                )
            )
        return pil_list, telem_list