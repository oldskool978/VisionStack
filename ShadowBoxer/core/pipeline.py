# ShadowBoxer/core/pipeline.py
from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
from safetensors.torch import load_file as load_safetensors
import torch
import torch.nn.functional as F

from models.modeling_moss import (
    MOSSImageConfig,
    MOSSImageTransformer,
    validate_skip_layers,
)
from models.modeling_qwen35 import Qwen3_5HybridCache, Qwen3_5Model
from models.modeling_vae import Flux2AutoencoderKL
from models.tokenizer import MOSSImageTokenizer
from schema import GenerationRequest, GenerationResponse, SUPPORTED_MODELS
from .caches import CachedReferenceRecord, PromptEmbedCache, ReferenceVisionCache
from .conditioning import (
    _normalize_vizinjector_framing,
    _resolve_boxed_reference_dims,
    _resolve_unboxed_reference_dims,
    build_box_aperture_mask,
    compose_latent_inpaint_mask,
    couple_splat_velocity_prior,
    load_image_tensor_16px,
    pack_vision_patches_32px,
    resample_packed_latents,
    splat_box_latents_to_canvas,
)
from .env import (
    ConditionedVisionBundle,
    HAS_VIZINJECTOR,
    ROOT_DIR,
    VIZJ_TIER_DUAL_LATENT,
    VizInjectorClient,
    VizInjectorRequest,
    _canonicalize_device,
    _evacuate_to_cpu,
)
from .loader import (
    _MetaWeightInitMode,
    load_sharded_safetensors_into,
    resolve_model_path,
    stream_safetensors_inplace,
)
from .memory import WDDMMemoryGuard, WDDMMemorySnapshot
from .prng import (
    _GENERATOR_CACHE,
    _MASK63,
    derive_stage_keys,
    philox_randn,
    subphase_radial_clamp,
)
from .schedulers import BifurcatedFlowMatchScheduler, MOSSImageScheduler
from .telemetry import EngineProgressReporter, compute_image_telemetry


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
        self, force: bool = False, required_headroom_gb: float = 1.25
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
        if self.vae is not None:
            self.vae.clear_caches()
        if torch.cuda.is_available():
            gc.collect()
            self._sync_device()
            torch.cuda.empty_cache()
        self.memory_guard.sync_pinned_baseline()

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
            if self.vae is not None:
                try:
                    self.vae.clear_caches()
                except Exception:
                    pass
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
                self.memory_guard.sync_pinned_baseline()
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
                    gc.collect()
                    self._trim_reserved_vram(force=False, required_headroom_gb=1.25)
                    self.memory_guard.sync_pinned_baseline()
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
                self.memory_guard.sync_pinned_baseline()
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
                self.memory_guard.sync_pinned_baseline()
                return False

            lora_sd: Dict[str, torch.Tensor] = {}
            for sf in sf_files:
                lora_sd.update(load_safetensors(str(sf), device="cpu"))

            fused = self.transformer.fuse_lora_state_dict(
                lora_sd, scale=edit_lora_scale, lora_tag=lora_tag_str
            )
            del lora_sd
            self._trim_reserved_vram(force=False, required_headroom_gb=1.25)
            self.memory_guard.sync_pinned_baseline()
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
                self.memory_guard.sync_pinned_baseline()
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
                    if (
                        bundle.vae_tensor.shape[-2] * bundle.vae_tensor.shape[-1]
                        >= 768 * 768
                    ):
                        self._trim_reserved_vram(force=True, required_headroom_gb=3.80)
                    raw_lat = self.vae.encode_to_packed_latents(bundle.vae_tensor)
                    clamped_lat = subphase_radial_clamp(
                        raw_lat, max_sigma=request.latent_clamp_sigma
                    )
                    packed_lat = _evacuate_to_cpu(clamped_lat)
                    del raw_lat, clamped_lat
                    bundle.vae_tensor = None
                    self._trim_reserved_vram(force=True, required_headroom_gb=3.80)
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
                self._trim_reserved_vram(force=True, required_headroom_gb=3.80)

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
                if eff_h * eff_w >= 768 * 768:
                    self._trim_reserved_vram(force=True, required_headroom_gb=3.80)
                raw_lat = self.vae.encode_to_packed_latents(img_t)
                clamped_lat = subphase_radial_clamp(
                    raw_lat, max_sigma=request.latent_clamp_sigma
                )
                packed_lat = _evacuate_to_cpu(clamped_lat)
                del img_t, raw_lat, clamped_lat
                self._trim_reserved_vram(force=True, required_headroom_gb=3.80)
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
                self._trim_reserved_vram(force=True, required_headroom_gb=3.80)

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
            self.memory_guard.sync_pinned_baseline()
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

        if use_cache and not is_negative:
            spliced = self.embed_cache.lookup_prefix_splice(
                tok_list,
                te_device,
                vision_signature=vision_signature,
            )
            if spliced is not None:
                del pixel_values, image_grid_thw, pre_vit_embeds
                prefix_cache, prefix_hidden, snap_len = spliced
                suffix_ids = torch.tensor(
                    [tok_list[snap_len:]], dtype=torch.long, device=te_device
                )
                self.text_encoder.rope_deltas = prefix_cache.rope_deltas
                suffix_out = self.text_encoder(
                    input_ids=suffix_ids,
                    past_key_values=prefix_cache,
                    use_cache=False,
                )
                full_hidden = torch.cat(
                    [prefix_hidden, suffix_out.last_hidden_state], dim=1
                )
                hidden_cpu = _evacuate_to_cpu(full_hidden)
                del suffix_ids, suffix_out, prefix_cache, prefix_hidden, full_hidden
                if hidden_cpu is not None:
                    self.embed_cache.insert(
                        tok_list, hidden_cpu, vision_signature=vision_signature
                    )
                    self.memory_guard.sync_pinned_baseline()
                hidden_out = hidden_cpu.to(device=out_device, non_blocking=True)
                mask_out = torch.ones(
                    (1, hidden_out.shape[1]), dtype=torch.bool, device=out_device
                )
                return hidden_out, mask_out

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

        total_seq_len = len(tok_list)
        vis_end_id = self.text_encoder.config.vision_end_token_id
        min_snap = 0
        for idx_t, tid in enumerate(tok_list):
            if tid == vis_end_id:
                min_snap = idx_t + 1

        can_snapshot_prefix = (
            use_cache
            and not is_negative
            and total_seq_len >= 24
            and (total_seq_len - min_snap) >= 16
        )
        traj_cache = (
            Qwen3_5HybridCache(
                record_checkpoints=True,
                min_checkpoint_token=min_snap,
            )
            if can_snapshot_prefix
            else None
        )

        output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            past_key_values=traj_cache,
            use_cache=can_snapshot_prefix,
        )
        hidden_cpu = _evacuate_to_cpu(output.last_hidden_state)
        hybrid_state = output.past_key_values
        del (
            output,
            input_ids,
            attention_mask,
            inputs_embeds,
            pixel_values,
            image_grid_thw,
        )

        if use_cache and hidden_cpu is not None:
            pinned_hidden = self.embed_cache.insert(
                tok_list, hidden_cpu, vision_signature=vision_signature
            )
            if can_snapshot_prefix and hybrid_state is not None:
                self.embed_cache.insert_prefix_state(
                    tok_list,
                    pinned_hidden if pinned_hidden is not None else hidden_cpu,
                    hybrid_state,
                    vision_signature=vision_signature,
                )
            self.memory_guard.sync_pinned_baseline()
        del hybrid_state

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
            f"  [3/3] Warmup: Flux2-VAE FP32 Encoder & Decoder ({width}x{height})...\n"
        )
        sys.stdout.flush()
        dummy_img_enc = torch.zeros(
            (1, 3, 256, 256),
            device=exec_dev,
            dtype=torch.float32,
        )
        dummy_enc = self.vae.encode_to_packed_latents(dummy_img_enc)
        del dummy_img_enc, dummy_enc
        self._sync_device()
        self._trim_reserved_vram(force=True, required_headroom_gb=1.50)

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
        paint_mask_cpu: Optional[torch.Tensor] = None
        box_mask_cpu: Optional[torch.Tensor] = None
        alpha_mask_cpu: Optional[torch.Tensor] = None
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
        has_mask_file = bool(
            request.inpaint_mask_path and Path(request.inpaint_mask_path).is_file()
        )
        has_control = bool(
            request.control_image_path and Path(request.control_image_path).is_file()
        )
        visual_ref_specs, _ = request.resolve_visual_references()
        valid_ref_specs = [
            spec for spec in visual_ref_specs if Path(spec["path"]).is_file()
        ]

        if has_init or has_mask_file or has_control or valid_ref_specs:
            if cpu_offload:
                self.vae.to(self.device)
                self.text_encoder.to(self.device)

            if has_init:
                init_p = Path(request.init_image_path)
                init_fm = (
                    "inscribed"
                    if str(request.framing_mode or "inscribed").lower() == "never_crop"
                    else request.framing_mode
                )
                init_rec, hit = self._condition_single_reference(
                    init_p,
                    request,
                    role_override="photo_ref",
                    framing_override=init_fm,
                    target_w_override=request.width,
                    target_h_override=request.height,
                )
                any_ref_cache_hit = any_ref_cache_hit or hit
                init_latents_cpu = _evacuate_to_cpu(init_rec.packed_latents)
                if init_rec.latent_mask is not None:
                    if float(torch.min(init_rec.latent_mask).item()) < 0.99:
                        alpha_mask_cpu = _evacuate_to_cpu(init_rec.latent_mask)

            if has_mask_file:
                mask_p = Path(request.inpaint_mask_path)
                if self.vizinjector is not None and VizInjectorRequest is not None:
                    mask_fm = _normalize_vizinjector_framing(request.framing_mode)
                    if mask_fm == "never_crop":
                        mask_fm = "inscribed"
                    m_req = VizInjectorRequest(
                        target_width=request.width,
                        target_height=request.height,
                        framing_mode=mask_fm,
                        input_role="inpaint_mask",
                        mask_feather_radius=request.mask_feather_radius,
                        prefer_vulkan=request.prefer_vulkan,
                    )
                    m_bundle = self.vizinjector.condition_image(mask_p, m_req)
                    if m_bundle.latent_mask is not None:
                        paint_mask_cpu = _evacuate_to_cpu(m_bundle.latent_mask)
                if paint_mask_cpu is None:
                    m_img = Image.open(mask_p).convert("L")
                    m_img = m_img.resize(
                        (request.width // 16, request.height // 16),
                        resample=Image.Resampling.BILINEAR,
                    )
                    m_arr = np.asarray(m_img, dtype=np.float32) / 255.0
                    paint_mask_cpu = (
                        torch.from_numpy(m_arr).unsqueeze(0).unsqueeze(0).contiguous()
                    )

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
                if norm_box is not None:
                    ref_w, ref_h = _resolve_boxed_reference_dims(
                        norm_box, request.width, request.height
                    )
                elif px_box is not None:
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

        if (has_init or bool(splat_entries)) and request.layout_items:
            want_box_aperture = (
                bool(request.preserve_canvas_outside_boxes)
                or str(request.mask_blend_mode or "auto").lower() == "boxes_only"
                or any(
                    isinstance(it, dict)
                    and (
                        bool(it.get("inpaint_box", False))
                        or (has_init and bool(it.get("paste_to_canvas", False)))
                    )
                    for it in request.layout_items
                )
            )
            if want_box_aperture:
                only_flagged = not (
                    bool(request.preserve_canvas_outside_boxes)
                    or str(request.mask_blend_mode or "auto").lower() == "boxes_only"
                )
                box_mask_cpu = build_box_aperture_mask(
                    layout_items=request.layout_items,
                    height=request.height,
                    width=request.width,
                    coords_are_pixels=request.layout_coords_are_pixels,
                    feather_radius_lat=float(request.mask_feather_radius or 1.25),
                    device=torch.device("cpu"),
                    only_flagged_boxes=only_flagged,
                    include_paste_boxes=has_init,
                )

        if has_init or bool(splat_entries):
            inpaint_mask_cpu = compose_latent_inpaint_mask(
                paint_mask=paint_mask_cpu,
                box_mask=box_mask_cpu,
                alpha_mask=alpha_mask_cpu,
                blend_mode=str(request.mask_blend_mode or "auto"),
                preserve_outside_boxes=bool(request.preserve_canvas_outside_boxes),
                invert_mask=bool(request.invert_inpaint_mask),
                h_lat=request.height // 16,
                w_lat=request.width // 16,
                device=torch.device("cpu"),
            )
        del paint_mask_cpu, box_mask_cpu, alpha_mask_cpu

        effective_prompt = request.compile_effective_prompt()
        pos_hits_before = self.embed_cache.pos_hits
        splice_hits_before = self.embed_cache.splice_hits

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
        cache_hit = (self.embed_cache.pos_hits > pos_hits_before) or (
            self.embed_cache.splice_hits > splice_hits_before
        )
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
                clean_init_latents = resample_packed_latents(
                    clean_init_latents, h_lat, w_lat
                )
        del init_latents_cpu

        composite_used = False
        splat_union_alpha: Optional[torch.Tensor] = None
        splat_denoise_map: Optional[torch.Tensor] = None
        eff_denoise_strength = float(request.denoise_strength)

        if splat_entries:
            composite_used = True
            default_splat_dn = (
                eff_denoise_strength if eff_denoise_strength < 0.999 else 0.75
            )
            (
                clean_init_latents,
                splat_union_alpha,
                splat_denoise_map,
            ) = splat_box_latents_to_canvas(
                base_canvas=clean_init_latents,
                splat_entries=splat_entries,
                h_lat=h_lat,
                w_lat=w_lat,
                latent_channels=self.transformer.config.latent_channels,
                feather_radius_lat=float(request.mask_feather_radius or 1.25),
                device=self.device,
                default_denoise_strength=default_splat_dn,
                return_denoise_map=True,
            )
            box_dns = [
                float(e[3]) for e in splat_entries if e[3] is not None
            ]
            if has_init and box_dns and eff_denoise_strength >= 0.999:
                eff_denoise_strength = max(
                    0.05, min(1.0, sum(box_dns) / len(box_dns))
                )
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
            if not has_init and splat_union_alpha is not None:
                bg_lock = (1.0 - inpaint_mask_tensor) * splat_union_alpha
                inpaint_mask_tensor = (1.0 - bg_lock).clamp(0.0, 1.0)
        del inpaint_mask_cpu

        start_step = 0
        if (
            has_init
            and clean_init_latents is not None
            and 0.0 < eff_denoise_strength < 1.0
        ):
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

        if (
            has_init
            and clean_init_latents is not None
            and (start_step > 0 or inpaint_mask_tensor is not None)
        ):
            sigma_0 = float(active_scheduler.sigmas[0].item())
            noised_init = (1.0 - sigma_0) * clean_init_latents + sigma_0 * noise
            if inpaint_mask_tensor is not None and start_step == 0:
                latents = (
                    inpaint_mask_tensor * noise
                    + (1.0 - inpaint_mask_tensor) * noised_init
                )
            else:
                latents = noised_init
            del noised_init
        else:
            latents = noise

        use_velocity_splat_prior = bool(
            composite_used
            and not has_init
            and clean_init_latents is not None
            and splat_union_alpha is not None
            and splat_denoise_map is not None
        )

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
                want_hist_out = (
                    active_scheduler.wants_early_history_output
                    and abs(early_cfg - late_cfg) > 1e-5
                )
            else:
                is_early = True
                step_frac = 1.0 - float(
                    active_scheduler.sigmas[min(pass_index, num_passes - 1)].item()
                )
                want_hist_out = False

            current_cfg = early_cfg if is_early else late_cfg
            do_step_cfg = abs(current_cfg - 1.0) > 1e-5
            skip_active = request.slg_start <= step_frac < request.slg_stop
            do_slg = bool(slg_skip_set and request.slg_scale > 0.0 and skip_active)
            eff_uncond_skip = (
                (uncond_skip_set - slg_skip_set) if do_slg else uncond_skip_set
            )

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
            hist_prediction: Optional[torch.Tensor] = None
            total_nfe += 1

            if do_step_cfg and negative is not None:
                skip_kwargs = (
                    {"skip_layers": eff_uncond_skip}
                    if (eff_uncond_skip and skip_active)
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
                cond_delta = prediction - uncond
                if want_hist_out:
                    hist_prediction = uncond + late_cfg * cond_delta
                prediction = uncond + current_cfg * cond_delta
                del uncond, cond_delta
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
                slg_delta = request.slg_scale * (conditional - skipped)
                prediction = prediction + slg_delta
                if hist_prediction is not None:
                    hist_prediction = hist_prediction + slg_delta
                del skipped, slg_delta
                total_nfe += 1
            del conditional

            if (
                use_velocity_splat_prior
                and clean_init_latents is not None
                and splat_union_alpha is not None
                and splat_denoise_map is not None
            ):
                sigma_eval = float(timestep.item()) / float(
                    active_scheduler.num_train_timesteps
                )
                prediction = couple_splat_velocity_prior(
                    v_dit=prediction,
                    latents=latents,
                    splat_canvas=clean_init_latents,
                    splat_union_alpha=splat_union_alpha,
                    splat_denoise_map=splat_denoise_map,
                    sigma_curr=sigma_eval,
                )
                if hist_prediction is not None:
                    hist_prediction = couple_splat_velocity_prior(
                        v_dit=hist_prediction,
                        latents=latents,
                        splat_canvas=clean_init_latents,
                        splat_union_alpha=splat_union_alpha,
                        splat_denoise_map=splat_denoise_map,
                        sigma_curr=sigma_eval,
                    )

            latents = active_scheduler.step(
                prediction,
                timestep,
                latents,
                generator=sde_gen,
                return_dict=False,
                history_output=hist_prediction,
            )
            del prediction, hist_prediction

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
            splat_union_alpha,
            splat_denoise_map,
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
            slg_layers_used=(
                list(request.slg_layers or [])
                if (request.slg_scale is not None and request.slg_scale > 0.0)
                else []
            ),
            slg_scale_used=float(request.slg_scale or 0.0),
            eta_used=request.eta,
            s_noise_used=float(request.s_noise),
            enable_thinking_used=request.enable_thinking,
            viztoken_active=True,
            vizinjector_active=self.vizinjector_active,
            vizinjector_backend=self.vizinjector_backend_name,
            prompt_cache_hit=cache_hit,
            reference_cache_hit=any_ref_cache_hit,
            common_prefix_tokens=common_prefix_len,
            edit_format_used=request.edit_format,
            framing_mode_used=request.framing_mode,
            input_role_used=request.input_role,
            mask_blend_mode_used=str(request.mask_blend_mode or "auto"),
            invert_inpaint_mask_used=bool(request.invert_inpaint_mask),
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


__all__ = ["MOSSImageEngine"]