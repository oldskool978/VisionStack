from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

from PIL import Image
from safetensors.torch import load_file as load_safetensors
import torch

from models.modeling_moss import MOSSImageConfig, MOSSImageTransformer
from models.modeling_qwen35 import Qwen3_5Model
from models.modeling_vae import Flux2AutoencoderKL
from models.tokenizer import MOSSImageTokenizer
from schema import SUPPORTED_MODELS
from ..caches import PromptEmbedCache, ReferenceVisionCache
from ..env import ROOT_DIR, _evacuate_to_cpu
from ..loader import (
    _MetaWeightInitMode,
    load_sharded_safetensors_into,
    resolve_model_path,
    stream_safetensors_inplace,
)
from ..memory import WDDMMemoryGuard, WDDMMemorySnapshot
from ..prng import _GENERATOR_CACHE


class PipelineLifecycleMixin:
    model_variant: str
    repo_id: str
    device: torch.device
    dtype: torch.dtype
    vae_dtype: torch.dtype
    tokenizer: Optional[MOSSImageTokenizer]
    text_encoder: Optional[Qwen3_5Model]
    transformer: Optional[MOSSImageTransformer]
    vae: Optional[Flux2AutoencoderKL]
    embed_cache: PromptEmbedCache
    ref_cache: ReferenceVisionCache
    memory_guard: WDDMMemoryGuard
    vizinjector: Optional[Any]
    _loaded_repo_id: Optional[str]
    _current_offload_state: Optional[bool]
    _warmed_up: bool

    def __enter__(self) -> Any:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
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
        models_info: List[Dict[str, object]] = []
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
        assert self.text_encoder is not None
        assert self.transformer is not None
        assert self.vae is not None
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
        assert dummy_ctx_cpu is not None
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


__all__ = ["PipelineLifecycleMixin"]