from __future__ import annotations

from pathlib import Path
import random
import sys
import time
from typing import Any, Callable, Optional, Union

import numpy as np
import torch

from models.modeling_moss import MOSSImageTransformer
from models.modeling_qwen35 import Qwen3_5Model
from models.modeling_vae import Flux2AutoencoderKL
from models.tokenizer import MOSSImageTokenizer
from schema import GenerationRequest, GenerationResponse, SUPPORTED_MODELS
from ..caches import PromptEmbedCache, ReferenceVisionCache
from ..env import (
    HAS_VIZINJECTOR,
    ROOT_DIR,
    VizInjectorClient,
    _canonicalize_device,
)
from ..memory import WDDMMemoryGuard
from ..telemetry import EngineProgressReporter
from .lifecycle import PipelineLifecycleMixin
from .stages import PipelineStagesMixin


class MOSSImageEngine(PipelineLifecycleMixin, PipelineStagesMixin):
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
            effective_cb: Callable[[str, int, int], None] = reporter
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

        vis_state = self._execute_stage_1_visual_conditioning(
            request=request,
            cpu_offload=cpu_offload,
        )
        ctx_state = self._execute_stage_2_context_encoding(
            request=request,
            vis_state=vis_state,
            cpu_offload=cpu_offload,
            reporter=reporter,
            effective_cb=effective_cb,
        )
        flow_state = self._execute_stage_3_flow_matching(
            request=request,
            explicit_seed=explicit_seed,
            vis_state=vis_state,
            ctx_state=ctx_state,
            cpu_offload=cpu_offload,
            reporter=reporter,
            effective_cb=effective_cb,
        )
        return self._execute_stage_4_decode_and_export(
            request=request,
            target_repo=target_repo,
            explicit_seed=explicit_seed,
            start_time=start_time,
            cpu_offload=cpu_offload,
            lora_active=lora_active,
            vis_state=vis_state,
            ctx_state=ctx_state,
            flow_state=flow_state,
            reporter=reporter,
            effective_cb=effective_cb,
        )


__all__ = ["MOSSImageEngine"]