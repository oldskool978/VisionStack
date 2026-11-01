# services/runtime/queue.py
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import gc
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

import numpy as np
import torch

from .env import (
    ARTIFACTS_DIR,
    ARTIFACT_TTL_SECONDS,
    GenerationRequest,
    STORAGE_ROOT,
    _close_preview_tokenizer,
    get_runtime_override,
)
from .vault import _resolve_user_upload_path


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

    def shutdown(self) -> None:
        _close_preview_tokenizer()
        if self._engine is not None:
            try:
                self._engine.prepare_for_exit()
            except Exception:
                pass
        if self._memory_guard is not None:
            try:
                self._memory_guard.close()
            except Exception:
                pass

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

            art_dir = Path(get_runtime_override("ARTIFACTS_DIR", ARTIFACTS_DIR))
            output_png_path = art_dir / f"{job.job_id}.png"
            export_vizj_path: Optional[Path] = (
                (art_dir / f"{job.job_id}.vizj")
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
            _, spec_cot_tokens = gen_req.resolve_thinking_trace()

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
                reanchor_telem: Optional[Dict[str, Any]] = None
                if spec_cot_tokens > 0 and getattr(eng, "text_encoder", None) is not None:
                    raw_rt = getattr(
                        eng.text_encoder, "last_reanchor_telemetry", None
                    )
                    if isinstance(raw_rt, dict) and raw_rt:
                        reanchor_telem = dict(raw_rt)

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
                    "max_sequence_length": int(
                        gen_req.max_sequence_length or 16384
                    ),
                    "max_vit_patches": int(gen_req.max_vit_patches or 2304),
                    "enable_thinking": resp.enable_thinking_used,
                    "thinking_trace_mode": gen_req.thinking_trace,
                    "speculative_cot_tokens": spec_cot_tokens,
                    "reanchor_telemetry": reanchor_telem,
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
                        resp,
                        "mask_blend_mode_used",
                        str(gen_req.mask_blend_mode or "auto"),
                    ),
                    "invert_inpaint_mask": getattr(
                        resp,
                        "invert_inpaint_mask_used",
                        bool(gen_req.invert_inpaint_mask),
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
                    "enable_thinking": gen_req.enable_thinking,
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
                    "early_solver": gen_req.early_solver,
                    "late_solver": gen_req.late_solver,
                    "handoff_threshold": gen_req.handoff_threshold,
                    "num_inference_steps": gen_req.num_inference_steps,
                    "num_images_per_prompt": gen_req.num_images_per_prompt,
                    "shift": gen_req.shift,
                    "use_v2_grid": gen_req.use_v2_grid,
                    "early_cfg": gen_req.early_cfg,
                    "late_cfg": gen_req.late_cfg,
                    "uncond_skip_layers": gen_req.uncond_skip_layers,
                    "slg_layers": gen_req.slg_layers,
                    "slg_scale": gen_req.slg_scale,
                    "slg_start": gen_req.slg_start,
                    "slg_stop": gen_req.slg_stop,
                    "eta": gen_req.eta,
                    "s_noise": gen_req.s_noise,
                    "max_sequence_length": gen_req.max_sequence_length,
                    "max_vit_patches": gen_req.max_vit_patches,
                    "denoise_strength": gen_req.denoise_strength,
                    "edit_format": gen_req.edit_format,
                    "framing_mode": gen_req.framing_mode,
                    "input_role": gen_req.input_role,
                    "mtf_boost": gen_req.mtf_boost,
                    "soft_knee_tau": gen_req.soft_knee_tau,
                    "mask_feather_radius": gen_req.mask_feather_radius,
                    "latent_clamp_sigma": gen_req.latent_clamp_sigma,
                    "deband_8bit": gen_req.deband_8bit,
                    "apply_bayer_dither": gen_req.apply_bayer_dither,
                    "use_reference_cache": gen_req.use_reference_cache,
                    "prefer_vulkan": gen_req.prefer_vulkan,
                    "cpu_offload": gen_req.cpu_offload,
                    "prepass_warmup": gen_req.prepass_warmup,
                    "reference_zero_timestep": gen_req.reference_zero_timestep,
                    "reference_isolated_kv": gen_req.reference_isolated_kv,
                    "edit_lora_path": gen_req.edit_lora_path,
                    "edit_lora_scale": gen_req.edit_lora_scale,
                }

                working_draft = {
                    "title": req_data.get("title", "Untitled Vision Master"),
                    "model_variant": gen_req.model_variant,
                    "prompt": gen_req.prompt,
                    "negative_prompt": gen_req.negative_prompt,
                    "system_prompt": gen_req.system_prompt,
                    "thinking_trace": gen_req.thinking_trace,
                    "add_vision_id": gen_req.add_vision_id,
                    "enable_thinking": gen_req.enable_thinking,
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
                    "num_images_per_prompt": gen_req.num_images_per_prompt,
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
                    "max_sequence_length": gen_req.max_sequence_length,
                    "max_vit_patches": gen_req.max_vit_patches,
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
                    "soft_knee_tau": gen_req.soft_knee_tau,
                    "mask_feather_radius": gen_req.mask_feather_radius,
                    "latent_clamp_sigma": gen_req.latent_clamp_sigma,
                    "deband_8bit": gen_req.deband_8bit,
                    "apply_bayer_dither": gen_req.apply_bayer_dither,
                    "use_reference_cache": gen_req.use_reference_cache,
                    "prefer_vulkan": gen_req.prefer_vulkan,
                    "cpu_offload": gen_req.cpu_offload,
                    "prepass_warmup": gen_req.prepass_warmup,
                    "reference_zero_timestep": gen_req.reference_zero_timestep,
                    "reference_isolated_kv": gen_req.reference_isolated_kv,
                    "edit_lora_path": gen_req.edit_lora_path,
                    "edit_lora_scale": gen_req.edit_lora_scale,
                    "export_vizj": bool(req_data.get("export_vizj", False)),
                    "export_vizj_tier": int(req_data.get("export_vizj_tier", 2)),
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

    def stop(self) -> None:
        for task in (self.worker_task, self.cleanup_task):
            if task is not None and not task.done():
                task.cancel()
        self.pipeline.shutdown()

    def count_user_inflight_jobs(self, user_slug: str) -> int:
        return sum(
            1
            for j in self.jobs.values()
            if j.user_slug == user_slug and j.status in ("QUEUED", "PROCESSING")
        )

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
            art_dir = Path(get_runtime_override("ARTIFACTS_DIR", ARTIFACTS_DIR))
            for pat in ("*.png", "*.vizj"):
                for artifact in art_dir.glob(pat):
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

                storage_root = Path(
                    get_runtime_override("STORAGE_ROOT", STORAGE_ROOT)
                )
                user_dir = storage_root / job.user_slug
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


__all__ = [
    "SynthesisJob",
    "EnginePipeline",
    "ComputeQueue",
]