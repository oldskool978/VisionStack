# ShadowBoxer/core/pipeline/stages.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from models.modeling_moss import MOSSImageTransformer, validate_skip_layers
from models.modeling_qwen35 import Qwen3_5HybridCache, Qwen3_5Model
from models.modeling_vae import Flux2AutoencoderKL
from models.tokenizer import MOSSImageTokenizer
from schema import (
    DEFAULT_BOX_REF_ISOLATION,
    GenerationRequest,
    GenerationResponse,
)

from ..caches import CachedReferenceRecord, PromptEmbedCache, ReferenceVisionCache
from ..conditioning import (
    _normalize_vizinjector_framing,
    _resolve_boxed_reference_dims,
    _resolve_unboxed_reference_dims,
    build_box_aperture_mask,
    build_box_denoise_horizon_map,
    compose_latent_inpaint_mask,
    couple_splat_velocity_prior,
    load_image_tensor_16px,
    pack_vision_patches_32px,
    resample_packed_latents,
    splat_box_latents_to_canvas,
    splat_box_masks_to_canvas,
)
from ..env import (
    ConditionedVisionBundle,
    ROOT_DIR,
    VIZJ_TIER_DUAL_LATENT,
    VizInjectorRequest,
    _canonicalize_device,
    _evacuate_to_cpu,
)
from ..memory import WDDMMemoryGuard
from ..prng import (
    _MASK63,
    derive_stage_keys,
    philox_randn,
    subphase_radial_clamp,
)
from ..schedulers import (
    BifurcatedFlowMatchScheduler,
    MOSSImageScheduler,
    unwarp_sigma_to_progress,
    warp_denoise_to_sigma_horizon,
)
from ..telemetry import EngineProgressReporter, compute_image_telemetry
from .regimes import (
    compute_differential_inpaint_gate,
    resolve_foreground_aperture_carveouts,
    resolve_skip_layer_regimes,
)


@dataclass
class VisualConditioningStageState:
    init_latents_cpu: Optional[torch.Tensor]
    inpaint_mask_cpu: Optional[torch.Tensor]
    ref_latents_cpu_list: List[torch.Tensor]
    ref_bboxes_list: List[Optional[Tuple[float, float, float, float]]]
    ref_box_iso_list: List[float]
    ref_masks_cpu_list: List[Optional[torch.Tensor]]
    ref_roles_list: List[str]
    ref_z_orders_list: List[float]
    ref_keep_in_uncond_list: List[bool]
    splat_entries: List[Tuple[Any, ...]]
    splatted_box_indices: Set[int]
    ref_records_for_prompt: List[CachedReferenceRecord]
    conditioned_ref_telem_map: Dict[Any, Dict[str, Any]]
    any_ref_cache_hit: bool
    exported_vizj_str: Optional[str]
    has_init: bool
    is_ait_format: bool


@dataclass
class ContextEncodingStageState:
    positive: torch.Tensor
    positive_mask: torch.Tensor
    negative: Optional[torch.Tensor]
    negative_mask: Optional[torch.Tensor]
    effective_prompt: str
    cache_hit: bool
    common_prefix_len: int
    spec_cot_tokens: int
    early_cfg: float
    late_cfg: float


@dataclass
class FlowMatchingStageState:
    latents_cpu: torch.Tensor
    total_nfe: int
    early_steps: int
    late_steps: int
    dit_elapsed: float
    max_box_iso_used: float
    composite_used: bool
    ref_zero_t: bool
    ref_iso_kv: bool
    uncond_uses_refs: bool


class PipelineStagesMixin:
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
        max_patches = max(64, int(request.max_vit_patches or 2304))

        cache_key = ReferenceVisionCache.make_key(
            path=image_path,
            target_w=eff_w,
            target_h=eff_h,
            framing_mode=eff_framing,
            input_role=eff_role,
            edit_format=request.edit_format,
            max_vit_patches=max_patches,
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
        vit_key = ReferenceVisionCache.make_vit_key(
            path=image_path,
            target_w=eff_w,
            target_h=eff_h,
            framing_mode=eff_framing,
            input_role=eff_role,
            max_vit_patches=max_patches,
            mtf_boost=request.mtf_boost,
            soft_knee_tau=request.soft_knee_tau,
            deband_8bit=bool(request.deband_8bit),
            seed=int(request.seed or 42),
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

        cached_vae_tuple: Optional[
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Any]]
        ] = None
        cached_vit_tuple: Optional[
            Tuple[
                Optional[torch.Tensor],
                Optional[torch.Tensor],
                Optional[torch.Tensor],
                int,
                Optional[Any],
            ]
        ] = None

        if request.use_reference_cache:
            cached = self.ref_cache.lookup(cache_key)
            if cached is not None:
                return cached, True
            if eff_role != "inpaint_mask":
                cached_vae_tuple = self.ref_cache.lookup_vae(vae_key)
            if need_vit:
                cached_vit_tuple = self.ref_cache.lookup_vit(vit_key)

        have_cached_vae = (
            cached_vae_tuple is not None and cached_vae_tuple[0] is not None
        )
        have_cached_vit = (
            need_vit
            and cached_vit_tuple is not None
            and cached_vit_tuple[3] > 0
            and (
                cached_vit_tuple[0] is not None
                or cached_vit_tuple[1] is not None
            )
        )

        if have_cached_vae and (not need_vit or have_cached_vit):
            rec = CachedReferenceRecord(
                packed_latents=cached_vae_tuple[0],
                vit_embeds=cached_vit_tuple[0] if have_cached_vit else None,
                vit_patches=cached_vit_tuple[1] if have_cached_vit else None,
                grid_thw=cached_vit_tuple[2] if have_cached_vit else None,
                llm_tokens=int(cached_vit_tuple[3]) if have_cached_vit else 0,
                latent_mask=cached_vae_tuple[1],
                viz_meta=cached_vae_tuple[2]
                or (cached_vit_tuple[4] if have_cached_vit else None),
                source_signature=vit_key,
            )
            if request.use_reference_cache:
                rec = self.ref_cache.insert(
                    cache_key,
                    rec,
                    vae_key=vae_key,
                    vit_key=vit_key if need_vit else None,
                )
                self.memory_guard.sync_pinned_baseline()
            return rec, True

        if self.vizinjector is not None and VizInjectorRequest is not None:
            viz_req = VizInjectorRequest(
                target_width=eff_w,
                target_height=eff_h,
                max_vit_patches=max_patches,
                edit_mode_384_vit=bool(max_patches <= 576),
                framing_mode=_normalize_vizinjector_framing(eff_framing),
                input_role=eff_role,
                edit_format=request.edit_format,
                emit_vae_branch=False if have_cached_vae else None,
                emit_mask_branch=False if have_cached_vae else None,
                emit_vit_branch=(
                    False if have_cached_vit else (True if need_vit else None)
                ),
                deband_8bit=request.deband_8bit,
                prefer_vulkan=request.prefer_vulkan,
                seed=int(request.seed or 42),
                mtf_boost=request.mtf_boost,
                soft_knee_tau=request.soft_knee_tau,
                mask_feather_radius=request.mask_feather_radius,
                latent_clamp_sigma=request.latent_clamp_sigma,
            )
            bundle = self.vizinjector.condition_image(image_path, viz_req)

            if have_cached_vae and cached_vae_tuple is not None:
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
                packed_lat = (
                    _evacuate_to_cpu(bundle.packed_latents)
                    if eff_role != "inpaint_mask"
                    else None
                )
                if (
                    packed_lat is None
                    and bundle.vae_tensor is not None
                    and eff_role != "inpaint_mask"
                ):
                    assert self.vae is not None
                    if (
                        bundle.vae_tensor.shape[-2] * bundle.vae_tensor.shape[-1]
                        >= 768 * 768
                    ):
                        self._trim_reserved_vram(force=False, required_headroom_gb=3.80)  # type: ignore[attr-defined]
                    raw_lat = self.vae.encode_to_packed_latents(bundle.vae_tensor)
                    clamped_lat = subphase_radial_clamp(
                        raw_lat, max_sigma=request.latent_clamp_sigma
                    )
                    packed_lat = _evacuate_to_cpu(clamped_lat)
                    del raw_lat, clamped_lat
                    bundle.vae_tensor = None
                    self._trim_reserved_vram(force=False, required_headroom_gb=3.80)  # type: ignore[attr-defined]
                latent_mask = _evacuate_to_cpu(bundle.latent_mask)

            if have_cached_vit and cached_vit_tuple is not None:
                vit_embeds = cached_vit_tuple[0]
                vit_patches_cpu = cached_vit_tuple[1]
                grid_thw_cpu = cached_vit_tuple[2]
                eff_llm_tokens = int(cached_vit_tuple[3])
                cached_vit_meta = cached_vit_tuple[4]
                if bundle.meta is not None and cached_vit_meta is not None:
                    bundle.meta.vit_size = getattr(
                        cached_vit_meta, "vit_size", bundle.meta.vit_size
                    )
                    bundle.meta.grid_thw = getattr(
                        cached_vit_meta, "grid_thw", bundle.meta.grid_thw
                    )
                    bundle.meta.llm_tokens = eff_llm_tokens
            else:
                vit_embeds = _evacuate_to_cpu(bundle.vit_embeds)
                vit_patches_cpu = _evacuate_to_cpu(bundle.vit_patches)
                grid_thw_cpu = _evacuate_to_cpu(bundle.grid_thw)
                eff_llm_tokens = int(bundle.llm_tokens) if need_vit else 0

                if (
                    need_vit
                    and vit_embeds is None
                    and vit_patches_cpu is None
                    and packed_lat is not None
                    and self.vae is not None
                ):
                    dec_rgb = self.vae.decode_packed_latents(
                        packed_lat.to(device=self.vae.device, non_blocking=True)
                    )
                    dec_rgb_cpu = _evacuate_to_cpu(dec_rgb)
                    del dec_rgb
                    assert dec_rgb_cpu is not None
                    self._trim_reserved_vram(force=False, required_headroom_gb=3.80)  # type: ignore[attr-defined]
                    vit_upgrade_req = VizInjectorRequest(
                        target_width=eff_w,
                        target_height=eff_h,
                        max_vit_patches=max_patches,
                        edit_mode_384_vit=bool(max_patches <= 576),
                        framing_mode="inscribed",
                        input_role="photo_ref",
                        edit_format=request.edit_format,
                        emit_vae_branch=False,
                        emit_mask_branch=False,
                        emit_vit_branch=True,
                        deband_8bit=False,
                        prefer_vulkan=request.prefer_vulkan,
                        seed=int(request.seed or 42),
                        mtf_boost=0.0,
                        soft_knee_tau=request.soft_knee_tau,
                    )
                    vit_bundle = self.vizinjector.condition_image(
                        dec_rgb_cpu, vit_upgrade_req
                    )
                    del dec_rgb_cpu
                    vit_patches_cpu = _evacuate_to_cpu(vit_bundle.vit_patches)
                    grid_thw_cpu = _evacuate_to_cpu(vit_bundle.grid_thw)
                    eff_llm_tokens = int(vit_bundle.llm_tokens)
                    if bundle.meta is not None and vit_bundle.meta is not None:
                        bundle.meta.vit_size = vit_bundle.meta.vit_size
                        bundle.meta.grid_thw = vit_bundle.meta.grid_thw
                        bundle.meta.llm_tokens = vit_bundle.meta.llm_tokens

                if (
                    vit_embeds is None
                    and need_vit
                    and vit_patches_cpu is not None
                    and grid_thw_cpu is not None
                ):
                    assert self.text_encoder is not None
                    img_splits = self.text_encoder.get_image_features(
                        vit_patches_cpu,
                        grid_thw_cpu.to(
                            device=self.text_encoder.visual.device, non_blocking=True
                        ),
                    )
                    vit_embeds = _evacuate_to_cpu(torch.cat(img_splits, dim=0))
                    del img_splits
                    self.text_encoder.visual.clear_caches()
                    self._trim_reserved_vram(force=False, required_headroom_gb=3.80)  # type: ignore[attr-defined]
                if vit_embeds is not None:
                    eff_llm_tokens = int(vit_embeds.shape[0])
                elif vit_patches_cpu is None:
                    eff_llm_tokens = 0

            rec = CachedReferenceRecord(
                packed_latents=packed_lat,
                vit_embeds=vit_embeds,
                vit_patches=vit_patches_cpu,
                grid_thw=grid_thw_cpu,
                llm_tokens=eff_llm_tokens if need_vit else 0,
                latent_mask=latent_mask,
                viz_meta=bundle.meta,
                source_signature=vit_key,
            )
            was_hit = bool(have_cached_vae or have_cached_vit)
        else:
            if eff_role == "inpaint_mask":
                m_img = Image.open(image_path).convert("L")
                m_img = m_img.resize(
                    (max(1, eff_w // 16), max(1, eff_h // 16)),
                    resample=Image.Resampling.BILINEAR,
                )
                m_arr = np.asarray(m_img, dtype=np.float32) / 255.0
                mask_t = _evacuate_to_cpu(
                    torch.from_numpy(m_arr).unsqueeze(0).unsqueeze(0).contiguous()
                )
                rec = CachedReferenceRecord(
                    packed_latents=None,
                    vit_embeds=None,
                    vit_patches=None,
                    grid_thw=None,
                    llm_tokens=0,
                    latent_mask=mask_t,
                    viz_meta=None,
                    source_signature=vit_key,
                )
                if request.use_reference_cache:
                    rec = self.ref_cache.insert(
                        cache_key, rec, vae_key=None, vit_key=None
                    )
                return rec, False

            if have_cached_vae and cached_vae_tuple is not None:
                packed_lat = cached_vae_tuple[0]
                was_hit = True
            else:
                assert self.vae is not None
                img_t = load_image_tensor_16px(image_path, eff_h, eff_w)
                if eff_h * eff_w >= 768 * 768:
                    self._trim_reserved_vram(force=False, required_headroom_gb=3.80)  # type: ignore[attr-defined]
                raw_lat = self.vae.encode_to_packed_latents(img_t)
                clamped_lat = subphase_radial_clamp(
                    raw_lat, max_sigma=request.latent_clamp_sigma
                )
                packed_lat = _evacuate_to_cpu(clamped_lat)
                del img_t, raw_lat, clamped_lat
                self._trim_reserved_vram(force=False, required_headroom_gb=3.80)  # type: ignore[attr-defined]
                was_hit = False

            if have_cached_vit and cached_vit_tuple is not None:
                vit_embeds = cached_vit_tuple[0]
                vit_patches = cached_vit_tuple[1]
                grid_thw = cached_vit_tuple[2]
                llm_tokens = int(cached_vit_tuple[3])
                was_hit = True
            else:
                vit_embeds = None
                vit_patches = None
                grid_thw = None
                llm_tokens = 0
                if need_vit:
                    assert self.text_encoder is not None
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
                    self._trim_reserved_vram(force=False, required_headroom_gb=3.80)  # type: ignore[attr-defined]

            rec = CachedReferenceRecord(
                packed_latents=packed_lat,
                vit_embeds=vit_embeds,
                vit_patches=vit_patches,
                grid_thw=grid_thw,
                llm_tokens=llm_tokens,
                latent_mask=None,
                viz_meta=None,
                source_signature=vit_key,
            )

        if request.use_reference_cache:
            rec = self.ref_cache.insert(
                cache_key,
                rec,
                vae_key=vae_key if eff_role != "inpaint_mask" else None,
                vit_key=vit_key if need_vit else None,
            )
            self.memory_guard.sync_pinned_baseline()
        return rec, was_hit

    @torch.inference_mode()
    def encode_prompt(
        self,
        prompt: str,
        max_sequence_length: int = 16384,
        enable_thinking: bool = True,
        device: Optional[torch.device] = None,
        use_cache: bool = True,
        is_negative: bool = False,
        vision_reference_paths: Optional[Sequence[str]] = None,
        prebuilt_vision_records: Optional[Sequence[CachedReferenceRecord]] = None,
        system_prompt: Optional[str] = None,
        thinking_trace: Optional[str] = None,
        add_vision_id: bool = True,
        speculative_cot_tokens: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.text_encoder is not None
        assert self.tokenizer is not None
        te_device = next(self.text_encoder.parameters()).device
        out_device = _canonicalize_device(device) if device is not None else te_device

        valid_recs: List[CachedReferenceRecord] = []
        ordered_path_items: List[
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]
        ] = []
        patch_tensors: List[torch.Tensor] = []
        grid_tensors: List[torch.Tensor] = []
        patch_counts: Optional[List[int]] = None
        vision_sig_list: List[Any] = []

        if prebuilt_vision_records:
            valid_recs = [
                r
                for r in prebuilt_vision_records
                if r.llm_tokens > 0
                and r.grid_thw is not None
                and (r.vit_embeds is not None or r.vit_patches is not None)
            ]
            if valid_recs:
                patch_counts = [int(r.llm_tokens) for r in valid_recs]
                for r in valid_recs:
                    vision_sig_list.append(
                        (r.source_signature, int(r.llm_tokens))
                    )
        elif vision_reference_paths:
            patch_counts = []
            for p_str in vision_reference_paths:
                p = Path(p_str)
                if p.is_file():
                    st = p.stat()
                    if self.vizinjector is not None and VizInjectorRequest is not None:
                        v_req = VizInjectorRequest(
                            max_vit_patches=2304,
                            edit_mode_384_vit=False,
                            emit_vae_branch=False,
                            emit_mask_branch=False,
                            emit_vit_branch=True,
                            prefer_vulkan=True,
                        )
                        v_bundle = self.vizinjector.condition_image(p, v_req)
                        if (
                            v_bundle.vit_embeds is None
                            and v_bundle.vit_patches is None
                            and v_bundle.packed_latents is not None
                            and self.vae is not None
                        ):
                            dec_rgb = self.vae.decode_packed_latents(
                                v_bundle.packed_latents.to(
                                    device=self.vae.device, non_blocking=True
                                )
                            )
                            dec_rgb_cpu = _evacuate_to_cpu(dec_rgb)
                            del dec_rgb
                            assert dec_rgb_cpu is not None
                            v_bundle = self.vizinjector.condition_image(
                                dec_rgb_cpu, v_req
                            )
                            del dec_rgb_cpu
                        if (
                            v_bundle.grid_thw is not None
                            and v_bundle.llm_tokens > 0
                            and (
                                v_bundle.vit_embeds is not None
                                or v_bundle.vit_patches is not None
                            )
                        ):
                            n_llm = int(v_bundle.llm_tokens)
                            grid_tensors.append(v_bundle.grid_thw)
                            patch_counts.append(n_llm)
                            ordered_path_items.append(
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
                    ordered_path_items.append((None, patches, g_thw))
                    vision_sig_list.append(
                        (str(p.resolve()), int(st.st_mtime_ns), int(n_llm))
                    )

        do_speculative_cot = bool(
            speculative_cot_tokens > 0
            and not is_negative
            and (thinking_trace is None or not str(thinking_trace).strip())
        )
        eff_enable_thinking = True if do_speculative_cot else bool(enable_thinking)

        tok_list = self.tokenizer.encode_chat(
            prompt=prompt,
            enable_thinking=eff_enable_thinking,
            max_sequence_length=max_sequence_length,
            image_patch_counts=patch_counts,
            system_prompt=system_prompt if not is_negative else None,
            thinking_trace=thinking_trace if not is_negative else None,
            add_vision_id=add_vision_id,
        )
        base_vision_signature = tuple(vision_sig_list)
        vision_signature = (
            base_vision_signature + (("spec_cot", int(speculative_cot_tokens)),)
            if do_speculative_cot
            else base_vision_signature
        )

        if use_cache:
            cached = self.embed_cache.lookup(
                tok_list,
                out_device,
                is_negative=is_negative,
                vision_signature=vision_signature,
            )
            if cached is not None:
                mask = torch.ones(
                    (1, cached.shape[1]), dtype=torch.bool, device=out_device
                )
                return cached, mask

        if use_cache and not is_negative and not do_speculative_cot:
            spliced = self.embed_cache.lookup_prefix_splice(
                tok_list,
                te_device,
                vision_signature=vision_signature,
            )
            if spliced is not None:
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
                assert hidden_cpu is not None
                del suffix_ids, suffix_out, prefix_cache, prefix_hidden, full_hidden
                self.embed_cache.insert(
                    tok_list, hidden_cpu, vision_signature=vision_signature
                )
                self.memory_guard.sync_pinned_baseline()
                hidden_out = hidden_cpu.to(device=out_device, non_blocking=True)
                mask_out = torch.ones(
                    (1, hidden_out.shape[1]), dtype=torch.bool, device=out_device
                )
                return hidden_out, mask_out

        pixel_values: Optional[torch.Tensor] = None
        image_grid_thw: Optional[torch.Tensor] = None
        pre_vit_embeds: Optional[torch.Tensor] = None

        if valid_recs:
            image_grid_thw = torch.cat(
                [r.grid_thw for r in valid_recs if r.grid_thw is not None], dim=0
            ).to(device=te_device, non_blocking=True)
            if all(r.vit_embeds is not None for r in valid_recs):
                pre_vit_embeds = torch.cat(
                    [r.vit_embeds for r in valid_recs if r.vit_embeds is not None],
                    dim=0,
                ).to(device=te_device, non_blocking=True)
            elif all(r.vit_patches is not None for r in valid_recs):
                pixel_values = torch.cat(
                    [r.vit_patches for r in valid_recs if r.vit_patches is not None],
                    dim=0,
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
        elif grid_tensors:
            image_grid_thw = torch.cat(grid_tensors, dim=0).to(
                device=te_device, non_blocking=True
            )
            if all(it[0] is not None for it in ordered_path_items):
                pre_vit_embeds = torch.cat(
                    [it[0] for it in ordered_path_items if it[0] is not None], dim=0
                ).to(device=te_device, non_blocking=True)
            elif len(patch_tensors) == len(grid_tensors):
                pixel_values = torch.cat(patch_tensors, dim=0).to(
                    device=te_device, non_blocking=True
                )
            else:
                resolved_embeds = []
                for emb_t, pat_t, g_t in ordered_path_items:
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

        im_end_id = int(self.text_encoder.config.text_config.eos_token_id or 248044)
        if do_speculative_cot:
            close_toks = self.tokenizer.trace_close_token_ids
            gen_ids, spec_hidden, spec_cache = (
                self.text_encoder.reanchor_speculative_generate(
                    input_ids=input_ids,
                    max_new_tokens=max(4, min(512, int(speculative_cot_tokens))),
                    block_size=4,
                    max_candidates=4,
                    entropy_threshold=1.25,
                    stop_token_ids=[im_end_id, 248043],
                    inputs_embeds=inputs_embeds,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    close_token_ids=close_toks,
                    return_cache=True,
                )
            )
            hidden_cpu = _evacuate_to_cpu(spec_hidden)
            assert hidden_cpu is not None
            del (
                spec_hidden,
                input_ids,
                attention_mask,
                inputs_embeds,
                pixel_values,
                image_grid_thw,
            )
            if use_cache:
                pinned_hidden = self.embed_cache.insert(
                    tok_list, hidden_cpu, vision_signature=vision_signature
                )
                full_tok_seq = list(tok_list) + list(gen_ids)
                self.embed_cache.insert(
                    full_tok_seq, hidden_cpu, vision_signature=base_vision_signature
                )
                if spec_cache is not None:
                    self.embed_cache.insert_prefix_state(
                        full_tok_seq,
                        pinned_hidden if pinned_hidden is not None else hidden_cpu,
                        spec_cache,
                        vision_signature=base_vision_signature,
                    )
                self.memory_guard.sync_pinned_baseline()
            del spec_cache
            hidden_out = hidden_cpu.to(device=out_device, non_blocking=True)
            mask_out = torch.ones(
                (1, hidden_out.shape[1]), dtype=torch.bool, device=out_device
            )
            return hidden_out, mask_out

        total_seq_len = len(tok_list)
        vis_end_id = self.text_encoder.config.vision_end_token_id
        min_snap, struct_cps = Qwen3_5HybridCache.derive_structural_checkpoints(
            tok_list,
            vision_end_token_id=vis_end_id,
            im_end_token_id=im_end_id,
        )
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
                structural_checkpoints=struct_cps,
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
        assert hidden_cpu is not None
        hybrid_state = output.past_key_values
        del (
            output,
            input_ids,
            attention_mask,
            inputs_embeds,
            pixel_values,
            image_grid_thw,
        )

        if use_cache:
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

    def _execute_stage_1_visual_conditioning(
        self,
        request: GenerationRequest,
        cpu_offload: bool,
    ) -> VisualConditioningStageState:
        init_latents_cpu: Optional[torch.Tensor] = None
        paint_mask_cpu: Optional[torch.Tensor] = None
        box_mask_cpu: Optional[torch.Tensor] = None
        alpha_mask_cpu: Optional[torch.Tensor] = None
        inpaint_mask_cpu: Optional[torch.Tensor] = None
        ref_latents_cpu_list: List[torch.Tensor] = []
        ref_bboxes_list: List[Optional[Tuple[float, float, float, float]]] = []
        ref_box_iso_list: List[float] = []
        ref_masks_cpu_list: List[Optional[torch.Tensor]] = []
        ref_roles_list: List[str] = []
        ref_z_orders_list: List[float] = []
        ref_keep_in_uncond_list: List[bool] = []
        splat_entries: List[Tuple[Any, ...]] = []
        box_mask_entries: List[Tuple[Any, ...]] = []
        splatted_box_indices: Set[int] = set()
        ref_records_for_prompt: List[CachedReferenceRecord] = []
        conditioned_ref_telem_map: Dict[Any, Dict[str, Any]] = {}
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
        is_ait_format = request.edit_format in (
            "ai_toolkit_isolated",
            "ai_toolkit_full",
        )

        if has_init or has_mask_file or has_control or valid_ref_specs:
            if cpu_offload:
                assert self.vae is not None
                assert self.text_encoder is not None
                self.vae.to(self.device)
                self.text_encoder.to(self.device)

            if has_init and request.init_image_path is not None:
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

            if has_mask_file and request.inpaint_mask_path is not None:
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
                    paint_mask_cpu = _evacuate_to_cpu(
                        torch.from_numpy(m_arr).unsqueeze(0).unsqueeze(0).contiguous()
                    )

            if has_control and request.control_image_path is not None:
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
                    max_vit_patches=int(request.max_vit_patches or 2304),
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
                    ctrl_lat_cpu = _evacuate_to_cpu(ctrl_rec.packed_latents)
                    if ctrl_lat_cpu is not None:
                        ref_latents_cpu_list.append(ctrl_lat_cpu)
                        ref_bboxes_list.append(None)
                        ref_box_iso_list.append(0.0)
                        ref_masks_cpu_list.append(None)
                        ref_roles_list.append(ctrl_role)
                        ref_z_orders_list.append(0.0)
                        ref_keep_in_uncond_list.append(True)

            for spec_idx, spec in enumerate(valid_ref_specs):
                r_path = Path(spec["path"])
                px_box = spec.get("pixel_bbox")
                norm_box = spec.get("norm_bbox")
                is_box_ref = norm_box is not None
                role_str = str(
                    spec.get("input_role") or request.input_role or "photo_ref"
                ).lower()
                emits_vit = bool(
                    spec.get(
                        "emits_vit",
                        role_str == "photo_ref"
                        and (is_box_ref or request.edit_format != "moss_image_edit"),
                    )
                )
                emits_dit = bool(
                    spec.get("emits_dit_latents", role_str != "inpaint_mask")
                )
                emits_box_mask = bool(
                    spec.get("emits_box_mask", role_str == "inpaint_mask")
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
                        max_vit_patches=int(request.max_vit_patches or 2304),
                    )
                r_rec, hit = self._condition_single_reference(
                    r_path,
                    request,
                    role_override=role_str,
                    framing_override=spec.get("framing_mode", request.framing_mode),
                    target_w_override=ref_w,
                    target_h_override=ref_h,
                    force_vit=emits_vit,
                )
                any_ref_cache_hit = any_ref_cache_hit or hit

                vae_sz = (
                    (
                        int(r_rec.packed_latents.shape[3]) * 16,
                        int(r_rec.packed_latents.shape[2]) * 16,
                    )
                    if (
                        r_rec.packed_latents is not None
                        and r_rec.packed_latents.ndim == 4
                    )
                    else (
                        tuple(r_rec.viz_meta.vae_size)
                        if (
                            r_rec.viz_meta is not None
                            and getattr(r_rec.viz_meta, "vae_size", None)
                        )
                        else (ref_w, ref_h)
                    )
                )
                box_sharpness: Optional[float] = None
                if (
                    r_rec.viz_meta is not None
                    and getattr(r_rec.viz_meta, "telemetry", None) is not None
                ):
                    box_sharpness = float(
                        r_rec.viz_meta.telemetry.laplacian_sharpness
                    )
                is_cutout_mask = (
                    r_rec.latent_mask is not None
                    and float(torch.min(r_rec.latent_mask).item()) < 0.99
                )
                telem_info = {
                    "vae_size": vae_sz,
                    "llm_tokens": int(r_rec.llm_tokens) if emits_vit else 0,
                    "has_cutout": bool(is_cutout_mask and role_str == "photo_ref"),
                    "sharpness": float(box_sharpness or 0.0),
                }
                conditioned_ref_telem_map[spec_idx] = telem_info
                conditioned_ref_telem_map[spec["path"]] = telem_info

                if emits_box_mask and r_rec.latent_mask is not None:
                    box_mask_entries.append(
                        (_evacuate_to_cpu(r_rec.latent_mask), norm_box)
                    )

                if emits_dit and r_rec.packed_latents is not None:
                    lat_cpu = _evacuate_to_cpu(r_rec.packed_latents)
                    mask_cpu = _evacuate_to_cpu(r_rec.latent_mask)
                    is_pasted = bool(spec.get("paste_to_canvas", False)) and (
                        norm_box is not None
                    )
                    z_val = float(spec.get("z_order") or 0.0)
                    if lat_cpu is not None:
                        ref_latents_cpu_list.append(lat_cpu)
                        ref_bboxes_list.append(norm_box)
                        ref_box_iso_list.append(
                            float(spec.get("isolate_box_attn") or 0.0)
                        )
                        ref_masks_cpu_list.append(
                            mask_cpu if is_cutout_mask else None
                        )
                        ref_roles_list.append(role_str)
                        ref_z_orders_list.append(z_val)
                        keep_in_uncond = (
                            True
                            if not is_ait_format
                            else (
                                role_str in ("depth_map", "color_map")
                                or is_pasted
                            )
                        )
                        ref_keep_in_uncond_list.append(keep_in_uncond)
                        if is_pasted and norm_box is not None:
                            b_idx = spec.get("box_index")
                            if isinstance(b_idx, int):
                                splatted_box_indices.add(b_idx)
                            splat_entries.append(
                                (
                                    lat_cpu,
                                    norm_box,
                                    mask_cpu,
                                    spec.get("denoise_strength"),
                                    box_sharpness,
                                    z_val,
                                )
                            )
                if (
                    r_rec.llm_tokens > 0
                    and emits_vit
                    and (
                        r_rec.vit_embeds is not None
                        or r_rec.vit_patches is not None
                    )
                ):
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
                assert self.vae is not None
                self.vae.to("cpu")
                self._trim_reserved_vram(force=True, required_headroom_gb=4.20)  # type: ignore[attr-defined]
            else:
                self._trim_reserved_vram(force=False, required_headroom_gb=1.25)  # type: ignore[attr-defined]

        if (
            has_init or bool(splat_entries) or bool(box_mask_entries)
        ) and request.layout_items:
            want_box_aperture = (
                bool(request.preserve_canvas_outside_boxes)
                or str(request.mask_blend_mode or "auto").lower() == "boxes_only"
                or any(
                    isinstance(it, dict)
                    and (
                        bool(it.get("inpaint_box", False))
                        or (
                            has_init
                            and bool(it.get("paste_to_canvas", False))
                            and str(it.get("input_role") or "").lower()
                            not in ("depth_map", "color_map", "inpaint_mask")
                        )
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
                    feather_radius_lat=float(
                        request.mask_feather_radius
                        if request.mask_feather_radius is not None
                        else 1.25
                    ),
                    device=torch.device("cpu"),
                    only_flagged_boxes=only_flagged,
                    include_paste_boxes=has_init,
                )

        if box_mask_entries:
            box_mask_cpu = splat_box_masks_to_canvas(
                base_mask=box_mask_cpu,
                mask_entries=box_mask_entries,
                h_lat=request.height // 16,
                w_lat=request.width // 16,
                feather_radius_lat=float(
                    request.mask_feather_radius
                    if request.mask_feather_radius is not None
                    else 1.25
                ),
                device=torch.device("cpu"),
                feather_box_edges=False,
            )
        del box_mask_entries

        if (
            has_init
            or bool(splat_entries)
            or box_mask_cpu is not None
            or paint_mask_cpu is not None
        ):
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

        return VisualConditioningStageState(
            init_latents_cpu=init_latents_cpu,
            inpaint_mask_cpu=inpaint_mask_cpu,
            ref_latents_cpu_list=ref_latents_cpu_list,
            ref_bboxes_list=ref_bboxes_list,
            ref_box_iso_list=ref_box_iso_list,
            ref_masks_cpu_list=ref_masks_cpu_list,
            ref_roles_list=ref_roles_list,
            ref_z_orders_list=ref_z_orders_list,
            ref_keep_in_uncond_list=ref_keep_in_uncond_list,
            splat_entries=splat_entries,
            splatted_box_indices=splatted_box_indices,
            ref_records_for_prompt=ref_records_for_prompt,
            conditioned_ref_telem_map=conditioned_ref_telem_map,
            any_ref_cache_hit=any_ref_cache_hit,
            exported_vizj_str=exported_vizj_str,
            has_init=has_init,
            is_ait_format=is_ait_format,
        )

    def _execute_stage_2_context_encoding(
        self,
        request: GenerationRequest,
        vis_state: VisualConditioningStageState,
        cpu_offload: bool,
        reporter: Optional[EngineProgressReporter],
        effective_cb: Callable[[str, int, int], None],
    ) -> ContextEncodingStageState:
        assert self.text_encoder is not None
        effective_prompt = request.compile_effective_prompt()
        resolved_trace, spec_cot_tokens = request.resolve_thinking_trace(
            conditioned_ref_telemetry=(
                vis_state.conditioned_ref_telem_map
                if vis_state.conditioned_ref_telem_map
                else None
            )
        )
        vis_state.conditioned_ref_telem_map.clear()
        eff_max_seq_len = int(request.max_sequence_length or 16384)
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
            max_sequence_length=eff_max_seq_len,
            enable_thinking=bool(
                True if spec_cot_tokens > 0 else request.enable_thinking
            ),
            device=torch.device("cpu"),
            is_negative=False,
            prebuilt_vision_records=vis_state.ref_records_for_prompt,
            system_prompt=request.system_prompt,
            thinking_trace=resolved_trace,
            add_vision_id=bool(
                request.add_vision_id if request.add_vision_id is not None else True
            ),
            speculative_cot_tokens=spec_cot_tokens,
        )
        cache_hit = (self.embed_cache.pos_hits > pos_hits_before) or (
            self.embed_cache.splice_hits > splice_hits_before
        )
        common_prefix_len = self.embed_cache.last_common_prefix_len
        self._sync_device()  # type: ignore[attr-defined]
        effective_cb("context", 1, ctx_total)

        neg_embeds: Optional[torch.Tensor] = None
        neg_mask: Optional[torch.Tensor] = None
        if any_cfg:
            neg_enable_thinking = bool(request.enable_thinking) and not (
                bool(resolved_trace) or spec_cot_tokens > 0
            )
            neg_embeds, neg_mask = self.encode_prompt(
                request.negative_prompt or "",
                max_sequence_length=eff_max_seq_len,
                enable_thinking=neg_enable_thinking,
                device=torch.device("cpu"),
                is_negative=True,
                add_vision_id=bool(
                    request.add_vision_id if request.add_vision_id is not None else True
                ),
            )
            self._sync_device()  # type: ignore[attr-defined]
            effective_cb("context", 2, ctx_total)

        vis_state.ref_records_for_prompt.clear()
        self.text_encoder.clear_caches()
        if cpu_offload:
            self.text_encoder.to("cpu")
            self._trim_reserved_vram(force=True, required_headroom_gb=11.0)  # type: ignore[attr-defined]
        else:
            self._trim_reserved_vram(force=False, required_headroom_gb=1.25)  # type: ignore[attr-defined]

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

        return ContextEncodingStageState(
            positive=positive,
            positive_mask=positive_mask,
            negative=negative,
            negative_mask=negative_mask,
            effective_prompt=effective_prompt,
            cache_hit=cache_hit,
            common_prefix_len=common_prefix_len,
            spec_cot_tokens=spec_cot_tokens,
            early_cfg=early_cfg,
            late_cfg=late_cfg,
        )

    def _execute_stage_3_flow_matching(
        self,
        request: GenerationRequest,
        explicit_seed: int,
        vis_state: VisualConditioningStageState,
        ctx_state: ContextEncodingStageState,
        cpu_offload: bool,
        reporter: Optional[EngineProgressReporter],
        effective_cb: Callable[[str, int, int], None],
    ) -> FlowMatchingStageState:
        assert self.transformer is not None
        if cpu_offload:
            self.transformer.to(self.device)

        depth = len(self.transformer.layers)
        uncond_skip_set = validate_skip_layers(
            request.uncond_skip_layers, depth, "uncond_skip_layers"
        )
        slg_skip_set = validate_skip_layers(request.slg_layers, depth, "slg_layers")

        ref_latents_list: List[torch.Tensor] = [
            r_cpu.to(device=self.device, dtype=torch.float32, non_blocking=True)
            for r_cpu in vis_state.ref_latents_cpu_list
            if r_cpu is not None
        ]
        vis_state.ref_latents_cpu_list.clear()

        ref_masks_list: List[Optional[torch.Tensor]] = [
            m_cpu.to(device=self.device, dtype=torch.float32, non_blocking=True)
            if m_cpu is not None
            else None
            for m_cpu in vis_state.ref_masks_cpu_list
        ]
        vis_state.ref_masks_cpu_list.clear()

        num_imgs = request.num_images_per_prompt
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
        if vis_state.init_latents_cpu is not None:
            clean_init_latents = vis_state.init_latents_cpu.to(
                device=self.device, dtype=torch.float32, non_blocking=True
            )
            if clean_init_latents.shape[-2:] != (h_lat, w_lat):
                clean_init_latents = resample_packed_latents(
                    clean_init_latents, h_lat, w_lat
                )
        vis_state.init_latents_cpu = None

        composite_used = False
        raw_splat_alpha: Optional[torch.Tensor] = None
        splat_union_alpha: Optional[torch.Tensor] = None
        splat_denoise_map: Optional[torch.Tensor] = None
        box_horizon_map: Optional[torch.Tensor] = None
        d_init = max(0.01, min(1.0, float(request.denoise_strength)))
        eff_denoise_strength = d_init
        eff_feather_lat = float(
            request.mask_feather_radius
            if request.mask_feather_radius is not None
            else 1.25
        )

        splat_boxes = [e[1] for e in vis_state.splat_entries]
        default_carve_iso = max(
            DEFAULT_BOX_REF_ISOLATION,
            float(request.box_attn_isolation or 0.0),
            max(vis_state.ref_box_iso_list) if vis_state.ref_box_iso_list else 0.0,
        )
        aperture_overrides, attn_carveouts = resolve_foreground_aperture_carveouts(
            layout_items=request.layout_items,
            height=request.height,
            width=request.width,
            coords_are_pixels=request.layout_coords_are_pixels,
            splat_boxes=splat_boxes,
            splatted_box_indices=vis_state.splatted_box_indices,
            ref_bboxes=vis_state.ref_bboxes_list,
            ref_z_orders=vis_state.ref_z_orders_list,
            default_isolation=default_carve_iso,
        )
        active_fg_carveouts = attn_carveouts if attn_carveouts else None

        if vis_state.splat_entries:
            composite_used = True
            default_splat_dn = d_init if d_init < 0.999 else 0.75
            (
                clean_init_latents,
                splat_union_alpha,
                splat_denoise_map,
            ) = splat_box_latents_to_canvas(
                base_canvas=clean_init_latents,
                splat_entries=vis_state.splat_entries,
                h_lat=h_lat,
                w_lat=w_lat,
                latent_channels=self.transformer.config.latent_channels,
                feather_radius_lat=eff_feather_lat,
                device=self.device,
                default_denoise_strength=default_splat_dn,
                return_denoise_map=True,
                aperture_overrides=aperture_overrides if aperture_overrides else None,
            )
            raw_splat_alpha = splat_union_alpha.clone()
            resolved_box_dns = [
                float(e[3]) if e[3] is not None else default_splat_dn
                for e in vis_state.splat_entries
            ] + [float(o[1]) for o in aperture_overrides]
            max_box_dn = (
                max(resolved_box_dns) if resolved_box_dns else default_splat_dn
            )
            if vis_state.has_init:
                if d_init >= 0.999 and any(
                    e[3] is not None for e in vis_state.splat_entries
                ):
                    eff_denoise_strength = max(0.01, min(1.0, max_box_dn))
                else:
                    eff_denoise_strength = max(
                        0.01, min(1.0, max(d_init, max_box_dn))
                    )
                if (
                    d_init < eff_denoise_strength - 1e-4
                    and splat_union_alpha is not None
                    and splat_denoise_map is not None
                ):
                    splat_denoise_map = (
                        splat_union_alpha * splat_denoise_map
                        + (1.0 - splat_union_alpha) * d_init
                    ).clamp(0.01, 1.0)
                    splat_union_alpha = torch.ones_like(splat_union_alpha)
        vis_state.splat_entries.clear()

        if (
            vis_state.has_init
            and clean_init_latents is not None
            and request.layout_items
        ):
            box_horizon_map = build_box_denoise_horizon_map(
                layout_items=request.layout_items,
                height=request.height,
                width=request.width,
                coords_are_pixels=request.layout_coords_are_pixels,
                feather_radius_lat=eff_feather_lat,
                default_denoise_strength=d_init,
                device=self.device,
                only_flagged_boxes=not bool(request.preserve_canvas_outside_boxes),
                include_paste_boxes=True,
                return_none_if_uniform=True,
            )
            if box_horizon_map is not None and not composite_used:
                max_h_dn = float(box_horizon_map.max().item())
                if d_init < 0.999:
                    eff_denoise_strength = max(
                        0.01, min(1.0, max(eff_denoise_strength, max_h_dn))
                    )

        sched_shift = float(request.shift if request.shift is not None else 5.0)
        if splat_denoise_map is not None:
            splat_denoise_map = warp_denoise_to_sigma_horizon(
                splat_denoise_map, shift=sched_shift
            )  # type: ignore[assignment]
        if box_horizon_map is not None:
            box_horizon_map = warp_denoise_to_sigma_horizon(
                box_horizon_map, shift=sched_shift
            )  # type: ignore[assignment]

        inpaint_mask_tensor: Optional[torch.Tensor] = None
        if vis_state.inpaint_mask_cpu is not None:
            inpaint_mask_tensor = vis_state.inpaint_mask_cpu.to(
                device=self.device, dtype=torch.float32, non_blocking=True
            )
            if inpaint_mask_tensor.shape[-2:] != (h_lat, w_lat):
                inpaint_mask_tensor = F.interpolate(
                    inpaint_mask_tensor,
                    size=(h_lat, w_lat),
                    mode="bilinear",
                    align_corners=False,
                )
            if not vis_state.has_init and splat_union_alpha is not None:
                bg_lock = torch.minimum(
                    (1.0 - inpaint_mask_tensor).clamp(0.0, 1.0),
                    splat_union_alpha,
                )
                inpaint_mask_tensor = (1.0 - bg_lock).clamp(0.0, 1.0)
        elif (
            vis_state.has_init
            and clean_init_latents is not None
            and box_horizon_map is not None
        ):
            inpaint_mask_tensor = torch.ones(
                (1, 1, h_lat, w_lat), device=self.device, dtype=torch.float32
            )
        vis_state.inpaint_mask_cpu = None

        start_step = 0
        if (
            vis_state.has_init
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

        early_cfg = ctx_state.early_cfg
        late_cfg = ctx_state.late_cfg
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

        assert active_scheduler.sigmas is not None
        sigma_0 = (
            float(active_scheduler._sigmas_cpu[0])
            if getattr(active_scheduler, "_sigmas_cpu", None)
            else float(active_scheduler.sigmas[0].item())
        )
        if (
            vis_state.has_init
            and clean_init_latents is not None
            and (start_step > 0 or inpaint_mask_tensor is not None)
        ):
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

        if (
            inpaint_mask_tensor is not None
            and clean_init_latents is not None
            and vis_state.has_init
        ):
            eff_horizon = (
                torch.minimum(
                    box_horizon_map,
                    torch.full_like(box_horizon_map, sigma_0),
                )
                if box_horizon_map is not None
                else sigma_0
            )
            inpaint_release_dn = (
                inpaint_mask_tensor.clamp(0.0, 1.0) * eff_horizon
            ).clamp(0.0, 1.0)
            horizon_hold_alpha = (
                ((sigma_0 - inpaint_release_dn) / max(sigma_0, 1e-4)).clamp(
                    0.0, 1.0
                )
            )
            bg_preserve_alpha = torch.maximum(
                (1.0 - inpaint_mask_tensor).clamp(0.0, 1.0),
                horizon_hold_alpha,
            )
            if splat_union_alpha is None or splat_denoise_map is None:
                splat_union_alpha = bg_preserve_alpha
                splat_denoise_map = inpaint_release_dn
            else:
                pasted_mask = (
                    raw_splat_alpha
                    if raw_splat_alpha is not None
                    else torch.zeros_like(inpaint_mask_tensor)
                )
                pure_inpaint_alpha = torch.minimum(
                    inpaint_mask_tensor.clamp(0.0, 1.0),
                    (1.0 - pasted_mask).clamp(0.0, 1.0),
                )
                attenuated_splat_alpha = torch.minimum(
                    splat_union_alpha,
                    (1.0 - pure_inpaint_alpha).clamp(0.0, 1.0),
                )
                splat_union_alpha = torch.maximum(
                    attenuated_splat_alpha, bg_preserve_alpha
                ).clamp(0.0, 1.0)
                splat_denoise_map = (
                    pasted_mask * splat_denoise_map
                    + (1.0 - pasted_mask) * inpaint_release_dn
                ).clamp(0.0, 1.0)
            del (
                bg_preserve_alpha,
                horizon_hold_alpha,
                inpaint_release_dn,
            )
        del raw_splat_alpha

        use_velocity_splat_prior = bool(
            clean_init_latents is not None
            and splat_union_alpha is not None
            and splat_denoise_map is not None
            and (
                composite_used
                or inpaint_mask_tensor is not None
                or box_horizon_map is not None
            )
        )

        sde_gen = torch.Generator(device=self.device).manual_seed(k_sde & _MASK63)
        timesteps = active_scheduler.timesteps
        assert timesteps is not None
        num_passes = len(timesteps)
        total_nfe = 0

        active_ref_latents = ref_latents_list if ref_latents_list else None
        active_ref_bboxes = (
            vis_state.ref_bboxes_list
            if (
                vis_state.ref_bboxes_list
                and any(b is not None for b in vis_state.ref_bboxes_list)
            )
            else None
        )
        active_ref_box_iso = (
            vis_state.ref_box_iso_list
            if (
                active_ref_bboxes is not None
                and any(v > 1e-6 for v in vis_state.ref_box_iso_list)
            )
            else None
        )
        active_ref_masks = (
            ref_masks_list
            if (
                active_ref_bboxes is not None
                and any(m is not None for m in ref_masks_list)
            )
            else None
        )
        active_ref_roles = (
            vis_state.ref_roles_list if active_ref_latents is not None else None
        )
        active_ref_z_orders = (
            vis_state.ref_z_orders_list if active_ref_latents is not None else None
        )

        if request.uncond_uses_references is not None:
            uncond_uses_refs = bool(request.uncond_uses_references)
            uncond_ref_latents = active_ref_latents if uncond_uses_refs else None
            uncond_ref_bboxes = active_ref_bboxes if uncond_uses_refs else None
            uncond_ref_box_iso = active_ref_box_iso if uncond_uses_refs else None
            uncond_ref_masks = active_ref_masks if uncond_uses_refs else None
            uncond_ref_roles = active_ref_roles if uncond_uses_refs else None
            uncond_ref_z_orders = active_ref_z_orders if uncond_uses_refs else None
        elif vis_state.is_ait_format and active_ref_latents is not None:
            keep_indices = [
                i
                for i, keep in enumerate(vis_state.ref_keep_in_uncond_list)
                if keep
            ]
            if not keep_indices:
                uncond_uses_refs = False
                uncond_ref_latents = None
                uncond_ref_bboxes = None
                uncond_ref_box_iso = None
                uncond_ref_masks = None
                uncond_ref_roles = None
                uncond_ref_z_orders = None
            elif len(keep_indices) == len(ref_latents_list):
                uncond_uses_refs = True
                uncond_ref_latents = active_ref_latents
                uncond_ref_bboxes = active_ref_bboxes
                uncond_ref_box_iso = active_ref_box_iso
                uncond_ref_masks = active_ref_masks
                uncond_ref_roles = active_ref_roles
                uncond_ref_z_orders = active_ref_z_orders
            else:
                uncond_uses_refs = True
                uncond_ref_latents = [ref_latents_list[i] for i in keep_indices]
                u_bboxes = [vis_state.ref_bboxes_list[i] for i in keep_indices]
                uncond_ref_bboxes = (
                    u_bboxes if any(b is not None for b in u_bboxes) else None
                )
                u_iso = [vis_state.ref_box_iso_list[i] for i in keep_indices]
                uncond_ref_box_iso = (
                    u_iso
                    if (
                        uncond_ref_bboxes is not None
                        and any(v > 1e-6 for v in u_iso)
                    )
                    else None
                )
                u_masks = [ref_masks_list[i] for i in keep_indices]
                uncond_ref_masks = (
                    u_masks
                    if (
                        uncond_ref_bboxes is not None
                        and any(m is not None for m in u_masks)
                    )
                    else None
                )
                uncond_ref_roles = [vis_state.ref_roles_list[i] for i in keep_indices]
                uncond_ref_z_orders = [
                    vis_state.ref_z_orders_list[i] for i in keep_indices
                ]
        else:
            uncond_uses_refs = request.effective_uncond_uses_references
            uncond_ref_latents = active_ref_latents if uncond_uses_refs else None
            uncond_ref_bboxes = active_ref_bboxes if uncond_uses_refs else None
            uncond_ref_box_iso = active_ref_box_iso if uncond_uses_refs else None
            uncond_ref_masks = active_ref_masks if uncond_uses_refs else None
            uncond_ref_roles = active_ref_roles if uncond_uses_refs else None
            uncond_ref_z_orders = active_ref_z_orders if uncond_uses_refs else None

        uncond_fg_carveouts = (
            active_fg_carveouts if bool(uncond_ref_latents) else None
        )

        ref_zero_t = request.effective_reference_zero_timestep
        ref_iso_kv = request.effective_reference_isolated_kv
        box_feather_lat = eff_feather_lat

        use_cuda_events = self.device.type == "cuda" and torch.cuda.is_available()
        step_events = (
            [torch.cuda.Event(), torch.cuda.Event()] if use_cuda_events else []
        )

        if reporter is not None:
            reporter.start_stage("dit", num_passes)
        dit_t0 = time.perf_counter()

        positive = ctx_state.positive
        positive_mask = ctx_state.positive_mask
        negative = ctx_state.negative
        negative_mask = ctx_state.negative_mask

        prev_skip_regime: Optional[Tuple[bool, bool]] = None

        for pass_index, timestep in enumerate(timesteps):
            if isinstance(active_scheduler, BifurcatedFlowMatchScheduler):
                is_early = active_scheduler.current_is_early
                step_frac = active_scheduler.current_progress
                want_hist_out = (
                    active_scheduler.wants_early_history_output
                    and abs(early_cfg - late_cfg) > 1e-5
                )
                cfg_step = active_scheduler._step_configs[
                    min(pass_index, num_passes - 1)
                ]
                step_dt = float(cfg_step["dt"])
                next_sigma = float(cfg_step["s_next"])
                sigma_eval = float(cfg_step["timestep"]) / float(
                    active_scheduler.num_train_timesteps
                )
            else:
                is_early = True
                sig_cpu = getattr(active_scheduler, "_sigmas_cpu", None)
                if sig_cpu:
                    s_curr_sched = float(sig_cpu[min(pass_index, num_passes - 1)])
                    next_sigma = float(
                        sig_cpu[min(pass_index + 1, len(sig_cpu) - 1)]
                    )
                else:
                    s_curr_sched = float(
                        active_scheduler.sigmas[min(pass_index, num_passes - 1)].item()
                    )
                    next_sigma = float(
                        active_scheduler.sigmas[
                            min(pass_index + 1, len(active_scheduler.sigmas) - 1)
                        ].item()
                    )
                step_dt = next_sigma - s_curr_sched
                step_frac = 1.0 - unwarp_sigma_to_progress(s_curr_sched, sched_shift)
                want_hist_out = False
                sigma_eval = s_curr_sched

            current_cfg = early_cfg if is_early else late_cfg
            do_step_cfg = abs(current_cfg - 1.0) > 1e-5
            _, eff_uncond_skip, eff_slg_skip, do_slg = resolve_skip_layer_regimes(
                uncond_skip_layers=uncond_skip_set,
                slg_layers=slg_skip_set,
                slg_scale=float(request.slg_scale or 0.0),
                slg_start=float(request.slg_start),
                slg_stop=float(request.slg_stop),
                step_progress=step_frac,
                depth=depth,
            )
            if do_slg and do_step_cfg and (eff_slg_skip & eff_uncond_skip):
                ortho_slg: Set[int] = set()
                for l_idx in eff_slg_skip:
                    if l_idx in eff_uncond_skip:
                        cand_idx = l_idx - 2 if l_idx >= 2 else l_idx + 2
                        if 0 <= cand_idx < depth and cand_idx not in eff_uncond_skip:
                            ortho_slg.add(cand_idx)
                        else:
                            ortho_slg.add(l_idx)
                    else:
                        ortho_slg.add(l_idx)
                eff_slg_skip = ortho_slg

            cur_skip_regime = (bool(eff_uncond_skip and do_step_cfg), bool(do_slg))
            if (
                prev_skip_regime is not None
                and cur_skip_regime != prev_skip_regime
                and isinstance(active_scheduler, BifurcatedFlowMatchScheduler)
            ):
                active_scheduler.notify_regime_change()
            prev_skip_regime = cur_skip_regime

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
                reference_masks=active_ref_masks,
                reference_roles=active_ref_roles,
                reference_z_orders=active_ref_z_orders,
                foreground_carveouts=active_fg_carveouts,
            ).sample.float()
            conditional = prediction if do_slg else None
            hist_prediction: Optional[torch.Tensor] = None
            total_nfe += 1

            if do_step_cfg and negative is not None:
                skip_kwargs = (
                    {"skip_layers": eff_uncond_skip} if eff_uncond_skip else {}
                )
                uncond = self.transformer(
                    latents,
                    timestep,
                    negative,
                    negative_mask,
                    reference_latents=uncond_ref_latents,
                    reference_zero_timestep=ref_zero_t and bool(uncond_ref_latents),
                    reference_isolated_kv=ref_iso_kv and bool(uncond_ref_latents),
                    reference_bboxes=uncond_ref_bboxes,
                    reference_box_isolation=uncond_ref_box_iso,
                    box_feather_radius=box_feather_lat,
                    reference_masks=uncond_ref_masks,
                    reference_roles=uncond_ref_roles,
                    reference_z_orders=uncond_ref_z_orders,
                    foreground_carveouts=uncond_fg_carveouts,
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
                    skip_layers=eff_slg_skip,
                    reference_latents=active_ref_latents,
                    reference_zero_timestep=ref_zero_t,
                    reference_isolated_kv=ref_iso_kv,
                    reference_bboxes=active_ref_bboxes,
                    reference_box_isolation=active_ref_box_iso,
                    box_feather_radius=box_feather_lat,
                    reference_masks=active_ref_masks,
                    reference_roles=active_ref_roles,
                    reference_z_orders=active_ref_z_orders,
                    foreground_carveouts=active_fg_carveouts,
                ).sample.float()
                if not do_step_cfg:
                    prediction = skipped
                slg_delta = float(request.slg_scale or 0.0) * (
                    conditional - skipped
                )
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
                prediction = couple_splat_velocity_prior(
                    v_dit=prediction,
                    latents=latents,
                    splat_canvas=clean_init_latents,
                    splat_union_alpha=splat_union_alpha,
                    splat_denoise_map=splat_denoise_map,
                    sigma_curr=sigma_eval,
                    noise=noise,
                    step_dt=step_dt,
                )
                if hist_prediction is not None:
                    hist_prediction = couple_splat_velocity_prior(
                        v_dit=hist_prediction,
                        latents=latents,
                        splat_canvas=clean_init_latents,
                        splat_union_alpha=splat_union_alpha,
                        splat_denoise_map=splat_denoise_map,
                        sigma_curr=sigma_eval,
                        noise=noise,
                        step_dt=step_dt,
                    )

            latents = active_scheduler.step(
                prediction,
                timestep,
                latents,
                generator=sde_gen,
                return_dict=False,
                history_output=hist_prediction,
            )  # type: ignore[assignment]
            del prediction, hist_prediction

            if inpaint_mask_tensor is not None and clean_init_latents is not None:
                noised_bg = (1.0 - next_sigma) * clean_init_latents + next_sigma * noise
                active_fg_gate, _ = compute_differential_inpaint_gate(
                    inpaint_mask=inpaint_mask_tensor,
                    next_sigma=next_sigma,
                    sigma_0=sigma_0,
                    box_horizon_map=(
                        box_horizon_map
                        if box_horizon_map is not None
                        else splat_denoise_map
                    ),
                )
                latents = (
                    active_fg_gate * latents
                    + (1.0 - active_fg_gate) * noised_bg
                )
                del noised_bg, active_fg_gate

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
            early_steps = len(timesteps)
            late_steps = 0

        max_box_iso_used = max(
            [float(request.box_attn_isolation or 0.0)]
            + [float(x) for x in vis_state.ref_box_iso_list]
            + [float(c[1]) for c in attn_carveouts]
        )

        latents_cpu = _evacuate_to_cpu(latents)
        assert latents_cpu is not None
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
            active_ref_masks,
            active_ref_roles,
            active_ref_z_orders,
            active_fg_carveouts,
            uncond_ref_latents,
            uncond_ref_bboxes,
            uncond_ref_box_iso,
            uncond_ref_masks,
            uncond_ref_roles,
            uncond_ref_z_orders,
            uncond_fg_carveouts,
            ref_latents_list,
            ref_masks_list,
            clean_init_latents,
            splat_union_alpha,
            splat_denoise_map,
            box_horizon_map,
            inpaint_mask_tensor,
            noise,
        )
        vis_state.ref_bboxes_list.clear()
        vis_state.ref_box_iso_list.clear()
        vis_state.ref_roles_list.clear()
        vis_state.ref_z_orders_list.clear()
        vis_state.ref_keep_in_uncond_list.clear()

        return FlowMatchingStageState(
            latents_cpu=latents_cpu,
            total_nfe=total_nfe,
            early_steps=early_steps,
            late_steps=late_steps,
            dit_elapsed=dit_elapsed,
            max_box_iso_used=max_box_iso_used,
            composite_used=composite_used,
            ref_zero_t=ref_zero_t,
            ref_iso_kv=ref_iso_kv,
            uncond_uses_refs=bool(uncond_uses_refs),
        )

    def _execute_stage_4_decode_and_export(
        self,
        request: GenerationRequest,
        target_repo: str,
        explicit_seed: int,
        start_time: float,
        cpu_offload: bool,
        lora_active: bool,
        vis_state: VisualConditioningStageState,
        ctx_state: ContextEncodingStageState,
        flow_state: FlowMatchingStageState,
        reporter: Optional[EngineProgressReporter],
        effective_cb: Callable[[str, int, int], None],
    ) -> GenerationResponse:
        assert self.transformer is not None
        assert self.vae is not None

        if cpu_offload:
            self.transformer.to("cpu")
            self._trim_reserved_vram(force=True, required_headroom_gb=4.40)  # type: ignore[attr-defined]
            self.vae.to(self.device)
        else:
            self._trim_reserved_vram(force=False, required_headroom_gb=1.25)  # type: ignore[attr-defined]

        num_imgs = request.num_images_per_prompt
        if reporter is not None:
            reporter.start_stage("vae", num_imgs)

        latents_vae = flow_state.latents_cpu.to(
            device=self.device, non_blocking=True
        )
        decoded_gpu = self.vae.decode_packed_latents(latents_vae)
        del latents_vae
        decoded_cpu = _evacuate_to_cpu(decoded_gpu)
        assert decoded_cpu is not None
        del decoded_gpu
        self._sync_device()  # type: ignore[attr-defined]

        if cpu_offload:
            self.vae.to("cpu")
            self._trim_reserved_vram(force=True, required_headroom_gb=4.20)  # type: ignore[attr-defined]
        else:
            self._trim_reserved_vram(force=False, required_headroom_gb=1.25)  # type: ignore[attr-defined]

        if self.vizinjector is not None:
            pil_images, viz_telems = self.vizinjector.postprocess_to_pil(
                decoded_cpu, apply_bayer_dither=request.apply_bayer_dither
            )
            mean_luma = viz_telems[0].mean_luminance
            rms_contrast = viz_telems[0].rms_contrast
            entropy = viz_telems[0].shannon_entropy_bits
            sharpness = viz_telems[0].laplacian_sharpness
        else:
            pil_images = self.postprocess_to_pil(decoded_cpu)  # type: ignore[attr-defined]
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

        step_ms = (flow_state.dit_elapsed * 1000.0) / float(
            max(1, flow_state.early_steps + flow_state.late_steps)
        )
        return GenerationResponse(
            output_path=str(out_path),
            additional_paths=extra_paths,
            exported_vizj_path=vis_state.exported_vizj_str,
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
            total_nfe=flow_state.total_nfe,
            shift_used=request.shift,
            use_v2_grid_used=request.use_v2_grid,
            early_solver_used=request.early_solver,
            late_solver_used=request.late_solver,
            handoff_threshold_used=request.handoff_threshold,
            early_steps=flow_state.early_steps,
            late_steps=flow_state.late_steps,
            early_cfg_used=ctx_state.early_cfg,
            late_cfg_used=ctx_state.late_cfg,
            uncond_skip_layers_used=list(request.uncond_skip_layers or []),
            slg_layers_used=(
                list(request.slg_layers or [])
                if (request.slg_scale is not None and request.slg_scale > 0.0)
                else []
            ),
            slg_scale_used=float(request.slg_scale or 0.0),
            eta_used=request.eta,
            s_noise_used=float(request.s_noise),
            enable_thinking_used=bool(
                True
                if ctx_state.spec_cot_tokens > 0
                else request.enable_thinking
            ),
            viztoken_active=True,
            vizinjector_active=self.vizinjector_active,  # type: ignore[attr-defined]
            vizinjector_backend=self.vizinjector_backend_name,  # type: ignore[attr-defined]
            prompt_cache_hit=ctx_state.cache_hit,
            reference_cache_hit=vis_state.any_ref_cache_hit,
            common_prefix_tokens=ctx_state.common_prefix_len,
            edit_format_used=request.edit_format,
            framing_mode_used=request.framing_mode,
            input_role_used=request.input_role,
            mask_blend_mode_used=str(request.mask_blend_mode or "auto"),
            invert_inpaint_mask_used=bool(request.invert_inpaint_mask),
            reference_zero_timestep_used=flow_state.ref_zero_t,
            reference_isolated_kv_used=flow_state.ref_iso_kv,
            box_attn_isolation_used=flow_state.max_box_iso_used,
            uncond_uses_references_used=flow_state.uncond_uses_refs,
            composite_box_images_used=flow_state.composite_used,
            edit_lora_active=lora_active,
            effective_prompt=ctx_state.effective_prompt,
            mean_luminance=mean_luma,
            rms_contrast=rms_contrast,
            shannon_entropy_bits=entropy,
            laplacian_sharpness=sharpness,
        )


__all__ = [
    "VisualConditioningStageState",
    "ContextEncodingStageState",
    "FlowMatchingStageState",
    "PipelineStagesMixin",
]