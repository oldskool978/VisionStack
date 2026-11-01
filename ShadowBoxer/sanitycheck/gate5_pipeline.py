#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

sys.modules["torchvision"] = None
sys.modules["torchvision.transforms"] = None
sys.modules["torchvision.io"] = None

import numpy as np
from PIL import Image
from safetensors.torch import save_file as save_safetensors
import torch
import torch.nn.functional as F

from core.caches import PromptEmbedCache, ReferenceVisionCache
from core.conditioning import (
    build_box_denoise_horizon_map,
    splat_box_masks_to_canvas,
)
from core.pipeline import (
    PipelineLifecycleMixin,
    PipelineStagesMixin,
    compute_differential_inpaint_gate,
    resolve_foreground_aperture_carveouts,
    resolve_skip_layer_regimes,
)
from download_model import inspect_shard_header
from engine import (
    BifurcatedFlowMatchScheduler,
    ConditionedVisionBundle,
    HAS_VIZINJECTOR,
    MOSSImageEngine,
    VIZJ_TIER_VAE_LATENT,
    VizInjectorClient,
    VizInjectorRequest,
    WDDMMemoryGuard,
    _AI_TOOLKIT_MAX_REF_DIM,
    _evacuate_to_cpu,
    _resolve_boxed_reference_dims,
    _resolve_unboxed_reference_dims,
    build_box_aperture_mask,
    compose_latent_inpaint_mask,
    couple_splat_velocity_prior,
    pack_vision_patches_32px,
    philox_randn,
    resample_packed_latents,
    splat_box_latents_to_canvas,
    warp_denoise_to_sigma_horizon,
)
from harness import (
    _apply_box_denoise_spec,
    build_default_request,
    stage_test_input_files,
)
from models.modeling_moss import (
    MOSSImageConfig,
    MOSSImageTransformer,
    _boxes_overlap_coefficient,
    _compute_collaborating_ref_groups,
)
from models.modeling_qwen35 import (
    Qwen3_5Config,
    Qwen3_5Model,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)
from models.modeling_vae import Flux2AutoencoderKL, Flux2VAEConfig
from refinery import (
    DTYPE_LOOKUP,
    DTYPE_MAP,
    FP8_DTYPES,
    clear_refinery_cache,
    refineTensor,
)
from schema import (
    BASELINE_ENGINE_DEFAULTS,
    DEFAULT_BOX_REF_ISOLATION,
    GenerationRequest,
    SUPPORTED_MASK_BLEND_MODES,
    bbox_to_normalized,
    bbox_to_pixel_rect,
    compile_layout_prompt,
    compile_spatial_thinking_trace,
    get_active_engine_defaults,
    get_canonical_box_order,
    has_custom_default_preset,
    pixel_bbox_to_abs_centralized,
    reconcile_layout_summary_quotes,
    scale_layout_items_to_canvas,
)
from vizinjector.schema import VizInjectorResponseMeta, VizInjectorTelemetry


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


class _Stage1UpgradeHarness(PipelineLifecycleMixin, PipelineStagesMixin):
    def __init__(
        self,
        device: torch.device,
        vae: Flux2AutoencoderKL,
        text_encoder: Qwen3_5Model,
        vizinjector: VizInjectorClient,
    ) -> None:
        self.model_variant = "sft"
        self.repo_id = "OpenMOSS-Team/MOSS-Image-SFT"
        self.device = device
        self.dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.vae_dtype = torch.float32
        self.tokenizer = None
        self.text_encoder = text_encoder
        self.transformer = None
        self.vae = vae
        self.embed_cache = PromptEmbedCache(max_entries=4)
        self.ref_cache = ReferenceVisionCache(max_entries=4)
        self.memory_guard = WDDMMemoryGuard(device)
        self.vizinjector = vizinjector
        self._loaded_repo_id = None
        self._current_offload_state = False
        self._warmed_up = True


def gate_5_refinery_schema_and_snapshots() -> None:
    print(
        "\n--- Gate 5: Gorilla DTO-CA Refinery, WDDM Guard, Preset Point-of-Truth & Spatio-Visual Graph ---"
    )
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32

    _assert(
        "bf16" in DTYPE_MAP and "fp16" in DTYPE_LOOKUP,
        "Refinery DTYPE_MAP/DTYPE_LOOKUP missing",
    )
    with torch.inference_mode():
        w = torch.randn((64, 128), device=dev, dtype=torch.float32) * 0.25
        q_bf16 = refineTensor(
            w,
            torch.bfloat16,
            FP8_DTYPES,
            DTYPE_LOOKUP,
            torch.float32,
            target_device=dev,
        )
        q_fp16 = refineTensor(
            w,
            torch.float16,
            FP8_DTYPES,
            DTYPE_LOOKUP,
            torch.float32,
            target_device=dev,
        )
        _assert(
            q_bf16.dtype == torch.bfloat16 and q_bf16.shape == w.shape,
            "TheContender BF16 quantization failed",
        )
        _assert(
            q_fp16.dtype == torch.float16 and q_fp16.shape == w.shape,
            "TheRefinery FP16 quantization failed",
        )
        e4m3_dtype = DTYPE_LOOKUP.get("e4m3fn")
        if e4m3_dtype is not None:
            q_fp8 = refineTensor(
                w,
                e4m3_dtype,
                FP8_DTYPES,
                DTYPE_LOOKUP,
                torch.float32,
                target_device=dev,
            )
            _assert(
                q_fp8.dtype == e4m3_dtype,
                "TheRefinery FP8 E4M3FN quantization failed",
            )
    clear_refinery_cache()
    print(
        "  [+] Gorilla DTO-CA Refinery  : TheContender (3D Gray-Code Z-Curve BF16) & TheRefinery (Perona-Malik FP16/FP8) Verified"
    )

    guard = WDDMMemoryGuard(dev)
    guard.sync_pinned_baseline()
    snap = guard.enforce_stage_budget(required_headroom_gb=2.50, force=False)
    _assert(
        snap.dedicated_total_gb >= 0.0
        and snap.dedicated_avail_gb >= 0.0
        and snap.contiguous_avail_gb >= 0.0
        and snap.pt_inactive_split_gb >= 0.0
        and snap.host_avail_gb >= 0.0,
        f"Invalid WDDMMemoryGuard snapshot: {snap}",
    )
    print(
        f"  [+] WDDMMemoryGuard (DXGI3)  : Active={guard.active} ({guard.adapter_name}) | "
        f"Avail={snap.dedicated_avail_gb:.2f}/{snap.dedicated_total_gb:.2f} GB "
        f"(Contig={snap.contiguous_avail_gb:.2f} GB) | HostAvail={snap.host_avail_gb:.2f} GB | "
        f"SpillDelta=+{snap.shared_delta_gb:.2f} GB"
    )
    guard.close()

    with tempfile.TemporaryDirectory() as tmpdir:
        test_sf = (
            Path(tmpdir)
            / "transformer"
            / "diffusion_pytorch_model-00001-of-00003.safetensors"
        )
        test_sf.parent.mkdir(parents=True, exist_ok=True)
        save_safetensors(
            {
                "rope.bands": torch.ones((16,), dtype=torch.float32),
                "layers.0.proj.q.weight": torch.randn((64, 64), dtype=torch.bfloat16),
                "layers.0.norm1.weight": torch.ones((64,), dtype=torch.bfloat16),
            },
            str(test_sf),
        )
        mod_native, q_nat, p_nat, non_bf16 = inspect_shard_header(
            test_sf, "transformer", None
        )
        _assert(
            not mod_native and q_nat == 0 and p_nat == 3 and not non_bf16,
            "Zero-copy native header audit failed",
        )
        mod_bf16, q_bf, p_bf, _ = inspect_shard_header(
            test_sf, "transformer", torch.bfloat16
        )
        _assert(
            not mod_bf16 and q_bf == 1 and p_bf == 2,
            "Zero-copy BF16 header audit failed",
        )
        mod_fp16, _, _, _ = inspect_shard_header(
            test_sf, "transformer", torch.float16
        )
        _assert(
            mod_fp16 is True, "Header audit failed to detect required FP16 conversion"
        )
    print(
        "  [+] Zero-Copy Header Auditor : safe_open Header Inspection Verified (0 GB Redundant Disk Writes)"
    )

    _assert(
        BASELINE_ENGINE_DEFAULTS["early_solver"] == "heun"
        and BASELINE_ENGINE_DEFAULTS["late_solver"] == "ipndm"
        and abs(float(BASELINE_ENGINE_DEFAULTS["early_cfg"]) - 3.2) < 1e-6
        and abs(float(BASELINE_ENGINE_DEFAULTS["late_cfg"]) - 2.8) < 1e-6
        and BASELINE_ENGINE_DEFAULTS["uncond_skip_layers"] == [24]
        and BASELINE_ENGINE_DEFAULTS["slg_layers"] == [24]
        and BASELINE_ENGINE_DEFAULTS["mask_blend_mode"] == "auto"
        and BASELINE_ENGINE_DEFAULTS["invert_inpaint_mask"] is False
        and BASELINE_ENGINE_DEFAULTS["max_sequence_length"] == 16384
        and _AI_TOOLKIT_MAX_REF_DIM == 768,
        f"Expected V03 Heun->IPNDM 16K/768px baseline defaults, got {BASELINE_ENGINE_DEFAULTS}",
    )

    skip_early, u_skip_early, slg_skip_early, do_slg_early = resolve_skip_layer_regimes(
        uncond_skip_layers=[24],
        slg_layers=[24],
        slg_scale=0.5,
        slg_start=0.20,
        slg_stop=0.70,
        step_progress=0.10,
        depth=42,
    )
    skip_mid, u_skip_mid, slg_skip_mid, do_slg_mid = resolve_skip_layer_regimes(
        uncond_skip_layers=[24],
        slg_layers=[24],
        slg_scale=0.5,
        slg_start=0.20,
        slg_stop=0.70,
        step_progress=0.35,
        depth=42,
    )
    skip_late, u_skip_late, slg_skip_late, do_slg_late = resolve_skip_layer_regimes(
        uncond_skip_layers=[24],
        slg_layers=[24],
        slg_scale=0.5,
        slg_start=0.20,
        slg_stop=0.70,
        step_progress=0.75,
        depth=42,
    )
    _assert(
        not skip_early
        and len(u_skip_early) == 0
        and not do_slg_early
        and skip_mid
        and u_skip_mid == {24}
        and slg_skip_mid == {24}
        and do_slg_mid
        and not skip_late
        and len(u_skip_late) == 0
        and len(slg_skip_late) == 0
        and not do_slg_late,
        "Gate 5.B resolve_skip_layer_regimes failed window gating across [slg_start, slg_stop)",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        staged_map = stage_test_input_files(dest_dir=Path(tmpdir), force=True)
        _assert(
            "ref_c" in staged_map and staged_map["ref_c"].is_file(),
            "stage_test_input_files failed to stage ref_cutout_c.png",
        )
        with Image.open(staged_map["ref_c"]) as cut_img:
            cut_arr = np.asarray(cut_img)
            _assert(
                cut_img.mode == "RGBA"
                and cut_arr.shape == (512, 512, 4)
                and int(cut_arr[0, 0, 3]) == 0
                and int(cut_arr[256, 256, 3]) == 255,
                "ref_cutout_c.png must be a 512x512 RGBA cutout with transparent borders and opaque interior",
            )
        ref_512_p = Path(tmpdir) / "ref_512.png"
        test_bw = np.zeros((512, 512, 3), dtype=np.uint8)
        test_bw[256:, :, :] = 255
        Image.fromarray(test_bw, mode="RGB").save(ref_512_p)
        unboxed_2304 = _resolve_unboxed_reference_dims(
            ref_512_p, 1024, 1024, "ai_toolkit_isolated", max_vit_patches=2304
        )
        unboxed_576 = _resolve_unboxed_reference_dims(
            ref_512_p, 1024, 1024, "ai_toolkit_isolated", max_vit_patches=576
        )
        _assert(
            unboxed_2304 == (768, 768) and unboxed_576 == (512, 512),
            f"_resolve_unboxed_reference_dims failed adaptive 4D-RoPE scaling: 2304->{unboxed_2304}, 576->{unboxed_576}",
        )
        fb_patches, fb_gthw, fb_llm = pack_vision_patches_32px(
            ref_512_p, max_patches=2304
        )
        _assert(
            fb_patches.shape == (2304, 3, 2, 16, 16)
            and fb_gthw.tolist() == [[1, 48, 48]]
            and fb_llm == 576
            and abs(float(fb_patches.min().item()) - (-1.0)) < 1e-5
            and abs(float(fb_patches.max().item()) - 1.0) < 1e-5,
            f"pack_vision_patches_32px failed full-budget upsampling or [-1, 1] sRGB normalization: shape={fb_patches.shape}, range=[{fb_patches.min().item()}, {fb_patches.max().item()}]",
        )

        if HAS_VIZINJECTOR and VizInjectorClient is not None:
            with torch.inference_mode(), VizInjectorClient(prefer_vulkan=True) as viz:
                mini_vae = (
                    Flux2AutoencoderKL(
                        config=Flux2VAEConfig(
                            block_out_channels=(16, 16, 32, 32),
                            decoder_block_out_channels=(16, 16, 32, 32),
                            latent_channels=32,
                            layers_per_block=1,
                            norm_num_groups=8,
                        )
                    )
                    .to(device=dev, dtype=torch.float32)
                    .eval()
                )
                mini_qwen = (
                    Qwen3_5Model(
                        Qwen3_5Config(
                            text_config=Qwen3_5TextConfig(
                                vocab_size=256,
                                hidden_size=64,
                                intermediate_size=128,
                                num_hidden_layers=1,
                                num_attention_heads=2,
                                num_key_value_heads=2,
                                head_dim=32,
                                linear_conv_kernel_dim=4,
                                linear_key_head_dim=16,
                                linear_value_head_dim=16,
                                linear_num_key_heads=2,
                                linear_num_value_heads=2,
                            ),
                            vision_config=Qwen3_5VisionConfig(
                                depth=1,
                                hidden_size=32,
                                intermediate_size=64,
                                num_heads=4,
                                in_channels=3,
                                patch_size=16,
                                spatial_merge_size=2,
                                temporal_patch_size=2,
                                out_hidden_size=64,
                                num_position_embeddings=64,
                            ),
                        )
                    )
                    .to(device=dev, dtype=dtype)
                    .eval()
                )
                stage_harness = _Stage1UpgradeHarness(
                    device=dev,
                    vae=mini_vae,
                    text_encoder=mini_qwen,
                    vizinjector=viz,
                )
                t1_lat = philox_randn(
                    (1, 128, 4, 4), seed=777, stream_id=1, device="cpu"
                )
                t1_bundle = ConditionedVisionBundle(
                    vae_tensor=None,
                    vit_patches=None,
                    grid_thw=torch.tensor([[1, 24, 24]], dtype=torch.long),
                    llm_tokens=0,
                    latent_mask=torch.ones((1, 1, 4, 4), dtype=torch.float32),
                    packed_latents=t1_lat,
                    vit_embeds=None,
                    wire_vizj_bytes=None,
                    meta=VizInjectorResponseMeta(
                        orig_size=(64, 64),
                        vae_size=(64, 64),
                        vit_size=(384, 384),
                        grid_thw=(1, 24, 24),
                        llm_tokens=0,
                        backend_used="VIZJ Zero-Copy Cache",
                        edit_format="ai_toolkit_isolated",
                        input_role="photo_ref",
                        reference_zero_timestep=True,
                        reference_isolated_kv=True,
                        telemetry=VizInjectorTelemetry(),
                    ),
                )
                t1_vizj_path = Path(tmpdir) / "stage1_tier1.vizj"
                viz.save_vizj(
                    t1_bundle,
                    dest_path=t1_vizj_path,
                    tier=VIZJ_TIER_VAE_LATENT,
                    use_bf16=False,
                )
                upgrade_req = GenerationRequest(
                    prompt="Stage 1 Tier-1 ViT upgrade audit",
                    width=256,
                    height=256,
                    max_vit_patches=576,
                    edit_format="ai_toolkit_isolated",
                    prefer_vulkan=True,
                    seed=42,
                    soft_knee_tau=0.015,
                    use_reference_cache=True,
                )
                upgraded_rec, was_hit = stage_harness._condition_single_reference(
                    t1_vizj_path,
                    upgrade_req,
                    role_override="photo_ref",
                    framing_override="inscribed",
                    target_w_override=64,
                    target_h_override=64,
                    force_vit=True,
                )
                dec_rgb_ref = mini_vae.decode_packed_latents(
                    t1_lat.to(device=dev, non_blocking=True)
                )
                dec_rgb_cpu_ref = _evacuate_to_cpu(dec_rgb_ref)
                assert dec_rgb_cpu_ref is not None
                vit_expected_req = VizInjectorRequest(
                    target_width=64,
                    target_height=64,
                    max_vit_patches=576,
                    edit_mode_384_vit=True,
                    framing_mode="inscribed",
                    input_role="photo_ref",
                    edit_format="ai_toolkit_isolated",
                    emit_vae_branch=False,
                    emit_mask_branch=False,
                    emit_vit_branch=True,
                    deband_8bit=False,
                    prefer_vulkan=True,
                    seed=42,
                    mtf_boost=0.0,
                    soft_knee_tau=0.015,
                )
                expected_f32_vit = viz.condition_image(
                    dec_rgb_cpu_ref, vit_expected_req
                )
                pil_8bit_ref, _ = viz.postprocess_to_pil(
                    dec_rgb_cpu_ref, apply_bayer_dither=False
                )
                quantized_8bit_vit = viz.condition_image(
                    pil_8bit_ref[0], vit_expected_req
                )
                f32_upgrade_drift = float(
                    (upgraded_rec.vit_patches - expected_f32_vit.vit_patches)
                    .abs()
                    .max()
                    .item()
                )
                pil_quant_err = float(
                    (expected_f32_vit.vit_patches - quantized_8bit_vit.vit_patches)
                    .abs()
                    .max()
                    .item()
                )
                _assert(
                    not was_hit
                    and upgraded_rec.packed_latents is not None
                    and upgraded_rec.vit_patches is not None
                    and upgraded_rec.vit_embeds is not None
                    and upgraded_rec.llm_tokens == 144
                    and f32_upgrade_drift == 0.0
                    and pil_quant_err > 1e-4,
                    f"Stage 1 Tier-1 .vizj ViT upgrade drifted from direct FP32 tensor conditioning: f32_drift={f32_upgrade_drift}, pil_err={pil_quant_err}",
                )
                if dev.type == "cuda":
                    _assert(
                        upgraded_rec.packed_latents.is_pinned()
                        and upgraded_rec.vit_patches.is_pinned()
                        and upgraded_rec.vit_embeds.is_pinned(),
                        "Stage 1 upgraded CachedReferenceRecord tensors must be page-locked",
                    )
                stage_harness.prepare_for_exit()
                del (
                    stage_harness,
                    mini_vae,
                    mini_qwen,
                    dec_rgb_ref,
                    dec_rgb_cpu_ref,
                )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_preset = Path(tmpdir) / "default.json"
        prev_env = os.environ.get("SHADOWBOXER_DEFAULT_JSON")
        try:
            os.environ["SHADOWBOXER_DEFAULT_JSON"] = str(tmp_preset / "nonexistent.json")
            base_def = get_active_engine_defaults()
            _assert(
                not has_custom_default_preset()
                and base_def["early_solver"] == "heun"
                and base_def["late_solver"] == "ipndm",
                "Expected hardcoded V03 defaults when default.json is absent",
            )
            harness_def_req = build_default_request()
            _assert(
                harness_def_req.early_solver == "heun"
                and harness_def_req.late_solver == "ipndm"
                and abs(harness_def_req.early_cfg - 3.2) < 1e-6
                and abs(harness_def_req.late_cfg - 2.8) < 1e-6
                and harness_def_req.uncond_skip_layers == [24]
                and harness_def_req.layout_summary is None
                and harness_def_req.max_sequence_length == 16384,
                f"build_default_request() did not match V03 baseline: {harness_def_req}",
            )
            harness_def_req.prompt_mode = "layout_json"
            harness_def_req.prompt = "A custom studio scene summary without shadowing"
            _assert(
                "A custom studio scene summary without shadowing"
                in harness_def_req.compile_effective_prompt(),
                "compile_effective_prompt() shadowed edited prompt in layout_json mode",
            )
            os.environ["SHADOWBOXER_DEFAULT_JSON"] = str(tmp_preset)
            custom_req = GenerationRequest(
                prompt="Preset hot-reload test",
                system_prompt="Spatial layout system prompt",
                thinking_trace="Explicit spatial chain-of-thought",
                add_vision_id=False,
                early_solver="sde_gpu_pp",
                late_solver="ipndm",
                early_cfg=3.5,
                late_cfg=2.5,
                uncond_skip_layers=[22, 24],
                box_attn_isolation=0.85,
                composite_box_images_to_init=True,
                uncond_uses_references=False,
                mask_blend_mode="union",
                invert_inpaint_mask=True,
                denoise_strength=0.02,
            )
            custom_req.save_preset(tmp_preset)
            _assert(
                has_custom_default_preset(),
                "has_custom_default_preset() failed to detect saved default.json",
            )
            reloaded_req = GenerationRequest(prompt="Inherited preset test")
            _assert(
                reloaded_req.early_solver == "sde_gpu_pp"
                and abs(reloaded_req.early_cfg - 3.5) < 1e-6
                and reloaded_req.uncond_skip_layers == [22, 24]
                and abs(float(reloaded_req.box_attn_isolation or 0.0) - 0.85) < 1e-6
                and reloaded_req.composite_box_images_to_init is True
                and reloaded_req.uncond_uses_references is False
                and reloaded_req.system_prompt == "Spatial layout system prompt"
                and reloaded_req.thinking_trace == "Explicit spatial chain-of-thought"
                and reloaded_req.add_vision_id is False
                and reloaded_req.mask_blend_mode == "union"
                and reloaded_req.invert_inpaint_mask is True,
                f"GenerationRequest failed to inherit active default.json overrides: {reloaded_req}",
            )
        finally:
            if prev_env is None:
                os.environ.pop("SHADOWBOXER_DEFAULT_JSON", None)
            else:
                os.environ["SHADOWBOXER_DEFAULT_JSON"] = prev_env
            get_active_engine_defaults()

    req_cli_dn = build_default_request()
    _apply_box_denoise_spec(req_cli_dn, "0:0.15,1:0.65")
    _assert(
        abs(float(req_cli_dn.layout_items[0]["denoise_strength"]) - 0.15) < 1e-6
        and abs(float(req_cli_dn.layout_items[1]["denoise_strength"]) - 0.65) < 1e-6,
        f"_apply_box_denoise_spec failed to set per-box denoise overrides: {req_cli_dn.layout_items}",
    )

    c_bbox = pixel_bbox_to_abs_centralized(0, 0, 1023, 1023, height=1024, width=1024)
    _assert(
        c_bbox == [-512, -512, 512, 512], f"Unexpected centralized bbox: {c_bbox}"
    )
    norm_b = bbox_to_normalized(
        [256, 512, 768, 1024], height=1024, width=1024, coords_are_pixels=True
    )
    _assert(
        norm_b == (0.25, 0.5, 0.75, 1.0),
        f"Unexpected normalized bbox: {norm_b}",
    )
    px_b = bbox_to_pixel_rect(
        [-256, 0, 256, 512], height=1024, width=1024, coords_are_pixels=False
    )
    _assert(
        px_b == (256, 512, 768, 1024),
        f"Unexpected pixel rect from centralized bbox: {px_b}",
    )

    stale_summary = (
        'A sleek brushed-steel refrigerator door with a yellow sticky note that reads: '
        '"Remember to test SFT & Base permutations @ 1024x1024"'
    )
    reconciled_str = reconcile_layout_summary_quotes(
        stale_summary, ["SHIP SOVEREIGN STUDIO v2"]
    )
    _assert(
        '"SHIP SOVEREIGN STUDIO v2"' in reconciled_str
        and "Remember to test SFT" not in reconciled_str,
        f"reconcile_layout_summary_quotes failed to update stale quote: {reconciled_str}",
    )
    compiled_reconciled = json.loads(
        compile_layout_prompt(
            caption=stale_summary,
            items=[
                {
                    "category": "text",
                    "text": "SHIP SOVEREIGN STUDIO v2",
                    "bbox": [288, 144, 544, 480],
                }
            ],
            height=1024,
            width=1024,
            coords_are_pixels=True,
        )
    )
    _assert(
        '"SHIP SOVEREIGN STUDIO v2"' in compiled_reconciled["summary"]["caption"]
        and compiled_reconciled["items"][0]["text"] == "SHIP SOVEREIGN STUDIO v2",
        f"compile_layout_prompt failed single-point-of-truth quote reconciliation: {compiled_reconciled}",
    )

    box_subject_item = {
        "category": "person",
        "caption": "Subject portrait in moody lighting",
        "bbox": [256, 256, 768, 768],
        "image_path": "box_subject.png",
        "isolate_box_attn": 0.9,
        "paste_to_canvas": True,
        "denoise_strength": 0.03,
        "z_order": 1,
    }
    box_text_item = {
        "category": "text",
        "text": "SHADOWBOXER",
        "bbox": [64, 256, 192, 768],
        "z_order": 5,
    }
    req_poly = GenerationRequest(
        prompt="A cinematic composition with a portrait and tattoo overlay",
        prompt_mode="layout_json",
        thinking_trace="auto",
        height=1024,
        width=1024,
        layout_coords_are_pixels=True,
        reference_image_paths=["global_scene.png"],
        layout_items=[box_subject_item, box_text_item],
    )
    req_poly_rev = GenerationRequest(
        prompt="A cinematic composition with a portrait and tattoo overlay",
        prompt_mode="layout_json",
        thinking_trace="speculative:32",
        height=1024,
        width=1024,
        layout_coords_are_pixels=True,
        reference_image_paths=["global_scene.png"],
        layout_items=[
            dict(box_text_item, z_order=1),
            dict(box_subject_item, z_order=9),
        ],
    )
    auto_trace_str, auto_spec_tok = req_poly.resolve_thinking_trace()
    spec_trace_str, spec_cot_tok = req_poly_rev.resolve_thinking_trace()
    _assert(
        auto_trace_str is not None
        and auto_spec_tok == 0
        and "Picture 2" in auto_trace_str
        and '"SHADOWBOXER"' in auto_trace_str
        and auto_trace_str == compile_spatial_thinking_trace(req_poly_rev)
        and spec_trace_str is None
        and spec_cot_tok == 32,
        f"resolve_thinking_trace failed 'auto' or 'speculative:32' routing: auto={auto_trace_str}, spec_tok={spec_cot_tok}",
    )
    live_telem_trace, _ = req_poly.resolve_thinking_trace(
        conditioned_ref_telemetry={
            1: {
                "vae_size": (512, 512),
                "llm_tokens": 576,
                "has_cutout": True,
                "sharpness": 0.184,
            }
        }
    )
    _assert(
        live_telem_trace is not None
        and "vae=512x512" in live_telem_trace
        and "vit=576tok" in live_telem_trace
        and "cutout=True" in live_telem_trace
        and "sharp=0.184" in live_telem_trace,
        f"resolve_thinking_trace failed to inject live VizInjector telemetry: {live_telem_trace}",
    )
    _assert(
        get_canonical_box_order(
            req_poly_rev.layout_items, height=1024, width=1024, coords_are_pixels=True
        )
        == [1, 0],
        "Canonical coarse-to-fine box sort failed to order larger subject box before smaller text box",
    )
    ref_specs, box_pic_map = req_poly.resolve_visual_references()
    ref_specs_rev, box_pic_map_rev = req_poly_rev.resolve_visual_references()
    _assert(
        len(ref_specs) == 2
        and box_pic_map == {0: 2}
        and box_pic_map_rev == {1: 2}
        and [s["path"] for s in ref_specs] == [s["path"] for s in ref_specs_rev],
        f"Expected permutation-invariant visual reference specs, got {box_pic_map} vs {box_pic_map_rev}",
    )
    _assert(
        ref_specs[0]["norm_bbox"] is None
        and ref_specs[1]["norm_bbox"] == (0.25, 0.25, 0.75, 0.75)
        and abs(float(ref_specs[1]["isolate_box_attn"]) - 0.9) < 1e-6
        and abs(float(ref_specs[1]["denoise_strength"]) - 0.03) < 1e-6
        and ref_specs[1]["paste_to_canvas"] is True,
        f"Unexpected reference spec mapping: {ref_specs}",
    )
    compiled_poly = req_poly.compile_effective_prompt()
    compiled_poly_rev = req_poly_rev.compile_effective_prompt()
    _assert(
        compiled_poly == compiled_poly_rev
        and '"reference": "Picture 2"' in compiled_poly
        and "box_subject.png" not in compiled_poly
        and "isolate_box_attn" not in compiled_poly
        and "paste_to_canvas" not in compiled_poly
        and "denoise_strength" not in compiled_poly
        and "z_order" not in compiled_poly,
        "Polymorphic layout compiler failed permutation invariance or leaked internal box routing keys",
    )
    _assert(
        req_poly.effective_reference_zero_timestep is True
        and req_poly.effective_reference_isolated_kv is True,
        "Box-bound reference images must automatically enable t_ref=0 and IsoKV",
    )

    req_mixed_roles = GenerationRequest(
        prompt="Mixed role multimodal indexing and box-conformal framing test",
        prompt_mode="layout_json",
        edit_format="moss_image_edit",
        framing_mode="never_crop",
        composite_box_images_to_init=True,
        height=1024,
        width=1024,
        layout_coords_are_pixels=True,
        reference_image_paths=["global_bg_edit.png"],
        layout_items=[
            {
                "category": "structure",
                "caption": "Depth map guide",
                "bbox": [0, 0, 512, 512],
                "image_path": "box_depth.png",
                "input_role": "depth_map",
            },
            {
                "category": "mask",
                "caption": "Regional stroke mask",
                "bbox": [128, 128, 640, 640],
                "image_path": "box_mask.png",
                "input_role": "inpaint_mask",
            },
            {
                "category": "subject",
                "caption": "Foreground character reference",
                "bbox": [256, 512, 768, 1024],
                "image_path": "box_hero.png",
                "input_role": "photo_ref",
            },
            {
                "category": "object",
                "caption": "Explicitly unisolated box reference",
                "bbox": [600, 600, 960, 960],
                "image_path": "box_uniso.png",
                "input_role": "photo_ref",
                "isolate_box_attn": 0.0,
            },
        ],
    )
    mixed_specs, mixed_pic_map = req_mixed_roles.resolve_visual_references()
    _assert(
        len(mixed_specs) == 5 and mixed_pic_map == {2: 1, 3: 2},
        f"Expected Picture 1 & 2 strictly bound to photo_ref boxes, got map={mixed_pic_map}",
    )
    _assert(
        mixed_specs[0]["emits_vit"] is False
        and mixed_specs[0]["emits_dit_latents"] is True
        and mixed_specs[0]["emits_box_mask"] is False
        and mixed_specs[0]["picture_index"] is None
        and mixed_specs[0]["dit_ref_index"] == 1
        and mixed_specs[0]["framing_mode"] == "never_crop"
        and abs(float(mixed_specs[0]["isolate_box_attn"]) - 0.0) < 1e-6,
        f"Global moss_image_edit ref spec mismatch: {mixed_specs[0]}",
    )
    _assert(
        mixed_specs[1]["emits_vit"] is False
        and mixed_specs[1]["emits_dit_latents"] is True
        and mixed_specs[1]["emits_box_mask"] is False
        and mixed_specs[1]["picture_index"] is None
        and mixed_specs[1]["dit_ref_index"] == 2
        and mixed_specs[1]["paste_to_canvas"] is False
        and mixed_specs[1]["framing_mode"] == "inscribed"
        and abs(float(mixed_specs[1]["isolate_box_attn"]) - DEFAULT_BOX_REF_ISOLATION)
        < 1e-6,
        f"Box 0 depth_map ref spec failed control-map canvas protection: {mixed_specs[1]}",
    )
    _assert(
        mixed_specs[2]["input_role"] == "inpaint_mask"
        and mixed_specs[2]["emits_vit"] is False
        and mixed_specs[2]["emits_dit_latents"] is False
        and mixed_specs[2]["emits_box_mask"] is True
        and mixed_specs[2]["picture_index"] is None
        and mixed_specs[2]["dit_ref_index"] is None
        and mixed_specs[2]["paste_to_canvas"] is False
        and abs(float(mixed_specs[2]["isolate_box_attn"]) - 0.0) < 1e-6,
        f"Box 1 inpaint_mask ref spec failed mask routing invariants: {mixed_specs[2]}",
    )
    _assert(
        mixed_specs[3]["emits_vit"] is True
        and mixed_specs[3]["emits_dit_latents"] is True
        and mixed_specs[3]["picture_index"] == 1
        and mixed_specs[3]["dit_ref_index"] == 3
        and mixed_specs[3]["paste_to_canvas"] is True
        and mixed_specs[3]["framing_mode"] == "inscribed"
        and abs(float(mixed_specs[3]["isolate_box_attn"]) - DEFAULT_BOX_REF_ISOLATION)
        < 1e-6,
        f"Box 2 photo_ref spec failed box-conformal framing, default isolation, or Picture 1 parity: {mixed_specs[3]}",
    )
    _assert(
        mixed_specs[4]["emits_vit"] is True
        and mixed_specs[4]["picture_index"] == 2
        and mixed_specs[4]["dit_ref_index"] == 4
        and mixed_specs[4]["paste_to_canvas"] is True
        and abs(float(mixed_specs[4]["isolate_box_attn"]) - 0.0) < 1e-6,
        f"Box 3 explicit isolate_box_attn=0.0 override was not respected: {mixed_specs[4]}",
    )
    compiled_mixed = req_mixed_roles.compile_effective_prompt()
    _assert(
        '"reference": "Picture 1"' in compiled_mixed
        and '"reference": "Picture 2"' in compiled_mixed
        and '"reference": "Picture 3"' not in compiled_mixed,
        f"Mixed-role layout prompt desynchronized Picture index: {compiled_mixed}",
    )

    v11_norm_boxes = (
        (0.0, 0.0, 1.0, 0.5625),
        (0.0, 0.0, 1.0, 0.5625),
        (0.4375, 0.5, 1.0, 1.0),
    )
    omega_01 = _boxes_overlap_coefficient(v11_norm_boxes[0], v11_norm_boxes[1])
    omega_02 = _boxes_overlap_coefficient(v11_norm_boxes[0], v11_norm_boxes[2])
    v11_collab = _compute_collaborating_ref_groups(3, v11_norm_boxes)
    _assert(
        abs(omega_01 - 1.0) < 1e-6
        and abs(omega_02 - 0.125) < 1e-6
        and v11_collab == ((0, 1), (0, 1), (2,)),
        f"Szymkiewicz-Simpson IsoKV peerhood failed on V11 boxes: omega_01={omega_01}, omega_02={omega_02}, groups={v11_collab}",
    )
    chain_boxes = (
        (0.0, 0.0, 0.5, 0.5),
        (0.0, 0.25, 0.5, 0.75),
        (0.0, 0.5, 0.5, 1.0),
    )
    chain_collab = _compute_collaborating_ref_groups(3, chain_boxes)
    _assert(
        chain_collab == ((0, 1), (0, 1, 2), (1, 2)),
        f"Direct pairwise IsoKV peerhood must prevent transitive A<->C leakage: got {chain_collab}",
    )

    aperture_mask = build_box_aperture_mask(
        layout_items=[{"bbox": [256, 256, 768, 768]}],
        height=1024,
        width=1024,
        coords_are_pixels=True,
        feather_radius_lat=2.0,
        device=dev,
    )
    _assert(
        aperture_mask is not None and aperture_mask.shape == (1, 1, 64, 64),
        f"Unexpected box aperture mask shape: {None if aperture_mask is None else aperture_mask.shape}",
    )
    _assert(
        float(aperture_mask[0, 0, 32, 32].item()) == 1.0
        and float(aperture_mask[0, 0, 0, 0].item()) == 0.0
        and 0.0 < float(aperture_mask[0, 0, 15, 32].item()) < 1.0,
        "Box aperture mask failed interior/feather/exterior boundary conditions",
    )
    aperture_kronecker = build_box_aperture_mask(
        layout_items=[{"bbox": [256, 256, 768, 768]}],
        height=1024,
        width=1024,
        coords_are_pixels=True,
        feather_radius_lat=0.0,
        device=dev,
    )
    _assert(
        aperture_kronecker is not None
        and float(aperture_kronecker[0, 0, 16, 32].item()) == 1.0
        and float(aperture_kronecker[0, 0, 15, 32].item()) == 0.0,
        "0.0-feather build_box_aperture_mask failed exact Kronecker step boundary",
    )

    sub_box_mask = torch.zeros((1, 1, 16, 16), device=dev, dtype=torch.float32)
    sub_box_mask[:, :, 4:12, 4:12] = 1.0
    splatted_box_mask = splat_box_masks_to_canvas(
        base_mask=None,
        mask_entries=[(sub_box_mask, (0.25, 0.25, 0.75, 0.75))],
        h_lat=32,
        w_lat=32,
        device=dev,
    )
    _assert(
        splatted_box_mask is not None
        and splatted_box_mask.shape == (1, 1, 32, 32)
        and float(splatted_box_mask[0, 0, 16, 16].item()) == 1.0
        and float(splatted_box_mask[0, 0, 9, 9].item()) == 0.0
        and float(splatted_box_mask[0, 0, 2, 2].item()) == 0.0,
        "splat_box_masks_to_canvas failed to place sub-box stroke mask at normalized bbox coordinates",
    )

    horizon_map = build_box_denoise_horizon_map(
        layout_items=[
            {
                "bbox": [0, 0, 512, 512],
                "inpaint_box": True,
                "denoise_strength": 0.25,
            },
            {
                "bbox": [512, 512, 1024, 1024],
                "inpaint_box": True,
                "denoise_strength": 0.85,
            },
        ],
        height=1024,
        width=1024,
        coords_are_pixels=True,
        feather_radius_lat=1.5,
        default_denoise_strength=1.0,
        device=dev,
        only_flagged_boxes=True,
        return_none_if_uniform=True,
    )
    _assert(
        horizon_map is not None
        and horizon_map.shape == (1, 1, 64, 64)
        and abs(float(horizon_map[0, 0, 16, 16].item()) - 0.25) < 1e-5
        and abs(float(horizon_map[0, 0, 48, 48].item()) - 0.85) < 1e-5
        and abs(float(horizon_map[0, 0, 8, 56].item()) - 1.0) < 1e-5,
        "build_box_denoise_horizon_map failed per-box denoise horizon synthesis",
    )

    nested_horizon_map = build_box_denoise_horizon_map(
        layout_items=[
            {
                "bbox": [128, 128, 896, 896],
                "inpaint_box": True,
                "denoise_strength": 0.80,
                "z_order": 1,
            },
            {
                "bbox": [384, 384, 640, 640],
                "inpaint_box": True,
                "denoise_strength": 0.20,
                "z_order": 5,
            },
        ],
        height=1024,
        width=1024,
        coords_are_pixels=True,
        feather_radius_lat=1.0,
        default_denoise_strength=1.0,
        device=dev,
        only_flagged_boxes=True,
        return_none_if_uniform=True,
    )
    _assert(
        nested_horizon_map is not None
        and abs(float(nested_horizon_map[0, 0, 32, 32].item()) - 0.20) < 1e-6
        and abs(float(nested_horizon_map[0, 0, 12, 12].item()) - 0.80) < 1e-6,
        "Porter-Duff occlusion in build_box_denoise_horizon_map failed to keep nested child interior 100% uncontaminated",
    )

    warped_horizon = warp_denoise_to_sigma_horizon(horizon_map, shift=5.0)
    _assert(
        isinstance(warped_horizon, torch.Tensor)
        and abs(float(warped_horizon[0, 0, 16, 16].item()) - (1.25 / 2.0)) < 1e-5
        and abs(float(warped_horizon[0, 0, 8, 56].item()) - 1.0) < 1e-5,
        "Mobius horizon warp failed on build_box_denoise_horizon_map tensor",
    )

    m_paint = torch.zeros((1, 1, 64, 64), device=dev, dtype=torch.float32)
    m_paint[:, :, :32, :] = 1.0
    m_alpha = torch.full((1, 1, 64, 64), 0.5, device=dev, dtype=torch.float32)

    m_auto_union = compose_latent_inpaint_mask(
        paint_mask=m_paint,
        box_mask=aperture_mask,
        alpha_mask=m_alpha,
        blend_mode="auto",
        preserve_outside_boxes=False,
        invert_mask=False,
        device=dev,
    )
    _assert(
        m_auto_union is not None
        and float(m_auto_union[0, 0, 5, 5].item()) == 1.0
        and float(m_auto_union[0, 0, 40, 32].item()) == 1.0
        and float(m_auto_union[0, 0, 55, 5].item()) == 0.0,
        "compose_latent_inpaint_mask auto-union failed",
    )

    m_auto_inter = compose_latent_inpaint_mask(
        paint_mask=m_paint,
        box_mask=aperture_mask,
        alpha_mask=m_alpha,
        blend_mode="auto",
        preserve_outside_boxes=True,
        invert_mask=False,
        device=dev,
    )
    _assert(
        m_auto_inter is not None
        and float(m_auto_inter[0, 0, 5, 5].item()) == 0.0
        and float(m_auto_inter[0, 0, 24, 32].item()) == 1.0
        and float(m_auto_inter[0, 0, 40, 32].item()) == 0.0,
        "compose_latent_inpaint_mask auto-intersect (preserve_outside_boxes) failed",
    )

    m_inv_alpha = compose_latent_inpaint_mask(
        paint_mask=m_paint,
        box_mask=aperture_mask,
        alpha_mask=m_alpha,
        blend_mode="alpha_only",
        invert_mask=True,
        device=dev,
    )
    _assert(
        m_inv_alpha is not None
        and len(SUPPORTED_MASK_BLEND_MODES) == 6
        and abs(float(m_inv_alpha[0, 0, 10, 10].item()) - 0.5) < 1e-6,
        "compose_latent_inpaint_mask alpha_only + invert_mask failed",
    )

    with torch.inference_mode():
        mini_dit = (
            MOSSImageTransformer(
                config=MOSSImageConfig(
                    dim=128,
                    ffn_dim=256,
                    num_heads=4,
                    num_layers=1,
                    text_dim=128,
                    freq_dim=32,
                    latent_channels=128,
                )
            )
            .to(device=dev, dtype=torch.float32)
            .eval()
        )
        fg_cutout_m = torch.zeros((1, 1, 6, 6), device=dev, dtype=torch.float32)
        fg_cutout_m[:, :, 2:4, 2:4] = 1.0
        bias_occ = mini_dit._get_box_aperture_bias(
            height=8,
            width=8,
            ref_token_lengths=[36, 36],
            text_tokens=8,
            ref_shapes=[(6, 6), (6, 6)],
            ref_bboxes=[(0.125, 0.125, 0.875, 0.875), (0.125, 0.125, 0.875, 0.875)],
            reference_box_isolation=[0.85, 0.90],
            feather_radius_lat=0.0,
            device=dev,
            dtype=torch.float32,
            reference_masks=[None, fg_cutout_m],
            reference_roles=["photo_ref", "photo_ref"],
            reference_z_orders=[1.0, 5.0],
        )
        q_opaque_center = 4 * 8 + 4
        q_transp_corner = 1 * 8 + 1
        k_bg_table = 64 + (3 * 6 + 3)
        k_fg_bowl_opaque = 64 + 36 + (2 * 6 + 2)
        _assert(
            bias_occ is not None
            and float(bias_occ[0, 0, q_opaque_center, k_fg_bowl_opaque].item()) == 0.0
            and float(bias_occ[0, 0, q_opaque_center, k_bg_table].item()) < -10.0
            and float(bias_occ[0, 0, q_transp_corner, k_bg_table].item()) == 0.0
            and float(bias_occ[0, 0, q_transp_corner, k_fg_bowl_opaque].item()) < -15.0,
            "Silhouette-conformal foreground occlusion failed: transparent cutout corner must leave parent reference unattenuated (0.0) while opaque center occludes parent",
        )
        del mini_dit, bias_occ

        boxed_w, boxed_h = _resolve_boxed_reference_dims(
            (0.25, 0.25, 0.75, 0.75), 1024, 1024
        )
        _assert(
            (boxed_w, boxed_h) == (512, 512),
            f"_resolve_boxed_reference_dims mismatch: {(boxed_w, boxed_h)}",
        )

        unpacked_ramp = (
            (torch.arange(16, device=dev, dtype=torch.float32) + 0.5)
            .view(1, 1, 1, 16)
            .expand(1, 32, 16, 16)
        )
        packed_ramp = (
            unpacked_ramp.reshape(1, 32, 8, 2, 8, 2)
            .permute(0, 1, 3, 5, 2, 4)
            .reshape(1, 128, 8, 8)
        )
        res_subphase = resample_packed_latents(packed_ramp, 16, 16)
        unpacked_sub = (
            res_subphase.reshape(1, 32, 2, 2, 16, 16)
            .permute(0, 1, 4, 2, 5, 3)
            .reshape(1, 32, 32, 32)
        )
        res_naive = F.interpolate(
            packed_ramp, size=(16, 16), mode="bilinear", align_corners=False
        )
        unpacked_naive = (
            res_naive.reshape(1, 32, 2, 2, 16, 16)
            .permute(0, 1, 4, 2, 5, 3)
            .reshape(1, 32, 32, 32)
        )
        gt_ramp_32 = (
            (torch.arange(32, device=dev, dtype=torch.float32) + 0.5) * 0.5
        ).view(1, 1, 1, 32)
        sub_err = float(
            (unpacked_sub[:, :, :, 2:-2] - gt_ramp_32[:, :, :, 2:-2])
            .abs()
            .max()
            .item()
        )
        naive_err = float(
            (unpacked_naive[:, :, :, 2:-2] - gt_ramp_32[:, :, :, 2:-2])
            .abs()
            .max()
            .item()
        )
        _assert(
            sub_err < 1e-5 and naive_err > 0.2,
            f"resample_packed_latents failed sub-phase preservation: sub_err={sub_err}, naive_err={naive_err}",
        )

        base_canvas = torch.zeros((1, 128, 32, 32), device=dev, dtype=torch.float32)
        box_lat_a = torch.full((1, 128, 16, 16), 2.5, device="cpu", dtype=torch.float32)
        splat_canvas, splat_alpha, splat_dn_map = splat_box_latents_to_canvas(
            base_canvas=base_canvas,
            splat_entries=[(box_lat_a, (0.25, 0.25, 0.75, 0.75), None, 0.60)],
            h_lat=32,
            w_lat=32,
            latent_channels=128,
            feather_radius_lat=2.0,
            device=dev,
            default_denoise_strength=0.75,
            return_denoise_map=True,
        )
        _assert(
            splat_canvas.shape == (1, 128, 32, 32)
            and splat_alpha.shape == (1, 1, 32, 32)
            and splat_dn_map.shape == (1, 1, 32, 32),
            "splat_box_latents_to_canvas output shape mismatch",
        )
        _assert(
            abs(float(splat_canvas[0, 0, 16, 16].item()) - 2.5) < 1e-5
            and float(splat_canvas[0, 0, 0, 0].item()) == 0.0
            and 0.0 < float(splat_canvas[0, 0, 8, 16].item()) < 2.5
            and abs(float(splat_dn_map[0, 0, 16, 16].item()) - 0.60) < 1e-5,
            "splat_box_latents_to_canvas failed interior/feather/exterior composite or denoise map values",
        )

        cutout_half = torch.full((1, 1, 16, 16), 0.5, device="cpu", dtype=torch.float32)
        _, cut_tnorm_alpha = splat_box_latents_to_canvas(
            base_canvas=base_canvas,
            splat_entries=[(box_lat_a, (0.25, 0.25, 0.75, 0.75), cutout_half, 0.60)],
            h_lat=32,
            w_lat=32,
            latent_channels=128,
            feather_radius_lat=2.0,
            device=dev,
            return_denoise_map=False,
        )
        _assert(
            abs(float(cut_tnorm_alpha[0, 0, 16, 16].item()) - 0.5) < 1e-6
            and abs(float(cut_tnorm_alpha[0, 0, 9, 16].item()) - 0.5) < 1e-6,
            "Gate 5.C Single-authority Godel t-norm failed: VizInjector cutout mask was double-feathered against box ramp",
        )

        _, kronecker_alpha, kronecker_dn = splat_box_latents_to_canvas(
            base_canvas=base_canvas,
            splat_entries=[(box_lat_a, (0.25, 0.25, 0.75, 0.75), None, 0.45)],
            h_lat=32,
            w_lat=32,
            latent_channels=128,
            feather_radius_lat=0.0,
            device=dev,
            default_denoise_strength=0.75,
            return_denoise_map=True,
        )
        _assert(
            float(kronecker_alpha[0, 0, 8, 8].item()) == 1.0
            and float(kronecker_alpha[0, 0, 7, 8].item()) == 0.0
            and abs(float(kronecker_dn[0, 0, 8, 8].item()) - 0.45) < 1e-5,
            "0.0-feather splat_box_latents_to_canvas failed exact Kronecker perimeter support",
        )

        _, low_dn_alpha = splat_box_latents_to_canvas(
            base_canvas=base_canvas,
            splat_entries=[(box_lat_a, (0.25, 0.25, 0.75, 0.75), None, 0.05)],
            h_lat=32,
            w_lat=32,
            latent_channels=128,
            feather_radius_lat=2.0,
            device=dev,
            return_denoise_map=False,
        )
        _assert(
            float(low_dn_alpha[0, 0, 8, 16].item())
            > float(splat_alpha[0, 0, 8, 16].item()) + 0.15,
            "Horizon-proportional feathering failed to tighten transition width for low-denoise preservation box",
        )

        bg_gauss = philox_randn((1, 128, 32, 32), seed=101, stream_id=1, device=dev)
        fg_gauss = philox_randn((1, 128, 16, 16), seed=202, stream_id=2, device="cpu")
        cov_canvas, cov_alpha = splat_box_latents_to_canvas(
            base_canvas=bg_gauss,
            splat_entries=[(fg_gauss, (0.25, 0.25, 0.75, 0.75), None, 0.50)],
            h_lat=32,
            w_lat=32,
            latent_channels=128,
            feather_radius_lat=3.0,
            device=dev,
            return_denoise_map=False,
        )
        feather_band = (cov_alpha[0, 0] > 0.25) & (cov_alpha[0, 0] < 0.75)
        cell_rms = torch.sqrt(torch.mean(cov_canvas[0] * cov_canvas[0], dim=0))
        feather_rms = float(cell_rms[feather_band].mean().item())
        _assert(
            feather_rms > 0.94,
            f"Covariance-preserving sub-phase splatting suffered variance droop in feather zone: rms={feather_rms}",
        )

        no_base_canvas, no_base_alpha, no_base_dn = splat_box_latents_to_canvas(
            base_canvas=None,
            splat_entries=[(fg_gauss, (0.25, 0.25, 0.75, 0.75), None, 0.40)],
            h_lat=32,
            w_lat=32,
            latent_channels=128,
            feather_radius_lat=3.0,
            device=dev,
            default_denoise_strength=0.75,
            return_denoise_map=True,
        )
        no_base_feather_rms = float(
            torch.sqrt(torch.mean(no_base_canvas[0] * no_base_canvas[0], dim=0))[
                (no_base_alpha[0, 0] > 0.15) & (no_base_alpha[0, 0] < 0.85)
            ]
            .mean()
            .item()
        )
        _assert(
            no_base_feather_rms > 0.94
            and abs(float(no_base_dn[0, 0, 16, 16].item()) - 0.40) < 1e-5
            and abs(float(no_base_dn[0, 0, 0, 0].item()) - 1.0) < 1e-5
            and 0.40 < float(no_base_dn[0, 0, 8, 16].item()) < 1.0,
            f"base_canvas=None splatting failed unattenuated feather energy or exterior 1.0 release ramp: rms={no_base_feather_rms}",
        )

        parent_lat = torch.full((1, 128, 24, 24), 2.0, device="cpu", dtype=torch.float32)
        child_lat = torch.full((1, 128, 8, 8), -2.0, device="cpu", dtype=torch.float32)
        p_entry = (parent_lat, (0.125, 0.125, 0.875, 0.875), None, 0.80)
        c_entry = (child_lat, (0.375, 0.375, 0.625, 0.625), None, 0.15)
        h_can_pc, h_alp_pc, h_dn_pc = splat_box_latents_to_canvas(
            base_canvas=base_canvas,
            splat_entries=[p_entry, c_entry],
            h_lat=32,
            w_lat=32,
            latent_channels=128,
            feather_radius_lat=1.5,
            device=dev,
            return_denoise_map=True,
        )
        h_can_cp, h_alp_cp, h_dn_cp = splat_box_latents_to_canvas(
            base_canvas=base_canvas,
            splat_entries=[c_entry, p_entry],
            h_lat=32,
            w_lat=32,
            latent_channels=128,
            feather_radius_lat=1.5,
            device=dev,
            return_denoise_map=True,
        )
        _assert(
            float((h_can_pc - h_can_cp).abs().max().item()) < 1e-6
            and float((h_alp_pc - h_alp_cp).abs().max().item()) < 1e-6
            and float((h_dn_pc - h_dn_cp).abs().max().item()) < 1e-6
            and abs(float(h_can_pc[0, 0, 16, 16].item()) - (-2.0)) < 1e-6
            and abs(float(h_dn_pc[0, 0, 16, 16].item()) - 0.15) < 1e-6
            and abs(float(h_can_pc[0, 0, 8, 8].item()) - 2.0) < 1e-6
            and abs(float(h_dn_pc[0, 0, 8, 8].item()) - 0.80) < 1e-6,
            "Porter-Duff hierarchical occlusion failed 100% uncontaminated child interior or permutation invariance",
        )

        child_cutout_mask = torch.zeros((1, 1, 16, 16), device="cpu", dtype=torch.float32)
        child_cutout_mask[:, :, 4:12, 4:12] = 1.0
        cut_child_lat = torch.full((1, 128, 16, 16), -2.0, device="cpu", dtype=torch.float32)
        cut_can, cut_alp, cut_dn = splat_box_latents_to_canvas(
            base_canvas=base_canvas,
            splat_entries=[
                (parent_lat, (0.125, 0.125, 0.875, 0.875), None, 0.80, None, 1.0),
                (
                    cut_child_lat,
                    (0.25, 0.25, 0.75, 0.75),
                    child_cutout_mask,
                    0.15,
                    None,
                    5.0,
                ),
            ],
            h_lat=32,
            w_lat=32,
            latent_channels=128,
            feather_radius_lat=0.0,
            device=dev,
            return_denoise_map=True,
        )
        _assert(
            abs(float(cut_can[0, 0, 16, 16].item()) - (-2.0)) < 1e-6
            and abs(float(cut_dn[0, 0, 16, 16].item()) - 0.15) < 1e-6
            and abs(float(cut_can[0, 0, 9, 9].item()) - 2.0) < 1e-6
            and abs(float(cut_dn[0, 0, 9, 9].item()) - 0.80) < 1e-6
            and float(cut_alp[0, 0, 9, 9].item()) == 1.0,
            "Porter-Duff cutout child on parent surface failed: opaque center must be -2.0 (d=0.15) while transparent corner reveals parent +2.0 (d=0.80)",
        )

        z_low_lat = torch.full((1, 128, 16, 16), -2.0, device="cpu", dtype=torch.float32)
        z_high_lat = torch.full((1, 128, 16, 16), 2.0, device="cpu", dtype=torch.float32)
        z_can, _, z_dn = splat_box_latents_to_canvas(
            base_canvas=base_canvas,
            splat_entries=[
                (z_low_lat, (0.25, 0.25, 0.75, 0.75), None, 0.80, None, 1.0),
                (z_high_lat, (0.25, 0.25, 0.75, 0.75), None, 0.20, None, 9.0),
            ],
            h_lat=32,
            w_lat=32,
            latent_channels=128,
            feather_radius_lat=1.5,
            device=dev,
            return_denoise_map=True,
        )
        _assert(
            abs(float(z_can[0, 0, 16, 16].item()) - 2.0) < 1e-6
            and abs(float(z_dn[0, 0, 16, 16].item()) - 0.20) < 1e-6,
            "z_order Porter-Duff dominance in splat_box_latents_to_canvas failed to elevate foreground layer",
        )

        ovr_can, ovr_alpha, ovr_dn_map = splat_box_latents_to_canvas(
            base_canvas=base_canvas,
            splat_entries=[(parent_lat, (0.125, 0.125, 0.875, 0.875), None, 0.15)],
            h_lat=32,
            w_lat=32,
            latent_channels=128,
            feather_radius_lat=1.5,
            device=dev,
            return_denoise_map=True,
            aperture_overrides=[((0.375, 0.375, 0.625, 0.625), 0.65, 1.0)],
        )
        _assert(
            abs(float(ovr_dn_map[0, 0, 16, 16].item()) - 0.65) < 1e-5
            and abs(float(ovr_dn_map[0, 0, 6, 6].item()) - 0.15) < 1e-5
            and float(ovr_alpha[0, 0, 16, 16].item()) == 0.0
            and float(ovr_alpha[0, 0, 6, 6].item()) == 1.0,
            "Gate 5.A aperture_overrides failed to carve union_alpha or elevate denoise temperature inside nested typography box",
        )
        v_dit_fg = torch.full((1, 128, 32, 32), -0.75, device=dev, dtype=torch.float32)
        v_coupled_carved = couple_splat_velocity_prior(
            v_dit=v_dit_fg,
            latents=torch.zeros_like(ovr_can),
            splat_canvas=ovr_can,
            splat_union_alpha=ovr_alpha,
            splat_denoise_map=ovr_dn_map,
            sigma_curr=0.65,
        )
        _assert(
            abs(float(v_coupled_carved[0, 0, 16, 16].item()) - (-0.75)) < 1e-6
            and abs(float(v_coupled_carved[0, 0, 6, 6].item()) - (-0.75)) > 0.5,
            "Gate 5.A couple_splat_velocity_prior must leave v_dit completely unattenuated inside carved foreground typography aperture",
        )

        v11_test_items = [
            {
                "label": "object",
                "bbox": [0, 0, 1024, 576],
                "image_path": "ref_a.png",
                "input_role": "photo_ref",
                "paste_to_canvas": True,
                "isolate_box_attn": 0.85,
                "denoise_strength": 0.55,
                "z_order": 1,
                "enabled": True,
            },
            {
                "label": "object",
                "bbox": [0, 0, 1024, 576],
                "image_path": "control.png",
                "input_role": "depth_map",
                "isolate_box_attn": 0.85,
                "z_order": 2,
                "enabled": True,
            },
            {
                "label": "text",
                "text": "SOVEREIGN V11 CONVERGED",
                "bbox": [256, 112, 512, 464],
                "denoise_strength": 0.65,
                "z_order": 5,
                "enabled": True,
            },
        ]
        v11_ovr, v11_attn_carve = resolve_foreground_aperture_carveouts(
            layout_items=v11_test_items,
            height=1024,
            width=1024,
            coords_are_pixels=True,
            splat_boxes=[(0.0, 0.0, 1.0, 0.5625)],
            splatted_box_indices={0},
            ref_bboxes=[(0.0, 0.0, 1.0, 0.5625), (0.0, 0.0, 1.0, 0.5625)],
            ref_z_orders=[1.0, 2.0],
            default_isolation=0.85,
        )
        _assert(
            len(v11_ovr) == 1
            and abs(float(v11_ovr[0][1]) - 0.65) < 1e-6
            and abs(float(v11_ovr[0][2]) - 1.0) < 1e-6
            and len(v11_attn_carve) == 1
            and abs(float(v11_attn_carve[0][1]) - 0.85) < 1e-6
            and v11_attn_carve[0][2] == 5.0,
            f"Gate 5.A resolve_foreground_aperture_carveouts failed V11 synthesis: ovr={v11_ovr}, attn={v11_attn_carve}",
        )

        v11_scaled_512 = scale_layout_items_to_canvas(
            v11_test_items,
            height=512,
            width=512,
            src_height=1024,
            src_width=1024,
            coords_are_pixels=True,
        )
        _assert(
            v11_scaled_512[0]["bbox"] == [0, 0, 512, 288]
            and v11_scaled_512[2]["bbox"] == [128, 48, 256, 224],
            f"scale_layout_items_to_canvas failed V11 512x512 scaling: {v11_scaled_512}",
        )

        alpha_sweep = torch.tensor(
            [
                0.0,
                1e-8,
                1e-7,
                1e-6,
                1.0001e-6,
                1e-5,
                1e-4,
                1e-3,
                1e-2,
                0.1,
                0.5,
                1.0,
            ],
            device=dev,
            dtype=torch.float32,
        ).view(1, 1, 1, -1)
        v_floor_out = couple_splat_velocity_prior(
            v_dit=torch.zeros((1, 4, 1, 12), device=dev, dtype=torch.float32),
            latents=torch.ones((1, 4, 1, 12), device=dev, dtype=torch.float32),
            splat_canvas=torch.zeros((1, 4, 1, 12), device=dev, dtype=torch.float32),
            splat_union_alpha=alpha_sweep,
            splat_denoise_map=torch.full(
                (1, 1, 1, 12), 0.85, device=dev, dtype=torch.float32
            ),
            sigma_curr=0.85,
        )
        v_floor_1d = v_floor_out[0, 0, 0]
        jump_at_1e6 = abs(
            float(v_floor_1d[4].item()) - float(v_floor_1d[3].item())
        )
        _assert(
            bool(torch.isfinite(v_floor_out).all().item())
            and float(v_floor_1d[0].item()) == 0.0
            and bool(torch.all(v_floor_1d[1:] >= v_floor_1d[:-1]).item())
            and jump_at_1e6 < 1e-6,
            f"Gate 5.C Micro-floor discontinuity detected at alpha=1e-6: jump={jump_at_1e6}, curve={v_floor_1d.tolist()}",
        )

        can_45, alp_45, dn_45 = splat_box_latents_to_canvas(
            base_canvas=base_canvas,
            splat_entries=[(box_lat_a, (0.25, 0.25, 0.75, 0.75), None, 0.45)],
            h_lat=32,
            w_lat=32,
            latent_channels=128,
            feather_radius_lat=2.0,
            device=dev,
            default_denoise_strength=1.0,
            return_denoise_map=True,
        )
        warped_dn_45 = warp_denoise_to_sigma_horizon(dn_45, shift=5.0)
        assert isinstance(warped_dn_45, torch.Tensor)
        v_mid_45 = couple_splat_velocity_prior(
            v_dit=torch.zeros_like(can_45),
            latents=torch.zeros_like(can_45),
            splat_canvas=can_45,
            splat_union_alpha=alp_45,
            splat_denoise_map=warped_dn_45,
            sigma_curr=0.82,
        )
        outer_cell_v = float(v_mid_45[0, :, 8, 16].abs().mean().item())
        inner_cell_v = float(v_mid_45[0, :, 9, 16].abs().mean().item())
        _assert(
            outer_cell_v > 0.02 and inner_cell_v > 0.15,
            f"Gate 5.C Feather rim collapsed to 0 at mid-trajectory sigma=0.82 under shift=5.0 Mobius warp: outer={outer_cell_v}, inner={inner_cell_v}",
        )

        v_dit_zero = torch.zeros((1, 128, 32, 32), device=dev, dtype=torch.float32)
        lat_high = torch.ones((1, 128, 32, 32), device=dev, dtype=torch.float32)
        v_coupled_hi = couple_splat_velocity_prior(
            v_dit=v_dit_zero,
            latents=lat_high,
            splat_canvas=splat_canvas,
            splat_union_alpha=splat_alpha,
            splat_denoise_map=splat_dn_map,
            sigma_curr=0.90,
        )
        v_expected_splat = (1.0 - 2.5) / 0.90
        v_interior_hi = float(v_coupled_hi[0, 0, 16, 16].item())
        rim_alpha = float(splat_alpha[0, 0, 8, 16].item())
        rim_splat_val = float(splat_canvas[0, 0, 8, 16].item())
        rim_v_splat = (1.0 - rim_splat_val) / 0.90
        rim_dn = float(splat_dn_map[0, 0, 8, 16].item())
        rim_sh = min(0.96, rim_dn)
        rim_hb = max(0.015, min(0.18, 0.35 * rim_sh))
        rim_tau = max(0.0, min(1.0, (0.90 - (rim_sh - rim_hb)) / (2.0 * rim_hb)))
        rim_psi = 0.5 * (1.0 - math.cos(math.pi * rim_tau))
        rim_gmax = max(0.35, min(1.0, 1.0 - 0.38 * rim_sh))
        naive_rim_v = abs(rim_alpha * rim_gmax * rim_psi * rim_v_splat)
        actual_rim_v = abs(float(v_coupled_hi[0, 0, 8, 16].item()))
        _assert(
            float(v_coupled_hi[0, 0, 0, 0].abs().item()) == 0.0
            and v_expected_splat < v_interior_hi < 0.0
            and actual_rim_v > naive_rim_v * 1.15,
            f"couple_splat_velocity_prior failed interior/perimeter/exterior coupling at sigma=0.90: interior={v_interior_hi}, rim={actual_rim_v} vs naive={naive_rim_v}",
        )
        v_dit_ones = torch.ones((1, 128, 32, 32), device=dev, dtype=torch.float32)
        v_coupled_lo = couple_splat_velocity_prior(
            v_dit=v_dit_ones,
            latents=lat_high,
            splat_canvas=splat_canvas,
            splat_union_alpha=splat_alpha,
            splat_denoise_map=splat_dn_map,
            sigma_curr=0.20,
        )
        _assert(
            float((v_coupled_lo - v_dit_ones).abs().max().item()) == 0.0,
            "couple_splat_velocity_prior must hand off cleanly to v_dit below the denoise threshold",
        )

        dn_low = torch.full((1, 1, 32, 32), 0.05, device=dev, dtype=torch.float32)
        alpha_full = torch.ones((1, 1, 32, 32), device=dev, dtype=torch.float32)
        noise_ref = torch.ones((1, 128, 32, 32), device=dev, dtype=torch.float32) * 0.5
        s_low = 0.08
        dt_step = -0.25
        z_exact_bridge = (1.0 - s_low) * splat_canvas + s_low * noise_ref
        v_exact_bridge = noise_ref - splat_canvas
        v_cfl_exact = couple_splat_velocity_prior(
            v_dit=v_exact_bridge,
            latents=z_exact_bridge,
            splat_canvas=splat_canvas,
            splat_union_alpha=alpha_full,
            splat_denoise_map=dn_low,
            sigma_curr=s_low,
            noise=noise_ref,
            step_dt=dt_step,
        )
        _assert(
            float((v_cfl_exact - v_exact_bridge).abs().max().item()) < 1e-5,
            "Exact bridge state must have zero drift restorer bias in couple_splat_velocity_prior",
        )
        drift_err = 0.40
        z_perturbed = z_exact_bridge + drift_err
        v_cfl_pert = couple_splat_velocity_prior(
            v_dit=v_exact_bridge,
            latents=z_perturbed,
            splat_canvas=splat_canvas,
            splat_union_alpha=alpha_full,
            splat_denoise_map=dn_low,
            sigma_curr=s_low,
            noise=noise_ref,
            step_dt=dt_step,
        )
        z_next_expected = z_exact_bridge + dt_step * v_exact_bridge
        z_next_actual = z_perturbed + dt_step * v_cfl_pert
        rem_drift = float((z_next_actual - z_next_expected)[0, 0, 16, 16].item())
        _assert(
            0.0 < rem_drift < drift_err,
            f"CFL drift restorer violated monotonic contraction at d=0.05, sigma=0.08, dt=-0.25: rem_drift={rem_drift}",
        )

        dn_locked = torch.zeros((1, 1, 32, 32), device=dev, dtype=torch.float32)
        v_locked_out = couple_splat_velocity_prior(
            v_dit=v_dit_ones,
            latents=z_exact_bridge,
            splat_canvas=splat_canvas,
            splat_union_alpha=alpha_full,
            splat_denoise_map=dn_locked,
            sigma_curr=s_low,
            noise=noise_ref,
            step_dt=dt_step,
        )
        _assert(
            float((v_locked_out - v_exact_bridge).abs().max().item()) < 1e-5,
            "Strictly locked exterior region (dn=0.0) failed velocity phase-locking to v_exact_bridge",
        )

        m_probe = torch.tensor(
            [0.0, 1e-6, 0.25, 0.50, 0.75, 1.0], device=dev, dtype=torch.float32
        ).view(1, 1, 1, -1)
        fg_gate_probe, sig_rel_probe = compute_differential_inpaint_gate(
            inpaint_mask=m_probe,
            next_sigma=0.45,
            sigma_0=1.0,
            box_horizon_map=None,
        )
        g_vals = fg_gate_probe[0, 0, 0]
        _assert(
            bool(torch.isfinite(fg_gate_probe).all().item())
            and float(g_vals[0].item()) == 0.0
            and float(g_vals[1].item()) == 0.0
            and float(g_vals[2].item()) == 0.0
            and 0.1 < float(g_vals[3].item()) < 0.99
            and float(g_vals[4].item()) == 1.0
            and float(g_vals[5].item()) == 1.0
            and abs(float(sig_rel_probe[0, 0, 0, 3].item()) - 0.50) < 1e-6,
            f"Gate 5.D Differential Diffusion temporal release gate failed continuous feather horizon mapping: {g_vals.tolist()}",
        )

        loop_noise = philox_randn((1, 128, 32, 32), seed=303, stream_id=3, device=dev)
        text_target_gauss = philox_randn(
            (1, 128, 32, 32), seed=404, stream_id=4, device=dev
        )
        cut_mask_loop = torch.ones((1, 1, 24, 24), device="cpu", dtype=torch.float32)
        cut_mask_loop[:, :, 0, :] = 0.50
        cut_mask_loop[:, :, -1, :] = 0.50
        fg_gauss_24 = philox_randn(
            (1, 128, 24, 24), seed=202, stream_id=5, device="cpu"
        )
        loop_can, loop_alp, loop_dn = splat_box_latents_to_canvas(
            base_canvas=bg_gauss,
            splat_entries=[
                (fg_gauss_24, (0.125, 0.125, 0.875, 0.875), cut_mask_loop, 0.55)
            ],
            h_lat=32,
            w_lat=32,
            latent_channels=128,
            feather_radius_lat=2.0,
            device=dev,
            default_denoise_strength=0.75,
            return_denoise_map=True,
            aperture_overrides=[((0.375, 0.375, 0.625, 0.625), 0.65, 1.0)],
        )
        loop_target_fg = loop_can.clone()
        loop_target_fg[:, :, 12:20, 12:20] = text_target_gauss[:, :, 12:20, 12:20]
        loop_inpaint_mask = build_box_aperture_mask(
            layout_items=[{"bbox": [0.125, 0.125, 0.875, 0.875]}],
            height=512,
            width=512,
            coords_are_pixels=False,
            feather_radius_lat=2.0,
            device=dev,
        )
        assert loop_inpaint_mask is not None
        loop_dn_warped = warp_denoise_to_sigma_horizon(loop_dn, shift=5.0)
        assert isinstance(loop_dn_warped, torch.Tensor)

        loop_sched = BifurcatedFlowMatchScheduler(
            early_solver="heun",
            late_solver="ipndm",
            handoff_threshold=0.35,
            shift=5.0,
            use_v2_grid=True,
        )
        loop_sched.set_timesteps(12, device=dev)
        s0_loop = float(loop_sched.sigmas[0].item())
        lat_loop = (1.0 - s0_loop) * loop_can + s0_loop * loop_noise

        for p_idx, t_loop in enumerate(loop_sched.timesteps):
            cfg_l = loop_sched._step_configs[p_idx]
            s_eval = float(t_loop.item()) / float(loop_sched.num_train_timesteps)
            dt_l = float(cfg_l["dt"])
            s_next_l = float(cfg_l["s_next"])
            s_eff_l = max(s_eval, 0.12, 1.85 * abs(dt_l))
            z_bridge_fg = (1.0 - s_eval) * loop_target_fg + s_eval * loop_noise
            v_model = (loop_noise - loop_target_fg) + (
                lat_loop - z_bridge_fg
            ) / s_eff_l
            v_coupled = couple_splat_velocity_prior(
                v_dit=v_model,
                latents=lat_loop,
                splat_canvas=loop_can,
                splat_union_alpha=loop_alp,
                splat_denoise_map=loop_dn_warped,
                sigma_curr=s_eval,
                noise=loop_noise,
                step_dt=dt_l,
            )
            lat_loop = loop_sched.step(v_coupled, t_loop, lat_loop)
            noised_bg_l = (1.0 - s_next_l) * loop_can + s_next_l * loop_noise
            fg_gate_l, _ = compute_differential_inpaint_gate(
                inpaint_mask=loop_inpaint_mask,
                next_sigma=s_next_l,
                sigma_0=s0_loop,
                box_horizon_map=loop_dn_warped,
            )
            lat_loop = fg_gate_l * lat_loop + (1.0 - fg_gate_l) * noised_bg_l
            _assert(
                bool(torch.isfinite(lat_loop).all().item()),
                f"Gate 5.D Non-finite latent state encountered at pass {p_idx}",
            )

        final_cell_rms = torch.sqrt(torch.mean(lat_loop[0] * lat_loop[0], dim=0))
        ext_rms = float(final_cell_rms[:2, :2].mean().item())
        box_int_rms = float(final_cell_rms[6:10, 6:10].mean().item())
        carved_txt_rms = float(final_cell_rms[14:18, 14:18].mean().item())
        rim_mask = (loop_inpaint_mask[0, 0] > 0.15) & (loop_inpaint_mask[0, 0] < 0.85)
        rim_loop_rms = float(final_cell_rms[rim_mask].mean().item())
        carved_txt_mae = float(
            (
                lat_loop[:, :, 14:18, 14:18]
                - text_target_gauss[:, :, 14:18, 14:18]
            )
            .abs()
            .mean()
            .item()
        )
        for z_name, z_rms in (
            ("exterior", ext_rms),
            ("splat_interior", box_int_rms),
            ("carved_text", carved_txt_rms),
            ("feather_rim", rim_loop_rms),
        ):
            _assert(
                0.70 <= z_rms <= 1.35,
                f"Gate 5.D Closed-loop regional variance collapse/explosion in {z_name}: rms={z_rms:.4f}",
            )
        _assert(
            carved_txt_mae < 0.05,
            f"Gate 5.D Carved foreground typography box failed to converge to target synthesis: mae={carved_txt_mae:.4f}",
        )

    for fmt in ("moss_image_edit", "ai_toolkit_isolated", "control_color_depth"):
        req = GenerationRequest(prompt="test", edit_format=fmt)
        _assert(
            req.effective_reference_zero_timestep is True,
            f"edit_format={fmt} must enable reference_zero_timestep",
        )
        _assert(
            req.effective_reference_isolated_kv is True,
            f"edit_format={fmt} must enable reference_isolated_kv",
        )
    req_ait = GenerationRequest(prompt="test", edit_format="ai_toolkit_isolated")
    _assert(
        req_ait.effective_uncond_uses_references is False,
        "Unboxed ai_toolkit_isolated must default effective_uncond_uses_references to False",
    )
    req_ait_override = GenerationRequest(
        prompt="test", edit_format="ai_toolkit_isolated", uncond_uses_references=True
    )
    _assert(
        req_ait_override.effective_uncond_uses_references is True,
        "Explicit uncond_uses_references=True override was not respected",
    )
    print(
        f"  [+] Spatio-Visual Graph      : PorterDuffOcclusion + NonTransitiveIsoKV + Stage1FP32Upgrade + ClosedLoopRMS (rim={rim_loop_rms:.3f}, txt={carved_txt_rms:.3f}) Verified"
    )

    eng = MOSSImageEngine(device=dev, prefer_vulkan=True)
    models_status = eng.list_available_models()
    cached_variants = [m["variant"].upper() for m in models_status if m["cached"]]
    print(
        f"  [+] Cached Model Snapshots   : {', '.join(cached_variants) if cached_variants else 'None (Run download_model.py)'}"
    )
    eng.prepare_for_exit()


if __name__ == "__main__":
    gate_5_refinery_schema_and_snapshots()