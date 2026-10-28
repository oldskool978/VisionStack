#!/usr/bin/env python3
# ShadowBoxer/sanitycheck.py
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

sys.modules["torchvision"] = None
sys.modules["torchvision.transforms"] = None
sys.modules["torchvision.io"] = None

import numpy as np
from PIL import Image
from safetensors.torch import save_file as save_safetensors
import torch

from download_model import inspect_shard_header
from engine import (
    BifurcatedFlowMatchScheduler,
    CachedReferenceRecord,
    MOSSImageEngine,
    MOSSImageScheduler,
    PromptEmbedCache,
    ReferenceVisionCache,
    WDDMMemoryGuard,
    _MetaWeightInitMode,
    _probe_image_dimensions,
    build_box_aperture_mask,
    compose_latent_inpaint_mask,
    compute_moss_v2_sigmas,
    derive_stage_keys,
    load_sharded_safetensors_into,
    philox_randn,
    splat_box_latents_to_canvas,
    stream_safetensors_inplace,
    subphase_radial_clamp,
)
from harness import build_default_request
from models.modeling_moss import (
    MOSSImageConfig,
    MOSSImageTransformer,
    _attention,
    _rotate,
)
from models.modeling_qwen35 import (
    Qwen3_5Config,
    Qwen3_5Model,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
    _STRICTLY_UPPER_MASK_CACHE,
    _solve_unit_lower_triangular,
)
from models.modeling_vae import Flux2AutoencoderKL, Flux2VAEConfig
from models.tokenizer import MOSSImageTokenizer, render_qwen35_chat_prompt
from refinery import (
    DTYPE_LOOKUP,
    DTYPE_MAP,
    FP8_DTYPES,
    clear_refinery_cache,
    refineTensor,
)
from schema import (
    BASELINE_ENGINE_DEFAULTS,
    GenerationRequest,
    SUPPORTED_MASK_BLEND_MODES,
    SUPPORTED_SOLVERS,
    bbox_to_normalized,
    bbox_to_pixel_rect,
    compile_layout_prompt,
    get_active_engine_defaults,
    get_canonical_box_order,
    has_custom_default_preset,
    pixel_bbox_to_abs_centralized,
    reconcile_layout_summary_quotes,
)
from vizinjector import (
    VIZJ_TIER_DUAL_LATENT,
    VizInjectorClient,
    VizInjectorRequest,
)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def gate_1_native_subsystems() -> None:
    print(
        "\n--- Gate 1: Sovereign Native Subsystems (VizToken SWISS5 & VizInjector Vulkan) ---"
    )
    vt_dir = ROOT_DIR / "viztoken"
    vi_dir = ROOT_DIR / "vizinjector"
    is_win = sys.platform == "win32"

    vt_required = (
        ("viztoken.dll", "viztoken-cli.exe", "qwen35.ranks.bin", "viztoken.py")
        if is_win
        else ("libviztoken.so", "viztoken-cli", "qwen35.ranks.bin", "viztoken.py")
    )
    for req_file in vt_required:
        p = vt_dir / req_file
        _assert(p.is_file(), f"Missing VizToken runtime artifact: {p}")

    vi_required = (
        (
            "vizinjector.dll",
            "vizinjector-cli.exe",
            "vulkan-1.dll",
            "ffmpeg.exe",
            "ffprobe.exe",
            "bindings.py",
            "schema.py",
            "modeling_vae.py",
        )
        if is_win
        else (
            "libvizinjector.so",
            "vizinjector-cli",
            "ffmpeg",
            "ffprobe",
            "bindings.py",
            "schema.py",
            "modeling_vae.py",
        )
    )
    for req_file in vi_required:
        p = vi_dir / req_file
        _assert(p.is_file(), f"Missing VizInjector runtime artifact: {p}")

    with MOSSImageTokenizer() as tok:
        _assert(
            tok.base_vocab_size >= 240000,
            f"Unexpected vocab size: {tok.base_vocab_size}",
        )
        _assert(
            tok.has_fast_tables, "SWISS5 fast tables not active in qwen35.ranks.bin"
        )

        raw_p1 = "A brushed-steel refrigerator with sticky note 'OPENMOSS'"
        raw_p2 = "A brushed-steel refrigerator with sticky note 'AGENTIC PASS 2'"
        p1 = render_qwen35_chat_prompt(
            raw_p1,
            enable_thinking=True,
            image_patch_counts=[144],
        )
        p2 = render_qwen35_chat_prompt(
            raw_p2,
            enable_thinking=True,
            image_patch_counts=[144],
        )
        ids_1 = tok.encode(p1, allowed_special="all")
        ids_2 = tok.encode(p2, allowed_special="all")

        chat_ids_1 = tok.encode_chat(
            raw_p1,
            enable_thinking=True,
            max_sequence_length=4096,
            image_patch_counts=[144],
        )
        _assert(
            ids_1 == chat_ids_1,
            "Zero-alloc encode_chat diverged from full-string ChatML encode",
        )

        for think_flag in (True, False):
            for p_counts in (None, [2304, 576]):
                for sys_p in (None, "You are a spatial composition architect."):
                    for t_trace in (
                        None,
                        "Place Picture 1 in the left quadrant and preserve typography.",
                    ):
                        for vis_id in (True, False):
                            ref_str = render_qwen35_chat_prompt(
                                raw_p2,
                                enable_thinking=think_flag,
                                image_patch_counts=p_counts,
                                system_prompt=sys_p,
                                thinking_trace=t_trace,
                                add_vision_id=vis_id,
                            )
                            _assert(
                                tok.encode(ref_str, allowed_special="all")
                                == tok.encode_chat(
                                    raw_p2,
                                    enable_thinking=think_flag,
                                    max_sequence_length=8192,
                                    image_patch_counts=p_counts,
                                    system_prompt=sys_p,
                                    thinking_trace=t_trace,
                                    add_vision_id=vis_id,
                                ),
                                f"encode_chat parity failure for thinking={think_flag}, patches={p_counts}, sys={sys_p is not None}, trace={t_trace is not None}, vis_id={vis_id}",
                            )

        shared_pfx = MOSSImageTokenizer.find_common_prefix(ids_1, ids_2)
        _assert(
            shared_pfx > 150, f"Expected shared prefix > 150 tokens, got {shared_pfx}"
        )
        decoded_p1 = tok.decode(ids_1)
        _assert("OPENMOSS" in decoded_p1, "Tokenizer roundtrip failed")
        print(
            f"  [+] VizToken Qwen3.5 SWISS5  : {tok.ranks_path.name} (Vocab={tok.n_vocab:,}, FastTables={tok.has_fast_tables}, SharedPrefix={shared_pfx} tok, System/Trace/VisionID Parity Verified)"
        )

    with VizInjectorClient(prefer_vulkan=True) as viz:
        _assert(
            viz.vulkan_active, "VizInjector Vulkan compute backend is not active"
        )
        print(
            f"  [+] VizInjector Backend      : {viz.backend_name} (Vulkan Active: {viz.vulkan_active})"
        )

        yy, xx = np.meshgrid(
            np.linspace(-1.0, 1.0, 512, dtype=np.float32),
            np.linspace(-1.0, 1.0, 512, dtype=np.float32),
            indexing="ij",
        )
        rgba = np.zeros((512, 512, 4), dtype=np.uint8)
        rgba[..., 0] = np.clip((0.5 + 0.45 * xx) * 255.0, 0, 255).astype(np.uint8)
        rgba[..., 1] = np.clip((0.5 + 0.45 * yy) * 255.0, 0, 255).astype(np.uint8)
        rgba[..., 2] = 180
        rgba[..., 3] = 255

        req_edit = VizInjectorRequest(
            target_width=512,
            target_height=512,
            edit_format="ai_toolkit_isolated",
            framing_mode="cover_crop",
            input_role="photo_ref",
            prefer_vulkan=True,
        )
        bundle = viz.condition_rgba8(rgba, req_edit)
        _assert(
            bundle.vae_tensor is not None
            and bundle.vae_tensor.shape == (1, 3, 512, 512),
            "Invalid VAE shape",
        )
        _assert(
            bundle.vit_patches is not None and bundle.llm_tokens == 144,
            f"Expected 144 LLM tokens for 384px ViT, got {bundle.llm_tokens}",
        )

        req_vit_only = VizInjectorRequest(
            target_width=1024,
            target_height=1024,
            edit_format="ai_toolkit_isolated",
            emit_vae_branch=False,
            emit_mask_branch=False,
            emit_vit_branch=True,
            prefer_vulkan=True,
        )
        b_vit_only = viz.condition_rgba8(rgba, req_vit_only)
        _assert(
            b_vit_only.vae_tensor is None
            and b_vit_only.meta.vae_size == (1024, 1024)
            and b_vit_only.meta.vit_size == (384, 384),
            f"ViT-only pass failed coupled grid geometry: vae={b_vit_only.meta.vae_size}, vit={b_vit_only.meta.vit_size}",
        )

        bundle.packed_latents = torch.randn((1, 128, 32, 32), dtype=torch.float32)
        bundle.vit_embeds = torch.randn((144, 4096), dtype=torch.bfloat16)
        bundle.meta.orig_size = (1024, 768)
        bundle.meta.vae_size = (0, 0)
        with tempfile.TemporaryDirectory() as tmpdir:
            vizj_p = Path(tmpdir) / "tier2_test.vizj"
            blob = viz.save_vizj(
                bundle,
                dest_path=vizj_p,
                tier=VIZJ_TIER_DUAL_LATENT,
                use_bf16=True,
            )
            probed_dims = _probe_image_dimensions(vizj_p)
            _assert(
                probed_dims == (512, 512),
                f"Expected _probe_image_dimensions to return healed VAE dims (512, 512), got {probed_dims}",
            )
        reloaded = viz.load_vizj_bytes(blob, req_edit)
        _assert(
            reloaded.packed_latents is not None
            and reloaded.packed_latents.shape == (1, 128, 32, 32),
            "Reloaded Tier-2 VAE latent shape mismatch",
        )
        _assert(
            reloaded.vit_embeds is not None
            and reloaded.vit_embeds.shape == (144, 4096),
            "Reloaded Tier-2 ViT embeds shape mismatch",
        )
        _assert(
            reloaded.meta.vae_size == (512, 512),
            f"Reloaded Tier-2 healed vae_size mismatch: {reloaded.meta.vae_size}",
        )
        print(
            f"  [+] .VIZJ Tier-2 Zero-Copy   : {len(blob):,} bytes | Latents={tuple(reloaded.packed_latents.shape)} | ViT={tuple(reloaded.vit_embeds.shape)}"
        )


def gate_2_pinned_memory_and_cache_isolation() -> None:
    print(
        "\n--- Gate 2: Page-Locked Host Memory (pin_memory) & Multi-Turn Cache Isolation ---"
    )
    has_gpu = torch.cuda.is_available()
    dev = torch.device("cuda" if has_gpu else "cpu")

    p_cache = PromptEmbedCache(max_entries=8)
    pos_tokens_1 = [248045, 872, 198, 101, 102, 103, 104, 105, 248044]
    neg_tokens = [248045, 872, 198, 248044]
    pos_tokens_2 = [248045, 872, 198, 101, 102, 103, 999, 888, 248044]

    with torch.inference_mode():
        h_pos1 = torch.randn(
            (1, len(pos_tokens_1), 64), device=dev, dtype=torch.bfloat16
        )
        h_neg = torch.randn(
            (1, len(neg_tokens), 64), device=dev, dtype=torch.bfloat16
        )

        _assert(
            p_cache.lookup(pos_tokens_1, dev, is_negative=False) is None,
            "Expected initial miss",
        )
        p_cache.insert(pos_tokens_1, h_pos1)

        stored_cpu = p_cache._cache[tuple(pos_tokens_1)]
        _assert(
            not stored_cpu.is_inference(),
            "Pinned CPU cache tensor must not be an InferenceTensor",
        )
        if has_gpu:
            _assert(
                stored_cpu.is_pinned(),
                "PromptEmbedCache CPU tensor is not page-locked (pinned)",
            )

        _assert(
            p_cache.lookup(neg_tokens, dev, is_negative=True) is None,
            "Expected initial neg miss",
        )
        p_cache.insert(neg_tokens, h_neg)

        _ = p_cache.lookup(neg_tokens, dev, is_negative=True)
        _assert(
            p_cache.pos_hits == 0,
            "Negative prompt hit must not increment pos_hits",
        )
        _assert(
            p_cache._last_tokens == pos_tokens_1,
            "Negative prompt lookup clobbered _last_tokens",
        )

        _ = p_cache.lookup(pos_tokens_2, dev, is_negative=False)
        _assert(
            p_cache.last_common_prefix_len == 6,
            f"Expected common prefix of 6 tokens between pos_1 and pos_2, got {p_cache.last_common_prefix_len}",
        )

        vis_sig = (("ref_a.png", 1024, 1024), 144)
        p_cache.insert(pos_tokens_2, h_pos1, vision_signature=vis_sig)
        hit_mm = p_cache.lookup(
            pos_tokens_2, dev, is_negative=False, vision_signature=vis_sig
        )
        _assert(
            hit_mm is not None and hit_mm.device.type == dev.type,
            "Multimodal prompt cache lookup failed",
        )

    print(
        f"  [+] PromptEmbedCache         : Pinned={stored_cpu.is_pinned() if has_gpu else 'CPU'} | NonInferenceCPU=True | PrefixSplice={p_cache.last_common_prefix_len} tok"
    )

    r_cache = ReferenceVisionCache(max_entries=8)
    with tempfile.TemporaryDirectory() as tmpdir:
        img_p = Path(tmpdir) / "test_ref.png"
        Image.fromarray(np.full((512, 512, 3), 128, dtype=np.uint8), mode="RGB").save(
            img_p
        )
        key_v7 = ReferenceVisionCache.make_key(
            img_p, 512, 512, "never_crop", "photo_ref", "moss_image_edit", 2304, 4.5
        )
        vae_key_v7 = ReferenceVisionCache.make_vae_key(
            img_p, 512, 512, "never_crop", "photo_ref", 4.5
        )
        vae_key_v8 = ReferenceVisionCache.make_vae_key(
            img_p, 512, 512, "inscribed", "photo_ref", 4.5
        )
        vae_key_cc = ReferenceVisionCache.make_vae_key(
            img_p, 512, 512, "cover_crop", "photo_ref", 4.5
        )
        _assert(
            vae_key_v7 == vae_key_v8 and vae_key_v8 == vae_key_cc,
            "Aspect-matching 'never_crop', 'inscribed', and 'cover_crop' must canonicalize to the same VAE cache key",
        )

        with torch.inference_mode():
            rec_in = CachedReferenceRecord(
                packed_latents=torch.randn((1, 128, 32, 32), device=dev),
                vit_embeds=None,
                vit_patches=None,
                grid_thw=None,
                llm_tokens=0,
                latent_mask=torch.ones((1, 1, 32, 32), device=dev),
                viz_meta=None,
            )
            pinned_rec = r_cache.insert(key_v7, rec_in, vae_key=vae_key_v7)

        if has_gpu:
            _assert(
                pinned_rec.packed_latents is not None
                and pinned_rec.packed_latents.is_pinned(),
                "ReferenceVisionCache packed_latents is not page-locked",
            )
        cross_hit = r_cache.lookup_vae(vae_key_v8)
        _assert(
            cross_hit is not None and cross_hit[0] is not None,
            "Cross-format VAE cache lookup missed",
        )
        print(
            f"  [+] ReferenceVisionCache     : Pinned={pinned_rec.packed_latents.is_pinned() if has_gpu else 'CPU'} | CrossFormatVAEHit=True (never_crop == inscribed == cover_crop)"
        )


def gate_3_models_zero_alloc_and_lora() -> None:
    print(
        "\n--- Gate 3: Sovereign Architectures, Meta-Init Streaming, Caches, LoRA & FP8 ---"
    )
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32

    cfg = MOSSImageConfig(
        dim=256,
        ffn_dim=512,
        num_heads=8,
        num_layers=4,
        text_dim=256,
        freq_dim=64,
        latent_channels=128,
    )
    dit = MOSSImageTransformer(config=cfg).to(device=dev, dtype=dtype).eval()

    with _MetaWeightInitMode():
        meta_dit = MOSSImageTransformer(config=cfg)
    _assert(
        next(meta_dit.parameters()).is_meta,
        "_MetaWeightInitMode failed to place parameters on meta device",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_sf = Path(tmpdir) / "diffusion_pytorch_model.safetensors"
        save_safetensors(
            {k: v.detach().cpu() for k, v in dit.state_dict().items()},
            str(tmp_sf),
        )
        load_sharded_safetensors_into(Path(tmpdir), meta_dit, device=dev, dtype=dtype)
        meta_dit.eval()
        _assert(
            not next(meta_dit.parameters()).is_meta,
            "load_sharded_safetensors_into left parameters on meta device",
        )

        ptrs_before = {k: p.data_ptr() for k, p in meta_dit.named_parameters()}
        with torch.no_grad():
            meta_dit.layers[0].proj["q"].weight.zero_()
        inplace_ok = stream_safetensors_inplace(
            Path(tmpdir), meta_dit, device=dev, dtype=dtype
        )
        ptrs_after = {k: p.data_ptr() for k, p in meta_dit.named_parameters()}
        _assert(
            inplace_ok and ptrs_before == ptrs_after,
            "stream_safetensors_inplace reallocated parameter storage instead of streaming in-place",
        )

    with torch.inference_mode():
        lat = torch.randn((1, 128, 8, 8), device=dev, dtype=torch.float32)
        ref1 = torch.randn((1, 128, 8, 8), device=dev, dtype=torch.float32)
        ref2 = torch.randn((1, 128, 6, 6), device=dev, dtype=torch.float32)
        txt = torch.randn((1, 16, 256), device=dev, dtype=dtype)
        mask = torch.ones((1, 16), device=dev, dtype=torch.bool)
        _assert(
            txt.is_inference() and mask.is_inference(),
            "Test tensors must be InferenceTensors",
        )

        rope_3d = dit.rope(8, 8, 16, dev, dtype=dtype, ref_shapes=[(8, 8), (6, 6)])
        (rope_4d_full, _) = dit.rope.get_expanded_rope(
            8, 8, 16, dev, dtype=dtype, ref_shapes=[(8, 8), (6, 6)]
        )
        dummy_q = torch.randn((1, rope_3d.shape[1], 8, 32), device=dev, dtype=dtype)
        rot_3d = _rotate(dummy_q, rope_3d)
        rot_4d = _rotate(dummy_q, rope_4d_full)
        _assert(
            float((rot_3d - rot_4d).abs().max().item()) == 0.0,
            "Pre-expanded 4D RoPE diverged from 3D RoPE",
        )

        t_tok = 64
        r1_tok = 64
        r2_tok = 36
        iso_mask = dit._get_isolated_reference_mask(
            t_tok, [r1_tok, r2_tok], 16, dev
        )
        _assert(
            bool(iso_mask[0, 0, 0, t_tok + 10].item()) is True,
            "Target must attend to Ref 1",
        )
        _assert(
            bool(iso_mask[0, 0, t_tok + 5, 0].item()) is False,
            "Ref 1 must not attend to Target",
        )
        _assert(
            bool(iso_mask[0, 0, t_tok + 5, t_tok + r1_tok + 5].item()) is False,
            "Disjoint Ref 1 must not attend to Ref 2",
        )
        _assert(
            bool(iso_mask[0, 0, t_tok + 5, t_tok + r1_tok + r2_tok + 2].item())
            is True,
            "Ref 1 must attend to Text",
        )

        q_test = torch.randn(
            (1, rope_3d.shape[1], 8, 32), device=dev, dtype=torch.float32
        )
        k_test = torch.randn(
            (1, rope_3d.shape[1], 8, 32), device=dev, dtype=torch.float32
        )
        v_test = torch.randn(
            (1, rope_3d.shape[1], 8, 32), device=dev, dtype=torch.float32
        )
        (rope_4d_fp32, _) = dit.rope.get_expanded_rope(
            8, 8, 16, dev, dtype=torch.float32, ref_shapes=[(8, 8), (6, 6)]
        )
        attn_masked = _attention(
            q_test, k_test, v_test, rope_4d_fp32, 8, attn_mask=iso_mask
        )
        attn_partitioned = _attention(
            q_test,
            k_test,
            v_test,
            rope_4d_fp32,
            8,
            attn_mask=None,
            iso_kv_spec=(t_tok, (r1_tok, r2_tok), 16, ((0,), (1,))),
        )
        iso_diff = float((attn_masked - attn_partitioned).abs().max().item())
        _assert(
            iso_diff < 1e-5,
            f"Block-partitioned FlashAttention IsoKV diverged from boolean mask: {iso_diff}",
        )

        overlap_boxes = [(0.125, 0.125, 0.75, 0.75), (0.25, 0.25, 0.875, 0.875)]
        collab_mask = dit._get_isolated_reference_mask(
            t_tok, [r1_tok, r2_tok], 16, dev, ref_bboxes=overlap_boxes
        )
        _assert(
            bool(collab_mask[0, 0, t_tok + 5, t_tok + r1_tok + 5].item()) is True,
            "Overlapping reference boxes must collaborate in IsoKV attention",
        )
        attn_collab_masked = _attention(
            q_test, k_test, v_test, rope_4d_fp32, 8, attn_mask=collab_mask
        )
        attn_collab_part = _attention(
            q_test,
            k_test,
            v_test,
            rope_4d_fp32,
            8,
            attn_mask=None,
            iso_kv_spec=(t_tok, (r1_tok, r2_tok), 16, ((0, 1), (0, 1))),
        )
        collab_diff = float(
            (attn_collab_masked - attn_collab_part).abs().max().item()
        )
        _assert(
            collab_diff < 1e-5,
            f"Collaborative block-partitioned IsoKV diverged from mask: {collab_diff}",
        )

        bias_none = dit._get_box_aperture_bias(
            height=8,
            width=8,
            ref_token_lengths=[r1_tok, r2_tok],
            text_tokens=16,
            ref_shapes=[(8, 8), (6, 6)],
            ref_bboxes=[(0.5, 0.5, 1.0, 1.0), None],
            reference_box_isolation=0.0,
            feather_radius_lat=1.0,
            device=dev,
            dtype=torch.float32,
        )
        _assert(
            bias_none is None,
            "Zero box_attn_isolation must return None without allocating a bias tensor",
        )

        bias_active = dit._get_box_aperture_bias(
            height=8,
            width=8,
            ref_token_lengths=[r1_tok, r2_tok],
            text_tokens=16,
            ref_shapes=[(8, 8), (6, 6)],
            ref_bboxes=[(0.5, 0.5, 1.0, 1.0), None],
            reference_box_isolation=[0.85, 0.0],
            feather_radius_lat=1.5,
            device=dev,
            dtype=torch.float32,
        )
        _assert(
            bias_active is not None and bias_active.shape == (1, 1, 64, 180),
            f"Unexpected box aperture bias shape: {None if bias_active is None else bias_active.shape}",
        )
        interior_idx = 6 * 8 + 6
        exterior_idx = 0 * 8 + 0
        feather_idx = 3 * 8 + 6
        _assert(
            float(bias_active[0, 0, interior_idx, t_tok + 5].item()) == 0.0,
            "Interior target token inside Ref 1 bbox must have 0.0 attention bias",
        )
        _assert(
            abs(float(bias_active[0, 0, exterior_idx, t_tok + 5].item()) - (-17.0))
            < 1e-4,
            f"Exterior target token outside Ref 1 bbox expected -17.0 bias, got {bias_active[0, 0, exterior_idx, t_tok + 5].item()}",
        )
        _assert(
            -17.0 < float(bias_active[0, 0, feather_idx, t_tok + 5].item()) < 0.0,
            "Feather boundary token must have smooth intermediate attention bias",
        )
        _assert(
            float(bias_active[0, 0, :, t_tok + r1_tok :].abs().max().item()) == 0.0,
            "Unisolated Ref 2 and Text tokens must remain completely unattenuated (0.0)",
        )

        phase_boxed = dit.rope._compute_phase(
            8,
            8,
            16,
            dev,
            ref_shapes=[(4, 4)],
            ref_bboxes=[(0.5, 0.5, 1.0, 1.0)],
        )
        ref_phase_slice = phase_boxed[64 : 64 + 16, 2:4, 0]
        _assert(
            float(ref_phase_slice.min().item()) >= 0.0,
            "Cell-centered bottom-right box-translated 4D-RoPE phase must reside strictly in positive quadrant",
        )

        t_step_1 = torch.tensor([500.0], device=dev)
        ref_boxes = [(0.125, 0.125, 0.875, 0.875), (0.25, 0.25, 1.0, 1.0)]
        out1 = dit(
            lat,
            t_step_1,
            txt,
            mask,
            reference_latents=[ref1, ref2],
            reference_zero_timestep=True,
            reference_isolated_kv=True,
            reference_bboxes=ref_boxes,
        ).sample
        out_meta = meta_dit(
            lat,
            t_step_1,
            txt,
            mask,
            reference_latents=[ref1, ref2],
            reference_zero_timestep=True,
            reference_isolated_kv=True,
            reference_bboxes=ref_boxes,
        ).sample
        _assert(
            float((out1 - out_meta).abs().max().item()) == 0.0,
            "Meta-initialized and in-place streamed DiT output diverged from standard DiT",
        )
        _assert(
            out1.shape == (1, 128, 8, 8), f"Unexpected DiT output shape: {out1.shape}"
        )

        out_box_iso = dit(
            lat,
            t_step_1,
            txt,
            mask,
            reference_latents=[ref1, ref2],
            reference_zero_timestep=True,
            reference_isolated_kv=True,
            reference_bboxes=ref_boxes,
            reference_box_isolation=[0.90, 0.50],
            box_feather_radius=1.25,
        ).sample
        out_box_iso_non_isokv = dit(
            lat,
            t_step_1,
            txt,
            mask,
            reference_latents=[ref1, ref2],
            reference_zero_timestep=True,
            reference_isolated_kv=False,
            reference_bboxes=ref_boxes,
            reference_box_isolation=0.85,
            box_feather_radius=1.25,
        ).sample
        _assert(
            torch.isfinite(out_box_iso).all()
            and torch.isfinite(out_box_iso_non_isokv).all()
            and float((out1 - out_box_iso).abs().max().item()) > 1e-5,
            "Regional box-aperture isolation forward pass failed or produced no spatial modulation",
        )
        _assert(
            len(dit._box_aperture_bias_cache) >= 2,
            "Box aperture bias cache was not populated",
        )

        _assert(dit._mod_seq_buf is not None, "_mod_seq_buf was not allocated")
        buf_ptr_1 = dit._mod_seq_buf.data_ptr()
        _assert(
            len(dit._mod_zero_cache) == 1, "Zero-timestep modulation was not cached"
        )
        _assert(
            len(dit._text_proj_cache) == 1,
            "Step-invariant _TextProjection was not cached",
        )
        _assert(
            len(dit._ref_proj_cache) == 2,
            "Reference latent projections were not cached",
        )
        _assert(len(dit._time_mod_cache) == 2, "Timestep modulation was not cached")
        cached_txt_ptr = next(iter(dit._text_proj_cache.values()))[2].data_ptr()

        _ = dit(
            lat,
            t_step_1,
            txt,
            mask,
            skip_layers={2},
            reference_latents=[ref1, ref2],
            reference_zero_timestep=True,
            reference_isolated_kv=True,
            reference_bboxes=ref_boxes,
        ).sample
        _assert(
            len(dit._time_mod_cache) == 2,
            "Same-timestep pass must hit _time_mod_cache",
        )

        t_step_2 = torch.tensor([250.0], device=dev)
        out2 = dit(
            lat,
            t_step_2,
            txt,
            mask,
            reference_latents=[ref1, ref2],
            reference_zero_timestep=True,
            reference_isolated_kv=True,
            reference_bboxes=ref_boxes,
        ).sample
        buf_ptr_2 = dit._mod_seq_buf.data_ptr()
        cached_txt_ptr_2 = next(iter(dit._text_proj_cache.values()))[2].data_ptr()
        _assert(
            buf_ptr_1 == buf_ptr_2,
            "_mod_seq_buf was reallocated instead of reused in-place",
        )
        _assert(
            cached_txt_ptr == cached_txt_ptr_2,
            "_text_proj_cache entry was not reused across steps",
        )
        _assert(
            len(dit._time_mod_cache) == 3,
            "Expected 3 cached time modulations (500, 0, 250)",
        )
        dit.clear_transient_caches()
        _assert(
            dit._mod_seq_buf is None,
            "clear_transient_caches must release _mod_seq_buf between turns",
        )
        dit.clear_caches()
        _assert(
            len(dit._box_aperture_bias_cache) == 0,
            "clear_caches must flush _box_aperture_bias_cache",
        )
    print(
        f"  [+] MOSSImageTransformer     : Meta-Init + InPlaceHotSwap (0-Alloc) + CellCenter-4D-RoPE + CollaborativeIsoKV (diff={collab_diff:.1e}) + ModSeqBuf (0x{buf_ptr_2:X}) Verified"
    )

    orig_q_weight = dit.layers[0].proj["q"].weight.data.detach().float().clone()
    lora_sd = {
        "transformer.layers.0.proj.q.lora_A.weight": torch.randn(
            (4, 256), dtype=torch.float32
        )
        * 0.1,
        "transformer.layers.0.proj.q.lora_B.weight": torch.randn(
            (256, 4), dtype=torch.float32
        )
        * 0.1,
    }
    fused_cnt = dit.fuse_lora_state_dict(lora_sd, scale=1.0, lora_tag="sanity_lora")
    _assert(fused_cnt == 1, f"Expected 1 fused LoRA layer, got {fused_cnt}")
    _assert(
        dit.is_lora_active("sanity_lora", 1.0),
        "is_lora_active failed to report active LoRA",
    )
    _assert(
        len(dit._text_proj_cache) == 0 and len(dit._ref_proj_cache) == 0,
        "LoRA fuse must invalidate step caches",
    )
    backup_tensor = dit._lora_backups["layers.0.proj.q.weight"]
    if dev.type == "cuda":
        _assert(backup_tensor.is_pinned(), "LoRA weight backup is not page-locked")
    dit.unfuse_lora()
    restored_diff = float(
        (dit.layers[0].proj["q"].weight.data.float() - orig_q_weight)
        .abs()
        .max()
        .item()
    )
    _assert(restored_diff == 0.0, f"LoRA unfuse drift: {restored_diff}")
    print(
        f"  [+] Pin-Backed LoRA Engine   : Fused={fused_cnt} | BackupPinned={backup_tensor.is_pinned() if dev.type == 'cuda' else 'CPU'} | UnfuseMaxDiff={restored_diff:.1e}"
    )

    q_cfg = Qwen3_5Config(
        text_config=Qwen3_5TextConfig(
            vocab_size=1024,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
                "mrope_section": [2, 3, 3],
            },
        ),
        vision_config=Qwen3_5VisionConfig(
            depth=2,
            hidden_size=64,
            intermediate_size=128,
            num_heads=4,
            in_channels=3,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=128,
            num_position_embeddings=64,
        ),
        image_token_id=900,
    )
    qwen = Qwen3_5Model(q_cfg).to(device=dev, dtype=dtype).eval()
    with torch.inference_mode():
        in_ids = torch.tensor(
            [[10, 20, 900, 900, 900, 900, 30, 40]], dtype=torch.long, device=dev
        )
        pix = torch.randn((16, 3, 2, 16, 16), device=dev, dtype=dtype)
        g_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=dev)

        ut = (
            torch.randn((1, 2, 1, 64, 64), device=dev, dtype=torch.float32).tril(-1)
            * 0.05
        )
        ut = ut + torch.eye(64, device=dev, dtype=torch.float32)
        rhs_a = torch.randn((1, 2, 1, 64, 32), device=dev, dtype=torch.float32)
        rhs_b = torch.randn((1, 2, 1, 64, 32), device=dev, dtype=torch.float32)
        sol_a = _solve_unit_lower_triangular(ut, rhs_a)
        sol_b = _solve_unit_lower_triangular(ut, rhs_b)
        sol_cat_a, sol_cat_b = torch.split(
            _solve_unit_lower_triangular(ut, torch.cat([rhs_a, rhs_b], dim=-1)),
            [32, 32],
            dim=-1,
        )
        tri_diff = max(
            float((sol_a - sol_cat_a).abs().max().item()),
            float((sol_b - sol_cat_b).abs().max().item()),
        )
        res_err = float(((ut @ sol_a) - rhs_a).abs().max().item())
        _assert(
            tri_diff < 1e-5 and res_err < 1e-4,
            f"Block-recursive triangular solver drift: cat_diff={tri_diff}, res_err={res_err}",
        )

        q_out = qwen(
            input_ids=in_ids,
            attention_mask=torch.ones_like(in_ids),
            pixel_values=pix,
            image_grid_thw=g_thw,
        )
        _assert(
            q_out.rope_deltas is not None,
            "Multimodal Qwen3.5 pass must populate rope_deltas",
        )
        q_txt_only = qwen(
            input_ids=in_ids[:, :4],
            attention_mask=torch.ones_like(in_ids[:, :4]),
        )
        _assert(
            q_txt_only.rope_deltas is None,
            "Text-only Qwen3.5 pass must reset rope_deltas to None",
        )
    _assert(
        q_out.last_hidden_state.shape == (1, 8, 128),
        f"Unexpected Qwen3.5 output shape: {q_out.last_hidden_state.shape}",
    )
    _assert(
        len(qwen.visual._geom_cache) == 1,
        "Vision geometry cache was not populated",
    )
    qwen.clear_caches()
    _assert(
        len(qwen.visual._geom_cache) == 0 and len(_STRICTLY_UPPER_MASK_CACHE) == 0,
        "Qwen3_5Model.clear_caches() must flush _geom_cache and _STRICTLY_UPPER_MASK_CACHE",
    )
    print(
        f"  [+] Qwen3_5Model Hybrid      : BlockRecTriSolve (diff={tri_diff:.1e}, res={res_err:.1e}) + CachedViTGeom + MaskCacheEvict Verified"
    )

    e4m3_dtype = getattr(torch, "float8_e4m3fn", None)
    if e4m3_dtype is not None:
        with torch.inference_mode():
            dit.layers[0].proj["q"].weight.data = (
                dit.layers[0].proj["q"].weight.data.to(e4m3_dtype)
            )
            qwen.language_model.layers[0].mlp.gate_proj.weight.data = (
                qwen.language_model.layers[0].mlp.gate_proj.weight.data.to(e4m3_dtype)
            )
            dit.clear_caches()
            fp8_dit_out = dit(lat, torch.tensor([500.0], device=dev), txt, mask).sample
            fp8_qwen_out = qwen(
                input_ids=in_ids[:, :4],
                attention_mask=torch.ones_like(in_ids[:, :4]),
            ).last_hidden_state
            _assert(
                torch.isfinite(fp8_dit_out).all()
                and torch.isfinite(fp8_qwen_out).all(),
                "NaN/Inf in FP8 weight-only dispatch",
            )
        print(
            "  [+] FP8 Weight-Only Dispatch : Verified float8_e4m3fn Linear Execution in MOSS-DiT & Qwen3.5"
        )

    v_cfg = Flux2VAEConfig(
        block_out_channels=(32, 64),
        decoder_block_out_channels=(32, 64),
        latent_channels=32,
        layers_per_block=1,
        norm_num_groups=16,
    )
    vae = Flux2AutoencoderKL(config=v_cfg).to(device=dev, dtype=torch.float32).eval()
    with torch.inference_mode():
        img_in = (
            torch.randn((1, 3, 64, 64), device=dev, dtype=torch.float32).clamp(
                -1.0, 1.0
            )
        )
        z_pack = vae.encode_to_packed_latents(img_in)
        z_clamped = subphase_radial_clamp(z_pack * 10.0, max_sigma=4.5)
        dec_out = vae.decode_packed_latents(z_clamped)
    _assert(
        z_pack.shape == (1, 128, 16, 16),
        f"Unexpected packed latent shape: {z_pack.shape}",
    )
    _assert(
        dec_out.shape == (1, 3, 64, 64),
        f"Unexpected decoded image shape: {dec_out.shape}",
    )
    print(
        "  [+] Flux2AutoencoderKL       : In-Place FP32 Silu/Shortcut Pack/Unpack & Sub-Phase Radial Clamp Verified"
    )


def gate_4_schedulers_and_prng() -> None:
    print(
        "\n--- Gate 4: Flow-Matching V2 Schedule, Lagrange TVD Solvers & Philox PRNG ---"
    )
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    v2_sigmas = compute_moss_v2_sigmas(50, shift=5.0)
    _assert(len(v2_sigmas) == 50, f"Expected 50 V2 sigmas, got {len(v2_sigmas)}")
    _assert(
        bool(torch.all(v2_sigmas[:-1] >= v2_sigmas[1:]).item()),
        "V2 sigmas are not monotonically decreasing",
    )

    def _upstream_ref_sigmas(
        steps: int = 50,
        shift: float = 5.0,
        training_shift: float = 1.0,
        total: int = 1000,
    ) -> torch.Tensor:
        def grid(count: int, s: float) -> torch.Tensor:
            lower = s / (total + s - 1)
            values = torch.linspace(1, lower, count)
            return s * values / (1 + (s - 1) * values)

        sigmas = grid(steps, shift)
        training = grid(total, training_shift).flip(0)
        upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
        lower = (upper - 1).clamp(min=0)
        nearest = torch.where(
            (sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(),
            lower,
            upper,
        )
        return torch.unique_consecutive(training[nearest])

    ref_sigmas = _upstream_ref_sigmas(50, 5.0)
    _assert(
        float((v2_sigmas - ref_sigmas).abs().max().item()) == 0.0,
        "V2 sigma schedule drifted from upstream FP32 reference",
    )

    for ord_m in (1, 2, 3, 4):
        for h_tuple in (
            (-0.08, -0.08, -0.08, -0.08),
            (-0.12, -0.09, -0.07, -0.05),
            (-0.25, -0.15, -0.10, -0.07),
        ):
            w_ab = BifurcatedFlowMatchScheduler._compute_ab_weights(
                h_tuple[0], h_tuple[1], h_tuple[2], h_tuple[3], ord_m
            )
            unity_err = abs(sum(w_ab) - h_tuple[0])
            l1_ratio = sum(abs(x) for x in w_ab) / abs(h_tuple[0])
            _assert(
                unity_err < 1e-12,
                f"Lagrange AB weights violated partition of unity (order={ord_m}, err={unity_err})",
            )
            _assert(
                l1_ratio <= 2.15 + 1e-9,
                f"Lagrange AB weights exceeded TVD L1 bound (order={ord_m}, ratio={l1_ratio})",
            )

    std_sched = MOSSImageScheduler(shift=5.0)
    std_sched.set_timesteps(20, device=dev)
    x = torch.randn((1, 128, 8, 8), device=dev)
    for t in std_sched.timesteps:
        x = std_sched.step(-x, t, x)
    _assert(not torch.isnan(x).any(), "NaN in MOSSImageScheduler integration")

    euler_ipndm = BifurcatedFlowMatchScheduler(
        early_solver="euler",
        late_solver="ipndm",
        handoff_threshold=0.5,
        shift=1.0,
        use_v2_grid=False,
    )
    euler_ipndm.set_timesteps(10, device=dev)
    _assert(
        euler_ipndm._early_steps == 5 and euler_ipndm._late_steps == 5,
        f"Expected 5 early and 5 late steps at handoff=0.5 (shift=1.0), got {euler_ipndm._early_steps}/{euler_ipndm._late_steps}",
    )
    s_test = torch.randn((1, 128, 8, 8), device=dev)
    for idx, t in enumerate(euler_ipndm.timesteps):
        was_early = euler_ipndm.current_is_early
        s_test = euler_ipndm.step(-0.5 * s_test, t, s_test)
        if was_early:
            _assert(
                len(euler_ipndm._history) == 0,
                "Euler step must not push to multistep _history",
            )
        elif idx < len(euler_ipndm.timesteps) - 1:
            _assert(
                len(euler_ipndm._history) > 0,
                "IPNDM step must populate multistep _history",
            )

    heun_ipndm_12 = BifurcatedFlowMatchScheduler(
        early_solver="heun",
        late_solver="ipndm",
        handoff_threshold=0.35,
        shift=5.0,
        use_v2_grid=True,
    )
    heun_ipndm_12.set_timesteps(12, device=dev)
    k_star = heun_ipndm_12._k_star
    _assert(
        0 < k_star < 12,
        f"Expected interior handoff index k_star in (0, 12), got {k_star}",
    )
    _assert(
        (1.0 - float(heun_ipndm_12.base_sigmas[k_star - 1].item())) < 0.35
        and (1.0 - float(heun_ipndm_12.base_sigmas[k_star].item())) >= 0.35 - 1e-7,
        "Physical signal progress handoff (1 - sigma >= t*) violated",
    )

    for solver in SUPPORTED_SOLVERS:
        bif = BifurcatedFlowMatchScheduler(
            early_solver=solver,
            late_solver="ipndm" if solver != "ipndm" else "euler",
            handoff_threshold=0.35,
            shift=5.0,
            eta=0.15 if solver in ("sde_gpu_pp", "multitree") else 0.0,
            s_noise=1.0,
            use_v2_grid=True,
        )
        bif.set_timesteps(16, device=dev)
        sample = torch.randn((1, 128, 8, 8), device=dev)
        gen = torch.Generator(device=dev).manual_seed(12345)
        for t in bif.timesteps:
            sample = bif.step(-0.5 * sample, t, sample, generator=gen)
        _assert(not torch.isnan(sample).any(), f"NaN encountered in solver {solver}")

    k_init, k_sde, k_aux = derive_stage_keys(42)
    _assert(len({k_init, k_sde, k_aux}) == 3, "Stage keys are not orthogonal")
    n1 = philox_randn((1, 128, 8, 8), seed=k_init, stream_id=1, device=dev)
    n2 = philox_randn((1, 128, 8, 8), seed=k_init, stream_id=1, device=dev)
    n3 = philox_randn((1, 128, 8, 8), seed=k_init, stream_id=2, device=dev)
    _assert(
        float((n1 - n2).abs().max().item()) == 0.0,
        "Philox PRNG not deterministic for identical seed/stream",
    )
    _assert(
        float((n1 - n3).abs().max().item()) > 0.1,
        "Philox PRNG stream collision",
    )
    print(
        f"  [+] Bifurcated Solvers & PRNG: Verified {', '.join(s.upper() for s in SUPPORTED_SOLVERS)} + Lagrange TVD Unity + Philox64"
    )


def gate_5_refinery_schema_and_snapshots() -> None:
    print(
        "\n--- Gate 5: Gorilla DTO-CA Refinery, WDDM Guard, Preset Point-of-Truth & Spatio-Visual Graph ---"
    )
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        and BASELINE_ENGINE_DEFAULTS["invert_inpaint_mask"] is False,
        f"Expected V03 Heun->IPNDM baseline defaults, got {BASELINE_ENGINE_DEFAULTS}",
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
                and harness_def_req.uncond_skip_layers == [24],
                f"build_default_request() did not match V03 baseline: {harness_def_req}",
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
        height=1024,
        width=1024,
        layout_coords_are_pixels=True,
        reference_image_paths=["global_scene.png"],
        layout_items=[box_subject_item, box_text_item],
    )
    req_poly_rev = GenerationRequest(
        prompt="A cinematic composition with a portrait and tattoo overlay",
        prompt_mode="layout_json",
        height=1024,
        width=1024,
        layout_coords_are_pixels=True,
        reference_image_paths=["global_scene.png"],
        layout_items=[
            dict(box_text_item, z_order=1),
            dict(box_subject_item, z_order=9),
        ],
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
                "category": "subject",
                "caption": "Foreground character reference",
                "bbox": [256, 512, 768, 1024],
                "image_path": "box_hero.png",
                "input_role": "photo_ref",
            },
        ],
    )
    mixed_specs, mixed_pic_map = req_mixed_roles.resolve_visual_references()
    _assert(
        len(mixed_specs) == 3 and mixed_pic_map == {1: 1},
        f"Expected Picture 1 strictly bound to box 1 (photo_ref), got map={mixed_pic_map}",
    )
    _assert(
        mixed_specs[0]["emits_vit"] is False
        and mixed_specs[0]["picture_index"] is None
        and mixed_specs[0]["dit_ref_index"] == 1
        and mixed_specs[0]["framing_mode"] == "never_crop",
        f"Global moss_image_edit ref spec mismatch: {mixed_specs[0]}",
    )
    _assert(
        mixed_specs[1]["emits_vit"] is False
        and mixed_specs[1]["picture_index"] is None
        and mixed_specs[1]["dit_ref_index"] == 2
        and mixed_specs[1]["framing_mode"] == "inscribed",
        f"Box 0 depth_map ref spec mismatch: {mixed_specs[1]}",
    )
    _assert(
        mixed_specs[2]["emits_vit"] is True
        and mixed_specs[2]["picture_index"] == 1
        and mixed_specs[2]["dit_ref_index"] == 3
        and mixed_specs[2]["framing_mode"] == "inscribed",
        f"Box 1 photo_ref spec failed box-conformal framing or Picture 1 parity: {mixed_specs[2]}",
    )
    compiled_mixed = req_mixed_roles.compile_effective_prompt()
    _assert(
        '"reference": "Picture 1"' in compiled_mixed
        and '"reference": "Picture 2"' not in compiled_mixed
        and '"reference": "Picture 3"' not in compiled_mixed,
        f"Mixed-role layout prompt desynchronized Picture index: {compiled_mixed}",
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
        base_canvas = torch.zeros((1, 128, 32, 32), device=dev, dtype=torch.float32)
        box_lat_a = torch.full((1, 128, 16, 16), 2.5, device="cpu", dtype=torch.float32)
        box_lat_b = torch.full((1, 128, 16, 16), -1.5, device="cpu", dtype=torch.float32)
        splat_canvas, splat_alpha = splat_box_latents_to_canvas(
            base_canvas=base_canvas,
            splat_entries=[(box_lat_a, (0.25, 0.25, 0.75, 0.75), None, 0.75)],
            h_lat=32,
            w_lat=32,
            latent_channels=128,
            feather_radius_lat=2.0,
            device=dev,
        )
        _assert(
            splat_canvas.shape == (1, 128, 32, 32)
            and splat_alpha.shape == (1, 1, 32, 32),
            "splat_box_latents_to_canvas output shape mismatch",
        )
        _assert(
            abs(float(splat_canvas[0, 0, 16, 16].item()) - 2.5) < 1e-5
            and float(splat_canvas[0, 0, 0, 0].item()) == 0.0
            and 0.0 < float(splat_canvas[0, 0, 8, 16].item()) < 2.5,
            "splat_box_latents_to_canvas failed interior/feather/exterior composite values",
        )

        entry_1 = (box_lat_a, (0.125, 0.125, 0.625, 0.625), None, 0.75)
        entry_2 = (box_lat_b, (0.375, 0.375, 0.875, 0.875), None, 0.75)
        canvas_12, alpha_12 = splat_box_latents_to_canvas(
            base_canvas=base_canvas,
            splat_entries=[entry_1, entry_2],
            h_lat=32,
            w_lat=32,
            latent_channels=128,
            feather_radius_lat=2.0,
            device=dev,
        )
        canvas_21, alpha_21 = splat_box_latents_to_canvas(
            base_canvas=base_canvas,
            splat_entries=[entry_2, entry_1],
            h_lat=32,
            w_lat=32,
            latent_channels=128,
            feather_radius_lat=2.0,
            device=dev,
        )
        _assert(
            float((canvas_12 - canvas_21).abs().max().item()) < 1e-6
            and float((alpha_12 - alpha_21).abs().max().item()) < 1e-6,
            "splat_box_latents_to_canvas violated commutative permutation invariance on overlapping boxes",
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
        "  [+] Spatio-Visual Graph      : V03 (Heun->IPNDM) + CanonicalSort + ComposableMaskAlgebra + CommutativeSplat Verified"
    )

    eng = MOSSImageEngine(device=dev, prefer_vulkan=True)
    models_status = eng.list_available_models()
    cached_variants = [m["variant"].upper() for m in models_status if m["cached"]]
    print(
        f"  [+] Cached Model Snapshots   : {', '.join(cached_variants) if cached_variants else 'None (Run download_model.py)'}"
    )
    eng.prepare_for_exit()


def main() -> None:
    t0 = time.perf_counter()
    print("=" * 92)
    print(
        "          SHADOWBOXER SOVEREIGN ENGINE CONVERGENCE & 5-GATE AUDIT SUITE"
    )
    print("=" * 92)

    gate_1_native_subsystems()
    gate_2_pinned_memory_and_cache_isolation()
    gate_3_models_zero_alloc_and_lora()
    gate_4_schedulers_and_prng()
    gate_5_refinery_schema_and_snapshots()

    elapsed = time.perf_counter() - t0
    print("\n" + "-" * 92)
    print(
        f"[+] STATUS: SECURE, ROBUST & CONVERGED IN {elapsed:.2f}s. ZERO DRIFT ACROSS ALL 5 GATES."
    )
    print("=" * 92)


if __name__ == "__main__":
    main()