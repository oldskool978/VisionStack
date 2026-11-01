#!/usr/bin/env python3
from __future__ import annotations

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
import torch

from engine import _probe_image_dimensions, pack_vision_patches_32px
from models.modeling_qwen35 import Qwen3_5HybridCache
from models.tokenizer import MOSSImageTokenizer, render_qwen35_chat_prompt
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
        _assert(
            len(tok.trace_close_token_ids) >= 2,
            f"Expected valid trace_close_token_ids, got {tok.trace_close_token_ids}",
        )
        raw_p1 = "A brushed-steel refrigerator with sticky note 'OPENMOSS'"
        raw_p2 = "A brushed-steel refrigerator with sticky note 'AGENTIC PASS 2'"
        p1 = render_qwen35_chat_prompt(
            raw_p1,
            enable_thinking=True,
            image_patch_counts=[576],
        )
        p2 = render_qwen35_chat_prompt(
            raw_p2,
            enable_thinking=True,
            image_patch_counts=[576],
        )
        ids_1 = tok.encode(p1, allowed_special="all")
        ids_2 = tok.encode(p2, allowed_special="all")
        chat_ids_1 = tok.encode_chat(
            raw_p1,
            enable_thinking=True,
            max_sequence_length=16384,
            image_patch_counts=[576],
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
                                    max_sequence_length=16384,
                                    image_patch_counts=p_counts,
                                    system_prompt=sys_p,
                                    thinking_trace=t_trace,
                                    add_vision_id=vis_id,
                                ),
                                f"encode_chat parity failure for thinking={think_flag}, patches={p_counts}, sys={sys_p is not None}, trace={t_trace is not None}, vis_id={vis_id}",
                            )
        trunc_chat_ids = tok.encode_chat(
            raw_p2 * 8,
            enable_thinking=True,
            max_sequence_length=16,
            image_patch_counts=[256],
            system_prompt="System invariant header.",
        )
        _assert(
            trunc_chat_ids.count(248056) == 256
            and trunc_chat_ids[-len(tok._enc._chat_tail_think_on) :]
            == tok._enc._chat_tail_think_on,
            "Structure-preserving encode_chat failed to protect <|image_pad|> tokens or assistant <think> suffix under truncation",
        )

        mm_grid_thw = [(1, 48, 48), (1, 24, 24)]
        mm_merged_counts = [(t * h * w) // 4 for t, h, w in mm_grid_thw]
        mm_chat_ids = tok.encode_chat(
            raw_p1,
            enable_thinking=True,
            max_sequence_length=16384,
            image_patch_counts=mm_merged_counts,
            system_prompt="Multimodal boundary audit.",
        )
        _assert(
            mm_chat_ids.count(tok.image_pad_id) == sum(mm_merged_counts),
            f"Expected {sum(mm_merged_counts)} <|image_pad|> tokens, got {mm_chat_ids.count(tok.image_pad_id)}",
        )
        last_vis_end_idx = max(
            idx for idx, tid in enumerate(mm_chat_ids) if tid == tok.vision_end_id
        )
        min_snap_mm, struct_cps_mm = Qwen3_5HybridCache.derive_structural_checkpoints(
            mm_chat_ids,
            vision_end_token_id=tok.vision_end_id,
            im_end_token_id=tok.im_end_id,
        )
        _assert(
            min_snap_mm == last_vis_end_idx + 1 and min_snap_mm in struct_cps_mm,
            f"Gate 1.B failed: min_snap={min_snap_mm} did not anchor post-<|vision_end|> index {last_vis_end_idx + 1}",
        )

        shared_pfx = MOSSImageTokenizer.find_common_prefix(ids_1, ids_2)
        _assert(
            shared_pfx > 580, f"Expected shared prefix > 580 tokens, got {shared_pfx}"
        )
        decoded_p1 = tok.decode(ids_1)
        _assert("OPENMOSS" in decoded_p1, "Tokenizer roundtrip failed")
        print(
            f"  [+] VizToken Qwen3.5 SWISS5  : {tok.ranks_path.name} (Vocab={tok.n_vocab:,}, FastTables={tok.has_fast_tables}, SharedPrefix={shared_pfx} tok, MMBoundary={min_snap_mm} tok)"
        )

    with VizInjectorClient(prefer_vulkan=True) as viz:
        _assert(
            viz.vulkan_active, "VizInjector Vulkan compute backend is not active"
        )
        print(
            f"  [+] VizInjector Backend      : {viz.backend_name} (Vulkan Active: {viz.vulkan_active})"
        )
        req_me_default = VizInjectorRequest(edit_format="moss_image_edit")
        req_me_box = VizInjectorRequest(
            edit_format="moss_image_edit", framing_mode="inscribed"
        )
        _assert(
            req_me_default.framing_mode == "never_crop"
            and req_me_box.framing_mode == "inscribed",
            f"moss_image_edit framing routing failed: default={req_me_default.framing_mode}, explicit={req_me_box.framing_mode}",
        )
        rng = np.random.default_rng(42)
        ident_rgb = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
        for prefer_vk in (True, False):
            b_id = viz.condition_rgba8(
                ident_rgb,
                VizInjectorRequest(
                    target_width=256,
                    target_height=256,
                    emit_vit_branch=False,
                    emit_mask_branch=False,
                    deband_8bit=False,
                    mtf_boost=0.08,
                    prefer_vulkan=prefer_vk,
                ),
            )
            recon_id_pil, _ = viz.postprocess_to_pil(
                b_id.vae_tensor, apply_bayer_dither=False
            )
            max_id_lsb = int(
                np.max(
                    np.abs(
                        np.asarray(recon_id_pil[0], dtype=np.int32)
                        - ident_rgb.astype(np.int32)
                    )
                )
            )
            _assert(
                max_id_lsb == 0,
                f"1:1 Kronecker identity passthrough at mtf_boost=0.08 drifted by {max_id_lsb} LSB",
            )

        iso_h, iso_w = 128, 192
        iso_patches = (iso_h // 16) * (iso_w // 16)
        yy_iso, xx_iso = np.meshgrid(
            np.arange(iso_h, dtype=np.int32),
            np.arange(iso_w, dtype=np.int32),
            indexing="ij",
        )
        iso_rgb = np.stack(
            [
                ((yy_iso * 7 + xx_iso * 3) & 0xFF).astype(np.uint8),
                ((yy_iso * 11 + xx_iso * 5 + 40) & 0xFF).astype(np.uint8),
                ((yy_iso * 13 + xx_iso * 17 + 90) & 0xFF).astype(np.uint8),
            ],
            axis=-1,
        )
        b_iso_vk = viz.condition_rgba8(
            iso_rgb,
            VizInjectorRequest(
                target_width=iso_w,
                target_height=iso_h,
                max_vit_patches=iso_patches,
                edit_mode_384_vit=False,
                framing_mode="inscribed",
                input_role="photo_ref",
                deband_8bit=False,
                mtf_boost=0.0,
                prefer_vulkan=True,
            ),
        )
        b_iso_cpu = viz.condition_rgba8(
            iso_rgb,
            VizInjectorRequest(
                target_width=iso_w,
                target_height=iso_h,
                max_vit_patches=iso_patches,
                edit_mode_384_vit=False,
                framing_mode="inscribed",
                input_role="photo_ref",
                deband_8bit=False,
                mtf_boost=0.0,
                prefer_vulkan=False,
            ),
        )
        _assert(
            b_iso_vk.vae_tensor is not None
            and b_iso_vk.vit_patches is not None
            and b_iso_vk.vit_patches.shape == (iso_patches, 3, 2, 16, 16)
            and b_iso_vk.grid_thw is not None
            and b_iso_vk.grid_thw.tolist() == [[1, iso_h // 16, iso_w // 16]],
            f"Unexpected Gate 1.A ViT patch geometry: {None if b_iso_vk.vit_patches is None else b_iso_vk.vit_patches.shape}",
        )
        x_9d = b_iso_vk.vae_tensor.unsqueeze(1).expand(1, 2, 3, iso_h, iso_w)
        canonical_9d_patches = (
            x_9d.reshape(1, 2, 3, iso_h // 32, 2, 16, iso_w // 32, 2, 16)
            .permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
            .contiguous()
            .view(iso_patches, 3, 2, 16, 16)
        )
        vk_9d_err = float(
            (b_iso_vk.vit_patches - canonical_9d_patches).abs().max().item()
        )
        cpu_9d_err = float(
            (b_iso_cpu.vit_patches - canonical_9d_patches).abs().max().item()
        )
        with tempfile.TemporaryDirectory() as tmpdir_iso:
            iso_p = Path(tmpdir_iso) / "iso_patch.png"
            Image.fromarray(iso_rgb, mode="RGB").save(iso_p)
            py_patches, py_gthw, py_llm = pack_vision_patches_32px(
                iso_p, max_patches=iso_patches
            )
        py_9d_err = float((py_patches - canonical_9d_patches).abs().max().item())
        _assert(
            vk_9d_err == 0.0
            and cpu_9d_err < 1e-5
            and py_9d_err < 1e-5
            and py_gthw.tolist() == [[1, iso_h // 16, iso_w // 16]]
            and py_llm == iso_patches // 4,
            f"Gate 1.A 9D Merge-Block Patchification Isomorphism failed: vk_err={vk_9d_err}, cpu_err={cpu_9d_err}, py_err={py_9d_err}",
        )
        print(
            f"  [+] 9D Patch Isomorphism     : VulkanErr={vk_9d_err:.1e} | AVX2Err={cpu_9d_err:.1e} | PyFallbackErr={py_9d_err:.1e}"
        )

        cutout = np.zeros((128, 128, 4), dtype=np.uint8)
        cutout[32:96, 32:64, :3] = [255, 40, 40]
        cutout[32:96, 64:96, :3] = [40, 40, 255]
        cutout[32:96, 32:96, 3] = 255
        b_cut_vk = viz.condition_rgba8(
            cutout,
            VizInjectorRequest(
                target_width=64,
                target_height=64,
                deband_8bit=False,
                mtf_boost=0.0,
                mask_feather_radius=1.25,
                prefer_vulkan=True,
            ),
        )
        b_cut_cpu = viz.condition_rgba8(
            cutout,
            VizInjectorRequest(
                target_width=64,
                target_height=64,
                deband_8bit=False,
                mtf_boost=0.0,
                mask_feather_radius=1.25,
                prefer_vulkan=False,
            ),
        )
        _assert(
            float(b_cut_vk.latent_mask[0, 0, 1, 1].item()) == 1.0
            and float(b_cut_vk.latent_mask[0, 0, 0, 0].item()) == 0.0
            and float(
                torch.max(torch.abs(b_cut_vk.latent_mask - b_cut_cpu.latent_mask)).item()
            )
            < 1e-6,
            "Cutout alpha C2 extrema bound or unblurred coverage parity failed",
        )
        stroke_mask = np.zeros((1024, 1024, 4), dtype=np.uint8)
        for i in range(128, 896):
            stroke_mask[i, i : i + 2, :3] = 255
        stroke_mask[..., 3] = 255
        b_m_raw = viz.condition_rgba8(
            stroke_mask,
            VizInjectorRequest(
                target_width=1024,
                target_height=1024,
                input_role="inpaint_mask",
                mask_feather_radius=0.0,
                prefer_vulkan=True,
            ),
        )
        b_m_fth = viz.condition_rgba8(
            stroke_mask,
            VizInjectorRequest(
                target_width=1024,
                target_height=1024,
                input_role="inpaint_mask",
                mask_feather_radius=1.25,
                prefer_vulkan=True,
            ),
        )
        raw_cells = int(torch.sum(b_m_raw.latent_mask > 0.05).item())
        fth_cells = int(torch.sum(b_m_fth.latent_mask > 0.05).item())
        _assert(
            raw_cells >= 48
            and fth_cells > raw_cells
            and bool(torch.all(b_m_fth.latent_mask >= b_m_raw.latent_mask - 1e-6).item()),
            f"Euclidean cell-perimeter SDF dilation envelope failed: raw={raw_cells}, fth={fth_cells}",
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
            bundle.vit_patches is not None
            and bundle.meta.vit_size == (768, 768)
            and bundle.llm_tokens == 576,
            f"Expected 576 LLM tokens for full-budget 768px ViT, got {bundle.llm_tokens} ({bundle.meta.vit_size})",
        )
        req_edit_384 = VizInjectorRequest(
            target_width=512,
            target_height=512,
            max_vit_patches=576,
            edit_format="ai_toolkit_isolated",
            framing_mode="cover_crop",
            input_role="photo_ref",
            prefer_vulkan=True,
        )
        bundle_384 = viz.condition_rgba8(rgba, req_edit_384)
        _assert(
            bundle_384.vit_patches is not None
            and bundle_384.meta.vit_size == (384, 384)
            and bundle_384.llm_tokens == 144,
            f"Expected 144 LLM tokens for compact 384px ViT, got {bundle_384.llm_tokens} ({bundle_384.meta.vit_size})",
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
            and b_vit_only.meta.vit_size == (768, 768)
            and b_vit_only.llm_tokens == 576,
            f"ViT-only pass failed coupled grid geometry: vae={b_vit_only.meta.vae_size}, vit={b_vit_only.meta.vit_size}, tok={b_vit_only.llm_tokens}",
        )
        bundle.packed_latents = torch.randn((1, 128, 32, 32), dtype=torch.float32)
        bundle.vit_embeds = torch.randn((576, 4096), dtype=torch.bfloat16)
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
            and reloaded.vit_embeds.shape == (576, 4096),
            "Reloaded Tier-2 ViT embeds shape mismatch",
        )
        _assert(
            reloaded.meta.vae_size == (512, 512),
            f"Reloaded Tier-2 healed vae_size mismatch: {reloaded.meta.vae_size}",
        )
        print(
            f"  [+] .VIZJ Tier-2 Zero-Copy   : {len(blob):,} bytes | Latents={tuple(reloaded.packed_latents.shape)} | ViT={tuple(reloaded.vit_embeds.shape)}"
        )


if __name__ == "__main__":
    gate_1_native_subsystems()