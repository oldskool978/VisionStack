#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import struct
import sys
import tempfile
import threading
from typing import List
import zlib

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

sys.modules["torchvision"] = None
sys.modules["torchvision.transforms"] = None
sys.modules["torchvision.io"] = None

import numpy as np
from PIL import Image
import torch

from models.tokenizer import MOSSImageTokenizer, render_qwen35_chat_prompt
from engine import (
    _evacuate_to_cpu,
    _probe_image_dimensions,
    _resolve_boxed_reference_dims,
    load_image_tensor_16px,
    pack_vision_patches_32px,
)
from models.modeling_qwen35 import Qwen3_5HybridCache
from schema import (
    DEFAULT_LAYOUT_ITEMS,
    bbox_to_abs_centralized,
    bbox_to_normalized,
    bbox_to_pixel_rect,
    pixel_bbox_to_abs_centralized,
    scale_layout_items_to_canvas,
)
from vizinjector import (
    VIZJ_TIER_DUAL_LATENT,
    VizInjectorClient,
    VizInjectorRequest,
)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _encode_png16_rgba_bytes(rgba16: np.ndarray) -> bytes:
    assert rgba16.ndim == 3 and rgba16.shape[2] == 4 and rgba16.dtype == np.uint16
    h, w, _ = rgba16.shape
    be_u16 = np.ascontiguousarray(rgba16, dtype=">u2")
    raw_rows = bytearray()
    for y in range(h):
        raw_rows.append(0)
        raw_rows.extend(be_u16[y].tobytes())

    def _chunk(tag: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 16, 6, 0, 0, 0)
    idat = zlib.compress(bytes(raw_rows), level=6)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )


def _encode_radiance_hdr_bytes(rgb_f32: np.ndarray) -> bytes:
    assert rgb_f32.ndim == 3 and rgb_f32.shape[2] == 3
    h, w, _ = rgb_f32.shape
    header = f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {h} +X {w}\n".encode("ascii")
    rgbe = np.zeros((h, w, 4), dtype=np.uint8)
    max_c = np.max(rgb_f32, axis=-1)
    mask = max_c > 1e-32
    mant, exp = np.frexp(np.where(mask, max_c, 1.0))
    scale = np.where(mask, (mant * 256.0) / np.maximum(max_c, 1e-32), 0.0)
    rgbe[..., 0] = np.clip(rgb_f32[..., 0] * scale, 0.0, 255.0).astype(np.uint8)
    rgbe[..., 1] = np.clip(rgb_f32[..., 1] * scale, 0.0, 255.0).astype(np.uint8)
    rgbe[..., 2] = np.clip(rgb_f32[..., 2] * scale, 0.0, 255.0).astype(np.uint8)
    rgbe[..., 3] = np.where(mask, np.clip(exp + 128, 0, 255), 0).astype(np.uint8)
    if rgbe[0, 0, 0] == 2 and rgbe[0, 0, 1] == 2:
        rgbe[0, 0, 0] = 3
    return header + rgbe.tobytes()


def gate_1_native_subsystems() -> None:
    print(
        "\n--- Gate 1: Sovereign Native Subsystems (VizToken SWISS5 & VizInjector Vulkan) ---"
    )
    vt_dir = ROOT_DIR / "viztoken"
    vi_dir = ROOT_DIR / "vizinjector"
    is_win = sys.platform == "win32"
    has_gpu = torch.cuda.is_available()

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

    for canvas_h, canvas_w in ((1024, 1024), (768, 1024), (1024, 768), (512, 512)):
        for gy0, gx0, gy1, gx1 in (
            (0, 0, canvas_h // 16, canvas_w // 16),
            (4, 6, 28, 30),
            (8, 8, 24, 28),
        ):
            py0, px0, py1, px1 = gy0 * 16, gx0 * 16, gy1 * 16, gx1 * 16
            norm_rect = bbox_to_normalized(
                [py0, px0, py1, px1],
                height=canvas_h,
                width=canvas_w,
                coords_are_pixels=True,
            )
            cent_rect = bbox_to_abs_centralized(
                [py0, px0, py1, px1],
                height=canvas_h,
                width=canvas_w,
                coords_are_pixels=True,
            )
            px_from_norm = bbox_to_pixel_rect(
                norm_rect,
                height=canvas_h,
                width=canvas_w,
                coords_are_pixels=False,
            )
            px_from_norm_auto = bbox_to_pixel_rect(
                norm_rect,
                height=canvas_h,
                width=canvas_w,
                coords_are_pixels=True,
            )
            px_from_cent = bbox_to_pixel_rect(
                cent_rect,
                height=canvas_h,
                width=canvas_w,
                coords_are_pixels=False,
            )
            cent_from_px = pixel_bbox_to_abs_centralized(
                py0, px0, py1, px1, height=canvas_h, width=canvas_w
            )
            box_w, box_h = _resolve_boxed_reference_dims(
                norm_rect, canvas_w=canvas_w, canvas_h=canvas_h
            )
            _assert(
                px_from_norm == (py0, px0, py1, px1)
                and px_from_norm_auto == (py0, px0, py1, px1)
                and px_from_cent == (py0, px0, py1, px1)
                and cent_from_px == cent_rect
                and (box_w, box_h) == (max(64, px1 - px0), max(64, py1 - py0)),
                f"Cross-domain bbox coordinate roundtrip drift on {canvas_w}x{canvas_h}: {(py0, px0, py1, px1)} -> norm={px_from_norm}, cent={px_from_cent}",
            )

    scaled_512 = scale_layout_items_to_canvas(
        DEFAULT_LAYOUT_ITEMS,
        height=512,
        width=512,
        src_height=1024,
        src_width=1024,
        coords_are_pixels=True,
    )
    _assert(
        scaled_512[0]["bbox"] == [0, 0, 512, 288]
        and scaled_512[1]["bbox"] == [128, 48, 256, 224]
        and scaled_512[2]["bbox"] == [224, 272, 512, 512],
        f"scale_layout_items_to_canvas failed 16px lattice-locked scaling to 512x512: {scaled_512}",
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

        rgba16 = np.zeros((64, 64, 4), dtype=np.uint16)
        ramp16 = np.linspace(16384, 49152, 64, dtype=np.uint16)
        rgba16[:, :, 0] = ramp16[None, :]
        rgba16[:, :, 1] = ramp16[:, None]
        rgba16[:, :, 2] = 32768
        rgba16[:, :, 3] = 65535
        rgba16[:8, :8, :3] = 0
        rgba16[:8, :8, 3] = 0
        png16_bytes = _encode_png16_rgba_bytes(rgba16)
        b_png16 = viz.condition_image(
            png16_bytes,
            VizInjectorRequest(
                target_width=64,
                target_height=64,
                deband_8bit=False,
                mtf_boost=0.0,
            ),
        )
        b_arr16 = viz.condition_image(
            rgba16,
            VizInjectorRequest(
                target_width=64,
                target_height=64,
                deband_8bit=False,
                mtf_boost=0.0,
            ),
        )
        _assert(
            b_png16.vae_tensor is not None
            and b_arr16.vae_tensor is not None
            and bool(torch.isfinite(b_png16.vae_tensor).all().item())
            and float((b_png16.vae_tensor - b_arr16.vae_tensor).abs().max().item())
            == 0.0,
            "16-bit PNG vs direct uint16 ndarray conditioning diverged or produced non-finite tensor",
        )
        row_srgb16 = (b_png16.vae_tensor[0, 0, 32, :].numpy() * 0.5 + 0.5) * 65535.0
        max_16bit_err = float(np.max(np.abs(row_srgb16 - ramp16.astype(np.float32))))
        _assert(
            max_16bit_err < 1.5
            and float(b_png16.vae_tensor[0, 1, 2, 2].item()) > -0.90,
            f"16-bit PNG radiometric precision or 8px outer ring propagation failed: {max_16bit_err:.3f} / 65535",
        )

        f32_ramp = (
            torch.linspace(-0.75, 0.75, 64, dtype=torch.float32)
            .view(1, 1, 1, 64)
            .expand(1, 3, 64, 64)
            .contiguous()
        )
        b_tensor_f32 = viz.condition_image(
            f32_ramp,
            VizInjectorRequest(
                target_width=64,
                target_height=64,
                deband_8bit=False,
                mtf_boost=0.0,
                emit_mask_branch=False,
            ),
        )
        b_tensor_bf16 = viz.condition_image(
            f32_ramp.to(torch.bfloat16),
            VizInjectorRequest(
                target_width=64,
                target_height=64,
                deband_8bit=False,
                mtf_boost=0.0,
                emit_mask_branch=False,
            ),
        )
        tensor_f32_err = float(
            (b_tensor_f32.vae_tensor - f32_ramp).abs().max().item()
        )
        tensor_bf16_err = float(
            (b_tensor_bf16.vae_tensor - f32_ramp).abs().max().item()
        )
        t_loaded_16 = load_image_tensor_16px(f32_ramp, target_h=64, target_w=64)
        t_patches_32, t_gthw_32, t_llm_32 = pack_vision_patches_32px(
            f32_ramp, max_patches=64
        )
        _assert(
            tensor_f32_err < 1e-4
            and tensor_bf16_err < 4e-3
            and float((t_loaded_16 - f32_ramp).abs().max().item()) < 1e-4
            and t_patches_32.shape == (64, 3, 2, 16, 16)
            and t_llm_32 == 16,
            f"Direct FP32/BF16 torch.Tensor conditioning failed sub-LSB accuracy: f32_err={tensor_f32_err:.2e}, bf16_err={tensor_bf16_err:.2e}",
        )
        if has_gpu:
            with torch.inference_mode():
                gpu_tmp = torch.randn((1, 4, 8, 8), device="cuda", dtype=torch.float32)
                evac_cpu = _evacuate_to_cpu(gpu_tmp)
            _assert(
                b_tensor_f32.vae_tensor.is_pinned()
                and b_tensor_f32.vit_patches.is_pinned()
                and t_loaded_16.is_pinned()
                and t_patches_32.is_pinned()
                and evac_cpu is not None
                and evac_cpu.is_pinned()
                and not evac_cpu.is_inference(),
                "Harvested or evacuated host tensors must be page-locked (is_pinned=True) on the first pass",
            )

        hdr_rgb = np.zeros((64, 64, 3), dtype=np.float32)
        hdr_x = np.linspace(0.1, 8.0, 64, dtype=np.float32)
        hdr_rgb[:, :, 0] = hdr_x[None, :]
        hdr_rgb[:, :, 1] = 0.5
        hdr_rgb[:, :, 2] = 0.25
        hdr_bytes = _encode_radiance_hdr_bytes(hdr_rgb)
        b_hdr = viz.condition_image(
            hdr_bytes,
            VizInjectorRequest(
                target_width=64,
                target_height=64,
                emit_vit_branch=False,
                emit_mask_branch=False,
                mtf_boost=0.0,
            ),
        )
        hdr_row_r = b_hdr.vae_tensor[0, 0, 32, :].numpy()
        _assert(
            b_hdr.vae_tensor is not None
            and bool(torch.isfinite(b_hdr.vae_tensor).all().item())
            and bool(np.all(np.diff(hdr_row_r) >= -1e-4))
            and float(hdr_row_r[-1]) > float(hdr_row_r[8]),
            "32-bit Radiance HDR C-infinity Pade [2/2] ACES tonemapping failed monotonic roll-off",
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
            f"  [+] 9D Patch & FP32/16b/HDR  : VK9DErr={vk_9d_err:.1e} | FP32TensorErr={tensor_f32_err:.1e} | 16bErr={max_16bit_err:.2f} LSB | PinnedDMA={has_gpu}"
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
        b_cut_1to1 = viz.condition_rgba8(
            cutout,
            VizInjectorRequest(
                target_width=128,
                target_height=128,
                deband_8bit=False,
                mtf_boost=0.0,
                emit_vit_branch=False,
                prefer_vulkan=True,
            ),
        )
        area_counts = [0, 1, 2, 3, 4, 8, 16, 32, 62, 63, 64, 65, 66, 128, 192, 256]
        cut_area_rgba = np.zeros((64, 64, 4), dtype=np.uint8)
        cut_area_rgba[..., :3] = 200
        for c_idx, k_cnt in enumerate(area_counts):
            cy_c, cx_c = divmod(c_idx, 4)
            cell_a = np.zeros(256, dtype=np.uint8)
            cell_a[:k_cnt] = 255
            cut_area_rgba[
                cy_c * 16 : (cy_c + 1) * 16, cx_c * 16 : (cx_c + 1) * 16, 3
            ] = cell_a.reshape(16, 16)
        b_area_vk = viz.condition_rgba8(
            cut_area_rgba,
            VizInjectorRequest(
                target_width=64,
                target_height=64,
                deband_8bit=False,
                mtf_boost=0.0,
                emit_vit_branch=False,
                prefer_vulkan=True,
            ),
        )
        expected_area = np.array(area_counts, dtype=np.float32) / 256.0
        area_err_vk = float(
            np.max(np.abs(b_area_vk.latent_mask.view(-1).numpy() - expected_area))
        )
        _assert(
            float(b_cut_vk.latent_mask[0, 0, 1, 1].item()) == 1.0
            and float(b_cut_vk.latent_mask[0, 0, 0, 0].item()) == 0.0
            and float(b_cut_1to1.vae_tensor[0, 0, 64, 26].item()) > 0.95
            and area_err_vk < 1e-6
            and float(
                torch.max(torch.abs(b_cut_vk.latent_mask - b_cut_cpu.latent_mask)).item()
            )
            < 1e-6,
            f"Cutout Mode 2 area integral or 8px outer ring propagation failed: area_err={area_err_vk:.2e}",
        )

        ramp_mask_rgba = np.zeros((256, 256, 4), dtype=np.uint8)
        ramp_mask_rgba[..., 3] = 255
        for cell_id in range(256):
            cy_cell, cx_cell = divmod(cell_id, 16)
            cell_r = np.zeros(256, dtype=np.uint8)
            cell_r[: cell_id + 1] = 255
            ramp_mask_rgba[
                cy_cell * 16 : (cy_cell + 1) * 16,
                cx_cell * 16 : (cx_cell + 1) * 16,
                0,
            ] = cell_r.reshape(16, 16)
        b_ramp_vk = viz.condition_rgba8(
            ramp_mask_rgba,
            VizInjectorRequest(
                target_width=256,
                target_height=256,
                input_role="inpaint_mask",
                mask_feather_radius=0.0,
                prefer_vulkan=True,
            ),
        )
        ramp_vals_vk = b_ramp_vk.latent_mask.view(-1).numpy()
        avg_k = np.arange(1, 257, dtype=np.float32) / 256.0
        expected_agm = 0.45 * avg_k + 0.55 * np.sqrt(avg_k)
        agm_err_vk = float(np.max(np.abs(ramp_vals_vk - expected_agm)))
        _assert(
            bool(np.all(np.diff(ramp_vals_vk) > 0.0)) and agm_err_vk < 1e-6,
            f"Mode 4 C-infinity AGM mask monotonicity failed: err={agm_err_vk:.2e}",
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
            f"Occupancy-proportional SDF dilation envelope failed: raw={raw_cells}, fth={fth_cells}",
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

        ref_cpu_512 = viz.condition_rgba8(
            rgba,
            req_edit.model_copy(update={"prefer_vulkan": False}),
        )
        soak_errors: List[str] = []

        def _soak_worker(tid: int) -> None:
            use_vk = (tid % 2) == 0
            w_req = req_edit.model_copy(update={"prefer_vulkan": use_vk})
            expected = bundle if use_vk else ref_cpu_512
            for _ in range(12):
                out_w = viz.condition_rgba8(rgba, w_req)
                if out_w.vae_tensor is None or out_w.vit_patches is None:
                    soak_errors.append(f"Thread {tid} returned None tensor")
                    return
                if (
                    float((out_w.vae_tensor - expected.vae_tensor).abs().max().item())
                    > 0.0
                    or float(
                        (out_w.vit_patches - expected.vit_patches).abs().max().item()
                    )
                    > 0.0
                ):
                    soak_errors.append(f"Thread {tid} (vk={use_vk}) race drift")
                    return

        threads = [threading.Thread(target=_soak_worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        _assert(
            not soak_errors,
            f"Multithreaded VizInjector contention failure: {soak_errors[0] if soak_errors else ''}",
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
            f"  [+] .VIZJ Tier-2 & MT Soak   : {len(blob):,} bytes | Latents={tuple(reloaded.packed_latents.shape)} | ViT={tuple(reloaded.vit_embeds.shape)} | 8-Thread Soak=0.0 Drift"
        )


if __name__ == "__main__":
    gate_1_native_subsystems()