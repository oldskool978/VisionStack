#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
BIN_DIR = ROOT_DIR / "bin"
CACHE_DIR = ROOT_DIR / ".forge_cache"
IS_WIN = sys.platform == "win32"
HERMETIC_PYTHON_EXE = BIN_DIR / "python" / ("python.exe" if IS_WIN else "bin/python")

if (
    HERMETIC_PYTHON_EXE.exists()
    and Path(sys.executable).resolve() != HERMETIC_PYTHON_EXE.resolve()
):
    respawn_cmd = [
        str(HERMETIC_PYTHON_EXE.resolve()),
        "-I",
        str(Path(__file__).resolve()),
    ] + sys.argv[1:]
    res = subprocess.run(respawn_cmd)
    sys.exit(res.returncode)

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "12.0.0"
os.environ["ROCM_PATH"] = str(BIN_DIR / "python")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"
os.environ["MIOPEN_FIND_MODE"] = "2"
os.environ["MIOPEN_LOG_LEVEL"] = "0"
os.environ["MIOPEN_ENABLE_LOGGING"] = "0"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from PIL import Image
import torch

from python import (
    VIZJ_TIER_DUAL_LATENT,
    VIZJ_TIER_VAE_LATENT,
    VizInjectorClient,
    VizInjectorRequest,
    resolve_local_vae_dir,
)


def srgb_to_linear_np(x: np.ndarray) -> np.ndarray:
    c = np.clip(x, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb_np(lin: np.ndarray) -> np.ndarray:
    v = np.clip(lin, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * (v ** (1.0 / 2.4)) - 0.055)


def upstream_pil_condition(
    pil_img: Image.Image,
    target_w: int = 1024,
    target_h: int = 1024,
    max_patches: int = 2304,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb = pil_img.convert("RGB")
    w_in, h_in = rgb.size
    h_out = (target_h // 16) * 16
    w_out = (target_w // 16) * 16
    rgb_vae = (
        rgb.resize((w_out, h_out), resample=Image.Resampling.BICUBIC)
        if (w_in, h_in) != (w_out, h_out)
        else rgb
    )
    arr_vae = np.asarray(rgb_vae, dtype=np.float32) / 127.5 - 1.0
    t_vae = torch.from_numpy(arr_vae).permute(2, 0, 1).unsqueeze(0).contiguous()

    max_pixels = max_patches * 16 * 16
    scale = min(1.0, math.sqrt(float(max_pixels) / float(max(1, w_in * h_in))))
    h_vis = max(32, (int(round(h_in * scale)) // 32) * 32)
    w_vis = max(32, (int(round(w_in * scale)) // 32) * 32)
    rgb_vis = (
        rgb.resize((w_vis, h_vis), resample=Image.Resampling.BICUBIC)
        if (w_in, h_in) != (w_vis, h_vis)
        else rgb
    )
    arr_vis = np.asarray(rgb_vis, dtype=np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    normed = (arr_vis - mean) / std
    chw = torch.from_numpy(normed).permute(2, 0, 1).contiguous()
    grid_h = h_vis // 16
    grid_w = w_vis // 16
    merge = 2
    patches = chw.unsqueeze(1).expand(3, 2, h_vis, w_vis)
    patches = patches.view(
        3, 2, grid_h // merge, merge, 16, grid_w // merge, merge, 16
    )
    patches = (
        patches.permute(2, 5, 3, 6, 0, 1, 4, 7)
        .contiguous()
        .view(grid_h * grid_w, 3, 2, 16, 16)
    )
    grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)
    return t_vae, patches, grid_thw


def benchmark_1_radiometric_photon_conservation(
    client: VizInjectorClient,
) -> Dict[str, Any]:
    print(
        "\n--- [Suite 1/6] Radiometric Photon Conservation & Step-Edge Halo Benchmark ---"
    )
    h_hi, w_hi = 1024, 1024
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, h_hi, dtype=np.float64),
        np.linspace(-1.0, 1.0, w_hi, dtype=np.float64),
        indexing="ij",
    )
    r2 = xx**2 + yy**2
    lin_hi = np.zeros((h_hi, w_hi, 3), dtype=np.float64)
    lin_hi[..., 0] = 0.5 + 0.48 * np.cos(140.0 * r2)
    lin_hi[..., 1] = np.where((xx + 0.3 * yy) > 0.0, 0.90, 0.04)
    lin_hi[..., 2] = 0.5 + 0.48 * np.sin(90.0 * (xx**2 - yy**2))

    gt_lin_lo = lin_hi.reshape(512, 2, 512, 2, 3).mean(axis=(1, 3))

    srgb_hi_u8 = np.clip(np.round(linear_to_srgb_np(lin_hi) * 255.0), 0, 255).astype(
        np.uint8
    )
    pil_hi = Image.fromarray(srgb_hi_u8, mode="RGB")

    up_vae, _, _ = upstream_pil_condition(pil_hi, 512, 512)
    up_srgb = np.clip(
        up_vae[0].permute(1, 2, 0).numpy() * 0.5 + 0.5, 0.0, 1.0
    )
    up_lin = srgb_to_linear_np(up_srgb)

    viz_bundle = client.condition_rgba8(
        srgb_hi_u8,
        VizInjectorRequest(
            target_width=512,
            target_height=512,
            emit_vit_branch=False,
            emit_mask_branch=False,
            deband_8bit=False,
            mtf_boost=0.0,
        ),
    )
    viz_srgb = np.clip(
        viz_bundle.vae_tensor[0].permute(1, 2, 0).numpy() * 0.5 + 0.5,
        0.0,
        1.0,
    )
    viz_lin = srgb_to_linear_np(viz_srgb)

    mse_up = float(np.mean((up_lin - gt_lin_lo) ** 2))
    mse_viz = float(np.mean((viz_lin - gt_lin_lo) ** 2))
    psnr_up = 10.0 * math.log10(1.0 / max(mse_up, 1e-12))
    psnr_viz = 10.0 * math.log10(1.0 / max(mse_viz, 1e-12))

    step_up_g = up_srgb[..., 1] * 255.0
    step_viz_g = viz_srgb[..., 1] * 255.0
    gt_min_g = float(linear_to_srgb_np(np.array([0.04]))[0] * 255.0)
    gt_max_g = float(linear_to_srgb_np(np.array([0.90]))[0] * 255.0)
    ring_up = max(0.0, gt_min_g - float(np.min(step_up_g))) + max(
        0.0, float(np.max(step_up_g)) - gt_max_g
    )
    ring_viz = max(0.0, gt_min_g - float(np.min(step_viz_g))) + max(
        0.0, float(np.max(step_viz_g)) - gt_max_g
    )

    print(
        f"  [Upstream PIL Bicubic] Linear Photon PSNR = {psnr_up:6.2f} dB | Total Step-Edge Ringing = {ring_up:5.2f} LSB"
    )
    print(
        f"  [VizInjector Sovereign] Linear Photon PSNR = {psnr_viz:6.2f} dB | Total Step-Edge Ringing = {ring_viz:5.2f} LSB"
    )
    print(
        f"  [+] Radiometric Gain  : +{psnr_viz - psnr_up:.2f} dB Linear Fidelity | {ring_up - ring_viz:.2f} LSB Ringing Reduction"
    )
    assert psnr_viz > psnr_up, "VizInjector must exceed upstream linear photon PSNR"
    return {
        "upstream_linear_psnr_db": round(psnr_up, 3),
        "vizinjector_linear_psnr_db": round(psnr_viz, 3),
        "upstream_ringing_lsb": round(ring_up, 3),
        "vizinjector_ringing_lsb": round(ring_viz, 3),
    }


def benchmark_2_geometric_isotropy_and_dual_pathway(
    client: VizInjectorClient,
) -> Dict[str, Any]:
    print(
        "\n--- [Suite 2/6] Anisotropic Geometric Shear & Dual-Pathway Coherence Benchmark ---"
    )
    h_in, w_in = 720, 1280
    yy, xx = np.ogrid[:h_in, :w_in]
    cy, cx = (h_in - 1) / 2.0, (w_in - 1) / 2.0
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    src_rgb = np.zeros((h_in, w_in, 3), dtype=np.uint8)
    src_rgb[r2 <= (220.0**2)] = 255

    pil_src = Image.fromarray(src_rgb, mode="RGB")
    up_vae, _, _ = upstream_pil_condition(pil_src, 1024, 1024)
    up_mask = (up_vae[0, 0].numpy() > 0.0).astype(np.float64)
    u_ys, u_xs = np.nonzero(up_mask)
    up_shear = abs(float(np.std(u_xs)) - float(np.std(u_ys))) / max(
        float(np.std(u_xs)), float(np.std(u_ys)), 1e-9
    )

    viz_b = client.condition_rgba8(
        src_rgb,
        VizInjectorRequest(
            target_width=1024,
            target_height=1024,
            framing_mode="inscribed",
        ),
    )
    v_mask = (viz_b.vae_tensor[0, 0].numpy() > 0.0).astype(np.float64)
    v_ys, v_xs = np.nonzero(v_mask)
    viz_shear = abs(float(np.std(v_xs)) - float(np.std(v_ys))) / max(
        float(np.std(v_xs)), float(np.std(v_ys)), 1e-9
    )

    print(
        f"  [Upstream PIL Resize ] 1280x720 -> 1024x1024 Circle Shear Distortion = {up_shear * 100.0:6.2f}%"
    )
    print(
        f"  [VizInjector Inscribed] 1280x720 -> 1024x1024 Circle Shear Distortion = {viz_shear * 100.0:6.3f}%"
    )
    assert viz_shear < 0.005, "VizInjector isotropic framing violated"
    return {
        "upstream_shear_pct": round(up_shear * 100.0, 3),
        "vizinjector_shear_pct": round(viz_shear * 100.0, 4),
    }


def benchmark_3_cutout_alpha_border_fidelity(
    client: VizInjectorClient,
) -> Dict[str, Any]:
    print(
        "\n--- [Suite 3/6] Cutout Alpha-Border Halo & Garbage-Bleed Immunity Benchmark ---"
    )
    sz = 512
    yy, xx = np.ogrid[:sz, :sz]
    r = np.sqrt((xx - 255.5) ** 2 + (yy - 255.5) ** 2)
    rgba = np.zeros((sz, sz, 4), dtype=np.uint8)
    inside = r <= 160.0
    rgba[inside, 0] = 240
    rgba[inside, 1] = 220
    rgba[inside, 2] = 180
    rgba[inside, 3] = 255
    rgba[~inside, 0] = 0
    rgba[~inside, 1] = 255
    rgba[~inside, 2] = 0
    rgba[~inside, 3] = 0

    pil_rgba = Image.fromarray(rgba, mode="RGBA")
    up_vae, _, _ = upstream_pil_condition(pil_rgba, 256, 256)
    up_rgb = (up_vae[0].permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255.0

    viz_b = client.condition_rgba8(
        rgba,
        VizInjectorRequest(
            target_width=256,
            target_height=256,
            deband_8bit=False,
            mtf_boost=0.0,
        ),
    )
    viz_rgb = (
        viz_b.vae_tensor[0].permute(1, 2, 0).numpy() * 0.5 + 0.5
    ) * 255.0

    yy_lo, xx_lo = np.ogrid[:256, :256]
    r_lo = np.sqrt((xx_lo - 127.5) ** 2 + (yy_lo - 127.5) ** 2)
    border_ring = (r_lo >= 78.5) & (r_lo <= 80.5)
    target_color = np.array([240.0, 220.0, 180.0], dtype=np.float32)

    up_border_err = float(
        np.mean(np.abs(up_rgb[border_ring] - target_color))
    )
    viz_border_err = float(
        np.mean(np.abs(viz_rgb[border_ring] - target_color))
    )

    print(
        f"  [Upstream PIL RGB Cast] Cutout Border Contamination Error = {up_border_err:6.2f} LSB"
    )
    print(
        f"  [VizInjector 2-Pass   ] Cutout Border Contamination Error = {viz_border_err:6.2f} LSB"
    )
    assert viz_border_err < up_border_err * 0.25
    return {
        "upstream_border_mae_lsb": round(up_border_err, 3),
        "vizinjector_border_mae_lsb": round(viz_border_err, 3),
    }


def benchmark_4_control_maps_and_masks(
    client: VizInjectorClient,
) -> Dict[str, Any]:
    print(
        "\n--- [Suite 4/6] Discrete Region Color-Map & Thin-Stroke Inpaint Mask Benchmark ---"
    )
    cmap = np.zeros((512, 512, 4), dtype=np.uint8)
    cmap[:256, :256] = [255, 0, 0, 255]
    cmap[:256, 256:] = [0, 255, 0, 255]
    cmap[256:, :256] = [0, 0, 255, 255]
    cmap[256:, 256:] = [255, 255, 0, 255]

    up_cmap_pil = (
        Image.fromarray(cmap[..., :3], mode="RGB").resize(
            (256, 256), resample=Image.Resampling.BICUBIC
        )
    )
    up_unique = len(
        np.unique(np.asarray(up_cmap_pil).reshape(-1, 3), axis=0)
    )
    up_hallucinated = max(0, up_unique - 4)

    viz_cmap = client.condition_rgba8(
        cmap,
        VizInjectorRequest(
            target_width=256,
            target_height=256,
            input_role="color_map",
        ),
    )
    viz_rgb = np.round(
        (viz_cmap.vae_tensor[0].permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255.0
    ).astype(np.int32)
    viz_unique = len(np.unique(viz_rgb.reshape(-1, 3), axis=0))
    viz_hallucinated = max(0, viz_unique - 4)

    mask_hi = np.zeros((1024, 1024), dtype=np.uint8)
    stroke_rows = [64 + 16 * k + (k % 7) * 2 for k in range(56)]
    for y in stroke_rows:
        mask_hi[y : y + 2, 128:896] = 255

    up_mask_lo = (
        np.asarray(
            Image.fromarray(mask_hi, mode="L").resize(
                (64, 64), resample=Image.Resampling.NEAREST
            ),
            dtype=np.float32,
        )
        / 255.0
    )
    up_strokes_detected = int(np.sum(up_mask_lo[:, 32] > 0.05))

    mask_rgba = np.stack([mask_hi, mask_hi, mask_hi, mask_hi], axis=-1)
    viz_mask_b = client.condition_rgba8(
        mask_rgba,
        VizInjectorRequest(
            target_width=1024,
            target_height=1024,
            input_role="inpaint_mask",
            mask_feather_radius=0.0,
        ),
    )
    viz_mask_lo = viz_mask_b.latent_mask[0, 0].numpy()
    viz_strokes_detected = int(np.sum(viz_mask_lo[:, 32] > 0.05))
    total_strokes = len(stroke_rows)

    print(
        f"  [Region Color-Map IDs] Upstream Hallucinated IDs = {up_hallucinated:3d} | VizInjector Hallucinated IDs = {viz_hallucinated:3d}"
    )
    print(
        f"  [2px Mask Survival   ] Upstream Detected = {up_strokes_detected:2d}/{total_strokes} ({up_strokes_detected / total_strokes * 100.0:5.1f}%) | VizInjector = {viz_strokes_detected:2d}/{total_strokes} ({viz_strokes_detected / total_strokes * 100.0:5.1f}%)"
    )
    assert viz_hallucinated == 0 and viz_strokes_detected == total_strokes
    return {
        "upstream_hallucinated_color_ids": up_hallucinated,
        "vizinjector_hallucinated_color_ids": viz_hallucinated,
        "upstream_mask_stroke_survival_pct": round(
            up_strokes_detected / total_strokes * 100.0, 2
        ),
        "vizinjector_mask_stroke_survival_pct": round(
            viz_strokes_detected / total_strokes * 100.0, 2
        ),
    }


def benchmark_5_vae_roundtrip_and_subphase(
    client: VizInjectorClient,
) -> Dict[str, Any]:
    print(
        "\n--- [Suite 5/6] Flux2AutoencoderKL Round-Trip & Sub-Phase Acutance Benchmark ---"
    )
    if resolve_local_vae_dir() is None:
        print(
            "  [i] Skipping live VAE pass (run `python tools/download_vae.py` to stage standalone Flux2VAE)."
        )
        return {"skipped": True}

    vae = client.get_or_load_local_vae()
    h, w = 768, 768
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, h, dtype=np.float32),
        np.linspace(-1.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    r = np.sqrt(xx**2 + yy**2)
    test_img = np.zeros((h, w, 3), dtype=np.uint8)
    test_img[..., 0] = np.clip(
        (0.5 + 0.45 * np.sin(45.0 * r)) * 255.0, 0, 255
    ).astype(np.uint8)
    test_img[..., 1] = np.clip(
        (0.5 + 0.45 * np.cos(35.0 * (xx - yy))) * 255.0, 0, 255
    ).astype(np.uint8)
    test_img[..., 2] = np.clip((0.5 + 0.4 * xx) * 255.0, 0, 255).astype(
        np.uint8
    )

    pil_in = Image.fromarray(test_img, mode="RGB")
    up_vae_in, _, _ = upstream_pil_condition(pil_in, 512, 512)
    up_lat = vae.encode_to_packed_latents(up_vae_in)
    up_rec = vae.decode_packed_latents(up_lat)
    _, up_telem = client.postprocess_to_pil(up_rec)

    viz_b = client.latentize(
        test_img,
        vae_module=vae,
        request=VizInjectorRequest(
            target_width=512,
            target_height=512,
            emit_vit_branch=False,
            mtf_boost=0.10,
        ),
    )
    viz_rec = vae.decode_packed_latents(viz_b.packed_latents)
    _, viz_telem = client.postprocess_to_pil(viz_rec)

    print(
        f"  [Upstream -> Flux2VAE  ] Decoded Sharpness = {up_telem[0].laplacian_sharpness:6.4f} | Entropy = {up_telem[0].shannon_entropy_bits:5.3f} bits"
    )
    print(
        f"  [VizInjector -> Flux2VAE] Decoded Sharpness = {viz_telem[0].laplacian_sharpness:6.4f} | Entropy = {viz_telem[0].shannon_entropy_bits:5.3f} bits"
    )
    return {
        "upstream_decoded_sharpness": round(
            up_telem[0].laplacian_sharpness, 4
        ),
        "vizinjector_decoded_sharpness": round(
            viz_telem[0].laplacian_sharpness, 4
        ),
    }


def benchmark_6_compute_throughput(
    client: VizInjectorClient,
) -> Dict[str, Any]:
    print(
        "\n--- [Suite 6/6] Compute Stack Throughput & .VIZJ Zero-Copy Latency Benchmark ---"
    )
    rng = np.random.default_rng(123)
    src_rgba = rng.integers(0, 256, size=(1536, 1536, 4), dtype=np.uint8)
    pil_img = Image.fromarray(src_rgba, mode="RGBA")
    iters = 10

    _ = upstream_pil_condition(pil_img, 1024, 1024)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = upstream_pil_condition(pil_img, 1024, 1024)
    up_ms = ((time.perf_counter() - t0) / iters) * 1000.0

    req_cpu = VizInjectorRequest(
        target_width=1024,
        target_height=1024,
        prefer_vulkan=False,
        deband_8bit=False,
    )
    _ = client.condition_rgba8(src_rgba, req_cpu)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = client.condition_rgba8(src_rgba, req_cpu)
    cpu_ms = ((time.perf_counter() - t0) / iters) * 1000.0

    req_vk = VizInjectorRequest(
        target_width=1024,
        target_height=1024,
        prefer_vulkan=True,
        deband_8bit=False,
    )
    b_vk = client.condition_rgba8(src_rgba, req_vk)
    t0 = time.perf_counter()
    for _ in range(iters):
        b_vk = client.condition_rgba8(src_rgba, req_vk)
    vk_ms = ((time.perf_counter() - t0) / iters) * 1000.0

    b_vk.packed_latents = torch.randn((1, 128, 64, 64), dtype=torch.float32)
    b_vk.vit_embeds = torch.randn((576, 4096), dtype=torch.bfloat16)
    vizj_t2 = client.save_vizj(b_vk, tier=VIZJ_TIER_DUAL_LATENT, use_bf16=True)

    t0 = time.perf_counter()
    for _ in range(iters * 5):
        _ = client.load_vizj_bytes(vizj_t2)
    vizj_ms = ((time.perf_counter() - t0) / (iters * 5)) * 1000.0

    print(
        f"  [1] Upstream Python (PIL + PyTorch Permute) : {up_ms:7.2f} ms / image (1.00x)"
    )
    print(
        f"  [2] VizInjector C11 AVX2 Polyphase Engine   : {cpu_ms:7.2f} ms / image ({up_ms / max(cpu_ms, 1e-3):.2f}x)"
    )
    print(
        f"  [3] VizInjector Vulkan Compute ({client.backend_name[:18]}) : {vk_ms:7.2f} ms / image ({up_ms / max(vk_ms, 1e-3):.2f}x)"
    )
    print(
        f"  [4] VizInjector .VIZJ Tier-2 Dual-Cache Load: {vizj_ms:7.3f} ms / image ({up_ms / max(vizj_ms, 1e-4):.1f}x)"
    )
    return {
        "upstream_pil_ms": round(up_ms, 3),
        "vizinjector_avx2_ms": round(cpu_ms, 3),
        "vizinjector_vulkan_ms": round(vk_ms, 3),
        "vizinjector_vizj_cache_ms": round(vizj_ms, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VizInjector vs. Upstream Paradigm Quantitative Benchmark Suite"
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=str(CACHE_DIR / "benchmark_report.json"),
        help="Path to write JSON benchmark report",
    )
    args = parser.parse_args()

    print("=" * 90)
    print(
        "        VIZINJECTOR VS. UPSTREAM PARADIGM: QUALITATIVE & COMPUTE SUPREMACY BENCHMARK"
    )
    print("=" * 90)

    client = VizInjectorClient(prefer_vulkan=True)
    report: Dict[str, Any] = {
        "backend": client.backend_name,
        "vulkan_active": client.vulkan_active,
        "suites": {},
    }

    report["suites"]["radiometric"] = benchmark_1_radiometric_photon_conservation(
        client
    )
    report["suites"]["isotropy"] = benchmark_2_geometric_isotropy_and_dual_pathway(
        client
    )
    report["suites"]["alpha_cutout"] = benchmark_3_cutout_alpha_border_fidelity(
        client
    )
    report["suites"]["control_and_masks"] = benchmark_4_control_maps_and_masks(
        client
    )
    report["suites"]["vae_roundtrip"] = benchmark_5_vae_roundtrip_and_subphase(
        client
    )
    report["suites"]["throughput"] = benchmark_6_compute_throughput(client)

    out_file = Path(args.json_out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print(f"[+] BENCHMARK COMPLETE. Full telemetry written to: {out_file}")
    print("=" * 90)


if __name__ == "__main__":
    main()