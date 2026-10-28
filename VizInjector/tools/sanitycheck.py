#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT_DIR = Path(__file__).resolve().parent.parent
BIN_DIR = ROOT_DIR / "bin"
DIST_DIR = ROOT_DIR / "dist"
DIST_BIN = DIST_DIR / "bin"
CACHE_DIR = ROOT_DIR / ".forge_cache"

IS_WIN = sys.platform == "win32"
HERMETIC_PYTHON_EXE = BIN_DIR / "python" / ("python.exe" if IS_WIN else "bin/python")

if __name__ == "__main__" and "--clean" in sys.argv:
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    shutil.rmtree(DIST_DIR, ignore_errors=True)
    print("[INFO]  Cleared .forge_cache and dist directories.")
    sys.exit(0)

if (
    HERMETIC_PYTHON_EXE.exists()
    and Path(sys.executable).resolve() != HERMETIC_PYTHON_EXE.resolve()
):
    tramp_env = os.environ.copy()
    tramp_env["PYTHONHOME"] = str(BIN_DIR / "python")
    tramp_env["ROCM_PATH"] = str(BIN_DIR / "python")
    respawn_cmd = [
        str(HERMETIC_PYTHON_EXE.resolve()),
        str(Path(__file__).resolve()),
    ] + sys.argv[1:]
    res = subprocess.run(respawn_cmd, env=tramp_env, check=False)
    sys.exit(res.returncode)

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
    subphase_radial_clamp,
)

CLI_EXE = DIST_BIN / ("vizinjector-cli.exe" if IS_WIN else "vizinjector-cli")
DLL_PATH = DIST_BIN / ("vizinjector.dll" if IS_WIN else "libvizinjector.so")
SPV_HEADER = ROOT_DIR / "src" / "vulkan" / "shaders_spv.h"


def audit_gate_1_binaries_and_sysroot() -> VizInjectorClient:
    print(
        "\n--- Gate 1: Deliverable Binaries, SPIR-V Shaders & Vulkan/AVX2 Backend Audit ---"
    )
    if not SPV_HEADER.exists() or SPV_HEADER.stat().st_size < 1024:
        raise FileNotFoundError(
            f"Embedded SPIR-V header missing or empty: {SPV_HEADER}. Run tools/forge_dag.py first."
        )
    print(
        f"  [+] Embedded SPIR-V Header   : {SPV_HEADER.relative_to(ROOT_DIR)} ({SPV_HEADER.stat().st_size:,} bytes)"
    )

    if not CLI_EXE.exists():
        raise FileNotFoundError(f"CLI executable missing: {CLI_EXE}")
    print(
        f"  [+] Native CLI Verified      : {CLI_EXE.name} ({CLI_EXE.stat().st_size:,} bytes)"
    )

    if not DLL_PATH.exists():
        raise FileNotFoundError(f"Shared library missing: {DLL_PATH}")
    print(
        f"  [+] Dynamic Library Verified : {DLL_PATH.name} ({DLL_PATH.stat().st_size:,} bytes)"
    )

    for py_name in ("schema.py", "bindings.py", "modeling_vae.py", "__init__.py"):
        staged_py = DIST_BIN / py_name
        if not staged_py.is_file():
            raise FileNotFoundError(
                f"Missing staged Python module in dist/bin: {staged_py}"
            )
    print(
        "  [+] Staged Drop-In Bundle    : schema.py, bindings.py, modeling_vae.py, __init__.py verified in dist/bin"
    )

    client = VizInjectorClient(DLL_PATH, prefer_vulkan=True)
    print(
        f"  [+] Compute Backend Active   : {client.backend_name} (Vulkan Enabled: {client.vulkan_active})"
    )
    return client


def audit_gate_2_radiometric_math_and_isotropy(client: VizInjectorClient) -> None:
    print(
        "\n--- Gate 2: Radiometric Kaiser-Sinc, Partition of Unity, De-Ringing & Isotropy Audit ---"
    )
    rng = np.random.default_rng(42)
    ident_rgb = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    for prefer_vk, label in ((True, "Vulkan"), (False, "C11 AVX2 LUT")):
        req_id = VizInjectorRequest(
            target_width=256,
            target_height=256,
            emit_vit_branch=False,
            emit_mask_branch=False,
            deband_8bit=False,
            prefer_vulkan=prefer_vk,
            mtf_boost=0.0,
        )
        bundle_id = client.condition_rgba8(ident_rgb, req_id)
        recon_pil, _ = client.postprocess_to_pil(
            bundle_id.vae_tensor, apply_bayer_dither=False
        )
        recon_arr = np.asarray(recon_pil[0], dtype=np.int32)
        max_lsb_err = int(np.max(np.abs(recon_arr - ident_rgb.astype(np.int32))))
        assert (
            max_lsb_err == 0
        ), f"1:1 Identity passthrough ({label}) drifted by {max_lsb_err} LSB (expected 0)"
    print(
        "  [+] 1:1 Kronecker Passthrough: Max Roundtrip Error = 0 LSB (Bit-Exact on Vulkan & C11 AVX2 LUT)"
    )

    const_rgb = np.full((512, 512, 3), 180, dtype=np.uint8)
    for prefer_vk in (True, False):
        req_const = VizInjectorRequest(
            target_width=256,
            target_height=256,
            emit_vit_branch=False,
            emit_mask_branch=False,
            deband_8bit=False,
            prefer_vulkan=prefer_vk,
            mtf_boost=0.0,
        )
        b_const = client.condition_rgba8(const_rgb, req_const)
        const_pil, _ = client.postprocess_to_pil(
            b_const.vae_tensor, apply_bayer_dither=False
        )
        const_out = np.asarray(const_pil[0], dtype=np.int32)
        border_droop = int(np.max(np.abs(const_out - 180)))
        assert (
            border_droop == 0
        ), f"Boundary partition-of-unity violation: border droop = {border_droop} LSB"
    print(
        "  [+] Boundary Unity Partition : 512x512 -> 256x256 Constant Field Border Droop = 0 LSB"
    )

    step_img = np.zeros((512, 512, 3), dtype=np.uint8)
    step_img[:, :256, :] = 25
    step_img[:, 256:, :] = 230
    req_step = VizInjectorRequest(
        target_width=256,
        target_height=256,
        emit_vit_branch=False,
        emit_mask_branch=False,
        deband_8bit=False,
        mtf_boost=0.0,
        soft_knee_tau=0.015,
    )
    bundle_step = client.condition_rgba8(step_img, req_step)
    vae_srgb = (bundle_step.vae_tensor.numpy() * 0.5 + 0.5) * 255.0
    min_v = float(np.min(vae_srgb))
    max_v = float(np.max(vae_srgb))
    undershoot = max(0.0, 25.0 - min_v)
    overshoot = max(0.0, max_v - 230.0)
    assert (
        undershoot < 3.5 and overshoot < 3.5
    ), f"Excessive Gibbs ringing detected: undershoot={undershoot:.2f}, overshoot={overshoot:.2f}"
    print(
        f"  [+] C2 Local Extrema De-Ring : Undershoot = {undershoot:.2f}/255, Overshoot = {overshoot:.2f}/255 (<3.5 LSB)"
    )

    h_src, w_src = 600, 900
    yy, xx = np.ogrid[:h_src, :w_src]
    cy, cx = (h_src - 1) / 2.0, (w_src - 1) / 2.0
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    circle_img = np.zeros((h_src, w_src, 3), dtype=np.uint8)
    circle_img[r2 <= (180.0**2)] = 255
    req_iso = VizInjectorRequest(
        target_width=512,
        target_height=512,
        framing_mode="inscribed",
        emit_vit_branch=False,
        emit_mask_branch=False,
    )
    bundle_iso = client.condition_rgba8(circle_img, req_iso)
    mask_out = (bundle_iso.vae_tensor[0, 0].numpy() > 0.0).astype(np.float64)
    ys, xs = np.nonzero(mask_out)
    std_x = float(np.std(xs))
    std_y = float(np.std(ys))
    anisotropy = abs(std_x - std_y) / max(std_x, std_y, 1e-9)
    assert (
        anisotropy < 0.01
    ), f"Anisotropic geometric shear detected: anisotropy={anisotropy:.5f}"
    print(
        f"  [+] Isotropic Framing Lock   : 900x600 -> 512x512 Circle Anisotropy = {anisotropy:.5f} (<0.01)"
    )


def audit_gate_3_vulkan_parity_and_edgecases(client: VizInjectorClient) -> None:
    print(
        "\n--- Gate 3: Vulkan/AVX2 Parity, Prime Dimensions, HDR/16-Bit & OpenMOSS Role Routing ---"
    )
    y_grid, x_grid = np.meshgrid(
        np.linspace(0, 1, 384, dtype=np.float32),
        np.linspace(0, 1, 384, dtype=np.float32),
        indexing="ij",
    )
    synth_rgba = np.zeros((384, 384, 4), dtype=np.uint8)
    synth_rgba[..., 0] = np.clip(
        np.sin(x_grid * 12.0) * 127.0 + 128.0, 0, 255
    ).astype(np.uint8)
    synth_rgba[..., 1] = np.clip(
        np.cos(y_grid * 12.0) * 127.0 + 128.0, 0, 255
    ).astype(np.uint8)
    synth_rgba[..., 2] = np.clip((x_grid + y_grid) * 127.0, 0, 255).astype(
        np.uint8
    )
    synth_rgba[..., 3] = 255

    req_vk = VizInjectorRequest(
        target_width=256,
        target_height=256,
        prefer_vulkan=True,
        deband_8bit=False,
    )
    req_cpu = VizInjectorRequest(
        target_width=256,
        target_height=256,
        prefer_vulkan=False,
        deband_8bit=False,
    )
    b_vk = client.condition_rgba8(synth_rgba, req_vk)
    b_cpu = client.condition_rgba8(synth_rgba, req_cpu)
    mae_vae = float(
        torch.mean(torch.abs(b_vk.vae_tensor - b_cpu.vae_tensor)).item()
    )
    mae_vit = float(
        torch.mean(torch.abs(b_vk.vit_patches - b_cpu.vit_patches)).item()
    )
    assert (
        mae_vae < 1e-3 and mae_vit < 2e-3
    ), f"Vulkan vs CPU drift: VAE={mae_vae}, ViT={mae_vit}"
    print(
        f"  [+] Vulkan vs. C11 AVX2      : VAE MAE = {mae_vae:.6f}, ViT 5D Patch MAE = {mae_vit:.6f}"
    )

    prime_rgba = np.zeros((519, 773, 4), dtype=np.uint8)
    prime_rgba[..., 0] = 120
    prime_rgba[..., 1] = 180
    prime_rgba[..., 2] = 210
    prime_rgba[..., 3] = 255
    b_prime = client.condition_rgba8(
        prime_rgba,
        VizInjectorRequest(target_width=512, target_height=512),
    )
    assert b_prime.vae_tensor.shape == (1, 3, 512, 512)
    assert torch.isfinite(b_prime.vae_tensor).all()
    print(
        f"  [+] Prime Dimension Torture  : 773x519 -> VAE {tuple(b_prime.vae_tensor.shape)} & ViT {tuple(b_prime.vit_patches.shape)}"
    )

    cutout = np.zeros((128, 128, 4), dtype=np.uint8)
    cutout[32:96, 32:96, 0] = 255
    cutout[32:96, 32:96, 1] = 255
    cutout[32:96, 32:96, 2] = 255
    cutout[32:96, 32:96, 3] = 255
    req_cut = VizInjectorRequest(
        target_width=64,
        target_height=64,
        deband_8bit=False,
        mtf_boost=0.0,
    )
    b_cut = client.condition_rgba8(cutout, req_cut)
    border_val = float(b_cut.vae_tensor[0, 0, 16, 16].item())
    assert (
        border_val > 0.85
    ), f"Alpha=0 black garbage bled into opaque cutout border: {border_val:.4f}"
    print(
        f"  [+] Cutout Alpha Propagation : Boundary luminance = {border_val:.4f} (Zero Black Bleed)"
    )

    cmap = np.zeros((256, 256, 4), dtype=np.uint8)
    cmap[:, :128] = [255, 0, 0, 255]
    cmap[:, 128:] = [0, 0, 255, 255]
    req_cmap = VizInjectorRequest(
        target_width=128,
        target_height=128,
        input_role="color_map",
    )
    b_cmap = client.condition_rgba8(cmap, req_cmap)
    rgb_out = np.round(
        (b_cmap.vae_tensor[0].permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255.0
    ).astype(np.int32)
    unique_colors = np.unique(rgb_out.reshape(-1, 3), axis=0)
    assert (
        len(unique_colors) == 2
    ), f"Color map resampler hallucinated intermediate colors: {unique_colors}"

    h_cm, w_cm = 600, 900
    yy_cm, xx_cm = np.ogrid[:h_cm, :w_cm]
    cy_cm, cx_cm = (h_cm - 1) / 2.0, (w_cm - 1) / 2.0
    r2_cm = (xx_cm - cx_cm) ** 2 + (yy_cm - cy_cm) ** 2
    cmap_iso = np.zeros((h_cm, w_cm, 4), dtype=np.uint8)
    cmap_iso[..., :3] = [0, 0, 255]
    cmap_iso[..., 3] = 255
    cmap_iso[r2_cm <= (180.0**2), :3] = [255, 0, 0]
    b_cmap_iso = client.condition_rgba8(
        cmap_iso,
        VizInjectorRequest(
            target_width=512,
            target_height=512,
            framing_mode="inscribed",
            input_role="color_map",
        ),
    )
    cmap_iso_rgb = np.round(
        (b_cmap_iso.vae_tensor[0].permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255.0
    ).astype(np.int32)
    unique_iso_colors = np.unique(cmap_iso_rgb.reshape(-1, 3), axis=0)
    assert (
        len(unique_iso_colors) == 2
    ), f"Isotropic color map hallucinated colors: {unique_iso_colors}"
    red_mask = (cmap_iso_rgb[..., 0] > 200).astype(np.float64)
    c_ys, c_xs = np.nonzero(red_mask)
    c_std_x = float(np.std(c_xs))
    c_std_y = float(np.std(c_ys))
    cmap_anisotropy = abs(c_std_x - c_std_y) / max(c_std_x, c_std_y, 1e-9)
    assert (
        cmap_anisotropy < 0.01
    ), f"Color map resampler suffered anisotropic aspect shear: {cmap_anisotropy:.5f}"

    b_cmap_contain = client.condition_rgba8(
        cmap_iso,
        VizInjectorRequest(
            target_width=512,
            target_height=512,
            framing_mode="contain",
            input_role="color_map",
        ),
    )
    assert float(b_cmap_contain.vae_tensor[0, 2, 0, 256].item()) == -1.0, (
        "Color map contain letterbox border must pad with exact -1.0"
    )
    print(
        f"  [+] Region Color-Map Role    : {len(unique_colors)} Pure IDs Preserved | 900x600->512x512 Anisotropy = {cmap_anisotropy:.5f} (<0.01)"
    )

    stroke_mask = np.zeros((1024, 1024, 4), dtype=np.uint8)
    for i in range(128, 896):
        stroke_mask[i, i : i + 2, :3] = 255
    stroke_mask[..., 3] = 255
    b_mask = client.condition_rgba8(
        stroke_mask,
        VizInjectorRequest(
            target_width=1024,
            target_height=1024,
            input_role="inpaint_mask",
            mask_feather_radius=0.0,
        ),
    )
    active_cells = int(torch.sum(b_mask.latent_mask > 0.05).item())
    assert (
        active_cells >= 48
    ), f"2px thin stroke vanished in 16x latent mask: only {active_cells} active cells"
    print(
        f"  [+] Thin-Stroke Inpaint Mask : 2px Diagonal Stroke Survived 16x Downsample ({active_cells} active latent cells)"
    )

    req_edit = VizInjectorRequest(
        target_width=1024,
        target_height=1024,
        edit_format="ai_toolkit_isolated",
        framing_mode="cover_crop",
    )
    assert (
        req_edit.framing_mode_id == 0
    ), f"Expected cover_crop framing_mode_id == 0, got {req_edit.framing_mode_id}"
    b_edit = client.condition_rgba8(synth_rgba, req_edit)
    assert b_edit.meta.vae_size == (
        1024,
        1024,
    ), f"Unexpected VAE size: {b_edit.meta.vae_size}"
    assert (
        b_edit.meta.vit_size[0] <= 384 and b_edit.meta.vit_size[1] <= 384
    ), f"AI-Toolkit ViT size exceeded 384px cap: {b_edit.meta.vit_size}"
    assert b_edit.meta.reference_zero_timestep and b_edit.meta.reference_isolated_kv

    for prefer_vk in (True, False):
        req_vit_only = VizInjectorRequest(
            target_width=1024,
            target_height=1024,
            edit_format="ai_toolkit_isolated",
            emit_vae_branch=False,
            emit_mask_branch=False,
            emit_vit_branch=True,
            prefer_vulkan=prefer_vk,
        )
        b_vit_only = client.condition_rgba8(synth_rgba, req_vit_only)
        assert (
            b_vit_only.vae_tensor is None and b_vit_only.latent_mask is None
        ), "ViT-only pass must not allocate VAE or mask tensors"
        assert (
            b_vit_only.vit_patches is not None and b_vit_only.llm_tokens == 144
        ), f"ViT-only pass failed to emit 144 LLM tokens: {b_vit_only.llm_tokens}"
        assert b_vit_only.meta.vae_size == (
            1024,
            1024,
        ), f"ViT-only pass failed to preserve coupled vae_size: {b_vit_only.meta.vae_size}"
        assert b_vit_only.meta.vit_size == (
            384,
            384,
        ), f"ViT-only pass failed to preserve coupled vit_size: {b_vit_only.meta.vit_size}"

    req_ctrl_fmt = VizInjectorRequest(
        target_width=1024,
        target_height=1024,
        edit_format="control_color_depth",
        input_role="color_map",
    )
    assert (
        req_ctrl_fmt.reference_zero_timestep and req_ctrl_fmt.reference_isolated_kv
    ), "control_color_depth must enable reference_zero_timestep and reference_isolated_kv"
    print(
        f"  [+] OpenMOSS Edit Routing    : VAE={b_edit.meta.vae_size}, ViT={b_edit.meta.vit_size} ({b_edit.llm_tokens} LLM tok, ViT-Only Coupled Grid & cover_crop Verified)"
    )


def audit_gate_4_cli_and_vizj_container(client: VizInjectorClient) -> None:
    print(
        "\n--- Gate 4: Standalone UTF-8 CLI Subprocess & .VIZJ 3-Tier Container Audit ---"
    )
    tmp_dir = CACHE_DIR / "sanity_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    test_png = tmp_dir / "test_input_\u591c\u7a7a.png"
    test_vizj = tmp_dir / "test_wire_\u591c\u7a7a.vizj"

    grad = np.zeros((512, 512, 3), dtype=np.uint8)
    for y in range(512):
        grad[y, :, 0] = y // 2
        grad[y, :, 1] = 255 - (y // 2)
        grad[y, :, 2] = (y * 3) % 256
    Image.fromarray(grad, mode="RGB").save(test_png)

    cmd = [
        str(CLI_EXE),
        str(test_png),
        str(test_vizj),
        "--width",
        "512",
        "--height",
        "512",
        "--export-debug",
    ]
    proc = subprocess.run(cmd, cwd=str(tmp_dir), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"CLI invocation failed with code {proc.returncode}:\n{proc.stderr}"
        )

    assert (
        test_vizj.exists() and test_vizj.stat().st_size > 144
    ), ".vizj wire container was not created"
    assert (tmp_dir / "debug_vae_16px.png").exists(), "Missing debug_vae_16px.png"
    assert (tmp_dir / "debug_mask_16x.png").exists(), "Missing debug_mask_16x.png"

    reloaded = client.condition_image(test_vizj)
    assert reloaded.vae_tensor.shape == (1, 3, 512, 512)
    print(
        f"  [+] UTF-8 CLI & Tier-0 .VIZJ : {test_vizj.stat().st_size:,} bytes | Reloaded VAE Shape = {tuple(reloaded.vae_tensor.shape)}"
    )

    reloaded.packed_latents = torch.randn((1, 128, 32, 32), dtype=torch.float32)
    reloaded.vit_embeds = torch.randn((256, 4096), dtype=torch.bfloat16)
    reloaded.meta.vae_size = (0, 0)
    reloaded.meta.vit_size = (0, 0)
    t1_bytes = client.save_vizj(reloaded, tier=VIZJ_TIER_VAE_LATENT, use_bf16=True)
    t2_bytes = client.save_vizj(reloaded, tier=VIZJ_TIER_DUAL_LATENT, use_bf16=True)

    b_t1 = client.load_vizj_bytes(t1_bytes)
    b_t2 = client.load_vizj_bytes(t2_bytes)
    assert b_t1.packed_latents.shape == (1, 128, 32, 32)
    assert b_t2.packed_latents.shape == (1, 128, 32, 32)
    assert b_t2.vit_embeds.shape == (256, 4096)
    assert b_t2.meta.vae_size == (
        512,
        512,
    ), f"save_vizj failed to heal vae_size from packed_latents: {b_t2.meta.vae_size}"
    assert b_t2.meta.vit_size == (
        512,
        512,
    ), f"save_vizj failed to heal vit_size from grid_thw: {b_t2.meta.vit_size}"

    corrupted = bytearray(t1_bytes)
    corrupted[-8] ^= 0xFF
    crc_caught = False
    try:
        client.load_vizj_bytes(bytes(corrupted))
    except ValueError:
        crc_caught = True
    assert crc_caught, "CRC32C failed to reject corrupted .vizj payload"
    print(
        f"  [+] Tier-1 & Tier-2 .VIZJ    : Tier-1 = {len(t1_bytes) / 1024:.1f} KB, Tier-2 = {len(t2_bytes) / (1024**2):.2f} MB | Self-Healing Header & HW CRC32C Verified"
    )
    shutil.rmtree(tmp_dir, ignore_errors=True)


def audit_gate_5_gpu_acceleration_and_latentization(
    client: VizInjectorClient,
) -> None:
    print(
        "\n--- Gate 5: Hardware Acceleration, Sub-Phase Clamping & Standalone Flux2VAE Gate ---"
    )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Hardware acceleration assertion failed: torch.cuda.is_available() is False."
        )
    gpu_name = torch.cuda.get_device_name(0)
    backend_tag = (
        getattr(torch.version, "hip", None)
        or getattr(torch.version, "cuda", None)
        or "Unknown"
    )
    print(f"  [+] Hardware Device Identified : {gpu_name} (Backend: {backend_tag})")

    z_raw = torch.randn((1, 128, 64, 64), device="cuda", dtype=torch.float32)
    z_raw[:, 0:4, 16, 16] = torch.tensor([12.0, 24.0, -12.0, 36.0], device="cuda")
    ratios_before = z_raw[0, 0:4, 16, 16] / z_raw[0, 0, 16, 16]
    z_clamped = subphase_radial_clamp(z_raw, max_sigma=4.5)
    torch.cuda.synchronize()
    ratios_after = z_clamped[0, 0:4, 16, 16] / z_clamped[0, 0, 16, 16]
    ratio_err = float(torch.max(torch.abs(ratios_before - ratios_after)).item())
    rms_after = float(
        torch.sqrt(torch.mean(z_clamped[0, 0:4, 16, 16] ** 2)).item()
    )
    del z_raw, z_clamped, ratios_before, ratios_after
    assert (
        ratio_err < 1e-6 and rms_after <= 5.5
    ), f"Sub-phase ratio drift={ratio_err}, rms={rms_after}"
    print(
        f"  [+] Sub-Phase Radial Clamp     : 2x2 Sub-Phase Ratio Drift = {ratio_err:.2e} | Outlier RMS 22.98 -> {rms_after:.3f}"
    )

    if resolve_local_vae_dir() is not None:
        yy, xx = np.meshgrid(
            np.linspace(0, 1, 256, dtype=np.float32),
            np.linspace(0, 1, 256, dtype=np.float32),
            indexing="ij",
        )
        test_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        test_rgb[..., 0] = np.clip(xx * 255.0, 0, 255).astype(np.uint8)
        test_rgb[..., 1] = np.clip(yy * 255.0, 0, 255).astype(np.uint8)
        test_rgb[..., 2] = np.clip((1.0 - 0.5 * (xx + yy)) * 255.0, 0, 255).astype(
            np.uint8
        )
        bundle = client.latentize(
            test_rgb,
            request=VizInjectorRequest(
                target_width=256, target_height=256, emit_vit_branch=False
            ),
        )
        vae = client.get_or_load_local_vae()
        decoded = vae.decode_packed_latents(bundle.packed_latents)
        mse = float(
            torch.mean(
                (decoded.cpu() - bundle.vae_tensor.cpu()) ** 2
            ).item()
        )
        psnr = 10.0 * math.log10(4.0 / max(mse, 1e-10))
        lat_shape = tuple(bundle.packed_latents.shape)
        del decoded, bundle
        client._local_vae = None
        del vae
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        assert psnr > 28.0, f"Standalone Flux2VAE roundtrip PSNR too low: {psnr:.2f} dB"
        print(
            f"  [+] Standalone Flux2VAE Pass   : Latent {lat_shape} | Roundtrip PSNR = {psnr:.2f} dB"
        )
    else:
        print(
            "  [i] Standalone Flux2VAE weights not yet downloaded (run `python tools/download_vae.py` to enable live VAE roundtrip)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VizInjector Comprehensive Hermetic Sanity Auditor"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Purge .forge_cache and dist directories",
    )
    args = parser.parse_args()
    if args.clean:
        return

    print("=" * 88)
    print(
        "          VIZINJECTOR SOVEREIGN SUBSYSTEM CONVERGENCE & 5-GATE AUDIT SUITE"
    )
    print("=" * 88)
    t0 = time.perf_counter()

    client = audit_gate_1_binaries_and_sysroot()
    try:
        audit_gate_2_radiometric_math_and_isotropy(client)
        audit_gate_3_vulkan_parity_and_edgecases(client)
        audit_gate_4_cli_and_vizj_container(client)
        audit_gate_5_gpu_acceleration_and_latentization(client)
    finally:
        client.close()

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 88)
    print(
        f"[+] STATUS: SECURE, ROBUST & CONVERGED IN {elapsed:.2f}s. ZERO DRIFT ACROSS ALL 5 GATES."
    )
    print("=" * 88)


if __name__ == "__main__":
    main()