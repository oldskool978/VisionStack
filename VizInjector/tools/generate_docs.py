#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
IS_WIN = sys.platform == "win32"
HERMETIC_PYTHON = ROOT_DIR / "bin" / "python" / ("python.exe" if IS_WIN else "bin/python")

if (
    __name__ == "__main__"
    and HERMETIC_PYTHON.exists()
    and Path(sys.executable).resolve() != HERMETIC_PYTHON.resolve()
):
    res = subprocess.run(
        [str(HERMETIC_PYTHON.resolve()), "-I", str(Path(__file__).resolve())]
        + sys.argv[1:],
        check=False,
    )
    sys.exit(res.returncode)


def clean_documentation() -> None:
    docs_dir = ROOT_DIR / "docs"
    if docs_dir.exists():
        shutil.rmtree(docs_dir, ignore_errors=True)
    print(f"[SUCCESS] Purged generated documentation at {docs_dir}")


def generate_documentation() -> None:
    docs_dir = ROOT_DIR / "docs"
    if docs_dir.exists():
        shutil.rmtree(docs_dir, ignore_errors=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    cb = "```"

    doc_00 = f"""# VizInjector: Sovereign Visual Conditioning & Latentization Engine

## 1. Zero-Dependency Hermetic Build Site
VizInjector autonomously provisions its own hermetic Python runtime, LLVM/Clang compiler, LLD linker, Ninja build system, CMake generator, Shaderc (`glslc`) SPIR-V compiler, Khronos Vulkan SDK (`Vulkan-Headers` + `Vulkan-Loader`), static math libraries (`AMD AOCL-BLIS`, `FFTW3`), and native C image codecs (`libwebp`, `libspng`, `miniz`, `stb_image`, `ffmpeg`) inside `bin/` and `bin/sysroot/`.

## 2. Build & Verification Sequence
{cb}powershell
python tools/hydrate.py
python tools/build_deps.py
python tools/forge_dag.py
python tools/sanitycheck.py
python tools/benchmark.py
{cb}

## 3. Self-Contained Distribution Bundle (`dist/bin/`)
`tools/forge_dag.py` stages `vizinjector.dll`, `vizinjector-cli.exe`, `schema.py`, `bindings.py`, `modeling_vae.py`, and `__init__.py` directly into `dist/bin/`.
"""

    doc_01 = """# Radiometric Linear-Light Resampling & Anomaly Prevention

## 1. Exact IEC 61966-2-1 Linearization & Boundary Alpha Propagation
* Standard image pipelines resize in nonlinear `uint8` sRGB space, violating photon additivity and darkening high-contrast typography strokes.
* VizInjector propagates opaque boundary chromaticity into transparent (`alpha <= 1/255`) pixels prior to premultiplication, eliminating dark halo fringes around cutouts.
* 8-bit inputs receive gradient-gated triangular PDF (TPDF) sub-LSB dither in flat regions before converting to linear `FP32`.

## 2. Scale-Routed Polyphase Filtering & C2 Extrema De-Ringing
* **Identity Passthrough (`delta == 1.0`)**: Collapses to an exact Kronecker delta with zero convolution drift.
* **Downsampling (`delta > 1.0`)**: Evaluates an 8-tap (`a = 4`, `beta = 8.5`) Kaiser-windowed Sinc filter with anti-alias cutoff `0.92 * pi / delta`.
* **Upsampling (`delta < 1.0`)**: Routes to a 4-tap Mitchell-Netravali spline.
* **C2 Local Extrema Bounding**: Bounds filter excursions against the local `2x2` linear source neighborhood via a rational Pade `tanh` soft-knee (`tau = 0.015`), eliminating Gibbs ringing and near-black sRGB slope amplification.
"""

    doc_02 = """# OpenMOSS Conditioning & Control Modes

## 1. Coupled Dual-Grid Geometry
Solves a single aspect-locked coordinate frame for both `Flux2AutoencoderKL` (`16px` stride) and `Qwen3_5VisionModel` (`32px` 2x2 merged patch stride) so 4D-RoPE reference latents and ViT semantic tokens share identical optical centers `((H - 1)/2, (W - 1)/2)`.

## 2. Edit & Control Routing (`VizInjectorRequest`)
* **`moss_image_edit`**: Enforces `never_crop` framing, emits `vae_16px`, and flags `reference_zero_timestep = True` and `reference_isolated_kv = True`.
* **`ai_toolkit_isolated` / `ai_toolkit_full`**: Emits a `<= 384x384` (`32px`-aligned) 5D ViT patch tensor `[N, 3, 2, 16, 16]` alongside full-resolution `16px` VAE latents.
* **`color_map`**: Routes to an exact majority-vote discrete ID resampler in sRGB space so region boundaries never blend into false intermediate colors.
* **`depth_map`**: Enforces monotonic single-channel luminance replicated across planar RGB in `[-1.0, 1.0]`.
* **`inpaint_mask`**: Computes `16x16` area-integral downsampling with stroke preservation and `C2` raised-cosine latent feathering.
"""

    doc_03 = """# .VIZJ 3-Tier Binary Container Specification

All `.vizj` containers begin with a 144-byte packed header (`VIZJ_CONTAINER01`) protected by a hardware SSE4.2/AVX2 Castagnoli CRC32C checksum over the 64-byte-aligned payload slab:
* **Tier 0 (`VIZJ_TIER_WIRE_WEBP`)**: Phase-locked WebP payload + `[1, 1, H/16, W/16]` feathered latent mask (~0.25 - 0.95 MB at `1024x1024`).
* **Tier 1 (`VIZJ_TIER_VAE_LATENT`)**: Sub-phase radial-clamped `Flux2` `[1, 128, H/16, W/16]` latent (`1.01 MB` in `BF16` or `2.01 MB` in `FP32` at `1024x1024`).
* **Tier 2 (`VIZJ_TIER_DUAL_LATENT`)**: Tier 1 `Flux2` latent + post-merger `Qwen3.5-VL` `[N_llm, 4096]` `BF16` embeddings (`5.51 MB` at `1024x1024`), bypassing both `Flux2Encoder` and `Qwen3_5VisionModel` at runtime.
"""

    files = {
        "00_ONBOARDING.md": doc_00,
        "01_RADIOMETRIC_MATH.md": doc_01,
        "02_OPENMOSS_MODES.md": doc_02,
        "03_VIZJ_CONTAINER_SPEC.md": doc_03,
    }
    for name, content in files.items():
        (docs_dir / name).write_text(content, encoding="utf-8")
    print(f"[SUCCESS] VizInjector documentation forged at {docs_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VizInjector Documentation Generator"
    )
    parser.add_argument(
        "--clean", action="store_true", help="Purge generated docs directory"
    )
    args = parser.parse_args()
    if args.clean:
        clean_documentation()
        sys.exit(0)
    generate_documentation()