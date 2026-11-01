#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT_DIR = Path(__file__).resolve().parent.parent
BIN_DIR = ROOT_DIR / "bin"
IS_WIN = sys.platform == "win32"
HERMETIC_PYTHON_EXE = BIN_DIR / "python" / ("python.exe" if IS_WIN else "bin/python")

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


def _resolve_cache_dir() -> Path:
    env_hf = os.environ.get("HF_HOME")
    if env_hf and env_hf.strip():
        return Path(env_hf).resolve()
    parent_cache = (ROOT_DIR.parent / ".hf_cache").resolve()
    if parent_cache.exists():
        return parent_cache
    return (ROOT_DIR / ".hf_cache").resolve()


CACHE_DIR = _resolve_cache_dir()
os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(CACHE_DIR / "hub"))

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

for _noisy_logger in ("httpx", "httpx2", "httpcore", "httpcore2", "urllib3"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
import torch

from python.modeling_vae import Flux2AutoencoderKL

DEFAULT_REPO_ID = "OpenMOSS-Team/MOSS-Image-SFT"
_FLOAT_ST_DTYPES = {"F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2"}


def resolve_local_vae_dir(repo_id: str = DEFAULT_REPO_ID) -> Path | None:
    hub_dir = Path(os.environ.get("HUGGINGFACE_HUB_CACHE") or (CACHE_DIR / "hub")).resolve()
    repo_folder = hub_dir / ("models--" + repo_id.replace("/", "--")) / "snapshots"
    if not repo_folder.is_dir():
        return None
    snapshots = sorted(
        [p for p in repo_folder.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for snap in snapshots:
        vae_dir = snap / "vae"
        if vae_dir.is_dir():
            sf_files = [sf for sf in vae_dir.glob("*.safetensors") if sf.exists()]
            if sf_files:
                return vae_dir.resolve()
    return None


def _inspect_vae_shard(sf_path: Path) -> tuple[bool, int]:
    needs_conversion = False
    tensor_count = 0
    resolved_sf = sf_path.resolve(strict=True)
    with safe_open(str(resolved_sf), framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor_count += 1
            st_dtype = f.get_slice(key).get_dtype()
            if st_dtype in _FLOAT_ST_DTYPES and st_dtype != "F32":
                needs_conversion = True
    return needs_conversion, tensor_count


def download_and_condition_vae(
    repo_id: str = DEFAULT_REPO_ID,
    skip_warmup: bool = False,
) -> Path:
    t0 = time.perf_counter()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[*] Acquiring standalone Flux2AutoencoderKL (vae/*) from {repo_id}...")
    snapshot_str = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=["vae/*.json", "vae/*.safetensors"],
        max_workers=8,
    )
    snapshot_path = Path(snapshot_str).resolve()
    vae_dir = snapshot_path / "vae"
    sf_files = sorted(vae_dir.glob("*.safetensors"))
    if not sf_files:
        raise FileNotFoundError(f"No .safetensors found in {vae_dir}")

    tensor_count = 0
    for sf_path in sf_files:
        needs_conv, count = _inspect_vae_shard(sf_path)
        tensor_count += count
        if not needs_conv:
            continue
        resolved_sf = sf_path.resolve(strict=True)
        raw = load_safetensors(str(resolved_sf), device="cpu")
        conditioned = {
            k: (v.to(torch.float32) if v.is_floating_point() else v)
            for k, v in raw.items()
        }
        tmp_path = sf_path.with_suffix(".safetensors.tmp")
        save_safetensors(conditioned, str(tmp_path))
        if sf_path.is_symlink():
            sf_path.unlink(missing_ok=True)
        tmp_path.replace(sf_path)
        del raw, conditioned
        gc.collect()

    elapsed = time.perf_counter() - t0
    manifest = {
        "repo_id": repo_id,
        "subsystem": "vae",
        "precision": "fp32",
        "tensors_conditioned": tensor_count,
        "conditioning_seconds": round(elapsed, 3),
    }
    (vae_dir / "vae_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"[+] Standalone Flux2AutoencoderKL staged at {vae_dir} ({tensor_count} FP32 tensors in {elapsed:.2f}s)"
    )

    if not skip_warmup and torch.cuda.is_available():
        print("[*] Executing GPU Flux2AutoencoderKL encode/decode warmup pass...")
        t_w = time.perf_counter()
        vae = Flux2AutoencoderKL.from_pretrained_local(
            vae_dir, device="cuda", dtype=torch.float32
        )
        dummy_img = torch.zeros((1, 3, 256, 256), device="cuda", dtype=torch.float32)
        lat = vae.encode_to_packed_latents(dummy_img)
        rec = vae.decode_packed_latents(lat)
        torch.cuda.synchronize()
        assert torch.isfinite(lat).all() and torch.isfinite(rec).all()
        lat_shape = tuple(lat.shape)
        del vae, dummy_img, lat, rec
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print(
            f"[+] Warmup complete in {time.perf_counter() - t_w:.2f}s | Latent shape: {lat_shape}"
        )

    return vae_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire and condition standalone Flux2AutoencoderKL weights for VizInjector."
    )
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--skip_warmup", action="store_true")
    args = parser.parse_args()
    download_and_condition_vae(repo_id=args.repo_id, skip_warmup=args.skip_warmup)


if __name__ == "__main__":
    main()