#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

sys.modules["torchvision"] = None
sys.modules["torchvision.transforms"] = None
sys.modules["torchvision.io"] = None


def _resolve_cache_dir() -> Path:
    env_hf = os.environ.get("HF_HOME")
    if env_hf and env_hf.strip():
        return Path(env_hf).resolve()
    parent_cache = (ROOT_DIR.parent / ".hf_cache").resolve()
    if parent_cache.exists():
        return parent_cache
    return (ROOT_DIR / ".hf_cache").resolve()


os.environ.pop("TORCH_BLAS_PREFER_HIPBLASLT", None)
CACHE_DIR = _resolve_cache_dir()
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_DIR / "hub")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors

from engine import MOSSImageEngine
from models.tokenizer import compile_qwen_ranks_binary

try:
    from refinery import (
        DTYPE_LOOKUP,
        DTYPE_MAP,
        FP8_DTYPES,
        clear_refinery_cache,
        refineTensor,
    )

    HAS_REFINERY = True
except ImportError:
    HAS_REFINERY = False
    DTYPE_MAP: Dict[str, Optional[torch.dtype]] = {
        "native": None,
        "none": None,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "e4m3fn": getattr(torch, "float8_e4m3fn", None),
        "e5m2": getattr(torch, "float8_e5m2", None),
    }
    DTYPE_LOOKUP: Dict[str, Optional[torch.dtype]] = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "e4m3fn": getattr(torch, "float8_e4m3fn", None),
        "e5m2": getattr(torch, "float8_e5m2", None),
        "e8m0fnu": getattr(torch, "float8_e8m0fnu", None),
    }
    FP8_DTYPES: List[torch.dtype] = [
        v for k, v in DTYPE_LOOKUP.items() if "e" in k and v is not None
    ]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

MODEL_REGISTRY: Dict[str, str] = {
    "sft": "OpenMOSS-Team/MOSS-Image-SFT",
    "base": "OpenMOSS-Team/MOSS-Image-Base",
}
EDIT_LORA_REPO_ID = "chinoll/Moss-Image-Edit"

SHARED_SUBSYSTEMS: Tuple[str, ...] = (
    "text_encoder",
    "vae",
    "tokenizer",
    "scheduler",
)

PRECISION_CHOICES = ["native", "bf16", "fp16", "e4m3fn", "e5m2"]
QUANT_ENGINES = ["clean_rtn", "dto_ca"]

_ST_TO_TORCH_DTYPE: Dict[str, Optional[torch.dtype]] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": getattr(torch, "float8_e4m3fn", None),
    "F8_E5M2": getattr(torch, "float8_e5m2", None),
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}
_FLOAT_ST_DTYPES = {"F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2"}

IGNORE_DOWNLOAD_PATTERNS: List[str] = [
    "assets/*",
    "assets/**",
    "*.webp",
    "*.png",
    "*.svg",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.msgpack",
    "*.h5",
]

PROTECTED_NAMESPACES: Tuple[str, ...] = (
    "vae",
    "visual/patch_embed",
    "visual/pos_embed",
    "visual/merger",
)

PROTECTED_KEY_PATTERNS: Tuple[str, ...] = (
    "rope.bands",
    "norm",
    "q_norm",
    "k_norm",
    "bias",
    "embed_tokens",
    "video_embedder",
    "final_linear_video",
    "final_norm_video",
    "time_embedding",
    "shared_block_modulation",
    "video_head_modulation",
    "text_embedder.proj",
    "linear_attn.a_log",
    "linear_attn.dt_bias",
    "linear_attn.conv1d",
    "linear_attn.in_proj_a",
    "linear_attn.in_proj_b",
    "visual.patch_embed",
    "visual.pos_embed",
    "visual.merger",
    "bn.",
    "quant_conv",
    "post_quant_conv",
)

DEAD_KEY_PATTERNS: Tuple[str, ...] = (
    "rotary_emb.inv_freq",
    "loss",
    "optimizer",
    "total_ops",
    "total_params",
)


def is_dead_key(key: str) -> bool:
    key_lower = key.lower()
    return any(pattern in key_lower for pattern in DEAD_KEY_PATTERNS)


def is_quantizable_meta(
    key: str, file_path: Path, st_dtype: str, shape: Sequence[int]
) -> bool:
    if st_dtype not in _FLOAT_ST_DTYPES or len(shape) < 2:
        return False
    if math.prod(int(d) for d in shape) < 128:
        return False
    path_norm = str(file_path).replace("\\", "/").lower()
    if any(
        f"/{ns}" in path_norm or path_norm.startswith(ns) for ns in PROTECTED_NAMESPACES
    ):
        return False
    key_lower = key.lower()
    if any(p in key_lower for p in PROTECTED_KEY_PATTERNS):
        return False
    return True


def is_quantizable(key: str, file_path: Path, tensor: torch.Tensor) -> bool:
    if not tensor.is_floating_point() or tensor.numel() < 128 or tensor.dim() < 2:
        return False
    path_norm = str(file_path).replace("\\", "/").lower()
    if any(
        f"/{ns}" in path_norm or path_norm.startswith(ns) for ns in PROTECTED_NAMESPACES
    ):
        return False
    key_lower = key.lower()
    if any(p in key_lower for p in PROTECTED_KEY_PATTERNS):
        return False
    return True


def inspect_shard_header(
    sf_path: Path,
    subsystem_name: str,
    target_dtype: Optional[torch.dtype],
) -> Tuple[bool, int, int, bool]:
    needs_modification = False
    quantized_count = 0
    preserved_count = 0
    has_non_bf16_quantizable = False

    with safe_open(str(sf_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            if is_dead_key(key):
                needs_modification = True
                continue
            sl = f.get_slice(key)
            st_dtype = sl.get_dtype()
            shape = sl.get_shape()
            t_dtype = _ST_TO_TORCH_DTYPE.get(st_dtype)
            is_fp = st_dtype in _FLOAT_ST_DTYPES

            if key == "rope.bands" or subsystem_name == "vae":
                if is_fp and t_dtype != torch.float32:
                    needs_modification = True
                preserved_count += 1
                continue

            if is_quantizable_meta(key, sf_path, st_dtype, shape):
                if t_dtype != torch.bfloat16:
                    has_non_bf16_quantizable = True
                if target_dtype is not None:
                    if t_dtype != target_dtype:
                        needs_modification = True
                    quantized_count += 1
                else:
                    if t_dtype != torch.bfloat16:
                        needs_modification = True
                    preserved_count += 1
            else:
                preserved_count += 1

    return (
        needs_modification,
        quantized_count,
        preserved_count,
        has_non_bf16_quantizable,
    )


def quantize_clean_rtn(
    tensor: torch.Tensor, target_dtype: torch.dtype, compute_device: torch.device
) -> torch.Tensor:
    with torch.inference_mode(False), torch.no_grad():
        t = tensor.to(compute_device, non_blocking=True)
        if target_dtype in (torch.bfloat16, torch.float16, torch.float32):
            out = t.to(target_dtype)
        else:
            finfo = torch.finfo(target_dtype)
            out = torch.clamp(t.float(), float(finfo.min), float(finfo.max)).to(
                target_dtype
            )
        return out.cpu()


def purge_unreferenced_artifacts(snapshot_path: Path) -> None:
    assets_dir = snapshot_path / "assets"
    if assets_dir.exists() and assets_dir.is_dir():
        shutil.rmtree(assets_dir, ignore_errors=True)

    safetensor_stems = {p.stem for p in snapshot_path.rglob("*.safetensors")}
    for bin_file in list(snapshot_path.rglob("*.bin")) + list(
        snapshot_path.rglob("*.pt")
    ):
        if bin_file.name.endswith(".ranks.bin"):
            continue
        if (
            bin_file.stem in safetensor_stems
            or bin_file.stem.replace("pytorch_model", "model") in safetensor_stems
        ):
            try:
                bin_file.unlink()
            except OSError:
                pass


def condition_subsystem_shards(
    subsystem_dir: Path,
    target_dtype: Optional[torch.dtype],
    quant_engine: str,
    compute_device: torch.device,
) -> Tuple[int, int, int]:
    safetensor_files = sorted(subsystem_dir.glob("*.safetensors"))
    total_files = len(safetensor_files)
    if total_files == 0:
        return 0, 0, 0

    quantized_count = 0
    preserved_count = 0
    stripped_count = 0

    with torch.inference_mode(False), torch.no_grad():
        for idx, sf_path in enumerate(safetensor_files, start=1):
            needs_mod, hdr_q, hdr_p, _ = inspect_shard_header(
                sf_path, subsystem_dir.name, target_dtype
            )
            if not needs_mod:
                logging.info(
                    "  [%d/%d] Verified %s/%s (Zero-Copy Header Audit: already converged)",
                    idx,
                    total_files,
                    subsystem_dir.name,
                    sf_path.name,
                )
                quantized_count += hdr_q
                preserved_count += hdr_p
                continue

            logging.info(
                "  [%d/%d] Conditioning %s/%s",
                idx,
                total_files,
                subsystem_dir.name,
                sf_path.name,
            )
            state_dict = load_safetensors(str(sf_path), device="cpu")
            processed_dict: Dict[str, torch.Tensor] = {}
            shard_modified = False

            for key, tensor in state_dict.items():
                if is_dead_key(key):
                    stripped_count += 1
                    shard_modified = True
                    continue

                if key == "rope.bands" or subsystem_dir.name == "vae":
                    if tensor.is_floating_point() and tensor.dtype != torch.float32:
                        processed_dict[key] = tensor.to(torch.float32)
                        shard_modified = True
                    else:
                        processed_dict[key] = tensor
                    preserved_count += 1
                    continue

                if target_dtype is not None and is_quantizable(key, sf_path, tensor):
                    if tensor.dtype != target_dtype:
                        if quant_engine == "dto_ca" and HAS_REFINERY:
                            refined = refineTensor(
                                origin_tensor=tensor,
                                target_dtype=target_dtype,
                                fp8_dtypes=FP8_DTYPES,
                                dtype_map=DTYPE_LOOKUP,
                                original_on_disk_dtype=tensor.dtype,
                                target_device=compute_device,
                            )
                            processed_dict[key] = refined.cpu()
                        else:
                            processed_dict[key] = quantize_clean_rtn(
                                tensor, target_dtype, compute_device
                            )
                        shard_modified = True
                    else:
                        processed_dict[key] = tensor
                    quantized_count += 1
                else:
                    processed_dict[key] = tensor
                    preserved_count += 1

            if shard_modified:
                tmp_path = sf_path.with_suffix(".safetensors.tmp")
                save_safetensors(processed_dict, str(tmp_path))
                tmp_path.replace(sf_path)

            del state_dict, processed_dict
            gc.collect()

    if compute_device.type == "cuda":
        torch.cuda.synchronize()
    if HAS_REFINERY:
        clear_refinery_cache()
    return quantized_count, preserved_count, stripped_count


def link_or_copy_shared_subsystem(src_sub: Path, dst_sub: Path) -> int:
    linked_files = 0
    if not src_sub.is_dir():
        return 0
    dst_sub.mkdir(parents=True, exist_ok=True)
    for src_file in src_sub.rglob("*"):
        if not src_file.is_file():
            continue
        real_src = src_file.resolve()
        if not real_src.is_file():
            continue
        rel = src_file.relative_to(src_sub)
        dst_file = dst_sub / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        if dst_file.exists() or dst_file.is_symlink():
            try:
                if (
                    dst_file.exists()
                    and not dst_file.is_symlink()
                    and dst_file.stat().st_ino == real_src.stat().st_ino
                ):
                    linked_files += 1
                    continue
                dst_file.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            os.link(real_src, dst_file)
        except OSError:
            shutil.copy2(real_src, dst_file)
        linked_files += 1
    return linked_files


def resolve_cached_snapshot(repo_id: str) -> Optional[Path]:
    hub_dir = CACHE_DIR / "hub"
    repo_folder = hub_dir / ("models--" + repo_id.replace("/", "--")) / "snapshots"
    if not repo_folder.is_dir():
        return None
    snapshots = sorted(
        [p for p in repo_folder.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return snapshots[0] if snapshots else None


def restore_pristine_shards_if_needed(
    repo_id: str, precision: str, quant_engine: str
) -> None:
    snap = resolve_cached_snapshot(repo_id)
    if snap is None:
        return
    target_dtype = DTYPE_MAP.get(precision.lower(), None)
    manifest_file = snap / "conditioning_manifest.json"
    manifest_mismatch = False
    if manifest_file.is_file():
        try:
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            prev_prec = str(data.get("precision", "native")).lower()
            prev_eng = str(data.get("quant_engine", "clean_rtn")).lower()
            if prev_prec != precision.lower() or (
                precision.lower() not in ("native", "none", "bf16")
                and prev_eng != quant_engine.lower()
            ):
                manifest_mismatch = True
        except Exception:
            manifest_mismatch = True

    unlinked_count = 0
    for sub_name in ("transformer", "text_encoder", "vae"):
        sub_dir = snap / sub_name
        if not sub_dir.is_dir():
            continue
        for sf_path in sorted(sub_dir.glob("*.safetensors")):
            if sf_path.is_symlink() and not sf_path.exists():
                try:
                    sf_path.unlink(missing_ok=True)
                    unlinked_count += 1
                except OSError:
                    pass
                continue
            if sub_name == "vae":
                continue
            try:
                _, _, _, has_non_bf16 = inspect_shard_header(
                    sf_path, sub_name, target_dtype
                )
            except Exception:
                has_non_bf16 = True
            if has_non_bf16 and (manifest_mismatch or target_dtype is None):
                try:
                    sf_path.unlink(missing_ok=True)
                    unlinked_count += 1
                except OSError:
                    pass

    if unlinked_count > 0:
        manifest_file.unlink(missing_ok=True)
        logging.info(
            "Unlinked %d stale or quantized shard(s) in %s to restore pristine BF16 blobs.",
            unlinked_count,
            repo_id,
        )


def find_donor_snapshot(
    current_repo_id: str, precision: str, quant_engine: str
) -> Optional[Path]:
    target_dtype = DTYPE_MAP.get(precision.lower(), None)
    for _, candidate_repo in MODEL_REGISTRY.items():
        if candidate_repo == current_repo_id:
            continue
        snap = resolve_cached_snapshot(candidate_repo)
        if snap is None:
            continue
        manifest_file = snap / "conditioning_manifest.json"
        if not manifest_file.is_file():
            continue
        try:
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            prev_prec = str(data.get("precision", "")).lower()
            prev_eng = str(data.get("quant_engine", "")).lower()
            prec_ok = prev_prec == precision.lower() or (
                precision.lower() in ("native", "bf16")
                and prev_prec in ("native", "bf16")
            )
            eng_ok = (
                precision.lower() in ("native", "bf16")
                or prev_eng == quant_engine.lower()
            )
            if (
                prec_ok
                and eng_ok
                and (snap / "text_encoder").is_dir()
                and (snap / "vae").is_dir()
            ):
                te_shards = sorted((snap / "text_encoder").glob("*.safetensors"))
                if len(te_shards) == 4 and all(s.resolve().is_file() for s in te_shards):
                    needs_mod, _, _, _ = inspect_shard_header(
                        te_shards[0].resolve(), "text_encoder", target_dtype
                    )
                    if not needs_mod:
                        return snap
        except Exception:
            continue
    return None


def download_and_condition_repo(
    repo_id: str,
    precision: str = "native",
    quant_engine: str = "clean_rtn",
    device_str: str = "auto",
    max_workers: int = 8,
    donor_snapshot: Optional[Path] = None,
    skip_warmup: bool = False,
) -> Path:
    t0 = time.perf_counter()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    restore_pristine_shards_if_needed(repo_id, precision, quant_engine)

    if donor_snapshot is None:
        donor_snapshot = find_donor_snapshot(repo_id, precision, quant_engine)

    ignore_list = list(IGNORE_DOWNLOAD_PATTERNS)
    if donor_snapshot is not None:
        logging.info(
            "Detected conditioned shared donor snapshot (%s); bypassing redundant 15.86 GB text_encoder/vae download.",
            donor_snapshot.parent.parent.name,
        )
        ignore_list.extend(
            [
                "text_encoder/*.safetensors",
                "vae/*.safetensors",
                "tokenizer/tokenizer.json",
            ]
        )

    logging.info("Acquiring repository snapshot: %s", repo_id)
    snapshot_path_str = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        max_workers=max_workers,
        ignore_patterns=ignore_list,
    )
    snapshot_path = Path(snapshot_path_str).resolve()
    purge_unreferenced_artifacts(snapshot_path)

    if donor_snapshot is not None and donor_snapshot != snapshot_path:
        for sub_name in SHARED_SUBSYSTEMS:
            cnt = link_or_copy_shared_subsystem(
                donor_snapshot / sub_name, snapshot_path / sub_name
            )
            logging.info(
                "Shared subsystem '%s' linked from donor (%d files).", sub_name, cnt
            )

    logging.info("Auditing Sovereign VizToken Qwen3.5 SWISS5 binary ranks cache...")
    ranks_bin = compile_qwen_ranks_binary(snapshot_path)
    viztoken_pkg_bin = ROOT_DIR / "viztoken" / "qwen35.ranks.bin"
    viztoken_pkg_bin.parent.mkdir(parents=True, exist_ok=True)
    if not viztoken_pkg_bin.is_file():
        try:
            shutil.copy2(ranks_bin, viztoken_pkg_bin)
        except OSError:
            pass
    logging.info(
        "Binary ranks cache confirmed: %s (%d bytes)",
        ranks_bin.name,
        ranks_bin.stat().st_size,
    )

    target_dtype = DTYPE_MAP.get(precision.lower(), None)
    if device_str == "auto":
        compute_device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    else:
        compute_device = torch.device(device_str)

    total_q, total_p, total_s = 0, 0, 0
    subsystems_to_condition = (
        ["transformer"]
        if donor_snapshot is not None
        else ["transformer", "text_encoder", "vae"]
    )
    for sub_name in subsystems_to_condition:
        sub_dir = snapshot_path / sub_name
        if sub_dir.is_dir():
            q, p, s = condition_subsystem_shards(
                sub_dir, target_dtype, quant_engine, compute_device
            )
            total_q += q
            total_p += p
            total_s += s

    elapsed = time.perf_counter() - t0
    manifest = {
        "repo_id": repo_id,
        "precision": precision,
        "quant_engine": quant_engine,
        "quantized_tensors": total_q,
        "preserved_tensors": total_p,
        "stripped_keys": total_s,
        "shared_donor": str(donor_snapshot) if donor_snapshot else None,
        "tokenizer_artifact": str(ranks_bin),
        "conditioning_seconds": round(elapsed, 3),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (snapshot_path / "conditioning_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logging.info(
        "Repository %s conditioned in %.2fs (%d quantized, %d preserved, %d dead keys removed).",
        repo_id,
        elapsed,
        total_q,
        total_p,
        total_s,
    )

    if not skip_warmup and compute_device.type == "cuda":
        logging.info("=" * 80)
        logging.info("INITIALIZING SOVEREIGN MOSS ENGINE WARMUP PASS")
        logging.info("=" * 80)
        engine = MOSSImageEngine(
            repo_id=str(snapshot_path),
            device=compute_device,
            dtype=torch.bfloat16,
        )
        engine.warmup()
        engine.prepare_for_exit()

    return snapshot_path


def download_edit_lora(max_workers: int = 4) -> Path:
    logging.info("Acquiring optional Edit LoRA: %s", EDIT_LORA_REPO_ID)
    lora_snap = snapshot_download(
        repo_id=EDIT_LORA_REPO_ID,
        repo_type="model",
        max_workers=max_workers,
        ignore_patterns=["*.png", "*.webp", "*.jpg"],
    )
    logging.info("Edit LoRA staged at: %s", lora_snap)
    return Path(lora_snap)


def prompt_acquisition_menu() -> Tuple[str, str, str, str, bool]:
    print("\n" + "=" * 84)
    print("             OPENMOSS (MOSS-IMAGE-SFT & BASE) ACQUISITION & CONDITIONER")
    print("=" * 84)
    print(" [1] MOSS-Image-SFT  - Supervised fine-tuned model (Sharp layout & typography)")
    print(" [2] MOSS-Image-Base - Base foundation model (Natural composition diversity)")
    print(" [3] Both Models     - Acquire SFT + Base with 15.86 GB shared-shard deduplication")
    print("-" * 84)
    m_choice = input("Select model target [1-3] (Default: 1): ").strip()
    model_map = {
        "1": "sft",
        "sft": "sft",
        "2": "base",
        "base": "base",
        "3": "both",
        "both": "both",
    }
    model_target = model_map.get(m_choice.lower(), "sft")

    print("\n" + "-" * 84)
    print(" [1] Native (None)   - Pristine BF16 DiT & Qwen3.5 + FP32 FLUX.2 VAE")
    print(" [2] BF16 (Contender)- Gray-Code Z-Curve + Bayer Dither BF16 pass")
    print(" [3] FP16            - Error-diffused + Perona-Malik seam-smoothed Float16")
    print(" [4] FP8 E4M3FN      - 3-bit Mantissa FP8 (VAE, Vision Stem, Norms & SSM preserved)")
    print(" [5] FP8 E5M2        - 2-bit Mantissa FP8 (VAE, Vision Stem, Norms & SSM preserved)")
    print("-" * 84)
    p_choice = input("Select precision [1-5] (Default: 1): ").strip()
    prec_map = {
        "1": "native",
        "native": "native",
        "2": "bf16",
        "bf16": "bf16",
        "3": "fp16",
        "fp16": "fp16",
        "4": "e4m3fn",
        "e4m3fn": "e4m3fn",
        "5": "e5m2",
        "e5m2": "e5m2",
    }
    precision = prec_map.get(p_choice.lower(), "native")

    quant_engine = "clean_rtn"
    device_str = "auto"
    if precision != "native":
        if HAS_REFINERY:
            print("\n" + "-" * 84)
            print(" [1] Clean RTN       - Round-to-nearest mathematical baseline")
            print(" [2] Gorilla DTO-CA  - Z-Curve Contender / Perona-Malik Refinery pass")
            print("-" * 84)
            e_choice = input("Select quantization engine [1-2] (Default: 2): ").strip()
            quant_engine = "clean_rtn" if e_choice == "1" else "dto_ca"

        print("\n" + "-" * 84)
        print(" [1] GPU (CUDA/ROCm) - Hardware-accelerated tensor conditioning")
        print(" [2] CPU (System RAM)- Zero VRAM allocation execution")
        print("-" * 84)
        d_choice = input("Select compute device [1-2] (Default: 1): ").strip()
        device_str = "cpu" if d_choice == "2" else "auto"

    print("\n" + "-" * 84)
    lora_in = (
        input("Fetch optional chinoll/Moss-Image-Edit LoRA (297 MB)? [y/N]: ")
        .strip()
        .lower()
    )
    include_lora = lora_in in ("y", "yes", "1")
    return model_target, precision, quant_engine, device_str, include_lora


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire and condition OpenMOSS (MOSS-Image-SFT & MOSS-Image-Base) weights."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["sft", "base", "both"],
        default=None,
    )
    parser.add_argument("--repo_id", type=str, default=None)
    parser.add_argument(
        "--precision",
        "--quant_target",
        dest="precision",
        type=str,
        choices=PRECISION_CHOICES,
        default=None,
    )
    parser.add_argument(
        "--engine",
        dest="quant_engine",
        type=str,
        choices=QUANT_ENGINES,
        default="dto_ca" if HAS_REFINERY else "clean_rtn",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default=None,
    )
    parser.add_argument("--include_edit_lora", action="store_true")
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--skip_warmup", action="store_true")
    args = parser.parse_args()

    model_target = args.model
    precision = args.precision
    quant_engine = args.quant_engine
    device_str = args.device
    include_lora = args.include_edit_lora

    if precision is None and model_target is None and args.repo_id is None:
        if sys.stdin.isatty():
            model_target, precision, quant_engine, dev_in, lora_in = (
                prompt_acquisition_menu()
            )
            if device_str is None:
                device_str = dev_in
            include_lora = include_lora or lora_in
        else:
            model_target = "sft"
            precision = "native"

    if model_target is None:
        model_target = "sft"
    if precision is None:
        precision = "native"
    if device_str is None:
        device_str = "auto"

    if precision in ("e4m3fn", "e5m2") and DTYPE_MAP[precision] is None:
        logging.error(
            "Selected precision '%s' is not supported in this PyTorch runtime.",
            precision,
        )
        sys.exit(1)

    try:
        if args.repo_id:
            download_and_condition_repo(
                repo_id=args.repo_id,
                precision=precision,
                quant_engine=quant_engine,
                device_str=device_str,
                max_workers=args.max_workers,
                skip_warmup=args.skip_warmup,
            )
        elif model_target == "both":
            sft_snap = download_and_condition_repo(
                repo_id=MODEL_REGISTRY["sft"],
                precision=precision,
                quant_engine=quant_engine,
                device_str=device_str,
                max_workers=args.max_workers,
                skip_warmup=True,
            )
            download_and_condition_repo(
                repo_id=MODEL_REGISTRY["base"],
                precision=precision,
                quant_engine=quant_engine,
                device_str=device_str,
                max_workers=args.max_workers,
                donor_snapshot=sft_snap,
                skip_warmup=args.skip_warmup,
            )
        else:
            download_and_condition_repo(
                repo_id=MODEL_REGISTRY[model_target],
                precision=precision,
                quant_engine=quant_engine,
                device_str=device_str,
                max_workers=args.max_workers,
                skip_warmup=args.skip_warmup,
            )

        if include_lora:
            download_edit_lora(max_workers=min(4, args.max_workers))
    except Exception as e:
        logging.error("Acquisition and conditioning failure: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()