# ShadowBoxer/core/loader.py
from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, Optional, Union

from safetensors import safe_open
import torch
from torch.overrides import TorchFunctionMode

from .env import CACHE_DIR, ROOT_DIR, _canonicalize_device, _same_device


class _MetaWeightInitMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs) if kwargs is not None else {}
        if (
            func in (torch.empty, torch.zeros, torch.ones)
            and kwargs.get("device") is None
        ):
            kwargs["device"] = torch.device("meta")
        return func(*args, **kwargs)


def resolve_model_path(repo_or_path: Union[str, Path]) -> Path:
    direct_path = Path(repo_or_path)
    if direct_path.exists():
        return direct_path.resolve()

    repo_str = str(repo_or_path).replace("\\", "/")
    hub_candidates = [
        CACHE_DIR / "hub",
        CACHE_DIR,
        ROOT_DIR.parent / ".hf_cache" / "hub",
        ROOT_DIR / ".hf_cache" / "hub",
    ]
    repo_folder_name = "models--" + repo_str.replace("/", "--")

    seen_dirs = set()
    for hub_dir in hub_candidates:
        resolved_hub = hub_dir.resolve()
        if resolved_hub in seen_dirs or not resolved_hub.exists():
            continue
        seen_dirs.add(resolved_hub)

        cand = resolved_hub / repo_folder_name / "snapshots"
        if cand.exists():
            snapshots = sorted(
                [p for p in cand.iterdir() if p.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if snapshots:
                return snapshots[0].resolve()

    raise FileNotFoundError(
        f"Cached snapshot not found for '{repo_or_path}' in {CACHE_DIR / 'hub'}. Run download_model.py first."
    )


def _resolve_target_tensor_dtype(
    clean_key: str,
    src_tensor: torch.Tensor,
    dtype: Optional[torch.dtype],
) -> torch.dtype:
    if not src_tensor.is_floating_point():
        return src_tensor.dtype
    if (
        clean_key.endswith("bands")
        or clean_key.endswith("inv_freq")
        or clean_key.startswith("bn.")
    ):
        return torch.float32
    if (
        clean_key.endswith("A_log")
        or clean_key.endswith("dt_bias")
        or dtype is None
    ):
        return src_tensor.dtype
    if src_tensor.dtype == torch.float32 and dtype != torch.float32:
        return dtype
    return src_tensor.dtype


def stream_safetensors_inplace(
    model_dir: Path,
    target_module: torch.nn.Module,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> bool:
    with torch.inference_mode(False), torch.no_grad():
        target_dev = _canonicalize_device(device)
        sf_files = sorted(set(model_dir.glob("*.safetensors")))
        if not sf_files:
            raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

        param_map: Dict[str, torch.nn.Parameter] = dict(
            target_module.named_parameters()
        )
        buf_map: Dict[str, torch.Tensor] = dict(target_module.named_buffers())

        for p in param_map.values():
            if p.is_meta or not _same_device(p.device, target_dev):
                return False

        for sf_path in sf_files:
            resolved_sf = sf_path.resolve(strict=True)
            with safe_open(str(resolved_sf), framework="pt", device="cpu") as f:
                for k in f.keys():
                    clean_key = k[6:] if k.startswith("model.") else k
                    target_t: Optional[torch.Tensor] = param_map.get(clean_key)
                    if target_t is None:
                        target_t = buf_map.get(clean_key)
                    if target_t is None:
                        continue
                    v = f.get_tensor(k)
                    t_dtype = _resolve_target_tensor_dtype(clean_key, v, dtype)
                    if (
                        not target_t.is_meta
                        and _same_device(target_t.device, target_dev)
                        and target_t.shape == v.shape
                        and target_t.dtype == t_dtype
                        and not target_t.is_inference()
                    ):
                        if v.dtype != t_dtype:
                            v_cast = v.to(dtype=t_dtype)
                            target_t.data.copy_(v_cast, non_blocking=True)
                            del v_cast
                        else:
                            target_t.data.copy_(v, non_blocking=True)
                    else:
                        target_t.data = v.to(device=target_dev, dtype=t_dtype)
                    del v

        clear_fn = getattr(target_module, "clear_caches", None)
        if callable(clear_fn):
            clear_fn()
        if target_dev.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(target_dev)
        return True


def load_sharded_safetensors_into(
    model_dir: Path,
    target_module: torch.nn.Module,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> None:
    with torch.inference_mode(False), torch.no_grad():
        target_dev = _canonicalize_device(device)
        first_param = next(target_module.parameters(), None)
        if (
            first_param is not None
            and not first_param.is_meta
            and _same_device(first_param.device, target_dev)
        ):
            if stream_safetensors_inplace(
                model_dir, target_module, device=target_dev, dtype=dtype
            ):
                return

        sf_files = sorted(set(model_dir.glob("*.safetensors")))
        if not sf_files:
            raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

        state_dict: Dict[str, torch.Tensor] = {}
        for sf_path in sf_files:
            resolved_sf = sf_path.resolve(strict=True)
            with safe_open(str(resolved_sf), framework="pt", device="cpu") as f:
                for k in f.keys():
                    clean_key = k[6:] if k.startswith("model.") else k
                    v = f.get_tensor(k)
                    t_dtype = _resolve_target_tensor_dtype(clean_key, v, dtype)
                    if v.dtype != t_dtype:
                        state_dict[clean_key] = v.to(device=target_dev, dtype=t_dtype)
                    else:
                        state_dict[clean_key] = v.to(device=target_dev)
                    del v

        target_module.load_state_dict(state_dict, strict=True, assign=True)
        del state_dict
        gc.collect()

        for name, param in target_module.named_parameters():
            if not _same_device(param.device, target_dev):
                t_dtype = _resolve_target_tensor_dtype(name, param, dtype)
                param.data = param.data.to(device=target_dev, dtype=t_dtype)

        for name, buf in target_module.named_buffers():
            if buf.is_floating_point():
                if (
                    name.endswith("bands")
                    or name.endswith("inv_freq")
                    or dtype is None
                ):
                    if (
                        not _same_device(buf.device, target_dev)
                        or buf.dtype != torch.float32
                    ):
                        buf.data = buf.data.to(device=target_dev, dtype=torch.float32)
                elif not _same_device(buf.device, target_dev):
                    buf.data = buf.data.to(device=target_dev)
            elif not _same_device(buf.device, target_dev):
                buf.data = buf.data.to(device=target_dev)


__all__ = [
    "_MetaWeightInitMode",
    "resolve_model_path",
    "_resolve_target_tensor_dtype",
    "stream_safetensors_inplace",
    "load_sharded_safetensors_into",
]