# ShadowBoxer/core/caches.py
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from models.modeling_qwen35 import Qwen3_5HybridCache
from models.tokenizer import MOSSImageTokenizer
from .conditioning import _probe_image_dimensions
from .env import _pin_cpu_tensor

_DIM_PROBE_CACHE: OrderedDict[Tuple[str, int, int], Optional[Tuple[int, int]]] = (
    OrderedDict()
)


def _cached_probe_dimensions(
    path: Path, mtime_ns: int, file_size: int
) -> Optional[Tuple[int, int]]:
    key = (str(path), int(mtime_ns), int(file_size))
    if key in _DIM_PROBE_CACHE:
        _DIM_PROBE_CACHE.move_to_end(key)
        return _DIM_PROBE_CACHE[key]
    dims = _probe_image_dimensions(path)
    _DIM_PROBE_CACHE[key] = dims
    while len(_DIM_PROBE_CACHE) > 128:
        _DIM_PROBE_CACHE.popitem(last=False)
    return dims


def _ensure_pinned_cpu(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None:
        return None
    if (
        t.device.type == "cpu"
        and not t.is_inference()
        and t.is_contiguous()
        and (not torch.cuda.is_available() or t.is_pinned())
    ):
        return t
    return _pin_cpu_tensor(t)


@dataclass
class PinnedPrefixState:
    prefix_tokens: Tuple[int, ...]
    vision_signature: Tuple[Any, ...]
    prefix_hidden: torch.Tensor
    conv_states: Dict[int, torch.Tensor]
    recurrent_states: Dict[int, torch.Tensor]
    key_cache: Dict[int, torch.Tensor]
    value_cache: Dict[int, torch.Tensor]
    rope_deltas: Optional[torch.Tensor]
    available_offsets: Tuple[int, ...] = ()
    conv_checkpoints: Dict[int, Dict[int, torch.Tensor]] = field(default_factory=dict)
    recurrent_checkpoints: Dict[int, Dict[int, torch.Tensor]] = field(
        default_factory=dict
    )

    def resolve_best_offset(self, max_allowed_len: int) -> Optional[int]:
        offsets = (
            self.available_offsets
            if self.available_offsets
            else (len(self.prefix_tokens),)
        )
        best: Optional[int] = None
        for off in offsets:
            if off <= max_allowed_len:
                if best is None or off > best:
                    best = off
        return best

    def restore(
        self, device: torch.device, snap_len: Optional[int] = None
    ) -> Tuple[Qwen3_5HybridCache, torch.Tensor, int]:
        full_len = len(self.prefix_tokens)
        target_len = int(snap_len) if snap_len is not None else full_len
        cache = Qwen3_5HybridCache()
        cache.past_seen_tokens = target_len
        if self.rope_deltas is not None:
            cache.rope_deltas = self.rope_deltas.to(device=device, non_blocking=True)

        conv_src = (
            self.conv_checkpoints[target_len]
            if target_len in self.conv_checkpoints
            else self.conv_states
        )
        rec_src = (
            self.recurrent_checkpoints[target_len]
            if target_len in self.recurrent_checkpoints
            else self.recurrent_states
        )

        for k, v in conv_src.items():
            cache.conv_states[k] = v.to(device=device, non_blocking=True).clone()
        for k, v in rec_src.items():
            cache.recurrent_states[k] = (
                v.to(device=device, dtype=torch.float32, non_blocking=True).clone()
            )
        for k, v in self.key_cache.items():
            kv_slice = v[:, :, :target_len, :] if v.shape[2] > target_len else v
            cache.key_cache[k] = kv_slice.to(
                device=device, non_blocking=True
            ).contiguous()
        for k, v in self.value_cache.items():
            vv_slice = v[:, :, :target_len, :] if v.shape[2] > target_len else v
            cache.value_cache[k] = vv_slice.to(
                device=device, non_blocking=True
            ).contiguous()

        hidden_slice = (
            self.prefix_hidden[:, :target_len, :]
            if self.prefix_hidden.shape[1] > target_len
            else self.prefix_hidden
        )
        hidden_dev = hidden_slice.to(device=device, non_blocking=True).contiguous()
        return cache, hidden_dev, target_len


class PromptEmbedCache:
    def __init__(self, max_entries: int = 32, max_prefix_entries: int = 8):
        self.max_entries = max_entries
        self.max_prefix_entries = max_prefix_entries
        self._cache: OrderedDict[Tuple[Any, ...], torch.Tensor] = OrderedDict()
        self._prefix_cache: OrderedDict[Tuple[Any, ...], PinnedPrefixState] = (
            OrderedDict()
        )
        self._last_tokens: Optional[List[int]] = None
        self.hits: int = 0
        self.pos_hits: int = 0
        self.splice_hits: int = 0
        self.misses: int = 0
        self.last_common_prefix_len: int = 0

    def clear(self) -> None:
        self._cache.clear()
        self._prefix_cache.clear()
        self._last_tokens = None

    def lookup(
        self,
        token_ids: Sequence[int],
        device: torch.device,
        is_negative: bool = False,
        vision_signature: Tuple[Any, ...] = (),
    ) -> Optional[torch.Tensor]:
        tok_list = (
            token_ids
            if isinstance(token_ids, list)
            else [int(x) for x in token_ids]
        )
        if not is_negative:
            if self._last_tokens is not None:
                self.last_common_prefix_len = MOSSImageTokenizer.find_common_prefix(
                    self._last_tokens, tok_list
                )
            else:
                self.last_common_prefix_len = 0
            self._last_tokens = list(tok_list)

        tok_tuple = tuple(tok_list)
        key: Tuple[Any, ...] = (
            (tok_tuple, vision_signature) if vision_signature else tok_tuple
        )
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            if not is_negative:
                self.pos_hits += 1
            return self._cache[key].to(device=device, non_blocking=True)

        self.misses += 1
        return None

    def insert(
        self,
        token_ids: Sequence[int],
        hidden: torch.Tensor,
        vision_signature: Tuple[Any, ...] = (),
    ) -> Optional[torch.Tensor]:
        tok_tuple = tuple(int(x) for x in token_ids)
        key: Tuple[Any, ...] = (
            (tok_tuple, vision_signature) if vision_signature else tok_tuple
        )
        pinned = _ensure_pinned_cpu(hidden)
        if pinned is not None:
            self._cache[key] = pinned
            self._cache.move_to_end(key)
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        return pinned

    def insert_prefix_state(
        self,
        prefix_tokens: Sequence[int],
        prefix_hidden: torch.Tensor,
        hybrid_cache: Qwen3_5HybridCache,
        vision_signature: Tuple[Any, ...] = (),
    ) -> None:
        pfx_tuple = tuple(int(x) for x in prefix_tokens)
        full_len = len(pfx_tuple)
        if full_len == 0:
            return
        pinned_hidden = _ensure_pinned_cpu(prefix_hidden)
        if pinned_hidden is None:
            return

        conv_pinned: Dict[int, torch.Tensor] = {}
        for k, v in hybrid_cache.conv_states.items():
            pv = _ensure_pinned_cpu(v)
            if pv is not None:
                conv_pinned[k] = pv

        rec_pinned: Dict[int, torch.Tensor] = {}
        for k, v in hybrid_cache.recurrent_states.items():
            pv = _ensure_pinned_cpu(v)
            if pv is not None:
                rec_pinned[k] = pv

        conv_cp_pinned: Dict[int, Dict[int, torch.Tensor]] = {}
        for cp, layer_map in hybrid_cache.conv_checkpoints.items():
            if 1 <= cp <= full_len:
                pinned_map: Dict[int, torch.Tensor] = {}
                for k, v in layer_map.items():
                    pv = _ensure_pinned_cpu(v)
                    if pv is not None:
                        pinned_map[k] = pv
                if pinned_map:
                    conv_cp_pinned[int(cp)] = pinned_map

        rec_cp_pinned: Dict[int, Dict[int, torch.Tensor]] = {}
        for cp, layer_map in hybrid_cache.recurrent_checkpoints.items():
            if 1 <= cp <= full_len:
                pinned_map = {}
                for k, v in layer_map.items():
                    pv = _ensure_pinned_cpu(v)
                    if pv is not None:
                        pinned_map[k] = pv
                if pinned_map:
                    rec_cp_pinned[int(cp)] = pinned_map

        valid_offsets = set(conv_cp_pinned.keys()) & set(rec_cp_pinned.keys())
        if conv_pinned or rec_pinned:
            valid_offsets.add(full_len)
            conv_cp_pinned.setdefault(full_len, conv_pinned)
            rec_cp_pinned.setdefault(full_len, rec_pinned)

        sorted_offsets = tuple(sorted(valid_offsets))
        if not sorted_offsets:
            return

        max_needed_kv = max(sorted_offsets)
        k_pinned: Dict[int, torch.Tensor] = {}
        for k, v in hybrid_cache.key_cache.items():
            v_slice = v[:, :, :max_needed_kv, :] if v.shape[2] > max_needed_kv else v
            pv = _ensure_pinned_cpu(v_slice)
            if pv is not None:
                k_pinned[k] = pv

        v_pinned: Dict[int, torch.Tensor] = {}
        for k, v in hybrid_cache.value_cache.items():
            v_slice = v[:, :, :max_needed_kv, :] if v.shape[2] > max_needed_kv else v
            pv = _ensure_pinned_cpu(v_slice)
            if pv is not None:
                v_pinned[k] = pv

        deltas_pinned = _ensure_pinned_cpu(hybrid_cache.rope_deltas)
        if pinned_hidden.shape[1] > max_needed_kv:
            pinned_hidden = _ensure_pinned_cpu(pinned_hidden[:, :max_needed_kv, :])
            if pinned_hidden is None:
                return

        key: Tuple[Any, ...] = (
            (pfx_tuple[:max_needed_kv], vision_signature)
            if vision_signature
            else pfx_tuple[:max_needed_kv]
        )
        self._prefix_cache[key] = PinnedPrefixState(
            prefix_tokens=pfx_tuple[:max_needed_kv],
            vision_signature=vision_signature,
            prefix_hidden=pinned_hidden,
            conv_states=conv_pinned,
            recurrent_states=rec_pinned,
            key_cache=k_pinned,
            value_cache=v_pinned,
            rope_deltas=deltas_pinned,
            available_offsets=sorted_offsets,
            conv_checkpoints=conv_cp_pinned,
            recurrent_checkpoints=rec_cp_pinned,
        )
        self._prefix_cache.move_to_end(key)
        while len(self._prefix_cache) > self.max_prefix_entries:
            self._prefix_cache.popitem(last=False)

    def lookup_prefix_splice(
        self,
        token_ids: Sequence[int],
        device: torch.device,
        vision_signature: Tuple[Any, ...] = (),
    ) -> Optional[Tuple[Qwen3_5HybridCache, torch.Tensor, int]]:
        tok_list = (
            token_ids
            if isinstance(token_ids, list)
            else [int(x) for x in token_ids]
        )
        total_len = len(tok_list)
        if total_len < 16 or not self._prefix_cache:
            return None

        best_key: Optional[Tuple[Any, ...]] = None
        best_state: Optional[PinnedPrefixState] = None
        best_len = 0

        for key, state in reversed(self._prefix_cache.items()):
            if state.vision_signature != vision_signature:
                continue
            common_len = MOSSImageTokenizer.find_common_prefix(
                state.prefix_tokens, tok_list
            )
            max_allowed = min(common_len, total_len - 1)
            if max_allowed <= best_len:
                continue
            cand_off = state.resolve_best_offset(max_allowed)
            if cand_off is not None and cand_off > best_len:
                best_key = key
                best_state = state
                best_len = cand_off

        if best_state is None or best_key is None or best_len <= 0:
            return None

        self._prefix_cache.move_to_end(best_key)
        self.splice_hits += 1
        return best_state.restore(device, snap_len=best_len)


@dataclass
class CachedReferenceRecord:
    packed_latents: Optional[torch.Tensor]
    vit_embeds: Optional[torch.Tensor]
    vit_patches: Optional[torch.Tensor]
    grid_thw: Optional[torch.Tensor]
    llm_tokens: int
    latent_mask: Optional[torch.Tensor]
    viz_meta: Optional[Any] = None
    source_signature: Tuple[Any, ...] = ()


class ReferenceVisionCache:
    def __init__(self, max_entries: int = 32):
        self.max_entries = max_entries
        self._cache: OrderedDict[Tuple[Any, ...], CachedReferenceRecord] = OrderedDict()
        self._vae_cache: OrderedDict[
            Tuple[Any, ...],
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Any]],
        ] = OrderedDict()
        self.hits: int = 0
        self.misses: int = 0

    def clear(self) -> None:
        self._cache.clear()
        self._vae_cache.clear()

    @staticmethod
    def canonicalize_framing(
        path: Path,
        target_w: int,
        target_h: int,
        framing_mode: str,
        mtime_ns: Optional[int] = None,
        file_size: Optional[int] = None,
    ) -> str:
        fm = str(framing_mode).lower()
        if fm == "cover_crop":
            fm = "inscribed"
        if fm == "never_crop":
            if mtime_ns is not None and file_size is not None:
                dims = _cached_probe_dimensions(path, mtime_ns, file_size)
            else:
                dims = _probe_image_dimensions(path)
            if dims is not None:
                src_w, src_h = dims
                if src_w * target_h == src_h * target_w:
                    return "inscribed"
        return fm

    @classmethod
    def make_vae_key(
        cls,
        path: Path,
        target_w: int,
        target_h: int,
        framing_mode: str,
        input_role: str,
        clamp_sigma: float,
        mtf_boost: float = 0.08,
        mask_feather_radius: float = 1.25,
    ) -> Tuple[Any, ...]:
        resolved = path.resolve()
        st = resolved.stat()
        norm_fm = cls.canonicalize_framing(
            resolved,
            target_w,
            target_h,
            framing_mode,
            mtime_ns=int(st.st_mtime_ns),
            file_size=int(st.st_size),
        )
        eff_mtf = round(float(mtf_boost), 4) if input_role == "photo_ref" else 0.0
        return (
            str(resolved),
            int(st.st_mtime_ns),
            int(st.st_size),
            int(target_w),
            int(target_h),
            norm_fm,
            str(input_role),
            round(float(clamp_sigma), 4),
            eff_mtf,
            round(float(mask_feather_radius), 4),
        )

    @classmethod
    def make_key(
        cls,
        path: Path,
        target_w: int,
        target_h: int,
        framing_mode: str,
        input_role: str,
        edit_format: str,
        max_vit_patches: int,
        clamp_sigma: float,
    ) -> Tuple[Any, ...]:
        resolved = path.resolve()
        st = resolved.stat()
        norm_fm = cls.canonicalize_framing(
            resolved,
            target_w,
            target_h,
            framing_mode,
            mtime_ns=int(st.st_mtime_ns),
            file_size=int(st.st_size),
        )
        return (
            str(resolved),
            int(st.st_mtime_ns),
            int(st.st_size),
            int(target_w),
            int(target_h),
            norm_fm,
            str(input_role),
            str(edit_format),
            int(max_vit_patches),
            round(float(clamp_sigma), 4),
        )

    def lookup(self, key: Tuple[Any, ...]) -> Optional[CachedReferenceRecord]:
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def lookup_vae(
        self, vae_key: Tuple[Any, ...]
    ) -> Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Any]]]:
        if vae_key in self._vae_cache:
            self._vae_cache.move_to_end(vae_key)
            self.hits += 1
            return self._vae_cache[vae_key]
        return None

    def insert(
        self,
        key: Tuple[Any, ...],
        record: CachedReferenceRecord,
        vae_key: Optional[Tuple[Any, ...]] = None,
    ) -> CachedReferenceRecord:
        pinned_rec = CachedReferenceRecord(
            packed_latents=_ensure_pinned_cpu(record.packed_latents),
            vit_embeds=_ensure_pinned_cpu(record.vit_embeds),
            vit_patches=_ensure_pinned_cpu(record.vit_patches),
            grid_thw=_ensure_pinned_cpu(record.grid_thw),
            llm_tokens=int(record.llm_tokens),
            latent_mask=_ensure_pinned_cpu(record.latent_mask),
            viz_meta=record.viz_meta,
            source_signature=record.source_signature or key[:6],
        )
        self._cache[key] = pinned_rec
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

        if vae_key is not None and pinned_rec.packed_latents is not None:
            self._vae_cache[vae_key] = (
                pinned_rec.packed_latents,
                pinned_rec.latent_mask,
                pinned_rec.viz_meta,
            )
            self._vae_cache.move_to_end(vae_key)
            while len(self._vae_cache) > self.max_entries:
                self._vae_cache.popitem(last=False)

        return pinned_rec


__all__ = [
    "PinnedPrefixState",
    "PromptEmbedCache",
    "CachedReferenceRecord",
    "ReferenceVisionCache",
]