# ShadowBoxer/core/caches.py
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import torch

from models.tokenizer import MOSSImageTokenizer

from .conditioning import _probe_image_dimensions
from .env import _pin_cpu_tensor


class PromptEmbedCache:
    def __init__(self, max_entries: int = 32):
        self.max_entries = max_entries
        self._cache: OrderedDict[Tuple[Any, ...], torch.Tensor] = OrderedDict()
        self._last_tokens: Optional[List[int]] = None
        self.hits: int = 0
        self.pos_hits: int = 0
        self.misses: int = 0
        self.last_common_prefix_len: int = 0

    def clear(self) -> None:
        self._cache.clear()
        self._last_tokens = None

    def lookup(
        self,
        token_ids: Sequence[int],
        device: torch.device,
        is_negative: bool = False,
        vision_signature: Tuple[Any, ...] = (),
    ) -> Optional[torch.Tensor]:
        tok_list = [int(x) for x in token_ids]
        if not is_negative:
            if self._last_tokens is not None:
                self.last_common_prefix_len = MOSSImageTokenizer.find_common_prefix(
                    self._last_tokens, tok_list
                )
            else:
                self.last_common_prefix_len = 0
            self._last_tokens = tok_list

        key: Tuple[Any, ...] = (
            (tuple(tok_list), vision_signature)
            if vision_signature
            else tuple(tok_list)
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
    ) -> None:
        key: Tuple[Any, ...] = (
            (tuple(int(x) for x in token_ids), vision_signature)
            if vision_signature
            else tuple(int(x) for x in token_ids)
        )
        pinned = _pin_cpu_tensor(hidden)
        if pinned is not None:
            self._cache[key] = pinned
            self._cache.move_to_end(key)
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)


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
        path: Path, target_w: int, target_h: int, framing_mode: str
    ) -> str:
        fm = str(framing_mode).lower()
        if fm == "cover_crop":
            fm = "inscribed"
        if fm == "never_crop":
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
        st = path.stat()
        norm_fm = cls.canonicalize_framing(path, target_w, target_h, framing_mode)
        eff_mtf = round(float(mtf_boost), 4) if input_role == "photo_ref" else 0.0
        return (
            str(path.resolve()),
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
        st = path.stat()
        norm_fm = cls.canonicalize_framing(path, target_w, target_h, framing_mode)
        return (
            str(path.resolve()),
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
            packed_latents=_pin_cpu_tensor(record.packed_latents),
            vit_embeds=_pin_cpu_tensor(record.vit_embeds),
            vit_patches=_pin_cpu_tensor(record.vit_patches),
            grid_thw=_pin_cpu_tensor(record.grid_thw),
            llm_tokens=int(record.llm_tokens),
            latent_mask=_pin_cpu_tensor(record.latent_mask),
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
    "PromptEmbedCache",
    "CachedReferenceRecord",
    "ReferenceVisionCache",
]