# ShadowBoxer/models/moss/rope.py
from __future__ import annotations

from collections import OrderedDict
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from .geometry import _normalize_ref_bboxes


def _rotate(
    x: torch.Tensor,
    rope: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    if isinstance(rope, tuple):
        sin_signed, cos = rope
    else:
        sin, cos = rope.chunk(2, -1)
        sin_signed = torch.cat((-sin, sin), -1).unsqueeze(-2)
        cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return torch.addcmul(x * cos, torch.cat((b, a), -1), sin_signed)


class _RoPE(nn.Module):
    def __init__(self, head_dim: int, base: float):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        self.register_buffer("bands", self._default_bands(torch.device("cpu")))
        self._phase_cache: OrderedDict[
            Tuple[Any, ...],
            torch.Tensor,
        ] = OrderedDict()
        self._expanded_cache: OrderedDict[
            Tuple[Any, ...],
            Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        ] = OrderedDict()
        self._text_rope_cache: OrderedDict[
            Tuple[int, str, torch.dtype],
            Tuple[torch.Tensor, torch.Tensor],
        ] = OrderedDict()

    def _default_bands(self, device: torch.device) -> torch.Tensor:
        exponent = (
            torch.arange(self.num_bands, device="cpu", dtype=torch.float32)
            / self.num_bands
        )
        return (1.0 / (self.base**exponent)).to(device)

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.bands = self._default_bands(self.bands.device)
        self.clear_cache()
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.bands = self._default_bands(self.bands.device)
        self.clear_cache()

    def clear_cache(self) -> None:
        self._phase_cache.clear()
        self._expanded_cache.clear()
        self._text_rope_cache.clear()

    def get_text_rope(
        self,
        text_length: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = (int(text_length), str(device), dtype)
        cached = self._text_rope_cache.get(cache_key)
        if cached is not None:
            self._text_rope_cache.move_to_end(cache_key)
            return cached

        if self.bands.dtype != torch.float32 or self.bands.device != device:
            self.bands = self._default_bands(device)
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        phase = text.unsqueeze(-1) * self.bands
        sin_half = phase.sin().flatten(-2).to(dtype=dtype)
        cos_half = phase.cos().flatten(-2).to(dtype=dtype)
        sin_text = (
            torch.cat((-sin_half, sin_half), dim=-1)[None, :, None, :].contiguous()
        )
        cos_text = (
            torch.cat((cos_half, cos_half), dim=-1)[None, :, None, :].contiguous()
        )
        pair = (sin_text, cos_text)
        self._text_rope_cache[cache_key] = pair
        while len(self._text_rope_cache) > 16:
            self._text_rope_cache.popitem(last=False)
        return pair

    def _compute_phase(
        self,
        height: int,
        width: int,
        text_length: int,
        device: torch.device,
        ref_shapes: Optional[List[Tuple[int, int]]] = None,
        ref_bboxes: Optional[Sequence[Optional[Sequence[float]]]] = None,
    ) -> torch.Tensor:
        if self.bands.dtype != torch.float32 or self.bands.device != device:
            self.bands = self._default_bands(device)

        cy = (height - 1) / 2.0
        cx = (width - 1) / 2.0

        h, w = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        image = (
            torch.stack(
                (torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1
            )
            .flatten(0, 1)
            .float()
        )
        image[:, 2] -= cy
        image[:, 3] -= cx

        coord_parts = [image]
        if ref_shapes:
            norm_boxes = _normalize_ref_bboxes(
                ref_shapes, ref_bboxes, height=height, width=width
            )
            for r_idx, ((rh, rw), r_box) in enumerate(
                zip(ref_shapes, norm_boxes), start=1
            ):
                if r_box is not None:
                    uy0, ux0, uy1, ux1 = r_box
                    y0 = max(0, min(height - 1, int(round(uy0 * height))))
                    x0 = max(0, min(width - 1, int(round(ux0 * width))))
                    y1 = max(y0 + 1, min(height, int(round(uy1 * height))))
                    x1 = max(x0 + 1, min(width, int(round(ux1 * width))))
                else:
                    y0, x0, y1, x1 = 0, 0, height, width

                if rh == (y1 - y0) and rw == (x1 - x0):
                    y_axis = (
                        torch.arange(y0, y1, device=device, dtype=torch.float32)
                        - cy
                    )
                    x_axis = (
                        torch.arange(x0, x1, device=device, dtype=torch.float32)
                        - cx
                    )
                else:
                    row_centers = (
                        torch.arange(rh, device=device, dtype=torch.float32) + 0.5
                    ) / float(max(1, rh))
                    col_centers = (
                        torch.arange(rw, device=device, dtype=torch.float32) + 0.5
                    ) / float(max(1, rw))
                    y_axis = (
                        float(y0) + row_centers * float(y1 - y0)
                    ) - 0.5 - cy
                    x_axis = (
                        float(x0) + col_centers * float(x1 - x0)
                    ) - 0.5 - cx

                gh, gw = torch.meshgrid(y_axis, x_axis, indexing="ij")
                ref_coord = torch.stack(
                    (
                        torch.full_like(gh, float(r_idx)),
                        torch.full_like(gh, float(text_length)),
                        gh,
                        gw,
                    ),
                    -1,
                ).flatten(0, 1)
                coord_parts.append(ref_coord)

        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        coord_parts.append(text)
        return torch.cat(coord_parts, 0).unsqueeze(-1) * self.bands

    def get_expanded_rope(
        self,
        height: int,
        width: int,
        text_length: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        ref_shapes: Optional[List[Tuple[int, int]]] = None,
        ref_bboxes: Optional[Sequence[Optional[Sequence[float]]]] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        ref_key = tuple(ref_shapes) if ref_shapes else ()
        bbox_key = _normalize_ref_bboxes(
            ref_shapes, ref_bboxes, height=height, width=width
        )
        cache_key = (height, width, text_length, ref_key, bbox_key, str(device), dtype)
        cached = self._expanded_cache.get(cache_key)
        if cached is not None:
            self._expanded_cache.move_to_end(cache_key)
            return cached

        phase = self._compute_phase(
            height, width, text_length, device, ref_shapes, bbox_key
        )
        sin_half = phase.sin().flatten(-2).to(dtype=dtype)
        cos_half = phase.cos().flatten(-2).to(dtype=dtype)
        sin_signed = (
            torch.cat((-sin_half, sin_half), dim=-1)[None, :, None, :].contiguous()
        )
        cos_full = torch.cat((cos_half, cos_half), dim=-1)[None, :, None, :].contiguous()

        total_img_tokens = sin_signed.shape[1] - text_length
        sin_text = sin_signed[:, total_img_tokens:, :, :]
        cos_text = cos_full[:, total_img_tokens:, :, :]

        text_key = (int(text_length), str(device), dtype)
        if text_key not in self._text_rope_cache:
            self._text_rope_cache[text_key] = (
                sin_text.contiguous(),
                cos_text.contiguous(),
            )
            while len(self._text_rope_cache) > 16:
                self._text_rope_cache.popitem(last=False)

        pair = ((sin_signed, cos_full), (sin_text, cos_text))
        self._expanded_cache[cache_key] = pair
        while len(self._expanded_cache) > 16:
            self._expanded_cache.popitem(last=False)
        return pair

    def forward(
        self,
        height: int,
        width: int,
        text_length: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        ref_shapes: Optional[List[Tuple[int, int]]] = None,
        ref_bboxes: Optional[Sequence[Optional[Sequence[float]]]] = None,
    ) -> torch.Tensor:
        ref_key = tuple(ref_shapes) if ref_shapes else ()
        bbox_key = _normalize_ref_bboxes(
            ref_shapes, ref_bboxes, height=height, width=width
        )
        cache_key = (height, width, text_length, ref_key, bbox_key, str(device), dtype)
        cached = self._phase_cache.get(cache_key)
        if cached is not None:
            self._phase_cache.move_to_end(cache_key)
            return cached

        phase = self._compute_phase(
            height, width, text_length, device, ref_shapes, bbox_key
        )
        rope_out = (
            torch.cat((phase.sin(), phase.cos()), -2)
            .flatten(-2)
            .unsqueeze(0)
            .to(dtype=dtype)
        )
        self._phase_cache[cache_key] = rope_out
        while len(self._phase_cache) > 16:
            self._phase_cache.popitem(last=False)
        return rope_out


__all__ = ["_rotate", "_RoPE"]