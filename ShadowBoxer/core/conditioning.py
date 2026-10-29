# ShadowBoxer/core/conditioning.py
from __future__ import annotations

import math
from pathlib import Path
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from schema import SUPPORTED_MASK_BLEND_MODES, bbox_to_normalized
from .env import _canonicalize_device

_AI_TOOLKIT_MAX_REF_DIM = 512


def _probe_image_dimensions(path: Path) -> Optional[Tuple[int, int]]:
    try:
        with open(path, "rb") as f:
            hdr = f.read(64)
        if len(hdr) >= 24 and hdr[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", hdr[16:24])
            if w > 0 and h > 0:
                return int(w), int(h)
        if len(hdr) >= 48 and hdr[:16] == b"VIZJ_CONTAINER01":
            orig_w, orig_h, vae_w, vae_h = struct.unpack("<IIII", hdr[32:48])
            if vae_w > 0 and vae_h > 0:
                return int(vae_w), int(vae_h)
            if orig_w > 0 and orig_h > 0:
                return int(orig_w), int(orig_h)
        with Image.open(path) as img:
            return int(img.width), int(img.height)
    except Exception:
        return None


def _resolve_unboxed_reference_dims(
    path: Path,
    req_w: int,
    req_h: int,
    edit_format: str,
) -> Tuple[int, int]:
    fmt = str(edit_format or "standard").strip().lower()
    cap_w = (
        min(req_w, _AI_TOOLKIT_MAX_REF_DIM)
        if fmt in ("ai_toolkit_isolated", "ai_toolkit_full")
        else req_w
    )
    cap_h = (
        min(req_h, _AI_TOOLKIT_MAX_REF_DIM)
        if fmt in ("ai_toolkit_isolated", "ai_toolkit_full")
        else req_h
    )
    max_area = float(max(64 * 64, cap_w * cap_h))

    dims = _probe_image_dimensions(path)
    if dims is not None and dims[0] > 0 and dims[1] > 0:
        src_w, src_h = dims
        src_area = float(src_w * src_h)
        scale = min(1.0, math.sqrt(max_area / src_area))
        out_w = max(64, int(round((src_w * scale) / 16.0)) * 16)
        out_h = max(64, int(round((src_h * scale) / 16.0)) * 16)
        return out_w, out_h

    return max(64, (cap_w // 16) * 16), max(64, (cap_h // 16) * 16)


def _resolve_boxed_reference_dims(
    norm_bbox: Tuple[float, float, float, float],
    canvas_w: int,
    canvas_h: int,
) -> Tuple[int, int]:
    h_lat = max(1, int(canvas_h) // 16)
    w_lat = max(1, int(canvas_w) // 16)
    uy0, ux0, uy1, ux1 = norm_bbox
    y0 = max(0, min(h_lat - 1, int(round(uy0 * h_lat))))
    x0 = max(0, min(w_lat - 1, int(round(ux0 * w_lat))))
    y1 = max(y0 + 1, min(h_lat, int(round(uy1 * h_lat))))
    x1 = max(x0 + 1, min(w_lat, int(round(ux1 * w_lat))))
    return max(64, (x1 - x0) * 16), max(64, (y1 - y0) * 16)


def _normalize_vizinjector_framing(framing_mode: Optional[str]) -> str:
    fm = str(framing_mode or "inscribed").strip().lower()
    if fm in ("inscribed", "cover_crop"):
        return "inscribed"
    if fm in ("contain", "stretch", "never_crop"):
        return fm
    return "inscribed"


def resample_packed_latents(
    packed_latents: torch.Tensor,
    target_h_lat: int,
    target_w_lat: int,
) -> torch.Tensor:
    if packed_latents.ndim != 4:
        return packed_latents
    b, c, h, w = packed_latents.shape
    th = max(1, int(target_h_lat))
    tw = max(1, int(target_w_lat))
    if (h, w) == (th, tw):
        return packed_latents
    if c % 4 != 0:
        return F.interpolate(
            packed_latents,
            size=(th, tw),
            mode="bilinear",
            align_corners=False,
        )
    orig_dtype = packed_latents.dtype
    unpacked = (
        packed_latents.float()
        .reshape(b, c // 4, 2, 2, h, w)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(b, c // 4, 2 * h, 2 * w)
    )
    resampled = F.interpolate(
        unpacked,
        size=(2 * th, 2 * tw),
        mode="bilinear",
        align_corners=False,
    )
    repacked = (
        resampled.reshape(b, c // 4, th, 2, tw, 2)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(b, c, th, tw)
        .contiguous()
    )
    return repacked.to(dtype=orig_dtype)


def build_box_aperture_mask(
    layout_items: Sequence[Dict[str, Any]],
    height: int,
    width: int,
    coords_are_pixels: bool,
    feather_radius_lat: float,
    device: torch.device,
    only_flagged_boxes: bool = False,
) -> Optional[torch.Tensor]:
    h_lat = max(1, height // 16)
    w_lat = max(1, width // 16)
    active_boxes: List[Tuple[int, int, int, int]] = []
    for item in layout_items:
        if not isinstance(item, dict) or item.get("enabled", True) is False:
            continue
        if only_flagged_boxes and not (
            bool(item.get("inpaint_box", False))
            or bool(item.get("paste_to_canvas", False))
        ):
            continue
        bbox = item.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            uy0, ux0, uy1, ux1 = bbox_to_normalized(
                bbox, height=height, width=width, coords_are_pixels=coords_are_pixels
            )
            if (uy1 - uy0) > 1e-4 and (ux1 - ux0) > 1e-4:
                y0 = max(0, min(h_lat - 1, int(round(uy0 * h_lat))))
                x0 = max(0, min(w_lat - 1, int(round(ux0 * w_lat))))
                y1 = max(y0 + 1, min(h_lat, int(round(uy1 * h_lat))))
                x1 = max(x0 + 1, min(w_lat, int(round(ux1 * w_lat))))
                active_boxes.append((y0, x0, y1, x1))
    if not active_boxes:
        return None

    yy = torch.arange(h_lat, device=device, dtype=torch.float32).unsqueeze(1) + 0.5
    xx = torch.arange(w_lat, device=device, dtype=torch.float32).unsqueeze(0) + 0.5
    feather = max(0.25, float(feather_radius_lat))
    mask = torch.zeros((h_lat, w_lat), device=device, dtype=torch.float32)

    for y0, x0, y1, x1 in active_boxes:
        ly0 = float(y0)
        ly1 = float(y1)
        lx0 = float(x0)
        lx1 = float(x1)
        dy = torch.maximum(torch.maximum(ly0 - yy, yy - ly1), torch.zeros_like(yy))
        dx = torch.maximum(torch.maximum(lx0 - xx, xx - lx1), torch.zeros_like(xx))
        dist = torch.sqrt(dy * dy + dx * dx)
        box_m = 0.5 * (1.0 + torch.cos(math.pi * (dist / feather).clamp(0.0, 1.0)))
        mask = torch.maximum(mask, box_m)

    return mask.unsqueeze(0).unsqueeze(0).clamp(0.0, 1.0)


def compose_latent_inpaint_mask(
    paint_mask: Optional[torch.Tensor],
    box_mask: Optional[torch.Tensor],
    alpha_mask: Optional[torch.Tensor],
    blend_mode: str = "auto",
    preserve_outside_boxes: bool = False,
    invert_mask: bool = False,
    h_lat: Optional[int] = None,
    w_lat: Optional[int] = None,
    device: Union[str, torch.device] = torch.device("cpu"),
) -> Optional[torch.Tensor]:
    target_dev = _canonicalize_device(device)
    mode = str(blend_mode or "auto").strip().lower()
    if mode not in SUPPORTED_MASK_BLEND_MODES:
        mode = "auto"

    ref_t = (
        paint_mask
        if paint_mask is not None
        else (box_mask if box_mask is not None else alpha_mask)
    )
    if ref_t is None:
        return None
    eff_h = int(h_lat) if h_lat is not None else int(ref_t.shape[-2])
    eff_w = int(w_lat) if w_lat is not None else int(ref_t.shape[-1])

    def _norm_mask(m: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if m is None:
            return None
        t = m.to(device=target_dev, dtype=torch.float32)
        if t.ndim == 2:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.ndim == 3:
            t = t.unsqueeze(0)
        if t.shape[-2:] != (eff_h, eff_w):
            t = F.interpolate(
                t, size=(eff_h, eff_w), mode="bilinear", align_corners=False
            )
        return t.clamp(0.0, 1.0)

    m_paint = _norm_mask(paint_mask)
    m_box = _norm_mask(box_mask)
    m_alpha = _norm_mask(alpha_mask)

    composed: Optional[torch.Tensor] = None
    if mode == "paint_only":
        composed = m_paint
    elif mode == "boxes_only":
        composed = m_box
    elif mode == "alpha_only":
        composed = m_alpha
    elif mode == "union":
        for cand in (m_paint, m_box, m_alpha):
            if cand is not None:
                composed = cand if composed is None else torch.maximum(composed, cand)
    elif mode == "intersect":
        for cand in (m_paint, m_box, m_alpha):
            if cand is not None:
                composed = cand if composed is None else (composed * cand)
    else:
        if m_paint is not None and m_box is not None:
            if preserve_outside_boxes:
                composed = m_paint * m_box
            else:
                composed = torch.maximum(m_paint, m_box)
        elif m_paint is not None:
            composed = m_paint
        elif m_box is not None:
            composed = m_box
        elif m_alpha is not None:
            composed = m_alpha

    if composed is not None and invert_mask:
        composed = (1.0 - composed).clamp(0.0, 1.0)
    return composed


def splat_box_latents_to_canvas(
    base_canvas: Optional[torch.Tensor],
    splat_entries: Sequence[
        Tuple[
            torch.Tensor,
            Tuple[float, float, float, float],
            Optional[torch.Tensor],
            Optional[float],
        ]
    ],
    h_lat: int,
    w_lat: int,
    latent_channels: int,
    feather_radius_lat: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if base_canvas is not None:
        canvas = base_canvas.to(device=device, dtype=torch.float32).clone()
        if canvas.shape[-2:] != (h_lat, w_lat):
            canvas = resample_packed_latents(canvas, h_lat, w_lat)
    else:
        canvas = torch.zeros(
            (1, latent_channels, h_lat, w_lat), device=device, dtype=torch.float32
        )

    if not splat_entries:
        return canvas, torch.zeros(
            (1, 1, h_lat, w_lat), device=device, dtype=torch.float32
        )

    accum_lat = torch.zeros(
        (1, latent_channels, h_lat, w_lat), device=device, dtype=torch.float32
    )
    accum_weight = torch.zeros(
        (1, 1, h_lat, w_lat), device=device, dtype=torch.float32
    )
    union_alpha = torch.zeros(
        (1, 1, h_lat, w_lat), device=device, dtype=torch.float32
    )
    feather = max(0.25, float(feather_radius_lat))

    for box_lat_cpu, norm_bbox, box_mask_cpu, _ in splat_entries:
        uy0, ux0, uy1, ux1 = norm_bbox
        y0 = max(0, min(h_lat - 1, int(round(uy0 * h_lat))))
        x0 = max(0, min(w_lat - 1, int(round(ux0 * w_lat))))
        y1 = max(y0 + 1, min(h_lat, int(round(uy1 * h_lat))))
        x1 = max(x0 + 1, min(w_lat, int(round(ux1 * w_lat))))
        bh = y1 - y0
        bw = x1 - x0

        b_lat = box_lat_cpu.to(device=device, dtype=torch.float32)
        if b_lat.shape[-2:] != (bh, bw):
            b_lat = resample_packed_latents(b_lat, bh, bw)

        yy = torch.arange(bh, device=device, dtype=torch.float32).unsqueeze(1) + 0.5
        xx = torch.arange(bw, device=device, dtype=torch.float32).unsqueeze(0) + 0.5
        d_edge = torch.minimum(
            torch.minimum(yy, float(bh) - yy),
            torch.minimum(xx, float(bw) - xx),
        )
        edge_ramp = 0.5 * (
            1.0 - torch.cos(math.pi * (d_edge / feather).clamp(0.0, 1.0))
        )
        local_alpha = edge_ramp.unsqueeze(0).unsqueeze(0)

        if box_mask_cpu is not None:
            b_mask = box_mask_cpu.to(device=device, dtype=torch.float32)
            if b_mask.shape[-2:] != (bh, bw):
                b_mask = F.interpolate(
                    b_mask, size=(bh, bw), mode="bilinear", align_corners=False
                )
            local_alpha = local_alpha * b_mask.clamp(0.0, 1.0)

        local_alpha = local_alpha.clamp(0.0, 1.0)
        accum_lat[:, :, y0:y1, x0:x1] = (
            accum_lat[:, :, y0:y1, x0:x1] + local_alpha * b_lat
        )
        accum_weight[:, :, y0:y1, x0:x1] = (
            accum_weight[:, :, y0:y1, x0:x1] + local_alpha
        )
        union_alpha[:, :, y0:y1, x0:x1] = torch.maximum(
            union_alpha[:, :, y0:y1, x0:x1], local_alpha
        )

    union_alpha = union_alpha.clamp(0.0, 1.0)
    norm_fg = torch.where(
        accum_weight > 1e-6,
        accum_lat / accum_weight.clamp(min=1e-6),
        canvas,
    )
    canvas = union_alpha * norm_fg + (1.0 - union_alpha) * canvas
    return canvas, union_alpha


def load_image_tensor_16px(
    image_path: Union[str, Path],
    target_h: Optional[int] = None,
    target_w: Optional[int] = None,
) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    w_in, h_in = img.size
    if target_h is not None and target_w is not None:
        h_out = max(64, (target_h // 16) * 16)
        w_out = max(64, (target_w // 16) * 16)
    else:
        scale = min(1.0, math.sqrt((1024.0 * 1024.0) / float(max(1, w_in * h_in))))
        h_out = max(64, (int(round(h_in * scale)) // 16) * 16)
        w_out = max(64, (int(round(w_in * scale)) // 16) * 16)

    if (w_in, h_in) != (w_out, h_out):
        img = img.resize((w_out, h_out), resample=Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def pack_vision_patches_32px(
    image_path: Union[str, Path], max_patches: int = 2304
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    img = Image.open(image_path).convert("RGB")
    w_in, h_in = img.size
    max_pixels = max_patches * 16 * 16
    scale = min(1.0, math.sqrt(float(max_pixels) / float(max(1, w_in * h_in))))
    h_vis = max(32, (int(round(h_in * scale)) // 32) * 32)
    w_vis = max(32, (int(round(w_in * scale)) // 32) * 32)

    if (w_in, h_in) != (w_vis, h_vis):
        img = img.resize((w_vis, h_vis), resample=Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    normed = (arr - mean) / std

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
    llm_tokens = (grid_h * grid_w) // (merge * merge)
    return patches, grid_thw, llm_tokens


__all__ = [
    "_AI_TOOLKIT_MAX_REF_DIM",
    "_probe_image_dimensions",
    "_resolve_unboxed_reference_dims",
    "_resolve_boxed_reference_dims",
    "_normalize_vizinjector_framing",
    "resample_packed_latents",
    "build_box_aperture_mask",
    "compose_latent_inpaint_mask",
    "splat_box_latents_to_canvas",
    "load_image_tensor_16px",
    "pack_vision_patches_32px",
]