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

from schema import (
    SUPPORTED_MASK_BLEND_MODES,
    bbox_to_normalized,
    extract_box_text_and_caption,
)
from .env import _canonicalize_device

_AI_TOOLKIT_MAX_REF_DIM = 768
_HIERARCHICAL_SPLAT_GAMMA = 4.0
_HIERARCHICAL_SPLAT_Z_GAMMA = 2.5
_CFL_SIGMA_FLOOR = 0.12
_CFL_STEP_MULTIPLIER = 1.85


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
    max_vit_patches: int = 2304,
) -> Tuple[int, int]:
    fmt = str(edit_format or "standard").strip().lower()
    is_ait = fmt in ("ai_toolkit_isolated", "ai_toolkit_full")
    if is_ait:
        eff_patches = max(64, int(max_vit_patches or 2304))
        vit_side = max(512, int(math.isqrt(eff_patches)) * 16)
        ref_cap = (
            min(_AI_TOOLKIT_MAX_REF_DIM, vit_side)
            if eff_patches <= 2304
            else vit_side
        )
        cap_w = min(int(req_w), ref_cap)
        cap_h = min(int(req_h), ref_cap)
    else:
        cap_w = int(req_w)
        cap_h = int(req_h)
    max_area = float(max(64 * 64, cap_w * cap_h))
    dims = _probe_image_dimensions(path)
    if dims is not None and dims[0] > 0 and dims[1] > 0:
        src_w, src_h = dims
        src_area = float(src_w * src_h)
        raw_scale = math.sqrt(max_area / src_area)
        scale = (
            raw_scale
            if (is_ait and int(max_vit_patches or 2304) >= 2304)
            else min(1.0, raw_scale)
        )
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


def _unpack_latents_2x2(packed: torch.Tensor) -> torch.Tensor:
    b, c, h, w = packed.shape
    return (
        packed.reshape(b, c // 4, 2, 2, h, w)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(b, c // 4, 2 * h, 2 * w)
    )


def _pack_latents_2x2(unpacked: torch.Tensor) -> torch.Tensor:
    b, c_sub, h2, w2 = unpacked.shape
    h = h2 // 2
    w = w2 // 2
    return (
        unpacked.reshape(b, c_sub, h, 2, w, 2)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(b, c_sub * 4, h, w)
        .contiguous()
    )


def resample_packed_latents(
    packed_latents: torch.Tensor,
    target_h_lat: int,
    target_w_lat: int,
) -> torch.Tensor:
    if packed_latents.ndim != 4:
        return packed_latents
    _, c, h, w = packed_latents.shape
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
    unpacked = _unpack_latents_2x2(packed_latents.float())
    resampled = F.interpolate(
        unpacked,
        size=(2 * th, 2 * tw),
        mode="bilinear",
        align_corners=False,
    )
    return _pack_latents_2x2(resampled).to(dtype=orig_dtype)


def build_box_aperture_mask(
    layout_items: Sequence[Dict[str, Any]],
    height: int,
    width: int,
    coords_are_pixels: bool,
    feather_radius_lat: float,
    device: torch.device,
    only_flagged_boxes: bool = False,
    include_paste_boxes: bool = True,
) -> Optional[torch.Tensor]:
    h_lat = max(1, height // 16)
    w_lat = max(1, width // 16)
    active_boxes: List[Tuple[int, int, int, int]] = []
    for item in layout_items:
        if not isinstance(item, dict) or item.get("enabled", True) is False:
            continue
        role_str = str(item.get("input_role") or "").strip().lower()
        has_img = bool(
            str(
                item.get("image_path")
                or item.get("image")
                or item.get("vizj_path")
                or ""
            ).strip()
        )
        if (
            role_str == "inpaint_mask"
            and has_img
            and not bool(item.get("inpaint_box", False))
        ):
            continue
        if only_flagged_boxes:
            is_inpaint = bool(item.get("inpaint_box", False))
            is_paste = (
                include_paste_boxes
                and bool(item.get("paste_to_canvas", False))
                and role_str not in ("depth_map", "color_map", "inpaint_mask")
            )
            if not (is_inpaint or is_paste):
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
    raw_feather = max(0.0, float(feather_radius_lat))
    mask = torch.zeros((h_lat, w_lat), device=device, dtype=torch.float32)

    for y0, x0, y1, x1 in active_boxes:
        ly0 = float(y0)
        ly1 = float(y1)
        lx0 = float(x0)
        lx1 = float(x1)
        dy = torch.maximum(torch.maximum(ly0 - yy, yy - ly1), torch.zeros_like(yy))
        dx = torch.maximum(torch.maximum(lx0 - xx, xx - lx1), torch.zeros_like(xx))
        dist = torch.sqrt(dy * dy + dx * dx)
        if raw_feather <= 1e-4:
            box_m = (dist <= 1e-5).to(dtype=torch.float32)
        else:
            box_m = 0.5 * (
                1.0 + torch.cos(math.pi * (dist / raw_feather).clamp(0.0, 1.0))
            )
        mask = torch.maximum(mask, box_m)

    return mask.unsqueeze(0).unsqueeze(0).clamp(0.0, 1.0)


def splat_box_masks_to_canvas(
    base_mask: Optional[torch.Tensor],
    mask_entries: Sequence[Tuple[Any, ...]],
    h_lat: int,
    w_lat: int,
    feather_radius_lat: float = 0.0,
    device: Union[str, torch.device] = torch.device("cpu"),
    feather_box_edges: bool = False,
) -> Optional[torch.Tensor]:
    if base_mask is None and not mask_entries:
        return None
    target_dev = _canonicalize_device(device)
    eff_h = max(1, int(h_lat))
    eff_w = max(1, int(w_lat))

    if base_mask is not None:
        canvas_mask = base_mask.to(device=target_dev, dtype=torch.float32).clone()
        if canvas_mask.ndim == 2:
            canvas_mask = canvas_mask.unsqueeze(0).unsqueeze(0)
        elif canvas_mask.ndim == 3:
            canvas_mask = canvas_mask.unsqueeze(0)
        if canvas_mask.shape[-2:] != (eff_h, eff_w):
            canvas_mask = F.interpolate(
                canvas_mask, size=(eff_h, eff_w), mode="bilinear", align_corners=False
            )
        canvas_mask = canvas_mask.clamp(0.0, 1.0)
    else:
        canvas_mask = torch.zeros(
            (1, 1, eff_h, eff_w), device=target_dev, dtype=torch.float32
        )

    if not mask_entries:
        return canvas_mask

    feather = max(0.0, float(feather_radius_lat))
    for entry in mask_entries:
        if len(entry) < 2 or entry[0] is None:
            continue
        box_mask_t = entry[0]
        norm_bbox = entry[1]
        m_dev = box_mask_t.to(device=target_dev, dtype=torch.float32)
        if m_dev.ndim == 2:
            m_dev = m_dev.unsqueeze(0).unsqueeze(0)
        elif m_dev.ndim == 3:
            m_dev = m_dev.unsqueeze(0)
        m_dev = m_dev.clamp(0.0, 1.0)

        if norm_bbox is None or len(norm_bbox) != 4:
            if m_dev.shape[-2:] != (eff_h, eff_w):
                m_dev = F.interpolate(
                    m_dev, size=(eff_h, eff_w), mode="bilinear", align_corners=False
                )
            canvas_mask = torch.maximum(canvas_mask, m_dev.clamp(0.0, 1.0))
            continue

        uy0, ux0, uy1, ux1 = norm_bbox
        y0 = max(0, min(eff_h - 1, int(round(float(uy0) * eff_h))))
        x0 = max(0, min(eff_w - 1, int(round(float(ux0) * eff_w))))
        y1 = max(y0 + 1, min(eff_h, int(round(float(uy1) * eff_h))))
        x1 = max(x0 + 1, min(eff_w, int(round(float(ux1) * eff_w))))
        bh = y1 - y0
        bw = x1 - x0

        if m_dev.shape[-2:] != (bh, bw):
            m_dev = F.interpolate(
                m_dev, size=(bh, bw), mode="bilinear", align_corners=False
            )

        if feather_box_edges and feather > 1e-4:
            yy = (
                torch.arange(bh, device=target_dev, dtype=torch.float32).unsqueeze(1)
                + 0.5
            )
            xx = (
                torch.arange(bw, device=target_dev, dtype=torch.float32).unsqueeze(0)
                + 0.5
            )
            d_edge = torch.minimum(
                torch.minimum(yy, float(bh) - yy),
                torch.minimum(xx, float(bw) - xx),
            )
            edge_ramp = 0.5 * (
                1.0 - torch.cos(math.pi * (d_edge / feather).clamp(0.0, 1.0))
            )
            m_dev = torch.minimum(m_dev, edge_ramp.unsqueeze(0).unsqueeze(0))

        canvas_mask[:, :, y0:y1, x0:x1] = torch.maximum(
            canvas_mask[:, :, y0:y1, x0:x1], m_dev.clamp(0.0, 1.0)
        )

    return canvas_mask.clamp(0.0, 1.0)


def build_box_denoise_horizon_map(
    layout_items: Sequence[Dict[str, Any]],
    height: int,
    width: int,
    coords_are_pixels: bool,
    feather_radius_lat: float,
    default_denoise_strength: float = 1.0,
    device: Union[str, torch.device] = torch.device("cpu"),
    only_flagged_boxes: bool = False,
    include_paste_boxes: bool = True,
    return_none_if_uniform: bool = False,
) -> Optional[torch.Tensor]:
    target_dev = _canonicalize_device(device)
    h_lat = max(1, int(height) // 16)
    w_lat = max(1, int(width) // 16)
    fallback_dn = max(0.01, min(1.0, float(default_denoise_strength)))
    total_cells = float(max(1, h_lat * w_lat))

    parsed_boxes: List[Tuple[int, int, int, int, float, float, bool]] = []
    deferred_text_boxes: List[Tuple[int, int, int, int, float, bool]] = []

    for item in layout_items:
        if not isinstance(item, dict) or item.get("enabled", True) is False:
            continue
        raw_dn = item.get("denoise_strength")
        has_explicit_dn = raw_dn is not None and str(raw_dn).strip() != ""
        role_str = str(item.get("input_role") or "").strip().lower()
        lbl, _, _ = extract_box_text_and_caption(item)
        img_p = (
            item.get("image_path")
            or item.get("image")
            or item.get("vizj_path")
        )
        has_ref = isinstance(img_p, str) and bool(img_p.strip())
        is_unreferenced_text = (lbl == "text") and not has_ref
        is_inpaint = bool(item.get("inpaint_box", False)) or (
            role_str == "inpaint_mask"
        )
        is_paste = (
            include_paste_boxes
            and bool(item.get("paste_to_canvas", False))
            and role_str not in ("depth_map", "color_map", "inpaint_mask")
        )
        if only_flagged_boxes and not (
            is_inpaint or is_paste or has_explicit_dn or is_unreferenced_text
        ):
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        uy0, ux0, uy1, ux1 = bbox_to_normalized(
            bbox, height=height, width=width, coords_are_pixels=coords_are_pixels
        )
        if (uy1 - uy0) <= 1e-4 or (ux1 - ux0) <= 1e-4:
            continue
        y0 = max(0, min(h_lat - 1, int(round(uy0 * h_lat))))
        x0 = max(0, min(w_lat - 1, int(round(ux0 * w_lat))))
        y1 = max(y0 + 1, min(h_lat, int(round(uy1 * h_lat))))
        x1 = max(x0 + 1, min(w_lat, int(round(ux1 * w_lat))))

        if is_unreferenced_text:
            text_dn = (
                max(0.01, min(1.0, float(raw_dn)))
                if has_explicit_dn
                else max(fallback_dn, 0.65)
            )
            deferred_text_boxes.append(
                (y0, x0, y1, x1, text_dn, has_explicit_dn)
            )
            continue

        box_dn = (
            max(0.01, min(1.0, float(raw_dn))) if has_explicit_dn else fallback_dn
        )
        raw_z = item.get("z_order")
        z_val = float(raw_z) if raw_z is not None else 0.0
        parsed_boxes.append((y0, x0, y1, x1, box_dn, z_val, has_explicit_dn))

    if not parsed_boxes and not deferred_text_boxes:
        return None

    has_non_uniform_base = any(
        has_exp and abs(dn - fallback_dn) > 1e-5
        for _, _, _, _, dn, _, has_exp in parsed_boxes
    )
    has_non_uniform_text = any(
        (has_exp and abs(dn - fallback_dn) > 1e-5) or has_non_uniform_base
        for _, _, _, _, dn, has_exp in deferred_text_boxes
    )
    if return_none_if_uniform and not (has_non_uniform_base or has_non_uniform_text):
        return None

    yy = torch.arange(h_lat, device=target_dev, dtype=torch.float32).unsqueeze(1) + 0.5
    xx = torch.arange(w_lat, device=target_dev, dtype=torch.float32).unsqueeze(0) + 0.5
    raw_feather = max(0.0, float(feather_radius_lat))

    if parsed_boxes:
        z_mean = sum(b[5] for b in parsed_boxes) / float(len(parsed_boxes))
        accum_dn = torch.zeros((h_lat, w_lat), device=target_dev, dtype=torch.float32)
        accum_w = torch.zeros((h_lat, w_lat), device=target_dev, dtype=torch.float32)
        union_alpha = torch.zeros(
            (h_lat, w_lat), device=target_dev, dtype=torch.float32
        )

        for y0, x0, y1, x1, box_dn, z_val, _ in parsed_boxes:
            area_frac = max(
                1e-4, min(1.0, float((y1 - y0) * (x1 - x0)) / total_cells)
            )
            priority = math.exp(
                _HIERARCHICAL_SPLAT_GAMMA * (1.0 - area_frac)
                + _HIERARCHICAL_SPLAT_Z_GAMMA * math.tanh((z_val - z_mean) / 4.0)
            )
            dy = torch.maximum(
                torch.maximum(float(y0) - yy, yy - float(y1)), torch.zeros_like(yy)
            )
            dx = torch.maximum(
                torch.maximum(float(x0) - xx, xx - float(x1)), torch.zeros_like(xx)
            )
            dist = torch.sqrt(dy * dy + dx * dx)
            if raw_feather <= 1e-4:
                w_inside = (dist <= 1e-5).to(dtype=torch.float32)
            else:
                w_inside = 0.5 * (
                    1.0 + torch.cos(math.pi * (dist / raw_feather).clamp(0.0, 1.0))
                )
            w_hier = w_inside * priority
            accum_dn = accum_dn + w_hier * box_dn
            accum_w = accum_w + w_hier
            union_alpha = torch.maximum(union_alpha, w_inside)

        safe_w = accum_w.clamp(min=1e-6)
        fg_dn = torch.where(
            accum_w > 1e-6,
            (accum_dn / safe_w).clamp(0.01, 1.0),
            torch.full_like(accum_w, fallback_dn),
        )
        horizon_map_2d = union_alpha * fg_dn + (1.0 - union_alpha) * fallback_dn
    else:
        horizon_map_2d = torch.full(
            (h_lat, w_lat), fallback_dn, device=target_dev, dtype=torch.float32
        )

    if deferred_text_boxes:
        text_peak = horizon_map_2d.clone()
        for ty0, tx0, ty1, tx1, text_dn, has_exp_t in deferred_text_boxes:
            dy = torch.maximum(
                torch.maximum(float(ty0) - yy, yy - float(ty1)),
                torch.zeros_like(yy),
            )
            dx = torch.maximum(
                torch.maximum(float(tx0) - xx, xx - float(tx1)),
                torch.zeros_like(xx),
            )
            dist = torch.sqrt(dy * dy + dx * dx)
            if raw_feather <= 1e-4:
                t_alpha = (dist <= 1e-5).to(dtype=torch.float32)
            else:
                t_alpha = 0.5 * (
                    1.0
                    + torch.cos(math.pi * (dist / raw_feather).clamp(0.0, 1.0))
                )
            cand_dn = t_alpha * text_dn + (1.0 - t_alpha) * horizon_map_2d
            if has_exp_t:
                text_peak = t_alpha * text_dn + (1.0 - t_alpha) * text_peak
            else:
                text_peak = torch.maximum(text_peak, cand_dn)
        horizon_map_2d = text_peak

    return horizon_map_2d.unsqueeze(0).unsqueeze(0).clamp(0.01, 1.0)


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
    splat_entries: Sequence[Tuple[Any, ...]],
    h_lat: int,
    w_lat: int,
    latent_channels: int,
    feather_radius_lat: float,
    device: torch.device,
    default_denoise_strength: float = 0.75,
    return_denoise_map: bool = False,
    aperture_overrides: Optional[
        Sequence[
            Union[
                Tuple[Tuple[float, float, float, float], float],
                Tuple[Tuple[float, float, float, float], float, float],
            ]
        ]
    ] = None,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    has_base_canvas = base_canvas is not None
    if has_base_canvas:
        canvas = base_canvas.to(device=device, dtype=torch.float32).clone()
        if canvas.shape[-2:] != (h_lat, w_lat):
            canvas = resample_packed_latents(canvas, h_lat, w_lat)
    else:
        canvas = torch.zeros(
            (1, latent_channels, h_lat, w_lat), device=device, dtype=torch.float32
        )

    fallback_dn = max(0.01, min(1.0, float(default_denoise_strength)))
    if not splat_entries:
        zero_alpha = torch.zeros(
            (1, 1, h_lat, w_lat), device=device, dtype=torch.float32
        )
        if return_denoise_map:
            dn_map = torch.full(
                (1, 1, h_lat, w_lat), fallback_dn, device=device, dtype=torch.float32
            )
            return canvas, zero_alpha, dn_map
        return canvas, zero_alpha

    use_subphase = (latent_channels % 4) == 0
    sub_scale = 2 if use_subphase else 1
    c_work = (latent_channels // 4) if use_subphase else latent_channels
    h_work = h_lat * sub_scale
    w_work = w_lat * sub_scale

    canvas_work = _unpack_latents_2x2(canvas) if use_subphase else canvas

    accum_lat = torch.zeros(
        (1, c_work, h_work, w_work), device=device, dtype=torch.float32
    )
    accum_weight = torch.zeros(
        (1, 1, h_work, w_work), device=device, dtype=torch.float32
    )
    accum_rms_w1 = torch.zeros(
        (1, 1, h_work, w_work), device=device, dtype=torch.float32
    )
    accum_rms_w2 = torch.zeros(
        (1, 1, h_work, w_work), device=device, dtype=torch.float32
    )
    accum_dn = torch.zeros(
        (1, 1, h_work, w_work), device=device, dtype=torch.float32
    )
    union_alpha_work = torch.zeros(
        (1, 1, h_work, w_work), device=device, dtype=torch.float32
    )

    raw_feather = max(0.0, float(feather_radius_lat))
    total_cells = float(max(1, h_lat * w_lat))
    z_vals = [
        float(e[5]) if (len(e) > 5 and e[5] is not None) else 0.0
        for e in splat_entries
    ]
    z_mean = (sum(z_vals) / float(len(z_vals))) if z_vals else 0.0

    for entry in splat_entries:
        box_lat_cpu = entry[0]
        norm_bbox = entry[1]
        box_mask_cpu = entry[2] if len(entry) > 2 else None
        box_dn = entry[3] if len(entry) > 3 else None
        box_sharpness = entry[4] if len(entry) > 4 else None
        box_z = float(entry[5]) if (len(entry) > 5 and entry[5] is not None) else 0.0

        uy0, ux0, uy1, ux1 = norm_bbox
        y0 = max(0, min(h_lat - 1, int(round(uy0 * h_lat))))
        x0 = max(0, min(w_lat - 1, int(round(ux0 * w_lat))))
        y1 = max(y0 + 1, min(h_lat, int(round(uy1 * h_lat))))
        x1 = max(x0 + 1, min(w_lat, int(round(ux1 * w_lat))))
        bh = y1 - y0
        bw = x1 - x0

        area_frac = max(1e-4, min(1.0, float(bh * bw) / total_cells))
        priority = math.exp(
            _HIERARCHICAL_SPLAT_GAMMA * (1.0 - area_frac)
            + _HIERARCHICAL_SPLAT_Z_GAMMA * math.tanh((box_z - z_mean) / 4.0)
        )

        eff_dn = (
            max(0.01, min(1.0, float(box_dn)))
            if box_dn is not None
            else fallback_dn
        )
        eff_feather = raw_feather
        if eff_feather > 1e-4:
            if box_sharpness is not None and float(box_sharpness) > 0.0:
                sharp_val = max(0.0, float(box_sharpness))
                acutance_scale = max(
                    0.55, min(1.15, 1.12 - 0.18 * math.log1p(sharp_val))
                )
                eff_feather = eff_feather * acutance_scale
            geom_cap = max(0.25, 0.25 * float(min(bh, bw)))
            horizon_scale = math.sin(
                0.5 * math.pi * min(1.0, max(0.20, eff_dn / 0.35))
            )
            eff_feather = max(0.125, min(eff_feather, geom_cap) * horizon_scale)

        b_lat = box_lat_cpu.to(device=device, dtype=torch.float32)
        if b_lat.shape[-2:] != (bh, bw):
            b_lat = resample_packed_latents(b_lat, bh, bw)
        b_work = _unpack_latents_2x2(b_lat) if use_subphase else b_lat

        sy0, sy1 = y0 * sub_scale, y1 * sub_scale
        sx0, sx1 = x0 * sub_scale, x1 * sub_scale
        sbh = sy1 - sy0
        sbw = sx1 - sx0

        if eff_feather <= 1e-4:
            local_alpha = torch.ones(
                (1, 1, sbh, sbw), device=device, dtype=torch.float32
            )
        else:
            yy = (
                torch.arange(sbh, device=device, dtype=torch.float32).unsqueeze(1)
                + 0.5
            ) / float(sub_scale)
            xx = (
                torch.arange(sbw, device=device, dtype=torch.float32).unsqueeze(0)
                + 0.5
            ) / float(sub_scale)
            d_edge = torch.minimum(
                torch.minimum(yy, float(bh) - yy),
                torch.minimum(xx, float(bw) - xx),
            )
            edge_ramp = 0.5 * (
                1.0 - torch.cos(math.pi * (d_edge / eff_feather).clamp(0.0, 1.0))
            )
            local_alpha = edge_ramp.unsqueeze(0).unsqueeze(0)

        if box_mask_cpu is not None:
            b_mask = box_mask_cpu.to(device=device, dtype=torch.float32)
            if b_mask.ndim == 2:
                b_mask = b_mask.unsqueeze(0).unsqueeze(0)
            elif b_mask.ndim == 3:
                b_mask = b_mask.unsqueeze(0)
            if b_mask.shape[-2:] != (sbh, sbw):
                b_mask = F.interpolate(
                    b_mask, size=(sbh, sbw), mode="bilinear", align_corners=False
                )
            local_alpha = torch.minimum(local_alpha, b_mask.clamp(0.0, 1.0))

        local_alpha = local_alpha.clamp(0.0, 1.0)
        w_hier = local_alpha * priority
        b_rms = torch.sqrt(torch.mean(b_work * b_work, dim=1, keepdim=True) + 1e-12)
        w_rms = w_hier * b_rms

        accum_lat[:, :, sy0:sy1, sx0:sx1] = (
            accum_lat[:, :, sy0:sy1, sx0:sx1] + w_hier * b_work
        )
        accum_weight[:, :, sy0:sy1, sx0:sx1] = (
            accum_weight[:, :, sy0:sy1, sx0:sx1] + w_hier
        )
        accum_rms_w1[:, :, sy0:sy1, sx0:sx1] = (
            accum_rms_w1[:, :, sy0:sy1, sx0:sx1] + w_rms
        )
        accum_rms_w2[:, :, sy0:sy1, sx0:sx1] = (
            accum_rms_w2[:, :, sy0:sy1, sx0:sx1] + w_rms * w_rms
        )
        accum_dn[:, :, sy0:sy1, sx0:sx1] = (
            accum_dn[:, :, sy0:sy1, sx0:sx1] + w_hier * eff_dn
        )
        union_alpha_work[:, :, sy0:sy1, sx0:sx1] = torch.maximum(
            union_alpha_work[:, :, sy0:sy1, sx0:sx1], local_alpha
        )

    union_alpha_work = union_alpha_work.clamp(0.0, 1.0)
    safe_weight = accum_weight.clamp(min=1e-5)
    linear_fg = accum_lat / safe_weight

    target_fg_rms = accum_rms_w1 / safe_weight
    orth_fg_rms = torch.sqrt(accum_rms_w2.clamp(min=0.0)) / safe_weight
    lin_fg_rms = torch.sqrt(
        torch.mean(linear_fg * linear_fg, dim=1, keepdim=True) + 1e-12
    )
    denom_fg_rms = torch.maximum(orth_fg_rms, lin_fg_rms).clamp(min=1e-6)
    cov_fg_gain = torch.where(
        (accum_weight > 1e-5) & (target_fg_rms > 1e-5),
        (target_fg_rms / denom_fg_rms).clamp(1.0, 1.41421356),
        torch.ones_like(target_fg_rms),
    )
    norm_fg = torch.where(
        accum_weight > 1e-6, linear_fg * cov_fg_gain, canvas_work
    )

    if has_base_canvas:
        lin_comp = (
            union_alpha_work * norm_fg + (1.0 - union_alpha_work) * canvas_work
        )
        fg_rms_final = torch.sqrt(
            torch.mean(norm_fg * norm_fg, dim=1, keepdim=True) + 1e-12
        )
        bg_rms_final = torch.sqrt(
            torch.mean(canvas_work * canvas_work, dim=1, keepdim=True) + 1e-12
        )
        s1_comp = (
            union_alpha_work * fg_rms_final
            + (1.0 - union_alpha_work) * bg_rms_final
        )
        s2_comp_rms = torch.hypot(
            union_alpha_work * fg_rms_final,
            (1.0 - union_alpha_work) * bg_rms_final,
        )
        lin_comp_rms = torch.sqrt(
            torch.mean(lin_comp * lin_comp, dim=1, keepdim=True) + 1e-12
        )
        denom_comp_rms = torch.maximum(s2_comp_rms, lin_comp_rms).clamp(min=1e-6)
        cov_comp_gain = torch.where(
            (bg_rms_final > 1e-5) & (fg_rms_final > 1e-5) & (accum_weight > 1e-5),
            (s1_comp / denom_comp_rms).clamp(1.0, 1.41421356),
            torch.ones_like(s1_comp),
        )
        canvas_work = lin_comp * cov_comp_gain
    else:
        canvas_work = norm_fg

    canvas_out = (
        _pack_latents_2x2(canvas_work) if use_subphase else canvas_work.contiguous()
    )
    if use_subphase:
        union_alpha = (
            union_alpha_work.reshape(1, 1, h_lat, 2, w_lat, 2)
            .mean(dim=(3, 5))
            .clamp(0.0, 1.0)
        )
    else:
        union_alpha = union_alpha_work

    dn_map: Optional[torch.Tensor] = None
    if return_denoise_map:
        ext_dn_target = fallback_dn if has_base_canvas else 1.0
        dn_work = torch.where(
            accum_weight > 1e-6,
            (accum_dn / safe_weight).clamp(0.01, 1.0),
            torch.full_like(accum_weight, ext_dn_target),
        )
        ambient_dn = torch.maximum(
            dn_work, torch.full_like(dn_work, ext_dn_target)
        )
        t_rim = (union_alpha_work / 0.20).clamp(0.0, 1.0)
        rim_keep = torch.where(
            t_rim > 1e-8,
            torch.pow(t_rim.clamp(min=1e-8), 0.20),
            torch.zeros_like(t_rim),
        )
        dn_work = rim_keep * dn_work + (1.0 - rim_keep) * ambient_dn
        if use_subphase:
            dn_map = (
                dn_work.reshape(1, 1, h_lat, 2, w_lat, 2)
                .mean(dim=(3, 5))
                .clamp(0.01, 1.0)
            )
        else:
            dn_map = dn_work.clamp(0.01, 1.0)

    if aperture_overrides:
        yy_lat = (
            torch.arange(h_lat, device=device, dtype=torch.float32).unsqueeze(1)
            + 0.5
        )
        xx_lat = (
            torch.arange(w_lat, device=device, dtype=torch.float32).unsqueeze(0)
            + 0.5
        )
        ovr_peak = dn_map.clone() if dn_map is not None else None
        for ovr_entry in aperture_overrides:
            if len(ovr_entry) < 2:
                continue
            ovr_bbox = ovr_entry[0]
            ovr_dn = ovr_entry[1]
            carve_strength = (
                max(0.0, min(1.0, float(ovr_entry[2])))
                if (len(ovr_entry) > 2 and ovr_entry[2] is not None)
                else 1.0
            )
            if len(ovr_bbox) != 4:
                continue
            oy0 = max(0, min(h_lat - 1, int(round(float(ovr_bbox[0]) * h_lat))))
            ox0 = max(0, min(w_lat - 1, int(round(float(ovr_bbox[1]) * w_lat))))
            oy1 = max(oy0 + 1, min(h_lat, int(round(float(ovr_bbox[2]) * h_lat))))
            ox1 = max(ox0 + 1, min(w_lat, int(round(float(ovr_bbox[3]) * w_lat))))
            dy = torch.maximum(
                torch.maximum(float(oy0) - yy_lat, yy_lat - float(oy1)),
                torch.zeros_like(yy_lat),
            )
            dx = torch.maximum(
                torch.maximum(float(ox0) - xx_lat, xx_lat - float(ox1)),
                torch.zeros_like(xx_lat),
            )
            dist = torch.sqrt(dy * dy + dx * dx)
            if raw_feather <= 1e-4:
                ovr_alpha = (
                    (dist <= 1e-5)
                    .to(dtype=torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
            else:
                ovr_alpha = (
                    0.5
                    * (
                        1.0
                        + torch.cos(
                            math.pi * (dist / raw_feather).clamp(0.0, 1.0)
                        )
                    )
                ).unsqueeze(0).unsqueeze(0)
            if carve_strength > 1e-6:
                union_alpha = torch.minimum(
                    union_alpha,
                    (1.0 - ovr_alpha * carve_strength).clamp(0.0, 1.0),
                )
            if ovr_peak is not None and dn_map is not None and ovr_dn is not None:
                eff_ovr_dn = max(0.01, min(1.0, float(ovr_dn)))
                cand_dn = ovr_alpha * eff_ovr_dn + (1.0 - ovr_alpha) * dn_map
                ovr_peak = torch.maximum(ovr_peak, cand_dn)
        if ovr_peak is not None:
            dn_map = ovr_peak.clamp(0.01, 1.0)

    if return_denoise_map:
        assert dn_map is not None
        return canvas_out, union_alpha, dn_map
    return canvas_out, union_alpha


def couple_splat_velocity_prior(
    v_dit: torch.Tensor,
    latents: torch.Tensor,
    splat_canvas: torch.Tensor,
    splat_union_alpha: torch.Tensor,
    splat_denoise_map: torch.Tensor,
    sigma_curr: float,
    noise: Optional[torch.Tensor] = None,
    step_dt: float = 0.0,
) -> torch.Tensor:
    s_val = float(sigma_curr)
    if s_val <= 1e-4:
        return v_dit
    d_clamped = splat_denoise_map.to(dtype=torch.float32).clamp(0.0, 1.0)
    sigma_hand = d_clamped.clamp(0.0, 0.96)
    half_band = (0.35 * sigma_hand).clamp(0.015, 0.18)
    tau = torch.where(
        d_clamped < 1e-3,
        torch.ones_like(d_clamped),
        ((s_val - (sigma_hand - half_band)) / (2.0 * half_band)).clamp(0.0, 1.0),
    )
    psi = 0.5 * (1.0 - torch.cos(math.pi * tau))
    g_max = (1.0 - 0.38 * sigma_hand).clamp(0.35, 1.0)
    alpha_f32 = splat_union_alpha.to(dtype=torch.float32).clamp(0.0, 1.0)
    t_floor = (alpha_f32 / 1e-4).clamp(0.0, 1.0)
    floor_gate = t_floor * t_floor * (3.0 - 2.0 * t_floor)
    eff_exp = 1.0 - (0.65 * psi * floor_gate)
    alpha_eff = torch.where(
        alpha_f32 > 1e-12,
        torch.pow(alpha_f32.clamp(min=1e-12), eff_exp) * floor_gate,
        torch.zeros_like(alpha_f32),
    )
    kappa = (alpha_eff * g_max * psi).clamp(0.0, 1.0)
    dt_abs = abs(float(step_dt))
    sigma_eff = max(s_val, _CFL_SIGMA_FLOOR, _CFL_STEP_MULTIPLIER * dt_abs)
    lat_f32 = latents.float()
    splat_f32 = splat_canvas.float()
    if noise is not None and noise.shape == lat_f32.shape:
        noise_f32 = noise.float()
        z_bridge = (1.0 - s_val) * splat_f32 + s_val * noise_f32
        v_splat = (noise_f32 - splat_f32) + (lat_f32 - z_bridge) / sigma_eff
    else:
        v_splat = (lat_f32 - splat_f32) / max(s_val, 0.05)
    coupled = ((1.0 - kappa) * v_dit.float() + kappa * v_splat).to(dtype=v_dit.dtype)
    return torch.where(kappa > 1e-6, coupled, v_dit)


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
    eff_max_patches = max(64, int(max_patches or 2304))
    max_pixels = eff_max_patches * 16 * 16
    scale = math.sqrt(float(max_pixels) / float(max(1, w_in * h_in)))
    h_vis = min(2048, max(32, (int(round(h_in * scale)) // 32) * 32))
    w_vis = min(2048, max(32, (int(round(w_in * scale)) // 32) * 32))
    while (w_vis // 16) * (h_vis // 16) > eff_max_patches and w_vis > 32 and h_vis > 32:
        if w_vis >= h_vis:
            w_vis -= 32
        else:
            h_vis -= 32
    if (w_in, h_in) != (w_vis, h_vis):
        img = img.resize((w_vis, h_vis), resample=Image.Resampling.BICUBIC)
    normed = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
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
    "splat_box_masks_to_canvas",
    "build_box_denoise_horizon_map",
    "compose_latent_inpaint_mask",
    "splat_box_latents_to_canvas",
    "couple_splat_velocity_prior",
    "load_image_tensor_16px",
    "pack_vision_patches_32px",
]