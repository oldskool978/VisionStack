from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

from models.modeling_moss import validate_skip_layers
from schema import (
    DEFAULT_BOX_REF_ISOLATION,
    DEFAULT_CHARACTER_CARVE_ISOLATION,
    DEFAULT_TEXT_CARVE_ISOLATION,
    bbox_to_normalized,
    extract_box_text_and_caption,
)


def _boxes_overlap_2d(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> bool:
    inter_h = min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])
    inter_w = min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])
    return (inter_h > 1e-5) and (inter_w > 1e-5)


def resolve_skip_layer_regimes(
    uncond_skip_layers: Optional[Iterable[int]],
    slg_layers: Optional[Iterable[int]],
    slg_scale: float,
    slg_start: float,
    slg_stop: float,
    step_progress: float,
    depth: int = 42,
) -> Tuple[bool, Set[int], Set[int], bool]:
    uncond_skip_set = validate_skip_layers(
        uncond_skip_layers, depth, "uncond_skip_layers"
    )
    slg_skip_set = validate_skip_layers(slg_layers, depth, "slg_layers")
    skip_active = float(slg_start) <= float(step_progress) < float(slg_stop)
    do_slg = bool(slg_skip_set and float(slg_scale or 0.0) > 0.0 and skip_active)
    eff_uncond_skip = set(uncond_skip_set) if skip_active else set()
    eff_slg_skip = set(slg_skip_set) if do_slg else set()
    return skip_active, eff_uncond_skip, eff_slg_skip, do_slg


def compute_differential_inpaint_gate(
    inpaint_mask: torch.Tensor,
    next_sigma: float,
    sigma_0: float,
    box_horizon_map: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    m = inpaint_mask.to(dtype=torch.float32).clamp(0.0, 1.0)
    s0 = max(1e-4, float(sigma_0))
    s_next = max(0.0, float(next_sigma))
    if box_horizon_map is not None:
        raw_h = box_horizon_map.to(dtype=torch.float32).clamp(0.0, 1.0)
        lo_snap = 0.955 * s0
        hi_snap = 0.980 * s0
        snap_span = max(1e-4, hi_snap - lo_snap)
        snap_t = ((raw_h - lo_snap) / snap_span).clamp(0.0, 1.0)
        snap_w = snap_t * snap_t * (3.0 - 2.0 * snap_t)
        snapped_h = raw_h + snap_w * (s0 - raw_h)
        eff_sigma_0 = torch.minimum(
            snapped_h,
            torch.full_like(m, s0),
        ).clamp(0.0, 1.0)
    else:
        eff_sigma_0 = torch.full_like(m, s0)

    sigma_release = (m * eff_sigma_0).clamp(0.0, 1.0)
    headroom = (s0 - sigma_release).clamp(min=1e-4)
    band = torch.minimum(
        (0.15 * sigma_release).clamp(min=1e-4, max=0.12),
        headroom,
    )
    tau = ((sigma_release - s_next) / band).clamp(0.0, 1.0)
    temporal_ramp = 0.5 * (1.0 - torch.cos(math.pi * tau))

    t_floor = (m / 1e-3).clamp(0.0, 1.0)
    floor_gate = t_floor * t_floor * (3.0 - 2.0 * t_floor)
    active_fg_gate = torch.where(
        (m <= 1e-7) | (s_next >= sigma_release),
        torch.zeros_like(m),
        (temporal_ramp * floor_gate).clamp(0.0, 1.0),
    )
    return active_fg_gate, sigma_release


def resolve_foreground_aperture_carveouts(
    layout_items: Sequence[Dict[str, Any]],
    height: int,
    width: int,
    coords_are_pixels: bool,
    splat_boxes: Sequence[Tuple[float, float, float, float]],
    splatted_box_indices: Set[int],
    ref_bboxes: Sequence[Optional[Tuple[float, float, float, float]]],
    ref_z_orders: Sequence[float],
    default_isolation: float = DEFAULT_BOX_REF_ISOLATION,
    splat_z_orders: Optional[Sequence[float]] = None,
) -> Tuple[
    List[Tuple[Any, ...]],
    List[Tuple[Any, ...]],
]:
    aperture_overrides: List[Tuple[Any, ...]] = []
    attn_carveouts: List[Tuple[Any, ...]] = []
    if not layout_items:
        return aperture_overrides, attn_carveouts

    eff_default_iso = max(
        DEFAULT_BOX_REF_ISOLATION, min(1.0, float(default_isolation or 0.0))
    )

    for idx, item in enumerate(layout_items):
        if (
            not isinstance(item, dict)
            or item.get("enabled", True) is False
            or idx in splatted_box_indices
        ):
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        ovr_box = bbox_to_normalized(
            bbox,
            height=height,
            width=width,
            coords_are_pixels=coords_are_pixels,
        )
        ovr_area = (ovr_box[2] - ovr_box[0]) * (ovr_box[3] - ovr_box[1])
        if ovr_area <= 1e-5:
            continue

        lbl, _, _ = extract_box_text_and_caption(item)
        role_item = str(item.get("input_role") or "").strip().lower()
        is_control_role = role_item in ("depth_map", "color_map")
        is_inp = bool(item.get("inpaint_box", False)) or (
            role_item == "inpaint_mask"
        )
        raw_dn = item.get("denoise_strength")
        has_exp_dn = raw_dn is not None and str(raw_dn).strip() != ""
        img_p = (
            item.get("image_path")
            or item.get("image")
            or item.get("vizj_path")
        )
        has_ref = isinstance(img_p, str) and bool(img_p.strip())
        raw_z = item.get("z_order")
        child_z: Optional[float] = float(raw_z) if raw_z is not None else None

        is_canvas_aperture = (
            (not has_ref and (lbl in ("text", "character") or is_inp or has_exp_dn))
            or is_inp
            or (has_exp_dn and not is_control_role)
        )
        if not is_canvas_aperture:
            continue

        if has_exp_dn:
            child_dn = max(0.0, min(1.0, float(raw_dn)))
        elif is_inp:
            child_dn = 1.0
        elif lbl == "text":
            child_dn = 0.65
        else:
            child_dn = 0.75

        if has_ref and not is_inp:
            splat_carve = 0.0
        elif lbl == "text" or is_inp:
            splat_carve = 1.0
        elif lbl == "character":
            if has_exp_dn and child_dn >= 0.85:
                splat_carve = 1.0
            elif not has_exp_dn or child_dn >= 0.35:
                splat_carve = 0.65
            else:
                splat_carve = 0.0
        else:
            splat_carve = 1.0 if child_dn >= 0.55 else 0.0

        if splat_boxes:
            overlaps_splat = False
            max_overlapping_splat_z = 0.0
            for s_idx, sb in enumerate(splat_boxes):
                if not _boxes_overlap_2d(ovr_box, sb):
                    continue
                sb_area = (sb[2] - sb[0]) * (sb[3] - sb[1])
                sb_z = (
                    float(splat_z_orders[s_idx])
                    if (splat_z_orders is not None and s_idx < len(splat_z_orders))
                    else 0.0
                )
                if (
                    lbl == "text"
                    or is_inp
                    or has_exp_dn
                    or (child_z is not None and child_z > sb_z + 1e-4)
                    or (sb_area > ovr_area + 1e-5)
                ):
                    overlaps_splat = True
                    if sb_z > max_overlapping_splat_z:
                        max_overlapping_splat_z = sb_z
            if overlaps_splat:
                eff_ovr_z = (
                    child_z
                    if child_z is not None
                    else (max_overlapping_splat_z + 1.0)
                )
                aperture_overrides.append(
                    (ovr_box, child_dn, splat_carve, lbl, has_exp_dn, eff_ovr_z)
                )

        if not has_ref and (lbl in ("text", "character") or is_inp) and ref_bboxes:
            raw_iso = item.get("isolate_box_attn")
            if raw_iso is not None and str(raw_iso).strip() != "":
                carve_iso = max(0.0, min(1.0, float(raw_iso)))
            elif lbl == "character" and not is_inp:
                carve_iso = DEFAULT_CHARACTER_CARVE_ISOLATION
            else:
                carve_iso = max(DEFAULT_TEXT_CARVE_ISOLATION, eff_default_iso)
            if carve_iso > 1e-6:
                overlaps_ref = False
                max_overlapping_ref_z = 0.0
                for r_idx, r_box in enumerate(ref_bboxes):
                    r_z = (
                        float(ref_z_orders[r_idx])
                        if r_idx < len(ref_z_orders)
                        else 0.0
                    )
                    if r_box is None:
                        if (
                            lbl == "text"
                            or (child_z is not None and child_z > r_z + 1e-4)
                            or (ovr_area < 0.98)
                        ):
                            overlaps_ref = True
                            if r_z > max_overlapping_ref_z:
                                max_overlapping_ref_z = r_z
                    elif _boxes_overlap_2d(ovr_box, r_box):
                        r_area = (r_box[2] - r_box[0]) * (r_box[3] - r_box[1])
                        if (
                            lbl == "text"
                            or (child_z is not None and child_z > r_z + 1e-4)
                            or (r_area > ovr_area + 1e-5)
                        ):
                            overlaps_ref = True
                            if r_z > max_overlapping_ref_z:
                                max_overlapping_ref_z = r_z
                if overlaps_ref:
                    if lbl in ("text", "character"):
                        eff_child_z: Optional[float] = (
                            child_z
                            if (
                                child_z is not None
                                and child_z > max_overlapping_ref_z + 1e-4
                            )
                            else (max_overlapping_ref_z + 1.0)
                        )
                    else:
                        eff_child_z = child_z
                    attn_carveouts.append(
                        (ovr_box, carve_iso, eff_child_z, lbl)
                    )

    return aperture_overrides, attn_carveouts


__all__ = [
    "_boxes_overlap_2d",
    "resolve_skip_layer_regimes",
    "compute_differential_inpaint_gate",
    "resolve_foreground_aperture_carveouts",
]