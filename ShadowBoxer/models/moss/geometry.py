from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import torch
from torch.nn import functional as F

from .config import (
    _BOX_ISOLATION_LOGIT_SCALE,
    _HIERARCHICAL_AREA_GAMMA,
    _HIERARCHICAL_SEMANTIC_GAMMA,
    _HIERARCHICAL_Z_GAMMA,
    _ISOKV_COLLAB_OVERLAP_MIN,
    CarveoutSpec,
    CollabGroups,
    NormBBox,
)

_CHARACTER_LABELS = {
    "character",
    "person",
    "people",
    "human",
    "portrait",
    "face",
    "man",
    "woman",
    "boy",
    "girl",
    "child",
    "subject",
    "figure",
    "actor",
    "model",
    "hero",
}

_TEXT_LABELS = {
    "text",
    "typography",
    "caption",
    "title",
    "sign",
    "label",
    "lettering",
}

_CHAR_DOME_R0 = 0.45
_CHAR_DOME_R1 = 1.05
_CHAR_DOME_EDGE_FLOOR = 0.15

_CHAR_WIDE_DOME_R0 = 0.88
_CHAR_WIDE_DOME_R1 = 1.18
_CHAR_WIDE_DOME_EDGE_FLOOR = 0.82


def _normalize_ref_label(label: Optional[str]) -> str:
    s = str(label or "object").strip().lower()
    if s in _TEXT_LABELS:
        return "text"
    if s in _CHARACTER_LABELS:
        return "character"
    return "object"


def _semantic_label_rank(label: Optional[str]) -> float:
    norm = _normalize_ref_label(label)
    if norm == "text":
        return 2.0
    if norm == "character":
        return 1.0
    return 0.0


def _normalize_ref_labels(
    num_refs: int,
    reference_labels: Optional[Sequence[Optional[str]]],
) -> Tuple[str, ...]:
    if num_refs <= 0:
        return ()
    if not reference_labels:
        return tuple("object" for _ in range(num_refs))
    out: List[str] = []
    for idx in range(num_refs):
        raw = reference_labels[idx] if idx < len(reference_labels) else None
        out.append(_normalize_ref_label(raw))
    return tuple(out)


def _evaluate_character_tukey_dome(
    yy: torch.Tensor,
    xx: torch.Tensor,
    y0_int: int,
    x0_int: int,
    y1_int: int,
    x1_int: int,
    wide_aperture: bool = False,
) -> torch.Tensor:
    r0 = _CHAR_WIDE_DOME_R0 if wide_aperture else _CHAR_DOME_R0
    r1 = _CHAR_WIDE_DOME_R1 if wide_aperture else _CHAR_DOME_R1
    edge_floor = (
        _CHAR_WIDE_DOME_EDGE_FLOOR if wide_aperture else _CHAR_DOME_EDGE_FLOOR
    )
    cy_mid = 0.5 * float(y0_int + y1_int)
    cx_mid = 0.5 * float(x0_int + x1_int)
    ry = max(0.5, 0.5 * float(y1_int - y0_int))
    rx = max(0.5, 0.5 * float(x1_int - x0_int))
    rho = torch.hypot((yy - cy_mid) / ry, (xx - cx_mid) / rx)
    tau = ((rho - r0) / (r1 - r0)).clamp(0.0, 1.0)
    return edge_floor + (1.0 - edge_floor) * 0.5 * (
        1.0 + torch.cos(math.pi * tau)
    )


def _snap_box_to_latent_grid(
    box: Optional[NormBBox],
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> Optional[NormBBox]:
    if box is None:
        return None
    uy0, ux0, uy1, ux1 = box
    if height is not None and width is not None and height > 0 and width > 0:
        h_lat = int(height)
        w_lat = int(width)
        y0 = max(0, min(h_lat - 1, int(round(uy0 * h_lat))))
        x0 = max(0, min(w_lat - 1, int(round(ux0 * w_lat))))
        y1 = max(y0 + 1, min(h_lat, int(round(uy1 * h_lat))))
        x1 = max(x0 + 1, min(w_lat, int(round(ux1 * w_lat))))
        if y0 == 0 and x0 == 0 and y1 == h_lat and x1 == w_lat:
            return None
        return (
            round(y0 / float(h_lat), 6),
            round(x0 / float(w_lat), 6),
            round(y1 / float(h_lat), 6),
            round(x1 / float(w_lat), 6),
        )
    if uy0 <= 1e-5 and ux0 <= 1e-5 and uy1 >= 1.0 - 1e-5 and ux1 >= 1.0 - 1e-5:
        return None
    return (round(uy0, 6), round(ux0, 6), round(uy1, 6), round(ux1, 6))


def _normalize_ref_bboxes(
    ref_shapes: Optional[List[Tuple[int, int]]],
    ref_bboxes: Optional[Sequence[Optional[Sequence[float]]]],
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> Tuple[Optional[NormBBox], ...]:
    if not ref_shapes:
        return ()
    if not ref_bboxes:
        return tuple(None for _ in ref_shapes)
    out: List[Optional[NormBBox]] = []
    for idx in range(len(ref_shapes)):
        b = ref_bboxes[idx] if idx < len(ref_bboxes) else None
        if b is None or len(b) != 4:
            out.append(None)
        else:
            y0 = max(0.0, min(1.0, float(b[0])))
            x0 = max(0.0, min(1.0, float(b[1])))
            y1 = max(y0, min(1.0, float(b[2])))
            x1 = max(x0, min(1.0, float(b[3])))
            out.append(
                _snap_box_to_latent_grid(
                    (y0, x0, y1, x1), height=height, width=width
                )
            )
    return tuple(out)


def _normalize_foreground_carveouts(
    foreground_carveouts: Optional[Sequence[Any]],
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> Tuple[CarveoutSpec, ...]:
    if not foreground_carveouts:
        return ()
    out: List[CarveoutSpec] = []
    for item in foreground_carveouts:
        if item is None:
            continue
        raw_box: Any = None
        strength: float = 1.0
        z_order: Optional[float] = None
        c_label: str = "text"
        if isinstance(item, dict):
            raw_box = item.get("bbox")
            raw_s = (
                item.get("carve_strength")
                if item.get("carve_strength") is not None
                else item.get("isolation", item.get("strength", 1.0))
            )
            strength = (
                max(0.0, min(1.0, float(raw_s))) if raw_s is not None else 1.0
            )
            raw_z = item.get("z_order")
            z_order = round(float(raw_z), 4) if raw_z is not None else None
            if item.get("label") is not None or item.get("category") is not None:
                c_label = _normalize_ref_label(
                    item.get("label") or item.get("category")
                )
        elif isinstance(item, (list, tuple)):
            if len(item) == 4 and all(isinstance(x, (int, float)) for x in item):
                raw_box = item
            elif len(item) >= 1:
                raw_box = item[0]
                if len(item) >= 2 and item[1] is not None:
                    strength = max(0.0, min(1.0, float(item[1])))
                if len(item) >= 3 and item[2] is not None:
                    z_order = round(float(item[2]), 4)
                if len(item) >= 4 and item[3] is not None:
                    c_label = _normalize_ref_label(str(item[3]))
        if (
            raw_box is None
            or not isinstance(raw_box, (list, tuple))
            or len(raw_box) != 4
            or strength <= 1e-6
        ):
            continue
        y0 = max(0.0, min(1.0, float(raw_box[0])))
        x0 = max(0.0, min(1.0, float(raw_box[1])))
        y1 = max(y0, min(1.0, float(raw_box[2])))
        x1 = max(x0, min(1.0, float(raw_box[3])))
        snapped = _snap_box_to_latent_grid(
            (y0, x0, y1, x1), height=height, width=width
        )
        if snapped is not None:
            out.append((snapped, round(strength, 6), z_order, c_label))
    return tuple(out)


def _boxes_intersect(
    box_a: Optional[NormBBox],
    box_b: Optional[NormBBox],
    pad_y: float = 0.0,
    pad_x: float = 0.0,
) -> bool:
    if box_a is None or box_b is None:
        return False
    ay0, ax0, ay1, ax1 = box_a
    by0, bx0, by1, bx1 = box_b
    inter_h = min(ay1 + pad_y, by1 + pad_y) - max(ay0 - pad_y, by0 - pad_y)
    inter_w = min(ax1 + pad_x, bx1 + pad_x) - max(ax0 - pad_x, bx0 - pad_x)
    return (inter_h > 1e-5) and (inter_w > 1e-5)


def _boxes_overlap_coefficient(
    box_a: Optional[NormBBox],
    box_b: Optional[NormBBox],
) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ay0, ax0, ay1, ax1 = box_a
    by0, bx0, by1, bx1 = box_b
    inter_h = min(ay1, by1) - max(ay0, by0)
    inter_w = min(ax1, bx1) - max(ax0, bx0)
    if inter_h <= 1e-6 or inter_w <= 1e-6:
        return 0.0
    inter_area = inter_h * inter_w
    area_a = max(0.0, ay1 - ay0) * max(0.0, ax1 - ax0)
    area_b = max(0.0, by1 - by0) * max(0.0, bx1 - bx0)
    min_area = min(area_a, area_b)
    if min_area <= 1e-6:
        return 0.0
    return float(inter_area / min_area)


def _compute_collaborating_ref_groups(
    num_refs: int,
    norm_boxes: Tuple[Optional[NormBBox], ...],
    pad_y: float = 0.0,
    pad_x: float = 0.0,
    min_overlap_coeff: float = _ISOKV_COLLAB_OVERLAP_MIN,
) -> CollabGroups:
    if num_refs <= 0:
        return ()
    adj: List[Set[int]] = [{i} for i in range(num_refs)]
    for i in range(num_refs):
        box_i = norm_boxes[i] if i < len(norm_boxes) else None
        if box_i is None:
            continue
        for j in range(i + 1, num_refs):
            box_j = norm_boxes[j] if j < len(norm_boxes) else None
            if box_j is None:
                continue
            if _boxes_overlap_coefficient(box_i, box_j) >= min_overlap_coeff:
                adj[i].add(j)
                adj[j].add(i)
    return tuple(tuple(sorted(adj[i])) for i in range(num_refs))


def _normalize_box_isolation(
    norm_boxes: Tuple[Optional[NormBBox], ...],
    reference_box_isolation: Optional[Union[float, Sequence[Optional[float]]]],
) -> Tuple[float, ...]:
    if not norm_boxes or reference_box_isolation is None:
        return tuple(0.0 for _ in norm_boxes)
    if isinstance(reference_box_isolation, (int, float)):
        global_val = max(0.0, min(1.0, float(reference_box_isolation)))
        return tuple(
            round(global_val, 6) if b is not None else 0.0 for b in norm_boxes
        )
    out: List[float] = []
    for idx, b in enumerate(norm_boxes):
        if b is None or idx >= len(reference_box_isolation):
            out.append(0.0)
        else:
            raw = reference_box_isolation[idx]
            val = max(0.0, min(1.0, float(raw))) if raw is not None else 0.0
            out.append(round(val, 6))
    return tuple(out)


def _normalize_ref_role_class(role: Optional[str]) -> str:
    r = str(role or "photo_ref").strip().lower()
    if r in ("depth_map", "color_map", "structural", "control"):
        return "structural"
    return "photo_ref"


def _normalize_ref_roles(
    num_refs: int,
    reference_roles: Optional[Sequence[Optional[str]]],
) -> Tuple[str, ...]:
    if num_refs <= 0:
        return ()
    if not reference_roles:
        return tuple("photo_ref" for _ in range(num_refs))
    out: List[str] = []
    for idx in range(num_refs):
        raw = reference_roles[idx] if idx < len(reference_roles) else None
        out.append(_normalize_ref_role_class(raw))
    return tuple(out)


def _normalize_ref_z_orders(
    num_refs: int,
    reference_z_orders: Optional[Sequence[Optional[float]]],
) -> Tuple[float, ...]:
    if num_refs <= 0:
        return ()
    if not reference_z_orders:
        return tuple(0.0 for _ in range(num_refs))
    out: List[float] = []
    for idx in range(num_refs):
        raw = reference_z_orders[idx] if idx < len(reference_z_orders) else None
        val = float(raw) if raw is not None else 0.0
        out.append(round(val, 4))
    return tuple(out)


def _build_isolated_reference_mask(
    target_tokens: int,
    ref_token_lengths: List[int],
    text_tokens: int,
    device: torch.device,
    collab_groups: Optional[CollabGroups] = None,
) -> torch.Tensor:
    total_ref = sum(ref_token_lengths)
    total_seq = target_tokens + total_ref + text_tokens
    mask = torch.zeros(
        (1, 1, total_seq, total_seq), dtype=torch.bool, device=device
    )
    mask[:, :, :target_tokens, :] = True
    text_start = target_tokens + total_ref
    mask[:, :, text_start:, :] = True
    ref_spans: List[Tuple[int, int]] = []
    cursor = target_tokens
    for r_len in ref_token_lengths:
        ref_spans.append((cursor, cursor + r_len))
        cursor += r_len
    for r_idx, (c_start, c_end) in enumerate(ref_spans):
        mask[:, :, c_start:c_end, c_start:c_end] = True
        mask[:, :, c_start:c_end, text_start:] = True
        if collab_groups and r_idx < len(collab_groups):
            for peer_idx in collab_groups[r_idx]:
                if 0 <= peer_idx < len(ref_spans):
                    p_start, p_end = ref_spans[peer_idx]
                    mask[:, :, c_start:c_end, p_start:p_end] = True
    return mask


def _build_box_aperture_bias(
    height: int,
    width: int,
    ref_token_lengths: Sequence[int],
    text_tokens: int,
    norm_boxes: Tuple[Optional[NormBBox], ...],
    iso_tuple: Tuple[float, ...],
    feather_radius_lat: float,
    device: torch.device,
    dtype: torch.dtype,
    ref_shapes: Optional[Sequence[Tuple[int, int]]] = None,
    reference_masks: Optional[Sequence[Optional[torch.Tensor]]] = None,
    role_tuple: Optional[Tuple[str, ...]] = None,
    z_order_tuple: Optional[Tuple[float, ...]] = None,
    carveout_tuple: Optional[Tuple[Any, ...]] = None,
    label_tuple: Optional[Tuple[str, ...]] = None,
) -> torch.Tensor:
    target_tokens = height * width
    total_ref = sum(ref_token_lengths)
    total_seq = target_tokens + total_ref + text_tokens
    bias = torch.zeros(
        (1, 1, target_tokens, total_seq), dtype=dtype, device=device
    )

    yy = torch.arange(height, device=device, dtype=torch.float32).unsqueeze(1) + 0.5
    xx = torch.arange(width, device=device, dtype=torch.float32).unsqueeze(0) + 0.5
    y_idx_grid = torch.arange(height, device=device, dtype=torch.long)
    x_idx_grid = torch.arange(width, device=device, dtype=torch.long)
    raw_feather = max(0.0, float(feather_radius_lat))

    precomputed_carveouts: List[
        Tuple[NormBBox, float, float, Optional[float], str, torch.Tensor]
    ] = []
    if carveout_tuple:
        for c_entry in carveout_tuple:
            c_box = c_entry[0]
            c_strength = float(c_entry[1])
            c_z = c_entry[2] if len(c_entry) > 2 else None
            c_label = (
                _normalize_ref_label(c_entry[3])
                if len(c_entry) > 3 and c_entry[3] is not None
                else "text"
            )
            cy0, cx0, cy1, cx1 = c_box
            cy0_int = max(0, min(height - 1, int(round(cy0 * height))))
            cx0_int = max(0, min(width - 1, int(round(cx0 * width))))
            cy1_int = max(cy0_int + 1, min(height, int(round(cy1 * height))))
            cx1_int = max(cx0_int + 1, min(width, int(round(cx1 * width))))
            cbh = cy1_int - cy0_int
            cbw = cx1_int - cx0_int
            c_area = max(
                1e-4,
                min(1.0, (float(cbh) / height) * (float(cbw) / width)),
            )
            cdy = torch.maximum(
                torch.maximum(float(cy0_int) - yy, yy - float(cy1_int)),
                torch.zeros_like(yy),
            )
            cdx = torch.maximum(
                torch.maximum(float(cx0_int) - xx, xx - float(cx1_int)),
                torch.zeros_like(xx),
            )
            c_dist = torch.sqrt(cdy * cdy + cdx * cdx)
            c_feather = (
                min(raw_feather, max(0.25, 0.35 * float(min(cbh, cbw))))
                if raw_feather > 1e-4
                else 0.0
            )
            if c_feather <= 1e-4:
                w_c = (c_dist <= 1e-5).to(dtype=torch.float32) * c_strength
            else:
                w_c = (
                    0.5
                    * (
                        1.0
                        + torch.cos(
                            math.pi * (c_dist / c_feather).clamp(0.0, 1.0)
                        )
                    )
                    * c_strength
                )
            if c_label == "character":
                core_env = _evaluate_character_tukey_dome(
                    yy,
                    xx,
                    cy0_int,
                    cx0_int,
                    cy1_int,
                    cx1_int,
                    wide_aperture=(c_strength >= 0.80),
                )
                w_c = w_c * core_env
            precomputed_carveouts.append(
                (c_box, c_area, c_strength, c_z, c_label, w_c)
            )

    class_z_vals: Dict[str, List[float]] = {}
    for r_idx, r_len in enumerate(ref_token_lengths):
        r_box = norm_boxes[r_idx] if r_idx < len(norm_boxes) else None
        iso_val = iso_tuple[r_idx] if r_idx < len(iso_tuple) else 0.0
        if r_len > 0 and (
            (r_box is not None and iso_val > 1e-6)
            or bool(precomputed_carveouts)
        ):
            r_class = (
                role_tuple[r_idx]
                if (role_tuple is not None and r_idx < len(role_tuple))
                else "photo_ref"
            )
            z_v = (
                float(z_order_tuple[r_idx])
                if (z_order_tuple is not None and r_idx < len(z_order_tuple))
                else 0.0
            )
            class_z_vals.setdefault(r_class, []).append(z_v)
    class_z_mean: Dict[str, float] = {
        rc: (sum(zs) / float(len(zs))) if zs else 0.0
        for rc, zs in class_z_vals.items()
    }

    raw_entries: List[
        Tuple[
            int,
            int,
            int,
            Optional[NormBBox],
            float,
            float,
            float,
            float,
            float,
            str,
            str,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Optional[torch.Tensor],
        ]
    ] = []
    cursor = target_tokens
    for r_idx, r_len in enumerate(ref_token_lengths):
        r_box = norm_boxes[r_idx] if r_idx < len(norm_boxes) else None
        iso_val = iso_tuple[r_idx] if r_idx < len(iso_tuple) else 0.0
        z_val = (
            float(z_order_tuple[r_idx])
            if (z_order_tuple is not None and r_idx < len(z_order_tuple))
            else 0.0
        )
        r_label = (
            _normalize_ref_label(label_tuple[r_idx])
            if (label_tuple is not None and r_idx < len(label_tuple))
            else "object"
        )
        if r_box is not None:
            uy0, ux0, uy1, ux1 = r_box
            y0_int = max(0, min(height - 1, int(round(uy0 * height))))
            x0_int = max(0, min(width - 1, int(round(ux0 * width))))
            y1_int = max(y0_int + 1, min(height, int(round(uy1 * height))))
            x1_int = max(x0_int + 1, min(width, int(round(ux1 * width))))
            bh = y1_int - y0_int
            bw = x1_int - x0_int
            ly0 = float(y0_int)
            ly1 = float(y1_int)
            lx0 = float(x0_int)
            lx1 = float(x1_int)
            area = max(
                1e-4,
                min(1.0, ((ly1 - ly0) / height) * ((lx1 - lx0) / width)),
            )
        else:
            y0_int, x0_int, y1_int, x1_int = 0, 0, height, width
            bh, bw = height, width
            ly0, ly1, lx0, lx1 = 0.0, float(height), 0.0, float(width)
            area = 1.0

        applicable_carveouts = [
            c
            for c in precomputed_carveouts
            if (r_box is None or _boxes_intersect(r_box, c[0]))
            and (
                (c[3] is not None and c[3] > z_val + 1e-4)
                or (c[1] < area - 1e-5)
                or (c[3] is None and c[1] <= area + 1e-5)
            )
        ]

        if r_len > 0 and (
            (r_box is not None and iso_val > 1e-6)
            or bool(applicable_carveouts)
        ):
            r_class = (
                role_tuple[r_idx]
                if (role_tuple is not None and r_idx < len(role_tuple))
                else "photo_ref"
            )
            z_mean = class_z_mean.get(r_class, 0.0)
            sem_rank = _semantic_label_rank(r_label)
            priority = math.exp(
                _HIERARCHICAL_AREA_GAMMA * (1.0 - area)
                + _HIERARCHICAL_Z_GAMMA * math.tanh((z_val - z_mean) / 4.0)
                + _HIERARCHICAL_SEMANTIC_GAMMA * sem_rank
            )
            box_iso = (
                float(iso_val)
                if (r_box is not None and iso_val > 1e-6)
                else 0.0
            )
            if box_iso > 1e-6:
                dy = torch.maximum(
                    torch.maximum(ly0 - yy, yy - ly1), torch.zeros_like(yy)
                )
                dx = torch.maximum(
                    torch.maximum(lx0 - xx, xx - lx1), torch.zeros_like(xx)
                )
                dist = torch.sqrt(dy * dy + dx * dx)
                if raw_feather <= 1e-4:
                    w_box_sil = (dist <= 1e-5).to(dtype=torch.float32)
                else:
                    w_box_sil = 0.5 * (
                        1.0
                        + torch.cos(
                            math.pi * (dist / raw_feather).clamp(0.0, 1.0)
                        )
                    )
            else:
                w_box_sil = torch.ones(
                    (height, width), device=device, dtype=torch.float32
                )

            max_eff_iso = max(
                box_iso,
                max((c[2] for c in applicable_carveouts), default=0.0),
            )
            key_atten: Optional[torch.Tensor] = None
            raw_rmask = (
                reference_masks[r_idx]
                if (reference_masks is not None and r_idx < len(reference_masks))
                else None
            )
            if raw_rmask is not None:
                m_dev = raw_rmask.to(device=device, dtype=torch.float32)
                if m_dev.ndim == 2:
                    m_dev = m_dev.unsqueeze(0).unsqueeze(0)
                elif m_dev.ndim == 3:
                    m_dev = m_dev.unsqueeze(0)
                m_dev = m_dev.clamp(0.0, 1.0)
                m_box = (
                    F.interpolate(
                        m_dev, size=(bh, bw), mode="bilinear", align_corners=False
                    )
                    if m_dev.shape[-2:] != (bh, bw)
                    else m_dev
                )
                m_box_2d = m_box[0, 0]
                cy_clamped = y_idx_grid.clamp(y0_int, y1_int - 1) - y0_int
                cx_clamped = x_idx_grid.clamp(x0_int, x1_int - 1) - x0_int
                sil_ext = torch.sqrt(
                    m_box_2d[cy_clamped][:, cx_clamped].clamp(0.0, 1.0)
                )
                w_box_sil = w_box_sil * sil_ext

                if ref_shapes is not None and r_idx < len(ref_shapes):
                    rh, rw = ref_shapes[r_idx]
                else:
                    rh, rw = bh, bw
                if rh * rw == r_len:
                    m_key = (
                        F.interpolate(
                            m_dev,
                            size=(rh, rw),
                            mode="bilinear",
                            align_corners=False,
                        )
                        if m_dev.shape[-2:] != (rh, rw)
                        else m_dev
                    )
                    m_key_root = torch.sqrt(
                        m_key.clamp(0.0, 1.0).reshape(1, 1, 1, r_len)
                    )
                    key_iso = box_iso if box_iso > 1e-6 else max_eff_iso
                    key_atten = (
                        (_BOX_ISOLATION_LOGIT_SCALE * float(key_iso))
                        * (1.0 - m_key_root)
                    )

            supp_box = box_iso * (1.0 - w_box_sil)
            supp_carve = torch.zeros_like(w_box_sil)
            w_inside = w_box_sil
            for _, _, _, _, _, w_c in applicable_carveouts:
                supp_carve = torch.maximum(supp_carve, w_c)
                w_inside = w_inside * (1.0 - w_c).clamp(0.0, 1.0)

            raw_entries.append(
                (
                    r_idx,
                    cursor,
                    r_len,
                    r_box,
                    area,
                    z_val,
                    box_iso,
                    float(max_eff_iso),
                    priority,
                    r_class,
                    r_label,
                    w_box_sil,
                    w_inside,
                    torch.maximum(supp_box, supp_carve),
                    key_atten,
                )
            )
        cursor += r_len

    if not raw_entries:
        return bias

    isolated_entries: List[
        Tuple[
            int,
            int,
            int,
            float,
            float,
            str,
            torch.Tensor,
            torch.Tensor,
            Optional[torch.Tensor],
        ]
    ] = []
    num_raw = len(raw_entries)
    for i, (
        r_idx_i,
        c_start_i,
        r_len_i,
        r_box_i,
        area_i,
        z_i,
        _,
        max_eff_iso_i,
        prio_i,
        r_class_i,
        _,
        _,
        w_inside_i,
        supp_query_i,
        key_atten_i,
    ) in enumerate(raw_entries):
        eff_w_inside = w_inside_i
        eff_supp_query = supp_query_i
        eff_max_iso = max_eff_iso_i
        if num_raw > 1:
            for j, (
                _,
                _,
                _,
                r_box_j,
                area_j,
                z_j,
                box_iso_j,
                _,
                _,
                r_class_j,
                _,
                w_box_sil_j,
                _,
                _,
                _,
            ) in enumerate(raw_entries):
                if (
                    i == j
                    or r_class_j != r_class_i
                    or box_iso_j <= 1e-6
                    or r_box_j is None
                ):
                    continue
                if not (r_box_i is None or _boxes_intersect(r_box_i, r_box_j)):
                    continue
                j_dominates = (z_j > z_i + 1e-4) or (
                    abs(z_j - z_i) <= 1e-4 and area_j < area_i - 1e-5
                )
                if j_dominates:
                    w_occ = (box_iso_j * w_box_sil_j).clamp(0.0, 1.0)
                    eff_w_inside = eff_w_inside * (1.0 - w_box_sil_j).clamp(0.0, 1.0)
                    eff_supp_query = torch.maximum(eff_supp_query, w_occ)
                    eff_max_iso = max(eff_max_iso, float(box_iso_j))
        isolated_entries.append(
            (
                r_idx_i,
                c_start_i,
                r_len_i,
                eff_max_iso,
                prio_i,
                r_class_i,
                eff_w_inside,
                eff_supp_query,
                key_atten_i,
            )
        )

    if len(isolated_entries) == 1:
        (
            _,
            c_start,
            r_len,
            max_eff_iso,
            _,
            _,
            _,
            supp_query,
            key_atten,
        ) = isolated_entries[0]
        min_logit = _BOX_ISOLATION_LOGIT_SCALE * max_eff_iso
        attenuation = (
            (_BOX_ISOLATION_LOGIT_SCALE * supp_query)
            .clamp(min=min_logit, max=0.0)
            .reshape(1, 1, target_tokens, 1)
        )
        if key_atten is not None:
            block_bias = (attenuation + key_atten).clamp(min=min_logit, max=0.0)
            bias[:, :, :, c_start : c_start + r_len] = block_bias.to(dtype=dtype)
        else:
            bias[:, :, :, c_start : c_start + r_len] = attenuation.to(
                dtype=dtype
            ).expand(1, 1, target_tokens, r_len)
        return bias

    w_sum_by_class: Dict[str, torch.Tensor] = {}
    p_sum_by_class: Dict[str, torch.Tensor] = {}
    class_counts: Dict[str, int] = {}
    for _, _, _, _, priority, r_class, w_inside, _, _ in isolated_entries:
        class_counts[r_class] = class_counts.get(r_class, 0) + 1
        if r_class not in w_sum_by_class:
            w_sum_by_class[r_class] = w_inside.clone()
            p_sum_by_class[r_class] = w_inside * priority
        else:
            w_sum_by_class[r_class] = w_sum_by_class[r_class] + w_inside
            p_sum_by_class[r_class] = (
                p_sum_by_class[r_class] + w_inside * priority
            )

    overlap_gate_by_class: Dict[str, torch.Tensor] = {}
    safe_p_sum_by_class: Dict[str, torch.Tensor] = {}
    for r_class, w_s in w_sum_by_class.items():
        overlap_gate_by_class[r_class] = (w_s - 1.0).clamp(0.0, 1.0)
        safe_p_sum_by_class[r_class] = p_sum_by_class[r_class].clamp(min=1e-6)

    for (
        _,
        c_start,
        r_len,
        max_eff_iso,
        priority,
        r_class,
        w_inside,
        supp_query,
        key_atten,
    ) in isolated_entries:
        min_logit = _BOX_ISOLATION_LOGIT_SCALE * max_eff_iso
        ext_atten = _BOX_ISOLATION_LOGIT_SCALE * supp_query
        if class_counts.get(r_class, 1) > 1:
            overlap_gate = overlap_gate_by_class[r_class]
            safe_p_sum = safe_p_sum_by_class[r_class]
            share = ((w_inside * priority) / safe_p_sum).clamp(1e-4, 1.0)
            log_part = overlap_gate * max_eff_iso * torch.log(share)
            attenuation = (
                (ext_atten + log_part)
                .clamp(min=min_logit, max=0.0)
                .reshape(1, 1, target_tokens, 1)
            )
        else:
            attenuation = ext_atten.clamp(min=min_logit, max=0.0).reshape(
                1, 1, target_tokens, 1
            )
        if key_atten is not None:
            block_bias = (attenuation + key_atten).clamp(min=min_logit, max=0.0)
            bias[:, :, :, c_start : c_start + r_len] = block_bias.to(dtype=dtype)
        else:
            bias[:, :, :, c_start : c_start + r_len] = attenuation.to(
                dtype=dtype
            ).expand(1, 1, target_tokens, r_len)

    return bias


__all__ = [
    "_normalize_ref_label",
    "_semantic_label_rank",
    "_normalize_ref_labels",
    "_evaluate_character_tukey_dome",
    "_snap_box_to_latent_grid",
    "_normalize_ref_bboxes",
    "_normalize_foreground_carveouts",
    "_boxes_intersect",
    "_boxes_overlap_coefficient",
    "_compute_collaborating_ref_groups",
    "_normalize_box_isolation",
    "_normalize_ref_role_class",
    "_normalize_ref_roles",
    "_normalize_ref_z_orders",
    "_build_isolated_reference_mask",
    "_build_box_aperture_bias",
]