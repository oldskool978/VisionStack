#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator

SHADOWBOXER_ROOT = Path(__file__).resolve().parent
DEFAULT_PRESET_FILENAME = "default.json"
DEFAULT_BOX_REF_ISOLATION = 0.65
DEFAULT_CHARACTER_CARVE_ISOLATION = 0.55
DEFAULT_TEXT_CARVE_ISOLATION = 0.85
DEFAULT_SPECULATIVE_COT_TOKENS = 64

_LEGACY_DEFAULT_LAYOUT_SUMMARY = (
    "A wide view of a sleek brushed-steel refrigerator beside a warm walnut "
    "kitchen counter with a ceramic bowl of fresh citrus fruit and a handwritten "
    "yellow sticky note on the door."
)

SUPPORTED_SOLVERS = ["euler", "heun", "ipndm", "sde_gpu_pp", "multitree"]

SUPPORTED_MODELS = {
    "sft": "OpenMOSS-Team/MOSS-Image-SFT",
    "base": "OpenMOSS-Team/MOSS-Image-Base",
}

SUPPORTED_PROMPT_MODES = ["natural", "layout", "layout_json"]
SUPPORTED_BOX_LABELS = ["object", "character", "text"]
SUPPORTED_FRAMING_MODES = [
    "inscribed",
    "cover_crop",
    "contain",
    "stretch",
    "never_crop",
]
SUPPORTED_INPUT_ROLES = ["photo_ref", "color_map", "depth_map", "inpaint_mask"]
SUPPORTED_EDIT_FORMATS = [
    "standard",
    "moss_image_edit",
    "ai_toolkit_isolated",
    "ai_toolkit_full",
    "control_color_depth",
]
SUPPORTED_MASK_BLEND_MODES = [
    "auto",
    "union",
    "intersect",
    "paint_only",
    "boxes_only",
    "alpha_only",
]

DEFAULT_LAYOUT_ITEMS: List[Dict[str, Any]] = [
    {
        "label": "object",
        "caption": "A sleek brushed-steel refrigerator door with a small yellow sticky note",
        "bbox": [0, 0, 1024, 576],
        "enabled": True,
        "z_order": 1,
    },
    {
        "label": "text",
        "caption": "Handwritten crisp black marker text on a small yellow sticky note",
        "text": "Remember to test SFT & Base permutations @ 1024x1024",
        "bbox": [256, 112, 512, 464],
        "enabled": True,
        "z_order": 3,
    },
    {
        "label": "object",
        "caption": "A ceramic bowl of fresh citrus fruit on a warm walnut kitchen counter",
        "bbox": [448, 544, 1024, 1024],
        "enabled": True,
        "z_order": 2,
    },
]

BASELINE_ENGINE_DEFAULTS: Dict[str, Any] = {
    "model_variant": "sft",
    "prompt_mode": "natural",
    "negative_prompt": "",
    "system_prompt": None,
    "thinking_trace": None,
    "add_vision_id": True,
    "layout_coords_are_pixels": True,
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "shift": 5.0,
    "use_v2_grid": True,
    "early_solver": "heun",
    "late_solver": "ipndm",
    "handoff_threshold": 0.35,
    "guidance_scale": 3.2,
    "early_cfg": 3.2,
    "late_cfg": 2.8,
    "uncond_skip_layers": [24],
    "slg_layers": [24],
    "slg_scale": 0.0,
    "slg_start": 0.20,
    "slg_stop": 0.70,
    "eta": 0.0,
    "s_noise": 1.0,
    "max_sequence_length": 16384,
    "enable_thinking": True,
    "denoise_strength": 1.0,
    "framing_mode": "inscribed",
    "input_role": "photo_ref",
    "edit_format": "standard",
    "mask_blend_mode": "auto",
    "invert_inpaint_mask": False,
    "max_vit_patches": 2304,
    "deband_8bit": True,
    "prefer_vulkan": True,
    "mtf_boost": 0.08,
    "soft_knee_tau": 0.015,
    "mask_feather_radius": 1.25,
    "latent_clamp_sigma": 4.5,
    "apply_bayer_dither": True,
    "use_reference_cache": True,
    "preserve_canvas_outside_boxes": False,
    "box_attn_isolation": 0.0,
    "uncond_uses_references": None,
    "composite_box_images_to_init": False,
    "edit_lora_path": None,
    "edit_lora_scale": 1.0,
    "num_images_per_prompt": 1,
    "seed": 42,
    "output_path": "output_moss_master.png",
    "cpu_offload": False,
    "prepass_warmup": True,
}

_PRESET_CACHE: Dict[str, Any] = {
    "mtime": -1.0,
    "path": None,
    "has_custom_default": False,
    "defaults": dict(BASELINE_ENGINE_DEFAULTS),
}

_INTERNAL_BOX_KEYS = {
    "image_path",
    "image",
    "vizj_path",
    "framing_mode",
    "input_role",
    "denoise_strength",
    "inpaint_box",
    "isolate_box_attn",
    "paste_to_canvas",
    "z_order",
    "id",
    "category",
    "enabled",
    "reference",
    "description",
}

_UPSTREAM_OPTIONAL_ITEM_KEYS = ("tags", "palette")

_TEXT_SYNONYMS = {
    "text",
    "typography",
    "caption",
    "title",
    "sign",
    "label",
    "lettering",
}

_CHARACTER_SYNONYMS = {
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


def _snap_lattice_floor_half(val: float, mult: int) -> int:
    return int(math.ceil((float(val) / float(mult)) - 0.5 - 1e-9)) * int(mult)


def normalize_box_label(raw_label: Any, has_text: bool = False) -> str:
    s = str(raw_label or "").strip().lower()
    if s in _TEXT_SYNONYMS:
        return "text"
    if s in _CHARACTER_SYNONYMS:
        return "character"
    if s == "object":
        return "object"
    if has_text:
        return "text"
    return "object"


def extract_box_text_and_caption(item: Dict[str, Any]) -> Tuple[str, str, str]:
    raw_txt = item.get("text")
    txt_val = str(raw_txt).strip() if isinstance(raw_txt, str) else ""
    raw_cap = item.get("caption") or item.get("description")
    cap_val = str(raw_cap).strip() if isinstance(raw_cap, str) else ""
    label_val = normalize_box_label(
        item.get("label") or item.get("category"),
        has_text=bool(txt_val),
    )
    if label_val == "text":
        if not txt_val and cap_val:
            txt_val = cap_val
        if not cap_val or cap_val == txt_val:
            cap_val = "Crisp legible typography with clean kerning"
    else:
        txt_val = ""
        if not cap_val:
            cap_val = "Character subject" if label_val == "character" else "Scene object"
    return label_val, txt_val, cap_val


def normalize_layout_boxes(
    items: Sequence[Dict[str, Any]],
    height: int = 1024,
    width: int = 1024,
    coords_are_pixels: bool = True,
) -> List[Dict[str, Any]]:
    if not items:
        return []
    out: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        d = dict(item)
        label_val, txt_val, cap_val = extract_box_text_and_caption(d)
        d["label"] = label_val
        d["caption"] = cap_val
        if label_val == "text":
            d["text"] = txt_val
        else:
            d.pop("text", None)
        raw_dn = d.get("denoise_strength")
        if raw_dn is not None and str(raw_dn).strip() != "":
            try:
                d["denoise_strength"] = max(0.0, min(1.0, float(raw_dn)))
            except (ValueError, TypeError):
                d.pop("denoise_strength", None)
        else:
            d.pop("denoise_strength", None)
        raw_iso = d.get("isolate_box_attn")
        if raw_iso is not None and str(raw_iso).strip() != "":
            try:
                d["isolate_box_attn"] = max(0.0, min(1.0, float(raw_iso)))
            except (ValueError, TypeError):
                d.pop("isolate_box_attn", None)
        else:
            d.pop("isolate_box_attn", None)
        bbox = d.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and coords_are_pixels:
            py0, px0, py1, px1 = bbox_to_pixel_rect(
                bbox, height=height, width=width, coords_are_pixels=True
            )
            d["bbox"] = [py0, px0, py1, px1]
        out.append(d)
    return out


def pixel_bbox_to_abs_centralized(
    ymin: float,
    xmin: float,
    ymax: float,
    xmax: float,
    height: int,
    width: int,
) -> List[int]:
    h_int = int(height)
    w_int = int(width)
    cy = h_int // 2
    cx = w_int // 2
    max_y = h_int - cy
    max_x = w_int - cx
    eff_ymax = float(h_int) if int(round(ymax)) == h_int - 1 else float(ymax)
    eff_xmax = float(w_int) if int(round(xmax)) == w_int - 1 else float(xmax)
    return [
        max(-cy, min(max_y, int(round(ymin - cy)))),
        max(-cx, min(max_x, int(round(xmin - cx)))),
        max(-cy, min(max_y, int(round(eff_ymax - cy)))),
        max(-cx, min(max_x, int(round(eff_xmax - cx)))),
    ]


def bbox_to_pixel_rect(
    bbox: Sequence[Union[int, float]],
    height: int,
    width: int,
    coords_are_pixels: bool = True,
) -> Tuple[int, int, int, int]:
    if len(bbox) != 4:
        return (0, 0, height, width)
    y0, x0, y1, x1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    min_v = min(y0, x0, y1, x1)
    max_v = max(y0, x0, y1, x1)
    if (
        -1e-5 <= min_v
        and 0.0 < max_v <= 1.0 + 1e-5
        and (not coords_are_pixels or (int(height) >= 64 and int(width) >= 64))
    ):
        py0 = y0 * height
        px0 = x0 * width
        py1 = y1 * height
        px1 = x1 * width
    elif not coords_are_pixels or min_v < 0.0:
        cy = float(int(height) // 2)
        cx = float(int(width) // 2)
        py0 = y0 + cy
        px0 = x0 + cx
        py1 = y1 + cy
        px1 = x1 + cx
    else:
        py0, px0, py1, px1 = y0, x0, y1, x1

    ymin = max(0, min(height - 1, int(round(min(py0, py1)))))
    xmin = max(0, min(width - 1, int(round(min(px0, px1)))))
    ymax = max(ymin + 1, min(height, int(round(max(py0, py1)))))
    xmax = max(xmin + 1, min(width, int(round(max(px0, px1)))))
    return (ymin, xmin, ymax, xmax)


def bbox_to_normalized(
    bbox: Sequence[Union[int, float]],
    height: int,
    width: int,
    coords_are_pixels: bool = True,
) -> Tuple[float, float, float, float]:
    ymin, xmin, ymax, xmax = bbox_to_pixel_rect(
        bbox, height=height, width=width, coords_are_pixels=coords_are_pixels
    )
    h_denom = float(max(1, height))
    w_denom = float(max(1, width))
    return (
        max(0.0, min(1.0, ymin / h_denom)),
        max(0.0, min(1.0, xmin / w_denom)),
        max(0.0, min(1.0, ymax / h_denom)),
        max(0.0, min(1.0, xmax / w_denom)),
    )


def bbox_to_abs_centralized(
    bbox: Sequence[Union[int, float]],
    height: int,
    width: int,
    coords_are_pixels: bool = True,
) -> List[int]:
    ymin, xmin, ymax, xmax = bbox_to_pixel_rect(
        bbox, height=height, width=width, coords_are_pixels=coords_are_pixels
    )
    return pixel_bbox_to_abs_centralized(
        ymin, xmin, ymax, xmax, height=height, width=width
    )


def scale_layout_items_to_canvas(
    items: Sequence[Dict[str, Any]],
    height: int,
    width: int,
    src_height: int = 1024,
    src_width: int = 1024,
    coords_are_pixels: bool = True,
    snap_multiple: int = 16,
) -> List[Dict[str, Any]]:
    if not items:
        return []
    h = max(16, int(height))
    w = max(16, int(width))
    sh = max(16, int(src_height))
    sw = max(16, int(src_width))
    if not coords_are_pixels or (h == sh and w == sw):
        return [dict(it) for it in items if isinstance(it, dict)]

    mult = max(1, int(snap_multiple))
    scaled_items: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        d = dict(item)
        bbox = d.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            uy0, ux0, uy1, ux1 = bbox_to_normalized(
                bbox, height=sh, width=sw, coords_are_pixels=True
            )
            if mult > 1 and h >= mult and w >= mult:
                ny0 = (
                    0
                    if uy0 <= 1e-5
                    else max(0, min(h - mult, _snap_lattice_floor_half(uy0 * h, mult)))
                )
                nx0 = (
                    0
                    if ux0 <= 1e-5
                    else max(0, min(w - mult, _snap_lattice_floor_half(ux0 * w, mult)))
                )
                ny1 = (
                    h
                    if uy1 >= 1.0 - 1e-5
                    else max(ny0 + mult, min(h, _snap_lattice_floor_half(uy1 * h, mult)))
                )
                nx1 = (
                    w
                    if ux1 >= 1.0 - 1e-5
                    else max(nx0 + mult, min(w, _snap_lattice_floor_half(ux1 * w, mult)))
                )
            else:
                ny0 = max(0, min(h - 1, _snap_lattice_floor_half(uy0 * h, 1)))
                nx0 = max(0, min(w - 1, _snap_lattice_floor_half(ux0 * w, 1)))
                ny1 = max(ny0 + 1, min(h, _snap_lattice_floor_half(uy1 * h, 1)))
                nx1 = max(nx0 + 1, min(w, _snap_lattice_floor_half(ux1 * w, 1)))
            d["bbox"] = [ny0, nx0, ny1, nx1]
        scaled_items.append(d)
    return scaled_items


def canonical_box_sort_key(
    item: Dict[str, Any],
    height: int = 1024,
    width: int = 1024,
    coords_are_pixels: bool = True,
) -> Tuple[float, float, float, float, float, str, str, str, str]:
    bbox = item.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        uy0, ux0, uy1, ux1 = bbox_to_normalized(
            bbox, height=height, width=width, coords_are_pixels=coords_are_pixels
        )
    else:
        uy0, ux0, uy1, ux1 = 0.0, 0.0, 1.0, 1.0

    area = round((uy1 - uy0) * (ux1 - ux0), 6)
    uyc = round(0.5 * (uy0 + uy1), 6)
    uxc = round(0.5 * (ux0 + ux1), 6)
    lbl, txt_str, cap_str = extract_box_text_and_caption(item)
    img_p = str(
        item.get("image_path") or item.get("image") or item.get("vizj_path") or ""
    ).strip()
    return (
        -area,
        uyc,
        uxc,
        round(uy0, 6),
        round(ux0, 6),
        lbl,
        txt_str,
        cap_str,
        img_p,
    )


def get_canonical_box_order(
    items: Sequence[Dict[str, Any]],
    height: int = 1024,
    width: int = 1024,
    coords_are_pixels: bool = True,
) -> List[int]:
    keyed: List[Tuple[Tuple[Any, ...], int]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict) or item.get("enabled", True) is False:
            continue
        keyed.append(
            (
                canonical_box_sort_key(
                    item,
                    height=height,
                    width=width,
                    coords_are_pixels=coords_are_pixels,
                ),
                idx,
            )
        )
    keyed.sort(key=lambda pair: pair[0])
    return [idx for _, idx in keyed]


def get_reading_order_texts(
    items: Sequence[Dict[str, Any]],
    height: int = 1024,
    width: int = 1024,
    coords_are_pixels: bool = True,
) -> List[str]:
    keyed: List[Tuple[Tuple[float, float, float, float, int], str]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict) or item.get("enabled", True) is False:
            continue
        lbl, txt_val, _ = extract_box_text_and_caption(item)
        if lbl != "text" or not txt_val:
            continue
        bbox = item.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            uy0, ux0, uy1, ux1 = bbox_to_normalized(
                bbox, height=height, width=width, coords_are_pixels=coords_are_pixels
            )
        else:
            uy0, ux0, uy1, ux1 = 0.0, 0.0, 1.0, 1.0
        keyed.append(
            (
                (round(uy0, 6), round(ux0, 6), round(uy1, 6), round(ux1, 6), idx),
                txt_val,
            )
        )
    keyed.sort(key=lambda pair: pair[0])
    return [txt for _, txt in keyed]


def reconcile_layout_summary_quotes(
    caption: str,
    active_texts: Sequence[str],
) -> str:
    clean_cap = caption.strip()
    if not clean_cap or not active_texts:
        return clean_cap
    matches = list(re.finditer(r'"([^"\r\n]+)"', clean_cap))
    if not matches:
        return clean_cap
    if len(matches) == len(active_texts):
        parts: List[str] = []
        last_idx = 0
        for m, new_txt in zip(matches, active_texts):
            parts.append(clean_cap[last_idx : m.start()])
            parts.append(f'"{new_txt}"')
            last_idx = m.end()
        parts.append(clean_cap[last_idx:])
        return "".join(parts)
    default_quote = '"Remember to test SFT & Base permutations @ 1024x1024"'
    if default_quote in clean_cap:
        return clean_cap.replace(default_quote, f'"{active_texts[0]}"', 1)
    return clean_cap


def compile_layout_prompt(
    caption: str,
    items: Sequence[Dict[str, Any]],
    height: int = 1024,
    width: int = 1024,
    coords_are_pixels: bool = False,
    box_picture_map: Optional[Dict[int, int]] = None,
) -> str:
    normalized_items: List[Dict[str, Any]] = []
    canonical_indices = get_canonical_box_order(
        items, height=height, width=width, coords_are_pixels=coords_are_pixels
    )
    active_texts = get_reading_order_texts(
        items, height=height, width=width, coords_are_pixels=coords_are_pixels
    )

    for idx in canonical_indices:
        item = items[idx]
        raw_entry = dict(item)
        bbox = raw_entry.get("bbox")
        label_val, txt_val, cap_val = extract_box_text_and_caption(raw_entry)

        centralized_bbox: Optional[List[int]] = None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            centralized_bbox = bbox_to_abs_centralized(
                bbox,
                height=height,
                width=width,
                coords_are_pixels=coords_are_pixels,
            )

        ref_val: Optional[str] = None
        if box_picture_map and idx in box_picture_map:
            ref_val = f"Picture {box_picture_map[idx]}"
        elif isinstance(raw_entry.get("reference"), str) and raw_entry["reference"].strip():
            ref_val = raw_entry["reference"].strip()

        if ref_val is not None:
            if not cap_val:
                cap_val = ref_val
            elif ref_val.lower() not in cap_val.lower():
                cap_val = f"{ref_val}: {cap_val}"

        clean_entry: Dict[str, Any] = {"label": label_val}
        if centralized_bbox is not None:
            clean_entry["bbox"] = centralized_bbox
        if cap_val:
            clean_entry["caption"] = cap_val
        if label_val == "text" and txt_val:
            clean_entry["text"] = txt_val

        for k in _UPSTREAM_OPTIONAL_ITEM_KEYS:
            if k in raw_entry and raw_entry[k] is not None:
                v = raw_entry[k]
                if isinstance(v, str) and not v.strip():
                    continue
                if isinstance(v, (list, tuple, dict)) and len(v) == 0:
                    continue
                clean_entry[k] = v

        normalized_items.append(clean_entry)

    reconciled_caption = reconcile_layout_summary_quotes(caption, active_texts)
    layout_payload = {
        "summary": {"caption": reconciled_caption},
        "items": normalized_items,
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
    }
    return json.dumps(layout_payload, ensure_ascii=False)


def _format_live_ref_telem_suffix(
    telem_entry: Optional[Dict[str, Any]],
) -> str:
    if not telem_entry or not isinstance(telem_entry, dict):
        return ""
    parts: List[str] = []
    vae_sz = telem_entry.get("vae_size")
    if (
        isinstance(vae_sz, (list, tuple))
        and len(vae_sz) == 2
        and int(vae_sz[0]) > 0
        and int(vae_sz[1]) > 0
    ):
        parts.append(f"vae={int(vae_sz[0])}x{int(vae_sz[1])}")
    llm_tok = telem_entry.get("llm_tokens")
    if llm_tok is not None and int(llm_tok) > 0:
        parts.append(f"vit={int(llm_tok)}tok")
    if bool(telem_entry.get("has_cutout", False)):
        parts.append("cutout=True")
    sharp = telem_entry.get("sharpness")
    if sharp is not None and float(sharp) > 0.0:
        parts.append(f"sharp={float(sharp):.3f}")
    return (", " + ", ".join(parts)) if parts else ""


def compile_spatial_thinking_trace(
    req: GenerationRequest,
    conditioned_ref_telemetry: Optional[
        Union[Sequence[Optional[Dict[str, Any]]], Dict[Any, Dict[str, Any]]]
    ] = None,
) -> str:
    h = int(req.height)
    w = int(req.width)
    cy = h // 2
    cx = w // 2
    ref_specs, box_pic_map = req.resolve_visual_references()
    canonical_indices = get_canonical_box_order(
        req.layout_items,
        height=h,
        width=w,
        coords_are_pixels=req.layout_coords_are_pixels,
    )
    active_texts = get_reading_order_texts(
        req.layout_items,
        height=h,
        width=w,
        coords_are_pixels=req.layout_coords_are_pixels,
    )
    clean_ls = (req.layout_summary or "").strip()
    clean_p = (req.prompt or "").strip()
    base_summary = (
        clean_ls
        if (clean_ls and (clean_ls != _LEGACY_DEFAULT_LAYOUT_SUMMARY or not clean_p))
        else clean_p
    )
    reconciled_summary = reconcile_layout_summary_quotes(base_summary, active_texts)
    scene_intent = re.sub(r'\s*"[^"\r\n]+"', "", reconciled_summary).strip(" :;-")
    if not scene_intent:
        scene_intent = reconciled_summary

    lines: List[str] = [
        f"1. Canvas Geometry: {w}x{h} px (latent grid {w // 16}x{h // 16}; abs_centralized yxyx range [{-cy}, {-cx}, {h - cy}, {w - cx}]).",
        f"2. Scene Intent: {scene_intent}",
    ]

    if ref_specs:
        ref_parts: List[str] = []
        for s_idx, spec in enumerate(ref_specs):
            pic_idx = spec.get("picture_index")
            dit_idx = spec.get("dit_ref_index")
            role = spec.get("input_role", "photo_ref")
            n_box = spec.get("norm_bbox")
            if pic_idx is not None:
                tag = f"Picture {pic_idx}"
            elif dit_idx is not None:
                tag = f"DiT-Ref {dit_idx}"
            else:
                tag = "Mask-Ref"

            telem_entry: Optional[Dict[str, Any]] = None
            if isinstance(conditioned_ref_telemetry, dict):
                telem_entry = (
                    conditioned_ref_telemetry.get(s_idx)
                    or conditioned_ref_telemetry.get(spec.get("path"))
                )
            elif (
                isinstance(conditioned_ref_telemetry, (list, tuple))
                and s_idx < len(conditioned_ref_telemetry)
            ):
                telem_entry = conditioned_ref_telemetry[s_idx]
            telem_sfx = _format_live_ref_telem_suffix(telem_entry)

            if n_box is not None:
                c_b = bbox_to_abs_centralized(
                    n_box, height=h, width=w, coords_are_pixels=False
                )
                iso = float(spec.get("isolate_box_attn") or 0.0)
                splat = bool(spec.get("paste_to_canvas", False))
                ref_parts.append(
                    f"{tag} ({role}, bbox={c_b}, iso={iso:.2f}, splat={splat}{telem_sfx})"
                )
            else:
                ref_parts.append(
                    f"{tag} ({role}, global_frame={spec.get('framing_mode')}{telem_sfx})"
                )
        lines.append(
            f"3. Visual Conditioning: {len(ref_specs)} reference(s) [t_ref=0: {req.effective_reference_zero_timestep}, IsoKV: {req.effective_reference_isolated_kv}] -> "
            + "; ".join(ref_parts)
            + "."
        )

    if canonical_indices:
        box_descs: List[str] = []
        for order_num, idx in enumerate(canonical_indices, start=1):
            item = req.layout_items[idx]
            lbl, txt_val, cap = extract_box_text_and_caption(item)
            raw_cap = str(item.get("caption") or item.get("description") or "").strip()
            bbox = item.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                c_b = bbox_to_abs_centralized(
                    bbox,
                    height=h,
                    width=w,
                    coords_are_pixels=req.layout_coords_are_pixels,
                )
                norm_b = bbox_to_normalized(
                    bbox,
                    height=h,
                    width=w,
                    coords_are_pixels=req.layout_coords_are_pixels,
                )
                area_pct = (norm_b[2] - norm_b[0]) * (norm_b[3] - norm_b[1]) * 100.0
            else:
                c_b = [-cy, -cx, h - cy, w - cx]
                area_pct = 100.0
            ref_tag = ""
            if idx in box_pic_map:
                ref_tag = f", ref=Picture {box_pic_map[idx]}"
            elif isinstance(item.get("reference"), str) and item["reference"].strip():
                ref_tag = f", ref={item['reference'].strip()}"
            dn_raw = item.get("denoise_strength")
            dn_tag = (
                f", d={float(dn_raw):.2f}"
                if (dn_raw is not None and str(dn_raw).strip() != "")
                else ""
            )
            if lbl == "text" and txt_val:
                style_note = (
                    raw_cap
                    if (raw_cap and raw_cap != txt_val)
                    else "crisp kerning"
                )
                box_descs.append(
                    f"R{order_num}[text {c_b}, {area_pct:.1f}%{ref_tag}{dn_tag}: {style_note}]"
                )
            else:
                box_descs.append(
                    f"R{order_num}[{lbl} {c_b}, {area_pct:.1f}%{ref_tag}{dn_tag}: {cap or lbl}]"
                )
        step_idx = 4 if ref_specs else 3
        lines.append(
            f"{step_idx}. Coarse-to-Fine Spatial Hierarchy: " + "; ".join(box_descs) + "."
        )

    if active_texts:
        step_idx = len(lines) + 1
        quoted = ", ".join(f'"{t}"' for t in active_texts)
        lines.append(
            f"{step_idx}. Typography Lock: Preserve exact character glyph sequence and stroke contrast for {quoted}."
        )

    return "\n".join(lines)


def parse_layer_list(
    val: Union[List[int], Tuple[int, ...], str, int, None],
) -> List[int]:
    if val is None:
        return []
    if isinstance(val, int):
        return [val] if 0 <= val < 42 else []
    if isinstance(val, (list, tuple)):
        out = []
        for x in val:
            try:
                ix = int(x)
                if 0 <= ix < 42:
                    out.append(ix)
            except (ValueError, TypeError):
                pass
        return sorted(set(out))
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("", "none", "null", "[]", "off"):
            return []
        matches = re.findall(r"\b(\d+)\b", s)
        return sorted({int(m) for m in matches if 0 <= int(m) < 42})
    return []


def harvest_engine_preset(data: Dict[str, Any]) -> Dict[str, Any]:
    if "engine_defaults" in data and isinstance(data["engine_defaults"], dict):
        data = data["engine_defaults"]
    harvested: Dict[str, Any] = {}
    if "model_variant" in data and isinstance(data["model_variant"], str):
        mv = data["model_variant"].strip().lower()
        if mv in SUPPORTED_MODELS:
            harvested["model_variant"] = mv
    if "prompt_mode" in data and isinstance(data["prompt_mode"], str):
        pm = data["prompt_mode"].strip().lower()
        if pm in SUPPORTED_PROMPT_MODES:
            harvested["prompt_mode"] = pm
    if "negative_prompt" in data and isinstance(data["negative_prompt"], str):
        harvested["negative_prompt"] = data["negative_prompt"]
    for str_opt_key in ("system_prompt", "thinking_trace"):
        if str_opt_key in data:
            val = data[str_opt_key]
            harvested[str_opt_key] = (
                str(val) if (isinstance(val, str) and val.strip()) else None
            )
    if (
        "layout_coords_are_pixels" in data
        and data["layout_coords_are_pixels"] is not None
    ):
        harvested["layout_coords_are_pixels"] = bool(data["layout_coords_are_pixels"])
    for dim_key in ("height", "width"):
        if dim_key in data and data[dim_key] is not None:
            try:
                v = max(256, min(2048, int(data[dim_key])))
                harvested[dim_key] = (v // 16) * 16
            except (ValueError, TypeError):
                pass
    if "num_inference_steps" in data and data["num_inference_steps"] is not None:
        try:
            harvested["num_inference_steps"] = max(
                1, min(200, int(data["num_inference_steps"]))
            )
        except (ValueError, TypeError):
            pass
    if "shift" in data and data["shift"] is not None:
        try:
            harvested["shift"] = max(0.1, min(20.0, float(data["shift"])))
        except (ValueError, TypeError):
            pass
    if "use_v2_grid" in data and data["use_v2_grid"] is not None:
        harvested["use_v2_grid"] = bool(data["use_v2_grid"])
    for s_key in ("early_solver", "late_solver"):
        if s_key in data and isinstance(data[s_key], str):
            slv = data[s_key].strip().lower()
            if slv in SUPPORTED_SOLVERS:
                harvested[s_key] = slv
    if "handoff_threshold" in data and data["handoff_threshold"] is not None:
        try:
            harvested["handoff_threshold"] = max(
                0.0, min(1.0, float(data["handoff_threshold"]))
            )
        except (ValueError, TypeError):
            pass
    for cfg_key in ("guidance_scale", "early_cfg", "late_cfg"):
        if cfg_key in data and data[cfg_key] is not None:
            try:
                harvested[cfg_key] = max(0.0, min(30.0, float(data[cfg_key])))
            except (ValueError, TypeError):
                pass
    if "guidance_scale" in harvested and "early_cfg" not in harvested:
        harvested["early_cfg"] = harvested["guidance_scale"]
    if "early_cfg" in harvested and "guidance_scale" not in harvested:
        harvested["guidance_scale"] = harvested["early_cfg"]
    if "uncond_skip_layers" in data:
        harvested["uncond_skip_layers"] = parse_layer_list(data["uncond_skip_layers"])
    if "slg_layers" in data:
        harvested["slg_layers"] = parse_layer_list(data["slg_layers"])
    if "slg_scale" in data and data["slg_scale"] is not None:
        try:
            harvested["slg_scale"] = max(0.0, min(20.0, float(data["slg_scale"])))
        except (ValueError, TypeError):
            pass
    for b_key in ("slg_start", "slg_stop", "eta", "box_attn_isolation"):
        if b_key in data and data[b_key] is not None:
            try:
                harvested[b_key] = max(0.0, min(1.0, float(data[b_key])))
            except (ValueError, TypeError):
                pass
    if "s_noise" in data and data["s_noise"] is not None:
        try:
            harvested["s_noise"] = max(0.0, min(5.0, float(data["s_noise"])))
        except (ValueError, TypeError):
            pass
    if "denoise_strength" in data and data["denoise_strength"] is not None:
        try:
            harvested["denoise_strength"] = max(
                0.01, min(1.0, float(data["denoise_strength"]))
            )
        except (ValueError, TypeError):
            pass
    if "max_sequence_length" in data and data["max_sequence_length"] is not None:
        try:
            harvested["max_sequence_length"] = max(
                64, min(32768, int(data["max_sequence_length"]))
            )
        except (ValueError, TypeError):
            pass
    if "framing_mode" in data and isinstance(data["framing_mode"], str):
        fm = data["framing_mode"].strip().lower()
        if fm in SUPPORTED_FRAMING_MODES:
            harvested["framing_mode"] = fm
    if "input_role" in data and isinstance(data["input_role"], str):
        ir = data["input_role"].strip().lower()
        if ir in SUPPORTED_INPUT_ROLES:
            harvested["input_role"] = ir
    if "edit_format" in data and isinstance(data["edit_format"], str):
        ef = data["edit_format"].strip().lower()
        if ef in SUPPORTED_EDIT_FORMATS:
            harvested["edit_format"] = ef
    if "mask_blend_mode" in data and isinstance(data["mask_blend_mode"], str):
        mb = data["mask_blend_mode"].strip().lower()
        if mb in SUPPORTED_MASK_BLEND_MODES:
            harvested["mask_blend_mode"] = mb
    if "max_vit_patches" in data and data["max_vit_patches"] is not None:
        try:
            harvested["max_vit_patches"] = max(
                64, min(9216, int(data["max_vit_patches"]))
            )
        except (ValueError, TypeError):
            pass
    for flt_key, lo, hi in (
        ("mtf_boost", 0.0, 0.5),
        ("soft_knee_tau", 0.001, 0.25),
        ("mask_feather_radius", 0.0, 6.0),
        ("latent_clamp_sigma", 2.0, 12.0),
        ("edit_lora_scale", 0.0, 4.0),
    ):
        if flt_key in data and data[flt_key] is not None:
            try:
                harvested[flt_key] = max(lo, min(hi, float(data[flt_key])))
            except (ValueError, TypeError):
                pass
    if "edit_lora_path" in data:
        val = data["edit_lora_path"]
        harvested["edit_lora_path"] = (
            str(val).strip() if (isinstance(val, str) and val.strip()) else None
        )
    if "uncond_uses_references" in data:
        val = data["uncond_uses_references"]
        harvested["uncond_uses_references"] = None if val is None else bool(val)
    for flag_key in (
        "enable_thinking",
        "add_vision_id",
        "cpu_offload",
        "prepass_warmup",
        "deband_8bit",
        "prefer_vulkan",
        "apply_bayer_dither",
        "use_reference_cache",
        "preserve_canvas_outside_boxes",
        "composite_box_images_to_init",
        "invert_inpaint_mask",
    ):
        if flag_key in data and data[flag_key] is not None:
            harvested[flag_key] = bool(data[flag_key])
    return harvested


def locate_default_preset_file() -> Optional[Path]:
    candidates = [
        SHADOWBOXER_ROOT / DEFAULT_PRESET_FILENAME,
        SHADOWBOXER_ROOT / "presets" / DEFAULT_PRESET_FILENAME,
        SHADOWBOXER_ROOT.parent / "config" / DEFAULT_PRESET_FILENAME,
        SHADOWBOXER_ROOT.parent / "webui" / "config" / DEFAULT_PRESET_FILENAME,
    ]
    env_preset = os.environ.get("SHADOWBOXER_DEFAULT_JSON")
    if env_preset:
        candidates.insert(0, Path(env_preset).resolve())
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def get_active_engine_defaults() -> Dict[str, Any]:
    global _PRESET_CACHE
    preset_file = locate_default_preset_file()
    if preset_file is None:
        if _PRESET_CACHE["has_custom_default"]:
            _PRESET_CACHE["has_custom_default"] = False
            _PRESET_CACHE["mtime"] = -1.0
            _PRESET_CACHE["path"] = None
            _PRESET_CACHE["defaults"] = dict(BASELINE_ENGINE_DEFAULTS)
        return dict(_PRESET_CACHE["defaults"])

    try:
        current_mtime = preset_file.stat().st_mtime
    except OSError:
        return dict(_PRESET_CACHE["defaults"])

    if (
        _PRESET_CACHE["path"] == preset_file
        and _PRESET_CACHE["mtime"] == current_mtime
    ):
        return dict(_PRESET_CACHE["defaults"])

    try:
        with open(preset_file, "r", encoding="utf-8") as f:
            raw_payload = json.load(f)
        if isinstance(raw_payload, dict):
            extracted = harvest_engine_preset(raw_payload)
            resolved = dict(BASELINE_ENGINE_DEFAULTS)
            resolved.update(extracted)
            _PRESET_CACHE["mtime"] = current_mtime
            _PRESET_CACHE["path"] = preset_file
            _PRESET_CACHE["has_custom_default"] = True
            _PRESET_CACHE["defaults"] = resolved
            return dict(resolved)
    except Exception:
        pass
    return dict(_PRESET_CACHE["defaults"])


def has_custom_default_preset() -> bool:
    get_active_engine_defaults()
    return bool(_PRESET_CACHE["has_custom_default"])


class GenerationRequest(BaseModel):
    model_variant: str = Field(default="sft")
    repo_id: Optional[str] = Field(default=None)

    prompt: str = Field(default="")
    negative_prompt: str = Field(default="")
    system_prompt: Optional[str] = Field(default=None)
    thinking_trace: Optional[str] = Field(default=None)
    add_vision_id: Optional[bool] = Field(default=None)
    prompt_mode: str = Field(default="natural")
    layout_summary: Optional[str] = Field(default=None)
    layout_items: List[Dict[str, Any]] = Field(default_factory=list)
    layout_coords_are_pixels: bool = Field(default=True)
    preserve_canvas_outside_boxes: Optional[bool] = Field(default=None)
    box_attn_isolation: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    composite_box_images_to_init: Optional[bool] = Field(default=None)

    height: int = Field(default=1024, ge=256, le=2048)
    width: int = Field(default=1024, ge=256, le=2048)

    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=200)
    shift: Optional[float] = Field(default=None, ge=0.1, le=20.0)
    use_v2_grid: Optional[bool] = Field(default=None)
    early_solver: Optional[str] = Field(default=None)
    late_solver: Optional[str] = Field(default=None)
    handoff_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    guidance_scale: Optional[float] = Field(default=None, ge=0.0, le=30.0)
    early_cfg: Optional[float] = Field(default=None, ge=0.0, le=30.0)
    late_cfg: Optional[float] = Field(default=None, ge=0.0, le=30.0)

    uncond_skip_layers: Optional[List[int]] = Field(default=None)
    slg_layers: Optional[List[int]] = Field(default=None)
    slg_scale: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    slg_start: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    slg_stop: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    eta: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    s_noise: Optional[float] = Field(default=None, ge=0.0, le=5.0)

    max_sequence_length: Optional[int] = Field(default=None, ge=64, le=32768)
    enable_thinking: Optional[bool] = Field(default=None)

    init_image_path: Optional[str] = Field(default=None)
    denoise_strength: float = Field(default=1.0, ge=0.01, le=1.0)
    inpaint_mask_path: Optional[str] = Field(default=None)
    mask_blend_mode: Optional[str] = Field(default=None)
    invert_inpaint_mask: Optional[bool] = Field(default=None)
    control_image_path: Optional[str] = Field(default=None)
    reference_image_paths: List[str] = Field(default_factory=list)

    framing_mode: Optional[str] = Field(default=None)
    input_role: Optional[str] = Field(default=None)
    edit_format: Optional[str] = Field(default=None)
    max_vit_patches: Optional[int] = Field(default=None, ge=64, le=9216)
    deband_8bit: Optional[bool] = Field(default=None)
    prefer_vulkan: Optional[bool] = Field(default=None)
    mtf_boost: Optional[float] = Field(default=None, ge=0.0, le=0.5)
    soft_knee_tau: Optional[float] = Field(default=None, ge=0.001, le=0.25)
    mask_feather_radius: Optional[float] = Field(default=None, ge=0.0, le=6.0)
    latent_clamp_sigma: Optional[float] = Field(default=None, ge=2.0, le=12.0)
    apply_bayer_dither: Optional[bool] = Field(default=None)
    use_reference_cache: Optional[bool] = Field(default=None)

    reference_zero_timestep: Optional[bool] = Field(default=None)
    reference_isolated_kv: Optional[bool] = Field(default=None)
    uncond_uses_references: Optional[bool] = Field(default=None)

    edit_lora_path: Optional[str] = Field(default=None)
    edit_lora_scale: float = Field(default=1.0, ge=0.0, le=4.0)

    export_vizj_path: Optional[str] = Field(default=None)
    export_vizj_tier: int = Field(default=2, ge=0, le=2)

    num_images_per_prompt: int = Field(default=1, ge=1, le=8)
    seed: Optional[int] = Field(default=42)
    output_path: str = Field(default="output_moss_master.png")
    device: str = Field(default="cuda")
    cpu_offload: Optional[bool] = Field(default=None)
    prepass_warmup: Optional[bool] = Field(default=None)

    @field_validator("uncond_skip_layers", "slg_layers", mode="before")
    @classmethod
    def _coerce_layers(cls, v: Any) -> Optional[List[int]]:
        if v is None:
            return None
        return parse_layer_list(v)

    @field_validator("height", "width", mode="before")
    @classmethod
    def _coerce_dimensions(cls, v: Any) -> int:
        val = max(256, min(2048, int(v)))
        return (val // 16) * 16

    @model_validator(mode="after")
    def _sync_defaults(self) -> GenerationRequest:
        active = get_active_engine_defaults()
        use_custom = has_custom_default_preset()

        if self.system_prompt is None and active.get("system_prompt") is not None:
            self.system_prompt = str(active["system_prompt"])
        if self.thinking_trace is None and active.get("thinking_trace") is not None:
            self.thinking_trace = str(active["thinking_trace"])
        if self.add_vision_id is None or (
            use_custom and self.add_vision_id == BASELINE_ENGINE_DEFAULTS["add_vision_id"]
        ):
            self.add_vision_id = bool(active.get("add_vision_id", True))

        if self.num_inference_steps is None or (
            use_custom
            and self.num_inference_steps
            == BASELINE_ENGINE_DEFAULTS["num_inference_steps"]
        ):
            self.num_inference_steps = int(active["num_inference_steps"])

        if self.shift is None or (
            use_custom and self.shift == BASELINE_ENGINE_DEFAULTS["shift"]
        ):
            self.shift = float(active["shift"])

        if self.use_v2_grid is None or (
            use_custom and self.use_v2_grid == BASELINE_ENGINE_DEFAULTS["use_v2_grid"]
        ):
            self.use_v2_grid = bool(active["use_v2_grid"])

        if self.early_solver is None or (
            use_custom
            and self.early_solver == BASELINE_ENGINE_DEFAULTS["early_solver"]
        ):
            self.early_solver = str(active["early_solver"])

        if self.late_solver is None or (
            use_custom and self.late_solver == BASELINE_ENGINE_DEFAULTS["late_solver"]
        ):
            self.late_solver = str(active["late_solver"])

        if self.handoff_threshold is None or (
            use_custom
            and self.handoff_threshold
            == BASELINE_ENGINE_DEFAULTS["handoff_threshold"]
        ):
            self.handoff_threshold = float(active["handoff_threshold"])

        explicit_gs = self.guidance_scale
        if self.early_cfg is None or (
            use_custom and self.early_cfg == BASELINE_ENGINE_DEFAULTS["early_cfg"]
        ):
            self.early_cfg = (
                float(explicit_gs)
                if explicit_gs is not None
                else float(active["early_cfg"])
            )
        if self.late_cfg is None or (
            use_custom and self.late_cfg == BASELINE_ENGINE_DEFAULTS["late_cfg"]
        ):
            self.late_cfg = (
                float(explicit_gs)
                if explicit_gs is not None
                else float(active["late_cfg"])
            )
        if self.guidance_scale is None:
            self.guidance_scale = float(self.early_cfg)

        if self.uncond_skip_layers is None or (
            use_custom
            and self.uncond_skip_layers
            == BASELINE_ENGINE_DEFAULTS["uncond_skip_layers"]
        ):
            self.uncond_skip_layers = list(active["uncond_skip_layers"])

        if self.slg_layers is None or (
            use_custom and self.slg_layers == BASELINE_ENGINE_DEFAULTS["slg_layers"]
        ):
            self.slg_layers = list(active["slg_layers"])

        if self.slg_scale is None or (
            use_custom and self.slg_scale == BASELINE_ENGINE_DEFAULTS["slg_scale"]
        ):
            self.slg_scale = float(active["slg_scale"])

        if self.slg_start is None or (
            use_custom and self.slg_start == BASELINE_ENGINE_DEFAULTS["slg_start"]
        ):
            self.slg_start = float(active["slg_start"])

        if self.slg_stop is None or (
            use_custom and self.slg_stop == BASELINE_ENGINE_DEFAULTS["slg_stop"]
        ):
            self.slg_stop = float(active["slg_stop"])

        if self.eta is None or (
            use_custom and self.eta == BASELINE_ENGINE_DEFAULTS["eta"]
        ):
            self.eta = float(active["eta"])

        if self.s_noise is None or (
            use_custom and self.s_noise == BASELINE_ENGINE_DEFAULTS["s_noise"]
        ):
            self.s_noise = float(active["s_noise"])

        if self.max_sequence_length is None or (
            use_custom
            and self.max_sequence_length
            == BASELINE_ENGINE_DEFAULTS["max_sequence_length"]
        ):
            self.max_sequence_length = int(active["max_sequence_length"])

        if self.enable_thinking is None or (
            use_custom
            and self.enable_thinking == BASELINE_ENGINE_DEFAULTS["enable_thinking"]
        ):
            self.enable_thinking = bool(active["enable_thinking"])

        if self.framing_mode is None or (
            use_custom
            and self.framing_mode == BASELINE_ENGINE_DEFAULTS["framing_mode"]
        ):
            self.framing_mode = str(active["framing_mode"])

        if self.input_role is None or (
            use_custom and self.input_role == BASELINE_ENGINE_DEFAULTS["input_role"]
        ):
            self.input_role = str(active["input_role"])

        if self.edit_format is None or (
            use_custom and self.edit_format == BASELINE_ENGINE_DEFAULTS["edit_format"]
        ):
            self.edit_format = str(active["edit_format"])

        if self.mask_blend_mode is None or (
            use_custom
            and self.mask_blend_mode == BASELINE_ENGINE_DEFAULTS["mask_blend_mode"]
        ):
            self.mask_blend_mode = str(active.get("mask_blend_mode", "auto"))

        if self.invert_inpaint_mask is None or (
            use_custom
            and self.invert_inpaint_mask
            == BASELINE_ENGINE_DEFAULTS["invert_inpaint_mask"]
        ):
            self.invert_inpaint_mask = bool(
                active.get("invert_inpaint_mask", False)
            )

        if self.max_vit_patches is None or (
            use_custom
            and self.max_vit_patches == BASELINE_ENGINE_DEFAULTS["max_vit_patches"]
        ):
            self.max_vit_patches = int(active["max_vit_patches"])

        if self.deband_8bit is None or (
            use_custom and self.deband_8bit == BASELINE_ENGINE_DEFAULTS["deband_8bit"]
        ):
            self.deband_8bit = bool(active["deband_8bit"])

        if self.prefer_vulkan is None or (
            use_custom
            and self.prefer_vulkan == BASELINE_ENGINE_DEFAULTS["prefer_vulkan"]
        ):
            self.prefer_vulkan = bool(active["prefer_vulkan"])

        if self.mtf_boost is None or (
            use_custom and self.mtf_boost == BASELINE_ENGINE_DEFAULTS["mtf_boost"]
        ):
            self.mtf_boost = float(active["mtf_boost"])

        if self.soft_knee_tau is None or (
            use_custom
            and self.soft_knee_tau == BASELINE_ENGINE_DEFAULTS["soft_knee_tau"]
        ):
            self.soft_knee_tau = float(active["soft_knee_tau"])

        if self.mask_feather_radius is None or (
            use_custom
            and self.mask_feather_radius
            == BASELINE_ENGINE_DEFAULTS["mask_feather_radius"]
        ):
            self.mask_feather_radius = float(active["mask_feather_radius"])

        if self.latent_clamp_sigma is None or (
            use_custom
            and self.latent_clamp_sigma
            == BASELINE_ENGINE_DEFAULTS["latent_clamp_sigma"]
        ):
            self.latent_clamp_sigma = float(active["latent_clamp_sigma"])

        if self.apply_bayer_dither is None or (
            use_custom
            and self.apply_bayer_dither
            == BASELINE_ENGINE_DEFAULTS["apply_bayer_dither"]
        ):
            self.apply_bayer_dither = bool(active["apply_bayer_dither"])

        if self.use_reference_cache is None or (
            use_custom
            and self.use_reference_cache
            == BASELINE_ENGINE_DEFAULTS["use_reference_cache"]
        ):
            self.use_reference_cache = bool(active["use_reference_cache"])

        if self.preserve_canvas_outside_boxes is None or (
            use_custom
            and self.preserve_canvas_outside_boxes
            == BASELINE_ENGINE_DEFAULTS["preserve_canvas_outside_boxes"]
        ):
            self.preserve_canvas_outside_boxes = bool(
                active.get("preserve_canvas_outside_boxes", False)
            )

        if self.box_attn_isolation is None or (
            use_custom
            and self.box_attn_isolation
            == BASELINE_ENGINE_DEFAULTS["box_attn_isolation"]
        ):
            self.box_attn_isolation = float(active.get("box_attn_isolation", 0.0))

        if self.composite_box_images_to_init is None or (
            use_custom
            and self.composite_box_images_to_init
            == BASELINE_ENGINE_DEFAULTS["composite_box_images_to_init"]
        ):
            self.composite_box_images_to_init = bool(
                active.get("composite_box_images_to_init", False)
            )

        if (
            self.uncond_uses_references is None
            and active.get("uncond_uses_references") is not None
        ):
            self.uncond_uses_references = bool(active["uncond_uses_references"])

        if self.cpu_offload is None or (
            use_custom and self.cpu_offload == BASELINE_ENGINE_DEFAULTS["cpu_offload"]
        ):
            self.cpu_offload = bool(active["cpu_offload"])

        if self.prepass_warmup is None or (
            use_custom
            and self.prepass_warmup == BASELINE_ENGINE_DEFAULTS["prepass_warmup"]
        ):
            self.prepass_warmup = bool(active["prepass_warmup"])

        return self

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GenerationRequest:
        return cls.model_validate(data)

    def has_box_images(self) -> bool:
        for item in self.layout_items:
            if isinstance(item, dict) and item.get("enabled", True) is not False:
                img_p = (
                    item.get("image_path")
                    or item.get("image")
                    or item.get("vizj_path")
                )
                if isinstance(img_p, str) and img_p.strip():
                    return True
        return False

    def resolve_visual_references(
        self,
    ) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
        specs: List[Dict[str, Any]] = []
        box_pic_map: Dict[int, int] = {}
        dit_ref_counter = 0
        vit_pic_counter = 0

        global_role = str(self.input_role or "photo_ref").lower()
        if global_role not in SUPPORTED_INPUT_ROLES:
            global_role = "photo_ref"
        global_framing = str(self.framing_mode or "inscribed").lower()
        if global_framing not in SUPPORTED_FRAMING_MODES:
            global_framing = "inscribed"

        global_emits_vit = (global_role == "photo_ref") and (
            self.edit_format != "moss_image_edit"
        )
        global_emits_dit = global_role != "inpaint_mask"
        global_emits_mask = global_role == "inpaint_mask"

        for p_str in self.reference_image_paths:
            if isinstance(p_str, str) and p_str.strip():
                dit_idx: Optional[int] = None
                if global_emits_dit:
                    dit_ref_counter += 1
                    dit_idx = dit_ref_counter
                pic_idx: Optional[int] = None
                if global_emits_vit:
                    vit_pic_counter += 1
                    pic_idx = vit_pic_counter
                specs.append(
                    {
                        "path": p_str.strip(),
                        "norm_bbox": None,
                        "pixel_bbox": None,
                        "framing_mode": global_framing,
                        "input_role": global_role,
                        "box_label": "object",
                        "box_index": None,
                        "picture_index": pic_idx,
                        "dit_ref_index": dit_idx,
                        "emits_vit": global_emits_vit,
                        "emits_dit_latents": global_emits_dit,
                        "emits_box_mask": global_emits_mask,
                        "isolate_box_attn": 0.0,
                        "paste_to_canvas": False,
                        "denoise_strength": None,
                        "z_order": 0.0,
                    }
                )

        canonical_indices = get_canonical_box_order(
            self.layout_items,
            height=self.height,
            width=self.width,
            coords_are_pixels=self.layout_coords_are_pixels,
        )

        for idx in canonical_indices:
            item = self.layout_items[idx]
            img_p = (
                item.get("image_path")
                or item.get("image")
                or item.get("vizj_path")
            )
            if not isinstance(img_p, str) or not img_p.strip():
                continue
            clean_p = img_p.strip()
            label_val, _, _ = extract_box_text_and_caption(item)
            bbox = item.get("bbox")
            norm_box: Optional[Tuple[float, float, float, float]] = None
            px_box: Optional[Tuple[int, int, int, int]] = None
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                norm_box = bbox_to_normalized(
                    bbox,
                    height=self.height,
                    width=self.width,
                    coords_are_pixels=self.layout_coords_are_pixels,
                )
                px_box = bbox_to_pixel_rect(
                    bbox,
                    height=self.height,
                    width=self.width,
                    coords_are_pixels=self.layout_coords_are_pixels,
                )
            raw_box_fm = item.get("framing_mode")
            if (
                isinstance(raw_box_fm, str)
                and raw_box_fm.strip().lower() in SUPPORTED_FRAMING_MODES
            ):
                fm = raw_box_fm.strip().lower()
            elif norm_box is not None and global_framing == "never_crop":
                fm = "inscribed"
            else:
                fm = global_framing

            role = str(
                item.get("input_role") or self.input_role or "photo_ref"
            ).lower()
            if role not in SUPPORTED_INPUT_ROLES:
                role = global_role

            emits_vit = role == "photo_ref"
            emits_dit = role != "inpaint_mask"
            emits_box_mask = role == "inpaint_mask"

            dit_idx = None
            if emits_dit:
                dit_ref_counter += 1
                dit_idx = dit_ref_counter

            pic_idx = None
            if emits_vit:
                vit_pic_counter += 1
                pic_idx = vit_pic_counter
                box_pic_map[idx] = pic_idx

            raw_iso = item.get("isolate_box_attn")
            has_explicit_box_iso = raw_iso is not None and str(raw_iso).strip() != ""
            if not emits_dit:
                box_iso = 0.0
            elif has_explicit_box_iso:
                box_iso = max(0.0, min(1.0, float(raw_iso)))
            elif norm_box is not None:
                global_iso = max(0.0, min(1.0, float(self.box_attn_isolation or 0.0)))
                if global_iso > 1e-6:
                    box_iso = global_iso
                else:
                    box_area = (norm_box[2] - norm_box[0]) * (
                        norm_box[3] - norm_box[1]
                    )
                    box_iso = (
                        DEFAULT_BOX_REF_ISOLATION if box_area < 0.98 else 0.0
                    )
            else:
                box_iso = 0.0

            raw_paste = item.get("paste_to_canvas")
            if not emits_dit:
                paste_flag = False
            elif raw_paste is not None:
                paste_flag = bool(raw_paste) and (norm_box is not None)
            else:
                paste_flag = (
                    bool(self.composite_box_images_to_init)
                    and (norm_box is not None)
                    and (role == "photo_ref")
                )

            raw_dn = item.get("denoise_strength")
            box_dn = (
                max(0.0, min(1.0, float(raw_dn)))
                if (raw_dn is not None and str(raw_dn).strip() != "")
                else None
            )
            raw_z = item.get("z_order")
            box_z = float(raw_z) if raw_z is not None else 0.0

            specs.append(
                {
                    "path": clean_p,
                    "norm_bbox": norm_box,
                    "pixel_bbox": px_box,
                    "framing_mode": fm,
                    "input_role": role,
                    "box_label": label_val,
                    "box_index": idx,
                    "picture_index": pic_idx,
                    "dit_ref_index": dit_idx,
                    "emits_vit": emits_vit,
                    "emits_dit_latents": emits_dit,
                    "emits_box_mask": emits_box_mask,
                    "isolate_box_attn": box_iso,
                    "paste_to_canvas": paste_flag,
                    "denoise_strength": box_dn,
                    "z_order": box_z,
                }
            )
        return specs, box_pic_map

    def resolve_thinking_trace(
        self,
        conditioned_ref_telemetry: Optional[
            Union[Sequence[Optional[Dict[str, Any]]], Dict[Any, Dict[str, Any]]]
        ] = None,
    ) -> Tuple[Optional[str], int]:
        raw = self.thinking_trace
        if raw is None:
            return None, 0
        s = str(raw).strip()
        if not s:
            return None, 0
        s_lower = s.lower()
        if s_lower in ("auto", "spatial", "compile"):
            return (
                compile_spatial_thinking_trace(
                    self, conditioned_ref_telemetry=conditioned_ref_telemetry
                ),
                0,
            )
        if s_lower == "speculative" or s_lower.startswith("speculative:"):
            tok_budget = DEFAULT_SPECULATIVE_COT_TOKENS
            if ":" in s_lower:
                _, tail = s_lower.split(":", 1)
                try:
                    tok_budget = int(tail.strip())
                except ValueError:
                    tok_budget = DEFAULT_SPECULATIVE_COT_TOKENS
            tok_budget = max(4, min(512, tok_budget))
            return None, tok_budget
        return s, 0

    @property
    def effective_reference_zero_timestep(self) -> bool:
        if self.reference_zero_timestep is not None:
            return bool(self.reference_zero_timestep)
        if self.has_box_images():
            return True
        return self.edit_format in (
            "moss_image_edit",
            "ai_toolkit_isolated",
            "ai_toolkit_full",
            "control_color_depth",
        )

    @property
    def effective_reference_isolated_kv(self) -> bool:
        if self.reference_isolated_kv is not None:
            return bool(self.reference_isolated_kv)
        if self.has_box_images():
            return True
        return self.edit_format in (
            "moss_image_edit",
            "ai_toolkit_isolated",
            "control_color_depth",
        )

    @property
    def effective_uncond_uses_references(self) -> bool:
        if self.uncond_uses_references is not None:
            return bool(self.uncond_uses_references)
        return not (
            self.edit_format in ("ai_toolkit_isolated", "ai_toolkit_full")
            and not self.has_box_images()
            and not bool(self.control_image_path)
        )

    def resolve_repo_id(self) -> str:
        if self.repo_id and self.repo_id.strip():
            return self.repo_id.strip()
        variant = (self.model_variant or "sft").strip().lower()
        return SUPPORTED_MODELS.get(variant, SUPPORTED_MODELS["sft"])

    def compile_effective_prompt(self) -> str:
        mode = (self.prompt_mode or "natural").strip().lower()
        if (
            mode in ("layout", "layout_json") or self.has_box_images()
        ) and self.layout_items:
            _, box_pic_map = self.resolve_visual_references()
            clean_ls = (self.layout_summary or "").strip()
            clean_p = (self.prompt or "").strip()
            if clean_ls and (
                clean_ls != _LEGACY_DEFAULT_LAYOUT_SUMMARY or not clean_p
            ):
                summary_caption = clean_ls
            else:
                summary_caption = clean_p
            return compile_layout_prompt(
                caption=summary_caption,
                items=self.layout_items,
                height=self.height,
                width=self.width,
                coords_are_pixels=self.layout_coords_are_pixels,
                box_picture_map=box_pic_map,
            )
        return self.prompt.strip()

    def validate(self) -> None:
        if self.height % 16 != 0 or self.width % 16 != 0:
            raise ValueError(
                f"Dimensions {self.width}x{self.height} must be multiples of 16."
            )
        mode = (self.prompt_mode or "natural").strip().lower()
        if mode not in SUPPORTED_PROMPT_MODES:
            raise ValueError(f"Unsupported prompt_mode: {self.prompt_mode}")
        if self.early_solver not in SUPPORTED_SOLVERS:
            raise ValueError(f"Unsupported early_solver: {self.early_solver}")
        if self.late_solver not in SUPPORTED_SOLVERS:
            raise ValueError(f"Unsupported late_solver: {self.late_solver}")
        if self.framing_mode not in SUPPORTED_FRAMING_MODES:
            raise ValueError(f"Unsupported framing_mode: {self.framing_mode}")
        if self.input_role not in SUPPORTED_INPUT_ROLES:
            raise ValueError(f"Unsupported input_role: {self.input_role}")
        if self.edit_format not in SUPPORTED_EDIT_FORMATS:
            raise ValueError(f"Unsupported edit_format: {self.edit_format}")
        mb_mode = str(self.mask_blend_mode or "auto").strip().lower()
        if mb_mode not in SUPPORTED_MASK_BLEND_MODES:
            raise ValueError(f"Unsupported mask_blend_mode: {self.mask_blend_mode}")
        if not (0.0 <= self.slg_start <= self.slg_stop <= 1.0):
            raise ValueError(
                "SLG window must satisfy 0.0 <= slg_start <= slg_stop <= 1.0."
            )
        if not (0.0 <= float(self.box_attn_isolation or 0.0) <= 1.0):
            raise ValueError(
                f"box_attn_isolation must be in [0.0, 1.0], got {self.box_attn_isolation}"
            )
        if not self.compile_effective_prompt():
            raise ValueError("Prompt cannot be empty.")

    def save_preset(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)
        temp_path.replace(path)
        if path.name == DEFAULT_PRESET_FILENAME:
            global _PRESET_CACHE
            _PRESET_CACHE["mtime"] = -1.0
            get_active_engine_defaults()

    @classmethod
    def load_preset(cls, path: Path) -> GenerationRequest:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)


class GenerationResponse(BaseModel):
    output_path: str
    additional_paths: List[str] = Field(default_factory=list)
    exported_vizj_path: Optional[str] = Field(default=None)
    model_variant_used: str
    repo_id_used: str
    width: int
    height: int
    seed_used: int
    generation_time_seconds: float
    step_latency_ms: float
    peak_vram_gb: float
    cpu_offload_active: bool
    num_inference_steps_used: int
    total_nfe: int
    shift_used: float
    use_v2_grid_used: bool
    early_solver_used: str
    late_solver_used: str
    handoff_threshold_used: float
    early_steps: int
    late_steps: int
    early_cfg_used: float
    late_cfg_used: float
    uncond_skip_layers_used: List[int]
    slg_layers_used: List[int]
    slg_scale_used: float
    eta_used: float
    s_noise_used: float
    enable_thinking_used: bool
    viztoken_active: bool = Field(default=True)
    vizinjector_active: bool = Field(default=False)
    vizinjector_backend: str = Field(default="PyTorch Fallback")
    prompt_cache_hit: bool
    reference_cache_hit: bool = Field(default=False)
    common_prefix_tokens: int
    edit_format_used: str = Field(default="standard")
    framing_mode_used: str = Field(default="inscribed")
    input_role_used: str = Field(default="photo_ref")
    mask_blend_mode_used: str = Field(default="auto")
    invert_inpaint_mask_used: bool = Field(default=False)
    reference_zero_timestep_used: bool = Field(default=False)
    reference_isolated_kv_used: bool = Field(default=False)
    box_attn_isolation_used: float = Field(default=0.0)
    uncond_uses_references_used: bool = Field(default=True)
    composite_box_images_used: bool = Field(default=False)
    edit_lora_active: bool = Field(default=False)
    effective_prompt: str
    mean_luminance: float
    rms_contrast: float
    shannon_entropy_bits: float
    laplacian_sharpness: float