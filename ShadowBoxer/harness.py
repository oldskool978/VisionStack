#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

sys.modules["torchvision"] = None
sys.modules["torchvision.transforms"] = None
sys.modules["torchvision.io"] = None

import numpy as np
from PIL import Image
import torch

from engine import MOSSImageEngine
from schema import (
    DEFAULT_LAYOUT_ITEMS,
    DEFAULT_PRESET_FILENAME,
    GenerationRequest,
    GenerationResponse,
    SUPPORTED_BOX_LABELS,
    SUPPORTED_EDIT_FORMATS,
    SUPPORTED_FRAMING_MODES,
    SUPPORTED_INPUT_ROLES,
    SUPPORTED_MASK_BLEND_MODES,
    SUPPORTED_MODELS,
    SUPPORTED_PROMPT_MODES,
    SUPPORTED_SOLVERS,
    get_active_engine_defaults,
    has_custom_default_preset,
    normalize_box_label,
    scale_layout_items_to_canvas,
)

DEFAULT_CONFIG_PATH = ROOT_DIR / DEFAULT_PRESET_FILENAME
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
TEST_INPUTS_DIR = ARTIFACTS_DIR / "test_inputs"
PERMUTATIONS_OUT_DIR = ARTIFACTS_DIR / "permutations"
SWEEPS_DIR = ARTIFACTS_DIR / "sweeps"
_TEST_INPUTS_VERSION_TAG = "studio_photometric_v5"

DEFAULT_POSITIVE_PROMPT = (
    "A wide view of a sleek brushed-steel refrigerator beside a warm walnut kitchen counter "
    "with a ceramic bowl of fresh citrus fruit. A small yellow sticky note on the refrigerator "
    'door is handwritten in crisp black marker: "Remember to test SFT & Base permutations @ 1024x1024"'
)

DEFAULT_LAYOUT_SUMMARY = (
    "A wide view of a sleek brushed-steel refrigerator beside a warm walnut "
    "kitchen counter with a ceramic bowl of fresh citrus fruit and a handwritten "
    "yellow sticky note on the door."
)


def _normalize_request_kwargs(raw: Dict[str, Any]) -> Dict[str, Any]:
    valid = set(GenerationRequest.model_fields.keys())
    return {k: v for k, v in raw.items() if k in valid}


def _graceful_exit(engine: Optional[MOSSImageEngine], code: int = 0) -> None:
    if engine is not None:
        try:
            engine.prepare_for_exit()
        except Exception:
            pass
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(code)


def build_default_request() -> GenerationRequest:
    d = get_active_engine_defaults()
    raw_ls = d.get("layout_summary", None)
    eff_ls = (
        str(raw_ls).strip()
        if (isinstance(raw_ls, str) and raw_ls.strip())
        else None
    )
    req_w = int(d.get("width", 1024))
    req_h = int(d.get("height", 1024))
    coords_px = bool(d.get("layout_coords_are_pixels", True))
    if "layout_items" in d and d["layout_items"]:
        init_items = [dict(b) for b in d["layout_items"] if isinstance(b, dict)]
    else:
        init_items = scale_layout_items_to_canvas(
            DEFAULT_LAYOUT_ITEMS,
            height=req_h,
            width=req_w,
            src_height=1024,
            src_width=1024,
            coords_are_pixels=coords_px,
        )
    raw_kwargs: Dict[str, Any] = {
        "prompt": str(d.get("prompt", DEFAULT_POSITIVE_PROMPT)),
        "negative_prompt": str(d.get("negative_prompt", "")),
        "system_prompt": d.get("system_prompt", None),
        "thinking_trace": d.get("thinking_trace", None),
        "add_vision_id": bool(d.get("add_vision_id", True)),
        "model_variant": str(d.get("model_variant", "sft")),
        "prompt_mode": str(d.get("prompt_mode", "natural")),
        "layout_summary": eff_ls,
        "layout_items": init_items,
        "layout_coords_are_pixels": coords_px,
        "preserve_canvas_outside_boxes": bool(
            d.get("preserve_canvas_outside_boxes", False)
        ),
        "box_attn_isolation": float(d.get("box_attn_isolation", 0.0)),
        "uncond_uses_references": d.get("uncond_uses_references", None),
        "composite_box_images_to_init": bool(
            d.get("composite_box_images_to_init", False)
        ),
        "enable_thinking": bool(d.get("enable_thinking", True)),
        "width": req_w,
        "height": req_h,
        "num_inference_steps": int(d.get("num_inference_steps", 50)),
        "shift": float(d.get("shift", 5.0)),
        "use_v2_grid": bool(d.get("use_v2_grid", True)),
        "early_solver": str(d.get("early_solver", "heun")),
        "late_solver": str(d.get("late_solver", "ipndm")),
        "handoff_threshold": float(d.get("handoff_threshold", 0.35)),
        "guidance_scale": float(d.get("guidance_scale", 3.2)),
        "early_cfg": float(d.get("early_cfg", 3.2)),
        "late_cfg": float(d.get("late_cfg", 2.8)),
        "uncond_skip_layers": list(d.get("uncond_skip_layers", [24])),
        "slg_layers": list(d.get("slg_layers", [24])),
        "slg_scale": float(d.get("slg_scale", 0.0)),
        "slg_start": float(d.get("slg_start", 0.20)),
        "slg_stop": float(d.get("slg_stop", 0.70)),
        "eta": float(d.get("eta", 0.0)),
        "s_noise": float(d.get("s_noise", 1.0)),
        "seed": int(d.get("seed", 42)),
        "num_images_per_prompt": int(d.get("num_images_per_prompt", 1)),
        "max_sequence_length": int(d.get("max_sequence_length", 16384)),
        "output_path": str(d.get("output_path", "output_moss_master.png")),
        "cpu_offload": bool(d.get("cpu_offload", False)),
        "prepass_warmup": bool(d.get("prepass_warmup", True)),
        "denoise_strength": float(d.get("denoise_strength", 1.0)),
        "mask_blend_mode": str(d.get("mask_blend_mode", "auto")),
        "invert_inpaint_mask": bool(d.get("invert_inpaint_mask", False)),
        "edit_format": str(d.get("edit_format", "standard")),
        "framing_mode": str(d.get("framing_mode", "inscribed")),
        "input_role": str(d.get("input_role", "photo_ref")),
        "max_vit_patches": int(d.get("max_vit_patches", 2304)),
        "deband_8bit": bool(d.get("deband_8bit", True)),
        "prefer_vulkan": bool(d.get("prefer_vulkan", True)),
        "mtf_boost": float(d.get("mtf_boost", 0.08)),
        "soft_knee_tau": float(d.get("soft_knee_tau", 0.015)),
        "mask_feather_radius": float(d.get("mask_feather_radius", 1.25)),
        "latent_clamp_sigma": float(d.get("latent_clamp_sigma", 4.5)),
        "apply_bayer_dither": bool(d.get("apply_bayer_dither", True)),
        "use_reference_cache": bool(d.get("use_reference_cache", True)),
        "edit_lora_path": d.get("edit_lora_path", None),
        "edit_lora_scale": float(d.get("edit_lora_scale", 1.0)),
    }
    return GenerationRequest(**_normalize_request_kwargs(raw_kwargs))


def request_to_dict(req: GenerationRequest) -> Dict[str, Any]:
    return req.to_dict()


def request_from_dict(data: Dict[str, Any]) -> GenerationRequest:
    merged = request_to_dict(build_default_request())
    merged.update(_normalize_request_kwargs(data))
    return GenerationRequest.from_dict(_normalize_request_kwargs(merged))


def _clone_request(req: GenerationRequest, **updates: Any) -> GenerationRequest:
    norm_updates = _normalize_request_kwargs(updates)
    return req.model_copy(update=norm_updates, deep=True)


def load_or_create_initial_request(
    config_path: Optional[Path] = None,
) -> Tuple[GenerationRequest, str]:
    target = config_path or DEFAULT_CONFIG_PATH
    if target.is_file():
        try:
            req = GenerationRequest.load_preset(target)
            if not req.prompt.strip():
                req.prompt = DEFAULT_POSITIVE_PROMPT
            if not req.layout_items:
                req.layout_items = scale_layout_items_to_canvas(
                    DEFAULT_LAYOUT_ITEMS,
                    height=req.height,
                    width=req.width,
                    src_height=1024,
                    src_width=1024,
                    coords_are_pixels=req.layout_coords_are_pixels,
                )
            req.validate()
            return req, f"ShadowBoxer/{target.name} (Active File)"
        except Exception as e:
            print(f"[!] Warning: Failed to parse {target}: {e}. Using baseline.")

    banner = (
        "ShadowBoxer/default.json (Active File)"
        if has_custom_default_preset()
        else "Discovered Optimal Baseline (V03 Heun->IPNDM)"
    )
    return build_default_request(), banner


def save_request_json(req: GenerationRequest, path: Path) -> None:
    req.save_preset(path)


load_request_from_json = lambda path: load_or_create_initial_request(path)[0]
save_request_to_json = save_request_json


def _truncate(text: Optional[str], max_len: int = 68) -> str:
    if not text:
        return "<Empty>"
    clean = " ".join(text.strip().split())
    if not clean:
        return "<Empty>"
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3] + "..."


def _short_path(p: Optional[str]) -> str:
    if not p or not p.strip():
        return "<None>"
    return Path(p).name


def _apply_box_denoise_spec(req: GenerationRequest, spec_str: str) -> None:
    s = spec_str.strip()
    if not s:
        return
    if not req.layout_items:
        req.layout_items = scale_layout_items_to_canvas(
            DEFAULT_LAYOUT_ITEMS,
            height=req.height,
            width=req.width,
            src_height=1024,
            src_width=1024,
            coords_are_pixels=req.layout_coords_are_pixels,
        )
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    for part in parts:
        sep = ":" if ":" in part else ("=" if "=" in part else None)
        if sep is not None:
            k_str, v_str = part.split(sep, 1)
            k_clean = k_str.strip().lower()
            v_clean = v_str.strip().lower()
            dn_val = (
                None
                if v_clean in ("auto", "none", "null", "default")
                else max(0.0, min(1.0, float(v_clean)))
            )
            if k_clean in ("all", "*"):
                for item in req.layout_items:
                    if isinstance(item, dict):
                        if dn_val is None:
                            item.pop("denoise_strength", None)
                        else:
                            item["denoise_strength"] = dn_val
            else:
                idx = int(k_clean)
                if 0 <= idx < len(req.layout_items) and isinstance(
                    req.layout_items[idx], dict
                ):
                    if dn_val is None:
                        req.layout_items[idx].pop("denoise_strength", None)
                    else:
                        req.layout_items[idx]["denoise_strength"] = dn_val
        else:
            v_clean = part.lower()
            dn_val = (
                None
                if v_clean in ("auto", "none", "null", "default")
                else max(0.0, min(1.0, float(v_clean)))
            )
            for item in req.layout_items:
                if isinstance(item, dict):
                    if dn_val is None:
                        item.pop("denoise_strength", None)
                    else:
                        item["denoise_strength"] = dn_val


def _interactive_edit_boxes(req: GenerationRequest) -> None:
    if not req.layout_items:
        req.layout_items = scale_layout_items_to_canvas(
            DEFAULT_LAYOUT_ITEMS,
            height=req.height,
            width=req.width,
            src_height=1024,
            src_width=1024,
            coords_are_pixels=req.layout_coords_are_pixels,
        )
    while True:
        print("\n" + "-" * 92)
        print("PER-BOX SPATIO-TEMPORAL OVERRIDES & COLLABORATIVE ROLES:")
        print("-" * 92)
        for idx, item in enumerate(req.layout_items):
            if not isinstance(item, dict):
                continue
            enabled = item.get("enabled", True) is not False
            raw_txt = item.get("text")
            has_txt = isinstance(raw_txt, str) and bool(raw_txt.strip())
            lbl = normalize_box_label(
                item.get("label") or item.get("category"), has_text=has_txt
            )
            bbox = item.get("bbox", [0, 0, 0, 0])
            img_p = (
                item.get("image_path")
                or item.get("image")
                or item.get("vizj_path")
            )
            paste_flag = item.get(
                "paste_to_canvas", req.composite_box_images_to_init
            )
            inp_flag = bool(item.get("inpaint_box", False))
            iso_val = item.get("isolate_box_attn")
            dn_val = item.get("denoise_strength")
            iso_str = f"{float(iso_val):.2f}" if iso_val is not None else "auto"
            dn_str = f"{float(dn_val):.2f}" if dn_val is not None else "auto"
            desc = (
                f'"{raw_txt}"'
                if (lbl == "text" and has_txt)
                else _truncate(
                    str(item.get("caption") or item.get("description") or ""), 34
                )
            )
            print(
                f" [{idx}] {'ON ' if enabled else 'OFF'} | {lbl:<9} | bbox={bbox} | "
                f"Ref={_short_path(img_p)} | Splat={bool(paste_flag)} | Inp={inp_flag} | "
                f"Iso={iso_str} | d={dn_str} | {desc}"
            )
        print("-" * 92)
        sel = (
            input(
                "Select box index [0..N-1], 'a' to set all box denoise, or Enter to return: "
            )
            .strip()
            .lower()
        )
        if not sel:
            return
        if sel == "a":
            all_dn = input(
                "Enter Denoise Strength for ALL boxes [0.0-1.0] or 'auto': "
            ).strip()
            if all_dn:
                _apply_box_denoise_spec(req, f"all:{all_dn}")
            continue
        idx = int(sel)
        if not (0 <= idx < len(req.layout_items)):
            print(f"[!] Box index out of range: {idx}")
            continue
        box = dict(req.layout_items[idx])
        en_in = input(
            f"  Enable Box [{idx}]? [Y/n] (Current: {box.get('enabled', True) is not False}): "
        ).strip()
        if en_in:
            box["enabled"] = en_in.lower() not in ("n", "no", "0", "false")
        lbl_in = input(
            f"  Box [{idx}] Label ({'/'.join(SUPPORTED_BOX_LABELS)}) (Current: {box.get('label', 'object')}): "
        ).strip()
        if lbl_in:
            norm_lbl = normalize_box_label(lbl_in, has_text=False)
            box["label"] = norm_lbl
            box.pop("category", None)
            if norm_lbl != "text":
                box.pop("text", None)
        dn_in = input(
            f"  Box [{idx}] Denoise Strength [0.0-1.0] or 'auto' (Current: {box.get('denoise_strength', 'auto')}): "
        ).strip()
        if dn_in.lower() in ("auto", "none", "default"):
            box.pop("denoise_strength", None)
        elif dn_in:
            box["denoise_strength"] = max(0.0, min(1.0, float(dn_in)))
        iso_in = input(
            f"  Box [{idx}] Attention Isolation [0.0-1.0] or 'auto' (Current: {box.get('isolate_box_attn', 'auto')}): "
        ).strip()
        if iso_in.lower() in ("auto", "none", "default"):
            box.pop("isolate_box_attn", None)
        elif iso_in:
            box["isolate_box_attn"] = max(0.0, min(1.0, float(iso_in)))
        pst_in = input(
            f"  Box [{idx}] Splat/Paste to Canvas? [y/n/auto] (Current: {box.get('paste_to_canvas', 'auto')}): "
        ).strip()
        if pst_in.lower() in ("auto", "default"):
            box.pop("paste_to_canvas", None)
        elif pst_in:
            box["paste_to_canvas"] = pst_in.lower() in ("y", "yes", "1", "true")
        inp_in = input(
            f"  Box [{idx}] Inpaint Box Aperture? [y/N] (Current: {bool(box.get('inpaint_box', False))}): "
        ).strip()
        if inp_in:
            box["inpaint_box"] = inp_in.lower() in ("y", "yes", "1", "true")
        img_in = input(
            f"  Box [{idx}] Reference Image Path or 'none' (Current: {box.get('image_path')}): "
        ).strip()
        if img_in.lower() == "none":
            box.pop("image_path", None)
            box.pop("image", None)
            box.pop("vizj_path", None)
        elif img_in:
            box["image_path"] = img_in
        req.layout_items[idx] = box


def render_menu(req: GenerationRequest, banner_tag: str) -> None:
    repo_str = SUPPORTED_MODELS.get(req.model_variant.lower(), req.repo_id or "Custom")
    n_boxes = len(req.layout_items or [])
    think_str = "ENABLED (<think>)" if req.enable_thinking else "DISABLED (</think>)"
    slg_str = (
        f"Layers {req.slg_layers} | Scale: {req.slg_scale:.2f} [{req.slg_start:.2f}-{req.slg_stop:.2f}]"
        if (req.slg_layers and req.slg_scale and req.slg_scale > 0.0)
        else f"DISABLED (Pre-staged: {req.slg_layers})"
    )
    refs_str = (
        ", ".join(Path(p).name for p in req.reference_image_paths)
        if req.reference_image_paths
        else "<None>"
    )
    lora_str = (
        f"{_short_path(req.edit_lora_path)} ({req.edit_lora_scale:.2f})"
        if (req.edit_lora_path and abs(req.edit_lora_scale) > 1e-6)
        else "<Disabled>"
    )
    uncond_ref_str = (
        "AUTO"
        if req.uncond_uses_references is None
        else ("ON" if req.uncond_uses_references else "OFF")
    )
    mask_alg_str = f"{req.mask_blend_mode or 'auto'}{'(Inv)' if req.invert_inpaint_mask else ''}"

    print("\n" + "=" * 92)
    print("               SHADOWBOXER: OPENMOSS ARCHITECTURAL EXPLORATION HARNESS")
    print(f"                     [{banner_tag}]")
    print("=" * 92)
    print(" --- MODEL & PROMPT CONDITIONING ---")
    print(f" [M]  Active Model Variant:   {req.model_variant.upper()} ({repo_str})")
    print(
        f" [1]  Prompt Mode:            {req.prompt_mode.upper()} ({n_boxes} layout boxes) | [B] Per-Box Overrides"
    )
    print(f" [2]  Positive Prompt:        {_truncate(req.prompt, 65)}")
    print(f" [3]  Negative Prompt:        {_truncate(req.negative_prompt, 65)}")
    print(
        f" [4]  Qwen3.5 Thinking Mode:  {think_str} | VisionID: {req.add_vision_id} | MaxSeq: {req.max_sequence_length}"
    )
    print(f" [5]  Canvas Resolution:      {req.width} x {req.height} px (Multiples of 16)")
    print(" --- FLOW-MATCHING SCHEDULE & SOLVER REGIME ---")
    print(
        f" [6]  Inference Steps & Shift:Steps: {req.num_inference_steps} | Shift: {req.shift:.2f} | V2 Grid: {req.use_v2_grid}"
    )
    print(
        f" [7]  Early / Late Solvers:   Early: {req.early_solver.upper()} | Late: {req.late_solver.upper()} | Handoff t*: {req.handoff_threshold:.2f}"
    )
    print(
        f" [8]  CFG Guidance Field:     Early CFG: {req.early_cfg:.2f} | Late CFG: {req.late_cfg:.2f}"
    )
    print(f" [9]  Negative Block Skip:    {req.uncond_skip_layers} (Default: [24])")
    print(f" [10] Positive SLG Guidance:  {slg_str}")
    print(
        f" [11] SDE Dispersion (Early): Eta: {req.eta:.3f} | S-Noise: {req.s_noise:.3f}"
    )
    print(" --- VIZINJECTOR FILE INPUT, INPAINT, CONTROL & AGENTIC EDITING ---")
    print(
        f" [12] Init, Inpaint & Algebra:Init: {_short_path(req.init_image_path)} (Denoise: {req.denoise_strength:.2f}) | Mask: {_short_path(req.inpaint_mask_path)} [{mask_alg_str}]"
    )
    print(
        f" [13] Multi-Reference (4D):   {refs_str} | Control Map: {_short_path(req.control_image_path)}"
    )
    print(
        f" [14] Edit & Framing Routing: Edit: {req.edit_format} | Framing: {req.framing_mode} | Role: {req.input_role} | ViT: {req.max_vit_patches}p"
    )
    print(
        f" [15] Radiometric & LoRA:     MTF: {req.mtf_boost:.2f} | Feather: {req.mask_feather_radius:.2f} | Clamp: {req.latent_clamp_sigma:.1f}s | LoRA: {lora_str}"
    )
    print(" --- HARDWARE, SEED & EXECUTION ---")
    print(f" [16] PRNG Seed & Output:     Seed: {req.seed} | Path: {req.output_path}")
    print(
        f" [17] Memory & Backends:      CPU Offload: {req.cpu_offload} | Vulkan: {req.prefer_vulkan} | RefCache: {req.use_reference_cache}"
    )
    print(
        f" [18] Spatio-Visual & ChatML: BoxIso: {float(req.box_attn_isolation or 0.0):.2f} | BoxSplat: {req.composite_box_images_to_init} | UncondRefs: {uncond_ref_str} | Sys/Trace: {_truncate(req.system_prompt or req.thinking_trace, 18)}"
    )
    print("-" * 92)
    print(
        " [G]  Generate Master Image   [B]  Inspect/Edit Boxes   [TP] Run 16-Vector Permutation Suite"
    )
    print(
        " [TI] Stage Test Input Files  [P]  Preview Prompt  [X] Sweep  [T] Reset  [L] Load  [S] Save  [D] Save default.json"
    )
    print(" [W]  Run Prepass Warmup      [Q]  Quit Harness")
    print("=" * 92)


def render_telemetry_report(
    resp: GenerationResponse, engine: Optional[MOSSImageEngine] = None
) -> None:
    vram_mode = "CPU OFFLOAD" if resp.cpu_offload_active else "RESIDENT VRAM"
    sched_str = "MOSS V2 NEAREST-GRID" if resp.use_v2_grid_used else "CONTINUOUS SHIFT"
    lat_h, lat_w = resp.height // 16, resp.width // 16
    tok_name = (
        "SOVEREIGN VIZTOKEN (C11 AVX2 SWISS5)"
        if resp.viztoken_active
        else "FALLBACK TOKENIZER"
    )
    mask_tag = f"{resp.mask_blend_mode_used}{'(Inv)' if resp.invert_inpaint_mask_used else ''}"
    print("\n" + "=" * 92)
    print("                          SHADOWBOXER: OPENMOSS TELEMETRY REPORT")
    print("=" * 92)
    print(f"Master Destination:     {resp.output_path}")
    if resp.additional_paths:
        print(f"Additional Outputs:     {', '.join(resp.additional_paths)}")
    if resp.exported_vizj_path:
        print(f"Exported .VIZJ Bundle:  {resp.exported_vizj_path}")
    print(
        f"Active Model Variant:   {resp.model_variant_used.upper()} ({resp.repo_id_used})"
    )
    print(
        f"Canvas Resolution:      {resp.width}x{resp.height} px (Latent Grid: {lat_w}x{lat_h})"
    )
    print(f"PRNG Seed:              {resp.seed_used}")
    print(
        f"Inference Latency:      {resp.generation_time_seconds:.3f}s ({resp.step_latency_ms:.1f} ms/step)"
    )
    print(f"Peak VRAM Footprint:    {resp.peak_vram_gb:.3f} GB ({vram_mode})")
    if engine is not None and getattr(engine, "memory_guard", None) is not None:
        hw = engine.memory_guard.poll()
        if engine.memory_guard.active:
            print(
                f"WDDM Hardware Residency:Dedicated: {hw.dedicated_used_gb:.2f}/{hw.dedicated_total_gb:.2f} GB "
                f"(Avail: {hw.dedicated_avail_gb:.2f} GB, Contig: {hw.contiguous_avail_gb:.2f} GB) | "
                f"Shared Spill Delta: +{hw.shared_delta_gb:.2f} GB"
            )
            if hw.host_total_gb > 0.0:
                print(
                    f"Host Physical Memory:   Available: {hw.host_avail_gb:.2f}/{hw.host_total_gb:.2f} GB "
                    f"| HIP Inactive Split: {hw.pt_inactive_split_gb * 1024.0:.1f} MB"
                )
    print(
        f"Tokenizer Engine:       {tok_name} | Prompt Cache Hit: {resp.prompt_cache_hit} (Shared Prefix: {resp.common_prefix_tokens} tok)"
    )
    print(
        f"Vision Conditioner:     {resp.vizinjector_backend} | Ref Cache Hit: {resp.reference_cache_hit}"
    )
    print(
        f"Conditioning Routing:   Edit={resp.edit_format_used} | Framing={resp.framing_mode_used} | Role={resp.input_role_used} | Mask={mask_tag} | t_ref=0: {resp.reference_zero_timestep_used} | IsoKV: {resp.reference_isolated_kv_used} | BoxIso: {resp.box_attn_isolation_used:.2f} | UncondRefs: {resp.uncond_uses_references_used} | BoxSplat: {resp.composite_box_images_used} | LoRA: {resp.edit_lora_active}"
    )
    print(f"Schedule Regime:        {sched_str} | Shift: {resp.shift_used:.2f}")
    print(
        f"ODE Solvers:            Early: {resp.early_solver_used.upper()} ({resp.early_steps} st) | Late: {resp.late_solver_used.upper()} ({resp.late_steps} st) | t* = {resp.handoff_threshold_used:.3f}"
    )
    print(
        f"Evaluated Passes (NFE): {resp.total_nfe} DiT forwards across {resp.num_inference_steps_used} steps"
    )
    print(
        f"Guidance Field (CFG):   Early CFG: {resp.early_cfg_used:.2f} | Late CFG: {resp.late_cfg_used:.2f}"
    )
    print(f"Negative Block Skip:    {resp.uncond_skip_layers_used}")
    if resp.slg_layers_used and resp.slg_scale_used > 0.0:
        print(
            f"Positive SLG Guidance:  Layers {resp.slg_layers_used} | Scale: {resp.slg_scale_used:.2f}"
        )
    if resp.eta_used > 0.0:
        print(
            f"Early SDE Dispersion:   Eta: {resp.eta_used:.3f} | S-Noise: {resp.s_noise_used:.3f}"
        )
    print(
        f"Signal Luminance / RMS: Mean={resp.mean_luminance:.4f} | Contrast={resp.rms_contrast:.4f}"
    )
    print(
        f"Information / Detail:   Entropy={resp.shannon_entropy_bits:.3f} bits | Sharpness={resp.laplacian_sharpness:.4f}"
    )
    print("-" * 92)
    print("Effective Prompt:")
    print(resp.effective_prompt)
    print("=" * 92 + "\n")


def print_telemetry_report(
    engine: MOSSImageEngine, resp: GenerationResponse
) -> None:
    render_telemetry_report(resp, engine=engine)


def stage_test_input_files(
    dest_dir: Path = TEST_INPUTS_DIR, force: bool = False
) -> Dict[str, Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    p_ref_a = dest_dir / "ref_subject_a.png"
    p_ref_b = dest_dir / "ref_style_b.png"
    p_ref_c = dest_dir / "ref_cutout_c.png"
    p_init = dest_dir / "init_canvas.png"
    p_mask = dest_dir / "inpaint_mask.png"
    p_ctrl = dest_dir / "control_map.png"
    p_ver = dest_dir / ".test_inputs_version"

    out_map = {
        "ref_a": p_ref_a,
        "ref_b": p_ref_b,
        "ref_c": p_ref_c,
        "init": p_init,
        "mask": p_mask,
        "control": p_ctrl,
        "ref_subject_a": p_ref_a,
        "ref_subject_b": p_ref_b,
        "ref_cutout_c": p_ref_c,
        "init_canvas": p_init,
        "inpaint_mask": p_mask,
        "control_color_map": p_ctrl,
    }
    ver_ok = (
        p_ver.is_file()
        and p_ver.read_text(encoding="utf-8").strip() == _TEST_INPUTS_VERSION_TAG
    )
    if (
        not force
        and ver_ok
        and all(
            p.is_file() and p.stat().st_size > 0
            for p in (p_ref_a, p_ref_b, p_ref_c, p_init, p_mask, p_ctrl)
        )
    ):
        return out_map

    h, w = 512, 512
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, h, dtype=np.float32),
        np.linspace(-1.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    rad = np.sqrt(xx * xx + yy * yy)

    steel_brush = (
        0.012 * np.sin(210.0 * yy + 0.35 * np.sin(14.0 * xx))
        + 0.006 * np.cos(370.0 * yy)
    )
    spec_lobe = 0.22 * np.exp(-((xx + 0.26) ** 2) / 0.14) * (1.0 - 0.14 * (yy ** 2))
    soft_shadow = -0.15 * np.exp(-((xx - 0.58) ** 2) / 0.24)
    door_seam = -0.34 * np.exp(-((xx - 0.12) ** 2) / 0.00035)
    handle_bar = (
        ((xx > -0.02) & (xx < 0.03) & (yy > -0.58) & (yy < 0.52)).astype(np.float32)
        * (0.15 * np.cos((xx - 0.005) * 65.0))
    )
    base_steel = (
        0.63 - 0.05 * yy + spec_lobe + soft_shadow + steel_brush + door_seam + handle_bar
    )
    ref_a_r = base_steel * 0.96
    ref_a_g = base_steel * 0.98
    ref_a_b = base_steel * 1.03 + 0.012
    ref_a = np.stack(
        [
            np.clip(ref_a_r * 255.0, 0, 255),
            np.clip(ref_a_g * 255.0, 0, 255),
            np.clip(ref_a_b * 255.0, 0, 255),
        ],
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(ref_a, mode="RGB").save(p_ref_a)

    is_counter = (yy > 0.16).astype(np.float32)
    wall_grad = 0.78 - 0.10 * yy - 0.06 * (xx ** 2)
    wood_wave = (
        0.035 * np.sin(54.0 * (yy + 0.06 * np.sin(6.5 * xx)))
        + 0.018 * np.cos(135.0 * yy)
    )
    walnut_r = 0.48 + 0.08 * (1.0 - yy) + wood_wave
    walnut_g = 0.31 + 0.05 * (1.0 - yy) + 0.75 * wood_wave
    walnut_b = 0.19 + 0.03 * (1.0 - yy) + 0.45 * wood_wave

    ref_b_r = np.where(is_counter > 0.5, walnut_r, wall_grad * 0.98)
    ref_b_g = np.where(is_counter > 0.5, walnut_g, wall_grad * 0.95)
    ref_b_b = np.where(is_counter > 0.5, walnut_b, wall_grad * 0.90)

    bowl_cx, bowl_cy = 0.46, 0.44
    bowl_ellip = np.sqrt(((xx - bowl_cx) / 0.38) ** 2 + ((yy - bowl_cy) / 0.22) ** 2)
    bowl_mask = np.clip((1.0 - bowl_ellip) * 14.0, 0.0, 1.0) * (yy > 0.20).astype(
        np.float32
    )
    bowl_shade = 0.88 - 0.22 * (xx - bowl_cx) - 0.18 * (yy - (bowl_cy - 0.04))
    ref_b_r = ref_b_r * (1.0 - bowl_mask) + bowl_mask * (bowl_shade * 0.95)
    ref_b_g = ref_b_g * (1.0 - bowl_mask) + bowl_mask * (bowl_shade * 0.93)
    ref_b_b = ref_b_b * (1.0 - bowl_mask) + bowl_mask * (bowl_shade * 0.88)

    for cx_f, cy_f, r_f, cr, cg, cb in (
        (0.32, 0.30, 0.14, 0.95, 0.52, 0.12),
        (0.56, 0.32, 0.13, 0.96, 0.58, 0.14),
        (0.45, 0.24, 0.12, 0.96, 0.84, 0.18),
    ):
        d_f = np.sqrt((xx - cx_f) ** 2 + (yy - cy_f) ** 2)
        f_mask = np.clip((r_f - d_f) * 28.0, 0.0, 1.0)
        nz = np.sqrt(np.maximum(0.0, 1.0 - np.minimum(1.0, (d_f / r_f) ** 2)))
        lambert = np.clip(
            0.35
            + 0.65
            * (
                -0.45 * ((xx - cx_f) / r_f)
                - 0.50 * ((yy - cy_f) / r_f)
                + 0.74 * nz
            ),
            0.25,
            1.12,
        )
        ref_b_r = ref_b_r * (1.0 - f_mask) + f_mask * (cr * lambert)
        ref_b_g = ref_b_g * (1.0 - f_mask) + f_mask * (cg * lambert)
        ref_b_b = ref_b_b * (1.0 - f_mask) + f_mask * (cb * lambert)

    ref_b = np.stack(
        [
            np.clip(ref_b_r * 255.0, 0, 255),
            np.clip(ref_b_g * 255.0, 0, 255),
            np.clip(ref_b_b * 255.0, 0, 255),
        ],
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(ref_b, mode="RGB").save(p_ref_b)

    cutout_alpha = np.clip((0.68 - rad) * 14.0, 0.0, 1.0)
    nz_cut = np.sqrt(np.maximum(0.0, 1.0 - np.minimum(1.0, (rad / 0.68) ** 2)))
    ceramic_shade = np.clip(
        0.78 - 0.18 * xx - 0.16 * yy + 0.14 * nz_cut, 0.42, 0.96
    )
    rim_ring = np.exp(-((rad - 0.58) ** 2) / 0.0025) * 0.10
    cut_r = ceramic_shade * 0.94 + rim_ring
    cut_g = ceramic_shade * 0.91 + rim_ring * 0.85
    cut_b = ceramic_shade * 0.85 + rim_ring * 0.60

    for cx_c, cy_c, r_c, fr, fg, fb in (
        (-0.20, 0.06, 0.24, 0.96, 0.54, 0.12),
        (0.22, 0.08, 0.22, 0.95, 0.48, 0.10),
        (0.00, -0.18, 0.21, 0.96, 0.85, 0.18),
        (-0.02, 0.22, 0.19, 0.94, 0.62, 0.14),
    ):
        d_c = np.sqrt((xx - cx_c) ** 2 + (yy - cy_c) ** 2)
        m_c = np.clip((r_c - d_c) * 26.0, 0.0, 1.0)
        nz_f = np.sqrt(np.maximum(0.0, 1.0 - np.minimum(1.0, (d_c / r_c) ** 2)))
        pore_tex = 0.012 * np.cos(80.0 * xx) * np.sin(80.0 * yy)
        lit = np.clip(
            0.36
            + 0.68
            * (
                -0.42 * ((xx - cx_c) / r_c)
                - 0.48 * ((yy - cy_c) / r_c)
                + 0.76 * nz_f
            )
            + pore_tex,
            0.22,
            1.10,
        )
        cut_r = cut_r * (1.0 - m_c) + m_c * (fr * lit)
        cut_g = cut_g * (1.0 - m_c) + m_c * (fg * lit)
        cut_b = cut_b * (1.0 - m_c) + m_c * (fb * lit)

    cutout_r = np.where(cutout_alpha > 0.01, cut_r, walnut_r)
    cutout_g = np.where(cutout_alpha > 0.01, cut_g, walnut_g)
    cutout_b = np.where(cutout_alpha > 0.01, cut_b, walnut_b)
    ref_c = np.stack(
        [
            np.clip(cutout_r * 255.0, 0, 255),
            np.clip(cutout_g * 255.0, 0, 255),
            np.clip(cutout_b * 255.0, 0, 255),
            np.clip(cutout_alpha * 255.0, 0, 255),
        ],
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(ref_c, mode="RGBA").save(p_ref_c)

    left_panel = (xx < 0.12).astype(np.float32)
    right_panel = (1.0 - left_panel).astype(np.float32)
    init_r = left_panel * (base_steel * 0.96) + right_panel * ref_b_r
    init_g = left_panel * (base_steel * 0.98) + right_panel * ref_b_g
    init_b = left_panel * (base_steel * 1.02) + right_panel * ref_b_b
    init_img = np.stack(
        [
            np.clip(init_r * 255.0, 0, 255),
            np.clip(init_g * 255.0, 0, 255),
            np.clip(init_b * 255.0, 0, 255),
        ],
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(init_img, mode="RGB").save(p_init)

    d_note_m = np.maximum(np.abs(xx + 0.44) - 0.30, np.abs(yy + 0.25) - 0.22)
    m_note = np.clip(1.0 - np.maximum(0.0, d_note_m) / 0.12, 0.0, 1.0)
    d_bowl_m = np.sqrt(((xx - 0.52) / 0.44) ** 2 + ((yy - 0.44) / 0.48) ** 2)
    m_bowl = np.clip((1.0 - d_bowl_m) * 3.5, 0.0, 1.0)
    mask = np.maximum(m_note, m_bowl)
    mask_u8 = (mask * 255.0).round().astype(np.uint8)
    Image.fromarray(mask_u8, mode="L").save(p_mask)

    ctrl = np.zeros((h, w, 3), dtype=np.uint8)
    is_left = xx < 0.12
    is_left_hi = xx < -0.22
    is_cnt = (xx >= 0.12) & (yy > 0.16)
    is_bwl = (xx >= 0.24) & (
        ((xx - 0.52) / 0.32) ** 2 + ((yy - 0.42) / 0.20) ** 2 <= 1.0
    )
    is_note = (xx > -0.68) & (xx < -0.24) & (yy > -0.46) & (yy < -0.04)
    ctrl[~is_left & ~is_cnt] = [194, 188, 178]
    ctrl[is_left] = [152, 158, 166]
    ctrl[is_left_hi] = [178, 184, 192]
    ctrl[is_cnt] = [124, 82, 52]
    ctrl[is_bwl] = [228, 186, 96]
    ctrl[is_note] = [242, 224, 108]
    Image.fromarray(ctrl, mode="RGB").save(p_ctrl)

    p_ver.write_text(f"{_TEST_INPUTS_VERSION_TAG}\n", encoding="utf-8")
    stale_vizj = PERMUTATIONS_OUT_DIR / "v08_tier2_bundle.vizj"
    if stale_vizj.is_file():
        try:
            stale_vizj.unlink()
        except OSError:
            pass
    print(f"[+] Staged 6 deterministic test input artifacts in: {dest_dir}")
    return out_map


stage_deterministic_test_inputs = stage_test_input_files


def _parse_int_list(raw: str) -> List[int]:
    s = raw.strip().strip("[]")
    if not s or s.lower() in ("none", "off", "empty", "clear"):
        return []
    out: List[int] = []
    for part in s.replace(";", ",").split(","):
        tok = part.strip()
        if not tok:
            continue
        if "-" in tok:
            bounds = tok.split("-", 1)
            for idx in range(int(bounds[0].strip()), int(bounds[1].strip()) + 1):
                out.append(idx)
        else:
            out.append(int(tok))
    return sorted(set(out))


def run_permutation_suite(
    engine: MOSSImageEngine,
    base_req: Optional[GenerationRequest] = None,
    steps_override: Optional[int] = None,
    width_override: Optional[int] = None,
    height_override: Optional[int] = None,
    steps: Optional[int] = None,
) -> List[Dict[str, Any]]:
    staged = stage_test_input_files(force=False)
    PERMUTATIONS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    active_base = base_req if base_req is not None else build_default_request()

    w = max(256, min(2048, (int(width_override or active_base.width) // 16) * 16))
    h = max(256, min(2048, (int(height_override or active_base.height) // 16) * 16))
    eff_steps = (
        steps_override
        if steps_override is not None
        else (
            steps
            if steps is not None
            else min(active_base.num_inference_steps, 12)
        )
    )

    models_info = {m["variant"]: m["cached"] for m in engine.list_available_models()}
    has_base = bool(models_info.get("base", False))

    if active_base.layout_items:
        base_layout_items = scale_layout_items_to_canvas(
            active_base.layout_items,
            height=h,
            width=w,
            src_height=active_base.height,
            src_width=active_base.width,
            coords_are_pixels=active_base.layout_coords_are_pixels,
        )
    else:
        base_layout_items = scale_layout_items_to_canvas(
            DEFAULT_LAYOUT_ITEMS,
            height=h,
            width=w,
            src_height=1024,
            src_width=1024,
            coords_are_pixels=True,
        )

    base_clean = _clone_request(
        active_base,
        width=w,
        height=h,
        num_inference_steps=eff_steps,
        early_solver="euler",
        late_solver="euler",
        handoff_threshold=0.35,
        early_cfg=3.0,
        late_cfg=3.0,
        guidance_scale=3.0,
        slg_layers=[24],
        slg_scale=0.0,
        eta=0.0,
        s_noise=1.0,
        thinking_trace=None,
        init_image_path=None,
        inpaint_mask_path=None,
        mask_blend_mode="auto",
        invert_inpaint_mask=False,
        preserve_canvas_outside_boxes=False,
        control_image_path=None,
        reference_image_paths=[],
        export_vizj_path=None,
        edit_format="standard",
        framing_mode="inscribed",
        input_role="photo_ref",
        edit_lora_path=None,
        edit_lora_scale=0.0,
        box_attn_isolation=0.0,
        uncond_uses_references=None,
        composite_box_images_to_init=False,
        layout_items=base_layout_items,
    )

    v06_horizon_boxes = scale_layout_items_to_canvas(
        DEFAULT_LAYOUT_ITEMS,
        height=h,
        width=w,
        src_height=1024,
        src_width=1024,
        coords_are_pixels=True,
    )
    v06_horizon_boxes[0] = dict(
        v06_horizon_boxes[0],
        label="object",
        caption="A sleek brushed-steel refrigerator door with a small yellow sticky note",
        inpaint_box=True,
        denoise_strength=0.88,
    )
    v06_horizon_boxes[1] = dict(
        v06_horizon_boxes[1],
        inpaint_box=True,
        denoise_strength=0.96,
    )
    v06_horizon_boxes[2] = dict(
        v06_horizon_boxes[2],
        inpaint_box=True,
        denoise_strength=0.94,
    )

    v11_canonical_1024_boxes: List[Dict[str, Any]] = [
        {
            "label": "object",
            "caption": "A sleek brushed-steel refrigerator door with a small yellow sticky note",
            "bbox": [0, 0, 1024, 576],
            "image_path": str(staged["ref_a"]),
            "input_role": "photo_ref",
            "paste_to_canvas": True,
            "isolate_box_attn": 0.85,
            "denoise_strength": 0.88,
            "z_order": 1,
            "enabled": True,
        },
        {
            "label": "object",
            "caption": "Structural depth guide for the left refrigerator panel",
            "bbox": [0, 0, 1024, 576],
            "image_path": str(staged["control"]),
            "input_role": "depth_map",
            "isolate_box_attn": 0.85,
            "z_order": 2,
            "enabled": True,
        },
        {
            "label": "object",
            "caption": "A sculpted ceramic bowl of fresh oranges and lemons resting on a warm walnut counter",
            "bbox": [448, 512, 1024, 1024],
            "image_path": str(staged["ref_c"]),
            "input_role": "photo_ref",
            "paste_to_canvas": True,
            "isolate_box_attn": 0.82,
            "denoise_strength": 0.90,
            "z_order": 3,
            "enabled": True,
        },
        {
            "label": "text",
            "caption": "Handwritten crisp black marker text on a small yellow sticky note",
            "text": "SOVEREIGN V11 CONVERGED",
            "bbox": [256, 112, 512, 464],
            "denoise_strength": 0.96,
            "z_order": 5,
            "enabled": True,
        },
    ]
    v11_spatiovisual_boxes = scale_layout_items_to_canvas(
        v11_canonical_1024_boxes,
        height=h,
        width=w,
        src_height=1024,
        src_width=1024,
        coords_are_pixels=True,
    )

    v12_prompt = (
        "A cinematic culinary studio portrait of a smiling chef in a tailored white chef coat "
        "and tall chef hat standing in front of a sleek brushed-steel kitchen backdrop beside a "
        "warm walnut counter with a ceramic bowl of fresh citrus fruit and a chiseled silver "
        'studio nameplate at the top reading: "CHARACTER SEMANTIC LOCK"'
    )
    v12_character_1024_boxes: List[Dict[str, Any]] = [
        {
            "label": "object",
            "caption": "A sleek brushed-steel kitchen backdrop with soft studio lighting",
            "bbox": [0, 0, 1024, 576],
            "image_path": str(staged["ref_a"]),
            "input_role": "photo_ref",
            "paste_to_canvas": True,
            "isolate_box_attn": 0.85,
            "denoise_strength": 0.88,
            "z_order": 1,
            "enabled": True,
        },
        {
            "label": "character",
            "caption": "A smiling culinary chef portrait in crisp studio rim lighting wearing a tailored white double-breasted chef coat and tall white chef hat",
            "bbox": [192, 160, 1024, 768],
            "isolate_box_attn": 0.92,
            "denoise_strength": 0.96,
            "z_order": 4,
            "enabled": True,
        },
        {
            "label": "object",
            "caption": "A sculpted ceramic bowl of fresh oranges and lemons resting on a warm walnut counter",
            "bbox": [576, 640, 1024, 1024],
            "image_path": str(staged["ref_c"]),
            "input_role": "photo_ref",
            "paste_to_canvas": True,
            "isolate_box_attn": 0.82,
            "denoise_strength": 0.90,
            "z_order": 3,
            "enabled": True,
        },
        {
            "label": "text",
            "caption": "Chiseled silver studio nameplate",
            "text": "CHARACTER SEMANTIC LOCK",
            "bbox": [48, 384, 192, 960],
            "denoise_strength": 0.96,
            "z_order": 6,
            "enabled": True,
        },
    ]
    v12_character_boxes = scale_layout_items_to_canvas(
        v12_character_1024_boxes,
        height=h,
        width=w,
        src_height=1024,
        src_width=1024,
        coords_are_pixels=True,
    )

    v13_prompt = (
        "An expressive culinary chef in a tailored white chef jacket standing beside a sleek "
        "brushed-steel refrigerator door in a warm walnut kitchen with a crisp black marker "
        'yellow sticky note on the upper door reading: "ABM4 MULTITREE VERIFIED"'
    )
    v13_multitree_1024_boxes: List[Dict[str, Any]] = [
        {
            "label": "object",
            "caption": "Sleek brushed-steel refrigerator door panel under warm kitchen studio lighting",
            "bbox": [0, 0, 1024, 448],
            "inpaint_box": True,
            "denoise_strength": 0.88,
            "z_order": 1,
            "enabled": True,
        },
        {
            "label": "character",
            "caption": "Expressive culinary chef portrait in a tailored white chef jacket standing in the studio kitchen",
            "bbox": [192, 384, 1024, 960],
            "inpaint_box": True,
            "denoise_strength": 0.96,
            "z_order": 3,
            "enabled": True,
        },
        {
            "label": "text",
            "caption": "Crisp black marker label on a yellow sticky note",
            "text": "ABM4 MULTITREE VERIFIED",
            "bbox": [64, 48, 256, 416],
            "inpaint_box": True,
            "denoise_strength": 0.96,
            "z_order": 5,
            "enabled": True,
        },
    ]
    v13_multitree_boxes = scale_layout_items_to_canvas(
        v13_multitree_1024_boxes,
        height=h,
        width=w,
        src_height=1024,
        src_width=1024,
        coords_are_pixels=True,
    )

    v15_prompt = (
        "A cinematic dual-portrait in a modern studio kitchen featuring a bearded executive chef "
        "in a charcoal chef jacket on the left and a smiling pastry sous-chef wearing round glasses "
        "and a white chef coat with a tall toque on the right, standing in front of a brushed-steel "
        'kitchen backdrop beneath two studio nameplates reading "CHEF DE CUISINE" and "PATISSIER"'
    )
    v15_multichar_1024_boxes: List[Dict[str, Any]] = [
        {
            "label": "object",
            "caption": "A wide brushed-steel studio kitchen backdrop with warm ambient illumination",
            "bbox": [0, 0, 1024, 640],
            "image_path": str(staged["ref_a"]),
            "input_role": "photo_ref",
            "paste_to_canvas": True,
            "isolate_box_attn": 0.85,
            "denoise_strength": 0.88,
            "z_order": 1,
            "enabled": True,
        },
        {
            "label": "character",
            "caption": "A bearded executive chef in a tailored charcoal double-breasted chef jacket with arms crossed in dramatic studio rim lighting",
            "bbox": [192, 48, 1024, 560],
            "isolate_box_attn": 0.92,
            "denoise_strength": 0.95,
            "z_order": 4,
            "enabled": True,
        },
        {
            "label": "character",
            "caption": "A smiling pastry sous-chef wearing round eyeglasses, a crisp white chef coat, and a tall white toque holding a stainless steel whisk",
            "bbox": [208, 464, 1024, 976],
            "isolate_box_attn": 0.92,
            "denoise_strength": 0.96,
            "z_order": 5,
            "enabled": True,
        },
        {
            "label": "text",
            "caption": "Brushed brass studio nameplate on the upper left",
            "text": "CHEF DE CUISINE",
            "bbox": [48, 48, 160, 480],
            "denoise_strength": 0.96,
            "z_order": 7,
            "enabled": True,
        },
        {
            "label": "text",
            "caption": "Brushed silver studio nameplate on the upper right",
            "text": "PATISSIER",
            "bbox": [48, 544, 160, 960],
            "denoise_strength": 0.96,
            "z_order": 8,
            "enabled": True,
        },
    ]
    v15_multichar_boxes = scale_layout_items_to_canvas(
        v15_multichar_1024_boxes,
        height=h,
        width=w,
        src_height=1024,
        src_width=1024,
        coords_are_pixels=True,
    )

    v16_prompt = (
        "An architectural studio kitchen scene with a sleek brushed-steel appliance panel on the "
        "left, an executive culinary chef and an artisanal sommelier in a linen apron conversing "
        "behind a warm walnut counter with a ceramic bowl of fresh citrus fruit, and a top studio "
        'plaque reading: "DUAL CHARACTER CONVERGED"'
    )
    v16_hybrid_1024_boxes: List[Dict[str, Any]] = [
        {
            "label": "object",
            "caption": "Sleek brushed-steel refrigerator appliance panel on the left",
            "bbox": [0, 0, 1024, 288],
            "inpaint_box": True,
            "denoise_strength": 0.88,
            "z_order": 1,
            "enabled": True,
        },
        {
            "label": "character",
            "caption": "Executive culinary chef in a crisp white chef jacket gesturing in conversation",
            "bbox": [208, 192, 1024, 624],
            "inpaint_box": True,
            "isolate_box_attn": 0.90,
            "denoise_strength": 0.96,
            "z_order": 4,
            "enabled": True,
        },
        {
            "label": "character",
            "caption": "Artisanal sommelier and barista in a dark shirt and leather apron holding a tasting glass",
            "bbox": [224, 544, 1024, 960],
            "inpaint_box": True,
            "isolate_box_attn": 0.90,
            "denoise_strength": 0.96,
            "z_order": 5,
            "enabled": True,
        },
        {
            "label": "object",
            "caption": "A ceramic bowl of fresh citrus fruit on the foreground walnut counter",
            "bbox": [640, 640, 1024, 1024],
            "image_path": str(staged["ref_c"]),
            "input_role": "photo_ref",
            "paste_to_canvas": True,
            "isolate_box_attn": 0.82,
            "denoise_strength": 0.90,
            "z_order": 6,
            "enabled": True,
        },
        {
            "label": "text",
            "caption": "Crisp studio header plaque",
            "text": "DUAL CHARACTER CONVERGED",
            "bbox": [48, 256, 176, 912],
            "inpaint_box": True,
            "denoise_strength": 0.96,
            "z_order": 8,
            "enabled": True,
        },
    ]
    v16_hybrid_boxes = scale_layout_items_to_canvas(
        v16_hybrid_1024_boxes,
        height=h,
        width=w,
        src_height=1024,
        src_width=1024,
        coords_are_pixels=True,
    )

    vectors: List[Tuple[str, GenerationRequest]] = [
        (
            "V01_SFT_Natural_Baseline",
            _clone_request(
                base_clean,
                model_variant="sft",
                prompt_mode="natural",
                output_path=str(PERMUTATIONS_OUT_DIR / "v01_sft_natural.png"),
            ),
        ),
        (
            "V02_SFT_Layout_JSON_SpatialCoT",
            _clone_request(
                base_clean,
                model_variant="sft",
                prompt_mode="layout_json",
                thinking_trace="auto",
                early_solver="heun",
                late_solver="ipndm",
                handoff_threshold=0.35,
                early_cfg=3.2,
                late_cfg=2.8,
                guidance_scale=3.2,
                output_path=str(PERMUTATIONS_OUT_DIR / "v02_sft_layout_json.png"),
            ),
        ),
        (
            "V03_Bifurcated_Heun_IPNDM",
            _clone_request(
                base_clean,
                early_solver="heun",
                late_solver="ipndm",
                handoff_threshold=0.35,
                early_cfg=3.2,
                late_cfg=2.8,
                output_path=str(PERMUTATIONS_OUT_DIR / "v03_bifurcated_heun_ipndm.png"),
            ),
        ),
        (
            "V04_SDE_Dispersion_And_SLG",
            _clone_request(
                base_clean,
                early_solver="sde_gpu_pp",
                late_solver="ipndm",
                handoff_threshold=0.35,
                eta=0.12,
                s_noise=1.0,
                slg_layers=[24],
                slg_scale=0.50,
                slg_start=0.20,
                slg_stop=0.70,
                output_path=str(PERMUTATIONS_OUT_DIR / "v04_sde_slg.png"),
            ),
        ),
        (
            "V05_VizToken_Prefix_Splice",
            _clone_request(
                base_clean,
                prompt=(
                    "A wide view of a sleek brushed-steel refrigerator beside a warm walnut kitchen counter "
                    "with a ceramic bowl of fresh citrus fruit. A small yellow sticky note on the refrigerator "
                    'door is handwritten in crisp black marker: "PASS 2 PREFIX SPLICE CONVERGED"'
                ),
                output_path=str(PERMUTATIONS_OUT_DIR / "v05_prefix_splice.png"),
            ),
        ),
        (
            "V06_Img2Img_Horizon_And_MaskAlgebra",
            _clone_request(
                base_clean,
                prompt_mode="layout_json",
                thinking_trace="auto",
                early_solver="heun",
                late_solver="ipndm",
                early_cfg=3.2,
                late_cfg=2.8,
                init_image_path=str(staged["init"]),
                inpaint_mask_path=str(staged["mask"]),
                denoise_strength=0.92,
                mask_blend_mode="union",
                layout_items=v06_horizon_boxes,
                output_path=str(PERMUTATIONS_OUT_DIR / "v06_img2img_inpaint.png"),
            ),
        ),
        (
            "V07_MossImageEdit_ZeroTimestep",
            _clone_request(
                base_clean,
                early_solver="heun",
                late_solver="ipndm",
                early_cfg=3.2,
                late_cfg=2.8,
                reference_image_paths=[str(staged["ref_a"])],
                edit_format="moss_image_edit",
                framing_mode="never_crop",
                edit_lora_path="auto",
                edit_lora_scale=1.0,
                output_path=str(PERMUTATIONS_OUT_DIR / "v07_moss_image_edit.png"),
            ),
        ),
        (
            "V08_AIToolkit_Isolated_MultiRef_4DRoPE",
            _clone_request(
                base_clean,
                early_solver="heun",
                late_solver="ipndm",
                early_cfg=3.2,
                late_cfg=2.8,
                reference_image_paths=[str(staged["ref_a"]), str(staged["ref_b"])],
                edit_format="ai_toolkit_isolated",
                framing_mode="inscribed",
                export_vizj_path=str(PERMUTATIONS_OUT_DIR / "v08_tier2_bundle.vizj"),
                export_vizj_tier=2,
                output_path=str(PERMUTATIONS_OUT_DIR / "v08_aitoolkit_multiref.png"),
            ),
        ),
        (
            "V09_Control_Color_Depth_Routing",
            _clone_request(
                base_clean,
                early_solver="heun",
                late_solver="ipndm",
                early_cfg=3.2,
                late_cfg=2.8,
                control_image_path=str(staged["control"]),
                edit_format="control_color_depth",
                input_role="color_map",
                output_path=str(PERMUTATIONS_OUT_DIR / "v09_control_map.png"),
            ),
        ),
        (
            "V10_Tier2_VIZJ_ZeroCopy_Reload",
            _clone_request(
                base_clean,
                model_variant="sft",
                early_solver="heun",
                late_solver="ipndm",
                early_cfg=3.2,
                late_cfg=2.8,
                reference_image_paths=[
                    str(PERMUTATIONS_OUT_DIR / "v08_tier2_bundle.vizj")
                ],
                edit_format="ai_toolkit_isolated",
                framing_mode="inscribed",
                output_path=str(PERMUTATIONS_OUT_DIR / "v10_tier2_vizj_reload.png"),
            ),
        ),
        (
            "V11_SpatioVisual_BoxGraph_CovSplat_Cutout",
            _clone_request(
                base_clean,
                model_variant="sft",
                prompt_mode="layout_json",
                thinking_trace="auto",
                early_solver="heun",
                late_solver="ipndm",
                handoff_threshold=0.35,
                early_cfg=3.2,
                late_cfg=2.8,
                box_attn_isolation=0.85,
                composite_box_images_to_init=True,
                layout_items=v11_spatiovisual_boxes,
                output_path=str(
                    PERMUTATIONS_OUT_DIR / "v11_spatiovisual_boxgraph.png"
                ),
            ),
        ),
        (
            "V12_Character_SemanticHierarchy_And_EllipticalCarve",
            _clone_request(
                base_clean,
                model_variant="sft",
                prompt=v12_prompt,
                layout_summary=v12_prompt,
                prompt_mode="layout_json",
                thinking_trace="auto",
                early_solver="heun",
                late_solver="ipndm",
                handoff_threshold=0.35,
                early_cfg=3.2,
                late_cfg=2.8,
                box_attn_isolation=0.85,
                composite_box_images_to_init=True,
                layout_items=v12_character_boxes,
                output_path=str(
                    PERMUTATIONS_OUT_DIR / "v12_character_elliptical_carve.png"
                ),
            ),
        ),
        (
            "V13_MultiTree_ABM4_And_ManualHorizonLock",
            _clone_request(
                base_clean,
                model_variant="sft",
                prompt=v13_prompt,
                layout_summary=v13_prompt,
                prompt_mode="layout_json",
                thinking_trace="auto",
                early_solver="multitree",
                late_solver="multitree",
                handoff_threshold=0.50,
                early_cfg=3.1,
                late_cfg=2.8,
                init_image_path=str(staged["init"]),
                denoise_strength=0.92,
                mask_blend_mode="boxes_only",
                preserve_canvas_outside_boxes=True,
                layout_items=v13_multitree_boxes,
                output_path=str(
                    PERMUTATIONS_OUT_DIR / "v13_multitree_horizon_lock.png"
                ),
            ),
        ),
        (
            "V14_Base_HotSwap_Speculative_CoT"
            if has_base
            else "V14_Speculative_CoT_Reasoning",
            _clone_request(
                base_clean,
                model_variant="base" if has_base else "sft",
                prompt_mode="layout_json",
                thinking_trace="speculative:24",
                early_solver="heun",
                late_solver="ipndm",
                handoff_threshold=0.35,
                early_cfg=3.2,
                late_cfg=2.8,
                output_path=str(PERMUTATIONS_OUT_DIR / "v14_speculative_cot.png"),
            ),
        ),
        (
            "V15_MultiCharacter_DualSubject_DepthHierarchy",
            _clone_request(
                base_clean,
                model_variant="sft",
                prompt=v15_prompt,
                layout_summary=v15_prompt,
                prompt_mode="layout_json",
                thinking_trace="auto",
                early_solver="heun",
                late_solver="ipndm",
                handoff_threshold=0.35,
                early_cfg=3.2,
                late_cfg=2.8,
                box_attn_isolation=0.88,
                composite_box_images_to_init=True,
                layout_items=v15_multichar_boxes,
                output_path=str(
                    PERMUTATIONS_OUT_DIR / "v15_multicharacter_dualsubject.png"
                ),
            ),
        ),
        (
            "V16_MultiCharacter_Hybrid_Control_And_SpeculativeCoT",
            _clone_request(
                base_clean,
                model_variant="sft",
                prompt=v16_prompt,
                layout_summary=v16_prompt,
                prompt_mode="layout_json",
                thinking_trace="speculative:24",
                early_solver="sde_gpu_pp",
                late_solver="ipndm",
                handoff_threshold=0.35,
                eta=0.10,
                s_noise=1.0,
                early_cfg=3.2,
                late_cfg=2.8,
                init_image_path=str(staged["init"]),
                control_image_path=str(staged["control"]),
                edit_format="control_color_depth",
                input_role="depth_map",
                denoise_strength=0.95,
                mask_blend_mode="boxes_only",
                preserve_canvas_outside_boxes=True,
                box_attn_isolation=0.88,
                composite_box_images_to_init=True,
                layout_items=v16_hybrid_boxes,
                output_path=str(
                    PERMUTATIONS_OUT_DIR / "v16_multicharacter_hybrid_cot.png"
                ),
            ),
        ),
    ]

    total_vecs = len(vectors)
    results: List[Dict[str, Any]] = []
    t_suite = time.perf_counter()
    print("\n" + "=" * 92)
    print(
        f"     EXECUTING {total_vecs}-VECTOR ARCHITECTURAL PERMUTATION SUITE ({w}x{h} @ {eff_steps} steps)"
    )
    print("=" * 92)

    for idx, (vec_name, vec_req) in enumerate(vectors, start=1):
        print(f"\n>>> [{idx:02d}/{total_vecs:02d}] {vec_name} ...")
        resp = engine.synthesize(vec_req)
        if not (
            0.01 <= resp.mean_luminance <= 0.99
            and resp.rms_contrast > 0.01
            and resp.shannon_entropy_bits > 2.0
            and resp.laplacian_sharpness > 0.001
        ):
            raise RuntimeError(
                f"Qualitative telemetry anomaly on {vec_name}: "
                f"luma={resp.mean_luminance:.4f}, contrast={resp.rms_contrast:.4f}, "
                f"entropy={resp.shannon_entropy_bits:.3f}, sharpness={resp.laplacian_sharpness:.4f}"
            )
        if vec_req.prompt_mode == "layout_json" or vec_req.has_box_images():
            parsed_eff = json.loads(resp.effective_prompt)
            for item_obj in parsed_eff.get("items", []):
                extra_keys = set(item_obj.keys()) - {
                    "label",
                    "caption",
                    "bbox",
                    "text",
                    "tags",
                    "palette",
                }
                if extra_keys:
                    raise RuntimeError(
                        f"Out-of-distribution layout_json keys on {vec_name}: {extra_keys}"
                    )
        hw = engine.memory_guard.poll()
        rec = {
            "index": idx,
            "vector": vec_name,
            "model_variant": resp.model_variant_used,
            "output_path": resp.output_path,
            "latency_s": round(resp.generation_time_seconds, 3),
            "step_ms": round(resp.step_latency_ms, 2),
            "peak_vram_gb": round(resp.peak_vram_gb, 3),
            "wddm_dedicated_gb": round(hw.dedicated_used_gb, 3),
            "wddm_dedicated_avail_gb": round(hw.dedicated_avail_gb, 3),
            "wddm_contiguous_avail_gb": round(hw.contiguous_avail_gb, 3),
            "wddm_shared_delta_gb": round(hw.shared_delta_gb, 3),
            "pt_inactive_split_gb": round(hw.pt_inactive_split_gb, 4),
            "nfe": resp.total_nfe,
            "early_steps": resp.early_steps,
            "late_steps": resp.late_steps,
            "prompt_cache_hit": resp.prompt_cache_hit,
            "ref_cache_hit": resp.reference_cache_hit,
            "shared_prefix_tokens": resp.common_prefix_tokens,
            "mask_blend_mode": resp.mask_blend_mode_used,
            "invert_mask": resp.invert_inpaint_mask_used,
            "t_ref_zero": resp.reference_zero_timestep_used,
            "iso_kv": resp.reference_isolated_kv_used,
            "box_iso": round(resp.box_attn_isolation_used, 3),
            "uncond_refs": resp.uncond_uses_references_used,
            "composite_boxes": resp.composite_box_images_used,
            "lora_active": resp.edit_lora_active,
            "thinking_trace_mode": vec_req.thinking_trace,
            "entropy_bits": round(resp.shannon_entropy_bits, 3),
            "sharpness": round(resp.laplacian_sharpness, 4),
        }
        results.append(rec)
        print(
            f"    [OK] {vec_name:<52} | {resp.generation_time_seconds:6.2f}s | "
            f"VRAM={resp.peak_vram_gb:5.2f}GB (Contig={hw.contiguous_avail_gb:5.2f}GB, Spill=+{hw.shared_delta_gb:4.2f}GB) | "
            f"Entropy={resp.shannon_entropy_bits:5.2f}b | Pfx={resp.common_prefix_tokens}tok"
        )

    if engine._loaded_repo_id != SUPPORTED_MODELS["sft"]:
        engine._init_or_swap_components(SUPPORTED_MODELS["sft"], cpu_offload=False)

    manifest_path = PERMUTATIONS_OUT_DIR / "permutation_telemetry.json"
    manifest_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n" + "-" * 92)
    print(
        f"[+] {total_vecs}-Vector Permutation Suite completed in {time.perf_counter() - t_suite:.2f}s. "
        f"Telemetry saved to {manifest_path}"
    )
    print("=" * 92 + "\n")
    return results


def run_custom_sweep(engine: MOSSImageEngine, base_req: GenerationRequest) -> None:
    raw_shifts = input(
        "Enter comma-separated Shift values (Default: 3.0,5.0,7.0): "
    ).strip()
    shifts = (
        [float(x.strip()) for x in raw_shifts.split(",") if x.strip()]
        if raw_shifts
        else [3.0, 5.0, 7.0]
    )
    raw_cfgs = input("Enter comma-separated CFG values (Default: 2.5,3.5): ").strip()
    cfgs = (
        [float(x.strip()) for x in raw_cfgs.split(",") if x.strip()]
        if raw_cfgs
        else [2.5, 3.5]
    )
    SWEEPS_DIR.mkdir(parents=True, exist_ok=True)
    total = len(shifts) * len(cfgs)
    idx = 0
    for s_val in shifts:
        for c_val in cfgs:
            idx += 1
            out_file = SWEEPS_DIR / f"sweep_shift{s_val:.1f}_cfg{c_val:.1f}.png"
            print(
                f"\n>>> [Sweep {idx}/{total}] Shift={s_val:.2f} | CFG={c_val:.2f} -> {out_file.name}"
            )
            s_req = _clone_request(
                base_req,
                shift=s_val,
                early_cfg=c_val,
                late_cfg=c_val,
                output_path=str(out_file),
            )
            resp = engine.synthesize(s_req)
            render_telemetry_report(resp, engine=engine)


def interactive_loop(
    engine: MOSSImageEngine, initial_req: GenerationRequest, banner_tag: str
) -> None:
    req = initial_req
    active_tag = banner_tag
    while True:
        render_menu(req, active_tag)
        try:
            choice = input("Select action or field: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[+] Exiting ShadowBoxer Harness...")
            _graceful_exit(engine, 0)

        cmd = choice.lower()
        if not cmd:
            continue
        if cmd in ("q", "quit", "exit"):
            print("[+] Exiting ShadowBoxer Harness...")
            _graceful_exit(engine, 0)

        try:
            if cmd == "g":
                resp = engine.synthesize(req)
                render_telemetry_report(resp, engine=engine)
            elif cmd == "b":
                _interactive_edit_boxes(req)
            elif cmd == "w":
                engine._init_or_swap_components(
                    req.resolve_repo_id(), cpu_offload=False
                )
                engine.warmup(height=req.height, width=req.width)
            elif cmd == "ti":
                stage_test_input_files(force=True)
            elif cmd == "tp":
                run_permutation_suite(engine, req)
            elif cmd == "x":
                run_custom_sweep(engine, req)
            elif cmd == "p":
                print("\n" + "-" * 92)
                print("COMPILED EFFECTIVE PROMPT PREVIEW:")
                print("-" * 92)
                print(req.compile_effective_prompt())
                resolved_trc, spec_tok = req.resolve_thinking_trace()
                if resolved_trc:
                    print("-" * 92)
                    print("COMPILED THINKING TRACE PREVIEW:")
                    print(resolved_trc)
                elif spec_tok > 0:
                    print("-" * 92)
                    print(
                        f"LIVE SPECULATIVE COT MODE ACTIVE: {spec_tok} reasoning tokens"
                    )
                print("-" * 92)
            elif cmd == "t":
                req = build_default_request()
                active_tag = (
                    "ShadowBoxer/default.json (Active File)"
                    if has_custom_default_preset()
                    else "Discovered Optimal Baseline (V03 Heun->IPNDM)"
                )
            elif cmd == "d":
                save_request_json(req, DEFAULT_CONFIG_PATH)
                active_tag = f"ShadowBoxer/{DEFAULT_CONFIG_PATH.name} (Active File)"
                print(f"[+] Authoritative default.json updated at {DEFAULT_CONFIG_PATH}")
            elif cmd == "s":
                dest_in = input(
                    "Enter destination JSON path (Default: custom_config.json): "
                ).strip()
                dest_p = Path(dest_in or "custom_config.json")
                if not dest_p.is_absolute():
                    dest_p = ROOT_DIR / dest_p
                save_request_json(req, dest_p)
                active_tag = f"Saved: {dest_p.name}"
                print(f"[+] Saved configuration to {dest_p}")
            elif cmd == "l":
                src_in = input(
                    "Enter source JSON path (Default: default.json): "
                ).strip()
                src_p = Path(src_in or "default.json")
                if not src_p.is_absolute():
                    src_p = ROOT_DIR / src_p
                if src_p.is_file():
                    req, active_tag = load_or_create_initial_request(src_p)
                else:
                    print(f"[!] File not found: {src_p}")
            elif cmd == "m":
                req.model_variant = (
                    "base" if req.model_variant.lower() == "sft" else "sft"
                )
                req.repo_id = None
                active_tag = f"Model Switched -> {req.model_variant.upper()}"
            elif cmd == "1":
                req.prompt_mode = (
                    "layout_json"
                    if req.prompt_mode.lower() == "natural"
                    else "natural"
                )
                if not req.layout_items:
                    req.layout_items = scale_layout_items_to_canvas(
                        DEFAULT_LAYOUT_ITEMS,
                        height=req.height,
                        width=req.width,
                        src_height=1024,
                        src_width=1024,
                        coords_are_pixels=req.layout_coords_are_pixels,
                    )
            elif cmd == "2":
                new_p = input("Enter Positive Prompt (Blank to keep current): ").strip()
                if new_p:
                    req.prompt = new_p
            elif cmd == "3":
                new_np = input("Enter Negative Prompt (Blank to clear): ").strip()
                req.negative_prompt = new_np
            elif cmd == "4":
                req.enable_thinking = not req.enable_thinking
                ms_in = input(
                    f"Max Textual Sequence Length [64-32768] (Current: {req.max_sequence_length}, Enter to keep): "
                ).strip()
                if ms_in:
                    req.max_sequence_length = max(64, min(32768, int(ms_in)))
            elif cmd == "5":
                old_w, old_h = req.width, req.height
                w_in = input(f"Enter Width (Current: {req.width}): ").strip()
                h_in = input(f"Enter Height (Current: {req.height}): ").strip()
                if w_in:
                    req.width = max(256, min(2048, (int(w_in) // 16) * 16))
                if h_in:
                    req.height = max(256, min(2048, (int(h_in) // 16) * 16))
                if (req.width, req.height) != (old_w, old_h) and req.layout_items:
                    req.layout_items = scale_layout_items_to_canvas(
                        req.layout_items,
                        height=req.height,
                        width=req.width,
                        src_height=old_h,
                        src_width=old_w,
                        coords_are_pixels=req.layout_coords_are_pixels,
                    )
            elif cmd == "6":
                st_in = input(
                    f"Enter Inference Steps (Current: {req.num_inference_steps}): "
                ).strip()
                sh_in = input(f"Enter Shift (Current: {req.shift:.2f}): ").strip()
                v2_in = input(
                    f"Use V2 Nearest-Grid Schedule? [Y/n] (Current: {req.use_v2_grid}): "
                ).strip()
                if st_in:
                    req.num_inference_steps = max(1, int(st_in))
                if sh_in:
                    req.shift = float(sh_in)
                if v2_in:
                    req.use_v2_grid = v2_in.lower() not in ("n", "no", "0", "false")
            elif cmd == "7":
                print(f"Supported Solvers: {', '.join(SUPPORTED_SOLVERS)}")
                es_in = (
                    input(f"Early Solver (Current: {req.early_solver}): ")
                    .strip()
                    .lower()
                )
                ls_in = (
                    input(f"Late Solver (Current: {req.late_solver}): ")
                    .strip()
                    .lower()
                )
                ho_in = input(
                    f"Handoff Threshold t* [0.0-1.0] (Current: {req.handoff_threshold:.2f}): "
                ).strip()
                if es_in in SUPPORTED_SOLVERS:
                    req.early_solver = es_in
                if ls_in in SUPPORTED_SOLVERS:
                    req.late_solver = ls_in
                if ho_in:
                    req.handoff_threshold = max(0.0, min(1.0, float(ho_in)))
            elif cmd == "8":
                ec_in = input(
                    f"Early CFG Scale (Current: {req.early_cfg:.2f}): "
                ).strip()
                lc_in = input(
                    f"Late CFG Scale (Current: {req.late_cfg:.2f}): "
                ).strip()
                if ec_in:
                    req.early_cfg = float(ec_in)
                    req.guidance_scale = float(ec_in)
                if lc_in:
                    req.late_cfg = float(lc_in)
            elif cmd == "9":
                sk_in = input(
                    f"Negative Block Skip Layers comma-sep or 'none' (Current: {req.uncond_skip_layers}): "
                ).strip()
                if sk_in:
                    req.uncond_skip_layers = _parse_int_list(sk_in)
            elif cmd == "10":
                sl_in = input(
                    f"Positive SLG Layers comma-sep or 'none' (Current: {req.slg_layers}): "
                ).strip()
                if sl_in:
                    req.slg_layers = _parse_int_list(sl_in)
                if req.slg_layers:
                    sc_in = input(
                        f"SLG Scale (Current: {req.slg_scale:.2f}, 0.0=Off, Suggested: 0.50): "
                    ).strip()
                    if sc_in:
                        req.slg_scale = max(0.0, float(sc_in))
                else:
                    req.slg_scale = 0.0
            elif cmd == "11":
                eta_in = input(f"Early SDE Eta (Current: {req.eta:.3f}): ").strip()
                sn_in = input(
                    f"SDE S-Noise (Current: {req.s_noise:.3f}): "
                ).strip()
                if eta_in:
                    req.eta = max(0.0, min(1.0, float(eta_in)))
                if sn_in:
                    req.s_noise = max(0.0, float(sn_in))
            elif cmd == "12":
                ii_in = input(
                    f"Init Image Path or 'none' (Current: {req.init_image_path}): "
                ).strip()
                if ii_in.lower() == "none":
                    req.init_image_path = None
                elif ii_in:
                    req.init_image_path = ii_in
                dn_in = input(
                    f"Denoise Strength [0.01-1.0] (Current: {req.denoise_strength:.2f}): "
                ).strip()
                if dn_in:
                    req.denoise_strength = max(0.01, min(1.0, float(dn_in)))
                mk_in = input(
                    f"Inpaint Mask Path or 'none' (Current: {req.inpaint_mask_path}): "
                ).strip()
                if mk_in.lower() == "none":
                    req.inpaint_mask_path = None
                elif mk_in:
                    req.inpaint_mask_path = mk_in
                print(
                    f"Supported Mask Blend Modes: {', '.join(SUPPORTED_MASK_BLEND_MODES)}"
                )
                mb_in = (
                    input(
                        f"Mask Blend Mode (Current: {req.mask_blend_mode or 'auto'}): "
                    )
                    .strip()
                    .lower()
                )
                if mb_in in SUPPORTED_MASK_BLEND_MODES:
                    req.mask_blend_mode = mb_in
                inv_in = input(
                    f"Invert Composed Inpaint Mask? [y/N] (Current: {req.invert_inpaint_mask}): "
                ).strip()
                if inv_in:
                    req.invert_inpaint_mask = inv_in.lower() in (
                        "y",
                        "yes",
                        "1",
                        "true",
                    )
                pr_in = input(
                    f"Preserve Canvas Outside Layout Boxes? [y/N] (Current: {req.preserve_canvas_outside_boxes}): "
                ).strip()
                if pr_in:
                    req.preserve_canvas_outside_boxes = pr_in.lower() in (
                        "y",
                        "yes",
                        "1",
                        "true",
                    )
            elif cmd == "13":
                rf_in = input(
                    "Reference Image Paths comma-sep or 'none': "
                ).strip()
                if rf_in.lower() == "none":
                    req.reference_image_paths = []
                elif rf_in:
                    req.reference_image_paths = [
                        p.strip() for p in rf_in.split(",") if p.strip()
                    ]
                ct_in = input(
                    f"Control Map Path or 'none' (Current: {req.control_image_path}): "
                ).strip()
                if ct_in.lower() == "none":
                    req.control_image_path = None
                elif ct_in:
                    req.control_image_path = ct_in
            elif cmd == "14":
                print(f"Edit Formats: {', '.join(SUPPORTED_EDIT_FORMATS)}")
                ef_in = (
                    input(f"Edit Format (Current: {req.edit_format}): ")
                    .strip()
                    .lower()
                )
                if ef_in in SUPPORTED_EDIT_FORMATS:
                    req.edit_format = ef_in
                print(f"Framing Modes: {', '.join(SUPPORTED_FRAMING_MODES)}")
                fm_in = (
                    input(f"Framing Mode (Current: {req.framing_mode}): ")
                    .strip()
                    .lower()
                )
                if fm_in in SUPPORTED_FRAMING_MODES:
                    req.framing_mode = fm_in
                print(f"Input Roles: {', '.join(SUPPORTED_INPUT_ROLES)}")
                ir_in = (
                    input(f"Input Role (Current: {req.input_role}): ")
                    .strip()
                    .lower()
                )
                if ir_in in SUPPORTED_INPUT_ROLES:
                    req.input_role = ir_in
                vp_in = input(
                    f"Max ViT Patches [64-9216, 2304=768px, 576=384px] (Current: {req.max_vit_patches}): "
                ).strip()
                if vp_in:
                    req.max_vit_patches = max(64, min(9216, int(vp_in)))
            elif cmd == "15":
                mtf_in = input(
                    f"MTF Boost [0.0-0.5] (Current: {req.mtf_boost:.2f}): "
                ).strip()
                fth_in = input(
                    f"Mask Feather Radius (Current: {req.mask_feather_radius:.2f}): "
                ).strip()
                clp_in = input(
                    f"Latent Radial Clamp Sigma (Current: {req.latent_clamp_sigma:.2f}): "
                ).strip()
                lr_in = input(
                    f"Edit LoRA Path ('auto', path, or 'none') (Current: {req.edit_lora_path}): "
                ).strip()
                if mtf_in:
                    req.mtf_boost = float(mtf_in)
                if fth_in:
                    req.mask_feather_radius = float(fth_in)
                if clp_in:
                    req.latent_clamp_sigma = float(clp_in)
                if lr_in.lower() == "none":
                    req.edit_lora_path = None
                elif lr_in:
                    req.edit_lora_path = lr_in
                    sc_in = input(
                        f"Edit LoRA Scale (Current: {req.edit_lora_scale:.2f}): "
                    ).strip()
                    if sc_in:
                        req.edit_lora_scale = float(sc_in)
                    elif abs(req.edit_lora_scale) < 1e-6:
                        req.edit_lora_scale = 1.0
            elif cmd == "16":
                sd_in = input(f"PRNG Seed (Current: {req.seed}): ").strip()
                op_in = input(f"Output Path (Current: {req.output_path}): ").strip()
                if sd_in:
                    req.seed = int(sd_in)
                if op_in:
                    req.output_path = op_in
            elif cmd == "17":
                co_in = input(
                    f"Enable CPU Offload? [y/N] (Current: {req.cpu_offload}): "
                ).strip()
                vk_in = input(
                    f"Prefer Vulkan VizInjector? [Y/n] (Current: {req.prefer_vulkan}): "
                ).strip()
                rc_in = input(
                    f"Use Reference Vision Cache? [Y/n] (Current: {req.use_reference_cache}): "
                ).strip()
                if co_in:
                    req.cpu_offload = co_in.lower() in ("y", "yes", "1", "true")
                if vk_in:
                    req.prefer_vulkan = vk_in.lower() not in (
                        "n",
                        "no",
                        "0",
                        "false",
                    )
                if rc_in:
                    req.use_reference_cache = rc_in.lower() not in (
                        "n",
                        "no",
                        "0",
                        "false",
                    )
            elif cmd == "18":
                iso_in = input(
                    f"Box Attention Isolation [0.0-1.0] (Current: {float(req.box_attn_isolation or 0.0):.2f}): "
                ).strip()
                spl_in = input(
                    f"Composite Box Images to Init Canvas? [y/N] (Current: {req.composite_box_images_to_init}): "
                ).strip()
                unc_in = input(
                    f"Uncond Uses References? ['auto', 'true', 'false'] (Current: {req.uncond_uses_references}): "
                ).strip()
                sys_in = input(
                    "System Prompt ('none' to clear, Blank to keep): "
                ).strip()
                trc_in = input(
                    "Thinking Trace ('auto', 'speculative:64', text, or 'none' to clear): "
                ).strip()
                vid_in = input(
                    f"Add 'Picture k:' Vision ID Prefix? [Y/n] (Current: {req.add_vision_id}): "
                ).strip()
                if iso_in:
                    req.box_attn_isolation = max(0.0, min(1.0, float(iso_in)))
                if spl_in:
                    req.composite_box_images_to_init = spl_in.lower() in (
                        "y",
                        "yes",
                        "1",
                        "true",
                    )
                if unc_in:
                    if unc_in.lower() == "auto":
                        req.uncond_uses_references = None
                    else:
                        req.uncond_uses_references = unc_in.lower() in (
                            "y",
                            "yes",
                            "1",
                            "true",
                        )
                if sys_in.lower() == "none":
                    req.system_prompt = None
                elif sys_in:
                    req.system_prompt = sys_in
                if trc_in.lower() == "none":
                    req.thinking_trace = None
                elif trc_in:
                    req.thinking_trace = trc_in
                if vid_in:
                    req.add_vision_id = vid_in.lower() not in (
                        "n",
                        "no",
                        "0",
                        "false",
                    )
            else:
                print(f"[!] Unrecognized option: '{choice}'")
        except Exception as e:
            print(f"[!] Operation error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ShadowBoxer: OpenMOSS Architectural Exploration Harness"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--test_permutations", action="store_true")
    parser.add_argument("--perm_steps", type=int, default=None)
    parser.add_argument("--stage_inputs", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--thinking_trace", type=str, default=None)
    parser.add_argument("--box_attn_isolation", type=float, default=None)
    parser.add_argument("--box_denoise", type=str, default=None)
    parser.add_argument("--composite_box_images", action="store_true")
    parser.add_argument(
        "--mask_blend_mode",
        type=str,
        choices=SUPPORTED_MASK_BLEND_MODES,
        default=None,
    )
    parser.add_argument("--invert_mask", action="store_true")
    parser.add_argument(
        "--model_variant",
        "--variant",
        dest="model_variant",
        type=str,
        choices=list(SUPPORTED_MODELS.keys()),
        default=None,
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_vit_patches", type=int, default=None)
    parser.add_argument("--max_sequence_length", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--save_preset", type=str, default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else None
    if cfg_path is not None and not cfg_path.is_absolute():
        cfg_path = ROOT_DIR / cfg_path

    req, banner_tag = load_or_create_initial_request(cfg_path)
    orig_w, orig_h = req.width, req.height
    if args.prompt is not None:
        req.prompt = args.prompt
    if args.system_prompt is not None:
        req.system_prompt = args.system_prompt
    if args.thinking_trace is not None:
        req.thinking_trace = args.thinking_trace
    if args.box_attn_isolation is not None:
        req.box_attn_isolation = max(0.0, min(1.0, float(args.box_attn_isolation)))
    if args.box_denoise is not None:
        _apply_box_denoise_spec(req, args.box_denoise)
    if args.composite_box_images:
        req.composite_box_images_to_init = True
    if args.mask_blend_mode is not None:
        req.mask_blend_mode = args.mask_blend_mode
    if args.invert_mask:
        req.invert_inpaint_mask = True
    if args.model_variant is not None:
        req.model_variant = args.model_variant
        req.repo_id = None
    if args.width is not None:
        req.width = max(256, min(2048, (int(args.width) // 16) * 16))
    if args.height is not None:
        req.height = max(256, min(2048, (int(args.height) // 16) * 16))
    if (req.width, req.height) != (orig_w, orig_h) and req.layout_items:
        req.layout_items = scale_layout_items_to_canvas(
            req.layout_items,
            height=req.height,
            width=req.width,
            src_height=orig_h,
            src_width=orig_w,
            coords_are_pixels=req.layout_coords_are_pixels,
        )
    if args.steps is not None:
        req.num_inference_steps = args.steps
    if args.seed is not None:
        req.seed = args.seed
    if args.max_vit_patches is not None:
        req.max_vit_patches = max(64, min(9216, int(args.max_vit_patches)))
    if args.max_sequence_length is not None:
        req.max_sequence_length = max(64, min(32768, int(args.max_sequence_length)))
    if args.output is not None:
        req.output_path = args.output
    if args.cpu_offload:
        req.cpu_offload = True

    if args.save_preset:
        target = Path(args.save_preset)
        if not target.is_absolute():
            target = ROOT_DIR / target
        save_request_json(req, target)
        print(f"[+] Preset exported to {target}")
        return

    if args.stage_inputs and not (args.generate or args.test_permutations):
        stage_test_input_files(force=True)
        return

    exec_dev = "cuda" if torch.cuda.is_available() else "cpu"
    engine = MOSSImageEngine(
        model_variant=req.model_variant,
        device=exec_dev,
        prefer_vulkan=req.prefer_vulkan,
    )

    if args.warmup:
        engine._init_or_swap_components(req.resolve_repo_id(), cpu_offload=False)
        engine.warmup(height=req.height, width=req.width)

    if args.test_permutations:
        try:
            eff_perm_steps = (
                args.perm_steps if args.perm_steps is not None else args.steps
            )
            run_permutation_suite(
                engine,
                req,
                steps_override=eff_perm_steps,
                width_override=args.width,
                height_override=args.height,
            )
        finally:
            _graceful_exit(engine, 0)

    if args.generate:
        try:
            resp = engine.synthesize(req)
            render_telemetry_report(resp, engine=engine)
        finally:
            _graceful_exit(engine, 0)

    interactive_loop(engine, req, banner_tag)


if __name__ == "__main__":
    main()