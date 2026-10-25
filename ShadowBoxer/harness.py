#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, fields, is_dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

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
import schema
from schema import (
    GenerationRequest,
    GenerationResponse,
    SUPPORTED_MODELS,
    SUPPORTED_SOLVERS,
)

DEFAULT_CONFIG_PATH = ROOT_DIR / "default.json"
TEST_INPUTS_DIR = ROOT_DIR / "artifacts" / "test_inputs"
PERMUTATIONS_OUT_DIR = ROOT_DIR / "artifacts" / "permutations"

DEFAULT_POSITIVE_PROMPT = (
    "A wide view of a sleek brushed-steel refrigerator beside a warm walnut kitchen counter "
    "with a ceramic bowl of fresh citrus fruit. A small yellow sticky note on the refrigerator "
    'door is handwritten in crisp black marker: "Remember to test SFT & Base permutations @ 1024x1024"'
)

DEFAULT_LAYOUT_ITEMS: List[Dict[str, Any]] = getattr(
    schema,
    "DEFAULT_LAYOUT_ITEMS",
    getattr(
        schema,
        "DEFAULT_LAYOUT_BOXES",
        [
            {
                "category": "refrigerator",
                "caption": "A sleek brushed-steel refrigerator door with a small yellow sticky note",
                "bbox": [32, 48, 992, 592],
            },
            {
                "category": "text",
                "text": "Remember to test SFT & Base permutations @ 1024x1024",
                "bbox": [288, 144, 544, 480],
            },
            {
                "category": "bowl",
                "caption": "A ceramic bowl of fresh citrus fruit on a warm walnut kitchen counter",
                "bbox": [592, 608, 976, 976],
            },
        ],
    ),
)

SUPPORTED_EDIT_FORMATS: Tuple[str, ...] = getattr(
    schema,
    "SUPPORTED_EDIT_FORMATS",
    (
        "standard",
        "moss_image_edit",
        "ai_toolkit_isolated",
        "ai_toolkit_full",
        "control_color_depth",
    ),
)

SUPPORTED_FRAMING_MODES: Tuple[str, ...] = getattr(
    schema,
    "SUPPORTED_FRAMING_MODES",
    ("inscribed", "cover_crop", "never_crop"),
)

SUPPORTED_INPUT_ROLES: Tuple[str, ...] = getattr(
    schema,
    "SUPPORTED_INPUT_ROLES",
    ("photo_ref", "color_map", "depth_map", "inpaint_mask"),
)


def _valid_request_fields() -> Set[str]:
    model_fields = getattr(GenerationRequest, "model_fields", None)
    if isinstance(model_fields, dict):
        return set(model_fields.keys())
    v1_fields = getattr(GenerationRequest, "__fields__", None)
    if isinstance(v1_fields, dict):
        return set(v1_fields.keys())
    if is_dataclass(GenerationRequest):
        return {f.name for f in fields(GenerationRequest)}
    return set()


def _normalize_request_kwargs(raw: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(raw)
    valid = _valid_request_fields()
    if "layout_boxes" in data and "layout_items" not in data:
        if not valid or "layout_items" in valid:
            data["layout_items"] = data.pop("layout_boxes")
    if "layout_coords_are_pixels" in data and "coords_are_pixels" in valid:
        data["coords_are_pixels"] = data.pop("layout_coords_are_pixels")
    if valid:
        return {k: v for k, v in data.items() if k in valid}
    return data


def _get_layout_items(req: GenerationRequest) -> List[Dict[str, Any]]:
    items = getattr(req, "layout_items", None)
    if items is None:
        items = getattr(req, "layout_boxes", None)
    return list(items) if items else []


def _set_layout_items(req: GenerationRequest, items: List[Dict[str, Any]]) -> None:
    valid = _valid_request_fields()
    if "layout_items" in valid or hasattr(req, "layout_items"):
        setattr(req, "layout_items", items)
    elif "layout_boxes" in valid or hasattr(req, "layout_boxes"):
        setattr(req, "layout_boxes", items)


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
    kwargs = _normalize_request_kwargs(
        {
            "prompt": DEFAULT_POSITIVE_PROMPT,
            "negative_prompt": "",
            "model_variant": "sft",
            "prompt_mode": "natural",
            "layout_items": [dict(b) for b in DEFAULT_LAYOUT_ITEMS],
            "layout_coords_are_pixels": True,
            "enable_thinking": True,
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 50,
            "shift": 5.0,
            "use_v2_grid": True,
            "early_solver": "euler",
            "late_solver": "euler",
            "handoff_threshold": 0.35,
            "early_cfg": 3.0,
            "late_cfg": 3.0,
            "uncond_skip_layers": [24],
            "slg_layers": [],
            "slg_scale": 0.0,
            "slg_start": 0.05,
            "slg_stop": 0.60,
            "eta": 0.0,
            "s_noise": 1.0,
            "seed": 42,
            "output_path": "output_moss_master.png",
            "cpu_offload": False,
            "prefer_vulkan": True,
            "use_reference_cache": True,
            "prepass_warmup": True,
        }
    )
    return GenerationRequest(**kwargs)


def request_to_dict(req: GenerationRequest) -> Dict[str, Any]:
    if hasattr(req, "to_dict") and callable(req.to_dict):
        return req.to_dict()
    if hasattr(req, "model_dump") and callable(req.model_dump):
        return req.model_dump()
    if is_dataclass(req):
        return asdict(req)
    return dict(req.__dict__)


def request_from_dict(data: Dict[str, Any]) -> GenerationRequest:
    norm_data = _normalize_request_kwargs(data)
    if hasattr(GenerationRequest, "from_dict") and callable(
        getattr(GenerationRequest, "from_dict")
    ):
        return GenerationRequest.from_dict(norm_data)
    if hasattr(GenerationRequest, "model_validate") and callable(
        getattr(GenerationRequest, "model_validate")
    ):
        return GenerationRequest.model_validate(norm_data)
    return GenerationRequest(**norm_data)


def _clone_request(req: GenerationRequest, **updates: Any) -> GenerationRequest:
    norm_updates = _normalize_request_kwargs(updates)
    if hasattr(req, "model_copy") and callable(req.model_copy):
        cloned = req.model_copy(update=norm_updates, deep=True)
        return cloned
    merged = request_to_dict(req)
    merged.update(norm_updates)
    return request_from_dict(merged)


def load_or_create_initial_request(
    config_path: Optional[Path] = None,
) -> Tuple[GenerationRequest, str]:
    target = config_path or DEFAULT_CONFIG_PATH
    if target.is_file():
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
            return request_from_dict(raw), f"Loaded: {target.name}"
        except Exception as e:
            print(f"[!] Warning: Failed to parse {target}: {e}. Using baseline.")
    return build_default_request(), "Discovered Optimal Baseline"


def save_request_json(req: GenerationRequest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request_to_dict(req), indent=2), encoding="utf-8")


def _truncate(text: str, max_len: int = 68) -> str:
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


def render_menu(req: GenerationRequest, banner_tag: str) -> None:
    repo_str = SUPPORTED_MODELS.get(req.model_variant.lower(), req.repo_id or "Custom")
    n_boxes = len(_get_layout_items(req))
    think_str = "ENABLED (<think>)" if req.enable_thinking else "DISABLED (</think>)"
    slg_str = (
        f"Layers {req.slg_layers} | Scale: {req.slg_scale:.2f} [{req.slg_start:.2f}-{req.slg_stop:.2f}]"
        if (req.slg_layers and req.slg_scale > 0.0)
        else "DISABLED"
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

    print("\n" + "=" * 92)
    print("               SHADOWBOXER: OPENMOSS ARCHITECTURAL EXPLORATION HARNESS")
    print(f"                              [{banner_tag}]")
    print("=" * 92)
    print(" --- MODEL & PROMPT CONDITIONING ---")
    print(f" [M]  Active Model Variant:   {req.model_variant.upper()} ({repo_str})")
    print(f" [1]  Prompt Mode:            {req.prompt_mode.upper()} ({n_boxes} layout boxes)")
    print(f" [2]  Positive Prompt:        {_truncate(req.prompt, 65)}")
    print(f" [3]  Negative Prompt:        {_truncate(req.negative_prompt, 65)}")
    print(f" [4]  Qwen3.5 Thinking Mode:  {think_str}")
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
        f" [12] Init Image & Inpaint:   Init: {_short_path(req.init_image_path)} (Denoise: {req.denoise_strength:.2f}) | Mask: {_short_path(req.inpaint_mask_path)}"
    )
    print(
        f" [13] Multi-Reference (4D):   {refs_str} | Control Map: {_short_path(req.control_image_path)}"
    )
    print(
        f" [14] Edit & Framing Routing: Edit: {req.edit_format} | Framing: {req.framing_mode} | Role: {req.input_role}"
    )
    print(
        f" [15] Radiometric & LoRA:     MTF: {req.mtf_boost:.2f} | Feather: {req.mask_feather_radius:.2f} | Clamp: {req.latent_clamp_sigma:.1f}s | LoRA: {lora_str}"
    )
    print(" --- HARDWARE, SEED & EXECUTION ---")
    print(f" [16] PRNG Seed & Output:     Seed: {req.seed} | Path: {req.output_path}")
    print(
        f" [17] Memory & Backends:      CPU Offload: {req.cpu_offload} | Vulkan: {req.prefer_vulkan} | RefCache: {req.use_reference_cache}"
    )
    print("-" * 92)
    print(
        " [G]  Generate Master Image   [TP] Run 10-Vector Permutation Suite   [X] Custom Sweep Matrix"
    )
    print(
        " [TI] Stage Test Input Files  [P]  Preview Prompt  [T] Reset  [L] Load  [S] Save  [D] Save default.json"
    )
    print(" [W]  Run Prepass Warmup      [Q]  Quit Harness")
    print("=" * 92)


def render_telemetry_report(
    resp: GenerationResponse, engine: Optional[MOSSImageEngine] = None
) -> None:
    vram_mode = "CPU OFFLOAD" if resp.cpu_offload_active else "RESIDENT VRAM"
    sched_str = "MOSS V2 NEAREST-GRID" if resp.use_v2_grid_used else "CONTINUOUS SHIFT"
    lat_h, lat_w = resp.height // 16, resp.width // 16

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
                f"(Avail: {hw.dedicated_avail_gb:.2f} GB) | Shared Spill Delta: +{hw.shared_delta_gb:.2f} GB"
            )
    print(
        f"Tokenizer Engine:       SOVEREIGN VIZTOKEN (C11 AVX2 SWISS5) | Prompt Cache Hit: {resp.prompt_cache_hit} (Shared Prefix: {resp.common_prefix_tokens} tok)"
    )
    print(
        f"Vision Conditioner:     {resp.vizinjector_backend} | Ref Cache Hit: {resp.reference_cache_hit}"
    )
    print(
        f"Conditioning Routing:   Edit={resp.edit_format_used} | Framing={resp.framing_mode_used} | Role={resp.input_role_used} | t_ref=0: {resp.reference_zero_timestep_used} | IsoKV: {resp.reference_isolated_kv_used} | LoRA: {resp.edit_lora_active}"
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


def stage_test_input_files(dest_dir: Path = TEST_INPUTS_DIR) -> Dict[str, Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    h, w = 512, 512
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, h, dtype=np.float32),
        np.linspace(-1.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )

    rad = np.sqrt(xx * xx + yy * yy)
    ref_a = np.zeros((h, w, 3), dtype=np.uint8)
    ref_a[..., 0] = np.clip((0.55 + 0.40 * np.cos(3.0 * xx)) * 255.0, 0, 255).astype(
        np.uint8
    )
    ref_a[..., 1] = np.clip((0.50 + 0.40 * np.sin(3.0 * yy)) * 255.0, 0, 255).astype(
        np.uint8
    )
    ref_a[..., 2] = np.clip((1.0 - 0.55 * rad) * 255.0, 0, 255).astype(np.uint8)
    p_ref_a = dest_dir / "ref_subject_a.png"
    Image.fromarray(ref_a, mode="RGB").save(p_ref_a)

    ref_b = np.zeros((h, w, 3), dtype=np.uint8)
    ref_b[..., 0] = np.clip((0.65 - 0.35 * yy) * 255.0, 0, 255).astype(np.uint8)
    ref_b[..., 1] = np.clip((0.45 + 0.35 * xx) * 255.0, 0, 255).astype(
        np.uint8
    )
    ref_b[..., 2] = np.clip(
        (0.35 + 0.35 * np.sin(4.0 * (xx + yy))) * 255.0, 0, 255
    ).astype(np.uint8)
    p_ref_b = dest_dir / "ref_style_b.png"
    Image.fromarray(ref_b, mode="RGB").save(p_ref_b)

    init_img = np.zeros((h, w, 3), dtype=np.uint8)
    init_img[..., 0] = np.clip((0.48 + 0.25 * xx) * 255.0, 0, 255).astype(np.uint8)
    init_img[..., 1] = np.clip((0.52 - 0.20 * yy) * 255.0, 0, 255).astype(np.uint8)
    init_img[..., 2] = np.clip((0.56 + 0.15 * (xx * yy)) * 255.0, 0, 255).astype(
        np.uint8
    )
    p_init = dest_dir / "init_canvas.png"
    Image.fromarray(init_img, mode="RGB").save(p_init)

    mask = np.clip((0.55 - rad) * 4.0, 0.0, 1.0)
    mask_u8 = (mask * 255.0).round().astype(np.uint8)
    p_mask = dest_dir / "inpaint_mask.png"
    Image.fromarray(mask_u8, mode="L").save(p_mask)

    ctrl = np.zeros((h, w, 3), dtype=np.uint8)
    ctrl_val = np.clip((1.0 - np.minimum(rad, 1.0)) * 255.0, 0, 255).astype(np.uint8)
    ctrl[..., 0] = ctrl_val
    ctrl[..., 1] = np.clip((0.5 + 0.5 * xx) * 255.0, 0, 255).astype(np.uint8)
    ctrl[..., 2] = np.clip((0.5 + 0.5 * yy) * 255.0, 0, 255).astype(np.uint8)
    p_ctrl = dest_dir / "control_map.png"
    Image.fromarray(ctrl, mode="RGB").save(p_ctrl)

    print(f"[+] Staged 5 deterministic test input artifacts in: {dest_dir}")
    return {
        "ref_a": p_ref_a,
        "ref_b": p_ref_b,
        "init": p_init,
        "mask": p_mask,
        "control": p_ctrl,
    }


def _parse_int_list(raw: str) -> List[int]:
    s = raw.strip()
    if not s or s.lower() in ("none", "[]", "clear"):
        return []
    parts = [p.strip() for p in s.replace("[", "").replace("]", "").split(",") if p.strip()]
    return [int(p) for p in parts]


def run_permutation_suite(
    engine: MOSSImageEngine,
    base_req: GenerationRequest,
    steps_override: Optional[int] = None,
    width_override: Optional[int] = None,
    height_override: Optional[int] = None,
) -> List[Dict[str, Any]]:
    staged = stage_test_input_files()
    PERMUTATIONS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    w = width_override or base_req.width
    h = height_override or base_req.height
    steps = steps_override or min(base_req.num_inference_steps, 12)

    models_info = {m["variant"]: m["cached"] for m in engine.list_available_models()}
    has_base = bool(models_info.get("base", False))

    base_clean = _clone_request(
        base_req,
        width=w,
        height=h,
        num_inference_steps=steps,
        init_image_path=None,
        inpaint_mask_path=None,
        control_image_path=None,
        reference_image_paths=[],
        export_vizj_path=None,
        edit_format="standard",
        framing_mode="inscribed",
        input_role="photo_ref",
        edit_lora_path=None,
        edit_lora_scale=0.0,
        layout_items=_get_layout_items(base_req) or [dict(b) for b in DEFAULT_LAYOUT_ITEMS],
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
            "V02_SFT_Layout_JSON_Centralized",
            _clone_request(
                base_clean,
                model_variant="sft",
                prompt_mode="layout_json",
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
                output_path=str(PERMUTATIONS_OUT_DIR / "v03_bifurcated_heun_ipndm.png"),
            ),
        ),
        (
            "V04_SDE_Dispersion_And_SLG",
            _clone_request(
                base_clean,
                early_solver="sde_gpu_pp",
                late_solver="ipndm",
                handoff_threshold=0.40,
                eta=0.12,
                s_noise=1.0,
                slg_layers=[22],
                slg_scale=1.25,
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
            "V06_Img2Img_Feather_Inpaint",
            _clone_request(
                base_clean,
                init_image_path=str(staged["init"]),
                inpaint_mask_path=str(staged["mask"]),
                denoise_strength=0.75,
                output_path=str(PERMUTATIONS_OUT_DIR / "v06_img2img_inpaint.png"),
            ),
        ),
        (
            "V07_MossImageEdit_ZeroTimestep",
            _clone_request(
                base_clean,
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
                control_image_path=str(staged["control"]),
                edit_format="control_color_depth",
                input_role="color_map",
                output_path=str(PERMUTATIONS_OUT_DIR / "v09_control_map.png"),
            ),
        ),
        (
            "V10_Base_Foundation_HotSwap" if has_base else "V10_Tier2_VIZJ_ZeroCopy_Reload",
            _clone_request(
                base_clean,
                model_variant="base" if has_base else "sft",
                reference_image_paths=(
                    []
                    if has_base
                    else [str(PERMUTATIONS_OUT_DIR / "v08_tier2_bundle.vizj")]
                ),
                edit_format="standard" if has_base else "ai_toolkit_isolated",
                output_path=str(PERMUTATIONS_OUT_DIR / "v10_final_vector.png"),
            ),
        ),
    ]

    results: List[Dict[str, Any]] = []
    t_suite = time.perf_counter()
    print("\n" + "=" * 92)
    print(
        f"     EXECUTING 10-VECTOR ARCHITECTURAL PERMUTATION SUITE ({w}x{h} @ {steps} steps)"
    )
    print("=" * 92)

    for idx, (vec_name, vec_req) in enumerate(vectors, start=1):
        print(f"\n>>> [{idx:02d}/10] {vec_name} ...")
        resp = engine.synthesize(vec_req)
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
            "wddm_shared_delta_gb": round(hw.shared_delta_gb, 3),
            "nfe": resp.total_nfe,
            "prompt_cache_hit": resp.prompt_cache_hit,
            "ref_cache_hit": resp.reference_cache_hit,
            "shared_prefix_tokens": resp.common_prefix_tokens,
            "t_ref_zero": resp.reference_zero_timestep_used,
            "iso_kv": resp.reference_isolated_kv_used,
            "lora_active": resp.edit_lora_active,
            "entropy_bits": round(resp.shannon_entropy_bits, 3),
            "sharpness": round(resp.laplacian_sharpness, 4),
        }
        results.append(rec)
        print(
            f"    [OK] {vec_name:<38} | {resp.generation_time_seconds:6.2f}s | "
            f"VRAM={resp.peak_vram_gb:5.2f}GB (Spill=+{hw.shared_delta_gb:4.2f}GB) | "
            f"Entropy={resp.shannon_entropy_bits:5.2f}b | Pfx={resp.common_prefix_tokens}tok"
        )

    manifest_path = PERMUTATIONS_OUT_DIR / "permutation_telemetry.json"
    manifest_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n" + "-" * 92)
    print(
        f"[+] 10-Vector Permutation Suite completed in {time.perf_counter() - t_suite:.2f}s. "
        f"Telemetry saved to {manifest_path}"
    )
    print("=" * 92 + "\n")
    return results


def run_custom_sweep(engine: MOSSImageEngine, base_req: GenerationRequest) -> None:
    raw_shifts = input("Enter comma-separated Shift values (Default: 3.0,5.0,7.0): ").strip()
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

    sweep_dir = ROOT_DIR / "artifacts" / "sweeps"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    total = len(shifts) * len(cfgs)
    idx = 0
    for s_val in shifts:
        for c_val in cfgs:
            idx += 1
            out_file = sweep_dir / f"sweep_shift{s_val:.1f}_cfg{c_val:.1f}.png"
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
            elif cmd == "w":
                engine._init_or_swap_components(
                    req.resolve_repo_id(), cpu_offload=False
                )
                engine.warmup(height=req.height, width=req.width)
            elif cmd == "ti":
                stage_test_input_files()
            elif cmd == "tp":
                run_permutation_suite(engine, req)
            elif cmd == "x":
                run_custom_sweep(engine, req)
            elif cmd == "p":
                print("\n" + "-" * 92)
                print("COMPILED EFFECTIVE PROMPT PREVIEW:")
                print("-" * 92)
                print(req.compile_effective_prompt())
                print("-" * 92)
            elif cmd == "t":
                req = build_default_request()
                active_tag = "Discovered Optimal Baseline (Reset)"
            elif cmd == "d":
                save_request_json(req, DEFAULT_CONFIG_PATH)
                active_tag = f"Saved: {DEFAULT_CONFIG_PATH.name}"
                print(f"[+] Saved configuration to {DEFAULT_CONFIG_PATH}")
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
                if not _get_layout_items(req):
                    _set_layout_items(req, [dict(b) for b in DEFAULT_LAYOUT_ITEMS])
            elif cmd == "2":
                new_p = input("Enter Positive Prompt (Blank to keep current): ").strip()
                if new_p:
                    req.prompt = new_p
            elif cmd == "3":
                new_np = input("Enter Negative Prompt (Blank to clear): ").strip()
                req.negative_prompt = new_np
            elif cmd == "4":
                req.enable_thinking = not req.enable_thinking
            elif cmd == "5":
                w_in = input(f"Enter Width (Current: {req.width}): ").strip()
                h_in = input(f"Enter Height (Current: {req.height}): ").strip()
                if w_in:
                    req.width = max(64, (int(w_in) // 16) * 16)
                if h_in:
                    req.height = max(64, (int(h_in) // 16) * 16)
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
                        f"SLG Scale (Current: {req.slg_scale:.2f}, Suggested: 1.25): "
                    ).strip()
                    req.slg_scale = float(sc_in) if sc_in else 1.25
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
                    f"Denoise Strength [0.05-1.0] (Current: {req.denoise_strength:.2f}): "
                ).strip()
                if dn_in:
                    req.denoise_strength = max(0.05, min(1.0, float(dn_in)))
                mk_in = input(
                    f"Inpaint Mask Path or 'none' (Current: {req.inpaint_mask_path}): "
                ).strip()
                if mk_in.lower() == "none":
                    req.inpaint_mask_path = None
                elif mk_in:
                    req.inpaint_mask_path = mk_in
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
    parser.add_argument("--stage_inputs", action="store_true")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--model_variant", type=str, choices=["sft", "base"], default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--cpu_offload", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else None
    req, banner_tag = load_or_create_initial_request(cfg_path)

    if args.prompt is not None:
        req.prompt = args.prompt
    if args.model_variant is not None:
        req.model_variant = args.model_variant
        req.repo_id = None
    if args.width is not None:
        req.width = args.width
    if args.height is not None:
        req.height = args.height
    if args.steps is not None:
        req.num_inference_steps = args.steps
    if args.seed is not None:
        req.seed = args.seed
    if args.output is not None:
        req.output_path = args.output
    if args.cpu_offload:
        req.cpu_offload = True

    if args.stage_inputs and not (args.generate or args.test_permutations):
        stage_test_input_files()
        return

    exec_dev = "cuda" if torch.cuda.is_available() else "cpu"
    engine = MOSSImageEngine(
        model_variant=req.model_variant,
        device=exec_dev,
        prefer_vulkan=req.prefer_vulkan,
    )

    if args.test_permutations:
        try:
            run_permutation_suite(
                engine,
                req,
                steps_override=args.steps,
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