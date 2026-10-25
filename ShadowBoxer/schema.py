from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator

SHADOWBOXER_ROOT = Path(__file__).resolve().parent
DEFAULT_PRESET_FILENAME = "default.json"

SUPPORTED_SOLVERS = ["euler", "heun", "ipndm", "sde_gpu_pp", "multitree"]
SUPPORTED_MODELS = {
    "sft": "OpenMOSS-Team/MOSS-Image-SFT",
    "base": "OpenMOSS-Team/MOSS-Image-Base",
}
SUPPORTED_PROMPT_MODES = ["natural", "layout", "layout_json"]
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

BASELINE_ENGINE_DEFAULTS: Dict[str, Any] = {
    "model_variant": "sft",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "shift": 5.0,
    "use_v2_grid": True,
    "early_solver": "euler",
    "late_solver": "euler",
    "handoff_threshold": 0.35,
    "guidance_scale": 3.0,
    "early_cfg": 3.0,
    "late_cfg": 3.0,
    "uncond_skip_layers": [22],
    "slg_layers": [],
    "slg_scale": 0.0,
    "slg_start": 0.0,
    "slg_stop": 1.0,
    "eta": 0.0,
    "s_noise": 1.0,
    "max_sequence_length": 4096,
    "enable_thinking": True,
    "denoise_strength": 1.0,
    "framing_mode": "inscribed",
    "input_role": "photo_ref",
    "edit_format": "standard",
    "max_vit_patches": 2304,
    "deband_8bit": True,
    "prefer_vulkan": True,
    "mtf_boost": 0.08,
    "soft_knee_tau": 0.015,
    "mask_feather_radius": 1.25,
    "latent_clamp_sigma": 4.5,
    "apply_bayer_dither": True,
    "use_reference_cache": True,
    "cpu_offload": False,
    "prepass_warmup": True,
}

_PRESET_CACHE: Dict[str, Any] = {
    "mtime": -1.0,
    "path": None,
    "has_custom_default": False,
    "defaults": dict(BASELINE_ENGINE_DEFAULTS),
}


def pixel_bbox_to_abs_centralized(
    ymin: float,
    xmin: float,
    ymax: float,
    xmax: float,
    height: int,
    width: int,
) -> List[int]:
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    return [
        int(round(ymin - cy)),
        int(round(xmin - cx)),
        int(round(ymax - cy)),
        int(round(xmax - cx)),
    ]


def compile_layout_prompt(
    caption: str,
    items: Sequence[Dict[str, Any]],
    height: int = 1024,
    width: int = 1024,
    coords_are_pixels: bool = False,
) -> str:
    normalized_items = []
    for item in items:
        entry = dict(item)
        bbox = entry.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            if coords_are_pixels:
                entry["bbox"] = pixel_bbox_to_abs_centralized(
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                    height=height,
                    width=width,
                )
            else:
                entry["bbox"] = [int(round(float(v))) for v in bbox]
        normalized_items.append(entry)

    layout_payload = {
        "bbox_format": "yxyx",
        "bbox_coord_space": "abs_centralized",
        "summary": {"caption": caption.strip()},
        "items": normalized_items,
    }
    return json.dumps(layout_payload, ensure_ascii=False)


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

    if "uncond_skip_layers" in data:
        harvested["uncond_skip_layers"] = parse_layer_list(data["uncond_skip_layers"])
    if "slg_layers" in data:
        harvested["slg_layers"] = parse_layer_list(data["slg_layers"])

    if "slg_scale" in data and data["slg_scale"] is not None:
        try:
            harvested["slg_scale"] = max(0.0, min(20.0, float(data["slg_scale"])))
        except (ValueError, TypeError):
            pass

    for b_key in ("slg_start", "slg_stop", "eta"):
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

    if "max_sequence_length" in data and data["max_sequence_length"] is not None:
        try:
            harvested["max_sequence_length"] = max(
                64, min(4096, int(data["max_sequence_length"]))
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
    ):
        if flt_key in data and data[flt_key] is not None:
            try:
                harvested[flt_key] = max(lo, min(hi, float(data[flt_key])))
            except (ValueError, TypeError):
                pass

    for flag_key in (
        "enable_thinking",
        "cpu_offload",
        "prepass_warmup",
        "deband_8bit",
        "prefer_vulkan",
        "apply_bayer_dither",
        "use_reference_cache",
    ):
        if flag_key in data and data[flag_key] is not None:
            harvested[flag_key] = bool(data[flag_key])

    return harvested


def locate_default_preset_file() -> Optional[Path]:
    candidates = [
        SHADOWBOXER_ROOT / DEFAULT_PRESET_FILENAME,
        SHADOWBOXER_ROOT / "presets" / DEFAULT_PRESET_FILENAME,
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

    if _PRESET_CACHE["path"] == preset_file and _PRESET_CACHE["mtime"] == current_mtime:
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
    prompt_mode: str = Field(default="natural")
    layout_items: List[Dict[str, Any]] = Field(default_factory=list)
    layout_coords_are_pixels: bool = Field(default=False)
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
    max_sequence_length: Optional[int] = Field(default=None, ge=64, le=4096)
    enable_thinking: Optional[bool] = Field(default=None)
    init_image_path: Optional[str] = Field(default=None)
    denoise_strength: float = Field(default=1.0, ge=0.01, le=1.0)
    inpaint_mask_path: Optional[str] = Field(default=None)
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
    edit_lora_path: Optional[str] = Field(default=None)
    edit_lora_scale: float = Field(default=1.0, ge=0.0, le=4.0)
    export_vizj_path: Optional[str] = Field(default=None)
    export_vizj_tier: int = Field(default=2, ge=0, le=2)
    num_images_per_prompt: int = Field(default=1, ge=1, le=8)
    seed: Optional[int] = Field(default=42)
    output_path: str = Field(default="output_moss.png")
    device: str = Field(default="cuda")
    cpu_offload: Optional[bool] = Field(default=None)
    prepass_warmup: Optional[bool] = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            out = dict(data)
            if "layout_boxes" in out and "layout_items" not in out:
                out["layout_items"] = out.pop("layout_boxes")
            return out
        return data

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
        if self.num_inference_steps is None:
            self.num_inference_steps = int(active["num_inference_steps"])
        if self.shift is None:
            self.shift = float(active["shift"])
        if self.use_v2_grid is None:
            self.use_v2_grid = bool(active["use_v2_grid"])
        if self.early_solver is None:
            self.early_solver = str(active["early_solver"])
        if self.late_solver is None:
            self.late_solver = str(active["late_solver"])
        if self.handoff_threshold is None:
            self.handoff_threshold = float(active["handoff_threshold"])
        if self.guidance_scale is None:
            self.guidance_scale = float(active["guidance_scale"])
        if self.early_cfg is None:
            self.early_cfg = float(self.guidance_scale)
        if self.late_cfg is None:
            self.late_cfg = float(self.guidance_scale)
        if self.uncond_skip_layers is None:
            self.uncond_skip_layers = list(active["uncond_skip_layers"])
        if self.slg_layers is None:
            self.slg_layers = list(active["slg_layers"])
        if self.slg_scale is None:
            self.slg_scale = float(active["slg_scale"])
        if self.slg_start is None:
            self.slg_start = float(active["slg_start"])
        if self.slg_stop is None:
            self.slg_stop = float(active["slg_stop"])
        if self.eta is None:
            self.eta = float(active["eta"])
        if self.s_noise is None:
            self.s_noise = float(active["s_noise"])
        if self.max_sequence_length is None:
            self.max_sequence_length = int(active["max_sequence_length"])
        if self.enable_thinking is None:
            self.enable_thinking = bool(active["enable_thinking"])
        if self.framing_mode is None:
            self.framing_mode = str(active["framing_mode"])
        if self.input_role is None:
            self.input_role = str(active["input_role"])
        if self.edit_format is None:
            self.edit_format = str(active["edit_format"])
        if self.max_vit_patches is None:
            self.max_vit_patches = int(active["max_vit_patches"])
        if self.deband_8bit is None:
            self.deband_8bit = bool(active["deband_8bit"])
        if self.prefer_vulkan is None:
            self.prefer_vulkan = bool(active["prefer_vulkan"])
        if self.mtf_boost is None:
            self.mtf_boost = float(active["mtf_boost"])
        if self.soft_knee_tau is None:
            self.soft_knee_tau = float(active["soft_knee_tau"])
        if self.mask_feather_radius is None:
            self.mask_feather_radius = float(active["mask_feather_radius"])
        if self.latent_clamp_sigma is None:
            self.latent_clamp_sigma = float(active["latent_clamp_sigma"])
        if self.apply_bayer_dither is None:
            self.apply_bayer_dither = bool(active["apply_bayer_dither"])
        if self.use_reference_cache is None:
            self.use_reference_cache = bool(active["use_reference_cache"])
        if self.cpu_offload is None:
            self.cpu_offload = bool(active["cpu_offload"])
        if self.prepass_warmup is None:
            self.prepass_warmup = bool(active["prepass_warmup"])
        return self

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GenerationRequest:
        return cls.model_validate(data)

    @property
    def effective_reference_zero_timestep(self) -> bool:
        if self.reference_zero_timestep is not None:
            return bool(self.reference_zero_timestep)
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
        return self.edit_format in (
            "moss_image_edit",
            "ai_toolkit_isolated",
            "control_color_depth",
        )

    def resolve_repo_id(self) -> str:
        if self.repo_id and self.repo_id.strip():
            return self.repo_id.strip()
        variant = (self.model_variant or "sft").strip().lower()
        return SUPPORTED_MODELS.get(variant, SUPPORTED_MODELS["sft"])

    def compile_effective_prompt(self) -> str:
        mode = (self.prompt_mode or "natural").strip().lower()
        if mode in ("layout", "layout_json") and self.layout_items:
            return compile_layout_prompt(
                caption=self.prompt,
                items=self.layout_items,
                height=self.height,
                width=self.width,
                coords_are_pixels=self.layout_coords_are_pixels,
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
        if not (0.0 <= self.slg_start <= self.slg_stop <= 1.0):
            raise ValueError(
                "SLG window must satisfy 0.0 <= slg_start <= slg_stop <= 1.0."
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
    tiktalkin_active: bool = Field(default=True)
    vizinjector_active: bool = Field(default=False)
    vizinjector_backend: str = Field(default="PyTorch Fallback")
    prompt_cache_hit: bool
    reference_cache_hit: bool = Field(default=False)
    common_prefix_tokens: int
    edit_format_used: str = Field(default="standard")
    framing_mode_used: str = Field(default="inscribed")
    input_role_used: str = Field(default="photo_ref")
    reference_zero_timestep_used: bool = Field(default=False)
    reference_isolated_kv_used: bool = Field(default=False)
    edit_lora_active: bool = Field(default=False)
    effective_prompt: str
    mean_luminance: float
    rms_contrast: float
    shannon_entropy_bits: float
    laplacian_sharpness: float