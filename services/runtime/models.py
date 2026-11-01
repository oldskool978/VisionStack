# services/runtime/models.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from .env import parse_layer_list


class PowSubmission(BaseModel):
    challenge: str
    signature: str
    solution_nonce: str


class AuthPayload(BaseModel):
    username: str


class ReferenceUploadPayload(BaseModel):
    filename: str = Field(default="reference.png", max_length=160)
    role: str = Field(default="photo_ref", max_length=40)
    data_url: str = Field(..., min_length=16)


class PromoteImagePayload(BaseModel):
    role: str = Field(default="photo_ref", max_length=40)
    variant_index: int = Field(default=1, ge=1, le=8)


class ProjectSavePayload(BaseModel):
    project_id: Optional[str] = Field(default=None, max_length=80)
    title: str = Field(default="Untitled Studio Project", max_length=120)
    working_draft: Dict[str, Any] = Field(default_factory=dict)
    telemetry: Optional[Dict[str, Any]] = Field(default=None)
    active_image_id: Optional[str] = Field(default=None, max_length=80)
    inpaint_canvas_data_url: Optional[str] = Field(default=None)
    include_all_vault_refs: bool = Field(default=False)


class ProjectImportPayload(BaseModel):
    filename: str = Field(default="project.vsproj", max_length=160)
    data_url: str = Field(..., min_length=16)


class PromptPreviewPayload(BaseModel):
    prompt: str = Field(default="", max_length=8000)
    prompt_mode: str = Field(default="natural")
    layout_summary: Optional[str] = Field(default=None, max_length=4000)
    layout_items: List[Dict[str, Any]] = Field(default_factory=list)
    layout_coords_are_pixels: bool = Field(default=True)
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    system_prompt: Optional[str] = Field(default=None, max_length=4000)
    thinking_trace: Optional[str] = Field(default=None, max_length=4000)
    enable_thinking: Optional[bool] = Field(default=True)
    add_vision_id: Optional[bool] = Field(default=True)
    max_sequence_length: Optional[int] = Field(default=16384, ge=64, le=32768)
    max_vit_patches: Optional[int] = Field(default=2304, ge=64, le=9216)
    edit_format: Optional[str] = Field(default="standard")
    framing_mode: Optional[str] = Field(default="inscribed")
    input_role: Optional[str] = Field(default="photo_ref")
    box_attn_isolation: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    composite_box_images_to_init: Optional[bool] = Field(default=False)
    uncond_uses_references: Optional[bool] = Field(default=None)
    reference_zero_timestep: Optional[bool] = Field(default=None)
    reference_isolated_kv: Optional[bool] = Field(default=None)
    reference_image_paths: List[str] = Field(default_factory=list)
    control_image_path: Optional[str] = Field(default=None)


class SynthesisPayload(BaseModel):
    title: str = Field(default="Untitled Vision Master", max_length=120)
    model_variant: Optional[str] = Field(default="sft")
    prompt: str = Field(default="", max_length=8000)
    negative_prompt: str = Field(default="", max_length=4000)
    system_prompt: Optional[str] = Field(default=None, max_length=4000)
    thinking_trace: Optional[str] = Field(default=None, max_length=4000)
    add_vision_id: Optional[bool] = Field(default=None)
    prompt_mode: str = Field(default="natural")
    layout_summary: Optional[str] = Field(default=None, max_length=4000)
    layout_items: List[Dict[str, Any]] = Field(default_factory=list)
    layout_coords_are_pixels: bool = Field(default=True)
    preserve_canvas_outside_boxes: Optional[bool] = Field(default=None)
    box_attn_isolation: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    uncond_uses_references: Optional[bool] = Field(default=None)
    composite_box_images_to_init: Optional[bool] = Field(default=None)

    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
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
    denoise_strength: Optional[float] = Field(default=None, ge=0.01, le=1.0)
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
    edit_lora_path: Optional[str] = Field(default=None)
    edit_lora_scale: Optional[float] = Field(default=None, ge=0.0, le=4.0)
    export_vizj: bool = Field(default=False)
    export_vizj_tier: int = Field(default=2, ge=0, le=2)

    num_images_per_prompt: int = Field(default=1, ge=1, le=8)
    seed: Optional[int] = Field(default=None, ge=0)
    cpu_offload: Optional[bool] = Field(default=None)
    prepass_warmup: Optional[bool] = Field(default=None)
    pow: PowSubmission

    @field_validator("seed", mode="before")
    @classmethod
    def _coerce_seed(cls, v: Any) -> Optional[int]:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        try:
            val = int(v)
            return val if val >= 0 else None
        except (ValueError, TypeError):
            return None

    @field_validator("width", "height", mode="before")
    @classmethod
    def _coerce_dims(cls, v: Any) -> int:
        try:
            val = max(256, min(2048, int(v)))
            return (val // 16) * 16
        except (ValueError, TypeError):
            return 1024

    @field_validator("uncond_skip_layers", "slg_layers", mode="before")
    @classmethod
    def _coerce_layer_lists(cls, v: Any) -> Optional[List[int]]:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return parse_layer_list(v)

    @field_validator("system_prompt", "thinking_trace", mode="before")
    @classmethod
    def _coerce_opt_strings(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    @field_validator(
        "shift",
        "handoff_threshold",
        "guidance_scale",
        "early_cfg",
        "late_cfg",
        "slg_scale",
        "slg_start",
        "slg_stop",
        "eta",
        "s_noise",
        "denoise_strength",
        "mtf_boost",
        "soft_knee_tau",
        "mask_feather_radius",
        "latent_clamp_sigma",
        "edit_lora_scale",
        "box_attn_isolation",
        mode="before",
    )
    @classmethod
    def _coerce_floats(cls, v: Any) -> Any:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return v


class PresetUpdatePayload(BaseModel):
    engine_defaults: Dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "PowSubmission",
    "AuthPayload",
    "ReferenceUploadPayload",
    "PromoteImagePayload",
    "ProjectSavePayload",
    "ProjectImportPayload",
    "PromptPreviewPayload",
    "SynthesisPayload",
    "PresetUpdatePayload",
]