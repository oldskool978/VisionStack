# VizInjector/python/schema.py
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, model_validator

FramingModeLiteral = Literal[
    "inscribed", "cover_crop", "contain", "stretch", "never_crop"
]
InputRoleLiteral = Literal["photo_ref", "color_map", "depth_map", "inpaint_mask"]
EditFormatLiteral = Literal[
    "standard",
    "moss_image_edit",
    "ai_toolkit_isolated",
    "ai_toolkit_full",
    "control_color_depth",
]

FRAMING_MODE_TO_INT = {
    "inscribed": 0,
    "cover_crop": 0,
    "contain": 1,
    "stretch": 2,
    "never_crop": 3,
}

INPUT_ROLE_TO_INT = {
    "photo_ref": 0,
    "color_map": 1,
    "depth_map": 2,
    "inpaint_mask": 3,
}


class VizInjectorTelemetry(BaseModel):
    mean_luminance: float = Field(default=0.0)
    rms_contrast: float = Field(default=0.0)
    shannon_entropy_bits: float = Field(default=0.0)
    laplacian_sharpness: float = Field(default=0.0)


class VizInjectorRequest(BaseModel):
    source: Optional[Union[str, Path, bytes]] = Field(default=None, exclude=True)
    target_width: int = Field(default=1024, ge=0, le=4096)
    target_height: int = Field(default=1024, ge=0, le=4096)
    max_vit_patches: int = Field(default=2304, ge=64, le=9216)
    framing_mode: FramingModeLiteral = Field(default="inscribed")
    input_role: InputRoleLiteral = Field(default="photo_ref")
    edit_format: EditFormatLiteral = Field(default="standard")
    edit_mode_384_vit: Optional[bool] = Field(default=None)
    emit_vae_branch: Optional[bool] = Field(default=None)
    emit_vit_branch: Optional[bool] = Field(default=None)
    emit_mask_branch: Optional[bool] = Field(default=None)
    emit_wire_vizj: bool = Field(default=False)
    deband_8bit: bool = Field(default=True)
    prefer_vulkan: bool = Field(default=True)
    seed: int = Field(default=42, ge=0)
    mtf_boost: float = Field(default=0.08, ge=0.0, le=0.5)
    soft_knee_tau: float = Field(default=0.015, ge=0.001, le=0.25)
    mask_feather_radius: float = Field(default=1.25, ge=0.0, le=6.0)
    latent_clamp_sigma: float = Field(default=4.5, ge=2.0, le=12.0)
    webp_quality: float = Field(default=98.0, ge=1.0, le=100.0)
    webp_lossless: bool = Field(default=True)

    @model_validator(mode="after")
    def _configure_regime_routing(self) -> VizInjectorRequest:
        if self.edit_format == "moss_image_edit":
            if self.framing_mode in ("inscribed", "cover_crop"):
                self.framing_mode = "never_crop"
            if self.emit_vit_branch is None:
                self.emit_vit_branch = False
            if self.edit_mode_384_vit is None:
                self.edit_mode_384_vit = False
        elif self.edit_format in ("ai_toolkit_isolated", "ai_toolkit_full"):
            if self.edit_mode_384_vit is None:
                self.edit_mode_384_vit = True
            if self.emit_vit_branch is None:
                self.emit_vit_branch = True
        else:
            if self.edit_mode_384_vit is None:
                self.edit_mode_384_vit = False

        if self.input_role in ("color_map", "depth_map"):
            self.emit_vae_branch = True
            self.emit_vit_branch = False
            self.emit_mask_branch = False
            self.mtf_boost = 0.0
        elif self.input_role == "inpaint_mask":
            self.emit_vae_branch = False
            self.emit_vit_branch = False
            self.emit_mask_branch = True
            self.mtf_boost = 0.0
        else:
            if self.emit_vae_branch is None:
                self.emit_vae_branch = True
            if self.emit_vit_branch is None:
                self.emit_vit_branch = True
            if self.emit_mask_branch is None:
                self.emit_mask_branch = True
        return self

    @property
    def framing_mode_id(self) -> int:
        return FRAMING_MODE_TO_INT[self.framing_mode]

    @property
    def input_role_id(self) -> int:
        return INPUT_ROLE_TO_INT[self.input_role]

    @property
    def reference_uses_zero_timestep(self) -> bool:
        return self.edit_format in (
            "moss_image_edit",
            "ai_toolkit_isolated",
            "ai_toolkit_full",
            "control_color_depth",
        )

    @property
    def reference_zero_timestep(self) -> bool:
        return self.reference_uses_zero_timestep

    @property
    def reference_uses_isolated_kv_cache(self) -> bool:
        return self.edit_format in (
            "moss_image_edit",
            "ai_toolkit_isolated",
            "control_color_depth",
        )

    @property
    def reference_isolated_kv(self) -> bool:
        return self.reference_uses_isolated_kv_cache


class VizInjectorResponseMeta(BaseModel):
    orig_size: Tuple[int, int]
    vae_size: Tuple[int, int]
    vit_size: Tuple[int, int]
    grid_thw: Tuple[int, int, int]
    llm_tokens: int
    backend_used: str
    edit_format: EditFormatLiteral
    input_role: InputRoleLiteral
    reference_zero_timestep: bool
    reference_isolated_kv: bool
    telemetry: VizInjectorTelemetry