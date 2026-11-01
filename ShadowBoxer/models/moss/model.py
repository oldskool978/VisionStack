# ShadowBoxer/models/moss/model.py
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from .config import (
    _FP8_DTYPES,
    IsoKVSpec,
    MOSSImageConfig,
    ModTuple6,
    PiecewiseModSpec,
    TransformerOutput,
    _RMSNorm,
    _linear,
    _maybe_pin_cpu_tensor,
    _resolve_compute_dtype,
    validate_skip_layers,
)
from .geometry import (
    _build_box_aperture_bias,
    _build_isolated_reference_mask,
    _compute_collaborating_ref_groups,
    _normalize_box_isolation,
    _normalize_foreground_carveouts,
    _normalize_ref_bboxes,
    _normalize_ref_roles,
    _normalize_ref_z_orders,
)
from .layers import _Block, _TextProjection
from .rope import _RoPE


class MOSSImageTransformer(nn.Module):
    _build_isolated_reference_mask = staticmethod(_build_isolated_reference_mask)
    _build_box_aperture_bias = staticmethod(_build_box_aperture_bias)

    def __init__(
        self,
        config: Optional[MOSSImageConfig] = None,
        dim: int = 3072,
        ffn_dim: int = 8192,
        num_heads: int = 24,
        num_layers: int = 42,
        text_dim: int = 4096,
        freq_dim: int = 256,
        latent_channels: int = 128,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        if config is not None:
            dim = config.dim
            ffn_dim = config.ffn_dim
            num_heads = config.num_heads
            num_layers = config.num_layers
            text_dim = config.text_dim
            freq_dim = config.freq_dim
            latent_channels = config.latent_channels
            rope_base = config.rope_base
        if dim % num_heads or (dim // num_heads) % 8 or freq_dim % 2:
            raise ValueError(
                "Require dim divisible by heads, head_dim divisible by 8, and even freq_dim."
            )

        self.config = MOSSImageConfig(
            dim=dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            text_dim=text_dim,
            freq_dim=freq_dim,
            latent_channels=latent_channels,
            rope_base=rope_base,
        )
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = _TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.rope = _RoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList(
            [_Block(dim, ffn_dim, num_heads) for _ in range(num_layers)]
        )
        self.final_norm_video = _RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

        self._lora_backups: Dict[str, torch.Tensor] = {}
        self._active_lora_tag: Optional[str] = None
        self._freq_cache: Dict[str, torch.Tensor] = {}
        self._isolated_mask_cache: OrderedDict[
            Tuple[Any, ...], torch.Tensor
        ] = OrderedDict()
        self._box_aperture_bias_cache: OrderedDict[
            Tuple[Any, ...], torch.Tensor
        ] = OrderedDict()
        self._box_aperture_mask_refs: OrderedDict[
            Tuple[Any, ...], Tuple[Optional[torch.Tensor], ...]
        ] = OrderedDict()
        self._mod_zero_cache: OrderedDict[
            Tuple[str, torch.dtype, Optional[str]], torch.Tensor
        ] = OrderedDict()
        self._time_mod_cache: OrderedDict[
            Tuple[Any, ...],
            Tuple[Any, torch.Tensor, torch.Tensor],
        ] = OrderedDict()
        self._text_proj_cache: OrderedDict[
            Tuple[Any, ...],
            Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor],
        ] = OrderedDict()
        self._ref_proj_cache: OrderedDict[
            Tuple[Any, ...],
            Tuple[torch.Tensor, torch.Tensor],
        ] = OrderedDict()
        self._mod_seq_buf: Optional[torch.Tensor] = None

    @classmethod
    def from_config_dir(cls, model_dir: Union[str, Path]) -> MOSSImageTransformer:
        cfg_file = Path(model_dir) / "config.json"
        if cfg_file.is_file():
            return cls(config=MOSSImageConfig.from_json_file(cfg_file))
        return cls()

    @staticmethod
    def format_lora_tag(lora_tag: str, scale: float) -> str:
        return f"{lora_tag}:{float(scale):.4f}"

    def is_lora_active(self, lora_tag: str, scale: float) -> bool:
        if abs(float(scale)) < 1e-7:
            return self._active_lora_tag is None
        return (
            self._active_lora_tag == self.format_lora_tag(lora_tag, scale)
            and bool(self._lora_backups)
        )

    def clear_transient_caches(self) -> None:
        self._time_mod_cache.clear()
        self._text_proj_cache.clear()
        self._ref_proj_cache.clear()
        self._isolated_mask_cache.clear()
        self._box_aperture_bias_cache.clear()
        self._box_aperture_mask_refs.clear()
        self._mod_seq_buf = None

    def clear_caches(self) -> None:
        self.rope.clear_cache()
        self._freq_cache.clear()
        self._isolated_mask_cache.clear()
        self._box_aperture_bias_cache.clear()
        self._box_aperture_mask_refs.clear()
        self._mod_zero_cache.clear()
        self.clear_transient_caches()

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.clear_caches()
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.clear_caches()

    @torch.no_grad()
    def unfuse_lora(self) -> None:
        if not self._lora_backups:
            self._active_lora_tag = None
            return
        param_dict = dict(self.named_parameters())
        touched_device: Optional[torch.device] = None
        for name, orig_cpu in self._lora_backups.items():
            if name in param_dict:
                p = param_dict[name]
                touched_device = p.device
                if p.dtype == orig_cpu.dtype:
                    p.data.copy_(orig_cpu, non_blocking=True)
                else:
                    p.data.copy_(
                        orig_cpu.to(device=p.device, dtype=p.dtype, non_blocking=True)
                    )
        if touched_device is not None and touched_device.type == "cuda":
            torch.cuda.synchronize(touched_device)
        self._lora_backups.clear()
        self._active_lora_tag = None
        self._mod_zero_cache.clear()
        self.clear_transient_caches()

    @torch.no_grad()
    def fuse_lora_state_dict(
        self,
        lora_state: Dict[str, torch.Tensor],
        scale: float = 1.0,
        lora_tag: str = "custom",
    ) -> int:
        target_tag = self.format_lora_tag(lora_tag, scale)
        if self._active_lora_tag == target_tag:
            return len(self._lora_backups)
        if self._lora_backups:
            self.unfuse_lora()
        if abs(scale) < 1e-7:
            return 0

        param_dict = dict(self.named_parameters())
        down_keys = [
            k
            for k in lora_state.keys()
            if k.endswith(".lora_A.weight") or k.endswith(".lora_down.weight")
        ]
        fused_count = 0
        for d_key in down_keys:
            if d_key.endswith(".lora_A.weight"):
                prefix = d_key[: -len(".lora_A.weight")]
                u_key = prefix + ".lora_B.weight"
            else:
                prefix = d_key[: -len(".lora_down.weight")]
                u_key = prefix + ".lora_up.weight"
            if u_key not in lora_state:
                continue

            clean_prefix = prefix
            for strip_tok in (
                "base_model.model.",
                "diffusion_model.",
                "transformer.",
                "model.",
            ):
                if clean_prefix.startswith(strip_tok):
                    clean_prefix = clean_prefix[len(strip_tok) :]

            target_param_name = f"{clean_prefix}.weight"
            if target_param_name not in param_dict:
                continue

            param = param_dict[target_param_name]
            if target_param_name not in self._lora_backups:
                self._lora_backups[target_param_name] = _maybe_pin_cpu_tensor(param.data)

            lora_a = lora_state[d_key].to(device=param.device, dtype=torch.float32)
            lora_b = lora_state[u_key].to(device=param.device, dtype=torch.float32)
            alpha_key = prefix + ".alpha"
            rank = float(lora_a.shape[0])
            alpha = (
                float(lora_state[alpha_key].item())
                if alpha_key in lora_state
                else rank
            )
            eff_scale = float(scale) * (alpha / max(rank, 1.0))

            delta = (lora_b @ lora_a) * eff_scale
            if delta.shape == param.shape:
                if param.dtype in _FP8_DTYPES:
                    finfo = torch.finfo(param.dtype)
                    updated = (param.data.float() + delta).clamp(
                        float(finfo.min), float(finfo.max)
                    )
                    param.data.copy_(updated.to(param.dtype))
                else:
                    param.data.copy_((param.data.float() + delta).to(param.dtype))
                fused_count += 1

        self._active_lora_tag = target_tag
        self._mod_zero_cache.clear()
        self.clear_transient_caches()
        return fused_count

    def _get_time_freq(self, device: torch.device) -> torch.Tensor:
        dev_key = str(device)
        freq = self._freq_cache.get(dev_key)
        if freq is None or freq.device != device:
            half = self.config.freq_dim // 2
            freq = 10000.0 ** (
                -torch.arange(half, device=device, dtype=torch.float64) / half
            )
            self._freq_cache[dev_key] = freq
        return freq

    def _compute_time_modulation(
        self,
        timestep: Union[float, int, torch.Tensor],
        batch: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        w_dtype = _resolve_compute_dtype(self.time_embedding[0].weight)
        if isinstance(timestep, (int, float)):
            cache_key: Tuple[Any, ...] = (
                "scalar",
                float(timestep),
                batch,
                str(device),
                w_dtype,
                self._active_lora_tag,
            )
        elif isinstance(timestep, torch.Tensor) and timestep.device.type == "cpu":
            t_vals = tuple(float(v) for v in timestep.detach().reshape(-1).tolist())
            cache_key = (
                "cpu_tensor",
                t_vals,
                batch,
                str(device),
                w_dtype,
                self._active_lora_tag,
            )
        elif isinstance(timestep, torch.Tensor):
            cache_key = (
                "dev_tensor",
                int(timestep.data_ptr()),
                int(timestep.storage_offset()),
                tuple(timestep.shape),
                timestep.dtype,
                batch,
                str(device),
                w_dtype,
                self._active_lora_tag,
            )
        else:
            cache_key = (
                "other",
                id(timestep),
                batch,
                str(device),
                w_dtype,
                self._active_lora_tag,
            )

        cached = self._time_mod_cache.get(cache_key)
        if cached is not None:
            self._time_mod_cache.move_to_end(cache_key)
            return cached[1], cached[2]

        times = torch.as_tensor(
            timestep, device=device, dtype=torch.float32
        ).reshape(-1)
        single_time = times.numel() == 1
        eff_batch = 1 if single_time else batch
        if not single_time and times.numel() != batch:
            times = times.expand(batch)

        freq = self._get_time_freq(device)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).to(w_dtype)

        h_time = F.silu(_linear(self.time_embedding[0], embedding))
        time_vec = _linear(self.time_embedding[2], h_time).float()

        mod_in = time_vec.to(
            _resolve_compute_dtype(self.shared_block_modulation.weight, w_dtype)
        )
        head_in = time_vec.to(
            _resolve_compute_dtype(self.video_head_modulation.weight, w_dtype)
        )

        mod = _linear(self.shared_block_modulation, mod_in).view(eff_batch, 6, -1)
        head = (
            _linear(self.video_head_modulation, head_in)
            .view(eff_batch, 2, -1)
            .float()
        )
        if single_time and batch > 1:
            mod = mod.expand(batch, -1, -1).contiguous()
            head = head.expand(batch, -1, -1).contiguous()

        self._time_mod_cache[cache_key] = (timestep, mod, head)
        while len(self._time_mod_cache) > 4:
            self._time_mod_cache.popitem(last=False)
        return mod, head

    def _get_cached_zero_modulation(self, device: torch.device) -> torch.Tensor:
        w_dtype = _resolve_compute_dtype(self.time_embedding[0].weight)
        key = (str(device), w_dtype, self._active_lora_tag)
        cached = self._mod_zero_cache.get(key)
        if cached is not None:
            self._mod_zero_cache.move_to_end(key)
            return cached
        mod_zero, _ = self._compute_time_modulation(0.0, 1, device)
        self._mod_zero_cache[key] = mod_zero
        while len(self._mod_zero_cache) > 4:
            self._mod_zero_cache.popitem(last=False)
        return mod_zero

    def _get_or_compute_text_projection(
        self,
        encoder_hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor],
        index: int,
        act_dtype: torch.dtype,
        height: int = 0,
        width: int = 0,
        ref_shapes: Optional[List[Tuple[int, int]]] = None,
        ref_bboxes: Optional[Sequence[Optional[Sequence[float]]]] = None,
        need_full_rope: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], int]:
        mask_id = id(mask) if mask is not None else 0
        mask_ptr = int(mask.data_ptr()) if mask is not None else 0
        mask_shape = tuple(mask.shape) if mask is not None else ()
        key = (
            id(encoder_hidden_states),
            int(encoder_hidden_states.data_ptr()),
            tuple(encoder_hidden_states.shape),
            index,
            mask_id,
            mask_ptr,
            mask_shape,
            act_dtype,
            str(encoder_hidden_states.device),
            self._active_lora_tag,
        )
        entry = self._text_proj_cache.get(key)
        if entry is not None:
            cached_ehs, cached_mask, cached_text = entry
            if cached_ehs is encoder_hidden_states and cached_mask is mask:
                self._text_proj_cache.move_to_end(key)
                text_len = int(cached_text.shape[1])
                full_rope_4d: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                if need_full_rope and height > 0 and width > 0:
                    full_rope_4d, _ = self.rope.get_expanded_rope(
                        height,
                        width,
                        text_len,
                        encoder_hidden_states.device,
                        dtype=act_dtype,
                        ref_shapes=ref_shapes,
                        ref_bboxes=ref_bboxes,
                    )
                return cached_text, full_rope_4d, text_len

        if mask is not None:
            context = encoder_hidden_states[index : index + 1, mask[index].bool()]
        else:
            context = encoder_hidden_states[index : index + 1]
        text_len = int(context.shape[1])
        if text_len == 0:
            raise ValueError("Each sample needs at least one valid text token.")

        text_rope_4d = self.rope.get_text_rope(
            text_len,
            encoder_hidden_states.device,
            dtype=act_dtype,
        )
        text = self.text_embedder(
            context if context.dtype == act_dtype else context.to(act_dtype),
            text_rope_4d,
        )
        self._text_proj_cache[key] = (encoder_hidden_states, mask, text)
        while len(self._text_proj_cache) > 8:
            self._text_proj_cache.popitem(last=False)

        full_rope_4d = None
        if need_full_rope and height > 0 and width > 0:
            full_rope_4d, _ = self.rope.get_expanded_rope(
                height,
                width,
                text_len,
                encoder_hidden_states.device,
                dtype=act_dtype,
                ref_shapes=ref_shapes,
                ref_bboxes=ref_bboxes,
            )
        return text, full_rope_4d, text_len

    def _get_or_compute_ref_projection(
        self,
        ref_lat: torch.Tensor,
        index: int,
        act_dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        eff_idx = index if ref_lat.shape[0] > index else 0
        key = (
            id(ref_lat),
            int(ref_lat.data_ptr()),
            tuple(ref_lat.shape),
            eff_idx,
            act_dtype,
            str(device),
            self._active_lora_tag,
        )
        entry = self._ref_proj_cache.get(key)
        if entry is not None:
            cached_ref, cached_emb = entry
            if cached_ref is ref_lat:
                self._ref_proj_cache.move_to_end(key)
                return cached_emb

        r_slice = ref_lat[eff_idx : eff_idx + 1]
        r_flat = (
            r_slice.flatten(2)
            .transpose(1, 2)
            .to(device=device, dtype=act_dtype)
            .contiguous()
        )
        r_emb = _linear(self.video_embedder, r_flat)
        self._ref_proj_cache[key] = (ref_lat, r_emb)
        while len(self._ref_proj_cache) > 8:
            self._ref_proj_cache.popitem(last=False)
        return r_emb

    def _acquire_mod_seq_buf(
        self,
        num_regimes: int,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int = 1,
    ) -> torch.Tensor:
        dim = self.config.dim
        alloc_regimes = max(2, int(num_regimes))
        alloc_batch = max(1, int(batch_size))
        if (
            self._mod_seq_buf is None
            or self._mod_seq_buf.device != device
            or self._mod_seq_buf.dtype != dtype
            or self._mod_seq_buf.shape[0] < alloc_batch
            or self._mod_seq_buf.shape[2] < alloc_regimes
            or self._mod_seq_buf.shape[3] != dim
        ):
            self._mod_seq_buf = torch.empty(
                (alloc_batch, 6, alloc_regimes, dim), device=device, dtype=dtype
            )
        return self._mod_seq_buf[:alloc_batch, :, :alloc_regimes, :]

    def _get_isolated_reference_mask(
        self,
        target_tokens: int,
        ref_token_lengths: List[int],
        text_tokens: int,
        device: torch.device,
        ref_bboxes: Optional[Sequence[Optional[Sequence[float]]]] = None,
        pad_y: float = 0.0,
        pad_x: float = 0.0,
    ) -> torch.Tensor:
        dummy_shapes = [(1, 1) for _ in ref_token_lengths]
        norm_boxes = _normalize_ref_bboxes(dummy_shapes, ref_bboxes)
        collab_groups = _compute_collaborating_ref_groups(
            len(ref_token_lengths), norm_boxes, pad_y=pad_y, pad_x=pad_x
        )
        key = (
            target_tokens,
            tuple(ref_token_lengths),
            text_tokens,
            collab_groups,
            str(device),
        )
        cached = self._isolated_mask_cache.get(key)
        if cached is not None:
            self._isolated_mask_cache.move_to_end(key)
            return cached
        mask = _build_isolated_reference_mask(
            target_tokens=target_tokens,
            ref_token_lengths=ref_token_lengths,
            text_tokens=text_tokens,
            device=device,
            collab_groups=collab_groups,
        )
        self._isolated_mask_cache[key] = mask
        while len(self._isolated_mask_cache) > 8:
            self._isolated_mask_cache.popitem(last=False)
        return mask

    def _get_box_aperture_bias(
        self,
        height: int,
        width: int,
        ref_token_lengths: Sequence[int],
        text_tokens: int,
        ref_shapes: Optional[List[Tuple[int, int]]],
        ref_bboxes: Optional[Sequence[Optional[Sequence[float]]]],
        reference_box_isolation: Optional[Union[float, Sequence[Optional[float]]]],
        feather_radius_lat: float,
        device: torch.device,
        dtype: torch.dtype,
        reference_masks: Optional[Sequence[Optional[torch.Tensor]]] = None,
        reference_roles: Optional[Sequence[Optional[str]]] = None,
        reference_z_orders: Optional[Sequence[Optional[float]]] = None,
        foreground_carveouts: Optional[Sequence[Any]] = None,
    ) -> Optional[torch.Tensor]:
        norm_boxes = _normalize_ref_bboxes(
            ref_shapes, ref_bboxes, height=height, width=width
        )
        iso_tuple = _normalize_box_isolation(norm_boxes, reference_box_isolation)
        carveout_tuple = _normalize_foreground_carveouts(
            foreground_carveouts, height=height, width=width
        )
        if not any(v > 1e-6 for v in iso_tuple) and not carveout_tuple:
            return None

        num_refs = len(ref_token_lengths)
        role_tuple = _normalize_ref_roles(num_refs, reference_roles)
        z_order_tuple = _normalize_ref_z_orders(num_refs, reference_z_orders)

        mask_key: Tuple[Any, ...] = ()
        mask_refs: Tuple[Optional[torch.Tensor], ...] = ()
        if reference_masks is not None and any(m is not None for m in reference_masks):
            mask_refs = tuple(reference_masks[:num_refs])
            mask_key = tuple(
                None
                if m is None
                else (
                    id(m),
                    int(m.data_ptr()),
                    int(m.storage_offset()),
                    tuple(m.shape),
                    0 if m.is_inference() else int(m._version),
                )
                for m in mask_refs
            )

        key = (
            int(height),
            int(width),
            tuple(int(x) for x in ref_token_lengths),
            int(text_tokens),
            norm_boxes,
            iso_tuple,
            role_tuple,
            z_order_tuple,
            carveout_tuple,
            round(float(feather_radius_lat), 4),
            mask_key,
            str(device),
            dtype,
        )
        cached = self._box_aperture_bias_cache.get(key)
        if cached is not None:
            cached_refs = self._box_aperture_mask_refs.get(key, ())
            if len(cached_refs) == len(mask_refs) and all(
                a is b for a, b in zip(cached_refs, mask_refs)
            ):
                self._box_aperture_bias_cache.move_to_end(key)
                if key in self._box_aperture_mask_refs:
                    self._box_aperture_mask_refs.move_to_end(key)
                return cached

        bias = _build_box_aperture_bias(
            height=height,
            width=width,
            ref_token_lengths=ref_token_lengths,
            text_tokens=text_tokens,
            norm_boxes=norm_boxes,
            iso_tuple=iso_tuple,
            feather_radius_lat=feather_radius_lat,
            device=device,
            dtype=dtype,
            ref_shapes=ref_shapes,
            reference_masks=reference_masks,
            role_tuple=role_tuple,
            z_order_tuple=z_order_tuple,
            carveout_tuple=carveout_tuple,
        )
        self._box_aperture_bias_cache[key] = bias
        self._box_aperture_mask_refs[key] = mask_refs
        while len(self._box_aperture_bias_cache) > 8:
            evicted_key, _ = self._box_aperture_bias_cache.popitem(last=False)
            self._box_aperture_mask_refs.pop(evicted_key, None)
        return bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Union[float, int, torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        skip_layers: Optional[Iterable[int]] = None,
        reference_latents: Optional[List[torch.Tensor]] = None,
        reference_zero_timestep: bool = False,
        reference_isolated_kv: bool = False,
        reference_bboxes: Optional[Sequence[Optional[Sequence[float]]]] = None,
        reference_box_isolation: Optional[Union[float, Sequence[Optional[float]]]] = None,
        box_feather_radius: float = 1.25,
        reference_masks: Optional[Sequence[Optional[torch.Tensor]]] = None,
        reference_roles: Optional[Sequence[Optional[str]]] = None,
        reference_z_orders: Optional[Sequence[Optional[float]]] = None,
        foreground_carveouts: Optional[Sequence[Any]] = None,
    ) -> Union[TransformerOutput, Tuple[torch.Tensor, ...]]:
        skip_set = validate_skip_layers(skip_layers, len(self.layers))
        if (
            hidden_states.ndim != 4
            or hidden_states.shape[1] != self.config.latent_channels
        ):
            raise ValueError("hidden_states must be [B, latent_channels, H, W].")
        batch, _, height, width = hidden_states.shape
        if (
            encoder_hidden_states.ndim != 3
            or encoder_hidden_states.shape[0] != batch
        ):
            raise ValueError("encoder_hidden_states must be [B, S, text_dim].")
        mask = encoder_attention_mask

        mod, head = self._compute_time_modulation(
            timestep, batch, hidden_states.device
        )
        mod_zero: Optional[torch.Tensor] = None
        if reference_latents and reference_zero_timestep:
            mod_zero = self._get_cached_zero_modulation(hidden_states.device)

        ref_shapes: Optional[List[Tuple[int, int]]] = None
        if reference_latents:
            ref_shapes = [
                (int(r.shape[-2]), int(r.shape[-1])) for r in reference_latents
            ]

        norm_boxes = _normalize_ref_bboxes(
            ref_shapes, reference_bboxes, height=height, width=width
        )
        pad_y = max(0.0, float(box_feather_radius)) / float(max(1, height))
        pad_x = max(0.0, float(box_feather_radius)) / float(max(1, width))
        collab_groups = _compute_collaborating_ref_groups(
            len(ref_shapes) if ref_shapes else 0,
            norm_boxes,
            pad_y=pad_y,
            pad_x=pad_x,
        )

        target_tokens = height * width
        act_dtype = _resolve_compute_dtype(self.video_embedder.weight)

        is_homogeneous = True
        if batch > 1 and mask is not None:
            valid_counts = mask.bool().sum(dim=-1)
            if not bool((valid_counts == valid_counts[0]).all().item()):
                is_homogeneous = False

        if is_homogeneous:
            text_0, full_rope_4d, text_len = self._get_or_compute_text_projection(
                encoder_hidden_states=encoder_hidden_states,
                mask=mask,
                index=0,
                act_dtype=act_dtype,
                height=height,
                width=width,
                ref_shapes=ref_shapes,
                ref_bboxes=reference_bboxes,
                need_full_rope=True,
            )
            assert full_rope_4d is not None

            if batch == 1:
                text = text_0
            else:
                same_ctx = bool(
                    (encoder_hidden_states[1:] == encoder_hidden_states[:1])
                    .all()
                    .item()
                ) and (
                    mask is None or bool((mask[1:] == mask[:1]).all().item())
                )
                if same_ctx:
                    text = text_0.expand(batch, -1, -1)
                else:
                    text_list = [text_0]
                    for idx in range(1, batch):
                        t_i, _, _ = self._get_or_compute_text_projection(
                            encoder_hidden_states=encoder_hidden_states,
                            mask=mask,
                            index=idx,
                            act_dtype=act_dtype,
                            need_full_rope=False,
                        )
                        text_list.append(t_i)
                    text = torch.cat(text_list, dim=0)

            ref_token_lengths: List[int] = [
                int(rh * rw) for rh, rw in (ref_shapes or [])
            ]
            total_img_tokens = target_tokens + sum(ref_token_lengths)
            total_seq_len = total_img_tokens + text_len

            x = torch.empty(
                (batch, total_seq_len, self.config.dim),
                device=hidden_states.device,
                dtype=act_dtype,
            )
            video = hidden_states.flatten(2).transpose(1, 2)
            video = _linear(self.video_embedder, video.to(act_dtype).contiguous())
            x[:, :target_tokens, :].copy_(video)
            del video

            if reference_latents:
                ref_cursor = target_tokens
                for r_idx, ref_lat in enumerate(reference_latents):
                    r_len = ref_token_lengths[r_idx]
                    if batch == 1 or ref_lat.shape[0] == 1:
                        r_emb_0 = self._get_or_compute_ref_projection(
                            ref_lat, 0, act_dtype, hidden_states.device
                        )
                        r_emb = (
                            r_emb_0.expand(batch, -1, -1) if batch > 1 else r_emb_0
                        )
                    else:
                        r_embs = [
                            self._get_or_compute_ref_projection(
                                ref_lat, idx, act_dtype, hidden_states.device
                            )
                            for idx in range(batch)
                        ]
                        r_emb = torch.cat(r_embs, dim=0)
                    x[:, ref_cursor : ref_cursor + r_len, :].copy_(r_emb)
                    ref_cursor += r_len

            x[:, total_img_tokens:total_seq_len, :].copy_(text)
            del text

            if mod_zero is not None and total_img_tokens > target_tokens:
                mod_pair = self._acquire_mod_seq_buf(
                    2, mod.device, mod.dtype, batch_size=batch
                )
                mod_pair[:, :, 0, :].copy_(mod)
                mod_pair[:, 1, 0, :].add_(1.0)
                mod_pair[:, 4, 0, :].add_(1.0)
                mod_pair[:, :, 1, :].copy_(mod_zero.expand(batch, -1, -1))
                mod_pair[:, 1, 1, :].add_(1.0)
                mod_pair[:, 4, 1, :].add_(1.0)
                mod_t_tuple: ModTuple6 = tuple(
                    mod_pair[:, :, 0:1, :].unbind(1)
                )  # type: ignore[assignment]
                mod_z_tuple: ModTuple6 = tuple(
                    mod_pair[:, :, 1:2, :].unbind(1)
                )  # type: ignore[assignment]
                active_mod: Union[ModTuple6, PiecewiseModSpec] = (
                    mod_t_tuple,
                    mod_z_tuple,
                    target_tokens,
                    total_img_tokens,
                )
            else:
                shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (
                    mod[:, :, None, :].unbind(1)
                )
                active_mod = (
                    shift_a,
                    1 + scale_a,
                    gate_a,
                    shift_f,
                    1 + scale_f,
                    gate_f,
                )

            iso_kv_spec: Optional[IsoKVSpec] = None
            if reference_latents and reference_isolated_kv and ref_token_lengths:
                iso_kv_spec = (
                    target_tokens,
                    tuple(ref_token_lengths),
                    text_len,
                    collab_groups,
                )

            target_attn_bias: Optional[torch.Tensor] = None
            if (
                reference_latents
                and ref_token_lengths
                and (
                    (
                        reference_bboxes is not None
                        and reference_box_isolation is not None
                    )
                    or foreground_carveouts is not None
                )
            ):
                target_attn_bias = self._get_box_aperture_bias(
                    height=height,
                    width=width,
                    ref_token_lengths=ref_token_lengths,
                    text_tokens=text_len,
                    ref_shapes=ref_shapes,
                    ref_bboxes=reference_bboxes,
                    reference_box_isolation=reference_box_isolation,
                    feather_radius_lat=box_feather_radius,
                    device=hidden_states.device,
                    dtype=x.dtype,
                    reference_masks=reference_masks,
                    reference_roles=reference_roles,
                    reference_z_orders=reference_z_orders,
                    foreground_carveouts=foreground_carveouts,
                )

            for layer_index, layer in enumerate(self.layers):
                if layer_index not in skip_set:
                    x = layer(
                        x,
                        full_rope_4d,
                        active_mod,
                        attn_mask=None,
                        iso_kv_spec=iso_kv_spec,
                        target_attn_bias=target_attn_bias,
                    )

            x_target = x[:, :target_tokens]
            if batch > 1:
                x_target = x_target.contiguous()
            x = self.final_norm_video(x_target.float())
            x.mul_(1.0 + head[:, 1, None]).add_(head[:, 0, None])
            x = _linear(self.final_linear_video, x.to(act_dtype))
            sample = x.transpose(1, 2).reshape(batch, -1, height, width)
            return TransformerOutput(sample=sample) if return_dict else (sample,)

        outputs: List[torch.Tensor] = []
        for index in range(batch):
            text, full_rope_4d, text_len = self._get_or_compute_text_projection(
                encoder_hidden_states=encoder_hidden_states,
                mask=mask,
                index=index,
                act_dtype=act_dtype,
                height=height,
                width=width,
                ref_shapes=ref_shapes,
                ref_bboxes=reference_bboxes,
                need_full_rope=True,
            )
            assert full_rope_4d is not None

            ref_token_lengths = [
                int(rh * rw) for rh, rw in (ref_shapes or [])
            ]
            total_img_tokens = target_tokens + sum(ref_token_lengths)
            total_seq_len = total_img_tokens + text_len

            x = torch.empty(
                (1, total_seq_len, self.config.dim),
                device=hidden_states.device,
                dtype=act_dtype,
            )
            video = hidden_states[index : index + 1].flatten(2).transpose(1, 2)
            video = _linear(self.video_embedder, video.to(act_dtype).contiguous())
            x[:, :target_tokens, :].copy_(video)
            del video

            if reference_latents:
                ref_cursor = target_tokens
                for r_idx, ref_lat in enumerate(reference_latents):
                    r_len = ref_token_lengths[r_idx]
                    r_emb = self._get_or_compute_ref_projection(
                        ref_lat, index, act_dtype, hidden_states.device
                    )
                    x[:, ref_cursor : ref_cursor + r_len, :].copy_(r_emb)
                    ref_cursor += r_len

            x[:, total_img_tokens:total_seq_len, :].copy_(text)
            del text

            mod_i = mod[index : index + 1]
            if mod_zero is not None and total_img_tokens > target_tokens:
                mod_pair = self._acquire_mod_seq_buf(
                    2, mod_i.device, mod_i.dtype, batch_size=1
                )
                mod_pair[:, :, 0, :].copy_(mod_i)
                mod_pair[:, 1, 0, :].add_(1.0)
                mod_pair[:, 4, 0, :].add_(1.0)
                mod_pair[:, :, 1, :].copy_(mod_zero)
                mod_pair[:, 1, 1, :].add_(1.0)
                mod_pair[:, 4, 1, :].add_(1.0)
                mod_t_tuple = tuple(
                    mod_pair[:, :, 0:1, :].unbind(1)
                )  # type: ignore[assignment]
                mod_z_tuple = tuple(
                    mod_pair[:, :, 1:2, :].unbind(1)
                )  # type: ignore[assignment]
                active_mod = (
                    mod_t_tuple,
                    mod_z_tuple,
                    target_tokens,
                    total_img_tokens,
                )
            else:
                shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (
                    mod_i[:, :, None, :].unbind(1)
                )
                active_mod = (
                    shift_a,
                    1 + scale_a,
                    gate_a,
                    shift_f,
                    1 + scale_f,
                    gate_f,
                )

            iso_kv_spec = None
            if reference_latents and reference_isolated_kv and ref_token_lengths:
                iso_kv_spec = (
                    target_tokens,
                    tuple(ref_token_lengths),
                    text_len,
                    collab_groups,
                )

            target_attn_bias = None
            if (
                reference_latents
                and ref_token_lengths
                and (
                    (
                        reference_bboxes is not None
                        and reference_box_isolation is not None
                    )
                    or foreground_carveouts is not None
                )
            ):
                target_attn_bias = self._get_box_aperture_bias(
                    height=height,
                    width=width,
                    ref_token_lengths=ref_token_lengths,
                    text_tokens=text_len,
                    ref_shapes=ref_shapes,
                    ref_bboxes=reference_bboxes,
                    reference_box_isolation=reference_box_isolation,
                    feather_radius_lat=box_feather_radius,
                    device=hidden_states.device,
                    dtype=x.dtype,
                    reference_masks=reference_masks,
                    reference_roles=reference_roles,
                    reference_z_orders=reference_z_orders,
                    foreground_carveouts=foreground_carveouts,
                )

            for layer_index, layer in enumerate(self.layers):
                if layer_index not in skip_set:
                    x = layer(
                        x,
                        full_rope_4d,
                        active_mod,
                        attn_mask=None,
                        iso_kv_spec=iso_kv_spec,
                        target_attn_bias=target_attn_bias,
                    )

            x = self.final_norm_video(x[:, :target_tokens].float())
            x.mul_(1.0 + head[index : index + 1, 1, None]).add_(
                head[index : index + 1, 0, None]
            )
            x = _linear(self.final_linear_video, x.to(act_dtype))
            outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))

        sample = torch.cat(outputs, dim=0)
        return TransformerOutput(sample=sample) if return_dict else (sample,)


__all__ = ["MOSSImageTransformer"]