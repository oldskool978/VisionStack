# ShadowBoxer/models/qwen35/model.py
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from .cache import Qwen3_5HybridCache
from .config import (
    Qwen3_5Config,
    Qwen3_5ModelOutput,
    _EYE_MATRIX_CACHE,
    _FP8_DTYPES,
    _STRICTLY_UPPER_MASK_CACHE,
    _linear,
    _resolve_compute_dtype,
)
from .speculative import Qwen3_5SpeculativeMixin
from .text import Qwen3_5MTPHead, Qwen3_5TextModel
from .vision import Qwen3_5VisionModel


class Qwen3_5Model(Qwen3_5SpeculativeMixin, nn.Module):
    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.config = config
        self.visual = Qwen3_5VisionModel(config.vision_config)
        self.language_model = Qwen3_5TextModel(config.text_config)
        self.lm_head: Optional[nn.Linear] = (
            nn.Linear(
                config.text_config.hidden_size,
                config.text_config.vocab_size,
                bias=False,
            )
            if config.include_lm_head
            else None
        )
        self.mtp: Optional[Qwen3_5MTPHead] = (
            Qwen3_5MTPHead(config.text_config) if config.include_mtp else None
        )
        self.rope_deltas: Optional[torch.Tensor] = None
        self.last_reanchor_telemetry: Dict[str, Any] = {}

    @classmethod
    def from_config_dir(cls, model_dir: Union[str, Path]) -> Qwen3_5Model:
        cfg_file = Path(model_dir) / "config.json"
        cfg = Qwen3_5Config.from_json_file(cfg_file)
        return cls(cfg)

    def enable_lm_head(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> nn.Linear:
        if self.lm_head is None:
            ref_w = self.language_model.embed_tokens.weight
            target_dev = device if device is not None else ref_w.device
            target_dtype = (
                dtype
                if dtype is not None
                else _resolve_compute_dtype(ref_w, torch.bfloat16)
            )
            self.lm_head = nn.Linear(
                self.config.text_config.hidden_size,
                self.config.text_config.vocab_size,
                bias=False,
            ).to(device=target_dev, dtype=target_dtype)
        return self.lm_head

    def enable_mtp(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Qwen3_5MTPHead:
        if self.mtp is None:
            ref_w = self.language_model.embed_tokens.weight
            target_dev = device if device is not None else ref_w.device
            target_dtype = (
                dtype
                if dtype is not None
                else _resolve_compute_dtype(ref_w, torch.bfloat16)
            )
            self.mtp = Qwen3_5MTPHead(self.config.text_config).to(
                device=target_dev, dtype=target_dtype
            )
        return self.mtp

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Any,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        if self.lm_head is None and f"{prefix}lm_head.weight" in state_dict:
            self.enable_lm_head()
        if self.mtp is None:
            mtp_prefix = f"{prefix}mtp."
            if any(k.startswith(mtp_prefix) for k in state_dict.keys()):
                self.enable_mtp()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def clear_caches(self) -> None:
        self.visual.clear_caches()
        self.rope_deltas = None
        _STRICTLY_UPPER_MASK_CACHE.clear()
        _EYE_MATRIX_CACHE.clear()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        return (
            self.lm_head
            if self.lm_head is not None
            else self.language_model.embed_tokens
        )

    def project_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.lm_head is not None:
            return _linear(self.lm_head, hidden_states)
        w = self.language_model.embed_tokens.weight
        if w.dtype in _FP8_DTYPES:
            compute_dtype = (
                hidden_states.dtype
                if hidden_states.dtype in (torch.bfloat16, torch.float16, torch.float32)
                else torch.bfloat16
            )
            x_in = (
                hidden_states
                if hidden_states.dtype == compute_dtype
                else hidden_states.to(compute_dtype)
            )
            return F.linear(x_in, w.to(compute_dtype))
        x_in = (
            hidden_states
            if hidden_states.dtype == w.dtype
            else hidden_states.to(w.dtype)
        )
        return F.linear(x_in, w)

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: torch.Tensor,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 2,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        llm_grid_t = int(grid_thw[0].item()) // temp_merge_size
        llm_grid_h = int(grid_thw[1].item()) // spatial_merge_size
        llm_grid_w = int(grid_thw[2].item()) // spatial_merge_size

        pos_t = torch.arange(llm_grid_t, device=device, dtype=torch.long) + start_position
        pos_h = torch.arange(llm_grid_h, device=device, dtype=torch.long) + start_position
        pos_w = torch.arange(llm_grid_w, device=device, dtype=torch.long) + start_position
        t_grid, h_grid, w_grid = torch.meshgrid(pos_t, pos_h, pos_w, indexing="ij")
        return torch.stack([t_grid, h_grid, w_grid], dim=0).reshape(3, -1)

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        position_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        mrope_deltas = []

        img_iter = iter(image_grid_thw) if image_grid_thw is not None else None
        if video_grid_thw is not None:
            vid_split = torch.repeat_interleave(
                video_grid_thw, video_grid_thw[:, 0], dim=0
            ).clone()
            vid_split[:, 0] = 1
            vid_iter = iter(vid_split)
        else:
            vid_iter = None

        img_tok_id = self.config.image_token_id
        vid_tok_id = self.config.video_token_id

        for batch_idx, cur_ids in enumerate(input_ids):
            if attention_mask is not None:
                valid_mask = attention_mask[batch_idx].bool()
                cur_ids = cur_ids[valid_mask]
            else:
                valid_mask = None

            token_types = torch.zeros_like(cur_ids, dtype=torch.int32)
            token_types[cur_ids == img_tok_id] = 1
            if vid_iter is not None:
                token_types[cur_ids == vid_tok_id] = 2

            groups = []
            for key, grp in itertools.groupby(
                enumerate(token_types.tolist()), lambda x: x[1]
            ):
                g_list = list(grp)
                groups.append((key, g_list[0][0], g_list[-1][0] + 1))

            current_pos = 0
            pos_list = []
            for mod_type, s_idx, e_idx in groups:
                if mod_type == 0:
                    t_len = e_idx - s_idx
                    pos_list.append(
                        torch.arange(t_len, device=input_ids.device)
                        .view(1, -1)
                        .expand(3, -1)
                        + current_pos
                    )
                    current_pos += t_len
                elif mod_type == 1 and img_iter is not None:
                    g_thw = next(img_iter)
                    v_pos = self.get_vision_position_ids(
                        current_pos,
                        g_thw,
                        1,
                        spatial_merge_size,
                        device=input_ids.device,
                    )
                    pos_list.append(v_pos)
                    current_pos += (
                        max(int(g_thw[1].item()), int(g_thw[2].item()))
                        // spatial_merge_size
                    )
                elif mod_type == 2 and vid_iter is not None:
                    grp_len = e_idx - s_idx
                    consumed = 0
                    while consumed < grp_len:
                        g_thw = next(vid_iter)
                        v_pos = self.get_vision_position_ids(
                            current_pos,
                            g_thw,
                            1,
                            spatial_merge_size,
                            device=input_ids.device,
                        )
                        pos_list.append(v_pos)
                        current_pos += (
                            max(int(g_thw[1].item()), int(g_thw[2].item()))
                            // spatial_merge_size
                        )
                        consumed += int(v_pos.shape[1])

            llm_pos = torch.cat(pos_list, dim=1).reshape(3, -1)
            if valid_mask is not None:
                position_ids[:, batch_idx, valid_mask] = llm_pos
            else:
                position_ids[:, batch_idx] = llm_pos
            mrope_deltas.append(llm_pos.max() + 1 - len(cur_ids))

        deltas_tensor = torch.tensor(mrope_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, deltas_tensor

    def get_image_features(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.LongTensor
    ) -> Tuple[torch.Tensor, ...]:
        pixel_values = pixel_values.to(
            device=self.visual.device, dtype=self.visual.dtype
        )
        vis_out = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (
            image_grid_thw.prod(-1) // (self.visual.spatial_merge_size**2)
        ).tolist()
        return torch.split(vis_out.pooler_output, split_sizes)

    def get_video_features(
        self, pixel_values_videos: torch.Tensor, video_grid_thw: torch.LongTensor
    ) -> Tuple[torch.Tensor, ...]:
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Qwen3_5HybridCache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Qwen3_5ModelOutput:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None and image_grid_thw is not None:
            img_splits = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(img_splits, dim=0).to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            del img_splits, image_embeds, image_mask

        if pixel_values_videos is not None and video_grid_thw is not None:
            vid_splits = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(vid_splits, dim=0).to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            del vid_splits, video_embeds, video_mask

        has_past = (
            past_key_values is not None and past_key_values.past_seen_tokens > 0
        )
        if position_ids is None and not has_past:
            if input_ids is not None and (
                image_grid_thw is not None or video_grid_thw is not None
            ):
                position_ids, self.rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                )
            else:
                self.rope_deltas = None

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        if outputs.past_key_values is not None and not has_past:
            outputs.past_key_values.rope_deltas = self.rope_deltas

        return Qwen3_5ModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            rope_deltas=self.rope_deltas,
            past_key_values=outputs.past_key_values,
        )


Qwen35ForConditionalGeneration = Qwen3_5Model

__all__ = [
    "Qwen3_5Model",
    "Qwen35ForConditionalGeneration",
]