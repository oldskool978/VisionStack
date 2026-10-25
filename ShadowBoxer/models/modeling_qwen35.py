from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

_FP8_DTYPES: Set[torch.dtype] = {
    dtype
    for name in (
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
    )
    if (dtype := getattr(torch, name, None)) is not None
}

_STRICTLY_UPPER_MASK_CACHE: OrderedDict[Tuple[int, str], torch.Tensor] = OrderedDict()


def _resolve_compute_dtype(
    weight: torch.Tensor, fallback: torch.dtype = torch.bfloat16
) -> torch.dtype:
    if weight.dtype in _FP8_DTYPES:
        return torch.bfloat16 if fallback in _FP8_DTYPES else fallback
    return weight.dtype


def _linear(layer: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    w = layer.weight
    if w.dtype in _FP8_DTYPES:
        compute_dtype = (
            x.dtype
            if x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            else torch.bfloat16
        )
        x_in = x if x.dtype == compute_dtype else x.to(compute_dtype)
        b = layer.bias.to(compute_dtype) if layer.bias is not None else None
        return F.linear(x_in, w.to(compute_dtype), b)
    if x.dtype != w.dtype:
        x = x.to(w.dtype)
    return layer(x)


def _get_strictly_upper_mask(chunk_size: int, device: torch.device) -> torch.Tensor:
    key = (chunk_size, str(device))
    cached = _STRICTLY_UPPER_MASK_CACHE.get(key)
    if cached is not None:
        _STRICTLY_UPPER_MASK_CACHE.move_to_end(key)
        return cached
    mask = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=device).triu(1)
    _STRICTLY_UPPER_MASK_CACHE[key] = mask
    while len(_STRICTLY_UPPER_MASK_CACHE) > 8:
        _STRICTLY_UPPER_MASK_CACHE.popitem(last=False)
    return mask


@dataclass
class Qwen3_5ModelOutput:
    last_hidden_state: torch.Tensor
    pooler_output: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None
    rope_deltas: Optional[torch.Tensor] = None


@dataclass
class Qwen3_5VisionConfig:
    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 4096
    num_position_embeddings: int = 2304
    initializer_range: float = 0.02
    rope_parameters: Dict[str, Any] = field(
        default_factory=lambda: {"rope_type": "axial", "rope_theta": 10000.0}
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Qwen3_5VisionConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        if "rope_parameters" not in filtered or not filtered["rope_parameters"]:
            filtered["rope_parameters"] = {
                "rope_type": "axial",
                "rope_theta": float(data.get("rope_theta", 10000.0)),
            }
        return cls(**filtered)


@dataclass
class Qwen3_5TextConfig:
    vocab_size: int = 248320
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0
    head_dim: int = 256
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    layer_types: Optional[List[str]] = None
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = 248044
    rope_parameters: Dict[str, Any] = field(
        default_factory=lambda: {
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
        }
    )

    def __post_init__(self) -> None:
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention" if bool((i + 1) % 4) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Qwen3_5TextConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        if "rope_parameters" not in filtered or not filtered["rope_parameters"]:
            rope_scaling = data.get("rope_scaling") or {}
            filtered["rope_parameters"] = {
                "rope_type": rope_scaling.get("rope_type", "default"),
                "rope_theta": float(data.get("rope_theta", 10000000.0)),
                "partial_rotary_factor": float(
                    data.get(
                        "partial_rotary_factor",
                        rope_scaling.get("partial_rotary_factor", 0.25),
                    )
                ),
                "mrope_section": list(
                    rope_scaling.get("mrope_section", [11, 11, 10])
                ),
            }
        return cls(**filtered)


@dataclass
class Qwen3_5Config:
    text_config: Qwen3_5TextConfig = field(default_factory=Qwen3_5TextConfig)
    vision_config: Qwen3_5VisionConfig = field(default_factory=Qwen3_5VisionConfig)
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    tie_word_embeddings: bool = False

    @classmethod
    def from_json_file(cls, config_path: Union[str, Path]) -> Qwen3_5Config:
        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
        t_dict = raw.get("text_config", raw)
        v_dict = raw.get("vision_config", {})
        return cls(
            text_config=Qwen3_5TextConfig.from_dict(t_dict),
            vision_config=Qwen3_5VisionConfig.from_dict(v_dict),
            image_token_id=int(raw.get("image_token_id", 248056)),
            video_token_id=int(raw.get("video_token_id", 248057)),
            vision_start_token_id=int(raw.get("vision_start_token_id", 248053)),
            vision_end_token_id=int(raw.get("vision_end_token_id", 248054)),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
        )


def _act_fn(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "silu":
        return F.silu(x)
    if name == "gelu_pytorch_tanh":
        return F.gelu(x, approximate="tanh")
    if name == "gelu":
        return F.gelu(x)
    return F.silu(x)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _l2norm_fp32(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


class Qwen3_5RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        normed = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * (1.0 + self.weight.float())).to(dtype=x.dtype)


class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        in_dtype = hidden_states.dtype
        h_f32 = hidden_states.float()
        var = h_f32.pow(2).mean(-1, keepdim=True)
        normed = h_f32 * torch.rsqrt(var + self.variance_epsilon)
        scaled = self.weight * normed.to(in_dtype)
        gated = scaled.float() * F.silu(gate.float())
        return gated.to(in_dtype)


def _causal_conv1d_fn(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: str = "silu",
) -> torch.Tensor:
    _, hidden_size, seq_len = hidden_states.shape
    padding = weight.shape[-1] - 1
    w_dtype = _resolve_compute_dtype(weight, hidden_states.dtype)
    w_conv = weight.to(w_dtype).unsqueeze(1)
    b_conv = bias.to(w_dtype) if bias is not None else None
    out = F.conv1d(
        hidden_states.to(w_dtype),
        weight=w_conv,
        bias=b_conv,
        padding=padding,
        groups=hidden_size,
    )[:, :, :seq_len]
    if activation:
        out = _act_fn(out, activation)
    return out.to(hidden_states.dtype)


def _solve_unit_lower_triangular(
    ut_system: torch.Tensor, rhs: torch.Tensor
) -> torch.Tensor:
    try:
        return torch.linalg.solve_triangular(
            ut_system, rhs, upper=False, unitriangular=True
        )
    except Exception:
        chunk_size = ut_system.shape[-1]
        mat = -ut_system.tril(-1)
        for i in range(1, chunk_size):
            row = mat[..., i, :i].clone()
            sub = mat[..., :i, :i].clone()
            mat[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        mat = mat + torch.eye(chunk_size, dtype=mat.dtype, device=mat.device)
        return mat @ rhs


def _chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
) -> torch.Tensor:
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
    padded_output_shape = (batch_size, num_v_heads, -1, v_head_dim)

    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, g)
    ]

    if use_qk_l2norm:
        query = _l2norm_fp32(query, dim=-1, eps=1e-6)
        key = _l2norm_fp32(key, dim=-1, eps=1e-6)
    query = query * (query.shape[-1] ** -0.5)

    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    if pad_size > 0:
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        decay = F.pad(decay, (0, pad_size))

    total_sequence_length = sequence_length + pad_size
    num_chunks = total_sequence_length // chunk_size

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)

    strictly_upper_mask = _get_strictly_upper_mask(chunk_size, query.device)
    cum_decay = decay.cumsum(dim=3)
    exp_cum_decay = cum_decay.exp()

    pairwise_decay = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)
    pairwise_decay = pairwise_decay.masked_fill(strictly_upper_mask, float("-inf")).exp()

    key_t = key.transpose(-1, -2)
    ut_system = (k_beta @ key_t) * pairwise_decay
    intra_chunk_attn = (query @ key_t) * pairwise_decay
    decayed_k_beta = k_beta * exp_cum_decay.unsqueeze(-1)

    rhs_combined = torch.cat([v_beta, decayed_k_beta], dim=-1)
    solved_combined = _solve_unit_lower_triangular(ut_system, rhs_combined)
    new_values, k_cumdecay = torch.split(
        solved_combined, [v_head_dim, k_head_dim], dim=-1
    )

    last_recurrent_state = torch.zeros(
        recurrent_state_shape, dtype=new_values.dtype, device=new_values.device
    )
    core_attn_out = torch.zeros_like(new_values)

    query = query * exp_cum_decay.unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = exp_cum_decay[..., -1][..., None, None]

    for i in range(num_chunks):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ last_recurrent_state
        inter_chunk_attn = query[:, :, i] @ last_recurrent_state
        core_attn_out[:, :, i] = inter_chunk_attn + intra_chunk_attn[:, :, i] @ v_new
        last_recurrent_state = (
            last_recurrent_state * chunk_decay[:, :, i]
            + key[:, :, i].transpose(-1, -2) @ v_new
        )

    core_attn_out = core_attn_out.reshape(padded_output_shape)[:, :, :sequence_length]
    return core_attn_out.transpose(1, 2).to(
        initial_dtype, memory_format=torch.contiguous_format
    )


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is not None and attention_mask.ndim == 2:
            hidden_states = (
                hidden_states * attention_mask[:, :, None].to(hidden_states.dtype)
            )
        batch_size, seq_len, _ = hidden_states.shape

        mixed_qkv = _linear(self.in_proj_qkv, hidden_states).transpose(1, 2)
        z = _linear(self.in_proj_z, hidden_states).reshape(
            batch_size, seq_len, -1, self.head_v_dim
        )
        b = _linear(self.in_proj_b, hidden_states)
        a = _linear(self.in_proj_a, hidden_states)

        mixed_qkv = _causal_conv1d_fn(
            mixed_qkv,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            activation=self.activation,
        ).transpose(1, 2)

        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())

        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        core_attn_out = _chunk_gated_delta_rule(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            use_qk_l2norm=True,
        )

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z).reshape(batch_size, seq_len, -1)
        return _linear(self.out_proj, core_attn_out)


class Qwen3_5TextRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.config = config
        rope_params = config.rope_parameters or {}
        self.base = float(rope_params.get("rope_theta", 10000000.0))
        partial_factor = float(rope_params.get("partial_rotary_factor", 0.25))
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.dim = int(head_dim * partial_factor)
        self.register_buffer(
            "inv_freq", self._compute_inv_freq(torch.device("cpu")), persistent=False
        )
        self.mrope_section = list(rope_params.get("mrope_section", [11, 11, 10]))

    def _compute_inv_freq(self, device: torch.device) -> torch.Tensor:
        return (
            1.0
            / (
                self.base
                ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
            )
        ).to(device)

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.inv_freq = self._compute_inv_freq(self.inv_freq.device)
        return self

    def _recompose(self, freq: torch.Tensor) -> torch.Tensor:
        freqs_thw = freq[0].clone()
        for dim_idx, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim_idx] * 3
            idx = slice(offset, length, 3)
            freqs_thw[..., idx] = freq[dim_idx, ..., idx]
        return torch.cat((freqs_thw, freqs_thw), dim=-1)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.inv_freq.dtype != torch.float32 or self.inv_freq.device != x.device:
            self.inv_freq = self._compute_inv_freq(x.device)
        pos_3d = position_ids.expand(3, -1, -1)
        freqs = pos_3d[..., None].float() * self.inv_freq
        cos = self._recompose(freqs.cos())
        sin = self._recompose(freqs.sin())
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _apply_rotary_pos_emb_partial(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


class Qwen3_5Attention(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim * 2,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            _linear(self.q_proj, hidden_states).view(
                *input_shape, -1, self.head_dim * 2
            ),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(
            _linear(self.k_proj, hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = (
            _linear(self.v_proj, hidden_states).view(hidden_shape).transpose(1, 2)
        )

        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb_partial(
            query_states, key_states, cos, sin
        )

        enable_gqa = self.num_key_value_groups > 1
        if attention_mask is not None and attention_mask.ndim == 4:
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                is_causal=False,
                scale=self.scaling,
                enable_gqa=enable_gqa,
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                is_causal=(query_states.shape[2] > 1),
                scale=self.scaling,
                enable_gqa=enable_gqa,
            )

        attn_output = (
            attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        )
        attn_output = attn_output * torch.sigmoid(gate)
        return _linear(self.o_proj, attn_output)


class Qwen3_5MLP(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, intermediate_size: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_name = config.hidden_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _linear(
            self.down_proj,
            _act_fn(_linear(self.gate_proj, x), self.act_name)
            * _linear(self.up_proj, x),
        )


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.block_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        linear_mask: Optional[torch.Tensor] = None,
        full_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.block_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states, attention_mask=linear_mask)
        else:
            hidden_states = self.self_attn(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=full_attn_mask,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3_5VisionRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        rope_params = config.rope_parameters or {}
        self.base = float(rope_params.get("rope_theta", 10000.0))
        dim = config.hidden_size // config.num_heads
        self.spatial_dim = dim // 2
        self.register_buffer(
            "inv_freq", self._compute_inv_freq(torch.device("cpu")), persistent=False
        )

    def _compute_inv_freq(self, device: torch.device) -> torch.Tensor:
        return (
            1.0
            / (
                self.base
                ** (
                    torch.arange(0, self.spatial_dim, 2, dtype=torch.float32)
                    / self.spatial_dim
                )
            )
        ).to(device)

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.inv_freq = self._compute_inv_freq(self.inv_freq.device)
        return self

    @staticmethod
    def _recompose(freq: torch.Tensor) -> torch.Tensor:
        freq_h, freq_w = freq[0], freq[1]
        freq_hw = torch.cat([freq_h, freq_w], dim=-1)
        return torch.cat([freq_hw, freq_hw], dim=-1)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.inv_freq.dtype != torch.float32 or self.inv_freq.device != x.device:
            self.inv_freq = self._compute_inv_freq(x.device)
        freqs = position_ids[..., None].float() * self.inv_freq
        cos = self._recompose(freqs.cos())
        sin = self._recompose(freqs.sin())
        return cos, sin


class Qwen3_5VisionMLP(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_name = config.hidden_act

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return _linear(
            self.linear_fc2,
            _act_fn(_linear(self.linear_fc1, hidden_state), self.act_name),
        )


class Qwen3_5VisionPatchEmbed(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel_size = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = _resolve_compute_dtype(self.proj.weight, hidden_states.dtype)
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        if self.proj.weight.dtype in _FP8_DTYPES:
            b = self.proj.bias.to(target_dtype) if self.proj.bias is not None else None
            out = F.conv3d(
                hidden_states.to(target_dtype),
                self.proj.weight.to(target_dtype),
                b,
                stride=self.proj.stride,
            )
            return out.view(-1, self.embed_dim)
        return self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)


class Qwen3_5VisionPatchMerger(nn.Module):
    def __init__(
        self, config: Qwen3_5VisionConfig, use_postshuffle_norm: bool = False
    ):
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6
        )
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(
            x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x
        ).view(-1, self.hidden_size)
        return _linear(self.linear_fc2, self.act_fn(_linear(self.linear_fc1, normed)))


def _apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q_f32, k_f32 = q.float(), k.float()
    cos_f32, sin_f32 = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q_f32 * cos_f32) + (_rotate_half(q_f32) * sin_f32)
    k_embed = (k_f32 * cos_f32) + (_rotate_half(k_f32) * sin_f32)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class Qwen3_5VisionAttention(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            _linear(self.qkv, hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb_vision(
            query_states, key_states, cos, sin
        )

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        if cu_seqlens.numel() == 2:
            attn_output = (
                F.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    is_causal=False,
                    scale=self.scaling,
                )
                .transpose(1, 2)
                .reshape(seq_length, -1)
                .contiguous()
            )
        else:
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            q_splits = torch.split(query_states, lengths, dim=2)
            k_splits = torch.split(key_states, lengths, dim=2)
            v_splits = torch.split(value_states, lengths, dim=2)
            attn_outputs = [
                F.scaled_dot_product_attention(
                    q, k, v, is_causal=False, scale=self.scaling
                ).transpose(1, 2)
                for q, k, v in zip(q_splits, k_splits, v_splits)
            ]
            attn_output = (
                torch.cat(attn_outputs, dim=1).reshape(seq_length, -1).contiguous()
            )
        return _linear(self.proj, attn_output)


class Qwen3_5VisionBlock(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3_5VisionAttention(config)
        self.mlp = Qwen3_5VisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3_5VisionModel(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3_5VisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(
            config.num_position_embeddings, config.hidden_size
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        self.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(config)
        self.blocks = nn.ModuleList(
            [Qwen3_5VisionBlock(config) for _ in range(config.depth)]
        )
        self.merger = Qwen3_5VisionPatchMerger(config, use_postshuffle_norm=False)
        self._geom_cache: OrderedDict[
            Tuple[Tuple[Tuple[int, int, int], ...], str, torch.dtype, int, int],
            Tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                Tuple[torch.Tensor, torch.Tensor],
            ],
        ] = OrderedDict()

    @property
    def dtype(self) -> torch.dtype:
        return _resolve_compute_dtype(self.patch_embed.proj.weight)

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def clear_caches(self) -> None:
        self._geom_cache.clear()

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.clear_caches()
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.clear_caches()

    def _compute_geometry(
        self, grid_thw: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        merge = self.spatial_merge_size
        side = self.num_grid_per_side

        pos_ids_list: List[torch.Tensor] = []
        interp_embeds_list: List[torch.Tensor] = []
        seqlens: List[int] = [0]

        for row in grid_thw.tolist():
            t, h, w = int(row[0]), int(row[1]), int(row[2])
            h_ids = torch.arange(h, device=device)
            w_ids = torch.arange(w, device=device)
            gh, gw = torch.meshgrid(h_ids, w_ids, indexing="ij")
            gh = (
                gh.reshape(h // merge, merge, w // merge, merge)
                .permute(0, 2, 1, 3)
                .flatten()
            )
            gw = (
                gw.reshape(h // merge, merge, w // merge, merge)
                .permute(0, 2, 1, 3)
                .flatten()
            )
            hw_ids = torch.stack([gh, gw], dim=0).repeat(1, t)
            pos_ids_list.append(hw_ids)

            if h == 1:
                y_coords = torch.zeros(1, device=device, dtype=torch.float32)
            else:
                y_coords = torch.linspace(
                    0, side - 1, h, device=device, dtype=torch.float32
                )
            if w == 1:
                x_coords = torch.zeros(1, device=device, dtype=torch.float32)
            else:
                x_coords = torch.linspace(
                    0, side - 1, w, device=device, dtype=torch.float32
                )

            y0 = y_coords.floor().long().clamp(0, side - 1)
            x0 = x_coords.floor().long().clamp(0, side - 1)
            y1 = (y0 + 1).clamp(0, side - 1)
            x1 = (x0 + 1).clamp(0, side - 1)
            wy = y_coords - y0.float()
            wx = x_coords - x0.float()

            idx_00 = (y0[:, None] * side + x0[None, :]).flatten()
            idx_01 = (y0[:, None] * side + x1[None, :]).flatten()
            idx_10 = (y1[:, None] * side + x0[None, :]).flatten()
            idx_11 = (y1[:, None] * side + x1[None, :]).flatten()

            w_00 = ((1.0 - wy)[:, None] * (1.0 - wx)[None, :]).flatten().unsqueeze(-1)
            w_01 = ((1.0 - wy)[:, None] * wx[None, :]).flatten().unsqueeze(-1)
            w_10 = (wy[:, None] * (1.0 - wx)[None, :]).flatten().unsqueeze(-1)
            w_11 = (wy[:, None] * wx[None, :]).flatten().unsqueeze(-1)

            emb_2d = (
                self.pos_embed(idx_00) * w_00
                + self.pos_embed(idx_01) * w_01
                + self.pos_embed(idx_10) * w_10
                + self.pos_embed(idx_11) * w_11
            )
            emb_merged = (
                emb_2d.view(h // merge, merge, w // merge, merge, -1)
                .permute(0, 2, 1, 3, 4)
                .reshape(h * w, -1)
                .repeat(t, 1)
            )
            interp_embeds_list.append(emb_merged)

            for _ in range(t):
                seqlens.append(seqlens[-1] + h * w)

        position_ids = torch.cat(pos_ids_list, dim=1)
        pos_embeds = torch.cat(interp_embeds_list, dim=0)
        cu_seqlens = torch.tensor(seqlens, dtype=torch.int32, device=device)
        return position_ids, pos_embeds, cu_seqlens

    def _get_cached_geometry_and_rotary(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        device = hidden_states.device
        grid_key = tuple(
            (int(r[0]), int(r[1]), int(r[2])) for r in grid_thw.tolist()
        )
        cache_key = (
            grid_key,
            str(device),
            self.pos_embed.weight.dtype,
            id(self.pos_embed.weight),
            int(self.pos_embed.weight.data_ptr()),
        )
        cached = self._geom_cache.get(cache_key)
        if cached is not None:
            self._geom_cache.move_to_end(cache_key)
            return cached

        position_ids, pos_embeds, cu_seqlens = self._compute_geometry(grid_thw, device)
        position_embeddings = self.rotary_pos_emb(hidden_states, position_ids)
        entry = (position_ids, pos_embeds, cu_seqlens, position_embeddings)
        self._geom_cache[cache_key] = entry
        while len(self._geom_cache) > 8:
            self._geom_cache.popitem(last=False)
        return entry

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> Qwen3_5ModelOutput:
        _, pos_embeds, cu_seqlens, position_embeddings = (
            self._get_cached_geometry_and_rotary(hidden_states, grid_thw)
        )
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )

        merged_hidden_states = self.merger(hidden_states)
        return Qwen3_5ModelOutput(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )


class Qwen3_5TextModel(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [
                Qwen3_5DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Qwen3_5ModelOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device

        if position_ids is None:
            pos = torch.arange(seq_len, device=device).view(1, 1, -1)
            position_ids = pos.expand(3, batch_size, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, batch_size, -1)
        elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]

        linear_mask: Optional[torch.Tensor] = None
        full_attn_mask: Optional[torch.Tensor] = None
        if attention_mask is not None and attention_mask.ndim == 2:
            if not attention_mask.bool().all():
                linear_mask = attention_mask.bool()
                causal = torch.tril(
                    torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
                )
                full_attn_mask = (
                    causal[None, None, :, :] & linear_mask[:, None, None, :]
                )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                linear_mask=linear_mask,
                full_attn_mask=full_attn_mask,
            )

        hidden_states = self.norm(hidden_states)
        return Qwen3_5ModelOutput(last_hidden_state=hidden_states)


class Qwen3_5Model(nn.Module):
    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.config = config
        self.visual = Qwen3_5VisionModel(config.vision_config)
        self.language_model = Qwen3_5TextModel(config.text_config)
        self.rope_deltas: Optional[torch.Tensor] = None

    @classmethod
    def from_config_dir(cls, model_dir: Union[str, Path]) -> Qwen3_5Model:
        cfg_file = Path(model_dir) / "config.json"
        cfg = Qwen3_5Config.from_json_file(cfg_file)
        return cls(cfg)

    def clear_caches(self) -> None:
        self.visual.clear_caches()
        self.rope_deltas = None

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.embed_tokens

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

        for batch_idx, cur_ids in enumerate(input_ids):
            if attention_mask is not None:
                valid_mask = attention_mask[batch_idx].bool()
                cur_ids = cur_ids[valid_mask]
            else:
                valid_mask = None

            token_types = (cur_ids == self.config.image_token_id).int().tolist()
            groups = []
            for key, grp in itertools.groupby(enumerate(token_types), lambda x: x[1]):
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
                else:
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
        pixel_values = pixel_values.to(device=self.visual.device, dtype=self.visual.dtype)
        vis_out = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (
            image_grid_thw.prod(-1) // (self.visual.spatial_merge_size**2)
        ).tolist()
        return torch.split(vis_out.pooler_output, split_sizes)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
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

        if position_ids is None:
            if input_ids is not None and image_grid_thw is not None:
                position_ids, self.rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask,
                )
            else:
                self.rope_deltas = None

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        return Qwen3_5ModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            rope_deltas=self.rope_deltas,
        )