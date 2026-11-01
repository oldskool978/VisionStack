# ShadowBoxer/models/qwen35/deltanet.py
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from .cache import Qwen3_5HybridCache
from .config import (
    Qwen3_5RMSNormGated,
    Qwen3_5TextConfig,
    _FP8_DTYPES,
    _act_fn,
    _get_eye_matrix,
    _get_strictly_upper_mask,
    _l2norm_fp32,
    _linear,
    _resolve_compute_dtype,
)


def _extract_conv_state_at_offset(
    mixed_qkv: torch.Tensor,
    prev_conv_state: Optional[torch.Tensor],
    local_offset: int,
    state_len: int = 3,
) -> torch.Tensor:
    if local_offset >= state_len:
        return mixed_qkv[:, :, local_offset - state_len : local_offset].contiguous()
    if prev_conv_state is not None:
        return torch.cat(
            [prev_conv_state[:, :, local_offset:], mixed_qkv[:, :, :local_offset]],
            dim=-1,
        ).contiguous()
    pad_left = state_len - local_offset
    return F.pad(mixed_qkv[:, :, :local_offset], (pad_left, 0)).contiguous()


def _causal_conv1d_fn(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: str = "silu",
    conv_state: Optional[torch.Tensor] = None,
    return_final_state: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    seq_len = hidden_states.shape[-1]
    w_dtype = _resolve_compute_dtype(weight, hidden_states.dtype)
    w = weight.squeeze(1) if weight.ndim == 3 else weight
    w = w if w.dtype == w_dtype else w.to(w_dtype)
    k_size = w.shape[-1]
    state_len = k_size - 1

    new_conv_state: Optional[torch.Tensor] = None
    if return_final_state:
        new_conv_state = _extract_conv_state_at_offset(
            hidden_states, conv_state, seq_len, state_len=state_len
        )

    x = hidden_states.transpose(1, 2)
    if x.dtype != w_dtype:
        x = x.to(w_dtype)

    out = x * w[:, k_size - 1]
    for tap in range(1, min(k_size, seq_len)):
        out[:, tap:, :].addcmul_(x[:, :-tap, :], w[:, k_size - 1 - tap])

    if conv_state is not None and state_len > 0:
        c_state = conv_state.transpose(1, 2)
        if c_state.dtype != w_dtype:
            c_state = c_state.to(w_dtype)
        for tap in range(1, k_size):
            n_from_state = min(tap, seq_len)
            state_start = state_len - tap
            out[:, :n_from_state, :].addcmul_(
                c_state[:, state_start : state_start + n_from_state, :],
                w[:, k_size - 1 - tap],
            )

    if bias is not None:
        out = out + bias.to(w_dtype)
    if activation:
        out = _act_fn(out, activation)

    res = out.to(hidden_states.dtype).transpose(1, 2)
    if return_final_state and new_conv_state is not None:
        return res, new_conv_state
    return res


def _invert_unit_lower_triangular_block(u_blk: torch.Tensor) -> torch.Tensor:
    b_sz = u_blk.shape[-1]
    eye = _get_eye_matrix(b_sz, u_blk.dtype, u_blk.device)
    a = -u_blk.tril(-1)
    inv = eye + a
    if b_sz == 16:
        a2 = a @ a
        a4 = a2 @ a2
        a8 = a4 @ a4
        return ((inv @ (eye + a2)) @ (eye + a4)) @ (eye + a8)
    cur = a
    step = 1
    while step < b_sz:
        cur = cur @ cur
        inv = inv @ (eye + cur)
        step <<= 1
    return inv


def _solve_unit_lower_triangular(
    ut_system: torch.Tensor, rhs: torch.Tensor
) -> torch.Tensor:
    n = ut_system.shape[-1]
    if n <= 16 or (n & (n - 1)) != 0:
        inv_mat = _invert_unit_lower_triangular_block(ut_system)
        return inv_mat @ rhs

    block_size = 16
    num_blocks = n // block_size
    batch_shape = ut_system.shape[:-2]

    u_blocks = ut_system.reshape(
        *batch_shape, num_blocks, block_size, num_blocks, block_size
    )
    u_diag = u_blocks.diagonal(dim1=-4, dim2=-2).movedim(-1, -3).contiguous()
    inv_blocks = _invert_unit_lower_triangular_block(u_diag)

    cur_size = block_size
    cur_num = num_blocks
    while cur_num > 1:
        half_num = cur_num // 2
        inv_reshaped = inv_blocks.reshape(
            *batch_shape, half_num, 2, cur_size, cur_size
        )
        t11 = inv_reshaped[..., 0, :, :]
        t22 = inv_reshaped[..., 1, :, :]

        u_part = ut_system.reshape(
            *batch_shape, half_num, 2, cur_size, half_num, 2, cur_size
        )
        u_sub = u_part[..., 1, :, :, 0, :]
        u21 = u_sub.diagonal(dim1=-4, dim2=-2).movedim(-1, -3).contiguous()

        t21 = -(t22 @ (u21 @ t11))

        next_inv = torch.zeros(
            (*batch_shape, half_num, cur_size * 2, cur_size * 2),
            dtype=ut_system.dtype,
            device=ut_system.device,
        )
        next_inv[..., :cur_size, :cur_size] = t11
        next_inv[..., cur_size:, :cur_size] = t21
        next_inv[..., cur_size:, cur_size:] = t22

        inv_blocks = next_inv
        cur_size *= 2
        cur_num = half_num

    mat = inv_blocks.squeeze(-3)
    return mat @ rhs


def _recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = True,
    checkpoint_positions: Optional[Sequence[int]] = None,
    checkpoint_states_out: Optional[Dict[int, torch.Tensor]] = None,
    past_seen_tokens: int = 0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]

    q_f32, k_f32, v_f32, beta_f32, g_f32 = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, g)
    ]
    if use_qk_l2norm:
        q_f32 = _l2norm_fp32(q_f32, dim=-1, eps=1e-6)
        k_f32 = _l2norm_fp32(k_f32, dim=-1, eps=1e-6)
    q_f32 = q_f32 * (k_head_dim**-0.5)
    decay_exp = g_f32.exp()

    state = (
        initial_state.to(device=q_f32.device, dtype=torch.float32).clone()
        if initial_state is not None
        else torch.zeros(
            (batch_size, num_v_heads, k_head_dim, v_head_dim),
            dtype=torch.float32,
            device=q_f32.device,
        )
    )
    out = torch.empty(
        (batch_size, num_v_heads, sequence_length, v_head_dim),
        dtype=torch.float32,
        device=q_f32.device,
    )
    cp_set = (
        set(int(x) for x in checkpoint_positions) if checkpoint_positions else None
    )

    for t in range(sequence_length):
        q_t = q_f32[:, :, t, :].unsqueeze(-2)
        k_t = k_f32[:, :, t, :].unsqueeze(-2)
        v_t = v_f32[:, :, t, :].unsqueeze(-2)
        b_t = beta_f32[:, :, t].unsqueeze(-1).unsqueeze(-1)
        d_t = decay_exp[:, :, t].unsqueeze(-1).unsqueeze(-1)

        state = state * d_t
        v_pred = k_t @ state
        v_new = b_t * (v_t - v_pred)
        state = state + k_t.transpose(-1, -2) @ v_new
        out[:, :, t : t + 1, :] = q_t @ state

        if cp_set is not None and checkpoint_states_out is not None:
            abs_tok = past_seen_tokens + t + 1
            if abs_tok in cp_set:
                checkpoint_states_out[abs_tok] = state.clone()

    out_t = out.transpose(1, 2).to(
        initial_dtype, memory_format=torch.contiguous_format
    )
    if output_final_state:
        return out_t, state
    return out_t


def _chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = True,
    checkpoint_positions: Optional[Sequence[int]] = None,
    checkpoint_states_out: Optional[Dict[int, torch.Tensor]] = None,
    past_seen_tokens: int = 0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    sequence_length = key.shape[1]
    if initial_state is not None and sequence_length <= 8:
        return _recurrent_gated_delta_rule(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm=use_qk_l2norm,
            checkpoint_positions=checkpoint_positions,
            checkpoint_states_out=checkpoint_states_out,
            past_seen_tokens=past_seen_tokens,
        )

    initial_dtype = query.dtype
    batch_size, _, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
    padded_output_shape = (batch_size, num_v_heads, -1, v_head_dim)

    if chunk_size == 64:
        if sequence_length <= 16:
            chunk_size = 16
        elif sequence_length <= 32:
            chunk_size = 32

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
    del value, beta

    query, key, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)

    strictly_upper_mask = _get_strictly_upper_mask(chunk_size, query.device)
    cum_decay = decay.cumsum(dim=3)
    del decay
    exp_cum_decay = cum_decay.exp()

    pairwise_decay = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)
    pairwise_decay = pairwise_decay.masked_fill(
        strictly_upper_mask, float("-inf")
    ).exp()

    key_t = key.transpose(-1, -2)
    ut_system = (k_beta @ key_t) * pairwise_decay
    intra_chunk_attn = (query @ key_t) * pairwise_decay
    del pairwise_decay, key_t

    decayed_k_beta = k_beta * exp_cum_decay.unsqueeze(-1)
    del k_beta
    rhs_combined = torch.cat([v_beta, decayed_k_beta], dim=-1)
    del v_beta, decayed_k_beta

    solved_combined = _solve_unit_lower_triangular(ut_system, rhs_combined)
    del ut_system, rhs_combined
    new_values, k_cumdecay = torch.split(
        solved_combined, [v_head_dim, k_head_dim], dim=-1
    )
    del solved_combined

    last_recurrent_state = (
        initial_state.to(dtype=new_values.dtype, device=new_values.device).clone()
        if initial_state is not None
        else torch.zeros(
            recurrent_state_shape, dtype=new_values.dtype, device=new_values.device
        )
    )

    core_attn_out = torch.zeros_like(new_values)

    has_subchunk_cp = False
    chunk_cp_map: Dict[int, List[Tuple[int, int]]] = {}
    if checkpoint_positions and checkpoint_states_out is not None:
        for cp in checkpoint_positions:
            local_cp = int(cp) - past_seen_tokens
            if 1 <= local_cp <= sequence_length:
                c_idx = (local_cp - 1) // chunk_size
                s_off = local_cp - c_idx * chunk_size
                chunk_cp_map.setdefault(c_idx, []).append((s_off, int(cp)))
                if s_off != chunk_size:
                    has_subchunk_cp = True

    key_unscaled = key if has_subchunk_cp else None
    cum_decay_saved = cum_decay if has_subchunk_cp else None

    query = query * exp_cum_decay.unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    if not has_subchunk_cp:
        del cum_decay
    chunk_decay = exp_cum_decay[..., -1][..., None, None]
    del exp_cum_decay

    for i in range(num_chunks):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ last_recurrent_state
        inter_chunk_attn = query[:, :, i] @ last_recurrent_state
        core_attn_out[:, :, i] = inter_chunk_attn + intra_chunk_attn[:, :, i] @ v_new
        next_recurrent_state = (
            last_recurrent_state * chunk_decay[:, :, i]
            + key[:, :, i].transpose(-1, -2) @ v_new
        )
        if i in chunk_cp_map and checkpoint_states_out is not None:
            for s_off, abs_cp in chunk_cp_map[i]:
                if s_off == chunk_size:
                    checkpoint_states_out[abs_cp] = next_recurrent_state.clone()
                elif key_unscaled is not None and cum_decay_saved is not None:
                    decay_s = cum_decay_saved[:, :, i, s_off - 1 : s_off]
                    k_s = key_unscaled[:, :, i, :s_off] * (
                        decay_s - cum_decay_saved[:, :, i, :s_off]
                    ).exp().unsqueeze(-1)
                    state_s = (
                        last_recurrent_state * decay_s.exp().unsqueeze(-1)
                        + k_s.transpose(-1, -2) @ v_new[:, :, :s_off]
                    )
                    checkpoint_states_out[abs_cp] = state_s.clone()
        last_recurrent_state = next_recurrent_state

    del (
        query,
        key,
        key_unscaled,
        cum_decay_saved,
        new_values,
        k_cumdecay,
        intra_chunk_attn,
        chunk_decay,
    )
    core_attn_out = core_attn_out.reshape(padded_output_shape)[:, :, :sequence_length]
    out_tensor = core_attn_out.transpose(1, 2).to(
        initial_dtype, memory_format=torch.contiguous_format
    )
    if output_final_state:
        return out_tensor, last_recurrent_state
    return out_tensor


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
        cache: Optional[Qwen3_5HybridCache] = None,
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

        prev_conv_state = (
            cache.conv_states.get(self.layer_idx) if cache is not None else None
        )
        prev_rec_state = (
            cache.recurrent_states.get(self.layer_idx) if cache is not None else None
        )
        past_seen = cache.past_seen_tokens if cache is not None else 0
        cp_offsets = (
            cache.checkpoint_offsets
            if (cache is not None and cache.record_checkpoints)
            else None
        )

        if cp_offsets and cache is not None:
            state_len = self.conv_kernel_size - 1
            for cp in cp_offsets:
                local_off = cp - past_seen
                if 1 <= local_off <= seq_len:
                    c_snap = _extract_conv_state_at_offset(
                        mixed_qkv, prev_conv_state, local_off, state_len=state_len
                    )
                    cache.conv_checkpoints.setdefault(cp, {})[self.layer_idx] = c_snap

        if cache is not None:
            conv_out, next_conv_state = _causal_conv1d_fn(
                mixed_qkv,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation=self.activation,
                conv_state=prev_conv_state,
                return_final_state=True,
            )  # type: ignore[misc]
            cache.conv_states[self.layer_idx] = next_conv_state
            mixed_qkv = conv_out.transpose(1, 2)
        else:
            mixed_qkv = _causal_conv1d_fn(
                mixed_qkv,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation=self.activation,
            ).transpose(1, 2)  # type: ignore[union-attr]

        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        del mixed_qkv
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        del a, b

        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        if cache is not None:
            layer_rec_cps: Optional[Dict[int, torch.Tensor]] = (
                {} if cp_offsets else None
            )
            core_attn_out, next_rec_state = _chunk_gated_delta_rule(
                query=query,
                key=key,
                value=value,
                g=g,
                beta=beta,
                initial_state=prev_rec_state,
                output_final_state=True,
                use_qk_l2norm=True,
                checkpoint_positions=cp_offsets,
                checkpoint_states_out=layer_rec_cps,
                past_seen_tokens=past_seen,
            )  # type: ignore[misc]
            cache.recurrent_states[self.layer_idx] = next_rec_state
            if layer_rec_cps:
                for cp, r_state in layer_rec_cps.items():
                    cache.recurrent_checkpoints.setdefault(cp, {})[
                        self.layer_idx
                    ] = r_state
        else:
            core_attn_out = _chunk_gated_delta_rule(
                query=query,
                key=key,
                value=value,
                g=g,
                beta=beta,
                use_qk_l2norm=True,
            )  # type: ignore[assignment]
        del query, key, value, g, beta

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z).reshape(batch_size, seq_len, -1)
        del z
        return _linear(self.out_proj, core_attn_out)


__all__ = [
    "_extract_conv_state_at_offset",
    "_causal_conv1d_fn",
    "_invert_unit_lower_triangular_block",
    "_solve_unit_lower_triangular",
    "_recurrent_gated_delta_rule",
    "_chunk_gated_delta_rule",
    "Qwen3_5GatedDeltaNet",
]