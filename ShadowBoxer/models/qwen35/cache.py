# ShadowBoxer/models/qwen35/cache.py
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch


class Qwen3_5HybridCache:
    def __init__(
        self,
        record_checkpoints: bool = False,
        checkpoint_positions: Optional[Sequence[int]] = None,
        min_checkpoint_token: int = 0,
        structural_checkpoints: Optional[Sequence[int]] = None,
    ) -> None:
        self.conv_states: Dict[int, torch.Tensor] = {}
        self.recurrent_states: Dict[int, torch.Tensor] = {}
        self.key_cache: Dict[int, torch.Tensor] = {}
        self.value_cache: Dict[int, torch.Tensor] = {}
        self.past_seen_tokens: int = 0
        self.rope_deltas: Optional[torch.Tensor] = None
        self.record_checkpoints: bool = bool(record_checkpoints)
        self.requested_checkpoints: Optional[Set[int]] = (
            {int(x) for x in checkpoint_positions}
            if checkpoint_positions is not None
            else None
        )
        self.structural_checkpoints: Set[int] = (
            {int(x) for x in structural_checkpoints}
            if structural_checkpoints is not None
            else set()
        )
        self.min_checkpoint_token: int = int(min_checkpoint_token)
        self.checkpoint_offsets: List[int] = []
        self.conv_checkpoints: Dict[int, Dict[int, torch.Tensor]] = {}
        self.recurrent_checkpoints: Dict[int, Dict[int, torch.Tensor]] = {}

    @staticmethod
    def derive_structural_checkpoints(
        token_ids: Sequence[int],
        vision_end_token_id: int = 248054,
        im_end_token_id: int = 248046,
        max_structural: int = 8,
    ) -> Tuple[int, List[int]]:
        n_tok = len(token_ids)
        min_snap = 0
        vis_boundaries: List[int] = []
        for idx, tid in enumerate(token_ids):
            t_val = int(tid)
            if t_val == vision_end_token_id:
                min_snap = idx + 1
                vis_boundaries.append(idx + 1)
            elif (
                t_val in (im_end_token_id, 248044)
                and idx + 1 < n_tok
                and min_snap == 0
            ):
                vis_boundaries.append(idx + 1)

        anchors: List[int] = list(vis_boundaries)
        tail_start = max(min_snap, 8)
        tail_end = n_tok - 1
        if tail_end - tail_start >= 16:
            step = max(12, (tail_end - tail_start) // max(2, max_structural))
            cur = ((tail_start + 3) // 4) * 4
            while cur <= tail_end:
                anchors.append(cur)
                cur = ((cur + step + 3) // 4) * 4

        dedup = sorted({pos for pos in anchors if 8 <= pos < n_tok})
        if len(dedup) > max_structural:
            stride = len(dedup) / float(max_structural)
            dedup = sorted(
                {
                    dedup[min(len(dedup) - 1, int(round(i * stride)))]
                    for i in range(max_structural)
                }
            )
        return min_snap, dedup

    def resolve_active_checkpoints(
        self, past_seen: int, seq_len: int
    ) -> List[int]:
        if not self.record_checkpoints:
            return []
        total_len = past_seen + seq_len
        if self.requested_checkpoints is not None:
            min_req = max(past_seen + 1, self.min_checkpoint_token)
            valid = sorted(
                pos
                for pos in self.requested_checkpoints
                if min_req <= pos <= total_len
            )
            self.checkpoint_offsets = valid
            return valid

        min_tok = max(past_seen + 1, self.min_checkpoint_token, 8)
        max_tok = total_len - 1
        if min_tok > max_tok:
            return []

        span = total_len - min_tok
        candidates: Set[int] = set()
        if (
            self.min_checkpoint_token >= min_tok
            and self.min_checkpoint_token <= max_tok
        ):
            candidates.add(self.min_checkpoint_token)
        for pos in self.structural_checkpoints:
            if min_tok <= pos <= max_tok:
                candidates.add(pos)

        for frac in (0.25, 0.50, 0.65, 0.75, 0.82, 0.90):
            raw_pos = min_tok + int(round(span * frac))
            snapped_4 = (raw_pos // 4) * 4
            if min_tok <= snapped_4 <= max_tok:
                candidates.add(snapped_4)
            elif min_tok <= raw_pos <= max_tok:
                candidates.add(raw_pos)

        first_64 = ((min_tok + 63) // 64) * 64
        for pos_64 in range(max(64, first_64), max_tok + 1, 64):
            if min_tok <= pos_64 <= max_tok:
                candidates.add(pos_64)

        valid = sorted(candidates)
        self.checkpoint_offsets = valid
        return valid

    def extract_checkpoint(self, token_offset: int) -> Qwen3_5HybridCache:
        offset = int(token_offset)
        if offset == self.past_seen_tokens and offset not in self.recurrent_checkpoints:
            return self.clone_and_expand(1)
        if (
            offset not in self.conv_checkpoints
            or offset not in self.recurrent_checkpoints
        ):
            raise KeyError(
                f"Checkpoint offset {offset} not recorded in Qwen3_5HybridCache (available: {self.checkpoint_offsets})"
            )
        extracted = Qwen3_5HybridCache()
        extracted.past_seen_tokens = offset
        if self.rope_deltas is not None:
            extracted.rope_deltas = self.rope_deltas.clone()
        for k, v in self.conv_checkpoints[offset].items():
            extracted.conv_states[k] = v.contiguous().clone()
        for k, v in self.recurrent_checkpoints[offset].items():
            extracted.recurrent_states[k] = v.contiguous().clone()
        for k, v in self.key_cache.items():
            extracted.key_cache[k] = v[:, :, :offset, :].contiguous().clone()
        for k, v in self.value_cache.items():
            extracted.value_cache[k] = v[:, :, :offset, :].contiguous().clone()
        return extracted

    def clone_and_expand(
        self,
        batch_repeats: int = 1,
        record_checkpoints: bool = False,
        checkpoint_positions: Optional[Sequence[int]] = None,
    ) -> Qwen3_5HybridCache:
        cloned = Qwen3_5HybridCache(
            record_checkpoints=record_checkpoints,
            checkpoint_positions=checkpoint_positions,
        )
        cloned.past_seen_tokens = int(self.past_seen_tokens)
        if self.rope_deltas is not None:
            cloned.rope_deltas = (
                self.rope_deltas.repeat(batch_repeats, 1).contiguous()
                if batch_repeats > 1
                else self.rope_deltas.clone()
            )
        for k, v in self.conv_states.items():
            cloned.conv_states[k] = (
                v.repeat(batch_repeats, 1, 1).contiguous()
                if batch_repeats > 1
                else v.clone()
            )
        for k, v in self.recurrent_states.items():
            cloned.recurrent_states[k] = (
                v.repeat(batch_repeats, 1, 1, 1).contiguous()
                if batch_repeats > 1
                else v.clone()
            )
        for k, v in self.key_cache.items():
            cloned.key_cache[k] = (
                v.repeat(batch_repeats, 1, 1, 1).contiguous()
                if batch_repeats > 1
                else v.clone()
            )
        for k, v in self.value_cache.items():
            cloned.value_cache[k] = (
                v.repeat(batch_repeats, 1, 1, 1).contiguous()
                if batch_repeats > 1
                else v.clone()
            )
        return cloned

    def select_batch_item(
        self, batch_idx: int, total_valid_tokens: Optional[int] = None
    ) -> Qwen3_5HybridCache:
        selected = Qwen3_5HybridCache()
        target_tokens = (
            int(total_valid_tokens)
            if total_valid_tokens is not None
            else int(self.past_seen_tokens)
        )
        selected.past_seen_tokens = target_tokens
        if self.rope_deltas is not None:
            idx_r = min(batch_idx, self.rope_deltas.shape[0] - 1)
            selected.rope_deltas = self.rope_deltas[idx_r : idx_r + 1].clone()
        conv_src = (
            self.conv_checkpoints[target_tokens]
            if (
                total_valid_tokens is not None
                and target_tokens in self.conv_checkpoints
            )
            else self.conv_states
        )
        rec_src = (
            self.recurrent_checkpoints[target_tokens]
            if (
                total_valid_tokens is not None
                and target_tokens in self.recurrent_checkpoints
            )
            else self.recurrent_states
        )
        for k, v in conv_src.items():
            selected.conv_states[k] = v[batch_idx : batch_idx + 1].contiguous().clone()
        for k, v in rec_src.items():
            selected.recurrent_states[k] = (
                v[batch_idx : batch_idx + 1].contiguous().clone()
            )
        for k, v in self.key_cache.items():
            kv_slice = v[batch_idx : batch_idx + 1]
            if total_valid_tokens is not None:
                kv_slice = kv_slice[:, :, :target_tokens, :]
            selected.key_cache[k] = kv_slice.contiguous().clone()
        for k, v in self.value_cache.items():
            vv_slice = v[batch_idx : batch_idx + 1]
            if total_valid_tokens is not None:
                vv_slice = vv_slice[:, :, :target_tokens, :]
            selected.value_cache[k] = vv_slice.contiguous().clone()
        return selected


__all__ = ["Qwen3_5HybridCache"]