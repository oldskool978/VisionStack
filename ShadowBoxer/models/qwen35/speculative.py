# ShadowBoxer/models/qwen35/speculative.py
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from .cache import Qwen3_5HybridCache
from .config import _INV_LN2, _shannon_entropy_bits

if TYPE_CHECKING:
    from .text import Qwen3_5MTPHead, Qwen3_5TextModel


def _missing_suffix_tokens(
    generated: Sequence[int], close_seq: Sequence[int]
) -> List[int]:
    c_list = [int(x) for x in close_seq]
    c_len = len(c_list)
    if c_len == 0:
        return []
    g_list = [int(x) for x in generated]
    g_len = len(g_list)
    if g_len >= c_len and g_list[-c_len:] == c_list:
        return []
    max_k = min(g_len, c_len - 1)
    for k in range(max_k, 0, -1):
        if g_list[-k:] == c_list[:k]:
            return c_list[k:]
    return c_list


def _levenshtein_distance(seq_a: Sequence[int], seq_b: Sequence[int]) -> int:
    la, lb = len(seq_a), len(seq_b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr[0] = i
        ai = seq_a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == seq_b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[lb]


@dataclass
class ReAnchorVerifierState:
    offset_d: int
    conditioning_token: int
    anchor_token: int
    hidden_state: torch.Tensor
    entropy_bits: float
    lookahead_tokens: Tuple[int, ...]


def _attach_shifted_continuation(
    base_chain: Sequence[int],
    shifted_chain: Sequence[int],
    block_len: int,
    conditioning_token: Optional[int] = None,
) -> Optional[List[int]]:
    if len(base_chain) < 2 or len(shifted_chain) == 0 or block_len <= 1:
        return None
    s_first = int(shifted_chain[0])
    for t_star in range(1, min(len(base_chain), block_len)):
        if int(base_chain[t_star]) == s_first:
            rem = block_len - t_star
            if len(shifted_chain) >= rem:
                attached = [int(x) for x in base_chain[:t_star]] + [
                    int(x) for x in shifted_chain[:rem]
                ]
                if attached != list(base_chain[:block_len]):
                    return attached
            break
    if conditioning_token is not None:
        c_tok = int(conditioning_token)
        for t_cond in range(1, min(len(base_chain), block_len - 1)):
            if int(base_chain[t_cond]) == c_tok:
                t_attach = t_cond + 1
                rem = block_len - t_attach
                if len(shifted_chain) >= rem:
                    attached = [int(x) for x in base_chain[:t_attach]] + [
                        int(x) for x in shifted_chain[:rem]
                    ]
                    if attached != list(base_chain[:block_len]):
                        return attached
                break
    return None


class _PromptSuffixDrafter:
    def __init__(self, history_tokens: Sequence[int], max_order: int = 4) -> None:
        self.max_order = max(1, int(max_order))
        self.table: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}
        self.transition_counts: Dict[Tuple[int, ...], Dict[int, int]] = {}
        self.total_counts: Dict[Tuple[int, ...], int] = {}
        self.recency_tick: int = 0
        self.last_seen_tick: Dict[Tuple[Tuple[int, ...], int], int] = {}
        self.add_tokens(history_tokens)

    def add_tokens(self, tokens: Sequence[int], block_len: int = 4) -> None:
        tok_list = [int(t) for t in tokens]
        n = len(tok_list)
        if n < 2:
            return
        for order in range(1, self.max_order + 1):
            if n <= order:
                continue
            for i in range(n - order):
                ctx = tuple(tok_list[i : i + order])
                nxt = tok_list[i + order]
                self.recency_tick += 1
                t_map = self.transition_counts.setdefault(ctx, {})
                t_map[nxt] = t_map.get(nxt, 0) + 1
                self.total_counts[ctx] = self.total_counts.get(ctx, 0) + 1
                self.last_seen_tick[(ctx, nxt)] = self.recency_tick
                if i + order + block_len <= n:
                    cont = tuple(tok_list[i + order : i + order + block_len])
                    bucket = self.table.setdefault(ctx, [])
                    if cont in bucket:
                        bucket.remove(cont)
                    bucket.append(cont)
                    if len(bucket) > 8:
                        bucket.pop(0)

    def context_entropy_bits(self, ctx: Tuple[int, ...]) -> float:
        t_map = self.transition_counts.get(ctx)
        tot = self.total_counts.get(ctx, 0)
        if not t_map or tot <= 1:
            return 0.0
        inv_tot = 1.0 / float(tot)
        entropy = 0.0
        for cnt in t_map.values():
            if cnt > 0:
                p = float(cnt) * inv_tot
                entropy -= p * math.log(p) * _INV_LN2
        return max(0.0, entropy)

    def _step_confidence(
        self, context_seq: Sequence[int], next_tok: int
    ) -> Tuple[float, int]:
        max_k = min(self.max_order, len(context_seq))
        for order in range(max_k, 0, -1):
            ctx = tuple(int(x) for x in context_seq[-order:])
            t_map = self.transition_counts.get(ctx)
            if t_map and next_tok in t_map:
                tot = self.total_counts.get(ctx, 1)
                prob = float(t_map[next_tok]) / float(max(1, tot))
                ent = self.context_entropy_bits(ctx)
                support = len(t_map)
                norm_ent = (
                    ent / max(1.0, math.log2(float(support)))
                    if support > 1
                    else 0.0
                )
                order_factor = (float(order) / float(self.max_order)) ** 0.25
                conf = max(
                    1e-4,
                    min(1.0, prob * (2.0 ** (-norm_ent)) * order_factor),
                )
                return conf, self.last_seen_tick.get((ctx, next_tok), 0)
        return 1e-3, 0

    def score_chain_prefix_survival(
        self, context_tokens: Sequence[int], chain: Sequence[int]
    ) -> Tuple[float, int]:
        running = [int(x) for x in context_tokens[-self.max_order :]]
        log_survival = 0.0
        cum_log_survival = 0.0
        recency_sum = 0
        for tok in chain:
            t_val = int(tok)
            conf, tick = self._step_confidence(running, t_val)
            log_survival += math.log(max(conf, 1e-6))
            cum_log_survival += log_survival
            recency_sum += tick
            running.append(t_val)
            if len(running) > self.max_order:
                running.pop(0)
        return cum_log_survival, recency_sum

    def _best_next_token(self, context_seq: Sequence[int]) -> Optional[int]:
        max_k = min(self.max_order, len(context_seq))
        for order in range(max_k, 0, -1):
            ctx = tuple(int(x) for x in context_seq[-order:])
            t_map = self.transition_counts.get(ctx)
            if t_map:
                best_tok = max(
                    t_map.keys(),
                    key=lambda t: (
                        t_map[t],
                        self.last_seen_tick.get((ctx, t), 0),
                    ),
                )
                return int(best_tok)
        return None

    def extend_chain(
        self,
        context_tokens: Sequence[int],
        seed_chain: Sequence[int],
        block_len: int,
    ) -> List[int]:
        chain = [int(x) for x in seed_chain[:block_len]]
        if not chain:
            return []
        running = [int(x) for x in context_tokens[-self.max_order :]] + list(chain)
        while len(chain) < block_len:
            nxt = self._best_next_token(running)
            if nxt is None:
                nxt = chain[-1]
            chain.append(int(nxt))
            running.append(int(nxt))
        return chain

    def propose_candidates(
        self,
        context_tokens: Sequence[int],
        block_len: int = 4,
        max_candidates: int = 4,
        first_token_override: Optional[int] = None,
    ) -> List[List[int]]:
        if block_len <= 0 or max_candidates <= 0:
            return []
        ctx_list = [int(x) for x in context_tokens]
        eval_ctx = (
            ctx_list + [int(first_token_override)]
            if first_token_override is not None
            else ctx_list
        )
        raw_pool: List[List[int]] = []

        for order in range(min(self.max_order, len(eval_ctx)), 0, -1):
            key = tuple(eval_ctx[-order:])
            matches = self.table.get(key)
            if matches:
                for m in reversed(matches):
                    seq = (
                        [int(first_token_override)] + list(m[: max(0, block_len - 1)])
                        if first_token_override is not None
                        else list(m[:block_len])
                    )
                    if len(seq) < block_len:
                        seq = self.extend_chain(ctx_list, seq, block_len)
                    if len(seq) == block_len and seq not in raw_pool:
                        raw_pool.append(seq)

            t_map = self.transition_counts.get(key)
            if t_map:
                ranked_next = sorted(
                    t_map.keys(),
                    key=lambda t: (
                        t_map[t],
                        self.last_seen_tick.get((key, t), 0),
                    ),
                    reverse=True,
                )
                for nxt_tok in ranked_next[: max_candidates * 2]:
                    seed = (
                        [int(first_token_override), int(nxt_tok)]
                        if first_token_override is not None
                        else [int(nxt_tok)]
                    )
                    seq = self.extend_chain(ctx_list, seed, block_len)
                    if len(seq) == block_len and seq not in raw_pool:
                        raw_pool.append(seq)

        if not raw_pool:
            return []

        scored: List[Tuple[float, int, List[int]]] = []
        for cand in raw_pool:
            surv_score, rec_tick = self.score_chain_prefix_survival(ctx_list, cand)
            scored.append((surv_score, rec_tick, cand))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored[:max_candidates]]


class Qwen3_5SpeculativeMixin:
    config: Any
    language_model: Qwen3_5TextModel
    mtp: Optional[Qwen3_5MTPHead]

    def get_input_embeddings(self) -> nn.Embedding:
        raise NotImplementedError

    def project_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @torch.inference_mode()
    def _propose_mtp_candidates(
        self,
        last_hidden: torch.Tensor,
        first_token: int,
        block_size: int,
        max_candidates: int,
        past_seen_tokens: int,
        rope_deltas: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        if self.mtp is None or block_size <= 1:
            return [[int(first_token)]] if block_size == 1 else []
        device = last_hidden.device
        embed_layer = self.get_input_embeddings()
        rotary_emb = self.language_model.rotary_emb

        delta_val = (
            int(rope_deltas.reshape(-1)[0].item())
            if (
                rope_deltas is not None
                and past_seen_tokens > 0
                and rope_deltas.numel() > 0
            )
            else 0
        )

        tok_1 = torch.tensor([[int(first_token)]], dtype=torch.long, device=device)
        emb_1 = embed_layer(tok_1)
        pos_1 = torch.tensor(
            [[[max(0, past_seen_tokens + delta_val)]]],
            dtype=torch.long,
            device=device,
        ).expand(3, 1, 1)

        h_mtp_1 = self.mtp(
            last_hidden,
            emb_1,
            rotary_emb=rotary_emb,
            position_ids=pos_1,
        )
        logits_1 = self.project_logits(h_mtp_1[:, -1, :]).float()
        k_top = min(max(1, int(max_candidates)), int(logits_1.shape[-1]))
        top_tokens = [
            int(t) for t in torch.topk(logits_1, k=k_top, dim=-1).indices[0].tolist()
        ]
        if block_size == 2:
            return [[int(first_token), t2] for t2 in top_tokens]

        cur_tokens = torch.tensor(
            top_tokens, dtype=torch.long, device=device
        ).unsqueeze(1)
        cur_h = (
            h_mtp_1.expand(k_top, -1, -1).contiguous() if k_top > 1 else h_mtp_1
        )
        draft_cols: List[List[int]] = [
            [int(first_token)] * k_top,
            top_tokens,
        ]
        for step in range(2, block_size):
            emb_s = embed_layer(cur_tokens)
            pos_s = torch.tensor(
                [[[max(0, past_seen_tokens + step - 1 + delta_val)]]],
                dtype=torch.long,
                device=device,
            ).expand(3, k_top, 1)
            cur_h = self.mtp(
                cur_h,
                emb_s,
                rotary_emb=rotary_emb,
                position_ids=pos_s,
            )
            logits_s = self.project_logits(cur_h[:, -1, :]).float()
            cur_tokens = torch.argmax(logits_s, dim=-1, keepdim=True)
            draft_cols.append([int(t) for t in cur_tokens.squeeze(1).tolist()])

        chains: List[List[int]] = []
        for idx in range(k_top):
            chain = [draft_cols[col][idx] for col in range(block_size)]
            if chain not in chains:
                chains.append(chain)
        return chains

    @torch.inference_mode()
    def _propose_batched_shifted_mtp(
        self,
        base_hidden: torch.Tensor,
        base_token: int,
        retained_anchors: Sequence[ReAnchorVerifierState],
        block_size: int,
        max_candidates: int,
        past_seen_tokens: int,
        rope_deltas: Optional[torch.Tensor] = None,
    ) -> Tuple[List[List[int]], List[Tuple[ReAnchorVerifierState, List[int]]]]:
        if self.mtp is None or block_size <= 1:
            base_c = [[int(base_token)]] if block_size == 1 else []
            return base_c, []
        if not retained_anchors:
            return (
                self._propose_mtp_candidates(
                    last_hidden=base_hidden,
                    first_token=base_token,
                    block_size=block_size,
                    max_candidates=max_candidates,
                    past_seen_tokens=past_seen_tokens,
                    rope_deltas=rope_deltas,
                ),
                [],
            )

        device = base_hidden.device
        embed_layer = self.get_input_embeddings()
        rotary_emb = self.language_model.rotary_emb
        delta_val = (
            int(rope_deltas.reshape(-1)[0].item())
            if (
                rope_deltas is not None
                and past_seen_tokens > 0
                and rope_deltas.numel() > 0
            )
            else 0
        )

        num_shifted = len(retained_anchors)
        batch_anchors = 1 + num_shifted
        h_cat = torch.cat(
            [base_hidden] + [a.hidden_state.to(device=device) for a in retained_anchors],
            dim=0,
        )
        tok_1_list = [int(base_token)] + [int(a.anchor_token) for a in retained_anchors]
        offsets_d = [0] + [int(a.offset_d) for a in retained_anchors]

        tok_1 = torch.tensor(tok_1_list, dtype=torch.long, device=device).unsqueeze(1)
        emb_1 = embed_layer(tok_1)
        pos_1_vals = [
            max(0, past_seen_tokens + d_off + delta_val) for d_off in offsets_d
        ]
        pos_1 = (
            torch.tensor(pos_1_vals, dtype=torch.long, device=device)
            .view(1, batch_anchors, 1)
            .expand(3, batch_anchors, 1)
        )

        h_mtp_1 = self.mtp(
            h_cat,
            emb_1,
            rotary_emb=rotary_emb,
            position_ids=pos_1,
        )
        logits_1 = self.project_logits(h_mtp_1[:, -1, :]).float()

        k_top = min(max(1, int(max_candidates)), int(logits_1.shape[-1]))
        base_top_tokens = [
            int(t)
            for t in torch.topk(logits_1[0:1], k=k_top, dim=-1).indices[0].tolist()
        ]
        shifted_step1_tokens = [
            int(t)
            for t in torch.argmax(logits_1[1:], dim=-1).tolist()
        ]

        if block_size == 2:
            base_chains = [[int(base_token), t2] for t2 in base_top_tokens]
            shifted_chains = [
                (
                    retained_anchors[idx],
                    [int(retained_anchors[idx].anchor_token), shifted_step1_tokens[idx]],
                )
                for idx in range(num_shifted)
            ]
            return base_chains, shifted_chains

        total_branches = k_top + num_shifted
        branch_offsets = [0] * k_top + [int(a.offset_d) for a in retained_anchors]
        cur_tokens = torch.tensor(
            base_top_tokens + shifted_step1_tokens,
            dtype=torch.long,
            device=device,
        ).unsqueeze(1)
        base_h_exp = (
            h_mtp_1[0:1].expand(k_top, -1, -1).contiguous()
            if k_top > 1
            else h_mtp_1[0:1]
        )
        cur_h = torch.cat([base_h_exp, h_mtp_1[1:]], dim=0)

        draft_cols: List[List[int]] = [
            [int(base_token)] * k_top
            + [int(a.anchor_token) for a in retained_anchors],
            base_top_tokens + shifted_step1_tokens,
        ]

        for step in range(2, block_size):
            emb_s = embed_layer(cur_tokens)
            pos_s_vals = [
                max(0, past_seen_tokens + d_off + step - 1 + delta_val)
                for d_off in branch_offsets
            ]
            pos_s = (
                torch.tensor(pos_s_vals, dtype=torch.long, device=device)
                .view(1, total_branches, 1)
                .expand(3, total_branches, 1)
            )
            cur_h = self.mtp(
                cur_h,
                emb_s,
                rotary_emb=rotary_emb,
                position_ids=pos_s,
            )
            logits_s = self.project_logits(cur_h[:, -1, :]).float()
            cur_tokens = torch.argmax(logits_s, dim=-1, keepdim=True)
            draft_cols.append([int(t) for t in cur_tokens.squeeze(1).tolist()])

        base_chains = []
        for idx in range(k_top):
            chain = [draft_cols[col][idx] for col in range(block_size)]
            if chain not in base_chains:
                base_chains.append(chain)

        shifted_chains = []
        for s_idx in range(num_shifted):
            row_idx = k_top + s_idx
            s_chain = [draft_cols[col][row_idx] for col in range(block_size)]
            shifted_chains.append((retained_anchors[s_idx], s_chain))

        return base_chains, shifted_chains

    @torch.inference_mode()
    def reanchor_speculative_generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 128,
        block_size: int = 4,
        max_candidates: int = 4,
        entropy_threshold: float = 1.25,
        stop_token_ids: Optional[Sequence[int]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        draft_proposer: Optional[
            Callable[[List[int], int, int, Optional[int]], List[List[int]]]
        ] = None,
        close_token_ids: Optional[Sequence[int]] = None,
        return_cache: bool = False,
    ) -> Union[
        Tuple[List[int], torch.Tensor],
        Tuple[List[int], torch.Tensor, Optional[Qwen3_5HybridCache]],
    ]:
        device = next(self.parameters()).device  # type: ignore[attr-defined]
        stop_set = set(int(x) for x in (stop_token_ids or [248044, 248046, 248069]))
        close_seq = [int(x) for x in close_token_ids] if close_token_ids else []
        has_close_seq = len(close_seq) > 0
        prompt_tokens = [int(x) for x in input_ids[0].tolist()]
        drafter = _PromptSuffixDrafter(prompt_tokens)
        eff_block_size = max(1, int(block_size))
        eff_max_cands = max(1, int(max_candidates))

        prefill_out = self.forward(
            input_ids=input_ids.to(device=device),
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=True,
        )
        cache: Optional[Qwen3_5HybridCache] = prefill_out.past_key_values
        hidden_chunks: List[torch.Tensor] = [prefill_out.last_hidden_state]
        generated: List[int] = []

        last_hidden = prefill_out.last_hidden_state[:, -1:, :]
        logits = self.project_logits(last_hidden[:, 0, :]).float()
        entropy = _shannon_entropy_bits(logits)
        next_tok = int(torch.argmax(logits, dim=-1).item())

        retained_anchors: List[ReAnchorVerifierState] = []
        total_rounds = 0
        total_accepted_tokens = 0
        reanchor_shift_candidates = 0
        reanchor_shift_wins = 0

        while len(generated) < max_new_tokens:
            if next_tok in stop_set:
                if not has_close_seq:
                    generated.append(next_tok)
                    step_ids = torch.tensor(
                        [[next_tok]], dtype=torch.long, device=device
                    )
                    step_out = self.forward(
                        input_ids=step_ids, past_key_values=cache, use_cache=True
                    )
                    cache = step_out.past_key_values
                    hidden_chunks.append(step_out.last_hidden_state)
                break

            generated.append(next_tok)
            if has_close_seq and len(generated) >= len(close_seq):
                if generated[-len(close_seq) :] == close_seq:
                    step_ids = torch.tensor(
                        [[next_tok]], dtype=torch.long, device=device
                    )
                    step_out = self.forward(
                        input_ids=step_ids, past_key_values=cache, use_cache=True
                    )
                    cache = step_out.past_key_values
                    hidden_chunks.append(step_out.last_hidden_state)
                    break

            if len(generated) >= max_new_tokens:
                step_ids = torch.tensor([[next_tok]], dtype=torch.long, device=device)
                step_out = self.forward(
                    input_ids=step_ids, past_key_values=cache, use_cache=True
                )
                cache = step_out.past_key_values
                hidden_chunks.append(step_out.last_hidden_state)
                break

            full_history = prompt_tokens + generated[:-1]
            high_entropy = entropy > entropy_threshold
            cand_limit = eff_max_cands if high_entropy else 1

            candidate_chains: List[List[int]] = []
            reanchor_chain_indices: set[int] = set()
            shifted_blocks: List[Tuple[ReAnchorVerifierState, List[int]]] = []

            if draft_proposer is not None:
                g0_list = draft_proposer(full_history, eff_block_size, 1, None)
                for anchor in retained_anchors:
                    s_cands = draft_proposer(
                        full_history + [next_tok],
                        eff_block_size,
                        1,
                        anchor.anchor_token,
                    )
                    s_blk = (
                        s_cands[0]
                        if s_cands
                        else drafter.extend_chain(
                            full_history + [next_tok],
                            list(anchor.lookahead_tokens),
                            eff_block_size,
                        )
                    )
                    if s_blk:
                        shifted_blocks.append((anchor, s_blk))
            elif self.mtp is not None:
                mtp_cands, shifted_blocks = self._propose_batched_shifted_mtp(
                    base_hidden=last_hidden,
                    base_token=next_tok,
                    retained_anchors=retained_anchors if cand_limit > 1 else (),
                    block_size=eff_block_size,
                    max_candidates=cand_limit,
                    past_seen_tokens=cache.past_seen_tokens if cache is not None else 0,
                    rope_deltas=cache.rope_deltas if cache is not None else None,
                )
                g0_list = mtp_cands[:1]
                for mc in mtp_cands:
                    if len(mc) == eff_block_size and mc not in candidate_chains:
                        candidate_chains.append(mc)
                        if len(candidate_chains) >= cand_limit:
                            break
            else:
                g0_list = drafter.propose_candidates(
                    full_history,
                    block_len=eff_block_size,
                    max_candidates=1,
                    first_token_override=next_tok,
                )
                for anchor in retained_anchors:
                    s_blk = drafter.extend_chain(
                        full_history + [next_tok],
                        list(anchor.lookahead_tokens),
                        eff_block_size,
                    )
                    if s_blk:
                        shifted_blocks.append((anchor, s_blk))

            g0 = (
                g0_list[0]
                if (g0_list and len(g0_list[0]) == eff_block_size and g0_list[0][0] == next_tok)
                else drafter.extend_chain(full_history, [next_tok], eff_block_size)
            )
            if len(g0) < eff_block_size:
                g0 = [next_tok] * eff_block_size

            if g0[0] == next_tok and g0 not in candidate_chains:
                candidate_chains.insert(0, g0)
                if len(candidate_chains) > cand_limit:
                    candidate_chains = candidate_chains[:cand_limit]

            if shifted_blocks and cand_limit > 1:
                reanchor_attached: List[Tuple[float, int, List[int]]] = []
                for anchor, s_chain in shifted_blocks:
                    attached = _attach_shifted_continuation(
                        g0,
                        s_chain,
                        eff_block_size,
                        conditioning_token=anchor.conditioning_token,
                    )
                    if (
                        attached is not None
                        and len(attached) == eff_block_size
                        and attached[0] == next_tok
                        and attached not in candidate_chains
                    ):
                        surv, _ = drafter.score_chain_prefix_survival(
                            full_history, attached
                        )
                        reanchor_attached.append(
                            (surv - 0.15 * anchor.entropy_bits, -anchor.offset_d, attached)
                        )
                reanchor_attached.sort(key=lambda item: (item[0], item[1]), reverse=True)
                for _, _, attached_chain in reanchor_attached:
                    if attached_chain not in candidate_chains:
                        if len(candidate_chains) < cand_limit:
                            reanchor_chain_indices.add(len(candidate_chains))
                            candidate_chains.append(attached_chain)
                            reanchor_shift_candidates += 1
                        elif len(candidate_chains) >= 2:
                            replace_idx = len(candidate_chains) - 1
                            if replace_idx not in reanchor_chain_indices:
                                candidate_chains[replace_idx] = attached_chain
                                reanchor_chain_indices.add(replace_idx)
                                reanchor_shift_candidates += 1
                                break

            if len(candidate_chains) < cand_limit:
                if draft_proposer is not None:
                    gd_list = draft_proposer(
                        full_history, eff_block_size, cand_limit, next_tok
                    )
                else:
                    gd_list = drafter.propose_candidates(
                        full_history,
                        block_len=eff_block_size,
                        max_candidates=cand_limit * 2,
                        first_token_override=next_tok,
                    )
                scored_markov: List[Tuple[int, float, int, List[int]]] = []
                for cand in gd_list:
                    if (
                        len(cand) == eff_block_size
                        and cand[0] == next_tok
                        and cand not in candidate_chains
                    ):
                        dist = _levenshtein_distance(g0[1:], cand[1:])
                        surv, rec_t = drafter.score_chain_prefix_survival(
                            full_history, cand
                        )
                        scored_markov.append((dist, -surv, -rec_t, cand))
                scored_markov.sort(key=lambda item: (item[0], item[1], item[2]))
                for _, _, _, cand in scored_markov:
                    if len(candidate_chains) < cand_limit:
                        candidate_chains.append(cand)

            if not candidate_chains:
                retained_anchors.clear()
                step_ids = torch.tensor([[next_tok]], dtype=torch.long, device=device)
                step_out = self.forward(
                    input_ids=step_ids, past_key_values=cache, use_cache=True
                )
                cache = step_out.past_key_values
                hidden_chunks.append(step_out.last_hidden_state)
                last_hidden = step_out.last_hidden_state[:, -1:, :]
                logits = self.project_logits(last_hidden[:, 0, :]).float()
                entropy = _shannon_entropy_bits(logits)
                next_tok = int(torch.argmax(logits, dim=-1).item())
                drafter.add_tokens(
                    (prompt_tokens + generated)[-16:], block_len=eff_block_size
                )
                total_rounds += 1
                total_accepted_tokens += 1
                continue

            num_k = len(candidate_chains)
            b_len = len(candidate_chains[0])
            cand_tensor = torch.tensor(
                candidate_chains, dtype=torch.long, device=device
            )

            prev_seen = cache.past_seen_tokens if cache is not None else 0
            step_cps = list(range(prev_seen + 1, prev_seen + b_len + 1))
            batched_cache = (
                cache.clone_and_expand(
                    num_k,
                    record_checkpoints=True,
                    checkpoint_positions=step_cps,
                )
                if cache is not None
                else Qwen3_5HybridCache(
                    record_checkpoints=True,
                    checkpoint_positions=step_cps,
                )
            )
            verify_out = self.forward(
                input_ids=cand_tensor,
                past_key_values=batched_cache,
                use_cache=True,
            )
            verify_logits = self.project_logits(verify_out.last_hidden_state).float()
            target_preds = torch.argmax(verify_logits, dim=-1)

            best_k = 0
            best_acc = 1
            for k_idx in range(num_k):
                acc = 1
                for pos in range(b_len - 1):
                    pred_t = int(target_preds[k_idx, pos].item())
                    if pred_t in stop_set:
                        break
                    if pred_t == candidate_chains[k_idx][pos + 1]:
                        acc += 1
                    else:
                        break
                if acc > best_acc:
                    best_acc = acc
                    best_k = k_idx

            if best_k in reanchor_chain_indices and best_acc > 1:
                reanchor_shift_wins += 1

            raw_best_acc = best_acc
            rem_budget = max_new_tokens - len(generated) + 1
            best_acc = min(best_acc, rem_budget)
            winning_chain = candidate_chains[best_k]

            if has_close_seq:
                c_len = len(close_seq)
                for pos in range(1, best_acc):
                    cand_prefix = generated + winning_chain[1 : pos + 1]
                    if len(cand_prefix) >= c_len and cand_prefix[-c_len:] == close_seq:
                        best_acc = pos + 1
                        break

            retained_anchors.clear()
            if raw_best_acc < b_len:
                win_preds = [
                    int(target_preds[best_k, p].item()) for p in range(b_len)
                ]
                for pos in range(raw_best_acc, b_len):
                    pred_d = win_preds[pos]
                    if pred_d in stop_set:
                        continue
                    d_offset = pos - raw_best_acc + 1
                    h_d = verify_out.last_hidden_state[
                        best_k : best_k + 1, pos : pos + 1, :
                    ].detach()
                    ent_d = _shannon_entropy_bits(
                        verify_logits[best_k : best_k + 1, pos, :]
                    )
                    lookahead = tuple(
                        t for t in win_preds[pos:b_len] if t not in stop_set
                    )
                    if lookahead:
                        retained_anchors.append(
                            ReAnchorVerifierState(
                                offset_d=d_offset,
                                conditioning_token=int(winning_chain[pos]),
                                anchor_token=pred_d,
                                hidden_state=h_d,
                                entropy_bits=ent_d,
                                lookahead_tokens=lookahead,
                            )
                        )
                        drafter.add_tokens(
                            [int(winning_chain[pos])] + list(lookahead),
                            block_len=eff_block_size,
                        )

            for pos in range(1, best_acc):
                generated.append(winning_chain[pos])

            total_rounds += 1
            total_accepted_tokens += best_acc

            accepted_hidden = verify_out.last_hidden_state[
                best_k : best_k + 1, :best_acc, :
            ].contiguous()
            hidden_chunks.append(accepted_hidden)
            last_hidden = accepted_hidden[:, -1:, :]

            if verify_out.past_key_values is not None:
                cache = verify_out.past_key_values.select_batch_item(
                    best_k, total_valid_tokens=prev_seen + best_acc
                )

            if has_close_seq and len(generated) >= len(close_seq):
                if generated[-len(close_seq) :] == close_seq:
                    break

            final_pos = best_acc - 1
            final_logits = verify_logits[best_k : best_k + 1, final_pos, :]
            entropy = _shannon_entropy_bits(final_logits)
            next_tok = int(target_preds[best_k, final_pos].item())
            drafter.add_tokens(
                (prompt_tokens + generated + [next_tok])[-20:],
                block_len=eff_block_size,
            )

            if next_tok in stop_set or any(
                t in stop_set for t in generated[-best_acc:]
            ):
                if (
                    not has_close_seq
                    and next_tok in stop_set
                    and len(generated) < max_new_tokens
                ):
                    generated.append(next_tok)
                    step_ids = torch.tensor(
                        [[next_tok]], dtype=torch.long, device=device
                    )
                    step_out = self.forward(
                        input_ids=step_ids, past_key_values=cache, use_cache=True
                    )
                    cache = step_out.past_key_values
                    hidden_chunks.append(step_out.last_hidden_state)
                break

        if has_close_seq:
            missing_close = _missing_suffix_tokens(generated, close_seq)
            if missing_close:
                generated.extend(missing_close)
                close_ids = torch.tensor(
                    [missing_close], dtype=torch.long, device=device
                )
                close_out = self.forward(
                    input_ids=close_ids, past_key_values=cache, use_cache=True
                )
                cache = close_out.past_key_values
                hidden_chunks.append(close_out.last_hidden_state)

        self.last_reanchor_telemetry = {
            "total_rounds": total_rounds,
            "total_accepted_tokens": total_accepted_tokens,
            "mean_acceptance_length": (
                float(total_accepted_tokens) / float(max(1, total_rounds))
            ),
            "reanchor_shift_candidates": reanchor_shift_candidates,
            "reanchor_shift_wins": reanchor_shift_wins,
        }

        full_hidden = torch.cat(hidden_chunks, dim=1)
        if return_cache:
            return generated, full_hidden, cache
        return generated, full_hidden


__all__ = [
    "_missing_suffix_tokens",
    "_levenshtein_distance",
    "ReAnchorVerifierState",
    "_attach_shifted_continuation",
    "_PromptSuffixDrafter",
    "Qwen3_5SpeculativeMixin",
]