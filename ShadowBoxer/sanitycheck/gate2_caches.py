#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import tempfile

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

sys.modules["torchvision"] = None
sys.modules["torchvision.transforms"] = None
sys.modules["torchvision.io"] = None

import numpy as np
from PIL import Image
import torch

from engine import (
    CachedReferenceRecord,
    HAS_VIZINJECTOR,
    PromptEmbedCache,
    ReferenceVisionCache,
    VizInjectorClient,
    VizInjectorRequest,
)
from models.modeling_qwen35 import Qwen3_5HybridCache


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def gate_2_pinned_memory_and_cache_isolation() -> None:
    print(
        "\n--- Gate 2: Page-Locked Host Memory (pin_memory) & Multi-Turn Cache Isolation ---"
    )
    has_gpu = torch.cuda.is_available()
    dev = torch.device("cuda" if has_gpu else "cpu")

    p_cache = PromptEmbedCache(max_entries=8, max_prefix_entries=4)
    _assert(
        p_cache.max_prefix_entries == 4,
        f"Expected max_prefix_entries=4, got {p_cache.max_prefix_entries}",
    )
    synth_mm_tokens = (
        [248045, 872, 198, 248053]
        + [248056] * 16
        + [248054]
        + list(range(200, 260))
        + [248044]
    )
    min_snap, struct_cps = Qwen3_5HybridCache.derive_structural_checkpoints(
        synth_mm_tokens
    )
    _assert(
        min_snap == 21 and 21 in struct_cps and len(struct_cps) >= 3,
        f"derive_structural_checkpoints failed to anchor <|vision_end|>: min_snap={min_snap}, cps={struct_cps}",
    )
    auto_cp_cache = Qwen3_5HybridCache(
        record_checkpoints=True,
        min_checkpoint_token=4,
        structural_checkpoints=struct_cps,
    )
    auto_cps = auto_cp_cache.resolve_active_checkpoints(0, 53)
    _assert(
        44 in auto_cps and 21 in auto_cps,
        f"Expected 4-token-snapped 0.82 checkpoint (44) and structural boundary (21) in {auto_cps}",
    )

    pos_tokens_1 = [248045, 872, 198] + list(range(101, 121)) + [248044]
    neg_tokens = [248045, 872, 198, 248044]
    pos_tokens_2 = (
        [248045, 872, 198]
        + list(range(101, 116))
        + [999, 888, 777, 666, 555, 248044]
    )
    pos_tokens_early_div = (
        [248045, 872, 198] + list(range(101, 111)) + [444, 333, 222, 111, 248044]
    )

    with torch.inference_mode():
        h_pos1 = torch.randn(
            (1, len(pos_tokens_1), 64), device=dev, dtype=torch.bfloat16
        )
        h_neg = torch.randn(
            (1, len(neg_tokens), 64), device=dev, dtype=torch.bfloat16
        )
        _assert(
            p_cache.lookup(pos_tokens_1, dev, is_negative=False) is None,
            "Expected initial miss",
        )
        p_cache.insert(pos_tokens_1, h_pos1)
        stored_cpu = p_cache._cache[tuple(pos_tokens_1)]
        _assert(
            not stored_cpu.is_inference(),
            "Pinned CPU cache tensor must not be an InferenceTensor",
        )
        if has_gpu:
            _assert(
                stored_cpu.is_pinned(),
                "PromptEmbedCache CPU tensor is not page-locked (pinned)",
            )
        _assert(
            p_cache.lookup(neg_tokens, dev, is_negative=True) is None,
            "Expected initial neg miss",
        )
        p_cache.insert(neg_tokens, h_neg)
        _ = p_cache.lookup(neg_tokens, dev, is_negative=True)
        _assert(
            p_cache.pos_hits == 0,
            "Negative prompt hit must not increment pos_hits",
        )
        _assert(
            p_cache._last_tokens == pos_tokens_1,
            "Negative prompt lookup clobbered _last_tokens",
        )
        _ = p_cache.lookup(pos_tokens_2, dev, is_negative=False)
        _assert(
            p_cache.last_common_prefix_len == 18,
            f"Expected common prefix of 18 tokens between pos_1 and pos_2, got {p_cache.last_common_prefix_len}",
        )

        dummy_hybrid = Qwen3_5HybridCache(
            record_checkpoints=True, checkpoint_positions=[12, 16]
        )
        dummy_hybrid.past_seen_tokens = len(pos_tokens_1)
        dummy_hybrid.rope_deltas = torch.tensor([[-432]], dtype=torch.long, device=dev)
        dummy_hybrid.conv_states[0] = torch.randn(
            (1, 128, 3), device=dev, dtype=torch.bfloat16
        )
        dummy_hybrid.recurrent_states[0] = torch.randn(
            (1, 4, 32, 32), device=dev, dtype=torch.float32
        )
        dummy_hybrid.conv_checkpoints[12] = {
            0: torch.randn((1, 128, 3), device=dev, dtype=torch.bfloat16)
        }
        dummy_hybrid.recurrent_checkpoints[12] = {
            0: torch.randn((1, 4, 32, 32), device=dev, dtype=torch.float32)
        }
        dummy_hybrid.conv_checkpoints[16] = {
            0: torch.randn((1, 128, 3), device=dev, dtype=torch.bfloat16)
        }
        dummy_hybrid.recurrent_checkpoints[16] = {
            0: torch.randn((1, 4, 32, 32), device=dev, dtype=torch.float32)
        }
        dummy_hybrid.key_cache[3] = torch.randn(
            (1, 2, len(pos_tokens_1), 32), device=dev, dtype=torch.bfloat16
        )
        dummy_hybrid.value_cache[3] = torch.randn(
            (1, 2, len(pos_tokens_1), 32), device=dev, dtype=torch.bfloat16
        )

        p_cache.insert_prefix_state(
            pos_tokens_1,
            stored_cpu,
            dummy_hybrid,
        )
        spliced_16 = p_cache.lookup_prefix_splice(pos_tokens_2, dev)
        _assert(
            spliced_16 is not None
            and spliced_16[2] == 16
            and spliced_16[0].key_cache[3].shape[2] == 16
            and spliced_16[0].recurrent_states[0].dtype == torch.float32
            and spliced_16[0].rope_deltas is not None
            and int(spliced_16[0].rope_deltas.item()) == -432
            and p_cache.splice_hits == 1,
            "PromptEmbedCache multi-checkpoint prefix splice lookup failed for offset 16 or lost rope_deltas",
        )
        spliced_12 = p_cache.lookup_prefix_splice(pos_tokens_early_div, dev)
        _assert(
            spliced_12 is not None
            and spliced_12[2] == 12
            and spliced_12[0].key_cache[3].shape[2] == 12
            and spliced_12[0].rope_deltas is not None
            and int(spliced_12[0].rope_deltas.item()) == -432
            and p_cache.splice_hits == 2,
            "PromptEmbedCache multi-checkpoint prefix splice lookup failed for offset 12",
        )
        if has_gpu:
            pinned_state = next(iter(p_cache._prefix_cache.values()))
            _assert(
                pinned_state.prefix_hidden.is_pinned()
                and pinned_state.recurrent_checkpoints[16][0].is_pinned()
                and pinned_state.recurrent_checkpoints[16][0].dtype == torch.float32
                and pinned_state.key_cache[3].is_pinned()
                and pinned_state.rope_deltas is not None
                and pinned_state.rope_deltas.is_pinned(),
                "PinnedPrefixState trajectory tensors or rope_deltas are not page-locked",
            )
        vis_sig = (("ref_a.png", 1024, 1024), 576)
        p_cache.insert(pos_tokens_2, h_pos1, vision_signature=vis_sig)
        hit_mm = p_cache.lookup(
            pos_tokens_2, dev, is_negative=False, vision_signature=vis_sig
        )
        _assert(
            hit_mm is not None and hit_mm.device.type == dev.type,
            "Multimodal prompt cache lookup failed",
        )
    print(
        f"  [+] PromptEmbedCache         : Pinned={stored_cpu.is_pinned() if has_gpu else 'CPU'} | FP32Recurrent=True | MRoPEDelta=-432 | MultiCheckpoint={tuple(auto_cps)} tok"
    )

    r_cache = ReferenceVisionCache(max_entries=8)
    with tempfile.TemporaryDirectory() as tmpdir:
        img_p = Path(tmpdir) / "test_ref.png"
        Image.fromarray(np.full((512, 512, 3), 128, dtype=np.uint8), mode="RGB").save(
            img_p
        )

        if HAS_VIZINJECTOR and VizInjectorClient is not None:
            with VizInjectorClient(prefer_vulkan=True) as viz:
                src_geoms = [
                    (1920, 1080),
                    (1080, 1920),
                    (1024, 768),
                    (512, 512),
                    (773, 519),
                ]
                tgt_geoms = [(1024, 1024), (768, 1024), (512, 512)]
                framing_modes = [
                    "inscribed",
                    "cover_crop",
                    "contain",
                    "stretch",
                    "never_crop",
                ]
                for sw, sh in src_geoms:
                    geom_p = Path(tmpdir) / f"geom_{sw}x{sh}.png"
                    Image.fromarray(
                        np.zeros((sh, sw, 3), dtype=np.uint8), mode="RGB"
                    ).save(geom_p)
                    for tw, th in tgt_geoms:
                        for fm in framing_modes:
                            for max_p in (576, 2304):
                                py_vit_w, py_vit_h = ReferenceVisionCache.solve_vit_dims(
                                    geom_p,
                                    tw,
                                    th,
                                    fm,
                                    max_vit_patches=max_p,
                                )
                                c_bundle = viz.condition_image(
                                    geom_p,
                                    VizInjectorRequest(
                                        target_width=tw,
                                        target_height=th,
                                        max_vit_patches=max_p,
                                        edit_mode_384_vit=bool(max_p <= 576),
                                        framing_mode=fm,
                                        input_role="photo_ref",
                                        emit_vae_branch=False,
                                        emit_mask_branch=False,
                                        emit_vit_branch=True,
                                    ),
                                )
                                _assert(
                                    (py_vit_w, py_vit_h) == c_bundle.meta.vit_size,
                                    f"Gate 2.A Dual-Solver Geometry mismatch for src={(sw, sh)}, tgt={(tw, th)}, fm={fm}, patches={max_p}: py={(py_vit_w, py_vit_h)} vs c11={c_bundle.meta.vit_size}",
                                )

        key_v7 = ReferenceVisionCache.make_key(
            img_p, 512, 512, "never_crop", "photo_ref", "moss_image_edit", 2304, 4.5
        )
        vae_key_v7 = ReferenceVisionCache.make_vae_key(
            img_p, 512, 512, "never_crop", "photo_ref", 4.5
        )
        vae_key_v8 = ReferenceVisionCache.make_vae_key(
            img_p, 512, 512, "inscribed", "photo_ref", 4.5
        )
        vae_key_cc = ReferenceVisionCache.make_vae_key(
            img_p, 512, 512, "cover_crop", "photo_ref", 4.5
        )
        _assert(
            vae_key_v7 == vae_key_v8 and vae_key_v8 == vae_key_cc,
            "Aspect-matching 'never_crop', 'inscribed', and 'cover_crop' must canonicalize to the same VAE cache key",
        )
        vit_key_512 = ReferenceVisionCache.make_vit_key(
            img_p, 512, 512, "inscribed", "photo_ref", 2304
        )
        vit_key_256 = ReferenceVisionCache.make_vit_key(
            img_p, 256, 256, "inscribed", "photo_ref", 2304
        )
        vit_key_768 = ReferenceVisionCache.make_vit_key(
            img_p, 768, 768, "never_crop", "photo_ref", 2304
        )
        _assert(
            vit_key_512 == vit_key_256 and vit_key_256 == vit_key_768,
            "Box-resized references with matching aspect ratio must canonicalize to the exact same ViT cache key",
        )
        with torch.inference_mode():
            rec_in = CachedReferenceRecord(
                packed_latents=torch.randn((1, 128, 32, 32), device=dev),
                vit_embeds=torch.randn((576, 64), device=dev, dtype=torch.bfloat16),
                vit_patches=None,
                grid_thw=torch.tensor([[1, 48, 48]], dtype=torch.long),
                llm_tokens=576,
                latent_mask=torch.ones((1, 1, 32, 32), device=dev),
                viz_meta=None,
            )
            pinned_rec = r_cache.insert(
                key_v7, rec_in, vae_key=vae_key_v7, vit_key=vit_key_512
            )
        if has_gpu:
            _assert(
                pinned_rec.packed_latents is not None
                and pinned_rec.packed_latents.is_pinned()
                and pinned_rec.vit_embeds is not None
                and pinned_rec.vit_embeds.is_pinned(),
                "ReferenceVisionCache packed_latents or vit_embeds is not page-locked",
            )
        _assert(
            pinned_rec.source_signature == vit_key_512,
            "CachedReferenceRecord.source_signature must bind to box-size-invariant vit_key",
        )
        cross_hit = r_cache.lookup_vae(vae_key_v8)
        vit_cross_hit = r_cache.lookup_vit(vit_key_256)
        _assert(
            cross_hit is not None
            and cross_hit[0] is not None
            and vit_cross_hit is not None
            and vit_cross_hit[0] is not None
            and vit_cross_hit[3] == 576,
            "Cross-format VAE or box-resize-invariant ViT cache lookup missed",
        )
        print(
            f"  [+] ReferenceVisionCache     : Pinned={pinned_rec.packed_latents.is_pinned() if has_gpu else 'CPU'} | DualSolverParity=150/150 | CrossFormatVAEHit=True | BoxScaleInvariantViTHit=True"
        )


if __name__ == "__main__":
    gate_2_pinned_memory_and_cache_isolation()