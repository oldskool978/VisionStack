#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

sys.modules["torchvision"] = None
sys.modules["torchvision.transforms"] = None
sys.modules["torchvision.io"] = None

from safetensors.torch import save_file as save_safetensors
import torch

from engine import (
    _MetaWeightInitMode,
    _resolve_target_tensor_dtype,
    load_sharded_safetensors_into,
    stream_safetensors_inplace,
    subphase_radial_clamp,
)
from models.modeling_moss import (
    MOSSImageConfig,
    MOSSImageTransformer,
    _attention,
    _rotate,
)
from models.modeling_qwen35 import (
    Qwen3_5Config,
    Qwen3_5HybridCache,
    Qwen3_5MTPHead,
    Qwen3_5Model,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
    ReAnchorVerifierState,
    _PromptSuffixDrafter,
    _STRICTLY_UPPER_MASK_CACHE,
    _attach_shifted_continuation,
    _causal_conv1d_fn,
    _chunk_gated_delta_rule,
    _recurrent_gated_delta_rule,
    _shannon_entropy_bits,
    _solve_unit_lower_triangular,
)
from models.modeling_vae import (
    Flux2AutoencoderKL,
    Flux2VAEConfig,
    _conv2d_3x3_strip_tiled,
)
import models.qwen35 as qwen35_pkg


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def gate_3_models_zero_alloc_and_lora() -> None:
    print(
        "\n--- Gate 3: Sovereign Architectures, Meta-Init Streaming, Caches, LoRA & FP8 ---"
    )
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32

    _assert(
        hasattr(qwen35_pkg, "Qwen3_5Model")
        and qwen35_pkg.Qwen3_5Model is Qwen3_5Model,
        "models.qwen35 subpackage failed to expose canonical Qwen3_5Model",
    )
    _assert(
        _resolve_target_tensor_dtype(
            "language_model.layers.0.linear_attn.dt_bias",
            torch.ones(4, dtype=torch.float32),
            torch.bfloat16,
        )
        == torch.float32
        and _resolve_target_tensor_dtype(
            "language_model.layers.0.linear_attn.A_log",
            torch.ones(4, dtype=torch.float32),
            torch.bfloat16,
        )
        == torch.float32
        and _resolve_target_tensor_dtype(
            "rope.bands",
            torch.ones(4, dtype=torch.bfloat16),
            torch.bfloat16,
        )
        == torch.float32,
        "_resolve_target_tensor_dtype failed to preserve SSM/RoPE parameters in FP32",
    )

    cfg = MOSSImageConfig(
        dim=256,
        ffn_dim=512,
        num_heads=8,
        num_layers=4,
        text_dim=256,
        freq_dim=64,
        latent_channels=128,
    )
    dit = MOSSImageTransformer(config=cfg).to(device=dev, dtype=dtype).eval()

    with _MetaWeightInitMode():
        meta_dit = MOSSImageTransformer(config=cfg)
    _assert(
        next(meta_dit.parameters()).is_meta,
        "_MetaWeightInitMode failed to place parameters on meta device",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_sf = Path(tmpdir) / "diffusion_pytorch_model.safetensors"
        save_safetensors(
            {k: v.detach().cpu() for k, v in dit.state_dict().items()},
            str(tmp_sf),
        )
        load_sharded_safetensors_into(Path(tmpdir), meta_dit, device=dev, dtype=dtype)
        meta_dit.eval()
        _assert(
            not next(meta_dit.parameters()).is_meta,
            "load_sharded_safetensors_into left parameters on meta device",
        )
        ptrs_before = {k: p.data_ptr() for k, p in meta_dit.named_parameters()}
        with torch.no_grad():
            meta_dit.layers[0].proj["q"].weight.zero_()
        inplace_ok = stream_safetensors_inplace(
            Path(tmpdir), meta_dit, device=dev, dtype=dtype
        )
        ptrs_after = {k: p.data_ptr() for k, p in meta_dit.named_parameters()}
        _assert(
            inplace_ok and ptrs_before == ptrs_after,
            "stream_safetensors_inplace reallocated parameter storage instead of streaming in-place",
        )

    with torch.inference_mode():
        _, head_f32 = dit._compute_time_modulation(500.0, 1, dev)
        _assert(
            head_f32.dtype == torch.float32,
            f"Expected FP32 video_head_modulation output, got {head_f32.dtype}",
        )
        sched_ts = torch.tensor([750.0, 500.0, 250.0], device=dev, dtype=torch.float32)
        mod_s1, head_s1 = dit._compute_time_modulation(sched_ts[0:1], 1, dev)
        cache_len_before = len(dit._time_mod_cache)
        mod_s2, head_s2 = dit._compute_time_modulation(sched_ts[0:1], 1, dev)
        _assert(
            mod_s1.data_ptr() == mod_s2.data_ptr()
            and head_s1.data_ptr() == head_s2.data_ptr()
            and len(dit._time_mod_cache) == cache_len_before,
            "View-slice device tensor timestep failed to hit _time_mod_cache",
        )

        lat = torch.randn((1, 128, 8, 8), device=dev, dtype=torch.float32)
        lat2 = torch.randn((1, 128, 8, 8), device=dev, dtype=torch.float32)
        ref1 = torch.randn((1, 128, 8, 8), device=dev, dtype=torch.float32)
        ref2 = torch.randn((1, 128, 6, 6), device=dev, dtype=torch.float32)
        txt = torch.randn((1, 16, 256), device=dev, dtype=dtype)
        mask = torch.ones((1, 16), device=dev, dtype=torch.bool)
        _assert(
            txt.is_inference() and mask.is_inference(),
            "Test tensors must be InferenceTensors",
        )

        rope_3d = dit.rope(8, 8, 16, dev, dtype=dtype, ref_shapes=[(8, 8), (6, 6)])
        (rope_4d_full, text_rope_from_exp) = dit.rope.get_expanded_rope(
            8, 8, 16, dev, dtype=dtype, ref_shapes=[(8, 8), (6, 6)]
        )
        text_rope_direct = dit.rope.get_text_rope(16, dev, dtype=dtype)
        _assert(
            float((text_rope_direct[0] - text_rope_from_exp[0]).abs().max().item())
            == 0.0
            and float((text_rope_direct[1] - text_rope_from_exp[1]).abs().max().item())
            == 0.0,
            "_RoPE.get_text_rope diverged from get_expanded_rope text slice",
        )

        phase_l16 = dit.rope._compute_phase(
            8, 8, 16, dev, ref_shapes=[(8, 8), (6, 6)]
        )
        phase_l11 = dit.rope._compute_phase(
            8, 8, 11, dev, ref_shapes=[(8, 8), (6, 6)]
        )
        expected_f16 = (16.0 * dit.rope.bands).unsqueeze(0).expand(164, -1)
        expected_f11 = (11.0 * dit.rope.bands).unsqueeze(0).expand(164, -1)
        _assert(
            float((phase_l16[:164, 1, :] - expected_f16).abs().max().item()) == 0.0
            and float((phase_l11[:164, 1, :] - expected_f11).abs().max().item())
            == 0.0
            and float(phase_l16[164:, 0, :].abs().max().item()) == 0.0
            and float(phase_l16[164:, 2:, :].abs().max().item()) == 0.0,
            "Gate 3.C Two-Tier 4D-RoPE failed to couple image/reference f-coordinate to L_txt",
        )
        dummy_q = torch.randn((1, rope_3d.shape[1], 8, 32), device=dev, dtype=dtype)
        rot_3d = _rotate(dummy_q, rope_3d)
        rot_4d = _rotate(dummy_q, rope_4d_full)
        _assert(
            float((rot_3d - rot_4d).abs().max().item()) == 0.0,
            "Pre-expanded 4D RoPE diverged from 3D RoPE",
        )

        t_tok = 64
        r1_tok = 64
        r2_tok = 36
        iso_mask = dit._get_isolated_reference_mask(
            t_tok, [r1_tok, r2_tok], 16, dev
        )
        _assert(
            bool(iso_mask[0, 0, 0, t_tok + 10].item()) is True,
            "Target must attend to Ref 1",
        )
        _assert(
            bool(iso_mask[0, 0, t_tok + 5, 0].item()) is False,
            "Ref 1 must not attend to Target",
        )
        _assert(
            bool(iso_mask[0, 0, t_tok + 5, t_tok + r1_tok + 5].item()) is False,
            "Disjoint Ref 1 must not attend to Ref 2",
        )
        _assert(
            bool(iso_mask[0, 0, t_tok + 5, t_tok + r1_tok + r2_tok + 2].item())
            is True,
            "Ref 1 must attend to Text",
        )

        q_test = torch.randn(
            (1, rope_3d.shape[1], 8, 32), device=dev, dtype=torch.float32
        )
        k_test = torch.randn(
            (1, rope_3d.shape[1], 8, 32), device=dev, dtype=torch.float32
        )
        v_test = torch.randn(
            (1, rope_3d.shape[1], 8, 32), device=dev, dtype=torch.float32
        )
        (rope_4d_fp32, _) = dit.rope.get_expanded_rope(
            8, 8, 16, dev, dtype=torch.float32, ref_shapes=[(8, 8), (6, 6)]
        )
        attn_masked = _attention(
            q_test, k_test, v_test, rope_4d_fp32, 8, attn_mask=iso_mask
        )
        attn_partitioned = _attention(
            q_test,
            k_test,
            v_test,
            rope_4d_fp32,
            8,
            attn_mask=None,
            iso_kv_spec=(t_tok, (r1_tok, r2_tok), 16, ((0,), (1,))),
        )
        iso_diff = float((attn_masked - attn_partitioned).abs().max().item())
        _assert(
            iso_diff < 1e-5,
            f"Block-partitioned FlashAttention IsoKV diverged from boolean mask: {iso_diff}",
        )

        overlap_boxes = [(0.125, 0.125, 0.75, 0.75), (0.25, 0.25, 0.875, 0.875)]
        collab_mask = dit._get_isolated_reference_mask(
            t_tok, [r1_tok, r2_tok], 16, dev, ref_bboxes=overlap_boxes
        )
        _assert(
            bool(collab_mask[0, 0, t_tok + 5, t_tok + r1_tok + 5].item()) is True,
            "Overlapping reference boxes must collaborate in IsoKV attention",
        )
        attn_collab_masked = _attention(
            q_test, k_test, v_test, rope_4d_fp32, 8, attn_mask=collab_mask
        )
        attn_collab_part = _attention(
            q_test,
            k_test,
            v_test,
            rope_4d_fp32,
            8,
            attn_mask=None,
            iso_kv_spec=(t_tok, (r1_tok, r2_tok), 16, ((0, 1), (0, 1))),
        )
        collab_diff = float(
            (attn_collab_masked - attn_collab_part).abs().max().item()
        )
        _assert(
            collab_diff < 1e-5,
            f"Collaborative block-partitioned IsoKV diverged from mask: {collab_diff}",
        )

        bias_none = dit._get_box_aperture_bias(
            height=8,
            width=8,
            ref_token_lengths=[r1_tok, r2_tok],
            text_tokens=16,
            ref_shapes=[(8, 8), (6, 6)],
            ref_bboxes=[(0.5, 0.5, 1.0, 1.0), None],
            reference_box_isolation=0.0,
            feather_radius_lat=1.0,
            device=dev,
            dtype=torch.float32,
        )
        _assert(
            bias_none is None,
            "Zero box_attn_isolation must return None without allocating a bias tensor",
        )
        bias_active = dit._get_box_aperture_bias(
            height=8,
            width=8,
            ref_token_lengths=[r1_tok, r2_tok],
            text_tokens=16,
            ref_shapes=[(8, 8), (6, 6)],
            ref_bboxes=[(0.5, 0.5, 1.0, 1.0), None],
            reference_box_isolation=[0.85, 0.0],
            feather_radius_lat=1.5,
            device=dev,
            dtype=torch.float32,
        )
        _assert(
            bias_active is not None and bias_active.shape == (1, 1, 64, 180),
            f"Unexpected box aperture bias shape: {None if bias_active is None else bias_active.shape}",
        )
        interior_idx = 6 * 8 + 6
        exterior_idx = 0 * 8 + 0
        feather_idx = 3 * 8 + 6
        _assert(
            float(bias_active[0, 0, interior_idx, t_tok + 5].item()) == 0.0,
            "Interior target token inside Ref 1 bbox must have 0.0 attention bias",
        )
        _assert(
            abs(float(bias_active[0, 0, exterior_idx, t_tok + 5].item()) - (-17.0))
            < 1e-4,
            f"Exterior target token outside Ref 1 bbox expected -17.0 bias, got {bias_active[0, 0, exterior_idx, t_tok + 5].item()}",
        )
        _assert(
            -17.0 < float(bias_active[0, 0, feather_idx, t_tok + 5].item()) < 0.0,
            "Feather boundary token must have smooth intermediate attention bias",
        )
        _assert(
            float(bias_active[0, 0, :, t_tok + r1_tok :].abs().max().item()) == 0.0,
            "Unisolated Ref 2 and Text tokens must remain completely unattenuated (0.0)",
        )
        bias_kronecker = dit._get_box_aperture_bias(
            height=8,
            width=8,
            ref_token_lengths=[r1_tok, r2_tok],
            text_tokens=16,
            ref_shapes=[(8, 8), (6, 6)],
            ref_bboxes=[(0.5, 0.5, 1.0, 1.0), None],
            reference_box_isolation=[0.85, 0.0],
            feather_radius_lat=0.0,
            device=dev,
            dtype=torch.float32,
        )
        _assert(
            bias_kronecker is not None
            and float(bias_kronecker[0, 0, interior_idx, t_tok + 5].item()) == 0.0
            and abs(float(bias_kronecker[0, 0, feather_idx, t_tok + 5].item()) - (-17.0))
            < 1e-4,
            "Kronecker 0.0-feather box aperture bias failed exact step transition at box boundary",
        )

        coloc_boxes = [(0.25, 0.25, 0.75, 0.75), (0.25, 0.25, 0.75, 0.75)]
        coloc_q_idx = 4 * 8 + 4
        bias_same_role = dit._get_box_aperture_bias(
            height=8,
            width=8,
            ref_token_lengths=[16, 16],
            text_tokens=16,
            ref_shapes=[(4, 4), (4, 4)],
            ref_bboxes=coloc_boxes,
            reference_box_isolation=[0.85, 0.85],
            feather_radius_lat=1.0,
            device=dev,
            dtype=torch.float32,
            reference_roles=["photo_ref", "photo_ref"],
            reference_z_orders=[1.0, 5.0],
        )
        bias_coop_roles = dit._get_box_aperture_bias(
            height=8,
            width=8,
            ref_token_lengths=[16, 16],
            text_tokens=16,
            ref_shapes=[(4, 4), (4, 4)],
            ref_bboxes=coloc_boxes,
            reference_box_isolation=[0.85, 0.85],
            feather_radius_lat=1.0,
            device=dev,
            dtype=torch.float32,
            reference_roles=["photo_ref", "depth_map"],
            reference_z_orders=[1.0, 5.0],
        )
        _assert(
            bias_same_role is not None
            and bias_coop_roles is not None
            and float(bias_coop_roles[0, 0, coloc_q_idx, t_tok + 2].item()) == 0.0
            and float(bias_coop_roles[0, 0, coloc_q_idx, t_tok + 16 + 2].item()) == 0.0
            and float(bias_same_role[0, 0, coloc_q_idx, t_tok + 16 + 2].item())
            > float(bias_same_role[0, 0, coloc_q_idx, t_tok + 2].item())
            and float(bias_same_role[0, 0, coloc_q_idx, t_tok + 2].item()) < -0.5,
            "Co-located multi-role cooperation or z_order hierarchy failed in _get_box_aperture_bias",
        )

        parent_boxes = [(0.125, 0.125, 0.875, 0.875), (0.125, 0.125, 0.875, 0.875)]
        carve_spec = [((0.25, 0.25, 0.625, 0.625), 0.90, 5.0)]
        bias_carved = dit._get_box_aperture_bias(
            height=8,
            width=8,
            ref_token_lengths=[36, 36],
            text_tokens=16,
            ref_shapes=[(6, 6), (6, 6)],
            ref_bboxes=parent_boxes,
            reference_box_isolation=[0.85, 0.85],
            feather_radius_lat=1.0,
            device=dev,
            dtype=torch.float32,
            reference_roles=["photo_ref", "depth_map"],
            reference_z_orders=[1.0, 2.0],
            foreground_carveouts=carve_spec,
        )
        carved_q_idx = 3 * 8 + 3
        uncarved_parent_q_idx = 6 * 8 + 6
        _assert(
            bias_carved is not None
            and float(bias_carved[0, 0, uncarved_parent_q_idx, t_tok + 2].item())
            == 0.0
            and float(bias_carved[0, 0, uncarved_parent_q_idx, t_tok + 36 + 2].item())
            == 0.0
            and float(bias_carved[0, 0, carved_q_idx, t_tok + 2].item()) < -15.0
            and float(bias_carved[0, 0, carved_q_idx, t_tok + 36 + 2].item()) < -15.0
            and float(bias_carved[0, 0, :, t_tok + 72 :].abs().max().item()) == 0.0,
            "Gate 3.D Foreground child-aperture carve-out failed to attenuate parent photo_ref/depth_map KV inside text box",
        )
        bias_global_carved = dit._get_box_aperture_bias(
            height=8,
            width=8,
            ref_token_lengths=[r1_tok],
            text_tokens=16,
            ref_shapes=[(8, 8)],
            ref_bboxes=[None],
            reference_box_isolation=0.0,
            feather_radius_lat=1.0,
            device=dev,
            dtype=torch.float32,
            reference_roles=["photo_ref"],
            reference_z_orders=[1.0],
            foreground_carveouts=carve_spec,
        )
        _assert(
            bias_global_carved is not None
            and float(bias_global_carved[0, 0, exterior_idx, t_tok + 2].item()) == 0.0
            and float(bias_global_carved[0, 0, carved_q_idx, t_tok + 2].item())
            < -15.0,
            "Gate 3.D Foreground carve-out failed on unboxed/global reference",
        )

        cutout_mask = torch.zeros((1, 1, 8, 8), device=dev, dtype=torch.float32)
        cutout_mask[:, :, 2:6, 2:6] = 1.0
        bias_silhouette = dit._get_box_aperture_bias(
            height=8,
            width=8,
            ref_token_lengths=[r1_tok, r2_tok],
            text_tokens=16,
            ref_shapes=[(8, 8), (6, 6)],
            ref_bboxes=[(0.5, 0.5, 1.0, 1.0), None],
            reference_box_isolation=[0.85, 0.0],
            feather_radius_lat=1.0,
            device=dev,
            dtype=torch.float32,
            reference_masks=[cutout_mask, None],
        )
        opaque_key_idx = t_tok + (4 * 8 + 4)
        transp_key_idx = t_tok + (0 * 8 + 0)
        box_center_q = 6 * 8 + 6
        box_corner_q = 4 * 8 + 4
        _assert(
            bias_silhouette is not None
            and float(bias_silhouette[0, 0, box_center_q, opaque_key_idx].item()) == 0.0
            and float(bias_silhouette[0, 0, box_corner_q, opaque_key_idx].item()) < -10.0
            and float(bias_silhouette[0, 0, box_center_q, transp_key_idx].item()) < -15.0,
            "Silhouette-conformal cutout reference_masks failed to isolate transparent query/key regions",
        )

        frac_mask = torch.zeros((1, 1, 4, 4), device=dev, dtype=torch.float32)
        frac_mask[0, 0, 0, 0] = 0.0
        frac_mask[0, 0, 0, 1] = 1e-6
        frac_mask[0, 0, 1, 1] = 0.25
        frac_mask[0, 0, 2, 2] = 0.50
        frac_mask[0, 0, 3, 3] = 1.0
        bias_frac = dit._get_box_aperture_bias(
            height=8,
            width=8,
            ref_token_lengths=[16],
            text_tokens=16,
            ref_shapes=[(4, 4)],
            ref_bboxes=[(0.25, 0.25, 0.75, 0.75)],
            reference_box_isolation=[0.85],
            feather_radius_lat=1.25,
            device=dev,
            dtype=torch.float32,
            reference_masks=[frac_mask],
            foreground_carveouts=[((0.5, 0.5, 0.75, 0.75), 0.90, 5.0)],
        )
        q_m0 = 2 * 8 + 2
        q_m1e6 = 2 * 8 + 3
        q_m025 = 3 * 8 + 3
        q_m050 = 4 * 8 + 4
        k_opaque_16 = t_tok + (3 * 4 + 3)
        k_half_16 = t_tok + (2 * 4 + 2)
        expected_half_q = -17.0 * (1.0 - math.sqrt(0.5))
        _assert(
            bias_frac is not None
            and bool(torch.isfinite(bias_frac).all().item())
            and float(bias_frac.min().item()) >= -18.0 - 1e-5
            and float(bias_frac.max().item()) <= 0.0
            and abs(float(bias_frac[0, 0, q_m0, k_opaque_16].item()) - (-17.0)) < 1e-4
            and -17.0 < float(bias_frac[0, 0, q_m1e6, k_opaque_16].item()) < -16.95
            and abs(float(bias_frac[0, 0, q_m025, k_opaque_16].item()) - (-8.5)) < 1e-4
            and abs(
                float(bias_frac[0, 0, q_m025, k_half_16].item())
                - (-8.5 + expected_half_q)
            )
            < 1e-4
            and float(bias_frac[0, 0, q_m050, k_opaque_16].item()) < -15.0,
            "Fractional cutout root-shaping or compounded foreground carve-out failed micro-floor continuity / bounds",
        )

        phase_boxed = dit.rope._compute_phase(
            8,
            8,
            16,
            dev,
            ref_shapes=[(4, 4)],
            ref_bboxes=[(0.49, 0.51, 0.99, 1.0)],
        )
        target_grid_phase = phase_boxed[:64, 2:4, :].view(8, 8, 2, -1)
        expected_box_phase = target_grid_phase[4:8, 4:8, :, :].reshape(16, 2, -1)
        ref_phase_slice = phase_boxed[64 : 64 + 16, 2:4, :]
        _assert(
            float((ref_phase_slice - expected_box_phase).abs().max().item()) == 0.0,
            "16px lattice-locked 4D-RoPE reference phase must match target canvas cell phase bit-for-bit",
        )

        t_step_1 = torch.tensor([500.0], device=dev)
        ref_boxes = [(0.125, 0.125, 0.875, 0.875), (0.25, 0.25, 1.0, 1.0)]
        out1 = dit(
            lat,
            t_step_1,
            txt,
            mask,
            reference_latents=[ref1, ref2],
            reference_zero_timestep=True,
            reference_isolated_kv=True,
            reference_bboxes=ref_boxes,
        ).sample
        out_meta = meta_dit(
            lat,
            t_step_1,
            txt,
            mask,
            reference_latents=[ref1, ref2],
            reference_zero_timestep=True,
            reference_isolated_kv=True,
            reference_bboxes=ref_boxes,
        ).sample
        _assert(
            float((out1 - out_meta).abs().max().item()) == 0.0,
            "Meta-initialized and in-place streamed DiT output diverged from standard DiT",
        )
        _assert(
            out1.shape == (1, 128, 8, 8), f"Unexpected DiT output shape: {out1.shape}"
        )

        out1_second = dit(
            lat2,
            t_step_1,
            txt,
            mask,
            reference_latents=[ref1, ref2],
            reference_zero_timestep=True,
            reference_isolated_kv=True,
            reference_bboxes=ref_boxes,
        ).sample
        out_b2 = dit(
            torch.cat([lat, lat2], dim=0),
            t_step_1,
            txt.repeat(2, 1, 1),
            mask.repeat(2, 1),
            reference_latents=[ref1, ref2],
            reference_zero_timestep=True,
            reference_isolated_kv=True,
            reference_bboxes=ref_boxes,
        ).sample
        b2_cat = torch.cat([out1, out1_second], dim=0)
        b2_diff = float((b2_cat - out_b2).abs().max().item())
        b2_mae = float((b2_cat - out_b2).abs().mean().item())
        _assert(
            b2_diff <= 0.03125 and b2_mae < 1e-3,
            f"Homogeneous B=2 vectorized DiT batch diverged from B=1 execution: max={b2_diff}, mae={b2_mae}",
        )

        mask_het = torch.ones((2, 16), device=dev, dtype=torch.bool)
        mask_het[1, 11:] = False
        out_het_1 = dit(
            lat2,
            t_step_1,
            txt,
            mask_het[1:2],
            reference_latents=[ref1, ref2],
            reference_zero_timestep=True,
            reference_isolated_kv=True,
            reference_bboxes=ref_boxes,
        ).sample
        out_het_b2 = dit(
            torch.cat([lat, lat2], dim=0),
            t_step_1,
            txt.repeat(2, 1, 1),
            mask_het,
            reference_latents=[ref1, ref2],
            reference_zero_timestep=True,
            reference_isolated_kv=True,
            reference_bboxes=ref_boxes,
        ).sample
        het_diff = float(
            (torch.cat([out1, out_het_1], dim=0) - out_het_b2).abs().max().item()
        )
        _assert(
            het_diff == 0.0,
            f"Gate 3.C Heterogeneous B=2 DiT batch diverged from individual B=1 passes: max_diff={het_diff}",
        )

        out_box_iso = dit(
            lat,
            t_step_1,
            txt,
            mask,
            reference_latents=[ref1, ref2],
            reference_zero_timestep=True,
            reference_isolated_kv=True,
            reference_bboxes=ref_boxes,
            reference_box_isolation=[0.90, 0.50],
            box_feather_radius=1.25,
            reference_masks=[cutout_mask, None],
            reference_roles=["photo_ref", "depth_map"],
            reference_z_orders=[1.0, 2.0],
        ).sample
        out_box_carved = dit(
            lat,
            t_step_1,
            txt,
            mask,
            reference_latents=[ref1, ref2],
            reference_zero_timestep=True,
            reference_isolated_kv=True,
            reference_bboxes=ref_boxes,
            reference_box_isolation=[0.90, 0.50],
            box_feather_radius=1.25,
            reference_masks=[cutout_mask, None],
            reference_roles=["photo_ref", "depth_map"],
            reference_z_orders=[1.0, 2.0],
            foreground_carveouts=carve_spec,
        ).sample
        out_box_iso_non_isokv = dit(
            lat,
            t_step_1,
            txt,
            mask,
            reference_latents=[ref1, ref2],
            reference_zero_timestep=True,
            reference_isolated_kv=False,
            reference_bboxes=ref_boxes,
            reference_box_isolation=0.85,
            box_feather_radius=1.25,
        ).sample
        _assert(
            torch.isfinite(out_box_iso).all()
            and torch.isfinite(out_box_carved).all()
            and torch.isfinite(out_box_iso_non_isokv).all()
            and float(out_box_carved.float().std().item()) > 0.05
            and float((out1 - out_box_iso).abs().max().item()) > 1e-5
            and float((out_box_iso - out_box_carved).abs().max().item()) > 1e-5,
            "Regional box-aperture isolation or foreground carve-out forward pass failed",
        )
        _assert(
            len(dit._box_aperture_bias_cache) >= 2,
            "Box aperture bias cache was not populated",
        )
        _assert(dit._mod_seq_buf is not None, "_mod_seq_buf was not allocated")
        buf_ptr_1 = dit._mod_seq_buf.data_ptr()
        _assert(
            len(dit._mod_zero_cache) == 1, "Zero-timestep modulation was not cached"
        )
        _assert(
            len(dit._text_proj_cache) >= 1,
            "Step-invariant _TextProjection was not cached",
        )
        _assert(
            len(dit._ref_proj_cache) == 2,
            "Reference latent projections were not cached",
        )

        txt_cache_len_before = len(dit._text_proj_cache)
        cached_txt_ptr = next(reversed(dit._text_proj_cache.values()))[2].data_ptr()
        txt_res_hit, _, _ = dit._get_or_compute_text_projection(
            encoder_hidden_states=txt,
            mask=mask,
            index=0,
            act_dtype=dtype,
            height=12,
            width=16,
            ref_shapes=None,
            ref_bboxes=None,
            need_full_rope=False,
        )
        _assert(
            txt_res_hit.data_ptr() == cached_txt_ptr
            and len(dit._text_proj_cache) == txt_cache_len_before,
            "Tier-1A _TextProjection cache must be invariant to canvas resolution and ref_bboxes",
        )

        _ = dit(
            lat,
            t_step_1,
            txt,
            mask,
            skip_layers={2},
            reference_latents=[ref1, ref2],
            reference_zero_timestep=True,
            reference_isolated_kv=True,
            reference_bboxes=ref_boxes,
        ).sample

        t_step_2 = torch.tensor([250.0], device=dev)
        _ = dit(
            lat,
            t_step_2,
            txt,
            mask,
            reference_latents=[ref1, ref2],
            reference_zero_timestep=True,
            reference_isolated_kv=True,
            reference_bboxes=ref_boxes,
        ).sample
        buf_ptr_2 = dit._mod_seq_buf.data_ptr()
        cached_txt_ptr_2 = next(reversed(dit._text_proj_cache.values()))[2].data_ptr()
        _assert(
            buf_ptr_1 == buf_ptr_2,
            "_mod_seq_buf was reallocated instead of reused in-place",
        )
        _assert(
            cached_txt_ptr == cached_txt_ptr_2,
            "_text_proj_cache entry was not reused across steps",
        )
        dit.clear_transient_caches()
        _assert(
            dit._mod_seq_buf is None,
            "clear_transient_caches must release _mod_seq_buf between turns",
        )
        dit.clear_caches()
        _assert(
            len(dit._box_aperture_bias_cache) == 0,
            "clear_caches must flush _box_aperture_bias_cache",
        )
    print(
        f"  [+] MOSSImageTransformer     : Meta-Init + InPlaceHotSwap + Tier1ATextRoPE + FgCarveOut4DRoPE + HetBatch (diff={het_diff:.1e}) + ModSeqBuf (0x{buf_ptr_2:X}) Verified"
    )

    orig_q_weight = dit.layers[0].proj["q"].weight.data.detach().float().clone()
    lora_sd = {
        "transformer.layers.0.proj.q.lora_A.weight": torch.randn(
            (4, 256), dtype=torch.float32
        )
        * 0.1,
        "transformer.layers.0.proj.q.lora_B.weight": torch.randn(
            (256, 4), dtype=torch.float32
        )
        * 0.1,
    }
    fused_cnt = dit.fuse_lora_state_dict(lora_sd, scale=1.0, lora_tag="sanity_lora")
    _assert(fused_cnt == 1, f"Expected 1 fused LoRA layer, got {fused_cnt}")
    _assert(
        dit.is_lora_active("sanity_lora", 1.0),
        "is_lora_active failed to report active LoRA",
    )
    _assert(
        len(dit._text_proj_cache) == 0 and len(dit._ref_proj_cache) == 0,
        "LoRA fuse must invalidate step caches",
    )
    backup_tensor = dit._lora_backups["layers.0.proj.q.weight"]
    if dev.type == "cuda":
        _assert(backup_tensor.is_pinned(), "LoRA weight backup is not page-locked")
    dit.unfuse_lora()
    restored_diff = float(
        (dit.layers[0].proj["q"].weight.data.float() - orig_q_weight)
        .abs()
        .max()
        .item()
    )
    _assert(restored_diff == 0.0, f"LoRA unfuse drift: {restored_diff}")
    print(
        f"  [+] Pin-Backed LoRA Engine   : Fused={fused_cnt} | BackupPinned={backup_tensor.is_pinned() if dev.type == 'cuda' else 'CPU'} | UnfuseMaxDiff={restored_diff:.1e}"
    )

    markov_drafter = _PromptSuffixDrafter(
        [10, 20, 30, 40, 50, 10, 20, 30, 40, 50, 10, 20, 31, 41, 51]
    )
    ent_ctx_20 = markov_drafter.context_entropy_bits((10, 20))
    _assert(
        0.90 <= ent_ctx_20 <= 0.93,
        f"Expected empirical Shannon entropy ~0.918 bits for 2:1 transition split, got {ent_ctx_20}",
    )
    markov_cands = markov_drafter.propose_candidates(
        [10], block_len=4, max_candidates=2, first_token_override=20
    )
    _assert(
        len(markov_cands) >= 2
        and markov_cands[0] == [20, 30, 40, 50]
        and markov_cands[1] == [20, 31, 41, 51],
        f"Shannon-entropy prefix-survival Markov drafter failed to rank high-probability continuation first: {markov_cands}",
    )
    attached_overlap = _attach_shifted_continuation(
        [100, 200, 300, 400], [200, 777, 888, 999], 4
    )
    attached_cond = _attach_shifted_continuation(
        [100, 200, 300, 400], [777, 888, 999], 4, conditioning_token=200
    )
    _assert(
        attached_overlap == [100, 200, 777, 888]
        and attached_cond == [100, 200, 777, 888],
        f"_attach_shifted_continuation failed ReAnchor overlap/conditioning attachment: overlap={attached_overlap}, cond={attached_cond}",
    )

    q_cfg = Qwen3_5Config(
        text_config=Qwen3_5TextConfig(
            vocab_size=1024,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
                "mrope_section": [2, 3, 3],
            },
        ),
        vision_config=Qwen3_5VisionConfig(
            depth=2,
            hidden_size=64,
            intermediate_size=128,
            num_heads=4,
            in_channels=3,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=128,
            num_position_embeddings=64,
        ),
        image_token_id=900,
        video_token_id=901,
    )
    qwen = Qwen3_5Model(q_cfg).to(device=dev, dtype=dtype).eval()
    with torch.inference_mode():
        g_thw_asym = torch.tensor([[1, 4, 6]], dtype=torch.long, device=dev)
        v_pos_ids_2d, v_pos_embeds, v_cu_seqlens = qwen.visual._compute_geometry(
            g_thw_asym, dev
        )
        _assert(
            v_cu_seqlens.tolist() == [0, 24]
            and v_pos_ids_2d.shape == (2, 24)
            and v_pos_ids_2d[0, :4].tolist() == [0, 0, 1, 1]
            and v_pos_ids_2d[1, :4].tolist() == [0, 1, 0, 1]
            and v_pos_ids_2d[0, 4:8].tolist() == [0, 0, 1, 1]
            and v_pos_ids_2d[1, 4:8].tolist() == [2, 3, 2, 3],
            f"Gate 3.A Vision 2D axial merge-block position_ids mismatch: {v_pos_ids_2d[:, :8].tolist()}",
        )
        side_v = qwen.visual.num_grid_per_side
        yc = torch.linspace(0, side_v - 1, 4, device=dev, dtype=torch.float32)
        xc = torch.linspace(0, side_v - 1, 6, device=dev, dtype=torch.float32)
        y0_v, x0_v = yc.floor().long(), xc.floor().long()
        y1_v = (y0_v + 1).clamp(max=side_v - 1)
        x1_v = (x0_v + 1).clamp(max=side_v - 1)
        wy_v = (yc - y0_v.float())[:, None]
        wx_v = (xc - x0_v.float())[None, :]
        pe_table = qwen.visual.pos_embed.weight.float().view(side_v, side_v, -1)
        ref_pe_2d = (
            pe_table[y0_v][:, x0_v] * ((1.0 - wy_v) * (1.0 - wx_v)).unsqueeze(-1)
            + pe_table[y0_v][:, x1_v] * ((1.0 - wy_v) * wx_v).unsqueeze(-1)
            + pe_table[y1_v][:, x0_v] * (wy_v * (1.0 - wx_v)).unsqueeze(-1)
            + pe_table[y1_v][:, x1_v] * (wy_v * wx_v).unsqueeze(-1)
        )
        ref_pe_merged = (
            ref_pe_2d.view(2, 2, 3, 2, -1)
            .permute(0, 2, 1, 3, 4)
            .reshape(24, -1)
        )
        pe_interp_err = float(
            (v_pos_embeds.float() - ref_pe_merged).abs().max().item()
        )
        v_cos, v_sin = qwen.visual.rotary_pos_emb(v_pos_embeds, v_pos_ids_2d)
        _assert(
            pe_interp_err < 1e-5
            and v_cos.shape == (24, 16)
            and float((v_cos[:, :8] - v_cos[:, 8:]).abs().max().item()) == 0.0
            and float((v_sin[:, :8] - v_sin[:, 8:]).abs().max().item()) == 0.0,
            f"Gate 3.A Vision bilinear pos_embed or 2D axial RoPE failed: pe_err={pe_interp_err}",
        )

        g_thw_multi = torch.tensor([[1, 4, 4], [1, 4, 6]], dtype=torch.long, device=dev)
        pix_multi = torch.randn((40, 3, 2, 16, 16), device=dev, dtype=dtype)
        feat_splits = qwen.get_image_features(pix_multi, g_thw_multi)
        _assert(
            len(feat_splits) == 2
            and feat_splits[0].shape == (4, 128)
            and feat_splits[1].shape == (6, 128),
            f"Multi-image cu_seqlens vision split failed: {[tuple(s.shape) for s in feat_splits]}",
        )

        ent_uniform = _shannon_entropy_bits(
            torch.zeros((1, 1024), device=dev, dtype=torch.float32)
        )
        peaked_logits = torch.full((1, 1024), -100.0, device=dev, dtype=torch.float32)
        peaked_logits[0, 0] = 100.0
        ent_dirac = _shannon_entropy_bits(peaked_logits)
        _assert(
            abs(ent_uniform - 10.0) < 1e-5 and abs(ent_dirac - 0.0) < 1e-6,
            f"_shannon_entropy_bits failed base-2 calibration: uniform={ent_uniform} (expected 10.0), dirac={ent_dirac} (expected 0.0)",
        )

        in_ids = torch.tensor(
            [[10, 20, 900, 900, 900, 900, 30, 40]], dtype=torch.long, device=dev
        )
        pix = torch.randn((16, 3, 2, 16, 16), device=dev, dtype=dtype)
        g_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=dev)

        vid_ids = torch.tensor(
            [[10, 901, 901, 901, 901, 25, 901, 901, 901, 901, 30]],
            dtype=torch.long,
            device=dev,
        )
        vid_pix = torch.randn((32, 3, 2, 16, 16), device=dev, dtype=dtype)
        v_thw = torch.tensor([[2, 4, 4]], dtype=torch.long, device=dev)
        v_pos_ids, v_deltas = qwen.get_rope_index(vid_ids, video_grid_thw=v_thw)
        _assert(
            v_pos_ids.shape == (3, 1, 11) and int(v_deltas.item()) == -4,
            f"Video M-RoPE index mismatch: shape={v_pos_ids.shape}, delta={v_deltas.item()}",
        )
        q_vid_out = qwen(
            input_ids=vid_ids,
            attention_mask=torch.ones_like(vid_ids),
            pixel_values_videos=vid_pix,
            video_grid_thw=v_thw,
        )
        _assert(
            q_vid_out.last_hidden_state.shape == (1, 11, 128)
            and q_vid_out.rope_deltas is not None
            and int(q_vid_out.rope_deltas.item()) == -4,
            f"Video forward pass failed: shape={q_vid_out.last_hidden_state.shape}, delta={q_vid_out.rope_deltas}",
        )

        conv_x = torch.randn((1, 64, 16), device=dev, dtype=torch.float32)
        conv_w = torch.randn((64, 4), device=dev, dtype=torch.float32)
        conv_full, conv_final_full = _causal_conv1d_fn(
            conv_x, conv_w, activation="silu", return_final_state=True
        )
        conv_pfx, conv_state_mid = _causal_conv1d_fn(
            conv_x[:, :, :10], conv_w, activation="silu", return_final_state=True
        )
        conv_sfx, conv_final_sfx = _causal_conv1d_fn(
            conv_x[:, :, 10:],
            conv_w,
            activation="silu",
            conv_state=conv_state_mid,
            return_final_state=True,
        )
        conv_diff = max(
            float((conv_full - torch.cat([conv_pfx, conv_sfx], dim=-1)).abs().max().item()),
            float((conv_final_full - conv_final_sfx).abs().max().item()),
        )
        _assert(
            conv_diff < 1e-6,
            f"Causal Conv1d state continuation drift: {conv_diff}",
        )

        ut = (
            torch.randn((1, 2, 1, 64, 64), device=dev, dtype=torch.float32).tril(-1)
            * 0.05
        )
        ut = ut + torch.eye(64, device=dev, dtype=torch.float32)
        rhs_a = torch.randn((1, 2, 1, 64, 32), device=dev, dtype=torch.float32)
        rhs_b = torch.randn((1, 2, 1, 64, 32), device=dev, dtype=torch.float32)
        sol_a = _solve_unit_lower_triangular(ut, rhs_a)
        sol_b = _solve_unit_lower_triangular(ut, rhs_b)
        sol_cat_a, sol_cat_b = torch.split(
            _solve_unit_lower_triangular(ut, torch.cat([rhs_a, rhs_b], dim=-1)),
            [32, 32],
            dim=-1,
        )
        tri_diff = max(
            float((sol_a - sol_cat_a).abs().max().item()),
            float((sol_b - sol_cat_b).abs().max().item()),
        )
        res_err = float(((ut @ sol_a) - rhs_a).abs().max().item())
        _assert(
            tri_diff < 1e-5 and res_err < 1e-4,
            f"Block-recursive triangular solver drift: cat_diff={tri_diff}, res_err={res_err}",
        )

        q_delta = torch.randn((1, 20, 4, 32), device=dev, dtype=torch.float32)
        k_delta = torch.randn((1, 20, 4, 32), device=dev, dtype=torch.float32)
        v_delta = torch.randn((1, 20, 4, 32), device=dev, dtype=torch.float32)
        g_delta = -torch.rand((1, 20, 4), device=dev, dtype=torch.float32) * 0.25
        b_delta = torch.sigmoid(torch.randn((1, 20, 4), device=dev, dtype=torch.float32))
        cp_states_single: dict[int, torch.Tensor] = {}
        full_delta = _chunk_gated_delta_rule(
            q_delta,
            k_delta,
            v_delta,
            g_delta,
            b_delta,
            output_final_state=False,
            checkpoint_positions=[12, 16],
            checkpoint_states_out=cp_states_single,
        )
        _, state_12_ref = _recurrent_gated_delta_rule(
            q_delta[:, :12],
            k_delta[:, :12],
            v_delta[:, :12],
            g_delta[:, :12],
            b_delta[:, :12],
            output_final_state=True,
        )
        pfx_delta, state_16 = _chunk_gated_delta_rule(
            q_delta[:, :16],
            k_delta[:, :16],
            v_delta[:, :16],
            g_delta[:, :16],
            b_delta[:, :16],
            output_final_state=True,
        )
        sfx_delta, _ = _recurrent_gated_delta_rule(
            q_delta[:, 16:],
            k_delta[:, 16:],
            v_delta[:, 16:],
            g_delta[:, 16:],
            b_delta[:, 16:],
            initial_state=cp_states_single[16],
            output_final_state=True,
        )
        cp_diff = max(
            float((cp_states_single[12] - state_12_ref).abs().max().item()),
            float((cp_states_single[16] - state_16).abs().max().item()),
        )
        rec_diff = float(
            (full_delta - torch.cat([pfx_delta, sfx_delta], dim=1)).abs().max().item()
        )
        _assert(
            cp_diff < 1e-5 and rec_diff < 1e-5,
            f"Single-pass trajectory checkpoint or step-recurrent GatedDeltaNet drift: cp_diff={cp_diff}, rec_diff={rec_diff}",
        )

        seq_80 = torch.randint(10, 500, (1, 80), dtype=torch.long, device=dev)
        traj_cache = Qwen3_5HybridCache(
            record_checkpoints=True, checkpoint_positions=[16, 48, 64]
        )
        out_single_80 = qwen(
            input_ids=seq_80,
            attention_mask=torch.ones_like(seq_80),
            past_key_values=traj_cache,
            use_cache=True,
        )
        cp64_cache = traj_cache.extract_checkpoint(64)
        out_sfx_64 = qwen(
            input_ids=seq_80[:, 64:],
            past_key_values=cp64_cache,
            use_cache=False,
        )
        spliced_80 = torch.cat(
            [out_single_80.last_hidden_state[:, :64, :], out_sfx_64.last_hidden_state],
            dim=1,
        )
        traj_splice_diff = float(
            (out_single_80.last_hidden_state - spliced_80).abs().max().item()
        )
        traj_splice_mae = float(
            (out_single_80.last_hidden_state - spliced_80).abs().mean().item()
        )
        _assert(
            traj_splice_diff <= 0.0625 and traj_splice_mae < 2e-3,
            f"Single-pass Qwen3.5 trajectory splice diverged from unsplit pass: max={traj_splice_diff}, mae={traj_splice_mae}",
        )

        mm_seq_a = torch.tensor(
            [
                [
                    10,
                    20,
                    900,
                    900,
                    900,
                    900,
                    25,
                    30,
                    31,
                    32,
                    33,
                    34,
                    35,
                    36,
                    37,
                    38,
                    101,
                    102,
                    103,
                    104,
                ]
            ],
            dtype=torch.long,
            device=dev,
        )
        mm_seq_b = torch.tensor(
            [
                [
                    10,
                    20,
                    900,
                    900,
                    900,
                    900,
                    25,
                    30,
                    31,
                    32,
                    33,
                    34,
                    35,
                    36,
                    37,
                    38,
                    201,
                    202,
                    203,
                    204,
                ]
            ],
            dtype=torch.long,
            device=dev,
        )
        mm_traj_cache = Qwen3_5HybridCache(
            record_checkpoints=True, checkpoint_positions=[12, 16]
        )
        out_mm_a = qwen(
            input_ids=mm_seq_a,
            attention_mask=torch.ones_like(mm_seq_a),
            pixel_values=pix,
            image_grid_thw=g_thw,
            past_key_values=mm_traj_cache,
            use_cache=True,
        )
        _assert(
            mm_traj_cache.rope_deltas is not None
            and int(mm_traj_cache.rope_deltas.item()) == -2,
            f"Gate 3.B Expected multimodal rope_deltas == -2, got {mm_traj_cache.rope_deltas}",
        )
        mm_cp16 = mm_traj_cache.extract_checkpoint(16)
        out_mm_b_sfx = qwen(
            input_ids=mm_seq_b[:, 16:],
            past_key_values=mm_cp16,
            use_cache=False,
        )
        out_mm_b_full = qwen(
            input_ids=mm_seq_b,
            attention_mask=torch.ones_like(mm_seq_b),
            pixel_values=pix,
            image_grid_thw=g_thw,
        )
        spliced_mm_b = torch.cat(
            [out_mm_a.last_hidden_state[:, :16, :], out_mm_b_sfx.last_hidden_state],
            dim=1,
        )
        mm_splice_diff = float(
            (out_mm_b_full.last_hidden_state - spliced_mm_b).abs().max().item()
        )
        mm_splice_mae = float(
            (out_mm_b_full.last_hidden_state - spliced_mm_b).abs().mean().item()
        )
        _assert(
            mm_splice_diff <= 0.0625 and mm_splice_mae < 2e-3,
            f"Gate 3.B Multimodal trajectory checkpoint splice diverged from unsplit pass: max={mm_splice_diff}, mae={mm_splice_mae}",
        )

        q_out = qwen(
            input_ids=in_ids,
            attention_mask=torch.ones_like(in_ids),
            pixel_values=pix,
            image_grid_thw=g_thw,
        )
        _assert(
            q_out.rope_deltas is not None,
            "Multimodal Qwen3.5 pass must populate rope_deltas",
        )
        q_txt_only = qwen(
            input_ids=in_ids[:, :4],
            attention_mask=torch.ones_like(in_ids[:, :4]),
        )
        _assert(
            q_txt_only.rope_deltas is None,
            "Text-only Qwen3.5 pass must reset rope_deltas to None",
        )

        prompt_seq = torch.tensor(
            [[11, 22, 33, 44, 11, 22, 33, 44, 55, 66, 11, 22]],
            dtype=torch.long,
            device=dev,
        )
        spec_ids_hi, spec_hidden_hi = qwen.reanchor_speculative_generate(
            prompt_seq,
            max_new_tokens=6,
            block_size=3,
            max_candidates=3,
            entropy_threshold=1.25,
            stop_token_ids=[999999],
        )
        spec_ids_lo, spec_hidden_lo = qwen.reanchor_speculative_generate(
            prompt_seq,
            max_new_tokens=6,
            block_size=3,
            max_candidates=3,
            entropy_threshold=20.0,
            stop_token_ids=[999999],
        )
        greedy_cache = Qwen3_5HybridCache()
        g_out = qwen(prompt_seq, past_key_values=greedy_cache, use_cache=True)
        greedy_ids: list[int] = []
        greedy_h_chunks: list[torch.Tensor] = [g_out.last_hidden_state]
        cur_logits = qwen.project_logits(g_out.last_hidden_state[:, -1, :]).float()
        for _ in range(6):
            nt = int(torch.argmax(cur_logits, dim=-1).item())
            greedy_ids.append(nt)
            step_t = torch.tensor([[nt]], dtype=torch.long, device=dev)
            g_out = qwen(step_t, past_key_values=greedy_cache, use_cache=True)
            greedy_h_chunks.append(g_out.last_hidden_state)
            cur_logits = qwen.project_logits(g_out.last_hidden_state[:, -1, :]).float()
        greedy_hidden = torch.cat(greedy_h_chunks, dim=1)
        spec_h_diff = max(
            float((spec_hidden_hi - greedy_hidden).abs().max().item()),
            float((spec_hidden_lo - greedy_hidden).abs().max().item()),
        )
        _assert(
            spec_ids_hi == greedy_ids
            and spec_ids_lo == greedy_ids
            and spec_hidden_hi.shape == greedy_hidden.shape
            and spec_hidden_lo.shape == greedy_hidden.shape
            and spec_h_diff <= 0.03125,
            f"Dual-regime speculative decoding diverged from greedy decoding: hi={spec_ids_hi}, lo={spec_ids_lo}, greedy={greedy_ids}, h_diff={spec_h_diff}",
        )

        close_test_ids = [701, 702, 703]
        spec_closed_ids, spec_closed_hidden, spec_closed_cache = (
            qwen.reanchor_speculative_generate(
                prompt_seq,
                max_new_tokens=4,
                block_size=3,
                max_candidates=3,
                entropy_threshold=1.25,
                stop_token_ids=[999999],
                close_token_ids=close_test_ids,
                return_cache=True,
            )
        )
        _assert(
            spec_closed_ids[-3:] == close_test_ids
            and spec_closed_hidden.shape[1] == prompt_seq.shape[1] + len(spec_closed_ids)
            and spec_closed_cache is not None
            and spec_closed_cache.past_seen_tokens
            == prompt_seq.shape[1] + len(spec_closed_ids),
            f"Speculative CoT auto-closure failed: ids={spec_closed_ids}, h_shape={spec_closed_hidden.shape}",
        )

    qwen_d03 = Qwen3_5Model(q_cfg).to(device=dev, dtype=dtype).eval()
    qwen_d03.load_state_dict(qwen.state_dict(), strict=False)
    qwen_d03.enable_lm_head(device=dev, dtype=dtype)
    qwen_d03.enable_mtp(device=dev, dtype=dtype)
    with torch.no_grad():
        qwen_d03.lm_head.weight.copy_(qwen.language_model.embed_tokens.weight)
    with (
        tempfile.TemporaryDirectory() as tmpdir_q,
        tempfile.TemporaryDirectory() as tmpdir_q_stripped,
    ):
        q_sf = Path(tmpdir_q) / "model.safetensors"
        save_safetensors(
            {k: v.detach().cpu() for k, v in qwen_d03.state_dict().items()},
            str(q_sf),
        )
        q_sf_stripped = Path(tmpdir_q_stripped) / "model.safetensors"
        save_safetensors(
            {k: v.detach().cpu() for k, v in qwen.state_dict().items()},
            str(q_sf_stripped),
        )
        with _MetaWeightInitMode():
            qwen_loaded = Qwen3_5Model(q_cfg)
        load_sharded_safetensors_into(
            Path(tmpdir_q), qwen_loaded, device=dev, dtype=dtype
        )
        qwen_loaded.eval()
        _assert(
            qwen_loaded.lm_head is not None
            and isinstance(qwen_loaded.mtp, Qwen3_5MTPHead),
            "load_sharded_safetensors_into failed to auto-materialize Domain 03 lm_head and mtp.*",
        )
        inplace_detach_ok = stream_safetensors_inplace(
            Path(tmpdir_q_stripped), qwen_loaded, device=dev, dtype=dtype
        )
        _assert(
            inplace_detach_ok
            and qwen_loaded.lm_head is None
            and qwen_loaded.mtp is None,
            "stream_safetensors_inplace failed to detach lm_head and mtp when loading Domain 01/02 shards",
        )
        inplace_attach_ok = stream_safetensors_inplace(
            Path(tmpdir_q), qwen_loaded, device=dev, dtype=dtype
        )
        _assert(
            inplace_attach_ok
            and qwen_loaded.lm_head is not None
            and isinstance(qwen_loaded.mtp, Qwen3_5MTPHead),
            "stream_safetensors_inplace failed to re-materialize lm_head and mtp when loading Domain 03 shards",
        )

    with torch.inference_mode():
        dummy_base_h = torch.randn((1, 1, 128), device=dev, dtype=dtype)
        dummy_retained = [
            ReAnchorVerifierState(
                offset_d=1,
                conditioning_token=33,
                anchor_token=44,
                hidden_state=torch.randn((1, 1, 128), device=dev, dtype=dtype),
                entropy_bits=1.2,
                lookahead_tokens=(44, 55),
            ),
            ReAnchorVerifierState(
                offset_d=2,
                conditioning_token=55,
                anchor_token=66,
                hidden_state=torch.randn((1, 1, 128), device=dev, dtype=dtype),
                entropy_bits=1.5,
                lookahead_tokens=(66,),
            ),
        ]
        base_mtp_cands, shifted_mtp_cands = qwen_d03._propose_batched_shifted_mtp(
            base_hidden=dummy_base_h,
            base_token=22,
            retained_anchors=dummy_retained,
            block_size=3,
            max_candidates=3,
            past_seen_tokens=12,
        )
        _assert(
            len(base_mtp_cands) >= 1
            and len(shifted_mtp_cands) == 2
            and shifted_mtp_cands[0][1][0] == 44
            and shifted_mtp_cands[1][1][0] == 66,
            f"_propose_batched_shifted_mtp failed parallel shifted MTP drafting: base={base_mtp_cands}, shifted={shifted_mtp_cands}",
        )

        mtp_ids, mtp_hidden = qwen_d03.reanchor_speculative_generate(
            prompt_seq,
            max_new_tokens=6,
            block_size=3,
            max_candidates=3,
            entropy_threshold=1.25,
            stop_token_ids=[999999],
        )
        loaded_ids, loaded_hidden = qwen_loaded.reanchor_speculative_generate(
            prompt_seq,
            max_new_tokens=6,
            block_size=3,
            max_candidates=3,
            entropy_threshold=1.25,
            stop_token_ids=[999999],
        )
        _assert(
            mtp_ids == greedy_ids
            and loaded_ids == greedy_ids
            and mtp_hidden.shape == greedy_hidden.shape
            and loaded_hidden.shape == greedy_hidden.shape
            and float((mtp_hidden - greedy_hidden).abs().max().item()) <= 0.03125
            and float((loaded_hidden - greedy_hidden).abs().max().item()) <= 0.03125
            and qwen_d03.last_reanchor_telemetry.get("total_rounds", 0) >= 1,
            f"Domain 03 Qwen3_5MTPHead ReAnchor speculative decoding diverged from greedy target: mtp={mtp_ids}, loaded={loaded_ids}, greedy={greedy_ids}",
        )

        mm_greedy_cache = Qwen3_5HybridCache()
        mm_g_out = qwen(
            input_ids=in_ids,
            pixel_values=pix,
            image_grid_thw=g_thw,
            past_key_values=mm_greedy_cache,
            use_cache=True,
        )
        mm_greedy_ids: list[int] = []
        mm_greedy_chunks: list[torch.Tensor] = [mm_g_out.last_hidden_state]
        mm_logits = qwen.project_logits(mm_g_out.last_hidden_state[:, -1, :]).float()
        for _ in range(4):
            nt_mm = int(torch.argmax(mm_logits, dim=-1).item())
            mm_greedy_ids.append(nt_mm)
            step_mm = torch.tensor([[nt_mm]], dtype=torch.long, device=dev)
            mm_g_out = qwen(
                input_ids=step_mm, past_key_values=mm_greedy_cache, use_cache=True
            )
            mm_greedy_chunks.append(mm_g_out.last_hidden_state)
            mm_logits = qwen.project_logits(
                mm_g_out.last_hidden_state[:, -1, :]
            ).float()
        mm_greedy_hidden = torch.cat(mm_greedy_chunks, dim=1)
        mm_mtp_ids, mm_mtp_hidden = qwen_d03.reanchor_speculative_generate(
            in_ids,
            max_new_tokens=4,
            block_size=3,
            max_candidates=3,
            entropy_threshold=1.25,
            stop_token_ids=[999999],
            pixel_values=pix,
            image_grid_thw=g_thw,
        )
        _assert(
            mm_mtp_ids == mm_greedy_ids
            and mm_mtp_hidden.shape == mm_greedy_hidden.shape
            and float((mm_mtp_hidden - mm_greedy_hidden).abs().max().item()) <= 0.03125,
            f"Multimodal Qwen3_5MTPHead speculative decoding diverged from greedy target: mtp={mm_mtp_ids}, greedy={mm_greedy_ids}",
        )

    _assert(
        q_out.last_hidden_state.shape == (1, 8, 128),
        f"Unexpected Qwen3.5 output shape: {q_out.last_hidden_state.shape}",
    )
    _assert(
        len(qwen.visual._geom_cache) >= 1,
        "Vision geometry cache was not populated",
    )
    qwen.clear_caches()
    _assert(
        len(qwen.visual._geom_cache) == 0 and len(_STRICTLY_UPPER_MASK_CACHE) == 0,
        "Qwen3_5Model.clear_caches() must flush _geom_cache and _STRICTLY_UPPER_MASK_CACHE",
    )
    print(
        f"  [+] Qwen3_5Model (Dom 01-03) : ViTGeom (err={pe_interp_err:.1e}) + MMSplice (mae={mm_splice_mae:.1e}) + ReAnchorMTP/Markov SpecVerify (h_diff={spec_h_diff:.1e}) Verified"
    )

    e4m3_dtype = getattr(torch, "float8_e4m3fn", None)
    if e4m3_dtype is not None:
        with torch.inference_mode():
            dit.layers[0].proj["q"].weight.data = (
                dit.layers[0].proj["q"].weight.data.to(e4m3_dtype)
            )
            qwen.language_model.layers[0].mlp.gate_proj.weight.data = (
                qwen.language_model.layers[0].mlp.gate_proj.weight.data.to(e4m3_dtype)
            )
            dit.clear_caches()
            fp8_dit_out = dit(lat, torch.tensor([500.0], device=dev), txt, mask).sample
            fp8_qwen_out = qwen(
                input_ids=in_ids[:, :4],
                attention_mask=torch.ones_like(in_ids[:, :4]),
            ).last_hidden_state
            _assert(
                torch.isfinite(fp8_dit_out).all()
                and torch.isfinite(fp8_qwen_out).all(),
                "NaN/Inf in FP8 weight-only dispatch",
            )
        print(
            "  [+] FP8 Weight-Only Dispatch : Verified float8_e4m3fn Linear Execution in MOSS-DiT & Qwen3.5"
        )

    v_cfg = Flux2VAEConfig(
        block_out_channels=(32, 64),
        decoder_block_out_channels=(32, 64),
        latent_channels=32,
        layers_per_block=1,
        norm_num_groups=16,
    )
    vae = Flux2AutoencoderKL(config=v_cfg).to(device=dev, dtype=torch.float32).eval()
    with torch.no_grad():
        vae.bn.running_mean.copy_(
            torch.linspace(-0.75, 0.75, 128, device=dev, dtype=torch.float32)
        )
        vae.bn.running_var.copy_(
            torch.linspace(0.35, 1.85, 128, device=dev, dtype=torch.float32)
        )
    vae.clear_caches()
    with torch.inference_mode():
        posterior_mode = torch.randn(
            (1, 32, 32, 32), device=dev, dtype=torch.float32
        )
        b_sz, c_lat, h2, w2 = posterior_mode.shape
        z_packed_alg = (
            posterior_mode.reshape(b_sz, c_lat, h2 // 2, 2, w2 // 2, 2)
            .permute(0, 1, 3, 5, 2, 4)
            .reshape(b_sz, c_lat * 4, h2 // 2, w2 // 2)
            .clone()
        )
        bn_mean, bn_std, bn_inv_std = vae._get_bn_stats(dev)
        z_packed_alg.sub_(bn_mean).mul_(bn_inv_std)
        z_unpacked_alg = z_packed_alg.clone().mul_(bn_std).add_(bn_mean)
        z_unpacked_alg = (
            z_unpacked_alg.reshape(b_sz, c_lat, 2, 2, h2 // 2, w2 // 2)
            .permute(0, 1, 4, 2, 5, 3)
            .reshape(b_sz, c_lat, h2, w2)
        )
        pack_invert_err = float(
            (posterior_mode - z_unpacked_alg).abs().max().item()
        )
        _assert(
            pack_invert_err < 1e-6,
            f"Gate 3.E 2x2 Pack + BatchNorm2d(128) algebraic inversion error exceeded 1e-6: {pack_invert_err}",
        )

        strip_conv = (
            torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
            .to(device=dev, dtype=torch.float32)
            .eval()
        )
        strip_x = torch.randn((1, 64, 512, 512), device=dev, dtype=torch.float32)
        out_untiled = strip_conv(strip_x)
        out_tiled = _conv2d_3x3_strip_tiled(strip_conv, strip_x, enabled=True)
        strip_err = float((out_untiled - out_tiled).abs().max().item())
        _assert(
            strip_err == 0.0,
            f"Halo-exact strip-tiled 3x3 Conv2d diverged from untiled Conv2d: max_err={strip_err}",
        )
        del strip_conv, strip_x, out_untiled, out_tiled

        img_in = (
            torch.randn((1, 3, 64, 64), device=dev, dtype=torch.float32).clamp(
                -1.0, 1.0
            )
        )
        z_pack = vae.encode_to_packed_latents(img_in)
        _assert(
            vae._bn_stats_cache is not None,
            "Flux2AutoencoderKL failed to populate _bn_stats_cache",
        )
        z_outlier = z_pack * 10.0
        z_outlier[:, 0:4, 8, 8] = torch.tensor(
            [12.0, 24.0, -12.0, 36.0], device=dev, dtype=torch.float32
        )
        ratios_before = z_outlier[0, 0:4, 8, 8] / z_outlier[0, 0, 8, 8]
        z_clamped = subphase_radial_clamp(z_outlier, max_sigma=4.5)
        ratios_after = z_clamped[0, 0:4, 8, 8] / z_clamped[0, 0, 8, 8]
        subphase_ratio_err = float((ratios_before - ratios_after).abs().max().item())
        _assert(
            subphase_ratio_err < 1e-6,
            f"subphase_radial_clamp distorted 2x2 sub-phase ratio: {subphase_ratio_err}",
        )
        dec_out = vae.decode_packed_latents(z_clamped)
        vae.clear_caches()
        _assert(
            vae._bn_stats_cache is None,
            "Flux2AutoencoderKL.clear_caches() must reset _bn_stats_cache",
        )
    _assert(
        z_pack.shape == (1, 128, 16, 16),
        f"Unexpected packed latent shape: {z_pack.shape}",
    )
    _assert(
        dec_out.shape == (1, 3, 64, 64),
        f"Unexpected decoded image shape: {dec_out.shape}",
    )
    print(
        f"  [+] Flux2AutoencoderKL       : Bit-Exact Halo-Tiled Conv2d (err={strip_err:.1e}) + 2x2PackBN128Inv (err={pack_invert_err:.1e}) + Radial Clamp Verified"
    )


if __name__ == "__main__":
    gate_3_models_zero_alloc_and_lora()