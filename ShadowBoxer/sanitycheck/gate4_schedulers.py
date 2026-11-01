#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

sys.modules["torchvision"] = None
sys.modules["torchvision.transforms"] = None
sys.modules["torchvision.io"] = None

import torch

from core.schedulers import (
    _AB_MAX_LEBESGUE_L1,
    _AM_MAX_LEBESGUE_L1,
    unwarp_sigma_to_progress,
)
from engine import (
    BifurcatedFlowMatchScheduler,
    MOSSImageScheduler,
    compute_moss_v2_sigmas,
    derive_stage_keys,
    philox_randn,
    warp_denoise_to_sigma_horizon,
)
from schema import SUPPORTED_SOLVERS


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def gate_4_schedulers_and_prng() -> None:
    print(
        "\n--- Gate 4: Flow-Matching V2 Schedule, Lagrange ABM4 Solvers & Philox PRNG ---"
    )
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    v2_sigmas = compute_moss_v2_sigmas(50, shift=5.0)
    _assert(len(v2_sigmas) == 50, f"Expected 50 V2 sigmas, got {len(v2_sigmas)}")
    _assert(
        bool(torch.all(v2_sigmas[:-1] >= v2_sigmas[1:]).item()),
        "V2 sigmas are not monotonically decreasing",
    )

    def _upstream_ref_sigmas(
        steps: int = 50,
        shift: float = 5.0,
        training_shift: float = 1.0,
        total: int = 1000,
    ) -> torch.Tensor:
        def grid(count: int, s: float) -> torch.Tensor:
            lower = s / (total + s - 1)
            values = torch.linspace(1, lower, count)
            return s * values / (1 + (s - 1) * values)

        sigmas = grid(steps, shift)
        training = grid(total, training_shift).flip(0)
        upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
        lower = (upper - 1).clamp(min=0)
        nearest = torch.where(
            (sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(),
            lower,
            upper,
        )
        return torch.unique_consecutive(training[nearest])

    ref_sigmas = _upstream_ref_sigmas(50, 5.0)
    _assert(
        float((v2_sigmas - ref_sigmas).abs().max().item()) == 0.0,
        "V2 sigma schedule drifted from upstream FP32 reference",
    )

    h_unif = -0.08
    w_ab3_unif = BifurcatedFlowMatchScheduler._compute_ab_weights(
        h_unif, h_unif, h_unif, h_unif, 3
    )
    w_ab4_unif = BifurcatedFlowMatchScheduler._compute_ab_weights(
        h_unif, h_unif, h_unif, h_unif, 4
    )
    _assert(
        abs(w_ab3_unif[0] / h_unif - 23.0 / 12.0) < 1e-12
        and abs(w_ab3_unif[1] / h_unif - (-16.0 / 12.0)) < 1e-12
        and abs(w_ab3_unif[2] / h_unif - 5.0 / 12.0) < 1e-12,
        f"Uniform Order-3 Adams-Bashforth weights did not match exact (23, -16, 5)/12: {w_ab3_unif}",
    )
    _assert(
        abs(w_ab4_unif[0] / h_unif - 55.0 / 24.0) < 1e-12
        and abs(w_ab4_unif[1] / h_unif - (-59.0 / 24.0)) < 1e-12
        and abs(w_ab4_unif[2] / h_unif - 37.0 / 24.0) < 1e-12
        and abs(w_ab4_unif[3] / h_unif - (-9.0 / 24.0)) < 1e-12,
        f"Uniform Order-4 Adams-Bashforth weights did not match exact (55, -59, 37, -9)/24: {w_ab4_unif}",
    )

    for ord_m in (1, 2, 3, 4):
        max_l1 = _AB_MAX_LEBESGUE_L1.get(ord_m, 1.0)
        for h_tuple in (
            (-0.08, -0.08, -0.08, -0.08),
            (-0.095, -0.082, -0.072, -0.064),
            (-0.25, -0.15, -0.10, -0.07),
        ):
            w_ab = BifurcatedFlowMatchScheduler._compute_ab_weights(
                h_tuple[0], h_tuple[1], h_tuple[2], h_tuple[3], ord_m
            )
            unity_err = abs(sum(w_ab) - h_tuple[0])
            l1_ratio = sum(abs(x) for x in w_ab) / abs(h_tuple[0])
            _assert(
                unity_err < 1e-12,
                f"Lagrange AB weights violated partition of unity (order={ord_m}, err={unity_err})",
            )
            _assert(
                l1_ratio <= max_l1 + 1e-9,
                f"Lagrange AB weights exceeded Dahlquist L1 bound (order={ord_m}, ratio={l1_ratio})",
            )

        h0, h1, h2, h3 = -0.085, -0.078, -0.072, -0.068
        w_poly = BifurcatedFlowMatchScheduler._compute_ab_weights(
            h0, h1, h2, h3, ord_m
        )
        nodes_ab = (0.0, -h1, -(h1 + h2), -(h1 + h2 + h3))
        for deg in range(ord_m):
            quad_val = sum(w_poly[k] * (nodes_ab[k] ** deg) for k in range(ord_m))
            exact_val = (h0 ** (deg + 1)) / float(deg + 1)
            _assert(
                abs(quad_val - exact_val) < 1e-12,
                f"Variable-step AB order {ord_m} failed polynomial exactness at degree {deg}: {quad_val} vs {exact_val}",
            )

    for ord_am in (2, 3, 4):
        h0, h1, h2 = -0.09, -0.08, -0.07
        w_am = BifurcatedFlowMatchScheduler._compute_am_weights(
            h0, h1, h2, ord_am
        )
        unity_am = abs(sum(w_am) - h0)
        l1_am = sum(abs(x) for x in w_am) / abs(h0)
        max_am_l1 = _AM_MAX_LEBESGUE_L1.get(ord_am, 2.15)
        _assert(
            unity_am < 1e-12 and l1_am <= max_am_l1 + 1e-9,
            f"Lagrange AM weights violated partition of unity or L1 bound (order={ord_am}): err={unity_am}, l1={l1_am}",
        )
        nodes_am = (h0, 0.0, -h1, -(h1 + h2))
        for deg in range(ord_am):
            quad_am = sum(w_am[k] * (nodes_am[k] ** deg) for k in range(ord_am))
            exact_am = (h0 ** (deg + 1)) / float(deg + 1)
            _assert(
                abs(quad_am - exact_am) < 1e-12,
                f"Variable-step AM order {ord_am} failed polynomial exactness at degree {deg}: {quad_am} vs {exact_am}",
            )

    warped_half = float(warp_denoise_to_sigma_horizon(0.5, shift=5.0))
    _assert(
        abs(warped_half - (2.5 / 3.0)) < 1e-6
        and abs(float(warp_denoise_to_sigma_horizon(0.5, shift=1.0)) - 0.5) < 1e-6,
        f"warp_denoise_to_sigma_horizon failed Mobius transformation: got {warped_half}",
    )
    for d_probe in (0.15, 0.35, 0.50, 0.65, 0.85):
        s_warped = float(warp_denoise_to_sigma_horizon(d_probe, shift=5.0))
        u_unwarped = float(unwarp_sigma_to_progress(s_warped, shift=5.0))
        _assert(
            abs(u_unwarped - d_probe) < 1e-6,
            f"Gate 4.B Mobius roundtrip inversion failed for d={d_probe}: got {u_unwarped}",
        )

    quad_sched = BifurcatedFlowMatchScheduler(
        early_solver="heun",
        late_solver="ipndm",
        handoff_threshold=0.35,
        shift=5.0,
        use_v2_grid=False,
    )
    quad_sched.set_timesteps(20, device=dev)
    z_quad = torch.zeros((1, 1, 1, 1), device=dev, dtype=torch.float32)
    c0_q, c1_q, c2_q = 0.75, -1.20, 0.90
    for t_q in quad_sched.timesteps:
        sig_q = float(t_q.item()) / float(quad_sched.num_train_timesteps)
        v_q = torch.full_like(
            z_quad, c0_q + c1_q * sig_q + c2_q * (sig_q * sig_q)
        )
        z_quad = quad_sched.step(v_q, t_q, z_quad)
    exact_quad_integral = -(c0_q + 0.5 * c1_q + (1.0 / 3.0) * c2_q)
    quad_int_err = abs(float(z_quad.item()) - exact_quad_integral)
    _assert(
        quad_int_err < 1e-3,
        f"Gate 4.A Shifted sigma-space polynomial quadrature drift too high: got {z_quad.item()} vs {exact_quad_integral} (err={quad_int_err})",
    )

    std_sched = MOSSImageScheduler(shift=5.0)
    std_sched.set_timesteps(20, device=dev)
    x = torch.randn((1, 128, 8, 8), device=dev)
    for t in std_sched.timesteps:
        x = std_sched.step(-x, t, x)
    _assert(
        bool(torch.isfinite(x).all().item()) and float(x.float().std().item()) > 0.1,
        "Non-finite or collapsed variance in MOSSImageScheduler integration",
    )

    euler_ipndm = BifurcatedFlowMatchScheduler(
        early_solver="euler",
        late_solver="ipndm",
        handoff_threshold=0.5,
        shift=1.0,
        use_v2_grid=False,
    )
    euler_ipndm.set_timesteps(10, device=dev)
    _assert(
        euler_ipndm._early_steps == 5 and euler_ipndm._late_steps == 5,
        f"Expected 5 early and 5 late steps at handoff=0.5 (shift=1.0), got {euler_ipndm._early_steps}/{euler_ipndm._late_steps}",
    )
    s_test = torch.randn((1, 128, 8, 8), device=dev)
    for idx, t in enumerate(euler_ipndm.timesteps):
        was_early = euler_ipndm.current_is_early
        s_test = euler_ipndm.step(-0.5 * s_test, t, s_test)
        if was_early:
            _assert(
                len(euler_ipndm._history) == 0,
                "Euler step must not push to multistep _history",
            )
        elif idx < len(euler_ipndm.timesteps) - 1:
            _assert(
                len(euler_ipndm._history) > 0,
                "IPNDM step must populate multistep _history",
            )

    heun_ipndm_12 = BifurcatedFlowMatchScheduler(
        early_solver="heun",
        late_solver="ipndm",
        handoff_threshold=0.35,
        shift=5.0,
        use_v2_grid=True,
    )
    heun_ipndm_12.set_timesteps(12, device=dev)
    k_star = heun_ipndm_12._k_star
    _assert(
        k_star == 4 and heun_ipndm_12._early_steps == 4 and heun_ipndm_12._late_steps == 8,
        f"Gate 4.B Expected unwarped trajectory handoff index k_star=4 (4 early, 8 late) for 12-step shift=5.0, got k_star={k_star}",
    )
    _assert(
        heun_ipndm_12.u_sigmas is not None
        and (1.0 - float(heun_ipndm_12.u_sigmas[k_star - 1].item())) < 0.35
        and (1.0 - float(heun_ipndm_12.u_sigmas[k_star].item())) >= 0.35 - 1e-7,
        "Gate 4.B Unwarped trajectory progress handoff (1 - u >= t*) violated",
    )
    _assert(
        heun_ipndm_12._can_warm_handoff is True,
        "Heun->IPNDM must enable Runge-Kutta warm handoff",
    )
    handoff_cfg = next(
        cfg for cfg in heun_ipndm_12._step_configs if not cfg["is_early"]
    )
    handoff_l1_ratio = sum(abs(x) for x in handoff_cfg["weights"]) / abs(
        handoff_cfg["dt"]
    )
    _assert(
        handoff_cfg["order"] == min(4, k_star + 1)
        and abs(handoff_cfg["weights"][2]) > 1e-6
        and abs(handoff_cfg["weights"][3]) > 1e-6
        and handoff_l1_ratio <= _AB_MAX_LEBESGUE_L1[4] + 1e-9,
        f"Expected warm-started IPNDM order {min(4, k_star + 1)} with active 4th-order weights at handoff, got order={handoff_cfg['order']}, weights={handoff_cfg['weights']}",
    )
    late_cfgs_12 = [
        cfg for cfg in heun_ipndm_12._step_configs if not cfg["is_early"]
    ]
    _assert(
        late_cfgs_12[-1]["order"] <= 2 and late_cfgs_12[-2]["order"] <= 2,
        f"Gate 4.A Dahlquist step-ratio guard failed to step down terminal IPNDM order on expanding V2 steps: {[c['order'] for c in late_cfgs_12]}",
    )
    s_warm = torch.ones((1, 4, 4, 4), device=dev, dtype=torch.float32)
    for t in heun_ipndm_12.timesteps:
        want_h = heun_ipndm_12.wants_early_history_output
        v_step = -0.5 * s_warm
        v_hist = -0.4 * s_warm if want_h else None
        s_warm = heun_ipndm_12.step(
            v_step, t, s_warm, history_output=v_hist
        )
    _assert(
        torch.isfinite(s_warm).all(),
        "Warm-started Heun->IPNDM with CFG-consistent history_output produced non-finite values",
    )

    heun_full = BifurcatedFlowMatchScheduler(
        early_solver="heun",
        late_solver="heun",
        handoff_threshold=0.5,
        shift=5.0,
        use_v2_grid=True,
    )
    heun_full.set_timesteps(8, device=dev)
    term_cfg = heun_full._step_configs[-1]
    _assert(
        term_cfg["mode"] == "multistep"
        and term_cfg["order"] == 2
        and abs(term_cfg["weights"][1]) > 1e-6,
        f"Terminal step of Heun solver must preserve 2nd-order Adams-Bashforth weights instead of dropping to Euler: {term_cfg}",
    )

    mt_sched = BifurcatedFlowMatchScheduler(
        early_solver="multitree",
        late_solver="multitree",
        handoff_threshold=0.5,
        shift=5.0,
        use_v2_grid=True,
    )
    mt_sched.set_timesteps(8, device=dev)
    corr_cfgs = [c for c in mt_sched._step_configs if c["mode"] == "corrector"]
    _assert(
        len(corr_cfgs) == 7
        and corr_cfgs[-1]["order"] == 4
        and abs(corr_cfgs[-1]["corr_weights"][2]) > 1e-6,
        f"multitree corrector failed to engage 4th-order Adams-Moulton weights: {corr_cfgs[-1]}",
    )

    sde_warm = BifurcatedFlowMatchScheduler(
        early_solver="sde_gpu_pp",
        late_solver="ipndm",
        handoff_threshold=0.35,
        shift=5.0,
        eta=0.15,
        s_noise=1.0,
        use_v2_grid=True,
    )
    sde_warm.set_timesteps(12, device=dev)
    sde_late_cfgs = [c for c in sde_warm._step_configs if not c["is_early"]]
    _assert(
        sde_warm._can_warm_handoff is True
        and sde_late_cfgs[0]["order"] == 2
        and sde_late_cfgs[1]["order"] >= 3,
        f"Gate 4.B sde_gpu_pp -> ipndm must use single-anchor Order-2 handoff before ramping order: {[c['order'] for c in sde_late_cfgs]}",
    )
    disp_unit = torch.ones((1, 4, 4, 4), device=dev, dtype=torch.float32)
    v_shift_sde = sde_warm._transport_history_for_sde_jump(disp_unit, s_next=0.80)
    expected_susc = 0.80 / ((1.0 - 0.80) ** 2 + 0.80**2)
    max_theoretical_susc = 0.5 * (1.0 + math.sqrt(2.0))
    _assert(
        v_shift_sde is not None
        and abs(float(v_shift_sde.max().item()) - expected_susc) < 1e-6
        and float(v_shift_sde.max().item()) <= max_theoretical_susc + 1e-6
        and float(v_shift_sde.max().item()) < (1.0 / 0.80),
        f"Gate 4.B SDE covariant susceptibility mismatch: got {None if v_shift_sde is None else v_shift_sde.max().item()} vs {expected_susc}",
    )

    k_init, k_sde, k_aux = derive_stage_keys(42)
    _assert(len({k_init, k_sde, k_aux}) == 3, "Stage keys are not orthogonal")

    z_target_het = philox_randn((1, 128, 8, 8), seed=k_init, stream_id=10, device=dev)
    z_target_het[:, :, 2:6, 2:6] *= 1.35
    noise_het = philox_randn((1, 128, 8, 8), seed=k_init, stream_id=11, device=dev)
    for early_s, eta_v in (("heun", 0.0), ("sde_gpu_pp", 0.15)):
        het_sched = BifurcatedFlowMatchScheduler(
            early_solver=early_s,
            late_solver="ipndm",
            handoff_threshold=0.35,
            shift=5.0,
            eta=eta_v,
            s_noise=1.0,
            use_v2_grid=True,
        )
        het_sched.set_timesteps(12, device=dev)
        z_cur = noise_het.clone()
        gen_het = torch.Generator(device=dev).manual_seed(k_sde & 0x7FFFFFFFFFFFFFFF)
        for t_h in het_sched.timesteps:
            s_h = max(float(t_h.item()) / float(het_sched.num_train_timesteps), 1e-3)
            v_bridge = (z_cur - z_target_het) / s_h
            z_cur = het_sched.step(v_bridge, t_h, z_cur, generator=gen_het)
        _assert(
            bool(torch.isfinite(z_cur).all().item())
            and float(z_cur[:, :, 2:6, 2:6].std().item()) > 0.5
            and float(z_cur[:, :, :2, :2].std().item()) > 0.5,
            f"Spatially heterogeneous bridge integration collapsed or diverged under {early_s}->ipndm",
        )

    for solver in SUPPORTED_SOLVERS:
        bif = BifurcatedFlowMatchScheduler(
            early_solver=solver,
            late_solver="ipndm" if solver != "ipndm" else "euler",
            handoff_threshold=0.35,
            shift=5.0,
            eta=0.15 if solver in ("sde_gpu_pp", "multitree") else 0.0,
            s_noise=1.0,
            use_v2_grid=True,
        )
        bif.set_timesteps(16, device=dev)
        sample = torch.randn((1, 128, 8, 8), device=dev)
        gen = torch.Generator(device=dev).manual_seed(12345)
        for t in bif.timesteps:
            sample = bif.step(-0.5 * sample, t, sample, generator=gen)
        _assert(
            bool(torch.isfinite(sample).all().item())
            and float(sample.float().std().item()) > 0.1,
            f"Non-finite or collapsed variance encountered in solver {solver}",
        )

    n1 = philox_randn((1, 128, 8, 8), seed=k_init, stream_id=1, device=dev)
    n2 = philox_randn((1, 128, 8, 8), seed=k_init, stream_id=1, device=dev)
    n3 = philox_randn((1, 128, 8, 8), seed=k_init, stream_id=2, device=dev)
    _assert(
        float((n1 - n2).abs().max().item()) == 0.0,
        "Philox PRNG not deterministic for identical seed/stream",
    )
    _assert(
        float((n1 - n3).abs().max().item()) > 0.1,
        "Philox PRNG stream collision",
    )
    print(
        f"  [+] Bifurcated Solvers & PRNG: Verified {', '.join(s.upper() for s in SUPPORTED_SOLVERS)} + SigmaABM4 (err={quad_int_err:.1e}) + SingleAnchorSDEHandoff + Philox64"
    )


if __name__ == "__main__":
    gate_4_schedulers_and_prng()