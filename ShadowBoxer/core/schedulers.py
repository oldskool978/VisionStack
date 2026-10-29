# ShadowBoxer/core/schedulers.py
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from . import env as _env  # noqa: F401


def compute_moss_v2_sigmas(
    num_inference_steps: int,
    shift: float = 5.0,
    training_shift: float = 1.0,
    num_train_timesteps: int = 1000,
) -> torch.Tensor:
    if num_inference_steps < 1 or shift <= 0 or training_shift <= 0:
        raise ValueError("Steps and shifts must be positive.")

    total = int(num_train_timesteps)

    def _grid(count: int, s: float) -> torch.Tensor:
        lower = s / (total + s - 1.0)
        values = torch.linspace(1.0, lower, count, dtype=torch.float32)
        return s * values / (1.0 + (s - 1.0) * values)

    sigmas = _grid(num_inference_steps, float(shift))
    training = _grid(total, float(training_shift)).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where(
        (sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(),
        lower,
        upper,
    )
    return torch.unique_consecutive(training[nearest])


class MOSSImageScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        training_shift: float = 1.0,
    ):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.training_shift = float(training_shift)
        self.timesteps: Optional[torch.Tensor] = None
        self.sigmas: Optional[torch.Tensor] = None
        self._step_index: Optional[int] = None

    def reset(self) -> None:
        self._step_index = 0

    def set_timesteps(
        self,
        num_inference_steps: int = 50,
        device: Optional[Union[str, torch.device]] = None,
        shift: Optional[float] = None,
    ) -> None:
        if shift is not None:
            self.shift = float(shift)
        target_device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        snapped = compute_moss_v2_sigmas(
            num_inference_steps=num_inference_steps,
            shift=self.shift,
            training_shift=self.training_shift,
            num_train_timesteps=self.num_train_timesteps,
        ).to(device=target_device)

        self.timesteps = snapped * float(self.num_train_timesteps)
        self.sigmas = torch.cat(
            [snapped, torch.zeros(1, device=target_device, dtype=torch.float32)]
        )
        self._step_index = 0

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = False,
        history_output: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        num_steps = len(self.timesteps)
        if self._step_index is None or self._step_index >= num_steps:
            self._step_index = 0

        idx = self._step_index
        dt = self.sigmas[idx + 1] - self.sigmas[idx]
        prev_sample = sample.float() + dt * model_output.float()

        self._step_index += 1
        if self._step_index >= num_steps:
            self._step_index = None

        return (prev_sample,) if return_dict else prev_sample


class BifurcatedFlowMatchScheduler:
    def __init__(
        self,
        early_solver: str = "euler",
        late_solver: str = "euler",
        handoff_threshold: float = 0.35,
        shift: float = 5.0,
        training_shift: float = 1.0,
        num_train_timesteps: int = 1000,
        ipndm_max_order: int = 4,
        eta: float = 0.0,
        s_noise: float = 1.0,
        use_v2_grid: bool = True,
    ):
        self.early_solver = str(early_solver).lower()
        self.late_solver = str(late_solver).lower()
        self.handoff_threshold = max(0.0, min(1.0, float(handoff_threshold)))
        self.shift = float(shift)
        self.training_shift = float(training_shift)
        self.num_train_timesteps = int(num_train_timesteps)
        self.ipndm_max_order = max(1, min(4, int(ipndm_max_order)))
        self.eta = max(0.0, min(1.0, float(eta)))
        self.s_noise = max(0.0, float(s_noise))
        self.use_v2_grid = bool(use_v2_grid)

        self.base_sigmas: Optional[torch.Tensor] = None
        self.sigmas: Optional[torch.Tensor] = None
        self.timesteps: Optional[torch.Tensor] = None
        self._step_configs: List[Dict[str, Any]] = []
        self._step_index: Optional[int] = None
        self._k_star: int = 0
        self._sample_anchor: Optional[torch.Tensor] = None
        self._v0: Optional[torch.Tensor] = None
        self._v0_hist: Optional[torch.Tensor] = None
        self._h: Optional[float] = None
        self._history: List[torch.Tensor] = []
        self._boundary_flushed: bool = False
        self._can_warm_handoff: bool = False
        self.has_corrector: bool = False
        self._early_steps: int = 0
        self._late_steps: int = 0

    def reset(self) -> None:
        self._step_index = 0
        self._sample_anchor = None
        self._v0 = None
        self._v0_hist = None
        self._h = None
        self._history.clear()
        self._boundary_flushed = False

    @property
    def current_is_early(self) -> bool:
        if self._step_index is None or self._step_index >= len(self._step_configs):
            return True
        return bool(self._step_configs[self._step_index]["is_early"])

    @property
    def wants_early_history_output(self) -> bool:
        if (
            not self._can_warm_handoff
            or self._step_index is None
            or self._step_index >= len(self._step_configs)
        ):
            return False
        cfg = self._step_configs[self._step_index]
        return bool(cfg["is_early"] and cfg["mode"] in ("predictor", "multistep"))

    @property
    def current_progress(self) -> float:
        if self._step_index is None or not self._step_configs:
            return 0.0
        cfg = self._step_configs[min(self._step_index, len(self._step_configs) - 1)]
        return float(max(0.0, min(1.0, 1.0 - float(cfg["s_curr"]))))

    @staticmethod
    def _lagrange_ab_weights_for_order(
        h_n: float,
        nodes: Sequence[float],
        m: int,
    ) -> Tuple[float, float, float, float]:
        if m <= 1:
            return (h_n, 0.0, 0.0, 0.0)
        active_nodes = nodes[:m]
        for a in range(m):
            for b in range(a + 1, m):
                if abs(active_nodes[a] - active_nodes[b]) < 1e-12:
                    return (h_n, 0.0, 0.0, 0.0)

        def _eval_basis(k: int, x: float) -> float:
            xk = active_nodes[k]
            num = 1.0
            den = 1.0
            for j in range(m):
                if j != k:
                    xj = active_nodes[j]
                    num *= x - xj
                    den *= xk - xj
            return num / den

        mid = 0.5 * h_n
        scale = h_n / 6.0
        w = [0.0, 0.0, 0.0, 0.0]
        for k in range(m):
            f0 = 1.0 if k == 0 else 0.0
            f_mid = _eval_basis(k, mid)
            f_end = _eval_basis(k, h_n)
            w[k] = scale * (f0 + 4.0 * f_mid + f_end)
        return (w[0], w[1], w[2], w[3])

    @classmethod
    def _compute_ab_weights(
        cls,
        h_n: float,
        h_n_1: float,
        h_n_2: float,
        h_n_3: float,
        order: int,
    ) -> Tuple[float, float, float, float]:
        if order <= 1 or abs(h_n) < 1e-12 or abs(h_n_1) < 1e-12:
            return (h_n, 0.0, 0.0, 0.0)

        max_m = min(int(order), 4)
        if max_m >= 2 and abs(h_n_1) < 1e-12:
            max_m = 1
        if max_m >= 3 and abs(h_n_2) < 1e-12:
            max_m = 2
        if max_m >= 4 and abs(h_n_3) < 1e-12:
            max_m = 3

        x0 = 0.0
        x1 = -h_n_1
        x2 = -(h_n_1 + h_n_2)
        x3 = -(h_n_1 + h_n_2 + h_n_3)
        nodes = (x0, x1, x2, x3)

        abs_h = abs(h_n)
        for m in range(max_m, 1, -1):
            cand = cls._lagrange_ab_weights_for_order(h_n, nodes, m)
            l1_sum = abs(cand[0]) + abs(cand[1]) + abs(cand[2]) + abs(cand[3])
            if l1_sum <= 2.15 * abs_h + 1e-12:
                return cand

        return (h_n, 0.0, 0.0, 0.0)

    def set_timesteps(
        self,
        num_inference_steps: int = 50,
        device: Optional[Union[str, torch.device]] = None,
        shift: Optional[float] = None,
        handoff_threshold: Optional[float] = None,
        start_step: int = 0,
    ) -> None:
        if shift is not None:
            self.shift = float(shift)
        if handoff_threshold is not None:
            self.handoff_threshold = max(0.0, min(1.0, float(handoff_threshold)))

        target_device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        if self.use_v2_grid:
            snapped = compute_moss_v2_sigmas(
                num_inference_steps=num_inference_steps,
                shift=self.shift,
                training_shift=self.training_shift,
                num_train_timesteps=self.num_train_timesteps,
            )
            sigmas_np = snapped.numpy().astype(np.float64)
        else:
            raw = np.linspace(
                1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=np.float64
            )
            sigmas_np = self.shift * raw / (1.0 + (self.shift - 1.0) * raw)

        if 0 < start_step < len(sigmas_np):
            sigmas_np = sigmas_np[start_step:]

        base_sigmas_np = np.append(sigmas_np, 0.0)
        self.base_sigmas = torch.from_numpy(base_sigmas_np.astype(np.float32)).to(
            device=target_device
        )

        active_steps = len(sigmas_np)
        if self.handoff_threshold >= 1.0:
            k_star = active_steps
        elif self.handoff_threshold <= 0.0:
            k_star = 0
        else:
            cand = [
                i
                for i in range(active_steps)
                if (1.0 - float(base_sigmas_np[i])) >= self.handoff_threshold - 1e-7
            ]
            k_star = cand[0] if cand else active_steps

        self._k_star = k_star
        self._early_steps = k_star
        self._late_steps = active_steps - k_star

        corrector_types = ("heun", "sde_gpu_pp", "multitree")
        early_has_corrector = self.early_solver in corrector_types
        late_has_corrector = self.late_solver in corrector_types

        self._can_warm_handoff = (
            self.eta <= 0.0
            and self.early_solver in ("heun", "ipndm", "multitree")
            and self.late_solver in ("ipndm", "multitree")
            and self.ipndm_max_order > 1
        )

        step_configs: List[Dict[str, Any]] = []
        eval_timesteps: List[float] = []

        for i in range(active_steps):
            is_early = i < k_star
            solver = self.early_solver if is_early else self.late_solver
            s_curr = float(base_sigmas_np[i])
            s_next = float(base_sigmas_np[i + 1])
            dt = s_next - s_curr

            interval_corrector = (
                (early_has_corrector if is_early else late_has_corrector)
                and (s_next > 0.0)
            )
            if interval_corrector:
                t_pred = s_curr * self.num_train_timesteps
                t_corr = s_next * self.num_train_timesteps
                eval_timesteps.extend([t_pred, t_corr])
                step_configs.append(
                    {
                        "mode": "predictor",
                        "interval_idx": i,
                        "is_early": is_early,
                        "s_curr": s_curr,
                        "s_next": s_next,
                        "dt": dt,
                        "timestep": t_pred,
                        "solver": solver,
                    }
                )
                step_configs.append(
                    {
                        "mode": "corrector",
                        "interval_idx": i,
                        "is_early": is_early,
                        "s_curr": s_curr,
                        "s_next": s_next,
                        "dt": dt,
                        "timestep": t_corr,
                        "solver": solver,
                    }
                )
            else:
                t_step = s_curr * self.num_train_timesteps
                eval_timesteps.append(t_step)
                j = i if (is_early or self._can_warm_handoff) else (i - k_star)
                order = min(self.ipndm_max_order, j + 1)
                h_n_1 = (
                    float(base_sigmas_np[i] - base_sigmas_np[i - 1]) if j >= 1 else 0.0
                )
                h_n_2 = (
                    float(base_sigmas_np[i - 1] - base_sigmas_np[i - 2])
                    if j >= 2
                    else 0.0
                )
                h_n_3 = (
                    float(base_sigmas_np[i - 2] - base_sigmas_np[i - 3])
                    if j >= 3
                    else 0.0
                )
                weights = self._compute_ab_weights(dt, h_n_1, h_n_2, h_n_3, order)
                step_configs.append(
                    {
                        "mode": "multistep",
                        "interval_idx": i,
                        "is_early": is_early,
                        "s_curr": s_curr,
                        "s_next": s_next,
                        "dt": dt,
                        "timestep": t_step,
                        "solver": solver,
                        "order": order,
                        "weights": weights,
                    }
                )

        self._step_configs = step_configs
        self.timesteps = torch.tensor(
            eval_timesteps, dtype=torch.float32, device=target_device
        )
        self.sigmas = self.base_sigmas
        self.has_corrector = any(cfg["mode"] == "corrector" for cfg in step_configs)
        self.reset()

    def _push_history(self, v: torch.Tensor) -> None:
        if len(self._history) >= self.ipndm_max_order - 1:
            self._history.pop(0)
        self._history.append(v.detach())

    def _calc_sde_dispersion(
        self,
        ref_tensor: torch.Tensor,
        v_tot: torch.Tensor,
        s_curr: float,
        s_next: float,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if self.eta <= 0.0 or self.s_noise <= 0.0 or self.handoff_threshold <= 0.0:
            return torch.zeros_like(ref_tensor)
        progress = self.current_progress
        if progress >= self.handoff_threshold or s_next <= 0.0:
            return torch.zeros_like(ref_tensor)

        envelope = max(
            0.0,
            min(1.0, (self.handoff_threshold - progress) / self.handoff_threshold),
        )
        dt_val = abs(s_next - s_curr)
        max_rho = 0.5 * s_next
        renoise = min(
            max_rho,
            self.eta * s_next * math.sqrt(max(dt_val, 1e-8)) * envelope * self.s_noise,
        )
        if renoise <= 0.0:
            return torch.zeros_like(ref_tensor)

        gen_dev = generator.device.type if generator is not None else None
        if generator is not None and gen_dev == ref_tensor.device.type:
            noise = torch.randn(
                ref_tensor.shape,
                generator=generator,
                device=ref_tensor.device,
                dtype=torch.float32,
            )
        else:
            noise = torch.randn(
                ref_tensor.shape, device=ref_tensor.device, dtype=torch.float32
            )

        eps_hat = ref_tensor + (1.0 - s_curr) * v_tot
        attenuation = (
            math.sqrt(max(0.0, s_next * s_next - renoise * renoise)) - s_next
        )
        return (attenuation * eps_hat) + (noise * renoise)

    def step(
        self,
        model_output: Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]],
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = False,
        history_output: Optional[
            Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]]
        ] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        num_passes = len(self._step_configs)
        if self._step_index is None or self._step_index >= num_passes:
            self.reset()

        v = (
            model_output[0].float()
            if isinstance(model_output, (tuple, list))
            else model_output.float()
        )
        if history_output is not None:
            v_hist = (
                history_output[0].float()
                if isinstance(history_output, (tuple, list))
                else history_output.float()
            )
        else:
            v_hist = v
        sample_f32 = sample.float()

        cfg = self._step_configs[self._step_index]
        mode = cfg["mode"]
        s_curr = cfg["s_curr"]
        s_next = cfg["s_next"]
        dt = cfg["dt"]
        is_early = cfg["is_early"]
        solver = cfg["solver"]

        if not is_early and not self._boundary_flushed:
            if not self._can_warm_handoff:
                self._history.clear()
            self._boundary_flushed = True

        if mode == "predictor":
            self._sample_anchor = sample_f32.clone()
            self._v0 = v.detach()
            self._v0_hist = v_hist.detach()
            self._h = dt
            prev_sample = sample_f32 + dt * v
        elif mode == "corrector":
            sample_0 = (
                self._sample_anchor if self._sample_anchor is not None else sample_f32
            )
            v0 = self._v0 if self._v0 is not None else v
            v0_hist = self._v0_hist if self._v0_hist is not None else v0
            h_dt = self._h if self._h is not None else dt

            if solver in ("heun", "sde_gpu_pp", "multitree"):
                dx = (h_dt / 2.0) * (v0 + v)
            else:
                dx = h_dt * v0

            if is_early and self.eta > 0.0 and s_next > 0.0:
                v_tot = 0.5 * (v0 + v)
                disp = self._calc_sde_dispersion(
                    sample_0, v_tot, s_curr, s_next, generator=generator
                )
                prev_sample = sample_0 + dx + disp
            else:
                prev_sample = sample_0 + dx

            if is_early and self._can_warm_handoff:
                self._push_history(v0_hist)

            self._sample_anchor = None
            self._v0 = None
            self._v0_hist = None
            self._h = None
        else:
            c0, c1, c2, c3 = cfg["weights"]
            if solver in ("ipndm", "multitree"):
                dx = c0 * v
                if len(self._history) >= 1 and c1 != 0.0:
                    dx = dx + c1 * self._history[-1]
                if len(self._history) >= 2 and c2 != 0.0:
                    dx = dx + c2 * self._history[-2]
                if len(self._history) >= 3 and c3 != 0.0:
                    dx = dx + c3 * self._history[-3]
            else:
                dx = dt * v

            if is_early and self.eta > 0.0 and s_next > 0.0:
                disp = self._calc_sde_dispersion(
                    sample_f32, v, s_curr, s_next, generator=generator
                )
                prev_sample = sample_f32 + dx + disp
            else:
                prev_sample = sample_f32 + dx

            if solver in ("ipndm", "multitree") and self.ipndm_max_order > 1:
                self._push_history(v_hist if is_early else v)

        self._step_index += 1
        if self._step_index >= num_passes:
            self.reset()

        return (prev_sample,) if return_dict else prev_sample


__all__ = [
    "compute_moss_v2_sigmas",
    "MOSSImageScheduler",
    "BifurcatedFlowMatchScheduler",
]