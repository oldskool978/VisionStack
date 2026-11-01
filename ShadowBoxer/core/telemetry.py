# ShadowBoxer/core/telemetry.py
from __future__ import annotations

import shutil
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from . import env as _env  # noqa: F401


def _format_seconds(seconds: float) -> str:
    s = max(0, int(seconds))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class EngineProgressReporter:
    def __init__(self) -> None:
        self.current_stage: Optional[str] = None
        self.stage_start: float = time.perf_counter()
        self.last_render: float = 0.0
        self.render_interval: float = 0.08
        self.last_count: int = 0
        self.stage_durations: Dict[str, float] = {}

    def start_stage(self, stage: str, total: int) -> None:
        now = time.perf_counter()
        if self.current_stage is not None and self.current_stage != stage:
            self._finish_stage(now)
        self.current_stage = stage
        self.stage_start = now
        self.last_render = 0.0
        self.last_count = 0
        self._render_bar(stage, 0, total, now)

    def __call__(self, stage: str, current: int, total: int) -> None:
        now = time.perf_counter()
        if stage != self.current_stage:
            if self.current_stage is not None:
                self._finish_stage(now)
            self.current_stage = stage
            self.stage_start = now
            self.last_render = 0.0
            self.last_count = 0

        self.last_count = current
        is_final = current >= total
        if not is_final and (now - self.last_render < self.render_interval):
            return

        self.last_render = now
        self._render_bar(stage, current, total, now)
        if is_final:
            self._finish_stage(now)

    def _render_bar(self, stage: str, current: int, total: int, now: float) -> None:
        elapsed = max(now - self.stage_start, 1e-4)
        pct = (current / max(total, 1)) * 100.0
        rate = (current / elapsed) if (elapsed >= 0.05 and current >= 1) else 0.0
        remaining_units = max(0, total - current)
        eta = (remaining_units / rate) if rate > 0.0 else 0.0
        time_str = f"[{_format_seconds(elapsed)}<{_format_seconds(eta)}"

        if stage == "context":
            prefix = "[1/3 Qwen3.5 Context]"
            rate_str = f"{rate:5.1f} seq/s" if rate > 0 else "  --.- seq/s"
            suffix = f" {pct:5.1f}% | {current:2d}/{total:2d} seq {time_str}, {rate_str}]"
        elif stage == "dit":
            prefix = "[2/3 MOSS-DiT Flow]"
            rate_str = f"{rate:5.2f} stp/s" if rate > 0 else "  --.-- stp/s"
            suffix = f" {pct:5.1f}% | {current:3d}/{total:3d} stp {time_str}, {rate_str}]"
        elif stage == "vae":
            prefix = "[3/3 Flux2-VAE FP32]"
            rate_str = f"{rate:5.2f} img/s" if rate > 0 else "  --.-- img/s"
            suffix = f" {pct:5.1f}% | {current:2d}/{total:2d} img {time_str}, {rate_str}]"
        else:
            prefix = f"[{stage.upper()}]"
            suffix = f" {pct:5.1f}% | {current}/{total} {time_str}]"

        cols = max(60, min(shutil.get_terminal_size((80, 20)).columns, 140))
        avail = cols - len(prefix) - len(suffix) - 3
        bar_len = max(8, min(avail, 30))
        filled = min(bar_len, int(bar_len * current / max(total, 1)))
        bar = "=" * filled + ">" if (filled < bar_len and current > 0) else "=" * filled
        bar = bar.ljust(bar_len, " ")
        line = f"\r{prefix} [{bar}]{suffix}"
        sys.stdout.write(line.ljust(cols - 1)[: cols - 1])
        sys.stdout.flush()

    def _finish_stage(self, now: float) -> None:
        elapsed = max(now - self.stage_start, 1e-4)
        if self.current_stage is not None:
            self.stage_durations[self.current_stage] = elapsed
        stage_names = {
            "context": "Qwen3.5 Hybrid Context",
            "dit": "MOSS-DiT Flow Matching",
            "vae": "Flux2-VAE FP32 Decode",
        }
        name = stage_names.get(
            self.current_stage,
            self.current_stage.upper() if self.current_stage else "Stage",
        )
        cols = max(60, min(shutil.get_terminal_size((80, 20)).columns, 140))
        summary = f"\r[+] {name:<26} finalized in {elapsed:6.2f}s ({self.last_count} units)."
        sys.stdout.write(summary.ljust(cols - 1)[: cols - 1] + "\n")
        sys.stdout.flush()
        self.current_stage = None

    def close(self) -> None:
        if self.current_stage is not None:
            self._finish_stage(time.perf_counter())


def compute_image_telemetry(pil_img: Image.Image) -> Tuple[float, float, float, float]:
    rgb = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    mean_luma = float(np.mean(luma))
    rms_contrast = float(np.std(luma))

    luma_u8 = np.clip(np.round(luma * 255.0), 0, 255).astype(np.uint8)
    hist = np.bincount(luma_u8.ravel(), minlength=256).astype(np.float64)
    probs = hist[hist > 0] / float(luma_u8.size)
    entropy = float(-np.sum(probs * np.log2(probs)))

    if luma.shape[0] >= 3 and luma.shape[1] >= 3:
        lap = (
            luma[:-2, 1:-1]
            + luma[2:, 1:-1]
            + luma[1:-1, :-2]
            + luma[1:-1, 2:]
            - 4.0 * luma[1:-1, 1:-1]
        )
        sharpness = float(np.var(lap) * 1000.0)
    else:
        sharpness = 0.0
    return mean_luma, rms_contrast, entropy, sharpness


__all__ = [
    "_format_seconds",
    "EngineProgressReporter",
    "compute_image_telemetry",
]