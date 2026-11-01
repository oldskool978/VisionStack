from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

sys.modules["torchvision"] = None
sys.modules["torchvision.transforms"] = None
sys.modules["torchvision.io"] = None


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


from .gate1_native import gate_1_native_subsystems
from .gate2_caches import gate_2_pinned_memory_and_cache_isolation
from .gate3_models import gate_3_models_zero_alloc_and_lora
from .gate4_schedulers import gate_4_schedulers_and_prng
from .gate5_pipeline import gate_5_refinery_schema_and_snapshots


def run_all_gates() -> float:
    t0 = time.perf_counter()
    print("=" * 92)
    print(
        "          SHADOWBOXER SOVEREIGN ENGINE CONVERGENCE & 5-GATE AUDIT SUITE"
    )
    print("=" * 92)
    gate_1_native_subsystems()
    gate_2_pinned_memory_and_cache_isolation()
    gate_3_models_zero_alloc_and_lora()
    gate_4_schedulers_and_prng()
    gate_5_refinery_schema_and_snapshots()
    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 92)
    print(
        f"[+] STATUS: SECURE, ROBUST & CONVERGED IN {elapsed:.2f}s. ZERO DRIFT ACROSS ALL 5 GATES."
    )
    print("=" * 92)
    return elapsed


__all__ = [
    "ROOT_DIR",
    "_assert",
    "gate_1_native_subsystems",
    "gate_2_pinned_memory_and_cache_isolation",
    "gate_3_models_zero_alloc_and_lora",
    "gate_4_schedulers_and_prng",
    "gate_5_refinery_schema_and_snapshots",
    "run_all_gates",
]