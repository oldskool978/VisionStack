#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

sys.modules["torchvision"] = None
sys.modules["torchvision.transforms"] = None
sys.modules["torchvision.io"] = None

from sanitycheck import (
    _assert,
    gate_1_native_subsystems,
    gate_2_pinned_memory_and_cache_isolation,
    gate_3_models_zero_alloc_and_lora,
    gate_4_schedulers_and_prng,
    gate_5_refinery_schema_and_snapshots,
    run_all_gates,
)

_GATE_DISPATCH = {
    1: gate_1_native_subsystems,
    2: gate_2_pinned_memory_and_cache_isolation,
    3: gate_3_models_zero_alloc_and_lora,
    4: gate_4_schedulers_and_prng,
    5: gate_5_refinery_schema_and_snapshots,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ShadowBoxer Sovereign Engine 5-Gate Domain Auditor"
    )
    parser.add_argument(
        "--gate",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="Execute a single domain gate (1..5) or all 5 gates by default.",
    )
    args = parser.parse_args()
    if args.gate is not None:
        t0 = time.perf_counter()
        _GATE_DISPATCH[args.gate]()
        elapsed = time.perf_counter() - t0
        print(f"\n[+] Gate {args.gate} converged in {elapsed:.2f}s.")
        return
    run_all_gates()


__all__ = [
    "_assert",
    "gate_1_native_subsystems",
    "gate_2_pinned_memory_and_cache_isolation",
    "gate_3_models_zero_alloc_and_lora",
    "gate_4_schedulers_and_prng",
    "gate_5_refinery_schema_and_snapshots",
    "run_all_gates",
    "main",
]

if __name__ == "__main__":
    main()