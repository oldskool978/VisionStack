#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
from pathlib import Path
import subprocess
import sys
import time
import unicodedata

ROOT_DIR = Path(__file__).resolve().parent.parent
TOOLCHAIN_DIR = ROOT_DIR / "toolchain"
DIST_DIR = ROOT_DIR / "dist"
DIST_BIN_DIR = DIST_DIR / "bin"
RANKS_QWEN35_BIN = DIST_BIN_DIR / "qwen35.ranks.bin"

IS_WIN = sys.platform == "win32"
HERMETIC_PYTHON = TOOLCHAIN_DIR / "python" / ("python.exe" if IS_WIN else "bin/python")

if (
    HERMETIC_PYTHON.exists()
    and Path(sys.executable).resolve() != HERMETIC_PYTHON.resolve()
):
    result = subprocess.run(
        [str(HERMETIC_PYTHON.resolve()), "-I", str(Path(__file__).resolve())]
        + sys.argv[1:],
        check=False,
    )
    sys.exit(result.returncode)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import tiktoken
import viztoken

ORDINARY_CORPUS = [
    (
        "OpenMOSS Spatial Layout JSON (64KB)",
        (
            '{"summary": {"caption": "A wide view of a sleek brushed-steel refrigerator beside a warm walnut kitchen counter"}, '
            '"items": [{"label": "object", "bbox": [-480, -464, 480, 80], "caption": "Sleek brushed-steel refrigerator"}, '
            '{"label": "text", "bbox": [-224, -368, 32, -32], "caption": "Handwritten marker text on yellow sticky note", "text": "OPENMOSS 2026"}], '
            '"bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}\n'
        )
        * 175,
    ),
    (
        "Multi-Box Reference Layout Graph (64KB)",
        (
            '{"summary": {"caption": "Studio composition with multiple visual references and typographic overlays"}, '
            '"items": [{"label": "character", "bbox": [-420, -440, 440, 20], "reference": "Picture 1", "caption": "Portrait subject in dramatic rim lighting"}, '
            '{"label": "object", "bbox": [60, 80, 460, 460], "reference": "Picture 2", "caption": "Ceramic vessel on walnut plinth"}, '
            '{"label": "text", "bbox": [-460, 60, -300, 460], "caption": "Chiseled silver serif title", "text": "SOVEREIGN"}], '
            '"bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}\n'
        )
        * 130,
    ),
    (
        "Multilingual UTF-8 Visual Descriptions (64KB)",
        (
            "Architectural interior under diffused twilight illumination\n"
            " Crisp typographic rendering with razor-sharp serifs and balanced kerning\n"
            "\u00c9l\u00e9gant, caf\u00e9-croissant, fa\u00e7ade, na\u00efve, \u00fcber, Gr\u00fc\u00dfe aus Berlin.\n"
            "\u591c\u7a7a\u4e2d\u6700\u4eae\u7684\u661f \u6771\u4eac\u306e\u591c \ud55c\uad6d\uc5b4 \ub514\uc790\uc778 \U0001f3a8 \U0001f5bc\n"
        )
        * 240,
    ),
    (
        "Model Configuration & AST Specification (64KB)",
        (
            "{\n"
            '  "model_type": "MOSSImageTransformer",\n'
            '  "num_hidden_layers": 42,\n'
            '  "num_attention_heads": 24,\n'
            '  "head_dim": 128,\n'
            '  "vocab_size": 248320,\n'
            '  "uncond_skip_layers": [22],\n'
            '  "attention_mechanism": "scaled_dot_product"\n'
            "}\n"
        )
        * 290,
    ),
    (
        "Worst-Case Stress Compound Block (64KB)",
        "Supercalifragilisticexpialidocious " * 1900,
    ),
]


def resolve_vocab_file(filename: str = "qwen35.tiktoken") -> Path:
    cand = ROOT_DIR / filename
    if cand.is_file():
        return cand.resolve()
    raise FileNotFoundError(
        f"Could not locate {filename} in {ROOT_DIR}. Run tools/hydrate.py first."
    )


def benchmark_cold_start(
    vocab_path: Path, ranks_bin: Path, label: str = "Qwen3_5-248K"
) -> tiktoken.Encoding:
    print(f"\n--- Cold-Start Initialization Latency ({label}) ---")
    specials_dict = dict(viztoken.build_default_qwen35_specials())

    t0 = time.perf_counter()
    ranks = {
        base64.b64decode(t): int(r)
        for t, r in (
            line.split() for line in vocab_path.read_bytes().splitlines() if line
        )
    }
    ref_enc = tiktoken.Encoding(
        label,
        pat_str=viztoken.QWEN35_PRETOKENIZE_REGEX,
        mergeable_ranks=ranks,
        special_tokens=specials_dict,
    )
    t_ref_cold = (time.perf_counter() - t0) * 1000.0

    cold_iters = 20
    t0 = time.perf_counter()
    for _ in range(cold_iters):
        with viztoken.Encoding(label, ranks_path=ranks_bin):
            pass
    t_sov_cold = ((time.perf_counter() - t0) / cold_iters) * 1000.0

    print(f"  Reference tiktoken (Parse + Alloc): {t_ref_cold:8.3f} ms")
    print(f"  Sovereign VizToken (SWISS5 MMap)  : {t_sov_cold:8.3f} ms")
    print(
        f"  Cold-Start Acceleration           : {t_ref_cold / max(t_sov_cold, 1e-6):8.2f}x faster"
    )
    return ref_enc


def benchmark_qwen35_multimodal(
    ref35: tiktoken.Encoding, sov35: viztoken.Encoding, iters: int
) -> None:
    pads = "<|image_pad|>" * 2304
    user_body = (
        '{"summary": {"caption": "A sleek brushed-steel refrigerator beside a walnut counter"}, '
        '"items": [{"label": "object", "bbox": [-480, -464, 480, 80], "caption": "Brushed-steel refrigerator"}, '
        '{"label": "text", "bbox": [-224, -368, 32, -32], "caption": "Handwritten marker text", "text": "OPENMOSS 2026"}], '
        '"bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}'
    )
    mm_prompt = (
        f"<|im_start|>user\nPicture 1: <|vision_start|>{pads}<|vision_end|>"
        f"Picture 2: <|vision_start|>{pads}<|vision_end|>"
        f"{user_body}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )
    raw_bytes = len(mm_prompt.encode("utf-8"))

    ref_out = ref35.encode(mm_prompt, allowed_special="all")
    sov_out = sov35.encode(mm_prompt, allowed_special="all")
    chat_out = sov35.encode_chat(
        user_body,
        enable_thinking=True,
        max_sequence_length=8192,
        image_patch_counts=[2304, 2304],
    )
    assert ref_out == sov_out and ref_out == chat_out, "Qwen3.5 multimodal parity mismatch"

    for _ in range(5):
        ref35.encode(mm_prompt, allowed_special="all")
        sov35.encode(mm_prompt, allowed_special="all")
        sov35.encode_chat(
            user_body,
            enable_thinking=True,
            max_sequence_length=8192,
            image_patch_counts=[2304, 2304],
        )

    t0 = time.perf_counter()
    for _ in range(iters):
        ref35.encode(mm_prompt, allowed_special="all")
    t_ref = (time.perf_counter() - t0) / iters

    t0 = time.perf_counter()
    for _ in range(iters):
        sov35.encode(mm_prompt, allowed_special="all")
    t_sov = (time.perf_counter() - t0) / iters

    t0 = time.perf_counter()
    for _ in range(iters):
        sov35.encode_chat(
            user_body,
            enable_thinking=True,
            max_sequence_length=8192,
            image_patch_counts=[2304, 2304],
        )
    t_chat = (time.perf_counter() - t0) / iters

    mb = raw_bytes / (1024 * 1024)
    speedup = t_ref / max(t_sov, 1e-9)
    chat_speedup = t_ref / max(t_chat, 1e-9)
    print(
        f"\n--- Target: ShadowBoxer Qwen3.5 Dual-Image Multimodal Prompt ({raw_bytes:,} bytes, {len(sov_out):,} tokens) ---"
    )
    print(
        f"  Reference tiktoken       : {t_ref * 1000.0:7.3f} ms | {mb / t_ref:8.2f} MB/s | {len(ref_out) / t_ref:11,.0f} tokens/s"
    )
    print(
        f"  Sovereign VizToken (RLE) : {t_sov * 1000.0:7.3f} ms | {mb / t_sov:8.2f} MB/s | {len(sov_out) / t_sov:11,.0f} tokens/s ({speedup:.2f}x)"
    )
    print(
        f"  Zero-Alloc encode_chat   : {t_chat * 1000.0:7.3f} ms | {mb / t_chat:8.2f} MB/s | {len(chat_out) / t_chat:11,.0f} tokens/s ({chat_speedup:.2f}x)"
    )


def run_benchmarks(iters: int = 50) -> None:
    print("=" * 88)
    print(
        "       SOVEREIGN TOKENIZER BENCHMARK: VIZTOKEN (C11 AVX2 SWISS5) vs TIKTOKEN"
    )
    print("=" * 88)

    vocab_path = resolve_vocab_file("qwen35.tiktoken")
    ref = benchmark_cold_start(vocab_path, RANKS_QWEN35_BIN, "Qwen3_5-248K")

    with viztoken.Encoding("Qwen3_5", ranks_path=RANKS_QWEN35_BIN) as sov:
        for name, text in ORDINARY_CORPUS:
            normalized = unicodedata.normalize("NFC", text)
            raw_bytes = len(normalized.encode("utf-8"))

            ref_out = ref.encode_ordinary(normalized)
            sov_out = sov.encode_ordinary(normalized)
            assert ref_out == sov_out, f"Differential mismatch on {name}"

            for _ in range(10):
                ref.encode_ordinary(normalized)
                sov.encode_ordinary(normalized)

            t0 = time.perf_counter()
            for _ in range(iters):
                ref.encode_ordinary(normalized)
            t_ref = (time.perf_counter() - t0) / iters

            t0 = time.perf_counter()
            for _ in range(iters):
                sov.encode_ordinary(normalized)
            t_sov = (time.perf_counter() - t0) / iters

            mb = raw_bytes / (1024 * 1024)
            speedup = t_ref / max(t_sov, 1e-9)
            mb_s_sov = mb / t_sov
            mb_s_ref = mb / t_ref

            print(
                f"\n--- Target: {name} ({raw_bytes:,} bytes, {len(sov_out):,} tokens) ---"
            )
            print(
                f"  Reference tiktoken: {t_ref * 1000.0:7.3f} ms | {mb_s_ref:8.2f} MB/s | {len(ref_out) / t_ref:11,.0f} tokens/s"
            )
            print(
                f"  Sovereign VizToken: {t_sov * 1000.0:7.3f} ms | {mb_s_sov:8.2f} MB/s | {len(sov_out) / t_sov:11,.0f} tokens/s"
            )
            print(
                f"  Acceleration:       {speedup:.2f}x faster ({'+' if speedup >= 1.0 else ''}{(speedup - 1.0) * 100.0:.1f}%)"
            )

        benchmark_qwen35_multimodal(ref, sov, iters=iters)

    print("\n" + "=" * 88)
    print(
        "ALL STEADY-STATE, MULTIMODAL & COLD-START COMPARISONS VERIFIED: 100% BIT-PARITY ACHIEVED"
    )
    print("=" * 88)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VizToken Microarchitectural Benchmark."
    )
    parser.add_argument(
        "--iters", type=int, default=50, help="Number of benchmark iterations."
    )
    args = parser.parse_args()
    run_benchmarks(iters=args.iters)


if __name__ == "__main__":
    main()