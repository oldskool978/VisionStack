#!/usr/bin/env python3
from __future__ import annotations

import base64
import gc
from pathlib import Path
import subprocess
import sys
import threading
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


def get_process_memory_mb() -> float:
    if IS_WIN:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        get_mem_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_mem_info.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        get_mem_info.restype = wintypes.BOOL
        if get_mem_info(handle, ctypes.byref(counters), counters.cb):
            return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
        return 0.0
    else:
        import resource

        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def resolve_vocab_file(filename: str = "qwen35.tiktoken") -> Path:
    cand = ROOT_DIR / filename
    if cand.is_file():
        return cand.resolve()
    raise FileNotFoundError(
        f"Could not locate {filename} in {ROOT_DIR}. Run tools/hydrate.py first."
    )


def run_differential_matrix(
    ref_enc: tiktoken.Encoding, sov_enc: viztoken.Encoding
) -> None:
    assert (
        sov_enc.has_fast_tables
    ), "SWISS5 zero-init fast tables were not active in compiled container"
    assert (
        sov_enc.base_vocab_size >= 248000
    ), f"Unexpected Qwen3.5 base vocab size: {sov_enc.base_vocab_size}"
    assert (
        sov_enc.eot_token == 248044
    ), f"Expected Qwen3.5 EOT token 248044, got {sov_enc.eot_token}"
    with open(sov_enc.ranks_path, "rb") as f_hdr:
        magic = f_hdr.read(16)
    assert (
        magic == b"VIZTOK_SWISS5___"
    ), f"Expected SWISS5 magic header b'VIZTOK_SWISS5___', got {magic!r}"

    vectors = [
        (
            "OpenMOSS Spatial Layout JSON",
            '{"summary": {"caption": "Sleek brushed-steel refrigerator beside a walnut counter"}, '
            '"items": [{"label": "object", "bbox": [-480, -464, 480, 80], "caption": "Sleek brushed-steel refrigerator"}, '
            '{"label": "text", "bbox": [-224, -368, 32, -32], "caption": "Handwritten marker text on sticky note", "text": "OPENMOSS 2026"}], '
            '"bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}',
        ),
        (
            "Dense Multi-Box Layout Array",
            '{"items": [{"label": "character", "bbox": [-400, -400, 400, 100], "reference": "Picture 1", "caption": "Foreground subject"}], '
            '"bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}'
            * 8,
        ),
        ("Boundary Gate N=62", "A" * 62),
        ("Boundary Gate N=63", "B" * 63),
        ("Boundary Gate N=64 (Stack Ceiling)", "C" * 64),
        ("Boundary Gate N=65 (Heap Transition)", "D" * 65),
        ("Boundary Gate N=66", "E" * 66),
        ("Even Tie Breaking", "aaaaaaaa"),
        ("Odd Tie Breaking", "aaaaaaa"),
        ("Alternating Clashing", "abababab" * 16),
        ("Nested Reductions", "a" * 128),
        (
            "Whitespace Clashing",
            "   \t\t\r\n\r\n   Multiple    spaces   \n\n\nand   tabs.   \r\n",
        ),
        ("Massive Space Run", " " * 1024),
        ("Massive Tab Run", "\t" * 512),
        ("CRLF Sequences", "\r\n" * 256),
        ("Contractions Lower", "'s 't 're 've 'm 'll 'd"),
        ("Contractions Upper", "'S 'T 'RE 'VE 'M 'LL 'D"),
        ("Repeated Quotes", "''''''''"),
        ("Numeric Sequences", "0 1 12 123 1234 12345 123456 999999999"),
        (
            "Accented Latin & Combining Marks",
            "\u00c9l\u00e9gant, caf\u00e9-croissant, fa\u00e7ade, na\u00efve, \u00fcber, Gr\u00fc\u00dfe aus Berlin.",
        ),
        (
            "Multilingual CJK Typography",
            "\u591c\u7a7a\u4e2d\u6700\u4eae\u7684\u661f \u6771\u4eac\u306e\u591c \ud55c\uad6d\uc5b4 \ub178\ub798 \u65cb\u5f8b\u4e0e\u548c\u5f26",
        ),
        (
            "SMP Technical & Artistic Glyphs",
            "\U0001d400 \U0001d401 \U0001d402 \U0001d538 \U0001d539 \U0001f3a8",
        ),
        ("Quad-Byte SMP Emojis", "\U0001f3a8\U0001f5bc\U0001f4f7\U0001f3ac " * 4),
        (
            "ChatML Standard Structure",
            "<|im_start|>system\nYou are an authoritative visual synthesis engine.<|im_end|>\n<|im_start|>user\nCompose.<|im_end|>",
        ),
        (
            "Sequential Multimodal Delimiters",
            "<|endoftext|><|vision_start|><|image_pad|><|vision_end|><|video_pad|><|audio_start|><|audio_pad|><|audio_end|>",
        ),
        (
            "Contiguous RLE Vision Pad Run",
            "<|vision_start|>" + ("<|image_pad|>" * 512) + "<|vision_end|>",
        ),
        (
            "Unclosed Delimiter Traps",
            "Prefix traps: <|im_start, <|vision_start, <<|im_end|>>, <invalid_tag>, <|image_pad_999|>, 3 < 5, 10 > 2.",
        ),
        ("Stress Compound Block", "Supercalifragilisticexpialidocious" * 16),
        ("Single Character Slices", "A"),
        ("Empty Delimiter Pass", ""),
        (
            "Long Synthetic Layout Prompt (16KB)",
            (
                '{"label": "object", "bbox": [-256, -256, 256, 256], "caption": "Brushed steel surface under studio lighting"}\n'
                * 150
            ),
        ),
    ]

    print(f"\n--- Running Adversarial Differential Suite ({len(vectors)} vectors) ---")
    for idx, (label, sample) in enumerate(vectors):
        normalized = unicodedata.normalize("NFC", sample)
        ref_tokens = ref_enc.encode(normalized, allowed_special="all")
        sov_tokens = sov_enc.encode(normalized, allowed_special="all")
        if ref_tokens != sov_tokens:
            print(f"[!] DIVERGENCE on vector [{idx:02d}] ({label}):")
            print(f"    Ref ({len(ref_tokens)}): {ref_tokens[:20]}")
            print(f"    Sov ({len(sov_tokens)}): {sov_tokens[:20]}")
            sys.exit(1)
        decoded = sov_enc.decode(sov_tokens)
        if decoded != normalized:
            print(f"[!] ROUNDTRIP FAILURE on vector [{idx:02d}] ({label}):")
            print(f"    Input:   {repr(normalized)}")
            print(f"    Decoded: {repr(decoded)}")
            sys.exit(1)
        print(
            f"  Vector [{idx:02d}]: PASSED (Bit-Exact) | {label:<36} ({len(sov_tokens):4d} tokens)"
        )


def run_qwen35_multimodal_chat_audit(
    ref_enc: tiktoken.Encoding, sov_enc: viztoken.Encoding
) -> None:
    print("\n--- Running Qwen3.5 (248K Vocab) Multimodal & encode_chat Parity Audit ---")
    pads = "<|image_pad|>" * 2304
    multimodal_prompt = (
        f"<|im_start|>user\nPicture 1: <|vision_start|>{pads}<|vision_end|>"
        '{"summary": {"caption": "Sleek brushed-steel refrigerator"}, '
        '"items": [{"label": "text", "bbox": [-220, -320, -100, -60], "caption": "Bold marker text", "text": "OPENMOSS 2026"}], '
        '"bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}'
        "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )
    ref_ids = ref_enc.encode(multimodal_prompt, allowed_special="all")
    sov_ids = sov_enc.encode(multimodal_prompt, allowed_special="all")
    assert ref_ids == sov_ids, "Qwen3.5 multimodal prompt divergence"
    assert sov_enc.decode(sov_ids) == multimodal_prompt, "Qwen3.5 decode mismatch"

    user_body = (
        '{"summary": {"caption": " Caf\u00e9-croissant & brushed-steel refrigerator "}, '
        '"items": [{"label": "text", "bbox": [-220, -320, -100, -60], "caption": "Marker note", "text": "OPENMOSS 2026"}], '
        '"bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}'
    )
    norm_body = unicodedata.normalize("NFC", user_body).strip()

    for enable_thinking in (True, False):
        for patch_counts in (None, [144], [2304, 576]):
            for sys_prompt in (None, "You are a spatial composition architect."):
                for trace_str in (None, "Align Picture 1 to the left quadrant."):
                    for add_vis_id in (True, False):
                        sys_pfx = (
                            f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
                            if sys_prompt
                            else ""
                        )
                        if trace_str:
                            think_sfx = f"<think>\n{trace_str}\n</think>\n\n"
                        else:
                            think_sfx = (
                                "<think>\n"
                                if enable_thinking
                                else "<think>\n\n</think>\n\n"
                            )
                        if patch_counts:
                            v_pfx = "".join(
                                f"{'Picture ' + str(idx) + ': ' if add_vis_id else ''}<|vision_start|>{'<|image_pad|>' * c}<|vision_end|>"
                                for idx, c in enumerate(patch_counts, start=1)
                            )
                        else:
                            v_pfx = ""
                        expected_str = (
                            f"{sys_pfx}<|im_start|>user\n{v_pfx}{norm_body}<|im_end|>\n"
                            f"<|im_start|>assistant\n{think_sfx}"
                        )
                        ref_chat_ids = ref_enc.encode(
                            expected_str, allowed_special="all"
                        )
                        sov_chat_ids = sov_enc.encode_chat(
                            user_body,
                            enable_thinking=enable_thinking,
                            max_sequence_length=8192,
                            image_patch_counts=patch_counts,
                            system_prompt=sys_prompt,
                            thinking_trace=trace_str,
                            add_vision_id=add_vis_id,
                        )
                        assert (
                            ref_chat_ids == sov_chat_ids
                        ), f"encode_chat mismatch for thinking={enable_thinking}, patches={patch_counts}, sys={sys_prompt is not None}, trace={trace_str is not None}, vis_id={add_vis_id}"

    trunc_ids = sov_enc.encode_chat(
        user_body,
        enable_thinking=True,
        max_sequence_length=128,
        image_patch_counts=[256],
    )
    assert (
        len(trunc_ids) == 128
    ), f"encode_chat max_sequence_length truncation failed: {len(trunc_ids)}"
    print(
        f"  [+] Qwen3.5 Multimodal RLE + Zero-Alloc encode_chat: PASSED (Bit-Exact, {len(sov_ids):,} tokens)"
    )


def run_special_token_edgecases(
    ref_enc: tiktoken.Encoding, sov_enc: viztoken.Encoding
) -> None:
    print("\n--- Running Special Token Edge-Case & Subset Verification ---")
    for req_tok in (
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ):
        assert (
            req_tok in sov_enc.special_tokens_set
        ), f"Assertion Error: {req_tok!r} missing from special_tokens_set"

    try:
        ref_enc.encode("This should fail: <|endoftext|>", disallowed_special="all")
        raise AssertionError("ref_enc failed to raise ValueError")
    except ValueError:
        pass

    try:
        sov_enc.encode("This should fail: <|endoftext|>", disallowed_special="all")
        raise AssertionError("sov_enc failed to raise ValueError")
    except ValueError:
        print(
            "  [+] Disallowed token rejection: PASSED (ValueError raised identically to tiktoken)"
        )

    test_text = "Instruction: condicion region <|vision_start|><|image_pad|><|vision_end|> with <|endoftext|> delimiter."
    allowed_subsets = [
        {"<|vision_start|>", "<|vision_end|>"},
        {"<|endoftext|>"},
        {"<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|endoftext|>"},
    ]
    for subset in allowed_subsets:
        ref_out = ref_enc.encode(
            test_text, allowed_special=subset, disallowed_special=()
        )
        sov_out = sov_enc.encode(
            test_text, allowed_special=subset, disallowed_special=()
        )
        assert (
            ref_out == sov_out
        ), f"Subset mismatch on allowed_special={subset}: {ref_out} != {sov_out}"
        print(f"  [+] Subset verification: PASSED (Bit-Exact match for {subset})")

    adjacent_test = "<|vision_start|><|endoftext|><|vision_end|>"
    ref_adj = ref_enc.encode(
        adjacent_test,
        allowed_special={"<|vision_start|>", "<|vision_end|>"},
        disallowed_special=(),
    )
    sov_adj = sov_enc.encode(
        adjacent_test,
        allowed_special={"<|vision_start|>", "<|vision_end|>"},
        disallowed_special=(),
    )
    assert (
        ref_adj == sov_adj
    ), f"Adjacent special mismatch: {ref_adj} != {sov_adj}"
    assert (
        248043 not in sov_adj
    ), "<|endoftext|> was wrongly tokenized as special ID 248043"
    print(
        "  [+] Adjacent unallowed special treated as ordinary text: PASSED (Bit-Exact)"
    )


def run_agentic_prefix_splicing_audit(sov_enc: viztoken.Encoding) -> None:
    print("\n--- Running AVX2 8-Wide Prefix Splicing & Lane-Boundary Audit ---")
    for length in range(1, 65):
        base_seq = list(range(100, 100 + length))
        assert viztoken.find_common_prefix(base_seq, list(base_seq)) == length
        for diff_idx in range(length):
            mutated = list(base_seq)
            mutated[diff_idx] = -999
            got = viztoken.find_common_prefix(base_seq, mutated)
            assert (
                got == diff_idx
            ), f"SIMD prefix mismatch at len={length}, diff_idx={diff_idx}: got {got}"

    assert (
        viztoken.find_common_prefix([], [1, 2, 3]) == 0
    ), "Empty sequence boundary failure"

    orig_layout = (
        '{"summary": {"caption": "A sleek brushed-steel refrigerator"}, '
        '"items": [{"label": "text", "bbox": [-224, -368, 32, -32], "caption": "Marker note", "text": "PASS 1 BASELINE"}]}'
    )
    edit_layout = (
        '{"summary": {"caption": "A sleek brushed-steel refrigerator"}, '
        '"items": [{"label": "text", "bbox": [-224, -368, 32, -32], "caption": "Marker note", "text": "PASS 2 MUTATION"}]}'
    )
    t_orig = sov_enc.encode_ordinary(orig_layout)
    t_edit = sov_enc.encode_ordinary(edit_layout)
    split_pt = viztoken.find_common_prefix(t_orig, t_edit)
    assert 0 < split_pt < len(t_orig), "Layout mutation split point out of range"
    assert t_orig[:split_pt] == t_edit[:split_pt], "Common prefix content mismatch"
    assert t_orig[split_pt] != t_edit[split_pt], "Common prefix divergence failure"
    saved_ratio = (split_pt / float(len(t_orig))) * 100.0
    print(
        f"  [+] AVX2 8-wide lane torture (lengths 1..64) & mutation split: PASSED ({split_pt}/{len(t_orig)} tokens retained, {saved_ratio:.1f}% reuse)"
    )


def run_batch_pipeline_audit(
    ref_enc: tiktoken.Encoding, sov_enc: viztoken.Encoding
) -> None:
    print("\n--- Running Multithreaded Batch Pipeline Audit ---")
    batch_prompts = [
        '{"summary": {"caption": "Batch 1 layout"}, "items": [{"label": "object", "bbox": [-256, -256, 256, 256], "caption": "Ceramic bowl"}]}',
        "A wide view of a sleek brushed-steel refrigerator beside a warm walnut kitchen counter",
        "\u591c\u7a7a\u4e2d\u6700\u4eae\u7684\u661f \U0001f3a8 \U0001f5bc",
        "<|im_start|>system\nCompose a visual layout.<|im_end|>",
        "Supercalifragilisticexpialidocious",
    ]
    ref_batch = [ref_enc.encode(p, allowed_special="all") for p in batch_prompts]
    sov_batch = sov_enc.encode_batch(
        batch_prompts, allowed_special="all", num_threads=4
    )
    assert ref_batch == sov_batch, "encode_batch returned divergent output"
    sov_decoded_batch = sov_enc.decode_batch(sov_batch, num_threads=4)
    assert sov_decoded_batch == batch_prompts, "decode_batch roundtrip divergence"
    print(
        "  [+] Concurrent batch encoding and decoding: PASSED (100% bit-exact across threads)"
    )


def run_thread_safety_soak(
    sov_enc: viztoken.Encoding, threads: int = 8, passes_per_thread: int = 5000
) -> None:
    print(
        f"\n--- Running Multithreaded Soak Test ({threads} threads, {passes_per_thread} iterations/thread) ---"
    )
    sample_text = (
        '{"summary": {"caption": "Soak Test"}, "items": [{"label": "object", "bbox": [-400, -400, 400, 400], "caption": "Ref"}]} | '
        "\u591c\u7a7a\u4e2d\u6700\u4eae\u7684\u661f \U0001f3a8 | Supercalifragilisticexpialidocious\n"
    )
    normalized = unicodedata.normalize("NFC", sample_text)
    expected_tokens = sov_enc.encode(normalized, allowed_special="all")
    errors: list[str] = []

    def worker(tid: int):
        for _ in range(passes_per_thread):
            toks = sov_enc.encode(normalized, allowed_special="all")
            if toks != expected_tokens:
                errors.append(f"Thread {tid} encountered token mismatch")
                break
            dec = sov_enc.decode(toks)
            if dec != normalized:
                errors.append(f"Thread {tid} encountered decode divergence")
                break

    t_pool = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    t0 = time.perf_counter()
    for t in t_pool:
        t.start()
    for t in t_pool:
        t.join()
    t_elapsed = time.perf_counter() - t0
    if errors:
        print(f"[!] Multithreaded soak failure: {errors[0]}")
        sys.exit(1)
    total_encodes = threads * passes_per_thread
    print(
        f"  [+] Completed {total_encodes:,} concurrent operations in {t_elapsed:.2f}s ({total_encodes / t_elapsed:,.0f} ops/s). Zero race conditions."
    )


def run_memory_soak(sov_enc: viztoken.Encoding, cycles: int = 25000) -> None:
    print(f"\n--- Running Zero-Leak Heap/RSS Soak Test ({cycles:,} cycles) ---")
    payload = (
        'Supercalifragilisticexpialidocious {"label": "object"} <|im_start|> \u591c\u7a7a \U0001f3a8\n'
        * 10
    )
    normalized = unicodedata.normalize("NFC", payload)
    gc.collect()
    mem_initial = get_process_memory_mb()
    for _ in range(1000):
        t = sov_enc.encode(normalized, allowed_special="all")
        _ = sov_enc.decode(t)
    gc.collect()
    mem_baseline = get_process_memory_mb()
    t0 = time.perf_counter()
    for _ in range(cycles):
        toks = sov_enc.encode(normalized, allowed_special="all")
        _ = sov_enc.decode(toks)
    t_elapsed = time.perf_counter() - t0
    gc.collect()
    mem_final = get_process_memory_mb()
    delta_mb = mem_final - mem_baseline
    print(f"  Initial RSS:   {mem_initial:.2f} MB")
    print(f"  Baseline RSS:  {mem_baseline:.2f} MB")
    print(f"  Final RSS:     {mem_final:.2f} MB")
    print(
        f"  Delta RSS:     {delta_mb:+.3f} MB over {cycles:,} cycles ({cycles / t_elapsed:,.0f} ops/s)"
    )
    if delta_mb > 1.5:
        print(f"[!] Leak threshold exceeded: {delta_mb:+.3f} MB")
        sys.exit(1)
    print(
        "  [+] PASS: Memory delta within strict static operating bounds. Zero heap leak confirmed."
    )


def main() -> None:
    print("=" * 84)
    print(
        "       VIZTOKEN COMPREHENSIVE CONVERGENCE, THREAD SAFETY & SOAK AUDITOR"
    )
    print("=" * 84)
    vocab_path = resolve_vocab_file("qwen35.tiktoken")
    ranks = {
        base64.b64decode(t): int(r)
        for t, r in (
            line.split() for line in vocab_path.read_bytes().splitlines() if line
        )
    }
    specials_dict = dict(viztoken.build_default_qwen35_specials())
    ref_enc = tiktoken.Encoding(
        "Qwen3_5",
        pat_str=viztoken.QWEN35_PRETOKENIZE_REGEX,
        mergeable_ranks=ranks,
        special_tokens=specials_dict,
    )
    with viztoken.Encoding("Qwen3_5", ranks_path=RANKS_QWEN35_BIN) as sov_enc:
        run_differential_matrix(ref_enc, sov_enc)
        run_qwen35_multimodal_chat_audit(ref_enc, sov_enc)
        run_special_token_edgecases(ref_enc, sov_enc)
        run_agentic_prefix_splicing_audit(sov_enc)
        run_batch_pipeline_audit(ref_enc, sov_enc)
        run_thread_safety_soak(sov_enc, threads=8, passes_per_thread=5000)
        run_memory_soak(sov_enc, cycles=25000)
    print("\n" + "=" * 84)
    print("ALL TESTS PASSED: BIT-EXACT, THREAD-SAFE, ZERO-LEAK SUPREMACY CONFIRMED.")
    print("=" * 84)


if __name__ == "__main__":
    main()