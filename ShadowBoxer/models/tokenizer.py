from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import unicodedata

try:
    from viztoken import (
        Encoding,
        compile_vocab as _native_compile_vocab,
        find_common_prefix as viztoken_find_common_prefix,
    )
except ImportError:
    from ..viztoken import (
        Encoding,
        compile_vocab as _native_compile_vocab,
        find_common_prefix as viztoken_find_common_prefix,
    )

SHADOWBOXER_ROOT = Path(__file__).resolve().parent.parent

PRETOKENIZE_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|"
    r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

EOS_TOKEN_ID = 248044
IM_START_TOKEN_ID = 248045
IM_END_TOKEN_ID = 248044
VISION_START_TOKEN_ID = 248053
VISION_END_TOKEN_ID = 248054
IMAGE_PAD_TOKEN_ID = 248056
VIDEO_PAD_TOKEN_ID = 248057


def _gpt2_bytes_to_unicode() -> Dict[int, str]:
    bs = (
        list(range(33, 127))
        + list(range(161, 173))
        + list(range(174, 256))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_BYTE_DECODER: Dict[str, int] = {v: k for k, v in _gpt2_bytes_to_unicode().items()}


def _resolve_tokenizer_dir(model_or_tok_dir: Path) -> Path:
    if (model_or_tok_dir / "tokenizer.json").is_file():
        return model_or_tok_dir
    sub = model_or_tok_dir / "tokenizer"
    if (sub / "tokenizer.json").is_file():
        return sub
    return model_or_tok_dir


def _is_valid_qwen35_swiss5(bin_path: Path) -> bool:
    if not bin_path.is_file() or bin_path.stat().st_size <= 2 * 1024 * 1024:
        return False
    try:
        with Encoding("Qwen3_5", ranks_path=bin_path) as probe:
            return bool(probe.base_vocab_size >= 240000 and probe.has_fast_tables)
    except Exception:
        return False


def compile_qwen_ranks_binary(
    model_or_tok_dir: Union[str, Path],
    target_bin_path: Optional[Union[str, Path]] = None,
) -> Path:
    tok_dir = _resolve_tokenizer_dir(Path(model_or_tok_dir).resolve())
    bin_path = (
        Path(target_bin_path).resolve()
        if target_bin_path is not None
        else (tok_dir / "qwen35.ranks.bin")
    )

    if _is_valid_qwen35_swiss5(bin_path):
        return bin_path

    tiktoken_path = tok_dir / "qwen35.tiktoken"
    tok_json_path = tok_dir / "tokenizer.json"
    tok_cfg_path = tok_dir / "tokenizer_config.json"

    specials_list: List[Tuple[str, int]] = []
    regex_pat = PRETOKENIZE_REGEX

    if tok_json_path.is_file():
        payload = json.loads(tok_json_path.read_text(encoding="utf-8"))
        vocab: Dict[str, int] = payload.get("model", {}).get("vocab", {})
        added_tokens: List[Dict[str, Any]] = payload.get("added_tokens", [])

        if (
            not tiktoken_path.is_file()
            or tiktoken_path.stat().st_size < 1024 * 1024
        ) and vocab:
            sorted_vocab = sorted(vocab.items(), key=lambda kv: int(kv[1]))
            lines: List[str] = []
            for token_str, rank in sorted_vocab:
                raw_bytes = bytes(
                    _BYTE_DECODER[ch] for ch in token_str if ch in _BYTE_DECODER
                )
                if len(raw_bytes) == len(token_str):
                    b64_tok = base64.b64encode(raw_bytes).decode("ascii")
                    lines.append(f"{b64_tok} {int(rank)}")
            tiktoken_path.parent.mkdir(parents=True, exist_ok=True)
            tiktoken_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        for entry in added_tokens:
            content = entry.get("content")
            tid = entry.get("id")
            if isinstance(content, str) and isinstance(tid, int):
                specials_list.append((content, tid))

    if tok_cfg_path.is_file():
        try:
            tcfg = json.loads(tok_cfg_path.read_text(encoding="utf-8"))
            if (
                isinstance(tcfg.get("pretokenize_regex"), str)
                and tcfg["pretokenize_regex"]
            ):
                regex_pat = tcfg["pretokenize_regex"]
        except Exception:
            pass

    if not tiktoken_path.is_file():
        matches = list(tok_dir.rglob("*.tiktoken"))
        if matches:
            tiktoken_path = matches[0]
        else:
            raise FileNotFoundError(
                f"Cannot locate tokenizer.json or .tiktoken in {tok_dir}"
            )

    bin_path.parent.mkdir(parents=True, exist_ok=True)
    telem_path = tok_dir / "viztoken_telemetry.json"
    _native_compile_vocab(
        in_tiktoken_path=tiktoken_path,
        out_bin_path=bin_path,
        out_telemetry_json=telem_path,
        regex_pattern=regex_pat,
        specials_list=specials_list if specials_list else None,
        eot_token_id=EOS_TOKEN_ID,
    )
    return bin_path


def render_qwen35_chat_prompt(
    prompt: str,
    enable_thinking: bool = True,
    image_patch_counts: Optional[Sequence[int]] = None,
) -> str:
    vision_prefix = ""
    if image_patch_counts:
        parts = []
        for idx, count in enumerate(image_patch_counts, start=1):
            pads = "<|image_pad|>" * max(1, int(count))
            parts.append(f"Picture {idx}: <|vision_start|>{pads}<|vision_end|>")
        vision_prefix = "".join(parts)
    body = f"{vision_prefix}{prompt.strip()}"
    think_suffix = "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"
    return f"<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n{think_suffix}"


def load_ranks_binary(ranks_path: Union[str, Path, None] = None) -> Encoding:
    return MOSSImageTokenizer(ranks_path)._enc


class MOSSImageTokenizer:
    def __init__(self, vocab_source: Union[str, Path, None] = None):
        source_path = Path(vocab_source).resolve() if vocab_source else None
        resolved_bin: Optional[Path] = None

        if source_path is not None:
            if source_path.is_file() and source_path.suffix == ".bin":
                if _is_valid_qwen35_swiss5(source_path):
                    resolved_bin = source_path
            elif source_path.is_dir():
                tok_dir = _resolve_tokenizer_dir(source_path)
                for b_name in ("qwen35.ranks.bin", "qwen.ranks.bin"):
                    cand = tok_dir / b_name
                    if _is_valid_qwen35_swiss5(cand):
                        resolved_bin = cand
                        break
                if resolved_bin is None and (tok_dir / "tokenizer.json").is_file():
                    resolved_bin = compile_qwen_ranks_binary(tok_dir)

        if resolved_bin is None or not resolved_bin.is_file():
            search_candidates = [
                SHADOWBOXER_ROOT / "viztoken" / "qwen35.ranks.bin",
                SHADOWBOXER_ROOT / ".hf_cache" / "viztoken_qwen35" / "qwen35.ranks.bin",
                SHADOWBOXER_ROOT / "viztoken" / "qwen.ranks.bin",
            ]
            env_ranks = os.environ.get("VIZTOKEN_RANKS_PATH")
            if env_ranks:
                search_candidates.insert(0, Path(env_ranks).resolve())
            for cand in search_candidates:
                if _is_valid_qwen35_swiss5(cand):
                    resolved_bin = cand
                    break

        if resolved_bin is None or not resolved_bin.is_file():
            raise FileNotFoundError(
                f"Unable to resolve Qwen3.5 SWISS5 qwen35.ranks.bin in {SHADOWBOXER_ROOT / 'viztoken'}. "
                "Stage qwen35.ranks.bin into ShadowBoxer/viztoken/ or run download_model.py."
            )

        self._enc = Encoding("Qwen3_5", ranks_path=resolved_bin)
        self.n_vocab = self._enc.n_vocab
        self.base_vocab_size = self._enc.base_vocab_size
        self.has_fast_tables = self._enc.has_fast_tables

    @property
    def ranks_path(self) -> Path:
        return self._enc.ranks_path

    def encode_ordinary(self, text: str) -> List[int]:
        normalized = unicodedata.normalize("NFC", text)
        return self._enc.encode_ordinary(normalized)

    def encode(
        self, text: str, allowed_special: Union[str, Set[str]] = "all"
    ) -> List[int]:
        normalized = unicodedata.normalize("NFC", text)
        if not allowed_special:
            return self._enc.encode_ordinary(normalized)
        return self._enc.encode(
            normalized, allowed_special=allowed_special, disallowed_special=()
        )

    def encode_batch(
        self,
        texts: Sequence[str],
        num_threads: int = 8,
        allowed_special: Union[str, Set[str]] = "all",
    ) -> List[List[int]]:
        norm_list = [unicodedata.normalize("NFC", t) for t in texts]
        if not allowed_special:
            return self._enc.encode_ordinary_batch(
                norm_list, num_threads=num_threads
            )
        return self._enc.encode_batch(
            norm_list,
            num_threads=num_threads,
            allowed_special=allowed_special,
            disallowed_special=(),
        )

    def encode_chat(
        self,
        prompt: str,
        enable_thinking: bool = True,
        max_sequence_length: int = 4096,
        image_patch_counts: Optional[Sequence[int]] = None,
    ) -> List[int]:
        formatted = render_qwen35_chat_prompt(
            prompt,
            enable_thinking=enable_thinking,
            image_patch_counts=image_patch_counts,
        )
        toks = self.encode(formatted, allowed_special="all")
        if len(toks) > max_sequence_length:
            toks = toks[:max_sequence_length]
        return toks

    def decode(self, ids: Sequence[int], errors: str = "replace") -> str:
        return self._enc.decode(ids, errors=errors)

    @staticmethod
    def find_common_prefix(tokens_a: Sequence[int], tokens_b: Sequence[int]) -> int:
        return viztoken_find_common_prefix(tokens_a, tokens_b)