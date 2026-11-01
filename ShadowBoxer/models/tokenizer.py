from __future__ import annotations

import base64
import importlib.util
import os
from pathlib import Path
import shutil
import sys
from typing import (
    AbstractSet,
    Any,
    Collection,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
import unicodedata

import torch

MODELS_DIR = Path(__file__).resolve().parent
SHADOWBOXER_ROOT = MODELS_DIR.parent
VIZTOKEN_STAGE_DIR = SHADOWBOXER_ROOT / "viztoken"
IS_WIN = sys.platform == "win32"
_DLL_NAME = "viztoken.dll" if IS_WIN else "libviztoken.so"

QWEN35_BASE_VOCAB_SIZE = 248044
QWEN35_TOTAL_VOCAB_SIZE = 248320
QWEN35_EOT_TOKEN_ID = 248044
QWEN35_IM_START_TOKEN_ID = 248045
QWEN35_IM_END_TOKEN_ID = 248046
QWEN35_OBJECT_REF_START_ID = 248047
QWEN35_OBJECT_REF_END_ID = 248048
QWEN35_BOX_START_ID = 248049
QWEN35_BOX_END_ID = 248050
QWEN35_QUAD_START_ID = 248051
QWEN35_QUAD_END_ID = 248052
QWEN35_VISION_START_ID = 248053
QWEN35_VISION_END_ID = 248054
QWEN35_VISION_PAD_ID = 248055
QWEN35_IMAGE_PAD_ID = 248056
QWEN35_VIDEO_PAD_ID = 248057
QWEN35_TOOL_CALL_START_ID = 248058
QWEN35_TOOL_CALL_END_ID = 248059
QWEN35_FIM_PREFIX_ID = 248060
QWEN35_FIM_MIDDLE_ID = 248061
QWEN35_FIM_SUFFIX_ID = 248062
QWEN35_FIM_PAD_ID = 248063
QWEN35_REPO_NAME_ID = 248064
QWEN35_FILE_SEP_ID = 248065
QWEN35_TOOL_RESPONSE_START_ID = 248066
QWEN35_TOOL_RESPONSE_END_ID = 248067
QWEN35_THINK_START_ID = 248068
QWEN35_THINK_END_ID = 248069
QWEN35_AUDIO_START_ID = 248070
QWEN35_AUDIO_END_ID = 248071
QWEN35_TTS_PAD_ID = 248072
QWEN35_TTS_TEXT_BOS_ID = 248073
QWEN35_TTS_TEXT_EOD_ID = 248074
QWEN35_TTS_TEXT_BOS_SINGLE_ID = 248075
QWEN35_AUDIO_PAD_ID = 248076

QWEN35_PRETOKENIZE_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|"
    r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def build_default_qwen35_specials() -> List[Tuple[str, int]]:
    return [
        ("<|endoftext|>", QWEN35_EOT_TOKEN_ID),
        ("<|im_start|>", QWEN35_IM_START_TOKEN_ID),
        ("<|im_end|>", QWEN35_IM_END_TOKEN_ID),
        ("<|object_ref_start|>", QWEN35_OBJECT_REF_START_ID),
        ("<|object_ref_end|>", QWEN35_OBJECT_REF_END_ID),
        ("<|box_start|>", QWEN35_BOX_START_ID),
        ("<|box_end|>", QWEN35_BOX_END_ID),
        ("<|quad_start|>", QWEN35_QUAD_START_ID),
        ("<|quad_end|>", QWEN35_QUAD_END_ID),
        ("<|vision_start|>", QWEN35_VISION_START_ID),
        ("<|vision_end|>", QWEN35_VISION_END_ID),
        ("<|vision_pad|>", QWEN35_VISION_PAD_ID),
        ("<|image_pad|>", QWEN35_IMAGE_PAD_ID),
        ("<|video_pad|>", QWEN35_VIDEO_PAD_ID),
        ("<tool_call>", QWEN35_TOOL_CALL_START_ID),
        ("</tool_call>", QWEN35_TOOL_CALL_END_ID),
        ("<|fim_prefix|>", QWEN35_FIM_PREFIX_ID),
        ("<|fim_middle|>", QWEN35_FIM_MIDDLE_ID),
        ("<|fim_suffix|>", QWEN35_FIM_SUFFIX_ID),
        ("<|fim_pad|>", QWEN35_FIM_PAD_ID),
        ("<|repo_name|>", QWEN35_REPO_NAME_ID),
        ("<|file_sep|>", QWEN35_FILE_SEP_ID),
        ("<tool_response>", QWEN35_TOOL_RESPONSE_START_ID),
        ("</tool_response>", QWEN35_TOOL_RESPONSE_END_ID),
        ("<think>", QWEN35_THINK_START_ID),
        ("</think>", QWEN35_THINK_END_ID),
        ("<|audio_start|>", QWEN35_AUDIO_START_ID),
        ("<|audio_end|>", QWEN35_AUDIO_END_ID),
        ("<tts_pad>", QWEN35_TTS_PAD_ID),
        ("<tts_text_bos>", QWEN35_TTS_TEXT_BOS_ID),
        ("<tts_text_eod>", QWEN35_TTS_TEXT_EOD_ID),
        ("<tts_text_bos_single>", QWEN35_TTS_TEXT_BOS_SINGLE_ID),
        ("<|audio_pad|>", QWEN35_AUDIO_PAD_ID),
    ]


def render_qwen35_chat_prompt(
    prompt: str,
    enable_thinking: bool = True,
    image_patch_counts: Optional[Sequence[int]] = None,
    system_prompt: Optional[str] = None,
    thinking_trace: Optional[str] = None,
    add_vision_id: bool = True,
) -> str:
    norm_prompt = unicodedata.normalize("NFC", prompt).strip()
    sys_prefix = ""
    if system_prompt is not None:
        norm_sys = unicodedata.normalize("NFC", system_prompt).strip()
        if norm_sys:
            sys_prefix = f"<|im_start|>system\n{norm_sys}<|im_end|>\n"
    if image_patch_counts:
        vis_prefix = "".join(
            f"{'Picture ' + str(idx) + ': ' if add_vision_id else ''}<|vision_start|>{'<|image_pad|>' * max(1, int(c))}<|vision_end|>"
            for idx, c in enumerate(image_patch_counts, start=1)
        )
    else:
        vis_prefix = ""
    norm_trace = (
        unicodedata.normalize("NFC", thinking_trace).strip()
        if thinking_trace is not None
        else ""
    )
    if norm_trace:
        think_suffix = f"<think>\n{norm_trace}\n</think>\n\n"
    else:
        think_suffix = (
            "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"
        )
    return (
        f"{sys_prefix}<|im_start|>user\n{vis_prefix}{norm_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{think_suffix}"
    )


_VIZTOKEN_MOD: Optional[Any] = None


def _load_viztoken_module() -> Optional[Any]:
    global _VIZTOKEN_MOD
    if _VIZTOKEN_MOD is not None:
        return _VIZTOKEN_MOD

    py_cand = VIZTOKEN_STAGE_DIR / "viztoken.py"
    dll_cand = VIZTOKEN_STAGE_DIR / _DLL_NAME
    if py_cand.is_file() and dll_cand.is_file():
        try:
            os.environ["VIZTOKEN_DLL_PATH"] = str(dll_cand.resolve())
            ranks_cand = VIZTOKEN_STAGE_DIR / "qwen35.ranks.bin"
            if ranks_cand.is_file():
                os.environ["VIZTOKEN_RANKS_PATH"] = str(ranks_cand.resolve())
            spec = importlib.util.spec_from_file_location(
                "_sb_viztoken", str(py_cand)
            )
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["_sb_viztoken"] = mod
                spec.loader.exec_module(mod)
                if hasattr(mod, "is_valid_qwen35_swiss5"):
                    _VIZTOKEN_MOD = mod
                    return _VIZTOKEN_MOD
        except Exception:
            pass

    try:
        import viztoken as mod  # type: ignore

        _VIZTOKEN_MOD = mod
        return _VIZTOKEN_MOD
    except Exception:
        return None


def find_common_prefix(
    tokens_a: Union[Sequence[int], torch.Tensor, Any],
    tokens_b: Union[Sequence[int], torch.Tensor, Any],
) -> int:
    vt = _load_viztoken_module()
    if vt is not None and hasattr(vt, "find_common_prefix"):
        return int(vt.find_common_prefix(tokens_a, tokens_b))

    if hasattr(tokens_a, "detach"):
        tokens_a = tokens_a.detach().cpu().tolist()
    if hasattr(tokens_b, "detach"):
        tokens_b = tokens_b.detach().cpu().tolist()

    limit = min(len(tokens_a), len(tokens_b))
    for i in range(limit):
        if int(tokens_a[i]) != int(tokens_b[i]):
            return i
    return limit


def compile_qwen_ranks_binary(
    source_dir_or_file: Union[str, Path],
    target_bin_path: Optional[Union[str, Path]] = None,
) -> Path:
    vt = _load_viztoken_module()
    src = Path(source_dir_or_file).resolve()
    if target_bin_path is not None:
        dest_bin = Path(target_bin_path).resolve()
    elif src.is_dir():
        tok_sub = src / "tokenizer"
        dest_bin = (tok_sub if tok_sub.is_dir() else src) / "qwen35.ranks.bin"
    else:
        dest_bin = src.with_suffix(".ranks.bin")

    if vt is not None and hasattr(vt, "is_valid_qwen35_swiss5"):
        if vt.is_valid_qwen35_swiss5(dest_bin):
            return dest_bin
        prebuilt = VIZTOKEN_STAGE_DIR / "qwen35.ranks.bin"
        if prebuilt.is_file() and vt.is_valid_qwen35_swiss5(prebuilt):
            if prebuilt.resolve() != dest_bin.resolve():
                dest_bin.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(prebuilt, dest_bin)
            return dest_bin

    if vt is not None and hasattr(vt, "compile_qwen_ranks_binary"):
        try:
            return Path(vt.compile_qwen_ranks_binary(src, dest_bin))
        except FileNotFoundError:
            for fallback_tik in (
                VIZTOKEN_STAGE_DIR / "qwen35.tiktoken",
                SHADOWBOXER_ROOT / "qwen35.tiktoken",
            ):
                if fallback_tik.is_file():
                    return Path(
                        vt.compile_qwen_ranks_binary(fallback_tik, dest_bin)
                    )
            raise

    raise RuntimeError("VizToken native compiler module is unavailable.")


def _resolve_ranks_or_tiktoken(
    model_path: Optional[Union[str, Path]] = None,
    ranks_path: Optional[Union[str, Path]] = None,
    vt_mod: Optional[Any] = None,
) -> Tuple[Optional[Path], Optional[Path]]:
    bin_candidates: List[Path] = []
    tik_candidates: List[Path] = []

    if ranks_path is not None:
        rp = Path(ranks_path).resolve()
        if rp.is_file() and rp.suffix == ".bin":
            bin_candidates.append(rp)
        elif rp.is_file() and rp.suffix == ".tiktoken":
            tik_candidates.append(rp)
        elif rp.is_dir():
            bin_candidates.append(rp / "qwen35.ranks.bin")
            tik_candidates.append(rp / "qwen35.tiktoken")

    if model_path is not None:
        mp = Path(model_path).resolve()
        if mp.is_file() and mp.suffix == ".bin":
            bin_candidates.append(mp)
        elif mp.is_file() and mp.suffix == ".tiktoken":
            tik_candidates.append(mp)
        elif mp.is_dir():
            bin_candidates.extend(
                [
                    mp / "qwen35.ranks.bin",
                    mp / "tokenizer" / "qwen35.ranks.bin",
                    mp / "text_encoder" / "qwen35.ranks.bin",
                ]
            )
            tik_candidates.extend(
                [
                    mp / "qwen35.tiktoken",
                    mp / "vocab.tiktoken",
                    mp / "qwen.tiktoken",
                    mp / "tokenizer" / "qwen35.tiktoken",
                    mp / "tokenizer" / "vocab.tiktoken",
                ]
            )

    env_ranks = os.environ.get("VIZTOKEN_RANKS_PATH")
    if env_ranks:
        bin_candidates.append(Path(env_ranks).resolve())

    bin_candidates.extend(
        [
            VIZTOKEN_STAGE_DIR / "qwen35.ranks.bin",
            SHADOWBOXER_ROOT / "qwen35.ranks.bin",
        ]
    )
    tik_candidates.extend(
        [
            VIZTOKEN_STAGE_DIR / "qwen35.tiktoken",
            SHADOWBOXER_ROOT / "qwen35.tiktoken",
        ]
    )

    resolved_bin: Optional[Path] = None
    if vt_mod is not None and hasattr(vt_mod, "is_valid_qwen35_swiss5"):
        for c in bin_candidates:
            if c.is_file() and vt_mod.is_valid_qwen35_swiss5(c):
                resolved_bin = c
                break
    if resolved_bin is None:
        resolved_bin = next((c for c in bin_candidates if c.is_file()), None)

    resolved_tik = next((c for c in tik_candidates if c.is_file()), None)
    return resolved_bin, resolved_tik


class MOSSImageTokenizer:
    BASE_VOCAB_SIZE = QWEN35_BASE_VOCAB_SIZE
    TOTAL_VOCAB_SIZE = QWEN35_TOTAL_VOCAB_SIZE
    EOT_TOKEN_ID = QWEN35_EOT_TOKEN_ID
    IM_START_TOKEN_ID = QWEN35_IM_START_TOKEN_ID
    IM_END_TOKEN_ID = QWEN35_IM_END_TOKEN_ID
    VISION_START_ID = QWEN35_VISION_START_ID
    VISION_END_ID = QWEN35_VISION_END_ID
    VISION_PAD_ID = QWEN35_VISION_PAD_ID
    IMAGE_PAD_ID = QWEN35_IMAGE_PAD_ID
    VIDEO_PAD_ID = QWEN35_VIDEO_PAD_ID
    THINK_START_ID = QWEN35_THINK_START_ID
    THINK_END_ID = QWEN35_THINK_END_ID

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        ranks_path: Optional[Union[str, Path]] = None,
    ):
        self._vt_mod = _load_viztoken_module()
        self._viztoken_active = False
        self._enc: Any = None
        self._ranks_path: Optional[Path] = None

        resolved_bin, resolved_tik = _resolve_ranks_or_tiktoken(
            model_path=model_path, ranks_path=ranks_path, vt_mod=self._vt_mod
        )

        if self._vt_mod is not None:
            if resolved_bin is not None and self._vt_mod.is_valid_qwen35_swiss5(
                resolved_bin
            ):
                self._enc = self._vt_mod.Encoding("Qwen3_5", ranks_path=resolved_bin)
                self._ranks_path = resolved_bin
                self._viztoken_active = True
            elif resolved_tik is not None:
                target_bin = VIZTOKEN_STAGE_DIR / "qwen35.ranks.bin"
                try:
                    target_bin.parent.mkdir(parents=True, exist_ok=True)
                    self._vt_mod.compile_qwen_ranks_binary(resolved_tik, target_bin)
                    self._enc = self._vt_mod.Encoding("Qwen3_5", ranks_path=target_bin)
                    self._ranks_path = target_bin
                    self._viztoken_active = True
                except Exception:
                    pass

        if not self._viztoken_active:
            if resolved_tik is None:
                raise FileNotFoundError(
                    "Neither compiled qwen35.ranks.bin nor qwen35.tiktoken could be located."
                )
            import tiktoken

            ranks = {
                base64.b64decode(t): int(r)
                for t, r in (
                    line.split()
                    for line in resolved_tik.read_bytes().splitlines()
                    if line
                )
            }
            specials_dict = dict(build_default_qwen35_specials())
            self._enc = tiktoken.Encoding(
                "Qwen3_5",
                pat_str=QWEN35_PRETOKENIZE_REGEX,
                mergeable_ranks=ranks,
                special_tokens=specials_dict,
            )
            self._ranks_path = resolved_tik

        self._special_token_to_id: Dict[str, int] = dict(
            build_default_qwen35_specials()
        )
        self._special_tokens_set: Set[str] = set(self._special_token_to_id.keys())
        self._id_to_special_token: Dict[int, str] = {
            v: k for k, v in self._special_token_to_id.items()
        }

        self.eos_token_id = QWEN35_EOT_TOKEN_ID
        self.pad_token_id = QWEN35_EOT_TOKEN_ID
        self.im_start_id = QWEN35_IM_START_TOKEN_ID
        self.im_end_id = QWEN35_IM_END_TOKEN_ID
        self.vision_start_id = QWEN35_VISION_START_ID
        self.vision_end_id = QWEN35_VISION_END_ID
        self.vision_pad_id = QWEN35_VISION_PAD_ID
        self.image_pad_id = QWEN35_IMAGE_PAD_ID
        self.video_pad_id = QWEN35_VIDEO_PAD_ID
        self.think_start_id = QWEN35_THINK_START_ID
        self.think_end_id = QWEN35_THINK_END_ID
        if self._viztoken_active and hasattr(self._enc, "_chat_tail_trace_close"):
            self._trace_close_token_ids: List[int] = list(
                self._enc._chat_tail_trace_close
            )
        else:
            self._trace_close_token_ids = list(
                self._enc.encode("\n</think>\n\n", allowed_special="all")
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> MOSSImageTokenizer:
        return cls(
            model_path=pretrained_model_name_or_path,
            ranks_path=kwargs.get("ranks_path"),
        )

    def __enter__(self) -> MOSSImageTokenizer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def viztoken_active(self) -> bool:
        return self._viztoken_active

    @property
    def ranks_path(self) -> Path:
        return self._ranks_path or (VIZTOKEN_STAGE_DIR / "qwen35.ranks.bin")

    @property
    def has_fast_tables(self) -> bool:
        if self._viztoken_active and hasattr(self._enc, "has_fast_tables"):
            return bool(self._enc.has_fast_tables)
        return False

    @property
    def base_vocab_size(self) -> int:
        if self._viztoken_active and hasattr(self._enc, "base_vocab_size"):
            return int(self._enc.base_vocab_size)
        return QWEN35_BASE_VOCAB_SIZE

    @property
    def vocab_size(self) -> int:
        return self.base_vocab_size

    @property
    def total_vocab_size(self) -> int:
        return QWEN35_TOTAL_VOCAB_SIZE

    @property
    def n_vocab(self) -> int:
        if self._viztoken_active and hasattr(self._enc, "n_vocab"):
            return int(self._enc.n_vocab)
        return QWEN35_TOTAL_VOCAB_SIZE

    @property
    def eot_token(self) -> int:
        if self._viztoken_active and hasattr(self._enc, "eot_token"):
            return int(self._enc.eot_token)
        return QWEN35_EOT_TOKEN_ID

    @property
    def trace_close_token_ids(self) -> List[int]:
        return list(self._trace_close_token_ids)

    @property
    def special_tokens_set(self) -> Set[str]:
        return self._special_tokens_set

    @property
    def special_token_to_id(self) -> Dict[str, int]:
        return self._special_token_to_id

    def close(self) -> None:
        if (
            self._viztoken_active
            and self._enc is not None
            and hasattr(self._enc, "close")
        ):
            self._enc.close()
            self._enc = None

    def __del__(self) -> None:
        self.close()

    def encode_ordinary(self, text: str) -> List[int]:
        if not text:
            return []
        norm = unicodedata.normalize("NFC", text)
        return list(self._enc.encode_ordinary(norm))

    def encode(
        self,
        text: str,
        *,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = "all",
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        if not text:
            return []
        norm = unicodedata.normalize("NFC", text)
        return list(
            self._enc.encode(
                norm,
                allowed_special=allowed_special,
                disallowed_special=disallowed_special,
            )
        )

    def encode_batch(
        self,
        texts: Iterable[str],
        *,
        num_threads: int = 8,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = "all",
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[List[int]]:
        norm_texts = [unicodedata.normalize("NFC", t) for t in texts]
        return list(
            self._enc.encode_batch(
                norm_texts,
                num_threads=num_threads,
                allowed_special=allowed_special,
                disallowed_special=disallowed_special,
            )
        )

    def decode(
        self,
        token_ids: Union[Sequence[int], torch.Tensor, Any],
        errors: str = "replace",
    ) -> str:
        if hasattr(token_ids, "detach"):
            token_ids = token_ids.detach().cpu().numpy().ravel().tolist()
        if not token_ids:
            return ""
        return str(self._enc.decode(token_ids, errors=errors))

    def decode_batch(
        self,
        batch: Iterable[Union[Sequence[int], torch.Tensor, Any]],
        *,
        num_threads: int = 8,
        errors: str = "replace",
    ) -> List[str]:
        clean_batch = [
            b.detach().cpu().numpy().ravel().tolist()
            if hasattr(b, "detach")
            else list(b)
            for b in batch
        ]
        return list(
            self._enc.decode_batch(
                clean_batch, num_threads=num_threads, errors=errors
            )
        )

    def encode_chat(
        self,
        prompt: str,
        enable_thinking: bool = True,
        max_sequence_length: int = 16384,
        image_patch_counts: Optional[Sequence[int]] = None,
        system_prompt: Optional[str] = None,
        thinking_trace: Optional[str] = None,
        add_vision_id: bool = True,
        return_tensors: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Union[List[int], torch.Tensor]:
        if self._viztoken_active and hasattr(self._enc, "encode_chat"):
            ids = list(
                self._enc.encode_chat(
                    prompt=prompt,
                    enable_thinking=enable_thinking,
                    max_sequence_length=max_sequence_length,
                    image_patch_counts=image_patch_counts,
                    system_prompt=system_prompt,
                    thinking_trace=thinking_trace,
                    add_vision_id=add_vision_id,
                )
            )
        else:
            ids = self._fallback_encode_chat(
                prompt=prompt,
                enable_thinking=enable_thinking,
                max_sequence_length=max_sequence_length,
                image_patch_counts=image_patch_counts,
                system_prompt=system_prompt,
                thinking_trace=thinking_trace,
                add_vision_id=add_vision_id,
            )

        if return_tensors == "pt":
            t = torch.tensor([ids], dtype=torch.long)
            return t.to(device=device) if device is not None else t
        return ids

    def tokenize_chat(
        self,
        prompt: str,
        enable_thinking: bool = True,
        max_sequence_length: int = 16384,
        image_patch_counts: Optional[Sequence[int]] = None,
        system_prompt: Optional[str] = None,
        thinking_trace: Optional[str] = None,
        add_vision_id: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = self.encode_chat(
            prompt=prompt,
            enable_thinking=enable_thinking,
            max_sequence_length=max_sequence_length,
            image_patch_counts=image_patch_counts,
            system_prompt=system_prompt,
            thinking_trace=thinking_trace,
            add_vision_id=add_vision_id,
            return_tensors=None,
        )
        assert isinstance(ids, list)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        return input_ids, attention_mask

    @staticmethod
    def find_common_prefix(
        tokens_a: Union[Sequence[int], torch.Tensor, Any],
        tokens_b: Union[Sequence[int], torch.Tensor, Any],
    ) -> int:
        return find_common_prefix(tokens_a, tokens_b)

    def _fallback_encode_chat(
        self,
        prompt: str,
        enable_thinking: bool = True,
        max_sequence_length: int = 16384,
        image_patch_counts: Optional[Sequence[int]] = None,
        system_prompt: Optional[str] = None,
        thinking_trace: Optional[str] = None,
        add_vision_id: bool = True,
    ) -> List[int]:
        norm_prompt = unicodedata.normalize("NFC", prompt).strip()
        toks: List[int] = []

        if system_prompt is not None:
            norm_sys = unicodedata.normalize("NFC", system_prompt).strip()
            if norm_sys:
                toks.extend(
                    self._enc.encode(
                        f"<|im_start|>system\n{norm_sys}<|im_end|>\n",
                        allowed_special="all",
                    )
                )

        if not image_patch_counts:
            toks.extend(
                self._enc.encode("<|im_start|>user\n", allowed_special="all")
            )
        else:
            for idx, count in enumerate(image_patch_counts, start=1):
                if add_vision_id:
                    vis_pfx = (
                        f"<|im_start|>user\nPicture {idx}: <|vision_start|>"
                        if idx == 1
                        else f"<|vision_end|>Picture {idx}: <|vision_start|>"
                    )
                else:
                    vis_pfx = (
                        "<|im_start|>user\n<|vision_start|>"
                        if idx == 1
                        else "<|vision_end|><|vision_start|>"
                    )
                toks.extend(self._enc.encode(vis_pfx, allowed_special="all"))
                toks.extend([QWEN35_IMAGE_PAD_ID] * max(1, int(count)))
            toks.append(QWEN35_VISION_END_ID)

        prompt_toks = (
            self._enc.encode(norm_prompt, allowed_special="all")
            if norm_prompt
            else []
        )

        norm_trace = (
            unicodedata.normalize("NFC", thinking_trace).strip()
            if thinking_trace is not None
            else ""
        )
        budget = int(max_sequence_length)

        if not norm_trace:
            if budget > 0 and len(prompt_toks) > budget:
                prompt_toks = prompt_toks[:budget]
            toks.extend(prompt_toks)
            tail = (
                "<|im_end|>\n<|im_start|>assistant\n<think>\n"
                if enable_thinking
                else "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
            toks.extend(self._enc.encode(tail, allowed_special="all"))
            return toks

        trace_open = self._enc.encode(
            "<|im_end|>\n<|im_start|>assistant\n<think>\n", allowed_special="all"
        )
        trace_close = self._enc.encode("\n</think>\n\n", allowed_special="all")
        trace_full = self._enc.encode(
            f"{norm_trace}\n</think>\n\n", allowed_special="all"
        )
        trace_body_len = max(1, len(trace_full) - len(trace_close))

        if budget <= 0 or (len(prompt_toks) + trace_body_len) <= budget:
            toks.extend(prompt_toks)
            toks.extend(trace_open)
            toks.extend(trace_full)
            return toks

        trace_body = self._enc.encode(norm_trace, allowed_special="all")
        if len(prompt_toks) <= budget:
            rem = budget - len(prompt_toks)
            trace_body = trace_body[:rem] if rem > 0 else []
        else:
            t_bud = min(len(trace_body), max(1, budget // 3)) if budget >= 3 else 0
            p_bud = max(1, budget - t_bud)
            prompt_toks = prompt_toks[:p_bud]
            trace_body = trace_body[:t_bud] if t_bud > 0 else []

        toks.extend(prompt_toks)
        if trace_body:
            toks.extend(trace_open)
            toks.extend(trace_body)
            toks.extend(trace_close)
        else:
            toks.extend(
                self._enc.encode(
                    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                    allowed_special="all",
                )
            )
        return toks


Qwen35Tokenizer = MOSSImageTokenizer
Qwen25VLTokenizer = MOSSImageTokenizer