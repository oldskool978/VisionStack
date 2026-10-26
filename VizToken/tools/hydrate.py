#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import gc
import gzip
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
import zlib

TOOLS_DIR = Path(__file__).parent.resolve()
ROOT_DIR = TOOLS_DIR.parent
TOOLCHAIN_DIR = ROOT_DIR / "toolchain"
LIBRARY_DIR = ROOT_DIR / "library"
CACHE_DIR = ROOT_DIR / ".forge_cache"
TMP_DIR = CACHE_DIR / "tmp"
PIP_CACHE_DIR = CACHE_DIR / "pip"
WHEELS_DIR = CACHE_DIR / "wheels"
HOME_DIR = CACHE_DIR / "home"
LOCAL_APPDATA_DIR = CACHE_DIR / "local_appdata"
ROAMING_APPDATA_DIR = CACHE_DIR / "appdata"
WINSDK_DIR = TOOLCHAIN_DIR / "winsdk"
PYTHON_DIR = TOOLCHAIN_DIR / "python"
NATIVE_SYSROOT = TOOLCHAIN_DIR / "native-sysroot"


def _resolve_hf_cache_dir() -> Path:
    env_hf = os.environ.get("HF_HOME")
    if env_hf and env_hf.strip():
        return Path(env_hf).resolve()
    parent_hf = (ROOT_DIR.parent / ".hf_cache").resolve()
    if parent_hf.exists():
        return parent_hf
    return (CACHE_DIR / "hf_cache").resolve()


HF_CACHE_DIR = _resolve_hf_cache_dir()
PARENT_WHEELS_DIR = (ROOT_DIR.parent / ".cache" / "downloads" / "wheels").resolve()

HOST_OS = sys.platform
HOST_ARCH = platform.machine().lower()
IS_WIN = HOST_OS == "win32"
IS_MAC = HOST_OS == "darwin"
IS_LINUX = HOST_OS.startswith("linux")
HTTPS = "https" + "://"

CLANG_EXE = TOOLCHAIN_DIR / "llvm" / "bin" / ("clang.exe" if IS_WIN else "clang")
CLANG_CL_EXE = (
    TOOLCHAIN_DIR / "llvm" / "bin" / ("clang-cl.exe" if IS_WIN else "clang-cl")
)
LLD_LINK_EXE = (
    TOOLCHAIN_DIR / "llvm" / "bin" / ("lld-link.exe" if IS_WIN else "lld-link")
)
CMAKE_EXE = TOOLCHAIN_DIR / "cmake" / "bin" / ("cmake.exe" if IS_WIN else "cmake")
NINJA_EXE = TOOLCHAIN_DIR / "ninja" / ("ninja.exe" if IS_WIN else "ninja")
NASM_EXE = TOOLCHAIN_DIR / "nasm" / ("nasm.exe" if IS_WIN else "nasm")
GIT_CMD_EXE = TOOLCHAIN_DIR / "git" / "cmd" / ("git.exe" if IS_WIN else "git")
HERMETIC_PYTHON_EXE = PYTHON_DIR / ("python.exe" if IS_WIN else "bin/python")

NUGET_PACKAGES = [
    "microsoft.windows.sdk.cpp",
    "microsoft.windows.sdk.cpp.x64",
    "microsoft.windows.sdk.buildtools",
]

SDK_REQUIRED_ANCHORS = [
    "kernel32.lib",
    "user32.lib",
    "advapi32.lib",
    "ucrt.lib",
    "libucrt.lib",
]

MSVC_REQUIRED_ANCHORS = [
    "libcmt.lib",
    "libvcruntime.lib",
    "oldnames.lib",
]

VOCAB_ENDPOINTS = [
    f"{HTTPS}huggingface.co/m-a-p/YuE2-3B/resolve/main/qwen.tiktoken",
    f"{HTTPS}huggingface.co/Qwen/Qwen-7B/resolve/main/qwen.tiktoken",
]
EXPECTED_VOCAB_LINES = 151643

QWEN35_TOKENIZER_JSON_ENDPOINTS = [
    f"{HTTPS}huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/resolve/main/tokenizer/tokenizer.json",
    f"{HTTPS}huggingface.co/OpenMOSS-Team/MOSS-Image-Base/resolve/main/tokenizer/tokenizer.json",
]

SEMVER_TAG_REGEX = re.compile(
    r"^(?:v|pcre2-|release-|rel-)?(\d+(?:\.\d+)+)$", re.IGNORECASE
)
URL_EXTRACTOR_REGEX = re.compile(r"(https?://[^\]\s)>\"']+)", re.IGNORECASE)
_VERIFIED_ARCHIVE_SIGNATURES: set[tuple[str, int, int]] = set()


def log_stage(msg: str) -> None:
    sys.stdout.write(f"[STAGE] {msg}\n")
    sys.stdout.flush()


def log_info(msg: str) -> None:
    sys.stdout.write(f"[INFO]  {msg}\n")
    sys.stdout.flush()


def log_warn(msg: str) -> None:
    sys.stderr.write(f"[WARN]  {msg}\n")
    sys.stderr.flush()


def sanitize_url(raw_url: str) -> str:
    cleaned = raw_url.strip()
    match = URL_EXTRACTOR_REGEX.search(cleaned)
    if match:
        cleaned = match.group(1)
    return cleaned.replace(" ", "%20")


def to_unc_path(path: Path) -> str:
    resolved = str(path.resolve())
    if IS_WIN and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved.replace("/", "\\")
    return resolved


def remove_readonly(func, path, _):
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        if func is not None:
            func(path)
    except Exception:
        pass


def safe_rmtree(
    dir_path: Path, max_retries: int = 12, initial_delay: float = 0.1
) -> None:
    if not dir_path.exists():
        return
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            if sys.version_info >= (3, 12):
                shutil.rmtree(
                    dir_path,
                    onexc=lambda func, path, _: remove_readonly(func, path, None),
                )
            else:
                shutil.rmtree(dir_path, onerror=remove_readonly)
            return
        except (PermissionError, OSError):
            if attempt == max_retries - 1:
                raise
            gc.collect()
            time.sleep(delay)
            delay *= 1.5


def copy_tree_contents(src_dir: Path, dest_dir: Path) -> int:
    copied_count = 0
    for item in src_dir.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(src_dir)
            target_file = dest_dir / rel_path
            target_str = to_unc_path(target_file)
            os.makedirs(os.path.dirname(target_str), exist_ok=True)
            if target_file.exists():
                remove_readonly(None, target_str, None)
            shutil.copy2(to_unc_path(item), target_str)
            copied_count += 1
    return copied_count


def promote_directory(src_dir: Path, dest_dir: Path) -> None:
    if not src_dir.exists():
        return
    safe_rmtree(dest_dir)
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(to_unc_path(src_dir), to_unc_path(dest_dir))
        return
    except (PermissionError, OSError):
        pass
    dest_dir.mkdir(parents=True, exist_ok=True)
    copy_tree_contents(src_dir, dest_dir)
    gc.collect()
    safe_rmtree(src_dir)


def build_isolated_env() -> dict[str, str]:
    for d in (
        CACHE_DIR,
        TMP_DIR,
        PIP_CACHE_DIR,
        WHEELS_DIR,
        HOME_DIR,
        LOCAL_APPDATA_DIR,
        ROAMING_APPDATA_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)

    tempfile.tempdir = str(TMP_DIR)
    env: dict[str, str] = {}
    for k in (
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE",
        "GITHUB_TOKEN",
        "WT_SESSION",
        "TERM",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    ):
        if k in os.environ:
            env[k] = os.environ[k]

    sys_root = env.get("SYSTEMROOT", r"C:\Windows")
    bins = [
        str(PYTHON_DIR if IS_WIN else PYTHON_DIR / "bin"),
        str(PYTHON_DIR / "Scripts" if IS_WIN else PYTHON_DIR / "bin"),
        str(TOOLCHAIN_DIR / "git" / "cmd"),
        str(TOOLCHAIN_DIR / "llvm" / "bin"),
        str(TOOLCHAIN_DIR / "ninja"),
        str(TOOLCHAIN_DIR / "cmake" / "bin"),
        str(TOOLCHAIN_DIR / "nasm"),
    ]
    if IS_WIN:
        bins.extend(
            [
                os.path.join(sys_root, "System32"),
                sys_root,
                os.path.join(sys_root, "System32", "Wbem"),
                os.path.join(sys_root, "System32", "WindowsPowerShell", "v1.0"),
            ]
        )
        env["PIP_CONFIG_FILE"] = "NUL"
    else:
        bins.extend(["/usr/bin", "/bin"])
        env["PIP_CONFIG_FILE"] = "/dev/null"

    env["PATH"] = os.pathsep.join(bins)
    env["PYTHONHOME"] = str(PYTHON_DIR)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PIP_NO_USER"] = "1"
    env["PIP_USER"] = "0"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["TEMP"] = str(TMP_DIR)
    env["TMP"] = str(TMP_DIR)
    env["TMPDIR"] = str(TMP_DIR)
    env["PIP_CACHE_DIR"] = str(PIP_CACHE_DIR)
    env["USERPROFILE"] = str(HOME_DIR)
    env["HOME"] = str(HOME_DIR)
    env["LOCALAPPDATA"] = str(LOCAL_APPDATA_DIR)
    env["APPDATA"] = str(ROAMING_APPDATA_DIR)
    env["HF_HOME"] = str(HF_CACHE_DIR)
    env["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
    return env


def enforce_pth_sandbox() -> None:
    if not IS_WIN or not PYTHON_DIR.exists():
        return
    (PYTHON_DIR / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)
    for pth_file in PYTHON_DIR.glob("*._pth"):
        orig_lines = pth_file.read_text(encoding="utf-8").splitlines()
        core_zips = [
            line.strip() for line in orig_lines if line.strip().endswith(".zip")
        ]
        if not core_zips:
            core_zips = [z.name for z in PYTHON_DIR.glob("python*.zip")]
        payload_paths = core_zips + [
            "../..",
            ".",
            "Lib/site-packages",
            "import site",
        ]
        pth_file.write_text("\n".join(payload_paths) + "\n", encoding="utf-8")


def parse_semver(ver_str: str) -> tuple[int, ...]:
    nums = [int(n) for n in re.findall(r"\d+", ver_str)]
    return tuple(nums) if nums else (0,)


def normalize_numeric_version(ver_str: str) -> str:
    m = re.match(r"^(\d+(?:\.\d+)+)", ver_str.strip())
    return m.group(1) if m else ver_str.strip()


def _http_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {"User-Agent": "viztoken-hydrator/5.0"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if extra:
        headers.update(extra)
    return headers


def probe_url(url: str, timeout: int = 8) -> bool:
    clean_url = sanitize_url(url)
    req = urllib.request.Request(clean_url, headers=_http_headers(), method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 400
    except Exception:
        return False


def fetch_http_text(url: str, timeout: int = 30) -> str:
    clean_url = sanitize_url(url)
    req = urllib.request.Request(
        clean_url,
        headers=_http_headers({"Accept-Encoding": "gzip, deflate"}),
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = response.read()
        content_encoding = response.headers.get("Content-Encoding", "").lower()
        if content_encoding == "gzip" or data.startswith(b"\x1f\x8b"):
            data = gzip.decompress(data)
        elif content_encoding == "deflate":
            try:
                data = zlib.decompress(data)
            except zlib.error:
                data = zlib.decompress(data, -zlib.MAX_WBITS)
        return data.decode("utf-8", errors="ignore")


def fetch_json(url: str, timeout: int = 30):
    text = fetch_http_text(url, timeout=timeout)
    stripped = text.lstrip()
    if stripped.startswith("<"):
        raise ValueError(f"Endpoint returned HTML instead of JSON: {url}")
    return json.loads(text)


def fetch_github_releases(repo: str, per_page: int = 15) -> list[dict]:
    try:
        data = fetch_json(
            f"{HTTPS}api.github.com/repos/{repo}/releases?per_page={per_page}",
            timeout=20,
        )
        if isinstance(data, list) and data:
            return data
    except Exception:
        pass
    try:
        latest = fetch_json(
            f"{HTTPS}api.github.com/repos/{repo}/releases/latest", timeout=15
        )
        if isinstance(latest, dict) and latest:
            return [latest]
    except Exception:
        pass
    return []


def read_version_marker(target_dir: Path, filename: str = ".version") -> str:
    ver_file = target_dir / filename
    if ver_file.exists():
        return ver_file.read_text(encoding="utf-8").strip()
    return ""


def write_version_marker(
    target_dir: Path, tag: str, filename: str = ".version"
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / filename).write_text(f"{tag}\n", encoding="utf-8")


def is_version_current(
    target_dir: Path, expected_tag: str, marker_file: Path, force: bool = False
) -> bool:
    if (
        force
        or not expected_tag
        or not target_dir.exists()
        or not marker_file.exists()
    ):
        return False
    current_tag = read_version_marker(target_dir)
    if not current_tag:
        return False
    return current_tag == expected_tag


def fetch_payload(url: str, destination: Path, timeout: int = 300) -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    clean_url = sanitize_url(url)
    log_info(f"Downloading: {clean_url}")
    req = urllib.request.Request(clean_url, headers=_http_headers())
    with urllib.request.urlopen(req, timeout=timeout) as response, open(
        to_unc_path(destination), "wb"
    ) as out_file:
        shutil.copyfileobj(response, out_file)
        out_file.flush()
        os.fsync(out_file.fileno())


def unpack_archive(
    archive_path: Path, extract_dir: Path, flatten_single_dir: bool = False
) -> None:
    safe_rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    archive_str = str(archive_path)
    if archive_str.endswith((".zip", ".vsix", ".nupkg")):
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            for member in zip_ref.infolist():
                rel_path = member.filename.replace("/", os.sep)
                target_str = to_unc_path(extract_dir / rel_path)
                if member.is_dir():
                    os.makedirs(target_str, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(target_str), exist_ok=True)
                    with zip_ref.open(member) as src, open(target_str, "wb") as dst:
                        shutil.copyfileobj(src, dst)
    elif archive_str.endswith((".tar.xz", ".tar.gz", ".tar.bz2", ".tgz")):
        with tarfile.open(archive_path, "r:*") as tar_ref:
            for member in tar_ref.getmembers():
                rel_path = member.name.replace("/", os.sep)
                target_str = to_unc_path(extract_dir / rel_path)
                if member.isdir():
                    os.makedirs(target_str, exist_ok=True)
                elif member.isfile():
                    os.makedirs(os.path.dirname(target_str), exist_ok=True)
                    f_src = tar_ref.extractfile(member)
                    if f_src is not None:
                        with f_src, open(target_str, "wb") as f_dst:
                            shutil.copyfileobj(f_src, f_dst)
    if flatten_single_dir:
        entries = list(extract_dir.iterdir())
        if len(entries) == 1 and entries[0].is_dir():
            single_dir = entries[0]
            temp_dir = extract_dir.parent / f"_tmp_{extract_dir.name}"
            promote_directory(single_dir, temp_dir)
            promote_directory(temp_dir, extract_dir)


def verify_tiktoken_file(
    target_file: Path, min_lines: int = EXPECTED_VOCAB_LINES
) -> bool:
    if not target_file.exists() or target_file.stat().st_size < 1_500_000:
        return False
    try:
        with open(target_file, "rb") as f:
            first_line = f.readline().strip()
            if not first_line or b" " not in first_line:
                return False
            parts = first_line.split(b" ")
            if len(parts) != 2:
                return False
            base64.b64decode(parts[0])
            int(parts[1])
            f.seek(0)
            line_count = sum(1 for line in f if line.strip())
            return line_count >= min_lines
    except Exception:
        return False


def _gpt2_byte_decoder() -> dict[str, int]:
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
    return {chr(c): b for b, c in zip(bs, cs)}


def _convert_tokenizer_json_to_tiktoken(
    json_path: Path, out_tiktoken_path: Path
) -> int:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    vocab = payload.get("model", {}).get("vocab", {})
    if len(vocab) < 240000:
        return 0
    byte_dec = _gpt2_byte_decoder()
    sorted_vocab = sorted(vocab.items(), key=lambda kv: int(kv[1]))
    lines: list[str] = []
    for token_str, rank in sorted_vocab:
        raw_bytes = bytes(byte_dec[ch] for ch in token_str if ch in byte_dec)
        if len(raw_bytes) == len(token_str):
            b64_tok = base64.b64encode(raw_bytes).decode("ascii")
            lines.append(f"{b64_tok} {int(rank)}")
    out_tiktoken_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def hydrate_vocabulary(force: bool = False) -> None:
    target_vocab = ROOT_DIR / "qwen.tiktoken"
    if not force and verify_tiktoken_file(target_vocab, EXPECTED_VOCAB_LINES):
        log_info(f"Authoritative base vocabulary converged: {target_vocab.name}")
    else:
        log_stage("Hydrating Authoritative Base Vocabulary (qwen.tiktoken)")
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        temp_target = TMP_DIR / f"qwen.tiktoken.{os.getpid()}.tmp"
        temp_target.unlink(missing_ok=True)
        downloaded = False
        for url in VOCAB_ENDPOINTS:
            try:
                log_info(f"Streaming vocabulary from: {url}")
                fetch_payload(url, temp_target, timeout=120)
                if verify_tiktoken_file(temp_target, EXPECTED_VOCAB_LINES):
                    downloaded = True
                    break
                log_warn(f"Integrity check failed for {url}")
                temp_target.unlink(missing_ok=True)
            except Exception as exc:
                log_warn(f"Mirror fetch failed for {url}: {exc}")
                temp_target.unlink(missing_ok=True)
        if not downloaded:
            raise RuntimeError(
                "Authoritative vocabulary hydration failed across all mirror endpoints."
            )
        if target_vocab.exists():
            remove_readonly(None, str(target_vocab), None)
        os.replace(str(temp_target), str(target_vocab))
        log_info(
            f"Staged {target_vocab.name} ({target_vocab.stat().st_size:,} bytes, {EXPECTED_VOCAB_LINES:,} tokens)."
        )

    target_qwen35 = ROOT_DIR / "qwen35.tiktoken"
    if not force and verify_tiktoken_file(target_qwen35, 240000):
        log_info(f"Authoritative Qwen3.5 vocabulary converged: {target_qwen35.name}")
        return

    log_stage("Hydrating Authoritative Qwen3.5 Vocabulary (qwen35.tiktoken)")
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_json = TMP_DIR / f"tokenizer_qwen35.{os.getpid()}.json"
    tmp_tik = TMP_DIR / f"qwen35.tiktoken.{os.getpid()}.tmp"

    hub_dir = HF_CACHE_DIR / "hub"
    if hub_dir.is_dir():
        for local_tok_json in sorted(
            hub_dir.glob("models--OpenMOSS-Team--*/snapshots/*/tokenizer/tokenizer.json")
        ):
            try:
                resolved_json = local_tok_json.resolve(strict=True)
                count = _convert_tokenizer_json_to_tiktoken(resolved_json, tmp_tik)
                if count >= 240000 and verify_tiktoken_file(tmp_tik, 240000):
                    if target_qwen35.exists():
                        remove_readonly(None, str(target_qwen35), None)
                    os.replace(str(tmp_tik), str(target_qwen35))
                    log_info(
                        f"Staged {target_qwen35.name} from shared HF_HOME cache ({target_qwen35.stat().st_size:,} bytes, {count:,} tokens)."
                    )
                    return
            except Exception:
                tmp_tik.unlink(missing_ok=True)

    for url in QWEN35_TOKENIZER_JSON_ENDPOINTS:
        try:
            fetch_payload(url, tmp_json, timeout=180)
            count = _convert_tokenizer_json_to_tiktoken(tmp_json, tmp_tik)
            if count >= 240000 and verify_tiktoken_file(tmp_tik, 240000):
                if target_qwen35.exists():
                    remove_readonly(None, str(target_qwen35), None)
                os.replace(str(tmp_tik), str(target_qwen35))
                log_info(
                    f"Staged {target_qwen35.name} ({target_qwen35.stat().st_size:,} bytes, {count:,} tokens)."
                )
                break
        except Exception as exc:
            log_warn(f"Optional Qwen3.5 vocabulary fetch skipped ({url}): {exc}")
        finally:
            tmp_json.unlink(missing_ok=True)
            tmp_tik.unlink(missing_ok=True)


def verify_archive_integrity(archive_path: Path) -> bool:
    if not archive_path.is_file() or archive_path.stat().st_size < 512:
        return False
    st = archive_path.stat()
    sig = (archive_path.name, st.st_size, st.st_mtime_ns)
    if sig in _VERIFIED_ARCHIVE_SIGNATURES:
        return True
    name_lower = archive_path.name.lower()
    try:
        if name_lower.endswith((".whl", ".zip")):
            with zipfile.ZipFile(archive_path, "r") as zf:
                if zf.testzip() is not None:
                    return False
        elif name_lower.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path, "r:gz") as tf:
                for member in tf.getmembers():
                    if member.isfile():
                        f_obj = tf.extractfile(member)
                        if f_obj is not None:
                            while f_obj.read(65536):
                                pass
        _VERIFIED_ARCHIVE_SIGNATURES.add(sig)
        return True
    except Exception:
        return False


def audit_wheels_dir() -> int:
    WHEELS_DIR.mkdir(parents=True, exist_ok=True)
    evicted = 0
    for f in sorted(WHEELS_DIR.iterdir()):
        if f.is_file() and not verify_archive_integrity(f):
            remove_readonly(None, str(f), None)
            f.unlink(missing_ok=True)
            evicted += 1
    return evicted


def stage_and_install_packages(
    isolated_env: dict[str, str], packages: list[str], max_attempts: int = 3
) -> None:
    find_links_args = ["--find-links", str(WHEELS_DIR)]
    if PARENT_WHEELS_DIR.is_dir() and PARENT_WHEELS_DIR != WHEELS_DIR:
        find_links_args.extend(["--find-links", str(PARENT_WHEELS_DIR)])

    for attempt in range(1, max_attempts + 1):
        audit_wheels_dir()
        dl_cmd = (
            [
                str(HERMETIC_PYTHON_EXE),
                "-I",
                "-m",
                "pip",
                "download",
                "--dest",
                str(WHEELS_DIR),
            ]
            + find_links_args
            + [
                "--cache-dir",
                str(PIP_CACHE_DIR),
                "--progress-bar",
                "off",
                "--retries",
                "10",
                "--timeout",
                "120",
                "--pre",
            ]
            + packages
        )
        dl_res = subprocess.run(dl_cmd, env=isolated_env, stdin=subprocess.DEVNULL)
        evicted = audit_wheels_dir()
        if dl_res.returncode != 0 or evicted > 0:
            if attempt < max_attempts:
                time.sleep(1.5 * attempt)
                continue
            dl_res.check_returncode()

        inst_cmd = (
            [
                str(HERMETIC_PYTHON_EXE),
                "-I",
                "-m",
                "pip",
                "install",
                "--no-index",
            ]
            + find_links_args
            + [
                "--no-warn-script-location",
                "--no-compile",
                "--progress-bar",
                "off",
                "--no-cache-dir",
                "--pre",
                "-U",
            ]
            + packages
        )
        inst_res = subprocess.run(inst_cmd, env=isolated_env, stdin=subprocess.DEVNULL)
        if inst_res.returncode == 0:
            return
        audit_wheels_dir()
        if attempt == max_attempts:
            inst_res.check_returncode()


def resolve_python_runtime(requested_version: str | None = None) -> tuple[str, str]:
    if IS_WIN:
        if requested_version:
            candidate_url = f"{HTTPS}www.python.org/ftp/python/{requested_version}/python-{requested_version}-embed-amd64.zip"
            if probe_url(candidate_url):
                return requested_version, candidate_url
        try:
            html = fetch_http_text(f"{HTTPS}www.python.org/ftp/python/", timeout=20)
            matches = re.findall(r'href="(\d+\.\d+\.\d+)/"', html)
            preferred = [
                v for v in matches if parse_semver(v)[:2] in ((3, 12), (3, 13), (3, 14))
            ]
            pool = preferred if preferred else matches
            for ver in sorted(set(pool), key=parse_semver, reverse=True):
                candidate_url = f"{HTTPS}www.python.org/ftp/python/{ver}/python-{ver}-embed-amd64.zip"
                if probe_url(candidate_url):
                    return ver, candidate_url
        except Exception as exc:
            log_warn(f"Python FTP index query failed: {exc}")
    else:
        arch = "x86_64" if sys.maxsize > 2**32 else "i686"
        triple = f"{arch}-unknown-linux-gnu" if IS_LINUX else f"{arch}-apple-darwin"
        releases = fetch_github_releases(
            "astral-sh/python-build-standalone", per_page=10
        )
        for rel in releases:
            assets = rel.get("assets", [])
            matching_assets = []
            for asset in assets:
                name = asset.get("name", "")
                if (
                    triple in name
                    and "install_only" in name
                    and name.endswith(".tar.gz")
                ):
                    if (
                        requested_version is None
                        or f"cpython-{requested_version}" in name
                    ):
                        ver_match = re.search(r"cpython-(\d+\.\d+\.\d+)", name)
                        ver_str = (
                            ver_match.group(1)
                            if ver_match
                            else rel.get("tag_name", "0")
                        )
                        matching_assets.append(
                            (ver_str, asset.get("browser_download_url", ""))
                        )
            if matching_assets:
                matching_assets.sort(
                    key=lambda item: parse_semver(item[0]), reverse=True
                )
                best_ver, best_url = matching_assets[0]
                if best_url and probe_url(best_url):
                    return best_ver, best_url
    cached = read_version_marker(PYTHON_DIR)
    if HERMETIC_PYTHON_EXE.exists():
        if not cached:
            res = subprocess.run(
                [
                    str(HERMETIC_PYTHON_EXE),
                    "-c",
                    "import platform; print(platform.python_version())",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            cached = res.stdout.strip() or "local"
            write_version_marker(PYTHON_DIR, cached)
        return cached, ""
    raise RuntimeError("Failed to dynamically resolve upstream Python runtime.")


def bootstrap_hermetic_python(
    requested_version: str | None = None, force: bool = False
) -> None:
    isolated_env = build_isolated_env()
    py_ver, py_url = resolve_python_runtime(requested_version)
    if py_url and not is_version_current(
        PYTHON_DIR, py_ver, HERMETIC_PYTHON_EXE, force
    ):
        log_stage(f"Hydrating Hermetic Python Runtime ({py_ver})")
        safe_rmtree(PYTHON_DIR)
        PYTHON_DIR.mkdir(parents=True, exist_ok=True)
        if IS_WIN:
            archive_path = TMP_DIR / "python_embed.zip"
            fetch_payload(py_url, archive_path)
            unpack_archive(archive_path, PYTHON_DIR, flatten_single_dir=False)
            archive_path.unlink(missing_ok=True)
            enforce_pth_sandbox()
        else:
            archive_path = TMP_DIR / "python_embed.tar.gz"
            fetch_payload(py_url, archive_path)
            temp_ext = TMP_DIR / "python_ext"
            unpack_archive(archive_path, temp_ext, flatten_single_dir=False)
            source_extracted = temp_ext / "python"
            if source_extracted.exists():
                for item in source_extracted.iterdir():
                    shutil.move(str(item), str(PYTHON_DIR / item.name))
            safe_rmtree(temp_ext)
            archive_path.unlink(missing_ok=True)
            os.chmod(HERMETIC_PYTHON_EXE, 0o755)
        pip_bootstrapper = TMP_DIR / "get-pip.py"
        fetch_payload(f"{HTTPS}bootstrap.pypa.io/get-pip.py", pip_bootstrapper)
        subprocess.run(
            [
                str(HERMETIC_PYTHON_EXE),
                "-I",
                str(pip_bootstrapper),
                "--no-warn-script-location",
                "--cache-dir",
                str(PIP_CACHE_DIR),
            ],
            env=isolated_env,
            stdin=subprocess.DEVNULL,
            check=True,
        )
        pip_bootstrapper.unlink(missing_ok=True)
        write_version_marker(PYTHON_DIR, py_ver)
    else:
        enforce_pth_sandbox()
        log_info(f"Hermetic Python runtime converged at ({py_ver}).")


def synchronize_python_packages(force: bool = False) -> None:
    isolated_env = build_isolated_env()
    pkg_marker = PYTHON_DIR / ".packages_converged"
    if not force and pkg_marker.exists():
        log_info("Reference tiktoken & numpy packages converged.")
        return
    log_info("Staging reference tiktoken & numpy in hermetic Python (--pre -U)...")
    stage_and_install_packages(
        isolated_env,
        ["pip", "setuptools", "wheel", "packaging", "tiktoken", "numpy"],
    )
    pkg_marker.write_text("converged\n", encoding="utf-8")


def resolve_release_asset(
    repo: str, target_dir: Path, predicate, per_page: int = 25
) -> tuple[str, str]:
    releases = fetch_github_releases(repo, per_page=per_page)
    for rel in releases:
        tag = rel.get("tag_name", "")
        for asset in rel.get("assets", []):
            name = asset.get("name", "")
            dl_url = asset.get("browser_download_url", "")
            if dl_url and predicate(name.lower()):
                if probe_url(dl_url):
                    return tag, dl_url
    cached = read_version_marker(target_dir)
    if cached:
        return cached, ""
    raise RuntimeError(f"Failed to dynamically resolve binary release asset for {repo}")


def resolve_nasm_url() -> tuple[str, str]:
    try:
        html = fetch_http_text(
            f"{HTTPS}www.nasm.us/pub/nasm/releasebuilds/", timeout=15
        )
        matches = re.findall(r'href="(\d+\.\d+(?:\.\d+)?)/"', html)
        for ver in sorted(set(matches), key=parse_semver, reverse=True):
            if IS_WIN:
                candidate_url = f"{HTTPS}www.nasm.us/pub/nasm/releasebuilds/{ver}/win64/nasm-{ver}-win64.zip"
            elif IS_MAC:
                candidate_url = f"{HTTPS}www.nasm.us/pub/nasm/releasebuilds/{ver}/macosx/nasm-{ver}-macosx.zip"
            else:
                candidate_url = f"{HTTPS}www.nasm.us/pub/nasm/releasebuilds/{ver}/linux/nasm-{ver}-linux.tar.xz"
            if probe_url(candidate_url):
                return ver, candidate_url
    except Exception as exc:
        log_warn(f"NASM index query failed: {exc}")
    cached = read_version_marker(TOOLCHAIN_DIR / "nasm")
    if cached and NASM_EXE.exists():
        return cached, ""
    raise RuntimeError("Failed to dynamically resolve upstream NASM release.")


def resolve_upstream_tool_urls() -> dict[str, tuple[str, str]]:
    def match_cmake(name: str) -> bool:
        if IS_WIN:
            return "windows-x86_64.zip" in name
        if IS_MAC:
            return "macos-universal" in name and name.endswith(".tar.gz")
        return "linux-x86_64.tar.gz" in name

    def match_ninja(name: str) -> bool:
        if IS_WIN:
            return name == "ninja-win.zip"
        if IS_MAC:
            return name == "ninja-mac.zip"
        return name == "ninja-linux.zip"

    def match_llvm(name: str) -> bool:
        if not name.endswith(".tar.xz"):
            return False
        if IS_WIN:
            return "x86_64-pc-windows-msvc" in name and (
                "clang+llvm" in name or "llvm" in name
            )
        if IS_MAC:
            arch_tok = (
                "arm64-apple-darwin"
                if "arm" in HOST_ARCH
                else "x86_64-apple-darwin"
            )
            return arch_tok in name and "clang+llvm" in name
        return "x86_64-linux-gnu" in name and "clang+llvm" in name

    def match_git(name: str) -> bool:
        return (
            name.startswith("mingit-")
            and name.endswith("-64-bit.zip")
            and "busybox" not in name
        )

    return {
        "cmake": resolve_release_asset(
            "Kitware/CMake", TOOLCHAIN_DIR / "cmake", match_cmake
        ),
        "ninja": resolve_release_asset(
            "ninja-build/ninja", TOOLCHAIN_DIR / "ninja", match_ninja
        ),
        "llvm": resolve_release_asset(
            "llvm/llvm-project", TOOLCHAIN_DIR / "llvm", match_llvm, per_page=30
        ),
        "nasm": resolve_nasm_url(),
        "git": (
            resolve_release_asset(
                "git-for-windows/git", TOOLCHAIN_DIR / "git", match_git
            )
            if IS_WIN
            else ("", "")
        ),
    }


def hydrate_toolchains(force: bool = False) -> None:
    log_stage("Resolving and Converging Upstream Host Toolchains")
    tools_map = resolve_upstream_tool_urls()

    git_tag, git_url = tools_map["git"]
    if IS_WIN and git_tag:
        if git_url and not is_version_current(
            TOOLCHAIN_DIR / "git", git_tag, GIT_CMD_EXE, force
        ):
            archive = TMP_DIR / "mingit.zip"
            fetch_payload(git_url, archive)
            unpack_archive(
                archive, TOOLCHAIN_DIR / "git", flatten_single_dir=False
            )
            archive.unlink(missing_ok=True)
            write_version_marker(TOOLCHAIN_DIR / "git", git_tag)
        else:
            log_info(f"MinGit converged ({git_tag}).")

    cmake_tag, cmake_url = tools_map["cmake"]
    if cmake_url and not is_version_current(
        TOOLCHAIN_DIR / "cmake", cmake_tag, CMAKE_EXE, force
    ):
        archive = TMP_DIR / (
            "cmake.zip" if cmake_url.endswith(".zip") else "cmake.tar.gz"
        )
        fetch_payload(cmake_url, archive)
        temp_ext = TMP_DIR / "cmake_ext"
        unpack_archive(archive, temp_ext, flatten_single_dir=False)
        inner = next(temp_ext.iterdir())
        if IS_MAC and (inner / "CMake.app").exists():
            inner = inner / "CMake.app" / "Contents"
        promote_directory(inner, TOOLCHAIN_DIR / "cmake")
        safe_rmtree(temp_ext)
        archive.unlink(missing_ok=True)
        write_version_marker(TOOLCHAIN_DIR / "cmake", cmake_tag)
    else:
        log_info(f"CMake converged ({cmake_tag}).")

    ninja_tag, ninja_url = tools_map["ninja"]
    if ninja_url and not is_version_current(
        TOOLCHAIN_DIR / "ninja", ninja_tag, NINJA_EXE, force
    ):
        archive = TMP_DIR / "ninja.zip"
        fetch_payload(ninja_url, archive)
        unpack_archive(
            archive, TOOLCHAIN_DIR / "ninja", flatten_single_dir=False
        )
        archive.unlink(missing_ok=True)
        if not IS_WIN:
            os.chmod(NINJA_EXE, 0o755)
        write_version_marker(TOOLCHAIN_DIR / "ninja", ninja_tag)
    else:
        log_info(f"Ninja converged ({ninja_tag}).")

    llvm_tag, llvm_url = tools_map["llvm"]
    llvm_marker = CLANG_CL_EXE if IS_WIN else CLANG_EXE
    if llvm_url and not is_version_current(
        TOOLCHAIN_DIR / "llvm", llvm_tag, llvm_marker, force
    ):
        archive = TMP_DIR / "llvm.tar.xz"
        fetch_payload(llvm_url, archive, timeout=900)
        temp_ext = TMP_DIR / "llvm_ext"
        unpack_archive(archive, temp_ext, flatten_single_dir=False)
        inner = next(temp_ext.iterdir())
        promote_directory(inner, TOOLCHAIN_DIR / "llvm")
        safe_rmtree(temp_ext)
        archive.unlink(missing_ok=True)
        if not IS_WIN:
            os.chmod(CLANG_EXE, 0o755)
            if LLD_LINK_EXE.exists():
                os.chmod(LLD_LINK_EXE, 0o755)
        write_version_marker(TOOLCHAIN_DIR / "llvm", llvm_tag)
    else:
        log_info(f"LLVM/Clang converged ({llvm_tag}).")

    nasm_tag, nasm_url = tools_map["nasm"]
    if nasm_url and not is_version_current(
        TOOLCHAIN_DIR / "nasm", nasm_tag, NASM_EXE, force
    ):
        archive = TMP_DIR / (
            "nasm.zip" if nasm_url.endswith(".zip") else "nasm.tar.xz"
        )
        fetch_payload(nasm_url, archive)
        temp_ext = TMP_DIR / "nasm_ext"
        unpack_archive(archive, temp_ext, flatten_single_dir=True)
        nasm_dest = TOOLCHAIN_DIR / "nasm"
        nasm_dest.mkdir(parents=True, exist_ok=True)
        for item in temp_ext.rglob("*"):
            if item.is_file() and item.name.lower() in [
                "nasm.exe",
                "nasm",
                "ndisasm.exe",
                "ndisasm",
            ]:
                shutil.copy2(to_unc_path(item), to_unc_path(nasm_dest / item.name))
        safe_rmtree(temp_ext)
        archive.unlink(missing_ok=True)
        if not IS_WIN:
            os.chmod(NASM_EXE, 0o755)
        write_version_marker(nasm_dest, nasm_tag)
    else:
        log_info(f"NASM converged ({nasm_tag}).")


def discover_latest_vs_channel() -> tuple[str, str]:
    for major in range(25, 14, -1):
        for track in ("release", "pre"):
            endpoint = f"{HTTPS}aka.ms/vs/{major}/{track}/channel"
            try:
                ch_data = fetch_json(endpoint, timeout=15)
                if not isinstance(ch_data, dict):
                    continue
                build_ver = ch_data.get("info", {}).get("buildVersion", "")
                for item in ch_data.get("channelItems", []):
                    item_type = item.get("type", "").lower()
                    item_id = item.get("id", "").lower()
                    if item_type == "manifest" or "manifest" in item_id:
                        ver = item.get("version") or build_ver or f"{major}.0"
                        for payload in item.get("payloads", []):
                            url = payload.get("url", "")
                            if url.endswith((".json", ".vsman")):
                                log_info(
                                    f"Locked Visual Studio Channel v{major} ({track}) -> Build {ver}"
                                )
                                return ver, url
            except Exception:
                continue
    raise RuntimeError(
        "Unable to dynamically discover any upstream Visual Studio channel manifest."
    )


def stream_vs_manifest_catalog(manifest_url: str) -> dict:
    log_info(f"Streaming Visual Studio manifest catalog: {manifest_url}")
    manifest_tmp = TMP_DIR / "vs_catalog.tmp"
    fetch_payload(manifest_url, manifest_tmp, timeout=300)
    try:
        with open(manifest_tmp, "rb") as mf:
            header = mf.read(2)
        if header == b"\x1f\x8b":
            with gzip.open(
                manifest_tmp, "rt", encoding="utf-8", errors="ignore"
            ) as gf:
                catalog = json.load(gf)
        else:
            with open(manifest_tmp, "r", encoding="utf-8", errors="ignore") as jf:
                catalog = json.load(jf)
        return catalog
    finally:
        manifest_tmp.unlink(missing_ok=True)


def resolve_latest_nuget_packages() -> list[tuple[str, str, str]]:
    resolved: list[tuple[str, str, str]] = []
    for pkg_id in NUGET_PACKAGES:
        index_url = f"{HTTPS}api.nuget.org/v3-flatcontainer/{pkg_id}/index.json"
        data = fetch_json(index_url, timeout=20)
        versions = data.get("versions", [])
        if not versions:
            raise RuntimeError(
                f"NuGet flatcontainer returned no versions for {pkg_id}"
            )
        latest_ver = sorted(versions, key=parse_semver)[-1]
        nupkg_url = f"{HTTPS}api.nuget.org/v3-flatcontainer/{pkg_id}/{latest_ver}/{pkg_id}.{latest_ver}.nupkg"
        resolved.append((pkg_id, latest_ver, nupkg_url))
    return resolved


def extract_and_stage_vs_payload(pkg: dict, kind: str, dest_dir: Path) -> None:
    payloads = pkg.get("payloads", [])
    vsix_payloads = [
        p
        for p in payloads
        if not p.get("fileName", "").lower().endswith((".msi", ".cab"))
    ]
    for payload in vsix_payloads:
        url = payload.get("url")
        if not url:
            continue
        file_name = payload.get("fileName", "payload.vsix")
        payload_file = TMP_DIR / file_name
        fetch_payload(url, payload_file, timeout=180)
        ext_dir = TMP_DIR / f"ext_{payload_file.stem}"
        try:
            unpack_archive(payload_file, ext_dir, flatten_single_dir=False)
            if kind == "crt_headers":
                matches = [
                    p.parent for p in ext_dir.rglob("vcruntime.h")
                ] or [p.parent for p in ext_dir.rglob("stdio.h")]
                copy_tree_contents(matches[0] if matches else ext_dir, dest_dir)
            elif kind == "crt_libs":
                for lib in ext_dir.rglob("*.lib"):
                    parts = [p.lower() for p in lib.parts]
                    if any(
                        excl in parts
                        for excl in (
                            "arm",
                            "arm64",
                            "x86",
                            "onecore",
                            "store",
                            "spectre",
                            "enclave",
                            "ucrt_enclave",
                        )
                    ):
                        continue
                    dest_lib = dest_dir / lib.name
                    if dest_lib.exists():
                        remove_readonly(None, str(dest_lib), None)
                    shutil.copy2(to_unc_path(lib), to_unc_path(dest_lib))
        finally:
            safe_rmtree(ext_dir)
            payload_file.unlink(missing_ok=True)


def _decontaminate_sdk_um_libs() -> None:
    kits_lib = WINSDK_DIR / "Windows Kits" / "10" / "Lib"
    if not kits_lib.exists():
        return
    for um_dir in kits_lib.glob("*/um/x64"):
        for bad_ucrt in um_dir.glob("*ucrt*.lib"):
            remove_readonly(None, str(bad_ucrt), None)
            bad_ucrt.unlink(missing_ok=True)


def audit_sdk_fakeroot() -> bool:
    if not WINSDK_DIR.exists():
        return False
    _decontaminate_sdk_um_libs()
    stdio_headers = list(
        WINSDK_DIR.glob("Windows Kits/10/Include/*/ucrt/stdio.h")
    )
    if not stdio_headers:
        return False
    found_libs = {anchor: False for anchor in SDK_REQUIRED_ANCHORS}
    for lib_path in (
        (WINSDK_DIR / "Windows Kits").rglob("*.lib")
        if (WINSDK_DIR / "Windows Kits").exists()
        else []
    ):
        name_lower = lib_path.name.lower()
        if name_lower in found_libs and lib_path.stat().st_size > 0:
            found_libs[name_lower] = True
    return all(found_libs.values())


def audit_msvc_fakeroot() -> bool:
    if not WINSDK_DIR.exists():
        return False
    vcruntime_headers = list(
        WINSDK_DIR.glob("VC/Tools/MSVC/*/include/vcruntime.h")
    )
    if not vcruntime_headers:
        return False
    found_libs = {anchor: False for anchor in MSVC_REQUIRED_ANCHORS}
    for lib_path in (
        (WINSDK_DIR / "VC").rglob("*.lib")
        if (WINSDK_DIR / "VC").exists()
        else []
    ):
        name_lower = lib_path.name.lower()
        if name_lower in found_libs and lib_path.stat().st_size > 0:
            found_libs[name_lower] = True
    return all(found_libs.values())


def hydrate_winsdk(force: bool = False) -> None:
    if not IS_WIN:
        return
    log_stage("Resolving Latest Upstream Windows SDK & Visual Studio MSVC Toolset")
    nuget_specs = resolve_latest_nuget_packages()
    raw_sdk_version = nuget_specs[0][1]
    clean_sdk_version = normalize_numeric_version(raw_sdk_version)
    sdk_composite_tag = "|".join(f"{p}:{v}" for p, v, _ in nuget_specs)
    vs_channel_ver, vs_manifest_url = discover_latest_vs_channel()

    sdk_valid = audit_sdk_fakeroot()
    msvc_valid = audit_msvc_fakeroot()
    current_sdk_tag = read_version_marker(WINSDK_DIR, ".sdk_version")
    current_msvc_tag = read_version_marker(WINSDK_DIR, ".msvc_version")
    sdk_needs_update = force or not sdk_valid or (current_sdk_tag != sdk_composite_tag)
    msvc_needs_update = force or not msvc_valid or (current_msvc_tag != vs_channel_ver)

    if not sdk_needs_update and not msvc_needs_update:
        log_info(
            f"Windows SDK ({raw_sdk_version}) and MSVC Channel ({vs_channel_ver}) converged."
        )
        return

    if sdk_needs_update:
        log_stage(
            f"Hydrating Windows SDK ({raw_sdk_version} -> {clean_sdk_version}) via NuGet Flatcontainer"
        )
        sdk_inc_target = (
            WINSDK_DIR / "Windows Kits" / "10" / "Include" / clean_sdk_version
        )
        sdk_lib_target = (
            WINSDK_DIR / "Windows Kits" / "10" / "Lib" / clean_sdk_version
        )
        os.makedirs(to_unc_path(sdk_inc_target), exist_ok=True)
        os.makedirs(to_unc_path(sdk_lib_target), exist_ok=True)
        header_buckets = ["ucrt", "um", "shared", "winrt", "cppwinrt"]
        for pkg_id, pkg_ver, url in nuget_specs:
            nupkg_path = TMP_DIR / f"{pkg_id}.{pkg_ver}.nupkg"
            ext_dir = TMP_DIR / f"ext_{pkg_id}"
            try:
                fetch_payload(url, nupkg_path, timeout=180)
                unpack_archive(nupkg_path, ext_dir, flatten_single_dir=False)
                for item in ext_dir.rglob("*"):
                    if not item.is_file():
                        continue
                    parts_lower = [p.lower() for p in item.parts]
                    file_name_lower = item.name.lower()
                    if any(
                        excl in parts_lower
                        for excl in (
                            "ucrt_enclave",
                            "enclave",
                            "onecore",
                            "store",
                            "arm",
                            "arm64",
                            "x86",
                        )
                    ):
                        continue
                    if file_name_lower.endswith(".lib") and (
                        "x64" in parts_lower or "amd64" in parts_lower
                    ):
                        if "ucrt" in parts_lower:
                            bucket = "ucrt"
                        elif "um" in parts_lower:
                            bucket = "um"
                        else:
                            continue
                        target = sdk_lib_target / bucket / "x64" / item.name
                        os.makedirs(to_unc_path(target.parent), exist_ok=True)
                        if target.exists():
                            remove_readonly(None, str(target), None)
                        shutil.copy2(to_unc_path(item), to_unc_path(target))
                    elif file_name_lower.endswith(
                        (".h", ".hpp", ".inl", ".idl", ".rh")
                    ):
                        found_bucket = next(
                            (p for p in parts_lower if p in header_buckets), None
                        )
                        if found_bucket:
                            b_idx = parts_lower.index(found_bucket)
                            rel_sub = Path(*item.parts[b_idx + 1 :])
                            target = sdk_inc_target / found_bucket / rel_sub
                        else:
                            target = (
                                sdk_inc_target / "um" / item.relative_to(ext_dir)
                            )
                        os.makedirs(to_unc_path(target.parent), exist_ok=True)
                        if target.exists():
                            remove_readonly(None, str(target), None)
                        shutil.copy2(to_unc_path(item), to_unc_path(target))
            finally:
                safe_rmtree(ext_dir)
                nupkg_path.unlink(missing_ok=True)
        _decontaminate_sdk_um_libs()
        write_version_marker(WINSDK_DIR, sdk_composite_tag, ".sdk_version")

    if msvc_needs_update:
        catalog = stream_vs_manifest_catalog(vs_manifest_url)
        catalog_packages = catalog.get("packages", [])

        def _extract_vc_toolset_ver(pkg_id_str: str) -> tuple[int, ...]:
            m = re.search(
                r"microsoft\.vc\.(\d+\.\d+\.\d+(?:\.\d+)?)",
                pkg_id_str,
                re.IGNORECASE,
            )
            return parse_semver(m.group(1)) if m else (0,)

        all_crt_headers = [
            p
            for p in catalog_packages
            if p.get("id", "").lower().startswith("microsoft.vc.")
            and p.get("id", "").lower().endswith(".crt.headers.base")
        ]
        if not all_crt_headers:
            raise RuntimeError(
                "No Microsoft.VC CRT header packages found in upstream Visual Studio catalog."
            )
        all_crt_headers.sort(
            key=lambda p: (
                _extract_vc_toolset_ver(p.get("id", "")),
                parse_semver(p.get("version", "0")),
            )
        )
        best_header_pkg = all_crt_headers[-1]
        best_id = best_header_pkg.get("id", "")
        id_ver_match = re.search(
            r"microsoft\.vc\.(\d+\.\d+\.\d+(?:\.\d+)?)", best_id, re.IGNORECASE
        )
        msvc_version = (
            id_ver_match.group(1)
            if id_ver_match
            else best_header_pkg.get("version", vs_channel_ver)
        )
        target_prefix = f"microsoft.vc.{msvc_version.lower()}.crt."
        crt_headers_pkgs = [
            p
            for p in all_crt_headers
            if p.get("id", "").lower().startswith(target_prefix)
        ]
        crt_libs_pkgs = []
        for pkg in catalog_packages:
            pkg_id = pkg.get("id", "").lower()
            if not pkg_id.startswith(target_prefix):
                continue
            if any(excl in pkg_id for excl in (".spectre", ".arm", ".x86.")):
                continue
            if ("crt.x64" in pkg_id or "crt.desktop" in pkg_id) and pkg.get(
                "payloads"
            ):
                crt_libs_pkgs.append(pkg)

        log_stage(
            f"Hydrating Latest MSVC Toolset ({msvc_version}) [{len(crt_headers_pkgs)} header pkg, {len(crt_libs_pkgs)} x64 lib pkgs]"
        )
        msvc_inc_target = (
            WINSDK_DIR / "VC" / "Tools" / "MSVC" / msvc_version / "include"
        )
        msvc_lib_target = (
            WINSDK_DIR / "VC" / "Tools" / "MSVC" / msvc_version / "lib" / "x64"
        )
        os.makedirs(to_unc_path(msvc_inc_target), exist_ok=True)
        os.makedirs(to_unc_path(msvc_lib_target), exist_ok=True)
        for pkg in crt_headers_pkgs:
            extract_and_stage_vs_payload(pkg, "crt_headers", msvc_inc_target)
        for pkg in crt_libs_pkgs:
            extract_and_stage_vs_payload(pkg, "crt_libs", msvc_lib_target)
        if not audit_msvc_fakeroot():
            raise RuntimeError(
                "MSVC CRT hydration incomplete: mandatory CRT headers or libraries missing."
            )
        write_version_marker(WINSDK_DIR, vs_channel_ver, ".msvc_version")

    write_version_marker(WINSDK_DIR, f"{vs_channel_ver}:{clean_sdk_version}")


def resolve_pcre2_asset(target_dir: Path) -> tuple[str, str]:
    def match_pcre2(name: str) -> bool:
        if not name.startswith("pcre2-"):
            return False
        if IS_WIN:
            return name.endswith(".zip") and "win" not in name
        return name.endswith(".tar.gz")

    return resolve_release_asset(
        "PCRE2Project/pcre2", target_dir, match_pcre2, per_page=15
    )


def hydrate_libraries(force: bool = False) -> None:
    log_stage("Resolving and Converging Upstream C/C++ Libraries (PCRE2)")
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    pcre2_dir = LIBRARY_DIR / "pcre2"
    pcre2_tag, pcre2_url = resolve_pcre2_asset(pcre2_dir)
    pcre2_marker = (
        pcre2_dir / "src" / "pcre2.h.generic"
        if (pcre2_dir / "src" / "pcre2.h.generic").exists()
        else pcre2_dir / "src" / "pcre2.h"
    )
    if pcre2_url and not is_version_current(
        pcre2_dir, pcre2_tag, pcre2_marker, force
    ):
        archive = TMP_DIR / (
            "pcre2.zip" if pcre2_url.endswith(".zip") else "pcre2.tar.gz"
        )
        fetch_payload(pcre2_url, archive, timeout=180)
        unpack_archive(archive, pcre2_dir, flatten_single_dir=True)
        archive.unlink(missing_ok=True)
        write_version_marker(pcre2_dir, pcre2_tag)
    else:
        log_info(f"PCRE2 converged ({pcre2_tag}).")


def orchestrate_hermetic_sitecustomize() -> None:
    if IS_WIN:
        site_packages_dirs = [PYTHON_DIR / "Lib" / "site-packages"]
    else:
        site_packages_dirs = list((PYTHON_DIR / "lib").glob("python*/site-packages"))

    sitecustomize_code = f"""from __future__ import annotations

import os
from pathlib import Path
import sys

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[{4 if IS_WIN else 5}]
_TC = _ROOT / "toolchain"
_PY = _TC / "python"
_CACHE = _ROOT / ".forge_cache"


def _env_default(key: str, val: str) -> None:
    cur = os.environ.get(key)
    if cur is None or not cur.strip():
        os.environ[key] = val


_env_default("PYTHONNOUSERSITE", "1")
_env_default("PYTHONUTF8", "1")
_env_default("PYTHONIOENCODING", "utf-8")
_env_default("TEMP", str(_CACHE / "tmp"))
_env_default("TMP", str(_CACHE / "tmp"))
_env_default("TMPDIR", str(_CACHE / "tmp"))

_hermetic_bins = [
    str(_PY if sys.platform == "win32" else _PY / "bin"),
    str(_PY / "Scripts" if sys.platform == "win32" else _PY / "bin"),
    str(_TC / "llvm" / "bin"),
    str(_TC / "cmake" / "bin"),
    str(_TC / "ninja"),
    str(_TC / "nasm"),
]
_cur_path = os.environ.get("PATH", "")
os.environ["PATH"] = os.pathsep.join(
    _hermetic_bins + ([_cur_path] if _cur_path else [])
)

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    for _dll_dir in (_ROOT, _ROOT / "dist" / "bin"):
        if _dll_dir.is_dir():
            try:
                os.add_dll_directory(str(_dll_dir))
            except Exception:
                pass
"""
    for sp_dir in site_packages_dirs:
        sp_dir.mkdir(parents=True, exist_ok=True)
        (sp_dir / "sitecustomize.py").write_text(sitecustomize_code, encoding="utf-8")


def clean_hydration() -> None:
    log_stage("Purging Hydrated Toolchains, Libraries, and Caches (--clean)")
    safe_rmtree(TOOLCHAIN_DIR)
    safe_rmtree(LIBRARY_DIR)
    safe_rmtree(CACHE_DIR)
    log_info("VizToken returned to unhydrated baseline.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VizToken Autonomous State-Convergent Hydrator"
    )
    parser.add_argument(
        "-p",
        "--python",
        type=str,
        default=None,
        help="Target Python version override (e.g. 3.12.8 or 3.14.10)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force full re-download regardless of .version match",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Purge all hydrated toolchains, libraries, and caches",
    )
    args, _ = parser.parse_known_args()
    if args.clean:
        clean_hydration()
        return

    TOOLCHAIN_DIR.mkdir(parents=True, exist_ok=True)
    (NATIVE_SYSROOT / "include").mkdir(parents=True, exist_ok=True)
    (NATIVE_SYSROOT / "lib").mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    bootstrap_hermetic_python(requested_version=args.python, force=args.force)

    current_exe = Path(sys.executable).resolve()
    target_exe = HERMETIC_PYTHON_EXE.resolve()
    if current_exe != target_exe:
        log_info(
            f"Detaching from host Python ({sys.version.split()[0]}). Trampolining into hermetic runtime: {target_exe}"
        )
        isolated_env = build_isolated_env()
        cmd = [str(target_exe), str(Path(__file__).resolve())] + sys.argv[1:]
        res = subprocess.run(cmd, env=isolated_env)
        sys.exit(res.returncode)

    hydrate_vocabulary(force=args.force)
    synchronize_python_packages(force=args.force)
    hydrate_toolchains(force=args.force)
    hydrate_winsdk(force=args.force)
    hydrate_libraries(force=args.force)
    orchestrate_hermetic_sitecustomize()
    safe_rmtree(TMP_DIR)
    log_info(
        "VizToken hermetic hydration converged to latest upstream state. Zero external host dependencies remaining."
    )


if __name__ == "__main__":
    main()