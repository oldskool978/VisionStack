#!/usr/bin/env python3
from __future__ import annotations

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

ROOT_DIR = Path(__file__).parent.parent.resolve()
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

HOST_OS = sys.platform
HOST_ARCH = platform.machine().lower()
IS_WIN = HOST_OS == "win32"
IS_MAC = HOST_OS == "darwin"
IS_LINUX = HOST_OS.startswith("linux")

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

VS_CHANNEL_ENDPOINTS = [
    "https://aka.ms/vs/17/release/channel",
    "https://aka.ms/vs/17/release.ltsc.17.12/channel",
    "https://aka.ms/vs/17/release.ltsc.17.10/channel",
]

NUGET_SDK_CPP_URL = "https://www.nuget.org/api/v2/package/Microsoft.Windows.SDK.CPP"
NUGET_SDK_CPP_X64_URL = (
    "https://www.nuget.org/api/v2/package/Microsoft.Windows.SDK.CPP.x64"
)
NUGET_SDK_BUILDTOOLS_URL = (
    "https://www.nuget.org/api/v2/package/Microsoft.Windows.SDK.BuildTools"
)

VOCAB_ENDPOINTS = [
    "https://huggingface.co/m-a-p/YuE2-3B/resolve/main/qwen.tiktoken",
    "https://huggingface.co/Qwen/Qwen-7B/resolve/main/qwen.tiktoken",
]
EXPECTED_VOCAB_LINES = 151643

QWEN35_TOKENIZER_JSON_ENDPOINTS = [
    "https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/resolve/main/tokenizer/tokenizer.json",
    "https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/resolve/main/tokenizer/tokenizer.json",
]

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
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    ):
        if k in os.environ:
            env[k] = os.environ[k]

    sys_root = env.get("SYSTEMROOT", r"C:\Windows")
    bins = [
        str(PYTHON_DIR if IS_WIN else PYTHON_DIR / "bin"),
        str(PYTHON_DIR / "Scripts" if IS_WIN else PYTHON_DIR / "bin"),
        str(TOOLCHAIN_DIR / "llvm" / "bin"),
        str(TOOLCHAIN_DIR / "ninja"),
        str(TOOLCHAIN_DIR / "cmake" / "bin"),
    ]
    if IS_WIN:
        bins.extend([os.path.join(sys_root, "System32"), sys_root])
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
    return env


def parse_semver(ver_str: str) -> tuple[int, ...]:
    clean = re.sub(r"[^\d.]", "", ver_str)
    parts = [int(p) for p in clean.split(".") if p.isdigit()]
    return tuple(parts)


def probe_url(url: str) -> bool:
    clean_url = url.replace(" ", "%20")
    req = urllib.request.Request(
        clean_url, headers={"User-Agent": "viztoken-hydrator/5.0"}, method="HEAD"
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return 200 <= resp.status < 400
    except Exception:
        return False


def fetch_http_text(url: str, timeout: int = 30) -> str:
    clean_url = url.replace(" ", "%20")
    req = urllib.request.Request(
        clean_url,
        headers={
            "User-Agent": "viztoken-hydrator/5.0",
            "Accept-Encoding": "gzip, deflate",
        },
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


def fetch_json(url: str, timeout: int = 30) -> dict:
    text = fetch_http_text(url, timeout=timeout)
    stripped = text.lstrip()
    if stripped.startswith("<"):
        raise ValueError(f"Endpoint returned HTML payload: {url}")
    return json.loads(text)


def fetch_github_latest_release(repo: str) -> dict:
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    try:
        return fetch_json(url, timeout=15)
    except Exception as exc:
        log_warn(f"GitHub release query failed for {repo}: {exc}")
        return {}


def fetch_payload(url: str, destination: Path, timeout: int = 300) -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    clean_url = url.replace(" ", "%20")
    log_info(f"Downloading: {clean_url}")
    req = urllib.request.Request(
        clean_url, headers={"User-Agent": "viztoken-hydrator/5.0"}
    )
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


def verify_tiktoken_file(target_file: Path, min_lines: int = EXPECTED_VOCAB_LINES) -> bool:
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


def hydrate_vocabulary() -> None:
    target_vocab = ROOT_DIR / "qwen.tiktoken"
    if verify_tiktoken_file(target_vocab, EXPECTED_VOCAB_LINES):
        log_info(f"Authoritative base vocabulary present: {target_vocab.name}")
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
    if verify_tiktoken_file(target_qwen35, 240000):
        log_info(f"Authoritative Qwen3.5 vocabulary present: {target_qwen35.name}")
        return

    log_stage("Hydrating Authoritative Qwen3.5 Vocabulary (qwen35.tiktoken)")
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_json = TMP_DIR / f"tokenizer_qwen35.{os.getpid()}.json"
    tmp_tik = TMP_DIR / f"qwen35.tiktoken.{os.getpid()}.tmp"
    for url in QWEN35_TOKENIZER_JSON_ENDPOINTS:
        try:
            fetch_payload(url, tmp_json, timeout=180)
            payload = json.loads(tmp_json.read_text(encoding="utf-8"))
            vocab = payload.get("model", {}).get("vocab", {})
            if len(vocab) >= 240000:
                byte_dec = _gpt2_byte_decoder()
                sorted_vocab = sorted(vocab.items(), key=lambda kv: int(kv[1]))
                lines: list[str] = []
                for token_str, rank in sorted_vocab:
                    raw_bytes = bytes(
                        byte_dec[ch] for ch in token_str if ch in byte_dec
                    )
                    if len(raw_bytes) == len(token_str):
                        b64_tok = base64.b64encode(raw_bytes).decode("ascii")
                        lines.append(f"{b64_tok} {int(rank)}")
                tmp_tik.write_text("\n".join(lines) + "\n", encoding="utf-8")
                if verify_tiktoken_file(tmp_tik, 240000):
                    if target_qwen35.exists():
                        remove_readonly(None, str(target_qwen35), None)
                    os.replace(str(tmp_tik), str(target_qwen35))
                    log_info(
                        f"Staged {target_qwen35.name} ({target_qwen35.stat().st_size:,} bytes, {len(lines):,} tokens)."
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
    for attempt in range(1, max_attempts + 1):
        audit_wheels_dir()
        dl_cmd = [
            str(HERMETIC_PYTHON_EXE),
            "-I",
            "-m",
            "pip",
            "download",
            "--dest",
            str(WHEELS_DIR),
            "--find-links",
            str(WHEELS_DIR),
            "--cache-dir",
            str(PIP_CACHE_DIR),
            "--progress-bar",
            "off",
            "--retries",
            "10",
            "--timeout",
            "120",
            "--pre",
        ] + packages
        dl_res = subprocess.run(dl_cmd, env=isolated_env, stdin=subprocess.DEVNULL)
        evicted = audit_wheels_dir()
        if dl_res.returncode != 0 or evicted > 0:
            if attempt < max_attempts:
                time.sleep(1.5 * attempt)
                continue
            dl_res.check_returncode()

        inst_cmd = [
            str(HERMETIC_PYTHON_EXE),
            "-I",
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(WHEELS_DIR),
            "--no-warn-script-location",
            "--no-compile",
            "--progress-bar",
            "off",
            "--no-cache-dir",
            "--pre",
            "-U",
        ] + packages
        inst_res = subprocess.run(inst_cmd, env=isolated_env, stdin=subprocess.DEVNULL)
        if inst_res.returncode == 0:
            return
        audit_wheels_dir()
        if attempt == max_attempts:
            inst_res.check_returncode()


def resolve_python_runtime() -> tuple[str, str]:
    fallback_ver = "3.12.8"
    fallback_url = (
        f"https://www.python.org/ftp/python/{fallback_ver}/python-{fallback_ver}-embed-amd64.zip"
        if IS_WIN
        else f"https://github.com/astral-sh/python-build-standalone/releases/download/20241016/cpython-{fallback_ver}+20241016-x86_64-unknown-linux-gnu-install_only.tar.gz"
    )
    if IS_WIN:
        try:
            html = fetch_http_text("https://www.python.org/ftp/python/", timeout=15)
            matches = re.findall(r'href="(\d+\.\d+\.\d+)/"', html)
            preferred = [
                v for v in matches if parse_semver(v)[:2] in ((3, 12), (3, 13), (3, 14))
            ]
            sorted_vers = sorted(
                set(preferred if preferred else matches),
                key=parse_semver,
                reverse=True,
            )
            for ver in sorted_vers:
                candidate_url = f"https://www.python.org/ftp/python/{ver}/python-{ver}-embed-amd64.zip"
                if probe_url(candidate_url):
                    return ver, candidate_url
        except Exception:
            pass
        return fallback_ver, fallback_url
    else:
        arch = "x86_64" if sys.maxsize > 2**32 else "i686"
        triple = f"{arch}-unknown-linux-gnu" if IS_LINUX else f"{arch}-apple-darwin"
        try:
            release = fetch_github_latest_release("astral-sh/python-build-standalone")
            tag = release.get("tag_name", "20241016")
            for asset in release.get("assets", []):
                name = asset.get("name", "")
                if (
                    triple in name
                    and "install_only" in name
                    and name.endswith(".tar.gz")
                ):
                    download_url = asset.get("browser_download_url")
                    if probe_url(download_url):
                        return tag, download_url
        except Exception:
            pass
        return fallback_ver, fallback_url


def resolve_git_url() -> str:
    fallback_url = "https://github.com/git-for-windows/git/releases/download/v2.48.1.windows.1/MinGit-2.48.1-64-bit.zip"
    if not IS_WIN:
        return ""
    try:
        release = fetch_github_latest_release("git-for-windows/git")
        for asset in release.get("assets", []):
            name = asset.get("name", "")
            if name.startswith("MinGit-") and name.endswith("-64-bit.zip"):
                download_url = asset.get("browser_download_url")
                if probe_url(download_url):
                    return download_url
    except Exception:
        pass
    return fallback_url


def resolve_nasm_url() -> str:
    fallback_ver = "2.16.03"
    fallback_url = (
        f"https://www.nasm.us/pub/nasm/releasebuilds/{fallback_ver}/win64/nasm-{fallback_ver}-win64.zip"
        if IS_WIN
        else (
            f"https://www.nasm.us/pub/nasm/releasebuilds/{fallback_ver}/macosx/nasm-{fallback_ver}-macosx.zip"
            if IS_MAC
            else f"https://www.nasm.us/pub/nasm/releasebuilds/{fallback_ver}/linux/nasm-{fallback_ver}-linux.tar.xz"
        )
    )
    try:
        html = fetch_http_text("https://www.nasm.us/pub/nasm/releasebuilds/", timeout=15)
        matches = re.findall(r'href="(\d+\.\d+(?:\.\d+)?)/"', html)
        sorted_vers = sorted(set(matches), key=parse_semver, reverse=True)
        for ver in sorted_vers:
            if IS_WIN:
                candidate_url = f"https://www.nasm.us/pub/nasm/releasebuilds/{ver}/win64/nasm-{ver}-win64.zip"
            elif IS_MAC:
                candidate_url = f"https://www.nasm.us/pub/nasm/releasebuilds/{ver}/macosx/nasm-{ver}-macosx.zip"
            else:
                candidate_url = f"https://www.nasm.us/pub/nasm/releasebuilds/{ver}/linux/nasm-{ver}-linux.tar.xz"
            if probe_url(candidate_url):
                return candidate_url
    except Exception:
        pass
    return fallback_url


def resolve_pcre2_url() -> str:
    fallback_url = "https://github.com/PCRE2Project/pcre2/releases/download/pcre2-10.45/pcre2-10.45.zip"
    try:
        release = fetch_github_latest_release("PCRE2Project/pcre2")
        for asset in release.get("assets", []):
            name = asset.get("name", "")
            if IS_WIN and name.endswith(".zip"):
                download_url = asset.get("browser_download_url")
                if probe_url(download_url):
                    return download_url
            elif not IS_WIN and name.endswith(".tar.gz"):
                download_url = asset.get("browser_download_url")
                if probe_url(download_url):
                    return download_url
    except Exception:
        pass
    return fallback_url


def resolve_upstream_tool_urls() -> tuple[str, str, str, str, str]:
    cmake_release = fetch_github_latest_release("Kitware/CMake")
    cmake_url = ""
    if cmake_release:
        for asset in cmake_release.get("assets", []):
            name = asset.get("name", "").lower()
            if IS_WIN and "windows-x86_64.zip" in name:
                cmake_url = asset.get("browser_download_url")
                break
            elif IS_MAC and "macos-universal" in name:
                cmake_url = asset.get("browser_download_url")
                break
            elif IS_LINUX and "linux-x86_64.tar.gz" in name:
                cmake_url = asset.get("browser_download_url")
                break
    if not cmake_url:
        cmake_url = "https://github.com/Kitware/CMake/releases/download/v3.31.5/cmake-3.31.5-windows-x86_64.zip"

    ninja_release = fetch_github_latest_release("ninja-build/ninja")
    ninja_url = ""
    if ninja_release:
        for asset in ninja_release.get("assets", []):
            name = asset.get("name", "").lower()
            if IS_WIN and "ninja-win.zip" in name:
                ninja_url = asset.get("browser_download_url")
                break
            elif IS_MAC and "ninja-mac.zip" in name:
                ninja_url = asset.get("browser_download_url")
                break
            elif IS_LINUX and "ninja-linux.zip" in name:
                ninja_url = asset.get("browser_download_url")
                break
    if not ninja_url:
        ninja_url = "https://github.com/ninja-build/ninja/releases/download/v1.12.1/ninja-win.zip"

    llvm_release = fetch_github_latest_release("llvm/llvm-project")
    llvm_url = ""
    if llvm_release:
        for asset in llvm_release.get("assets", []):
            name = asset.get("name", "").lower()
            if IS_WIN and "x86_64-pc-windows-msvc" in name and name.endswith(".tar.xz"):
                llvm_url = asset.get("browser_download_url")
                break
            elif (
                IS_MAC
                and ("arm64-apple-darwin" in name or "x86_64-apple-darwin" in name)
                and name.endswith(".tar.xz")
            ):
                if ("arm" in HOST_ARCH) == ("arm64" in name):
                    llvm_url = asset.get("browser_download_url")
                    break
            elif IS_LINUX and "x86_64-linux-gnu" in name and name.endswith(".tar.xz"):
                llvm_url = asset.get("browser_download_url")
                break
    if not llvm_url:
        llvm_url = "https://github.com/llvm/llvm-project/releases/download/llvmorg-18.1.8/clang+llvm-18.1.8-x86_64-pc-windows-msvc.tar.xz"

    nasm_url = resolve_nasm_url()
    git_url = resolve_git_url()
    return cmake_url, ninja_url, llvm_url, nasm_url, git_url


def hydrate_hermetic_python() -> None:
    isolated_env = build_isolated_env()
    if not HERMETIC_PYTHON_EXE.exists():
        log_stage("Hydrating Hermetic Python Runtime")
        PYTHON_DIR.mkdir(parents=True, exist_ok=True)
        _, py_url = resolve_python_runtime()
        if IS_WIN:
            archive_path = TMP_DIR / "python_embed.zip"
            fetch_payload(py_url, archive_path)
            unpack_archive(archive_path, PYTHON_DIR, flatten_single_dir=False)
            archive_path.unlink(missing_ok=True)
            (PYTHON_DIR / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)
            for pth_file in PYTHON_DIR.glob("*._pth"):
                orig_lines = pth_file.read_text(encoding="utf-8").splitlines()
                core_zips = [
                    line.strip()
                    for line in orig_lines
                    if line.strip().endswith(".zip")
                ]
                payload_paths = core_zips + [
                    ".",
                    "Lib/site-packages",
                    "import site",
                ]
                pth_file.write_text("\n".join(payload_paths) + "\n", encoding="utf-8")
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
        fetch_payload("https://bootstrap.pypa.io/get-pip.py", pip_bootstrapper)
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

    log_info("Staging reference tiktoken & numpy in hermetic Python (--pre -U)...")
    stage_and_install_packages(
        isolated_env,
        ["pip", "setuptools", "wheel", "packaging", "tiktoken", "numpy"],
    )


def hydrate_toolchains() -> None:
    log_stage("Resolving and Hydrating Upstream Host Toolchains")
    cmake_url, ninja_url, llvm_url, nasm_url, git_url = resolve_upstream_tool_urls()
    if IS_WIN and not GIT_CMD_EXE.exists() and git_url:
        archive = TMP_DIR / "mingit.zip"
        fetch_payload(git_url, archive)
        unpack_archive(archive, TOOLCHAIN_DIR / "git", flatten_single_dir=False)
        archive.unlink(missing_ok=True)
    if not CMAKE_EXE.exists():
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
    if not NINJA_EXE.exists():
        archive = TMP_DIR / "ninja.zip"
        fetch_payload(ninja_url, archive)
        unpack_archive(archive, TOOLCHAIN_DIR / "ninja", flatten_single_dir=False)
        archive.unlink(missing_ok=True)
        if not IS_WIN:
            os.chmod(NINJA_EXE, 0o755)
    if not (CLANG_CL_EXE if IS_WIN else CLANG_EXE).exists():
        archive = TMP_DIR / "llvm.tar.xz"
        fetch_payload(llvm_url, archive, timeout=600)
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
    if IS_WIN and not NASM_EXE.exists() and nasm_url:
        archive = TMP_DIR / "nasm.zip"
        fetch_payload(nasm_url, archive)
        temp_ext = TMP_DIR / "nasm_ext"
        unpack_archive(archive, temp_ext, flatten_single_dir=True)
        nasm_dest = TOOLCHAIN_DIR / "nasm"
        nasm_dest.mkdir(parents=True, exist_ok=True)
        for item in temp_ext.rglob("*"):
            if item.is_file() and item.name.lower() in ["nasm.exe", "ndisasm.exe"]:
                shutil.copy2(to_unc_path(item), to_unc_path(nasm_dest / item.name))
        safe_rmtree(temp_ext)
        archive.unlink(missing_ok=True)


def fetch_msvc_manifest_catalog() -> dict:
    last_err = None
    for endpoint in VS_CHANNEL_ENDPOINTS:
        try:
            log_info(f"Querying Visual Studio Channel: {endpoint}")
            channel_data = fetch_json(endpoint, timeout=30)
            if "packages" in channel_data:
                return channel_data
            for item in channel_data.get("channelItems", []):
                item_type = item.get("type", "").lower()
                item_id = item.get("id", "").lower()
                if item_type == "manifest" or "manifest" in item_id:
                    for manifest_payload in item.get("payloads", []):
                        url = manifest_payload.get("url", "")
                        if url.endswith((".json", ".vsman")):
                            log_info(
                                f"Streaming full Visual Studio manifest catalog: {url}"
                            )
                            manifest_tmp = TMP_DIR / "vs_catalog.tmp"
                            fetch_payload(url, manifest_tmp, timeout=300)
                            try:
                                with open(manifest_tmp, "rb") as mf:
                                    header = mf.read(2)
                                if header == b"\x1f\x8b":
                                    with gzip.open(
                                        manifest_tmp,
                                        "rt",
                                        encoding="utf-8",
                                        errors="ignore",
                                    ) as gf:
                                        catalog = json.load(gf)
                                else:
                                    with open(
                                        manifest_tmp,
                                        "r",
                                        encoding="utf-8",
                                        errors="ignore",
                                    ) as jf:
                                        catalog = json.load(jf)
                                manifest_tmp.unlink(missing_ok=True)
                                if "packages" in catalog:
                                    return catalog
                            except Exception as parse_err:
                                manifest_tmp.unlink(missing_ok=True)
                                last_err = parse_err
        except Exception as exc:
            last_err = exc
            continue
    raise RuntimeError(
        f"Visual Studio Catalog Manifest resolution failed across all endpoints: {last_err}"
    )


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


def audit_winsdk_fakeroot() -> bool:
    if not WINSDK_DIR.exists():
        return False
    kits_lib = WINSDK_DIR / "Windows Kits" / "10" / "Lib"
    if kits_lib.exists():
        for um_dir in kits_lib.glob("*/um/x64"):
            for bad_ucrt in um_dir.glob("*ucrt*.lib"):
                remove_readonly(None, str(bad_ucrt), None)
                bad_ucrt.unlink(missing_ok=True)
    vcruntime_headers = list(WINSDK_DIR.glob("VC/Tools/MSVC/*/include/vcruntime.h"))
    if not vcruntime_headers:
        return False
    crt_libs = list(WINSDK_DIR.glob("VC/Tools/MSVC/*/lib/x64/libcmt.lib"))
    if not crt_libs:
        return False
    kernel32_libs = list(WINSDK_DIR.glob("Windows Kits/10/Lib/*/um/x64/kernel32.lib"))
    ucrt_libs = list(WINSDK_DIR.glob("Windows Kits/10/Lib/*/ucrt/x64/ucrt.lib"))
    if not kernel32_libs or not ucrt_libs:
        return False
    for target in [vcruntime_headers[0], crt_libs[0], kernel32_libs[0], ucrt_libs[0]]:
        if target.stat().st_size == 0:
            return False
    return True


def hydrate_winsdk() -> None:
    if not IS_WIN:
        return
    if audit_winsdk_fakeroot():
        log_info("Windows SDK and MSVC CRT fakeroot verified. Staged assets complete.")
        return
    msvc_version = "14.44.35220"
    sdk_version = "10.0.26100.0"
    msvc_inc_target = WINSDK_DIR / "VC" / "Tools" / "MSVC" / msvc_version / "include"
    msvc_lib_target = (
        WINSDK_DIR / "VC" / "Tools" / "MSVC" / msvc_version / "lib" / "x64"
    )
    sdk_inc_target = WINSDK_DIR / "Windows Kits" / "10" / "Include" / sdk_version
    sdk_lib_target = WINSDK_DIR / "Windows Kits" / "10" / "Lib" / sdk_version
    version_marker = msvc_lib_target / ".version_14.44"

    log_stage("Hydrating Windows SDK and MSVC CRT Fakeroot")
    os.makedirs(to_unc_path(msvc_inc_target), exist_ok=True)
    os.makedirs(to_unc_path(msvc_lib_target), exist_ok=True)
    os.makedirs(to_unc_path(sdk_inc_target), exist_ok=True)
    os.makedirs(to_unc_path(sdk_lib_target), exist_ok=True)

    packages = [
        ("winsdk_cpp.nupkg", NUGET_SDK_CPP_URL),
        ("winsdk_cpp_x64.nupkg", NUGET_SDK_CPP_X64_URL),
        ("winsdk_buildtools.nupkg", NUGET_SDK_BUILDTOOLS_URL),
    ]
    header_buckets = ["ucrt", "um", "shared", "winrt", "cppwinrt"]
    for nupkg_name, url in packages:
        nupkg_path = TMP_DIR / nupkg_name
        ext_dir = TMP_DIR / f"ext_{nupkg_path.stem}"
        try:
            fetch_payload(url, nupkg_path, timeout=120)
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
                    bucket = "ucrt" if "ucrt" in parts_lower else "um"
                    target = sdk_lib_target / bucket / "x64" / item.name
                    os.makedirs(to_unc_path(target.parent), exist_ok=True)
                    if target.exists():
                        remove_readonly(None, str(target), None)
                    shutil.copy2(to_unc_path(item), to_unc_path(target))
                elif file_name_lower.endswith((".h", ".hpp", ".idl")):
                    found_bucket = next(
                        (p for p in parts_lower if p in header_buckets), None
                    )
                    if found_bucket:
                        b_idx = parts_lower.index(found_bucket)
                        rel_sub = Path(*item.parts[b_idx + 1 :])
                        target = sdk_inc_target / found_bucket / rel_sub
                    else:
                        target = sdk_inc_target / "um" / item.relative_to(ext_dir)
                    os.makedirs(to_unc_path(target.parent), exist_ok=True)
                    if target.exists():
                        remove_readonly(None, str(target), None)
                    shutil.copy2(to_unc_path(item), to_unc_path(target))
        finally:
            safe_rmtree(ext_dir)
            nupkg_path.unlink(missing_ok=True)

    catalog = fetch_msvc_manifest_catalog()
    catalog_packages = catalog.get("packages", [])
    crt_headers_pkgs = []
    crt_libs_pkgs = []
    for pkg in catalog_packages:
        pkg_id = pkg.get("id", "").lower()
        if pkg_id.startswith("microsoft.vc.") and pkg_id.endswith(
            ".crt.headers.base"
        ):
            crt_headers_pkgs.append(pkg)
        elif pkg_id.startswith("microsoft.vc.") and (
            "crt.x64" in pkg_id or "crt.desktop" in pkg_id
        ):
            crt_libs_pkgs.append(pkg)

    crt_headers_pkgs.sort(key=lambda p: parse_semver(p.get("version", "0")))
    crt_libs_pkgs.sort(key=lambda p: parse_semver(p.get("version", "0")))
    for pkg in crt_headers_pkgs:
        try:
            extract_and_stage_vs_payload(pkg, "crt_headers", msvc_inc_target)
        except Exception as exc:
            log_warn(f"CRT header extraction skipped ({pkg.get('id')}): {exc}")
    for pkg in crt_libs_pkgs:
        try:
            extract_and_stage_vs_payload(pkg, "crt_libs", msvc_lib_target)
        except Exception as exc:
            log_warn(f"CRT lib extraction skipped ({pkg.get('id')}): {exc}")

    if not audit_winsdk_fakeroot():
        raise RuntimeError(
            "Sysroot hydration failure: vcruntime.h or CRT libraries could not be extracted."
        )
    version_marker.write_text(f"{msvc_version}\n", encoding="utf-8")


def hydrate_libraries() -> None:
    log_stage("Hydrating Upstream C/C++ Libraries (PCRE2)")
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    pcre2_dir = LIBRARY_DIR / "pcre2"
    if (
        not (pcre2_dir / "src" / "pcre2.h.generic").exists()
        and not (pcre2_dir / "src" / "pcre2.h").exists()
    ):
        pcre2_url = resolve_pcre2_url()
        archive = TMP_DIR / (
            "pcre2.zip" if pcre2_url.endswith(".zip") else "pcre2.tar.gz"
        )
        fetch_payload(pcre2_url, archive, timeout=120)
        unpack_archive(archive, pcre2_dir, flatten_single_dir=True)
        archive.unlink(missing_ok=True)


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    hydrate_vocabulary()
    hydrate_hermetic_python()
    hydrate_toolchains()
    hydrate_winsdk()
    hydrate_libraries()
    safe_rmtree(TMP_DIR)
    log_info("VizToken hermetic hydration complete. Zero external host dependencies remaining.")


if __name__ == "__main__":
    main()