#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import ctypes
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
BASE_DIR = TOOLS_DIR.parent
BIN_DIR = BASE_DIR / "bin"
SYSROOT_DIR = BIN_DIR / "sysroot"
SOURCES_DIR = BIN_DIR / "sources"
WINSDK_DIR = BIN_DIR / "winsdk"
PYTHON_DIR = BIN_DIR / "python"
FFMPEG_DIR = BIN_DIR / "ffmpeg"

CACHE_ROOT = BASE_DIR / ".forge_cache"
ENV_CACHE_DIR = CACHE_ROOT / "env"
HOME_DIR = ENV_CACHE_DIR / "home"
TMP_DIR = ENV_CACHE_DIR / "tmp"
DOWNLOADS_DIR = ENV_CACHE_DIR / "downloads"
WHEELS_DIR = DOWNLOADS_DIR / "wheels"
PIP_CACHE_DIR = ENV_CACHE_DIR / "pip"
LOCAL_APPDATA_DIR = ENV_CACHE_DIR / "local_appdata"
ROAMING_APPDATA_DIR = ENV_CACHE_DIR / "appdata"


def _resolve_hf_cache_dir() -> Path:
    env_hf = os.environ.get("HF_HOME")
    if env_hf and env_hf.strip():
        return Path(env_hf).resolve()
    parent_hf = (BASE_DIR.parent / ".hf_cache").resolve()
    if parent_hf.exists():
        return parent_hf
    return (BASE_DIR / ".hf_cache").resolve()


HF_CACHE_DIR = _resolve_hf_cache_dir()
PARENT_WHEELS_DIR = (BASE_DIR.parent / ".cache" / "downloads" / "wheels").resolve()

HOST_OS = sys.platform
HOST_ARCH = platform.machine().lower()
IS_WIN = HOST_OS == "win32"
IS_MAC = HOST_OS == "darwin"
IS_LINUX = HOST_OS.startswith("linux")
HTTPS = "https" + "://"

ALLOC_CONF_POLICY = "garbage_collection_threshold:0.8"

CLANG_EXE = BIN_DIR / "llvm" / "bin" / ("clang.exe" if IS_WIN else "clang")
CLANG_CL_EXE = BIN_DIR / "llvm" / "bin" / ("clang-cl.exe" if IS_WIN else "clang-cl")
LLD_LINK_EXE = BIN_DIR / "llvm" / "bin" / ("lld-link.exe" if IS_WIN else "lld-link")
CMAKE_EXE = BIN_DIR / "cmake" / "bin" / ("cmake.exe" if IS_WIN else "cmake")
NINJA_EXE = BIN_DIR / "ninja" / ("ninja.exe" if IS_WIN else "ninja")
NASM_EXE = BIN_DIR / "nasm" / ("nasm.exe" if IS_WIN else "nasm")
GIT_CMD_EXE = BIN_DIR / "git" / "cmd" / ("git.exe" if IS_WIN else "git")
FFMPEG_EXE = FFMPEG_DIR / ("ffmpeg.exe" if IS_WIN else "ffmpeg")
HERMETIC_PYTHON_EXE = PYTHON_DIR / ("python.exe" if IS_WIN else "bin/python")

REQUIREMENTS_FILE = BASE_DIR / "requirements.txt"
LOCK_FILE = PYTHON_DIR / "constraints.lock"

NUGET_PACKAGES = [
    "microsoft.windows.sdk.cpp",
    "microsoft.windows.sdk.cpp.x64",
    "microsoft.windows.sdk.buildtools",
]

SDK_REQUIRED_ANCHORS = [
    "kernel32.lib",
    "user32.lib",
    "ucrt.lib",
    "libucrt.lib",
]

MSVC_REQUIRED_ANCHORS = [
    "libcmt.lib",
    "libcpmt.lib",
    "libvcruntime.lib",
    "oldnames.lib",
]

IMPORT_TO_PYPI_MAP = {
    "numpy": "numpy",
    "PIL": "Pillow",
    "pydantic": "pydantic",
    "safetensors": "safetensors",
    "huggingface_hub": "huggingface-hub",
}

LOCAL_MODULE_EXCLUSIONS = {
    "python",
    "tools",
    "vizinjector",
    "bindings",
    "schema",
    "modeling_vae",
}

AMD_STAGING_FAMILY_MAP = {
    "gfx1201": "gfx120X-all",
    "gfx1200": "gfx120X-all",
    "gfx120X": "gfx120X-all",
    "gfx1150": "gfx1150",
    "gfx1151": "gfx1151",
    "gfx1152": "gfx1152",
    "gfx1100": "gfx110X-all",
    "gfx1101": "gfx110X-all",
    "gfx1102": "gfx110X-all",
    "gfx1103": "gfx110X-all",
    "gfx1030": "gfx103X-all",
    "gfx1031": "gfx103X-all",
    "gfx1035": "gfx103X-all",
    "gfx942": "gfx94X-dcgpu",
    "gfx950": "gfx950",
    "gfx90a": "gfx90a-all",
    "gfx908": "gfx908",
}

HSA_OVERRIDES = {
    "gfx1150": "11.5.0",
    "gfx1151": "11.5.0",
    "gfx1152": "11.5.0",
    "gfx1103": "11.0.0",
    "gfx1035": "10.3.0",
    "gfx1031": "10.3.0",
}

SEMVER_TAG_REGEX = re.compile(
    r"^(?:v|vulkan-sdk-|sdk-|release-|rel-)?(\d+(?:\.\d+)+)$", re.IGNORECASE
)
URL_EXTRACTOR_REGEX = re.compile(r"(https?://[^\]\s)>\"']+)", re.IGNORECASE)


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


def safe_subprocess(cmd: list, env: dict[str, str] | None = None) -> str:
    try:
        kwargs = (
            {"creationflags": subprocess.CREATE_NO_WINDOW} if IS_WIN else {}
        )
        return subprocess.check_output(
            cmd,
            stderr=subprocess.DEVNULL,
            timeout=15,
            text=True,
            env=env,
            **kwargs,
        ).strip()
    except Exception:
        return ""


def resolve_rocm_sdk_layout(target_sandbox: Path) -> dict[str, str]:
    site_packages = (
        target_sandbox / "Lib" / "site-packages"
        if IS_WIN
        else next(target_sandbox.glob("lib/python*/site-packages"), None)
    )
    rocm_root: Path | None = None
    rocm_bin: Path | None = None
    rocm_cmake: Path | None = None

    if site_packages is not None and site_packages.is_dir():
        devel_dir = site_packages / "_rocm_sdk_devel"
        core_dir = site_packages / "_rocm_sdk_core"
        if (devel_dir / "bin").is_dir():
            rocm_root = devel_dir.resolve()
        elif (core_dir / "bin").is_dir():
            rocm_root = core_dir.resolve()
        if rocm_root is not None:
            if (rocm_root / "bin").is_dir():
                rocm_bin = (rocm_root / "bin").resolve()
            if (rocm_root / "lib" / "cmake").is_dir():
                rocm_cmake = (rocm_root / "lib" / "cmake").resolve()

    if rocm_root is None:
        rocm_root = target_sandbox.resolve()

    llvm_bin = rocm_root / "lib" / "llvm" / "bin"
    device_lib = rocm_root / "lib" / "llvm" / "amdgcn" / "bitcode"
    miopen_sys_db = rocm_root / "share" / "miopen" / "db"

    extra_bins: list[str] = []
    if rocm_bin is not None and rocm_bin.is_dir():
        extra_bins.append(str(rocm_bin))
    if llvm_bin.is_dir():
        extra_bins.append(str(llvm_bin.resolve()))
    if site_packages is not None and site_packages.is_dir():
        for pkg_dir in sorted(site_packages.glob("_rocm_sdk_*")):
            b_dir = pkg_dir / "bin"
            if b_dir.is_dir():
                b_str = str(b_dir.resolve())
                if b_str not in extra_bins:
                    extra_bins.append(b_str)

    return {
        "rocm_root": str(rocm_root),
        "rocm_bin": str(rocm_bin) if rocm_bin is not None else "",
        "rocm_cmake": str(rocm_cmake) if rocm_cmake is not None else "",
        "llvm_bin": str(llvm_bin.resolve()) if llvm_bin.is_dir() else "",
        "device_lib": str(device_lib.resolve()) if device_lib.is_dir() else "",
        "miopen_sys_db": (
            str(miopen_sys_db.resolve()) if miopen_sys_db.is_dir() else ""
        ),
        "extra_bins": os.pathsep.join(extra_bins),
    }


def get_isolated_env() -> dict[str, str]:
    for d in (
        CACHE_ROOT,
        ENV_CACHE_DIR,
        HOME_DIR,
        TMP_DIR,
        DOWNLOADS_DIR,
        WHEELS_DIR,
        PIP_CACHE_DIR,
        HF_CACHE_DIR,
        LOCAL_APPDATA_DIR,
        ROAMING_APPDATA_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)

    tempfile.tempdir = str(TMP_DIR)
    rocm_layout = resolve_rocm_sdk_layout(PYTHON_DIR)
    rocm_extra_bins = [
        p for p in rocm_layout["extra_bins"].split(os.pathsep) if p
    ]

    env: dict[str, str] = {}
    for os_var in (
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
        if os_var in os.environ:
            env[os_var] = os.environ[os_var]

    sys_root = env.get("SYSTEMROOT", r"C:\Windows")
    hermetic_bins = (
        [
            str(PYTHON_DIR if IS_WIN else PYTHON_DIR / "bin"),
            str(PYTHON_DIR / "Scripts" if IS_WIN else PYTHON_DIR / "bin"),
        ]
        + rocm_extra_bins
        + [
            str(BIN_DIR / "git" / "cmd"),
            str(BIN_DIR / "llvm" / "bin"),
            str(BIN_DIR / "cmake" / "bin"),
            str(BIN_DIR / "ninja"),
            str(BIN_DIR / "nasm"),
            str(BIN_DIR / "shaderc" / "bin"),
            str(FFMPEG_DIR),
            str(SYSROOT_DIR / "bin"),
        ]
    )
    if IS_WIN:
        hermetic_bins.extend(
            [
                os.path.join(sys_root, "System32"),
                sys_root,
                os.path.join(sys_root, "System32", "Wbem"),
                os.path.join(sys_root, "System32", "WindowsPowerShell", "v1.0"),
            ]
        )
        env["PIP_CONFIG_FILE"] = "NUL"
    else:
        hermetic_bins.extend(["/usr/bin", "/bin"])
        env["PIP_CONFIG_FILE"] = "/dev/null"

    env["PATH"] = os.pathsep.join(hermetic_bins)
    env["PYTHONHOME"] = str(PYTHON_DIR)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PIP_NO_USER"] = "1"
    env["PIP_USER"] = "0"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["USERPROFILE"] = str(HOME_DIR)
    env["HOME"] = str(HOME_DIR)
    env["TEMP"] = str(TMP_DIR)
    env["TMP"] = str(TMP_DIR)
    env["TMPDIR"] = str(TMP_DIR)
    env["PIP_CACHE_DIR"] = str(PIP_CACHE_DIR)
    env["LOCALAPPDATA"] = str(LOCAL_APPDATA_DIR)
    env["APPDATA"] = str(ROAMING_APPDATA_DIR)
    env["XDG_CACHE_HOME"] = str(ENV_CACHE_DIR)
    env["XDG_CONFIG_HOME"] = str(ENV_CACHE_DIR / "config")
    env["XDG_DATA_HOME"] = str(ENV_CACHE_DIR / "share")
    env["ROCM_PATH"] = rocm_layout["rocm_root"]
    env["ROCM_HOME"] = rocm_layout["rocm_root"]
    env["HIP_PATH"] = rocm_layout["rocm_root"]

    if rocm_layout["llvm_bin"]:
        env["HIP_CLANG_PATH"] = rocm_layout["llvm_bin"]
    if rocm_layout["device_lib"]:
        env["HIP_DEVICE_LIB_PATH"] = rocm_layout["device_lib"]
    if rocm_layout["miopen_sys_db"]:
        env["MIOPEN_SYSTEM_DB_PATH"] = rocm_layout["miopen_sys_db"]

    contract_defaults = {
        "HF_HOME": str(HF_CACHE_DIR),
        "TORCH_HOME": str(HF_CACHE_DIR / "torch"),
        "TRANSFORMERS_CACHE": str(HF_CACHE_DIR / "transformers"),
        "HUGGINGFACE_HUB_CACHE": str(HF_CACHE_DIR / "hub"),
        "MIOPEN_USER_DB_PATH": str(HF_CACHE_DIR / "miopen" / "db"),
        "MIOPEN_CUSTOM_CACHE_DIR": str(HF_CACHE_DIR / "miopen" / "kernels"),
        "TORCHINDUCTOR_CACHE_DIR": str(HF_CACHE_DIR / "torch_inductor"),
        "TRITON_CACHE_DIR": str(HF_CACHE_DIR / "triton"),
        "MIOPEN_FIND_MODE": "1",
        "MIOPEN_LOG_LEVEL": "0",
        "MIOPEN_ENABLE_LOGGING": "0",
        "AMD_LOG_LEVEL": "0",
        "TORCH_BLAS_PREFER_HIPBLASLT": "1",
        "HSA_ENABLE_SDMA": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTORCH_ALLOC_CONF": ALLOC_CONF_POLICY,
        "PYTORCH_CUDA_ALLOC_CONF": ALLOC_CONF_POLICY,
        "PYTORCH_HIP_ALLOC_CONF": ALLOC_CONF_POLICY,
    }
    for k, v in contract_defaults.items():
        cur = os.environ.get(k)
        env[k] = cur if (cur is not None and cur.strip()) else v

    for passthrough_rocm in (
        "HIP_PLATFORM",
        "PYTORCH_ROCM_ARCH",
        "ROCM_SDK_TARGET_FAMILY",
        "HSA_OVERRIDE_GFX_VERSION",
    ):
        if os.environ.get(passthrough_rocm, "").strip():
            env[passthrough_rocm] = os.environ[passthrough_rocm].strip()

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
    headers = {"User-Agent": "vizinjector-hydrator/2.0"}
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


def resolve_github_repo_snapshot(
    repo: str, target_dir: Path, prefer_default_branch: bool = False
) -> tuple[str, str]:
    if not prefer_default_branch:
        releases = fetch_github_releases(repo, per_page=10)
        for rel in releases:
            if rel.get("draft"):
                continue
            tag = rel.get("tag_name", "")
            if tag:
                candidate = f"{HTTPS}github.com/{repo}/archive/refs/tags/{tag}.zip"
                if probe_url(candidate):
                    return tag, candidate
        candidate_tags: list[str] = []
        try:
            refs_data = fetch_json(
                f"{HTTPS}api.github.com/repos/{repo}/git/matching-refs/tags",
                timeout=20,
            )
            if isinstance(refs_data, list):
                for ref_obj in refs_data:
                    ref_str = ref_obj.get("ref", "")
                    if ref_str.startswith("refs/tags/"):
                        t_name = ref_str[len("refs/tags/") :]
                        if not t_name.endswith("^{}") and SEMVER_TAG_REGEX.match(
                            t_name
                        ):
                            candidate_tags.append(t_name)
        except Exception:
            pass
        if not candidate_tags:
            try:
                tags_data = fetch_json(
                    f"{HTTPS}api.github.com/repos/{repo}/tags?per_page=100",
                    timeout=15,
                )
                if isinstance(tags_data, list):
                    for tag_obj in tags_data:
                        t_name = tag_obj.get("name", "")
                        if SEMVER_TAG_REGEX.match(t_name):
                            candidate_tags.append(t_name)
            except Exception:
                pass
        if candidate_tags:
            candidate_tags.sort(key=parse_semver, reverse=True)
            for best_tag in candidate_tags[:5]:
                candidate = (
                    f"{HTTPS}github.com/{repo}/archive/refs/tags/{best_tag}.zip"
                )
                if probe_url(candidate):
                    return best_tag, candidate
    default_branch = "main"
    try:
        repo_meta = fetch_json(f"{HTTPS}api.github.com/repos/{repo}", timeout=15)
        if isinstance(repo_meta, dict) and repo_meta.get("default_branch"):
            default_branch = repo_meta["default_branch"]
    except Exception:
        pass
    for branch in dict.fromkeys([default_branch, "main", "master"]):
        try:
            commit_data = fetch_json(
                f"{HTTPS}api.github.com/repos/{repo}/commits/{branch}", timeout=15
            )
            if isinstance(commit_data, dict):
                sha = commit_data.get("sha", "")
                if sha:
                    candidate = f"{HTTPS}github.com/{repo}/archive/{sha}.zip"
                    if probe_url(candidate):
                        return sha, candidate
        except Exception:
            continue
    cached = read_version_marker(target_dir)
    if cached:
        return cached, ""
    for branch in ("main", "master"):
        candidate = f"{HTTPS}github.com/{repo}/archive/refs/heads/{branch}.zip"
        if probe_url(candidate):
            return branch, candidate
    raise RuntimeError(f"Unable to dynamically resolve upstream snapshot for {repo}")


def fetch_payload(url: str, destination: Path, timeout: int = 600) -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    clean_url = sanitize_url(url)
    log_info(f"Fetching payload: {clean_url}")
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


def query_dxgi_adapters_win32() -> list[dict]:
    if not IS_WIN:
        return []
    try:

        class DXGI_ADAPTER_DESC1(ctypes.Structure):
            _fields_ = [
                ("Description", ctypes.c_wchar * 128),
                ("VendorId", ctypes.c_uint),
                ("DeviceId", ctypes.c_uint),
                ("SubSysId", ctypes.c_uint),
                ("Revision", ctypes.c_uint),
                ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t),
                ("SharedSystemMemory", ctypes.c_size_t),
                ("AdapterLuid", ctypes.c_longlong),
                ("Flags", ctypes.c_uint),
            ]

        class GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", ctypes.c_ulong),
                ("Data2", ctypes.c_ushort),
                ("Data3", ctypes.c_ushort),
                ("Data4", ctypes.c_ubyte * 8),
            ]

        def call_com(ptr, index, argtypes, *args):
            vtable = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_void_p))[0]
            func_addr = ctypes.cast(vtable, ctypes.POINTER(ctypes.c_void_p))[index]
            func = ctypes.WINFUNCTYPE(ctypes.c_long, *argtypes)(func_addr)
            return func(ptr, *args)

        iid_dxgi_factory1 = GUID(
            0x770AAE78,
            0xF26F,
            0x4DBA,
            (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87),
        )
        factory = ctypes.c_void_p()
        if (
            ctypes.windll.dxgi.CreateDXGIFactory1(
                ctypes.byref(iid_dxgi_factory1), ctypes.byref(factory)
            )
            != 0
        ):
            return []
        adapters = []
        idx = 0
        while True:
            adapter = ctypes.c_void_p()
            hr = call_com(
                factory,
                12,
                [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)],
                idx,
                ctypes.byref(adapter),
            )
            if hr < 0:
                break
            desc = DXGI_ADAPTER_DESC1()
            call_com(
                adapter,
                10,
                [ctypes.c_void_p, ctypes.POINTER(DXGI_ADAPTER_DESC1)],
                ctypes.byref(desc),
            )
            if not (desc.Flags & 2):
                adapters.append(
                    {
                        "name": str(desc.Description),
                        "vendor_id": desc.VendorId,
                        "device_id": desc.DeviceId,
                        "vram_gb": desc.DedicatedVideoMemory / (1024**3),
                    }
                )
            call_com(adapter, 2, [ctypes.c_void_p])
            idx += 1
        call_com(factory, 2, [ctypes.c_void_p])
        return adapters
    except Exception:
        return []


def resolve_hardware_matrix() -> tuple[str, str]:
    if IS_MAC:
        uname = safe_subprocess(["uname", "-m"])
        return ("MPS", "") if "arm64" in uname else ("CPU", "")
    has_nvidia, has_amd, has_intel = False, False, False
    detected_amd_targets = []
    if IS_WIN:
        dxgi_devices = query_dxgi_adapters_win32()
        for dev in dxgi_devices:
            name = dev["name"]
            vid = dev["vendor_id"]
            if vid == 0x10DE:
                has_nvidia = True
            elif vid == 0x8086:
                has_intel = True
            elif vid == 0x1002:
                has_amd = True
                if re.search(r"(?i)MI325|MI350|MI355", name):
                    detected_amd_targets.append("gfx950")
                elif re.search(r"(?i)MI300", name):
                    detected_amd_targets.append("gfx942")
                elif re.search(r"(?i)MI250|MI210", name):
                    detected_amd_targets.append("gfx90a")
                elif re.search(
                    r"(?i)R9700|9070|Navi\s*48|Navi\s*44|RX\s*9\d{3}|AI\s*PRO\s*R9",
                    name,
                ):
                    detected_amd_targets.append("gfx1201")
                elif re.search(
                    r"(?i)890M|880M|Strix|Ryzen\s*AI\s*(?:PRO\s*)?3\d{2}|Kraken",
                    name,
                ):
                    detected_amd_targets.append("gfx1150")
                elif re.search(r"(?i)7900|W7900|Navi\s*31", name):
                    detected_amd_targets.append("gfx1100")
                elif re.search(r"(?i)7800|7700|Navi\s*32", name):
                    detected_amd_targets.append("gfx1101")
                elif re.search(r"(?i)7600|W7600|W7500|Navi\s*33", name):
                    detected_amd_targets.append("gfx1102")
                elif re.search(r"(?i)6950|6900|6800|6700|W6800|Navi\s*21", name):
                    detected_amd_targets.append("gfx1030")
                elif re.search(r"(?i)780M|760M|740M|Phoenix|Hawk", name):
                    detected_amd_targets.append("gfx1103")
                elif re.search(r"(?i)680M|660M|Rembrandt", name):
                    detected_amd_targets.append("gfx1035")
                else:
                    detected_amd_targets.append(
                        "gfx1201" if "Radeon" in name else "gfx1100"
                    )
    else:
        lspci = safe_subprocess(["lspci"])
        if lspci:
            if re.search(r"(?i)NVIDIA", lspci):
                has_nvidia = True
            if re.search(r"(?i)AMD|Radeon", lspci):
                has_amd = True
            if re.search(r"(?i)Intel.*(Arc|Graphics)", lspci):
                has_intel = True
        if has_amd:
            rocm_info = safe_subprocess(["rocminfo"])
            gfx_match = re.search(r"gfx\d+[a-zA-Z]?", rocm_info)
            if gfx_match:
                detected_amd_targets.append(gfx_match.group(0))
            else:
                detected_amd_targets.append("gfx1201")
    if has_nvidia:
        smi = safe_subprocess(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        )
        if smi:
            try:
                major = int(smi.split(".")[0])
                if major >= 620:
                    return "CUDA_13_2", ""
                if major >= 600:
                    return "CUDA_13_0", ""
                if major >= 570:
                    return "CUDA_12_8", ""
                if major >= 560:
                    return "CUDA_12_6", ""
                if major >= 520:
                    return "CUDA_12_1", ""
                if major >= 450:
                    return "CUDA_11_8", ""
            except Exception:
                pass
        return "CUDA_12_6", ""
    if has_amd:
        best_target = detected_amd_targets[0] if detected_amd_targets else "gfx1201"
        return "ROCM", best_target
    if has_intel:
        return "INTEL_XPU", ""
    return "CPU", ""


def scan_ast_dependencies() -> list[str]:
    stdlib_names = (
        set(getattr(sys, "stdlib_module_names", set()))
        | set(sys.builtin_module_names)
        | {"__future__", "winreg", "msvcrt", "_winapi"}
    )
    scan_targets: list[Path] = []
    for folder in ["python", "tools"]:
        folder_path = BASE_DIR / folder
        if folder_path.exists():
            scan_targets.extend(folder_path.rglob("*.py"))
    discovered_roots: set[str] = {
        "numpy",
        "PIL",
        "pydantic",
        "safetensors",
        "huggingface_hub",
    }
    for py_file in scan_targets:
        if not py_file.exists():
            continue
        try:
            source = py_file.read_text(encoding="utf-8-sig")
            tree = ast.parse(source, filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root_mod = alias.name.split(".")[0]
                        if root_mod:
                            discovered_roots.add(root_mod)
                elif isinstance(node, ast.ImportFrom):
                    if node.level == 0 and node.module:
                        root_mod = node.module.split(".")[0]
                        if root_mod:
                            discovered_roots.add(root_mod)
        except Exception:
            pass
    third_party_packages: set[str] = set()
    for mod in discovered_roots:
        if mod in stdlib_names or mod in LOCAL_MODULE_EXCLUSIONS:
            continue
        if mod in ("torch", "torchvision", "torchaudio"):
            continue
        if (BASE_DIR / f"{mod}.py").exists() or (BASE_DIR / mod).is_dir():
            continue
        pypi_name = IMPORT_TO_PYPI_MAP.get(mod, mod)
        third_party_packages.add(pypi_name)
    merged_packages = sorted(third_party_packages, key=str.lower)
    req_content = (
        "# AUTO-SYNTHESIZED BY VIZINJECTOR AST DEPENDENCY SCANNER\n"
        + "\n".join(merged_packages)
        + "\n"
    )
    REQUIREMENTS_FILE.write_text(req_content, encoding="utf-8")
    return merged_packages


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
    isolated_env = get_isolated_env()
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


def salvage_cached_wheels() -> None:
    WHEELS_DIR.mkdir(parents=True, exist_ok=True)
    for whl in PIP_CACHE_DIR.rglob("*.whl"):
        dest = WHEELS_DIR / whl.name
        if not dest.exists():
            try:
                shutil.copy2(whl, dest)
            except Exception:
                pass
    http_cache = PIP_CACHE_DIR / "http-v2"
    if not http_cache.exists():
        return
    for body_file in http_cache.rglob("*"):
        if not body_file.is_file() or body_file.stat().st_size < 10 * 1024 * 1024:
            continue
        try:
            with open(body_file, "rb") as f:
                if f.read(4) != b"PK\x03\x04":
                    continue
            with zipfile.ZipFile(body_file, "r") as zf:
                dist_infos = [
                    n for n in zf.namelist() if n.endswith(".dist-info/WHEEL")
                ]
                if not dist_infos:
                    continue
                pkg_prefix = dist_infos[0].split(".dist-info/")[0]
                wheel_meta = zf.read(dist_infos[0]).decode(
                    "utf-8", errors="ignore"
                )
                tags = re.findall(r"Tag:\s*(\S+)", wheel_meta)
                tag_str = tags[0] if tags else "py3-none-any"
                wheel_name = f"{pkg_prefix}-{tag_str}.whl"
                dest = WHEELS_DIR / wheel_name
                if not dest.exists():
                    shutil.copy2(body_file, dest)
                    log_info(
                        f"Recovered cached wheel: {wheel_name} ({dest.stat().st_size / (1024**2):.1f} MB)"
                    )
        except Exception:
            pass


def initialize_rocm_sdk_devel(isolated_env: dict[str, str]) -> None:
    site_packages = (
        PYTHON_DIR / "Lib" / "site-packages"
        if IS_WIN
        else next(PYTHON_DIR.glob("lib/python*/site-packages"), None)
    )
    if site_packages is None or not (site_packages / "rocm_sdk").is_dir():
        return
    devel_dir = site_packages / "_rocm_sdk_devel"
    if (devel_dir / "_devel.tar.xz").is_file() or not (devel_dir / "bin").is_dir():
        log_info("Materializing TheRock rocm_sdk devel sysroot (rocm_sdk init)...")
        subprocess.run(
            [str(HERMETIC_PYTHON_EXE), "-I", "-m", "rocm_sdk", "init"],
            env=isolated_env,
            stdin=subprocess.DEVNULL,
            check=False,
        )


def synchronize_pip_and_pytorch(
    force: bool = False,
) -> tuple[str, str, str]:
    isolated_env = get_isolated_env()
    discovered_pkgs = scan_ast_dependencies()
    profile, llvm_target = resolve_hardware_matrix()
    hsa_override = HSA_OVERRIDES.get(llvm_target, "")
    torch_marker = PYTHON_DIR / ".torch_profile"
    expected_profile_tag = f"{profile}:{llvm_target}"
    salvage_cached_wheels()

    find_links_args = ["--find-links", str(WHEELS_DIR)]
    if PARENT_WHEELS_DIR.is_dir() and PARENT_WHEELS_DIR != WHEELS_DIR:
        find_links_args.extend(["--find-links", str(PARENT_WHEELS_DIR)])

    pip_base = (
        [
            str(HERMETIC_PYTHON_EXE),
            "-I",
            "-m",
            "pip",
            "install",
            "--no-warn-script-location",
            "--no-compile",
            "--progress-bar",
            "off",
            "--no-cache-dir",
        ]
        + find_links_args
    )

    if (
        force
        or not torch_marker.exists()
        or torch_marker.read_text(encoding="utf-8").strip() != expected_profile_tag
    ):
        log_stage(
            f"Conditioning Hardware-Aligned PyTorch Runtime: {profile} ({llvm_target or 'Generic'})"
        )
        subprocess.run(
            pip_base + ["-U", "pip", "setuptools", "wheel", "packaging"],
            env=isolated_env,
            stdin=subprocess.DEVNULL,
            check=True,
        )
        if profile == "ROCM":
            family = AMD_STAGING_FAMILY_MAP.get(llvm_target, f"{llvm_target}-all")
            isolated_env["ROCM_SDK_TARGET_FAMILY"] = family
            if llvm_target:
                isolated_env["PYTORCH_ROCM_ARCH"] = llvm_target
            index_url = f"{HTTPS}rocm.nightlies.amd.com/v2-staging/{family}/"
            log_info(f"Locking Sovereign Staging Nightly Index: {index_url}")
            if IS_WIN:
                subprocess.run(
                    pip_base
                    + [
                        "--index-url",
                        index_url,
                        "--pre",
                        "-U",
                        "--no-build-isolation",
                        "rocm[libraries,devel]",
                    ],
                    env=isolated_env,
                    stdin=subprocess.DEVNULL,
                    check=True,
                )
            subprocess.run(
                pip_base
                + [
                    "--index-url",
                    index_url,
                    "--pre",
                    "-U",
                    "torch",
                    "torchvision",
                    "torchaudio",
                ],
                env=isolated_env,
                stdin=subprocess.DEVNULL,
                check=True,
            )
            initialize_rocm_sdk_devel(isolated_env)
        elif profile.startswith("CUDA_"):
            cuda_urls = {
                "CUDA_13_2": f"{HTTPS}download.pytorch.org/whl/nightly/cu132",
                "CUDA_13_0": f"{HTTPS}download.pytorch.org/whl/nightly/cu130",
                "CUDA_12_8": f"{HTTPS}download.pytorch.org/whl/nightly/cu128",
                "CUDA_12_6": f"{HTTPS}download.pytorch.org/whl/nightly/cu126",
                "CUDA_12_1": f"{HTTPS}download.pytorch.org/whl/cu121",
                "CUDA_11_8": f"{HTTPS}download.pytorch.org/whl/cu118",
            }
            index_url = cuda_urls.get(
                profile, f"{HTTPS}download.pytorch.org/whl/nightly/cu126"
            )
            subprocess.run(
                pip_base
                + [
                    "--index-url",
                    index_url,
                    "--pre",
                    "-U",
                    "torch",
                    "torchvision",
                    "torchaudio",
                ],
                env=isolated_env,
                stdin=subprocess.DEVNULL,
                check=True,
            )
        elif profile == "INTEL_XPU":
            subprocess.run(
                pip_base
                + [
                    "--index-url",
                    f"{HTTPS}pytorch-extension.intel.com/release-whl/stable/xpu/us/",
                    "-U",
                    "torch",
                    "torchvision",
                    "torchaudio",
                    "intel-extension-for-pytorch",
                ],
                env=isolated_env,
                stdin=subprocess.DEVNULL,
                check=True,
            )
        elif profile == "MPS":
            subprocess.run(
                pip_base + ["-U", "torch", "torchvision", "torchaudio"],
                env=isolated_env,
                stdin=subprocess.DEVNULL,
                check=True,
            )
        else:
            subprocess.run(
                pip_base
                + [
                    "--index-url",
                    f"{HTTPS}download.pytorch.org/whl/cpu",
                    "--pre",
                    "-U",
                    "torch",
                    "torchvision",
                    "torchaudio",
                ],
                env=isolated_env,
                stdin=subprocess.DEVNULL,
                check=True,
            )
        torch_marker.write_text(f"{expected_profile_tag}\n", encoding="utf-8")
    else:
        if profile == "ROCM":
            initialize_rocm_sdk_devel(isolated_env)
        log_info(f"Hardware-aligned PyTorch runtime converged ({expected_profile_tag}).")

    log_info(
        f"Synchronizing AST requirements ({', '.join(discovered_pkgs)}) with --pre -U..."
    )
    subprocess.run(
        pip_base + ["--pre", "-U", "-r", str(REQUIREMENTS_FILE)],
        env=isolated_env,
        stdin=subprocess.DEVNULL,
        check=True,
    )
    freeze_res = subprocess.run(
        [str(HERMETIC_PYTHON_EXE), "-I", "-m", "pip", "freeze"],
        env=isolated_env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
    )
    LOCK_FILE.write_text(freeze_res.stdout, encoding="utf-8")
    return profile, llvm_target, hsa_override


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
    cached = read_version_marker(BIN_DIR / "nasm")
    if cached and NASM_EXE.exists():
        return cached, ""
    raise RuntimeError("Failed to dynamically resolve upstream NASM release.")


def resolve_fftw_url() -> tuple[str, str]:
    try:
        html = fetch_http_text(f"{HTTPS}www.fftw.org/download.html", timeout=15)
        matches = re.findall(r"fftw-(\d+\.\d+\.\d+)\.tar\.gz", html)
        for ver in sorted(set(matches), key=parse_semver, reverse=True):
            candidate_url = f"{HTTPS}www.fftw.org/fftw-{ver}.tar.gz"
            if probe_url(candidate_url):
                return ver, candidate_url
    except Exception as exc:
        log_warn(f"FFTW index query failed: {exc}")
    cached = read_version_marker(SOURCES_DIR / "fftw3")
    if cached and (SOURCES_DIR / "fftw3" / "CMakeLists.txt").exists():
        return cached, ""
    raise RuntimeError("Failed to dynamically resolve upstream FFTW release.")


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

    def match_ffmpeg(name: str) -> bool:
        if IS_WIN:
            return "win64-lgpl.zip" in name
        if IS_LINUX:
            return "linux64-lgpl.tar.xz" in name
        return "macos" in name

    return {
        "cmake": resolve_release_asset(
            "Kitware/CMake", BIN_DIR / "cmake", match_cmake
        ),
        "ninja": resolve_release_asset(
            "ninja-build/ninja", BIN_DIR / "ninja", match_ninja
        ),
        "llvm": resolve_release_asset(
            "llvm/llvm-project", BIN_DIR / "llvm", match_llvm, per_page=30
        ),
        "nasm": resolve_nasm_url(),
        "git": (
            resolve_release_asset(
                "git-for-windows/git", BIN_DIR / "git", match_git
            )
            if IS_WIN
            else ("", "")
        ),
        "ffmpeg": resolve_release_asset(
            "BtbN/FFmpeg-Builds", FFMPEG_DIR, match_ffmpeg, per_page=10
        ),
    }


def hydrate_host_tools(force: bool = False) -> None:
    log_stage("Resolving and Converging Upstream Host Toolchains")
    tools_map = resolve_upstream_tool_urls()
    git_tag, git_url = tools_map["git"]
    if IS_WIN and git_tag:
        if git_url and not is_version_current(
            BIN_DIR / "git", git_tag, GIT_CMD_EXE, force
        ):
            archive = TMP_DIR / "mingit.zip"
            fetch_payload(git_url, archive)
            unpack_archive(archive, BIN_DIR / "git", flatten_single_dir=False)
            archive.unlink(missing_ok=True)
            write_version_marker(BIN_DIR / "git", git_tag)
        else:
            log_info(f"MinGit converged ({git_tag}).")
    cmake_tag, cmake_url = tools_map["cmake"]
    if cmake_url and not is_version_current(
        BIN_DIR / "cmake", cmake_tag, CMAKE_EXE, force
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
        promote_directory(inner, BIN_DIR / "cmake")
        safe_rmtree(temp_ext)
        archive.unlink(missing_ok=True)
        write_version_marker(BIN_DIR / "cmake", cmake_tag)
    else:
        log_info(f"CMake converged ({cmake_tag}).")
    ninja_tag, ninja_url = tools_map["ninja"]
    if ninja_url and not is_version_current(
        BIN_DIR / "ninja", ninja_tag, NINJA_EXE, force
    ):
        archive = TMP_DIR / "ninja.zip"
        fetch_payload(ninja_url, archive)
        unpack_archive(archive, BIN_DIR / "ninja", flatten_single_dir=False)
        archive.unlink(missing_ok=True)
        if not IS_WIN:
            os.chmod(NINJA_EXE, 0o755)
        write_version_marker(BIN_DIR / "ninja", ninja_tag)
    else:
        log_info(f"Ninja converged ({ninja_tag}).")
    llvm_tag, llvm_url = tools_map["llvm"]
    llvm_marker = CLANG_CL_EXE if IS_WIN else CLANG_EXE
    if llvm_url and not is_version_current(
        BIN_DIR / "llvm", llvm_tag, llvm_marker, force
    ):
        archive = TMP_DIR / "llvm.tar.xz"
        fetch_payload(llvm_url, archive, timeout=900)
        temp_ext = TMP_DIR / "llvm_ext"
        unpack_archive(archive, temp_ext, flatten_single_dir=False)
        inner = next(temp_ext.iterdir())
        promote_directory(inner, BIN_DIR / "llvm")
        safe_rmtree(temp_ext)
        archive.unlink(missing_ok=True)
        if not IS_WIN:
            os.chmod(CLANG_EXE, 0o755)
            if LLD_LINK_EXE.exists():
                os.chmod(LLD_LINK_EXE, 0o755)
        write_version_marker(BIN_DIR / "llvm", llvm_tag)
    else:
        log_info(f"LLVM/Clang converged ({llvm_tag}).")
    nasm_tag, nasm_url = tools_map["nasm"]
    if nasm_url and not is_version_current(
        BIN_DIR / "nasm", nasm_tag, NASM_EXE, force
    ):
        archive = TMP_DIR / (
            "nasm.zip" if nasm_url.endswith(".zip") else "nasm.tar.xz"
        )
        fetch_payload(nasm_url, archive)
        temp_ext = TMP_DIR / "nasm_ext"
        unpack_archive(archive, temp_ext, flatten_single_dir=True)
        nasm_dest = BIN_DIR / "nasm"
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
    ff_tag, ff_url = tools_map["ffmpeg"]
    ff_bins = {"ffmpeg.exe", "ffprobe.exe"} if IS_WIN else {"ffmpeg", "ffprobe"}
    if ff_url and not is_version_current(FFMPEG_DIR, ff_tag, FFMPEG_EXE, force):
        archive = TMP_DIR / (
            "ffmpeg.zip" if ff_url.endswith(".zip") else "ffmpeg.tar.xz"
        )
        fetch_payload(ff_url, archive)
        temp_ext = TMP_DIR / "ff_ext"
        unpack_archive(archive, temp_ext, flatten_single_dir=False)
        safe_rmtree(FFMPEG_DIR)
        FFMPEG_DIR.mkdir(parents=True, exist_ok=True)
        for item in temp_ext.rglob("*"):
            if item.is_file() and (
                item.name.lower() in ff_bins
                or item.suffix.lower() in (".dll", ".so", ".dylib")
            ):
                shutil.copy2(to_unc_path(item), to_unc_path(FFMPEG_DIR / item.name))
                if not IS_WIN:
                    os.chmod(FFMPEG_DIR / item.name, 0o755)
        safe_rmtree(temp_ext)
        archive.unlink(missing_ok=True)
        write_version_marker(FFMPEG_DIR, ff_tag)
    else:
        if FFMPEG_DIR.exists():
            for stale_doc in FFMPEG_DIR.glob("*.html"):
                remove_readonly(None, str(stale_doc), None)
                stale_doc.unlink(missing_ok=True)
        log_info(f"FFmpeg converged ({ff_tag}).")


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


def hydrate_sources(force: bool = False) -> None:
    log_stage(
        "Resolving and Converging Upstream Vulkan, Shaderc, Math, and Image Codec Sources"
    )
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    sysroot_inc = SYSROOT_DIR / "include"
    sysroot_inc.mkdir(parents=True, exist_ok=True)
    vk_headers_dir = SOURCES_DIR / "vulkan_headers"
    vkh_tag, vkh_url = resolve_github_repo_snapshot(
        "KhronosGroup/Vulkan-Headers", vk_headers_dir
    )
    if vkh_url and not is_version_current(
        vk_headers_dir,
        vkh_tag,
        vk_headers_dir / "include" / "vulkan" / "vulkan.h",
        force,
    ):
        archive = TMP_DIR / "vulkan_headers.zip"
        fetch_payload(vkh_url, archive)
        unpack_archive(archive, vk_headers_dir, flatten_single_dir=True)
        archive.unlink(missing_ok=True)
        write_version_marker(vk_headers_dir, vkh_tag)
        copy_tree_contents(vk_headers_dir / "include", sysroot_inc)
    else:
        log_info(f"Vulkan-Headers converged ({vkh_tag}).")
        if not (sysroot_inc / "vulkan" / "vulkan.h").exists():
            copy_tree_contents(vk_headers_dir / "include", sysroot_inc)
    vk_loader_dir = SOURCES_DIR / "vulkan_loader"
    paired_loader_url = f"{HTTPS}github.com/KhronosGroup/Vulkan-Loader/archive/refs/tags/{vkh_tag}.zip"
    if probe_url(paired_loader_url):
        vkl_tag, vkl_url = vkh_tag, paired_loader_url
    else:
        vkl_tag, vkl_url = resolve_github_repo_snapshot(
            "KhronosGroup/Vulkan-Loader", vk_loader_dir
        )
    if vkl_url and not is_version_current(
        vk_loader_dir, vkl_tag, vk_loader_dir / "CMakeLists.txt", force
    ):
        archive = TMP_DIR / "vulkan_loader.zip"
        fetch_payload(vkl_url, archive)
        unpack_archive(archive, vk_loader_dir, flatten_single_dir=True)
        archive.unlink(missing_ok=True)
        write_version_marker(vk_loader_dir, vkl_tag)
    else:
        log_info(f"Vulkan-Loader converged ({vkl_tag}).")
    shaderc_dir = SOURCES_DIR / "shaderc"
    sc_tag, sc_url = resolve_github_repo_snapshot("google/shaderc", shaderc_dir)
    shaderc_updated = False
    if sc_url and not is_version_current(
        shaderc_dir, sc_tag, shaderc_dir / "CMakeLists.txt", force
    ):
        archive = TMP_DIR / "shaderc.zip"
        fetch_payload(sc_url, archive)
        unpack_archive(archive, shaderc_dir, flatten_single_dir=True)
        archive.unlink(missing_ok=True)
        write_version_marker(shaderc_dir, sc_tag)
        shaderc_updated = True
    else:
        log_info(f"Shaderc converged ({sc_tag}).")
    shaderc_tp = shaderc_dir / "third_party"
    if (
        shaderc_updated
        or force
        or not (shaderc_tp / "glslang" / "CMakeLists.txt").exists()
        or not (shaderc_tp / "spirv-tools" / "CMakeLists.txt").exists()
    ):
        log_stage(
            "Synchronizing Shaderc Third-Party Dependencies (glslang, spirv-tools, spirv-headers)"
        )
        sync_env = get_isolated_env()
        subprocess.run(
            [
                str(HERMETIC_PYTHON_EXE),
                "-I",
                str(shaderc_dir / "utils" / "git-sync-deps"),
            ],
            cwd=str(shaderc_dir),
            env=sync_env,
            check=True,
        )
    blis_dir = SOURCES_DIR / "blis"
    blis_tag, blis_url = resolve_github_repo_snapshot("amd/blis", blis_dir)
    if blis_url and not is_version_current(
        blis_dir, blis_tag, blis_dir / "CMakeLists.txt", force
    ):
        archive = TMP_DIR / "amd_blis.zip"
        fetch_payload(blis_url, archive)
        unpack_archive(archive, blis_dir, flatten_single_dir=True)
        archive.unlink(missing_ok=True)
        write_version_marker(blis_dir, blis_tag)
    else:
        log_info(f"AMD AOCL-BLIS converged ({blis_tag}).")
    fftw_dir = SOURCES_DIR / "fftw3"
    fftw_ver, fftw_url = resolve_fftw_url()
    if fftw_url and not is_version_current(
        fftw_dir, fftw_ver, fftw_dir / "CMakeLists.txt", force
    ):
        archive = TMP_DIR / "fftw3.tar.gz"
        fetch_payload(fftw_url, archive)
        unpack_archive(archive, fftw_dir, flatten_single_dir=True)
        archive.unlink(missing_ok=True)
        write_version_marker(fftw_dir, fftw_ver)
    else:
        log_info(f"FFTW3 converged ({fftw_ver}).")
    webp_dir = SOURCES_DIR / "libwebp"
    webp_tag, webp_url = resolve_github_repo_snapshot(
        "webmproject/libwebp", webp_dir
    )
    if webp_url and not is_version_current(
        webp_dir, webp_tag, webp_dir / "src" / "webp" / "decode.h", force
    ):
        archive = TMP_DIR / "libwebp.zip"
        fetch_payload(webp_url, archive)
        unpack_archive(archive, webp_dir, flatten_single_dir=True)
        archive.unlink(missing_ok=True)
        write_version_marker(webp_dir, webp_tag)
    else:
        log_info(f"libwebp converged ({webp_tag}).")
    spng_dir = SOURCES_DIR / "libspng"
    spng_tag, spng_url = resolve_github_repo_snapshot(
        "randy408/libspng", spng_dir
    )
    if spng_url and not is_version_current(
        spng_dir, spng_tag, spng_dir / "spng" / "spng.c", force
    ):
        archive = TMP_DIR / "libspng.zip"
        fetch_payload(spng_url, archive)
        unpack_archive(archive, spng_dir, flatten_single_dir=True)
        archive.unlink(missing_ok=True)
        write_version_marker(spng_dir, spng_tag)
    else:
        log_info(f"libspng converged ({spng_tag}).")
    miniz_dir = SOURCES_DIR / "miniz"
    miniz_tag, miniz_url = resolve_github_repo_snapshot(
        "richgel999/miniz", miniz_dir
    )
    if miniz_url and not is_version_current(
        miniz_dir, miniz_tag, miniz_dir / "miniz.c", force
    ):
        archive = TMP_DIR / "miniz.zip"
        fetch_payload(miniz_url, archive)
        unpack_archive(archive, miniz_dir, flatten_single_dir=True)
        archive.unlink(missing_ok=True)
        write_version_marker(miniz_dir, miniz_tag)
    else:
        log_info(f"miniz converged ({miniz_tag}).")
    stb_dir = SOURCES_DIR / "stb"
    stb_tag, stb_url = resolve_github_repo_snapshot(
        "nothings/stb", stb_dir, prefer_default_branch=True
    )
    if stb_url and not is_version_current(
        stb_dir, stb_tag, stb_dir / "stb_image.h", force
    ):
        archive = TMP_DIR / "stb.zip"
        fetch_payload(stb_url, archive)
        unpack_archive(archive, stb_dir, flatten_single_dir=True)
        archive.unlink(missing_ok=True)
        write_version_marker(stb_dir, stb_tag)
    else:
        log_info(f"stb converged ({stb_tag[:12]}).")


def orchestrate_hermetic_runtime_environment(
    profile: str, llvm_target: str, hsa_override: str = ""
) -> None:
    miopen_db = HF_CACHE_DIR / "miopen" / "db"
    miopen_kernels = HF_CACHE_DIR / "miopen" / "kernels"
    inductor_dir = HF_CACHE_DIR / "torch_inductor"
    triton_dir = HF_CACHE_DIR / "triton"
    for p in (miopen_db, miopen_kernels, inductor_dir, triton_dir):
        p.mkdir(parents=True, exist_ok=True)

    (BASE_DIR / "launch_env.bat").unlink(missing_ok=True)
    (BASE_DIR / "launch_env.ps1").unlink(missing_ok=True)

    if IS_WIN:
        site_packages_dirs = [PYTHON_DIR / "Lib" / "site-packages"]
    else:
        site_packages_dirs = list((PYTHON_DIR / "lib").glob("python*/site-packages"))

    rocm_family = (
        AMD_STAGING_FAMILY_MAP.get(llvm_target, f"{llvm_target}-all")
        if (profile == "ROCM" and llvm_target)
        else ""
    )

    sitecustomize_code = f"""from __future__ import annotations

import os
from pathlib import Path
import sys

_HERE = Path(__file__).resolve()
_SP_DIR = _HERE.parent
_ROOT = _HERE.parents[{4 if IS_WIN else 5}]
_BIN = _ROOT / "bin"
_PY = _BIN / "python"
_CACHE = _ROOT / ".forge_cache" / "env"

_env_hf = os.environ.get("HF_HOME")
if _env_hf and _env_hf.strip():
    _HF = Path(_env_hf).resolve()
elif (_ROOT.parent / ".hf_cache").exists():
    _HF = (_ROOT.parent / ".hf_cache").resolve()
else:
    _HF = (_ROOT / ".hf_cache").resolve()


def _env_default(key: str, val: str) -> None:
    cur = os.environ.get(key)
    if cur is None or not cur.strip():
        os.environ[key] = val


_ROCM_DEVEL = _SP_DIR / "_rocm_sdk_devel"
_ROCM_CORE = _SP_DIR / "_rocm_sdk_core"
if (_ROCM_DEVEL / "bin").is_dir():
    _ROCM_ROOT = _ROCM_DEVEL.resolve()
elif (_ROCM_CORE / "bin").is_dir():
    _ROCM_ROOT = _ROCM_CORE.resolve()
else:
    _ROCM_ROOT = _PY.resolve()

_LLVM_BIN = _ROCM_ROOT / "lib" / "llvm" / "bin"
_DEVICE_LIB = _ROCM_ROOT / "lib" / "llvm" / "amdgcn" / "bitcode"
_ROCM_CMAKE = _ROCM_ROOT / "lib" / "cmake"
_MIOPEN_SYS_DB = _ROCM_ROOT / "share" / "miopen" / "db"

_env_default("PYTHONNOUSERSITE", "1")
_env_default("PYTHONUTF8", "1")
_env_default("PYTHONIOENCODING", "utf-8")

_cur_rocm = os.environ.get("ROCM_PATH", "")
if not _cur_rocm.strip() or Path(_cur_rocm).resolve() == _PY.resolve():
    os.environ["ROCM_PATH"] = str(_ROCM_ROOT)
_env_default("ROCM_HOME", str(_ROCM_ROOT))
_env_default("HIP_PATH", str(_ROCM_ROOT))
if _LLVM_BIN.is_dir():
    _env_default("HIP_CLANG_PATH", str(_LLVM_BIN))
if _DEVICE_LIB.is_dir():
    _env_default("HIP_DEVICE_LIB_PATH", str(_DEVICE_LIB))
if _ROCM_CMAKE.is_dir():
    _env_default("CMAKE_PREFIX_PATH", str(_ROCM_CMAKE))
if _MIOPEN_SYS_DB.is_dir():
    _env_default("MIOPEN_SYSTEM_DB_PATH", str(_MIOPEN_SYS_DB))

if {repr(profile == "ROCM")}:
    _env_default("HIP_PLATFORM", "amd")
if {repr(bool(llvm_target))}:
    _env_default("PYTORCH_ROCM_ARCH", {repr(llvm_target)})
if {repr(bool(rocm_family))}:
    _env_default("ROCM_SDK_TARGET_FAMILY", {repr(rocm_family)})

_env_default("HF_HOME", str(_HF))
_env_default("TORCH_HOME", str(_HF / "torch"))
_env_default("TRANSFORMERS_CACHE", str(_HF / "transformers"))
_env_default("HUGGINGFACE_HUB_CACHE", str(_HF / "hub"))
_env_default("MIOPEN_USER_DB_PATH", str(_HF / "miopen" / "db"))
_env_default("MIOPEN_CUSTOM_CACHE_DIR", str(_HF / "miopen" / "kernels"))
_env_default("TORCHINDUCTOR_CACHE_DIR", str(_HF / "torch_inductor"))
_env_default("TRITON_CACHE_DIR", str(_HF / "triton"))
_env_default("MIOPEN_FIND_MODE", "1")
_env_default("MIOPEN_LOG_LEVEL", "0")
_env_default("MIOPEN_ENABLE_LOGGING", "0")
_env_default("AMD_LOG_LEVEL", "0")
_env_default("TORCH_BLAS_PREFER_HIPBLASLT", "1")
_env_default("HSA_ENABLE_SDMA", "1")
_env_default("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
_env_default("PYTORCH_ALLOC_CONF", "{ALLOC_CONF_POLICY}")
_env_default("PYTORCH_CUDA_ALLOC_CONF", "{ALLOC_CONF_POLICY}")
_env_default("PYTORCH_HIP_ALLOC_CONF", "{ALLOC_CONF_POLICY}")

if {repr(bool(hsa_override))}:
    _env_default("HSA_OVERRIDE_GFX_VERSION", {repr(hsa_override)})
elif os.environ.get("HSA_OVERRIDE_GFX_VERSION", "") == "":
    os.environ.pop("HSA_OVERRIDE_GFX_VERSION", None)

_hermetic_bins = [
    str(_PY if sys.platform == "win32" else _PY / "bin"),
    str(_PY / "Scripts" if sys.platform == "win32" else _PY / "bin"),
]
if (_ROCM_ROOT / "bin").is_dir():
    _hermetic_bins.append(str(_ROCM_ROOT / "bin"))
if _LLVM_BIN.is_dir():
    _hermetic_bins.append(str(_LLVM_BIN))
for _pkg_b in sorted(_SP_DIR.glob("_rocm_sdk_*/bin")):
    if _pkg_b.is_dir() and str(_pkg_b) not in _hermetic_bins:
        _hermetic_bins.append(str(_pkg_b))
_hermetic_bins.extend(
    [
        str(_BIN / "llvm" / "bin"),
        str(_BIN / "cmake" / "bin"),
        str(_BIN / "ninja"),
        str(_BIN / "nasm"),
        str(_BIN / "shaderc" / "bin"),
        str(_BIN / "ffmpeg"),
        str(_BIN / "sysroot" / "bin"),
    ]
)
_cur_path = os.environ.get("PATH", "")
os.environ["PATH"] = os.pathsep.join(
    _hermetic_bins + ([_cur_path] if _cur_path else [])
)

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _dll_dirs = [
        _ROOT / "dist" / "bin",
        _BIN / "sysroot" / "bin",
        _BIN / "ffmpeg",
        _ROCM_ROOT / "bin",
        _LLVM_BIN,
    ] + list(_SP_DIR.glob("_rocm_sdk_*/bin"))
    for _dll_dir in _dll_dirs:
        if _dll_dir.is_dir():
            try:
                os.add_dll_directory(str(_dll_dir.resolve()))
            except Exception:
                pass
"""
    for sp_dir in site_packages_dirs:
        sp_dir.mkdir(parents=True, exist_ok=True)
        (sp_dir / "sitecustomize.py").write_text(sitecustomize_code, encoding="utf-8")


def clean_hydration() -> None:
    log_stage("Purging Hydrated Toolchains, Sources, and Caches (--clean)")
    safe_rmtree(BIN_DIR)
    safe_rmtree(ENV_CACHE_DIR)
    (BASE_DIR / "launch_env.bat").unlink(missing_ok=True)
    (BASE_DIR / "launch_env.ps1").unlink(missing_ok=True)
    log_info("Repository returned to unhydrated baseline.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VizInjector Autonomous State-Convergent Hydrator"
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
        help="Purge all hydrated toolchains, sources, and caches",
    )
    args, _ = parser.parse_known_args()
    if args.clean:
        clean_hydration()
        return

    BIN_DIR.mkdir(parents=True, exist_ok=True)
    (SYSROOT_DIR / "include").mkdir(parents=True, exist_ok=True)
    (SYSROOT_DIR / "lib").mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    bootstrap_hermetic_python(requested_version=args.python, force=args.force)

    current_exe = Path(sys.executable).resolve()
    target_exe = HERMETIC_PYTHON_EXE.resolve()
    if current_exe != target_exe:
        log_info(
            f"Detaching from host Python ({sys.version.split()[0]}). Trampolining into hermetic runtime: {target_exe}"
        )
        isolated_env = get_isolated_env()
        cmd = [str(target_exe), str(Path(__file__).resolve())] + sys.argv[1:]
        res = subprocess.run(cmd, env=isolated_env)
        sys.exit(res.returncode)

    profile, llvm_target, hsa_override = synchronize_pip_and_pytorch(
        force=args.force
    )
    hydrate_host_tools(force=args.force)
    hydrate_winsdk(force=args.force)
    hydrate_sources(force=args.force)
    orchestrate_hermetic_runtime_environment(
        profile=profile, llvm_target=llvm_target, hsa_override=hsa_override
    )
    safe_rmtree(TMP_DIR)
    log_info("VizInjector hermetic hydration converged to latest upstream state.")


if __name__ == "__main__":
    main()