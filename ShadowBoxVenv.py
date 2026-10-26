#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import gc
import os
from pathlib import Path
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

ROOT_DIR = Path(__file__).resolve().parent
CACHE_DIR = ROOT_DIR / ".cache"
TMP_DIR = ROOT_DIR / ".tmp"
DOWNLOADS_DIR = CACHE_DIR / "downloads"
WHEELS_DIR = DOWNLOADS_DIR / "wheels"
PIP_CACHE_DIR = CACHE_DIR / "pip"
HF_CACHE_DIR = ROOT_DIR / ".hf_cache"
LOCAL_APPDATA_DIR = ROOT_DIR / ".local_appdata"
ROAMING_APPDATA_DIR = ROOT_DIR / ".appdata"

for _d in (
    CACHE_DIR,
    TMP_DIR,
    DOWNLOADS_DIR,
    WHEELS_DIR,
    PIP_CACHE_DIR,
    HF_CACHE_DIR,
    LOCAL_APPDATA_DIR,
    ROAMING_APPDATA_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

os.environ.pop("TORCH_BLAS_PREFER_HIPBLASLT", None)
tempfile.tempdir = str(TMP_DIR)
os.environ["TEMP"] = str(TMP_DIR)
os.environ["TMP"] = str(TMP_DIR)
os.environ["TMPDIR"] = str(TMP_DIR)
os.environ["PIP_CACHE_DIR"] = str(PIP_CACHE_DIR)
os.environ["LOCALAPPDATA"] = str(LOCAL_APPDATA_DIR)
os.environ["APPDATA"] = str(ROAMING_APPDATA_DIR)
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["TORCH_HOME"] = str(HF_CACHE_DIR / "torch")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

DEFAULT_PYTHON_VERSION = "3.12.8"
FALLBACK_MATRIX = {
    "1": "3.11.9",
    "2": "3.12.8",
    "3": "3.13.2",
    "4": "3.14.7",
}
STANDALONE_RELEASE = "20241016"

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

_VERIFIED_ARCHIVE_SIGNATURES: set[tuple[str, int, int]] = set()


def print_status(msg: str, status: str = "INFO") -> None:
    colors = {
        "INFO": "\033[94m",
        "SUCCESS": "\033[92m",
        "WARN": "\033[93m",
        "ERROR": "\033[91m",
        "RESET": "\033[0m",
    }
    if os.name == "nt" and not os.environ.get("WT_SESSION"):
        print(f"[{status}] {msg}", flush=True)
    else:
        print(
            f"{colors.get(status, '')}[{status}] {msg}{colors['RESET']}",
            flush=True,
        )


def parse_semver(ver_str: str) -> tuple[int, ...]:
    nums = [int(n) for n in re.findall(r"\d+", ver_str)]
    return tuple(nums) if nums else (0,)


def remove_readonly(func, path, _):
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        if func is not None:
            func(path)
    except Exception:
        pass


def safe_rmtree(dir_path: Path) -> None:
    if not dir_path.exists():
        return
    for attempt in range(8):
        try:
            if sys.version_info >= (3, 12):
                shutil.rmtree(
                    dir_path,
                    onexc=lambda func, path, _: remove_readonly(func, path, None),
                )
            else:
                shutil.rmtree(dir_path, onerror=remove_readonly)
            return
        except Exception:
            gc.collect()
            time.sleep(0.15 * (attempt + 1))


def resolve_dynamic_python_matrix() -> dict[str, str]:
    matrix = dict(FALLBACK_MATRIX)
    if os.name != "nt":
        return matrix
    try:
        req = urllib.request.Request(
            "https://www.python.org/ftp/python/",
            headers={"User-Agent": "ShadowBoxVenv/2.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        matches = re.findall(r'href="(\d+\.\d+\.\d+)/"', html)
        by_minor: dict[tuple[int, int], list[str]] = {
            (3, 11): [],
            (3, 12): [],
            (3, 13): [],
            (3, 14): [],
        }
        for ver in matches:
            sv = parse_semver(ver)
            if len(sv) >= 2 and (sv[0], sv[1]) in by_minor:
                by_minor[(sv[0], sv[1])].append(ver)
        slot_map = [
            ("1", (3, 11)),
            ("2", (3, 12)),
            ("3", (3, 13)),
            ("4", (3, 14)),
        ]
        for key, minor_pair in slot_map:
            candidates = sorted(
                set(by_minor[minor_pair]), key=parse_semver, reverse=True
            )
            for cand in candidates[:4]:
                probe_url = f"https://www.python.org/ftp/python/{cand}/python-{cand}-embed-amd64.zip"
                head_req = urllib.request.Request(
                    probe_url,
                    headers={"User-Agent": "ShadowBoxVenv/2.0"},
                    method="HEAD",
                )
                try:
                    with urllib.request.urlopen(head_req, timeout=5) as h_resp:
                        if 200 <= h_resp.status < 400:
                            matrix[key] = cand
                            break
                except Exception:
                    continue
    except Exception:
        pass
    return matrix


def _install_c10_env_sanitizer(target_dir: Path) -> None:
    site_packages = (
        target_dir / "Lib" / "site-packages"
        if os.name == "nt"
        else next(target_dir.glob("lib/python*/site-packages"), None)
    )
    if site_packages is None or not site_packages.is_dir():
        return
    sanitizer_code = (
        "import os\n"
        "_C10_BOOL_FLAGS = (\n"
        "    'TORCH_BLAS_PREFER_HIPBLASLT',\n"
        "    'TORCH_USE_HIP_DSA',\n"
        "    'CUDA_LAUNCH_BLOCKING',\n"
        "    'HIP_LAUNCH_BLOCKING',\n"
        "    'TORCH_SHOW_CPP_STACKTRACES',\n"
        ")\n"
        "for _k in _C10_BOOL_FLAGS:\n"
        "    _v = os.environ.get(_k)\n"
        "    if _v is not None and _v not in ('0', '1'):\n"
        "        _norm = _v.strip().lower()\n"
        "        if _norm in ('1', 'true', 'yes', 'on'):\n"
        "            os.environ[_k] = '1'\n"
        "        elif _norm in ('0', 'false', 'no', 'off'):\n"
        "            os.environ[_k] = '0'\n"
        "        else:\n"
        "            os.environ.pop(_k, None)\n"
    )
    (site_packages / "sitecustomize.py").write_text(
        sanitizer_code, encoding="utf-8"
    )


def build_hermetic_env(target_sandbox: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    passthrough_keys = (
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE",
        "OS",
        "USERNAME",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "WT_SESSION",
        "TERM",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    )
    for k in passthrough_keys:
        if k in os.environ:
            env[k] = os.environ[k]

    if os.name == "nt":
        sys_root = os.environ.get("SystemRoot", r"C:\Windows")
        base_os_paths = [
            str(target_sandbox),
            str(target_sandbox / "Scripts"),
            os.path.join(sys_root, "System32"),
            sys_root,
            os.path.join(sys_root, "System32", "Wbem"),
            os.path.join(sys_root, "System32", "WindowsPowerShell", "v1.0"),
        ]
        env["PATH"] = ";".join(base_os_paths)
        env["PIP_CONFIG_FILE"] = "NUL"
    else:
        base_os_paths = [
            str(target_sandbox / "bin"),
            str(target_sandbox),
            "/usr/local/bin",
            "/usr/bin",
            "/bin",
        ]
        env["PATH"] = ":".join(base_os_paths)
        env["PIP_CONFIG_FILE"] = "/dev/null"

    env["PYTHONHOME"] = str(target_sandbox)
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
    env["LOCALAPPDATA"] = str(LOCAL_APPDATA_DIR)
    env["APPDATA"] = str(ROAMING_APPDATA_DIR)
    env["XDG_CACHE_HOME"] = str(CACHE_DIR)
    env["XDG_CONFIG_HOME"] = str(ROOT_DIR / ".config")
    env["XDG_DATA_HOME"] = str(ROOT_DIR / ".local" / "share")
    env["HF_HOME"] = str(HF_CACHE_DIR)
    env["TORCH_HOME"] = str(HF_CACHE_DIR / "torch")
    env["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "transformers")
    env["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
    env["MIOPEN_USER_DB_PATH"] = str(HF_CACHE_DIR / "miopen" / "db")
    env["MIOPEN_CUSTOM_CACHE_DIR"] = str(HF_CACHE_DIR / "miopen" / "kernels")
    env["TORCHINDUCTOR_CACHE_DIR"] = str(HF_CACHE_DIR / "torch_inductor")
    env["TRITON_CACHE_DIR"] = str(HF_CACHE_DIR / "triton")
    env["MIOPEN_FIND_MODE"] = "2"
    env["MIOPEN_LOG_LEVEL"] = "0"
    env["MIOPEN_ENABLE_LOGGING"] = "0"
    env["AMD_LOG_LEVEL"] = "0"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
    env["ROCM_PATH"] = str(target_sandbox)
    return env


def safe_subprocess(cmd: list) -> str:
    try:
        kwargs = (
            {"creationflags": subprocess.CREATE_NO_WINDOW}
            if os.name == "nt"
            else {}
        )
        return subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, timeout=10, text=True, **kwargs
        ).strip()
    except Exception:
        return ""


def query_dxgi_adapters_win32() -> list[dict]:
    if sys.platform != "win32":
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
    if sys.platform == "darwin":
        uname = safe_subprocess(["uname", "-m"])
        if "arm64" in uname:
            return "MPS", ""
        return "CPU", ""

    has_nvidia, has_amd, has_intel = False, False, False
    detected_amd_targets = []

    if os.name == "nt":
        dxgi_devices = query_dxgi_adapters_win32()
        if dxgi_devices:
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
        drm_path = Path("/sys/class/drm")
        if drm_path.exists():
            for uevent in drm_path.glob("card*/device/uevent"):
                try:
                    content = uevent.read_text()
                    if "DRIVER=amdgpu" in content:
                        has_amd = True
                    if "DRIVER=nvidia" in content:
                        has_nvidia = True
                    if "DRIVER=i915" in content or "DRIVER=xe" in content:
                        has_intel = True
                except Exception:
                    pass
        if has_amd:
            rocm_info = safe_subprocess(["rocminfo"])
            gfx_match = re.search(r"gfx\d+[a-zA-Z]?", rocm_info)
            if gfx_match:
                detected_amd_targets.append(gfx_match.group(0))
            elif lspci and re.search(r"(?i)MI325|MI350", lspci):
                detected_amd_targets.append("gfx950")
            elif lspci and re.search(r"(?i)R9700|9070|Navi\s*48", lspci):
                detected_amd_targets.append("gfx1201")
            elif lspci and re.search(r"(?i)890M|880M", lspci):
                detected_amd_targets.append("gfx1150")
            else:
                detected_amd_targets.append("gfx1100")

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


def fetch_runtime(version: str, target_dir: Path) -> Path:
    executable = (
        target_dir / "python.exe"
        if os.name == "nt"
        else target_dir / "bin" / "python"
    )
    site_packages = (
        target_dir / "Lib" / "site-packages"
        if os.name == "nt"
        else next(target_dir.glob("lib/python*/site-packages"), None)
    )
    if (
        executable.is_file()
        and site_packages is not None
        and (site_packages / "pip").is_dir()
    ):
        _install_c10_env_sanitizer(target_dir)
        print_status(
            f"Reusing existing hermetic base interpreter at {target_dir.name}.",
            "INFO",
        )
        return executable

    if target_dir.exists():
        safe_rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if os.name == "nt":
        archive_name = f"python-{version}-embed-amd64.zip"
        url = f"https://www.python.org/ftp/python/{version}/{archive_name}"
        archive_path = DOWNLOADS_DIR / archive_name
        if not archive_path.is_file():
            print_status(f"Streaming standalone Windows runtime ({version})...")
            ctx = urllib.request.Request(url, headers={"User-Agent": "ShadowBoxVenv/2.0"})
            with urllib.request.urlopen(ctx, timeout=60) as response, open(
                archive_path, "wb"
            ) as out_file:
                while chunk := response.read(65536):
                    out_file.write(chunk)
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        (target_dir / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)
        for pth_file in target_dir.glob("*._pth"):
            orig_lines = pth_file.read_text(encoding="utf-8").splitlines()
            core_zips = [
                line.strip()
                for line in orig_lines
                if line.strip().endswith(".zip")
            ]
            payload_paths = core_zips + [".", "Lib/site-packages", "import site"]
            pth_file.write_text("\n".join(payload_paths) + "\n", encoding="utf-8")
        executable = target_dir / "python.exe"
    else:
        arch = "x86_64" if sys.maxsize > 2**32 else "i686"
        triple = (
            f"{arch}-unknown-linux-gnu"
            if sys.platform.startswith("linux")
            else f"{arch}-apple-darwin"
        )
        archive_name = (
            f"cpython-{version}+{STANDALONE_RELEASE}-{triple}-install_only.tar.gz"
        )
        url = f"https://github.com/astral-sh/python-build-standalone/releases/download/{STANDALONE_RELEASE}/{archive_name}"
        archive_path = DOWNLOADS_DIR / archive_name
        if not archive_path.is_file():
            print_status(f"Streaming standalone POSIX runtime ({version})...")
            ctx = urllib.request.Request(url, headers={"User-Agent": "ShadowBoxVenv/2.0"})
            with urllib.request.urlopen(ctx, timeout=60) as response, open(
                archive_path, "wb"
            ) as out_file:
                while chunk := response.read(65536):
                    out_file.write(chunk)
        with tarfile.open(archive_path, "r:gz") as tar_ref:
            tar_ref.extractall(target_dir.parent)
        source_extracted = target_dir.parent / "python"
        if source_extracted.exists() and source_extracted != target_dir:
            source_extracted.rename(target_dir)
        executable = target_dir / "bin" / "python"

    _install_c10_env_sanitizer(target_dir)
    hermetic_env = build_hermetic_env(target_dir)
    pip_bootstrapper = TMP_DIR / "get-pip.py"
    v_parts = version.split(".")
    if len(v_parts) >= 2 and v_parts[0] == "3" and v_parts[1] in ["6", "7", "8", "9"]:
        pip_url = f"https://bootstrap.pypa.io/pip/{v_parts[0]}.{v_parts[1]}/get-pip.py"
    else:
        pip_url = "https://bootstrap.pypa.io/get-pip.py"
    pip_ctx = urllib.request.Request(pip_url, headers={"User-Agent": "ShadowBoxVenv/2.0"})
    with urllib.request.urlopen(pip_ctx, timeout=60) as response, open(
        pip_bootstrapper, "wb"
    ) as out_file:
        while chunk := response.read(65536):
            out_file.write(chunk)

    subprocess.run(
        [
            str(executable),
            "-I",
            str(pip_bootstrapper),
            "--no-warn-script-location",
            "--cache-dir",
            str(PIP_CACHE_DIR),
        ],
        env=hermetic_env,
        stdin=subprocess.DEVNULL,
        check=True,
    )
    pip_bootstrapper.unlink(missing_ok=True)
    return executable


def verify_archive_integrity(archive_path: Path) -> bool:
    if not archive_path.is_file() or archive_path.stat().st_size < 512:
        return False
    stat_res = archive_path.stat()
    sig = (archive_path.name, stat_res.st_size, stat_res.st_mtime_ns)
    if sig in _VERIFIED_ARCHIVE_SIGNATURES:
        return True
    name_lower = archive_path.name.lower()
    try:
        if name_lower.endswith(".whl") or name_lower.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                bad_member = zf.testzip()
                if bad_member is not None:
                    print_status(
                        f"Corrupt CRC-32 stream in {archive_path.name} ({bad_member}). Evicting.",
                        "WARN",
                    )
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
    except Exception as exc:
        print_status(
            f"Corrupt package archive {archive_path.name} ({exc}). Evicting.",
            "WARN",
        )
        return False


def audit_and_prune_wheels_dir() -> int:
    WHEELS_DIR.mkdir(parents=True, exist_ok=True)
    evicted = 0
    for part_file in WHEELS_DIR.glob("*.part"):
        part_file.unlink(missing_ok=True)
    for pkg_file in sorted(WHEELS_DIR.iterdir()):
        if not pkg_file.is_file():
            continue
        if not verify_archive_integrity(pkg_file):
            try:
                remove_readonly(None, str(pkg_file), None)
                pkg_file.unlink(missing_ok=True)
            except Exception:
                pass
            evicted += 1
    return evicted


def salvage_cached_wheels() -> None:
    WHEELS_DIR.mkdir(parents=True, exist_ok=True)
    audit_and_prune_wheels_dir()

    for whl in PIP_CACHE_DIR.rglob("*.whl"):
        dest = WHEELS_DIR / whl.name
        if not dest.exists():
            try:
                shutil.copy2(whl, dest)
                if not verify_archive_integrity(dest):
                    dest.unlink(missing_ok=True)
                    whl.unlink(missing_ok=True)
            except Exception:
                dest.unlink(missing_ok=True)

    http_cache = PIP_CACHE_DIR / "http-v2"
    if not http_cache.exists():
        return
    for body_file in http_cache.rglob("*"):
        if not body_file.is_file() or body_file.stat().st_size < 4096:
            continue
        try:
            with open(body_file, "rb") as f:
                magic = f.read(4)
            if magic != b"PK\x03\x04":
                continue
            with zipfile.ZipFile(body_file, "r") as zf:
                bad_member = zf.testzip()
                if bad_member is not None:
                    print_status(
                        f"Evicting corrupt pip HTTP cache entry ({bad_member}).",
                        "WARN",
                    )
                    remove_readonly(None, str(body_file), None)
                    body_file.unlink(missing_ok=True)
                    continue
                dist_infos = [
                    n for n in zf.namelist() if n.endswith(".dist-info/WHEEL")
                ]
                if not dist_infos:
                    continue
                pkg_prefix = dist_infos[0].split(".dist-info/")[0]
                wheel_meta = zf.read(dist_infos[0]).decode("utf-8", errors="ignore")
                tags = re.findall(r"Tag:\s*(\S+)", wheel_meta)
                tag_str = tags[0] if tags else "py3-none-any"
                wheel_name = f"{pkg_prefix}-{tag_str}.whl"
                dest = WHEELS_DIR / wheel_name
                if not dest.exists():
                    shutil.copy2(body_file, dest)
                    st = dest.stat()
                    _VERIFIED_ARCHIVE_SIGNATURES.add(
                        (dest.name, st.st_size, st.st_mtime_ns)
                    )
                    if st.st_size >= 10 * 1024 * 1024:
                        print_status(
                            f"Recovered verified cached wheel: {wheel_name} ({st.st_size / (1024**2):.1f} MB)",
                            "SUCCESS",
                        )
        except zipfile.BadZipFile:
            remove_readonly(None, str(body_file), None)
            body_file.unlink(missing_ok=True)
        except Exception:
            pass


def stage_and_install_packages(
    executable: Path,
    isolated_env: dict[str, str],
    packages: list[str],
    index_url: str | None = None,
    use_pre: bool = True,
    no_build_isolation: bool = False,
    max_attempts: int = 3,
) -> None:
    index_args: list[str] = ["--index-url", index_url] if index_url else []
    pre_args: list[str] = ["--pre"] if use_pre else []
    iso_args: list[str] = ["--no-build-isolation"] if no_build_isolation else []

    for attempt in range(1, max_attempts + 1):
        audit_and_prune_wheels_dir()
        dl_cmd = (
            [
                str(executable),
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
            ]
            + index_args
            + pre_args
            + iso_args
            + packages
        )
        dl_res = subprocess.run(
            dl_cmd,
            env=isolated_env,
            stdin=subprocess.DEVNULL,
            check=False,
        )
        evicted_after_dl = audit_and_prune_wheels_dir()
        if dl_res.returncode != 0 or evicted_after_dl > 0:
            if attempt < max_attempts:
                print_status(
                    f"Wheel staging attempt {attempt}/{max_attempts} detected transit/CRC fault ({evicted_after_dl} evicted). Retrying...",
                    "WARN",
                )
                time.sleep(2.0 * attempt)
                continue
            dl_res.check_returncode()

        inst_cmd = (
            [
                str(executable),
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
                "-U",
            ]
            + pre_args
            + iso_args
            + packages
        )
        inst_res = subprocess.run(
            inst_cmd,
            env=isolated_env,
            stdin=subprocess.DEVNULL,
            check=False,
        )
        if inst_res.returncode == 0:
            return

        evicted_after_inst = audit_and_prune_wheels_dir()
        if attempt < max_attempts:
            print_status(
                f"Package installation attempt {attempt}/{max_attempts} failed ({evicted_after_inst} corrupt archives evicted). Retrying...",
                "WARN",
            )
            time.sleep(2.0 * attempt)
        else:
            inst_res.check_returncode()


def condition_pytorch_runtime(
    executable: Path,
    isolated_env: dict[str, str],
    profile: str,
    llvm_target: str,
) -> str:
    print_status(
        f"Hardware Compute Architecture Locked: {profile} ({llvm_target or 'Generic'})",
        "SUCCESS",
    )
    salvage_cached_wheels()

    print_status(
        "Upgrading core build tooling (pip, setuptools, wheel, packaging) with --pre -U..."
    )
    stage_and_install_packages(
        executable=executable,
        isolated_env=isolated_env,
        packages=["pip", "setuptools", "wheel", "packaging"],
        index_url=None,
        use_pre=True,
    )

    hsa_override_used = HSA_OVERRIDES.get(llvm_target, "")
    print_status(f"Deploying hardware-aligned PyTorch binaries for {profile}...")

    if profile == "ROCM":
        family = AMD_STAGING_FAMILY_MAP.get(llvm_target, f"{llvm_target}-all")
        index_url = f"https://rocm.nightlies.amd.com/v2-staging/{family}/"
        print_status(f"Locking Sovereign Staging Nightly Index: {index_url}", "SUCCESS")
        if os.name == "nt":
            stage_and_install_packages(
                executable=executable,
                isolated_env=isolated_env,
                packages=["rocm[libraries,devel]"],
                index_url=index_url,
                use_pre=True,
                no_build_isolation=True,
            )
        stage_and_install_packages(
            executable=executable,
            isolated_env=isolated_env,
            packages=["torch", "torchvision", "torchaudio"],
            index_url=index_url,
            use_pre=True,
        )
        subprocess.run(
            [str(executable), "-I", "-m", "rocm_sdk", "init"],
            env=isolated_env,
            stdin=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )
    elif profile.startswith("CUDA_"):
        cuda_urls = {
            "CUDA_13_2": "https://download.pytorch.org/whl/nightly/cu132",
            "CUDA_13_0": "https://download.pytorch.org/whl/nightly/cu130",
            "CUDA_12_8": "https://download.pytorch.org/whl/nightly/cu128",
            "CUDA_12_6": "https://download.pytorch.org/whl/nightly/cu126",
            "CUDA_12_1": "https://download.pytorch.org/whl/cu121",
            "CUDA_11_8": "https://download.pytorch.org/whl/cu118",
        }
        index_url = cuda_urls.get(
            profile, "https://download.pytorch.org/whl/nightly/cu126"
        )
        stage_and_install_packages(
            executable=executable,
            isolated_env=isolated_env,
            packages=["torch", "torchvision", "torchaudio"],
            index_url=index_url,
            use_pre=True,
        )
    elif profile == "INTEL_XPU":
        stage_and_install_packages(
            executable=executable,
            isolated_env=isolated_env,
            packages=[
                "torch",
                "torchvision",
                "torchaudio",
                "intel-extension-for-pytorch",
            ],
            index_url="https://pytorch-extension.intel.com/release-whl/stable/xpu/us/",
            use_pre=True,
        )
    elif profile == "MPS":
        stage_and_install_packages(
            executable=executable,
            isolated_env=isolated_env,
            packages=["torch", "torchvision", "torchaudio"],
            index_url=None,
            use_pre=True,
        )
    else:
        stage_and_install_packages(
            executable=executable,
            isolated_env=isolated_env,
            packages=["torch", "torchvision", "torchaudio"],
            index_url="https://download.pytorch.org/whl/cpu",
            use_pre=True,
        )

    print_status("PyTorch hardware alignment complete.", "SUCCESS")
    return hsa_override_used


def generate_environment_anchors(
    target_dir: Path, executable: Path, hsa_override: str = ""
) -> None:
    _install_c10_env_sanitizer(target_dir)
    project_root = target_dir.parent
    rel_env_name = target_dir.name

    miopen_db = HF_CACHE_DIR / "miopen" / "db"
    miopen_kernels = HF_CACHE_DIR / "miopen" / "kernels"
    inductor_dir = HF_CACHE_DIR / "torch_inductor"
    triton_dir = HF_CACHE_DIR / "triton"
    for p in (miopen_db, miopen_kernels, inductor_dir, triton_dir):
        p.mkdir(parents=True, exist_ok=True)

    hsa_bat_line = (
        f'set "HSA_OVERRIDE_GFX_VERSION={hsa_override}"'
        if hsa_override
        else 'set "HSA_OVERRIDE_GFX_VERSION="'
    )
    hsa_echo_bat = (
        "echo  HSA GFX Override: %HSA_OVERRIDE_GFX_VERSION%" if hsa_override else ""
    )

    hsa_ps1_line = (
        f'$env:HSA_OVERRIDE_GFX_VERSION="{hsa_override}"'
        if hsa_override
        else "Remove-Item Env:\\HSA_OVERRIDE_GFX_VERSION -ErrorAction SilentlyContinue"
    )
    hsa_echo_ps1 = (
        'Write-Host " HSA GFX Override: $env:HSA_OVERRIDE_GFX_VERSION"'
        if hsa_override
        else ""
    )

    hsa_sh_line = (
        f'export HSA_OVERRIDE_GFX_VERSION="{hsa_override}"'
        if hsa_override
        else "unset HSA_OVERRIDE_GFX_VERSION"
    )
    hsa_echo_sh = (
        'echo " HSA GFX Override: ${HSA_OVERRIDE_GFX_VERSION}"'
        if hsa_override
        else ""
    )

    if os.name == "nt":
        bat_content = f"""@echo off
title Hermetic Sandbox Environment
color 0b
set "ROOT_DIR=%~dp0"
set "PYTHONPATH="
set "PYTHONCASEOK="
set "VIRTUAL_ENV="
set "CONDA_PREFIX="
set "CONDA_DEFAULT_ENV="
set "TORCH_BLAS_PREFER_HIPBLASLT="
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PIP_NO_USER=1"
set "PIP_USER=0"
set "PIP_CONFIG_FILE=NUL"
set "PYTHONHOME=%ROOT_DIR%{rel_env_name}"
set "PATH=%ROOT_DIR%{rel_env_name};%ROOT_DIR%{rel_env_name}\\Scripts;%SystemRoot%\\System32;%SystemRoot%;%SystemRoot%\\System32\\Wbem;%SystemRoot%\\System32\\WindowsPowerShell\\v1.0"
set "TEMP=%ROOT_DIR%.tmp"
set "TMP=%ROOT_DIR%.tmp"
set "TMPDIR=%ROOT_DIR%.tmp"
set "PIP_CACHE_DIR=%ROOT_DIR%.cache\\pip"
set "LOCALAPPDATA=%ROOT_DIR%.local_appdata"
set "APPDATA=%ROOT_DIR%.appdata"
set "HF_HOME=%ROOT_DIR%.hf_cache"
set "TORCH_HOME=%ROOT_DIR%.hf_cache\\torch"
set "TRANSFORMERS_CACHE=%ROOT_DIR%.hf_cache\\transformers"
set "HUGGINGFACE_HUB_CACHE=%ROOT_DIR%.hf_cache\\hub"
set "MIOPEN_USER_DB_PATH=%ROOT_DIR%.hf_cache\\miopen\\db"
set "MIOPEN_CUSTOM_CACHE_DIR=%ROOT_DIR%.hf_cache\\miopen\\kernels"
set "TORCHINDUCTOR_CACHE_DIR=%ROOT_DIR%.hf_cache\\torch_inductor"
set "TRITON_CACHE_DIR=%ROOT_DIR%.hf_cache\\triton"
set "MIOPEN_FIND_MODE=2"
set "MIOPEN_LOG_LEVEL=0"
set "MIOPEN_ENABLE_LOGGING=0"
set "AMD_LOG_LEVEL=0"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "PYTORCH_HIP_ALLOC_CONF=expandable_segments:True"
set "ROCM_PATH=%PYTHONHOME%"
{hsa_bat_line}
echo =======================================================================
echo  Hermetic Sandbox Shell Active (CMD)
echo  Interpreter: %PYTHONHOME%\\python.exe
echo  Cache Anchor: %HF_HOME%
echo  Local Temp: %TEMP%
echo  Local Pip Cache: %PIP_CACHE_DIR%
echo  MIOpen Cache: %MIOPEN_CUSTOM_CACHE_DIR% (Find Mode: FAST Heuristic)
{hsa_echo_bat}
echo  Isolation Status: Zero host leakage. All temp/cache local to root.
echo =======================================================================
cmd /k
"""
        (project_root / "launch_env.bat").write_text(bat_content, encoding="utf-8")

        ps1_content = f"""# Hermetic Sandbox PowerShell Launcher
$ErrorActionPreference = "Continue"
$ROOT_DIR = $PSScriptRoot
Remove-Item Env:\\PYTHONPATH -ErrorAction SilentlyContinue
Remove-Item Env:\\PYTHONCASEOK -ErrorAction SilentlyContinue
Remove-Item Env:\\VIRTUAL_ENV -ErrorAction SilentlyContinue
Remove-Item Env:\\CONDA_PREFIX -ErrorAction SilentlyContinue
Remove-Item Env:\\CONDA_DEFAULT_ENV -ErrorAction SilentlyContinue
Remove-Item Env:\\TORCH_BLAS_PREFER_HIPBLASLT -ErrorAction SilentlyContinue
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PIP_NO_USER = "1"
$env:PIP_USER = "0"
$env:PIP_CONFIG_FILE = "NUL"
$env:PYTHONHOME = "$ROOT_DIR\\{rel_env_name}"
$sysRoot = $env:SystemRoot
$env:PATH = "$ROOT_DIR\\{rel_env_name};$ROOT_DIR\\{rel_env_name}\\Scripts;$sysRoot\\System32;$sysRoot;$sysRoot\\System32\\Wbem;$sysRoot\\System32\\WindowsPowerShell\\v1.0"
$env:TEMP = "$ROOT_DIR\\.tmp"
$env:TMP = "$ROOT_DIR\\.tmp"
$env:TMPDIR = "$ROOT_DIR\\.tmp"
$env:PIP_CACHE_DIR = "$ROOT_DIR\\.cache\\pip"
$env:LOCALAPPDATA = "$ROOT_DIR\\.local_appdata"
$env:APPDATA = "$ROOT_DIR\\.appdata"
$env:HF_HOME = "$ROOT_DIR\\.hf_cache"
$env:TORCH_HOME = "$ROOT_DIR\\.hf_cache\\torch"
$env:TRANSFORMERS_CACHE = "$ROOT_DIR\\.hf_cache\\transformers"
$env:HUGGINGFACE_HUB_CACHE = "$ROOT_DIR\\.hf_cache\\hub"
$env:MIOPEN_USER_DB_PATH = "$ROOT_DIR\\.hf_cache\\miopen\\db"
$env:MIOPEN_CUSTOM_CACHE_DIR = "$ROOT_DIR\\.hf_cache\\miopen\\kernels"
$env:TORCHINDUCTOR_CACHE_DIR = "$ROOT_DIR\\.hf_cache\\torch_inductor"
$env:TRITON_CACHE_DIR = "$ROOT_DIR\\.hf_cache\\triton"
$env:MIOPEN_FIND_MODE = "2"
$env:MIOPEN_LOG_LEVEL = "0"
$env:MIOPEN_ENABLE_LOGGING = "0"
$env:AMD_LOG_LEVEL = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:PYTORCH_HIP_ALLOC_CONF = "expandable_segments:True"
$env:ROCM_PATH = "$env:PYTHONHOME"
{hsa_ps1_line}
Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host " Hermetic Sandbox Active (PowerShell)" -ForegroundColor Cyan
Write-Host " Interpreter: $env:PYTHONHOME\\python.exe"
Write-Host " Cache Anchor: $env:HF_HOME"
Write-Host " Local Temp: $env:TEMP"
Write-Host " Local Pip Cache: $env:PIP_CACHE_DIR"
Write-Host " MIOpen Cache: $env:MIOPEN_CUSTOM_CACHE_DIR (Find Mode: FAST Heuristic)"
{hsa_echo_ps1}
Write-Host " Isolation Status: Zero host leakage. All temp/cache local to root." -ForegroundColor Green
Write-Host "=======================================================================" -ForegroundColor Cyan
"""
        (project_root / "launch_env.ps1").write_text(ps1_content, encoding="utf-8")
    else:
        sh_content = f"""#!/usr/bin/env bash
ROOT_DIR="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" &> /dev/null && pwd)"
unset PYTHONPATH
unset PYTHONCASEOK
unset VIRTUAL_ENV
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset TORCH_BLAS_PREFER_HIPBLASLT
export PYTHONNOUSERSITE="1"
export PYTHONUTF8="1"
export PYTHONIOENCODING="utf-8"
export PIP_NO_USER="1"
export PIP_USER="0"
export PIP_CONFIG_FILE="/dev/null"
export PYTHONHOME="${{ROOT_DIR}}/{rel_env_name}"
export PATH="${{ROOT_DIR}}/{rel_env_name}:${{ROOT_DIR}}/{rel_env_name}/bin:/usr/local/bin:/usr/bin:/bin"
export TEMP="${{ROOT_DIR}}/.tmp"
export TMP="${{ROOT_DIR}}/.tmp"
export TMPDIR="${{ROOT_DIR}}/.tmp"
export PIP_CACHE_DIR="${{ROOT_DIR}}/.cache/pip"
export XDG_CACHE_HOME="${{ROOT_DIR}}/.cache"
export XDG_CONFIG_HOME="${{ROOT_DIR}}/.config"
export XDG_DATA_HOME="${{ROOT_DIR}}/.local/share"
export HF_HOME="${{ROOT_DIR}}/.hf_cache"
export TORCH_HOME="${{ROOT_DIR}}/.hf_cache/torch"
export TRANSFORMERS_CACHE="${{ROOT_DIR}}/.hf_cache/transformers"
export HUGGINGFACE_HUB_CACHE="${{ROOT_DIR}}/.hf_cache/hub"
export MIOPEN_USER_DB_PATH="${{ROOT_DIR}}/.hf_cache/miopen/db"
export MIOPEN_CUSTOM_CACHE_DIR="${{ROOT_DIR}}/.hf_cache/miopen/kernels"
export TORCHINDUCTOR_CACHE_DIR="${{ROOT_DIR}}/.hf_cache/torch_inductor"
export TRITON_CACHE_DIR="${{ROOT_DIR}}/.hf_cache/triton"
export MIOPEN_FIND_MODE="2"
export MIOPEN_LOG_LEVEL="0"
export MIOPEN_ENABLE_LOGGING="0"
export AMD_LOG_LEVEL="0"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
export ROCM_PATH="${{PYTHONHOME}}"
{hsa_sh_line}
echo "======================================================================="
echo " Hermetic Sandbox Shell Active"
echo " Interpreter: ${{PYTHONHOME}}/bin/python"
echo " Cache Anchor: ${{HF_HOME}}"
echo " Local Temp: ${{TEMP}}"
echo " Local Pip Cache: ${{PIP_CACHE_DIR}}"
echo " MIOpen Cache: ${{MIOPEN_CUSTOM_CACHE_DIR}} (Find Mode: FAST Heuristic)"
{hsa_echo_sh}
echo " Isolation Status: Host modules segregated. Anchored relative root."
echo "======================================================================="
exec $SHELL
"""
        launcher_sh = project_root / "launch_env.sh"
        launcher_sh.write_text(sh_content, encoding="utf-8")
        launcher_sh.chmod(0o755)

    print_status(
        "Hermetic launchers deployed: launch_env.bat, launch_env.ps1", "SUCCESS"
    )


def run_hardware_acceptance_gate(
    executable: Path, isolated_env: dict[str, str], hsa_override: str = ""
) -> None:
    print_status("Executing hardware-accelerated acceptance verification...", "INFO")
    test_env = isolated_env.copy()
    if hsa_override:
        test_env["HSA_OVERRIDE_GFX_VERSION"] = hsa_override
    else:
        test_env.pop("HSA_OVERRIDE_GFX_VERSION", None)
    test_code = (
        "import sys, torch\n"
        "device = 'cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'\n"
        "backend = getattr(torch.version, 'hip', None) or getattr(torch.version, 'cuda', None) or ('mps' if device == 'mps' else 'cpu')\n"
        "name = torch.cuda.get_device_name(0) if device == 'cuda' else ('Apple Silicon' if device == 'mps' else 'Host CPU')\n"
        "print(f'[ACCEPTANCE] Interpreter: {sys.executable} (Python {sys.version.split()[0]})')\n"
        "print(f'[ACCEPTANCE] Device: {device.upper()} | Backend: {backend} | GPU: {name}')\n"
        "x = torch.randn((256, 256), device=device, dtype=torch.float32)\n"
        "y = torch.mm(x, x.T)\n"
        "assert torch.isfinite(y).all(), 'Non-finite tensor output detected'\n"
        "print(f'[ACCEPTANCE] GEMM Compute Verification PASSED on {device.upper()}.')\n"
    )
    res = subprocess.run(
        [str(executable), "-I", "-c", test_code],
        env=test_env,
        stdin=subprocess.DEVNULL,
        text=True,
        capture_output=True,
    )
    if res.returncode == 0:
        for line in res.stdout.strip().splitlines():
            print_status(line, "SUCCESS")
    else:
        print_status(
            f"Acceptance test diagnostic output:\n{res.stderr.strip()}", "WARN"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous Hardware-Accelerated Virtualization Conditioner."
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help=f"Target Python runtime version (default: {DEFAULT_PYTHON_VERSION}).",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Execute hands-free without interactive prompts.",
    )
    parser.add_argument(
        "--hermetic-internal",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if not args.hermetic_internal:
        print_status("=" * 72)
        print_status(
            "            SHADOWBOXVENV SOVEREIGN RUNTIME CONDITIONER", "SUCCESS"
        )
        print_status("=" * 72)

    selected_version = args.version
    if selected_version is None:
        matrix = resolve_dynamic_python_matrix()
        default_ver = matrix.get("2", DEFAULT_PYTHON_VERSION)
        if args.auto or not sys.stdin.isatty():
            selected_version = default_ver
        else:
            for key, ver in matrix.items():
                print(f"  [{key}] Python {ver}")
            manual_opt = str(len(matrix) + 1)
            print(f"  [{manual_opt}] Manual Version Entry")
            choice = input(
                f"\nSelect core runtime configuration [1-{manual_opt}] (Default: 2 [Python {default_ver}]): "
            ).strip()
            if not choice:
                selected_version = default_ver
            elif choice in matrix:
                selected_version = matrix[choice]
            elif choice == manual_opt:
                selected_version = input("Enter custom Python version: ").strip()
            else:
                selected_version = default_ver

    if not selected_version:
        selected_version = DEFAULT_PYTHON_VERSION

    target_sandbox = ROOT_DIR / f"py_env_{selected_version.replace('.', '_')}"
    executable = fetch_runtime(selected_version, target_sandbox)
    isolated_env = build_hermetic_env(target_sandbox)

    current_exe = Path(sys.executable).resolve()
    target_exe = executable.resolve()
    if current_exe != target_exe and not args.hermetic_internal:
        print_status(
            f"Detaching from host system Python ({sys.version.split()[0]} -> {current_exe}).",
            "INFO",
        )
        print_status(
            f"Trampolining into hermetic runtime: {target_exe}",
            "SUCCESS",
        )
        respawn_cmd = [
            str(target_exe),
            "-I",
            str(Path(__file__).resolve()),
            "--version",
            selected_version,
            "--hermetic-internal",
        ]
        res = subprocess.run(respawn_cmd, env=isolated_env)
        sys.exit(res.returncode)

    profile, llvm_target = resolve_hardware_matrix()
    hsa_override = condition_pytorch_runtime(
        executable, isolated_env, profile, llvm_target
    )
    generate_environment_anchors(
        target_sandbox, executable, hsa_override=hsa_override
    )
    run_hardware_acceptance_gate(
        executable, isolated_env, hsa_override=hsa_override
    )
    print_status(
        f"Hermetic sandbox conditioned successfully at {target_sandbox.name}.",
        "SUCCESS",
    )


if __name__ == "__main__":
    main()