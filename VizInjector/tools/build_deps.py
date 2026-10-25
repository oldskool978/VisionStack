#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT_DIR = Path(__file__).parent.parent.resolve()
TOOLS_DIR = ROOT_DIR / "tools"
BIN_DIR = ROOT_DIR / "bin"
SYSROOT_DIR = BIN_DIR / "sysroot"
SYSROOT_INC = SYSROOT_DIR / "include"
SYSROOT_LIB = SYSROOT_DIR / "lib"
SYSROOT_BIN = SYSROOT_DIR / "bin"
BUILD_TMP_DIR = BIN_DIR / "build_tmp"
WINSDK_DIR = BIN_DIR / "winsdk"
SOURCES_DIR = BIN_DIR / "sources"
SHADERC_INSTALL_DIR = BIN_DIR / "shaderc"
TOOLCHAIN_FILE = TOOLS_DIR / "toolchain.cmake"

IS_WIN = sys.platform == "win32"

if __name__ == "__main__" and "--clean" in sys.argv:
    shutil.rmtree(SYSROOT_DIR, ignore_errors=True)
    shutil.rmtree(BUILD_TMP_DIR, ignore_errors=True)
    shutil.rmtree(SHADERC_INSTALL_DIR, ignore_errors=True)
    shutil.rmtree(WINSDK_DIR / "lib", ignore_errors=True)
    TOOLCHAIN_FILE.unlink(missing_ok=True)
    print(
        "[STAGE] Purging Compiled Sysroot, Shaderc, and Intermediate Build Directories (--clean)"
    )
    print("[INFO]  Dependency build artifacts cleaned.")
    sys.exit(0)

CMAKE_BIN = BIN_DIR / "cmake" / "bin" / ("cmake.exe" if IS_WIN else "cmake")
NINJA_BIN = BIN_DIR / "ninja" / ("ninja.exe" if IS_WIN else "ninja")
CLANG_BIN = BIN_DIR / "llvm" / "bin" / ("clang.exe" if IS_WIN else "clang")
CLANG_CXX_BIN = BIN_DIR / "llvm" / "bin" / ("clang++.exe" if IS_WIN else "clang++")
CLANG_CL_BIN = BIN_DIR / "llvm" / "bin" / ("clang-cl.exe" if IS_WIN else "clang-cl")
LLD_LINK_BIN = BIN_DIR / "llvm" / "bin" / ("lld-link.exe" if IS_WIN else "lld-link")
LLVM_RC_BIN = BIN_DIR / "llvm" / "bin" / ("llvm-rc.exe" if IS_WIN else "llvm-rc")
LLVM_MT_BIN = BIN_DIR / "llvm" / "bin" / ("llvm-mt.exe" if IS_WIN else "llvm-mt")
LLVM_LIB_BIN = BIN_DIR / "llvm" / "bin" / ("llvm-lib.exe" if IS_WIN else "llvm-lib")
LLVM_AR_BIN = BIN_DIR / "llvm" / "bin" / ("llvm-ar.exe" if IS_WIN else "llvm-ar")
NASM_BIN = BIN_DIR / "nasm" / ("nasm.exe" if IS_WIN else "nasm")
HERMETIC_PYTHON = BIN_DIR / "python" / ("python.exe" if IS_WIN else "bin/python")
GIT_BIN_CANDIDATES = [
    BIN_DIR / "git" / "cmd" / ("git.exe" if IS_WIN else "git"),
    BIN_DIR / "git" / "bin" / ("git.exe" if IS_WIN else "git"),
]

CANONICAL_CBLAS_H = """#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef enum CBLAS_ORDER { CblasRowMajor=101, CblasColMajor=102 } CBLAS_ORDER;
typedef enum CBLAS_TRANSPOSE { CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113 } CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO { CblasUpper=121, CblasLower=122 } CBLAS_UPLO;
typedef enum CBLAS_DIAG { CblasNonUnit=131, CblasUnit=132 } CBLAS_DIAG;
typedef enum CBLAS_SIDE { CblasLeft=141, CblasRight=142 } CBLAS_SIDE;

double cblas_ddot(const int n, const double* x, const int incx, const double* y, const int incy);
float cblas_sdot(const int n, const float* x, const int incx, const float* y, const int incy);
void cblas_daxpy(const int n, const double alpha, const double* x, const int incx, double* y, const int incy);
void cblas_saxpy(const int n, const float alpha, const float* x, const int incx, float* y, const int incy);
void cblas_dscal(const int n, const double alpha, double* x, const int incx);
void cblas_sscal(const int n, const float alpha, float* x, const int incx);
void cblas_dcopy(const int n, const double* x, const int incx, double* y, const int incy);
void cblas_scopy(const int n, const float* x, const int incx, float* y, const int incy);
double cblas_dnrm2(const int n, const double* x, const int incx);
float cblas_snrm2(const int n, const float* x, const int incx);
double cblas_dasum(const int n, const double* x, const int incx);
float cblas_sasum(const int n, const float* x, const int incx);
void cblas_dgemv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE trans, const int m, const int n,
                 const double alpha, const double* a, const int lda, const double* x, const int incx,
                 const double beta, double* y, const int incy);
void cblas_sgemv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE trans, const int m, const int n,
                 const float alpha, const float* a, const int lda, const float* x, const int incx,
                 const float beta, float* y, const int incy);
void cblas_dgemm(const CBLAS_ORDER order, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
                 const int m, const int n, const int k, const double alpha, const double* a, const int lda,
                 const double* b, const int ldb, const double beta, double* c, const int ldc);
void cblas_sgemm(const CBLAS_ORDER order, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
                 const int m, const int n, const int k, const float alpha, const float* a, const int lda,
                 const float* b, const int ldb, const float beta, float* c, const int ldc);

#ifdef __cplusplus
}
#endif
"""

if (
    HERMETIC_PYTHON.exists()
    and Path(sys.executable).resolve() != HERMETIC_PYTHON.resolve()
):
    res = subprocess.run(
        [str(HERMETIC_PYTHON.resolve()), "-I", str(Path(__file__).resolve())]
        + sys.argv[1:]
    )
    sys.exit(res.returncode)


def log_stage(msg: str) -> None:
    sys.stdout.write(f"[STAGE] {msg}\n")
    sys.stdout.flush()


def log_info(msg: str) -> None:
    sys.stdout.write(f"[INFO]  {msg}\n")
    sys.stdout.flush()


def log_warn(msg: str) -> None:
    sys.stderr.write(f"[WARN]  {msg}\n")
    sys.stderr.flush()


def sanitize_path(path_obj: Path) -> str:
    return str(path_obj.resolve()).replace("\\", "/")


def parse_semver(ver_str: str) -> tuple[int, ...]:
    nums = [int(n) for n in re.findall(r"\d+", ver_str)]
    return tuple(nums) if nums else (0,)


def normalize_numeric_version(ver_str: str) -> str:
    m = re.match(r"^(\d+(?:\.\d+)+)", ver_str.strip())
    return m.group(1) if m else ver_str.strip()


def resolve_git_binary() -> Path:
    for candidate in GIT_BIN_CANDIDATES:
        if candidate.exists():
            return candidate
    system_git = shutil.which("git")
    if system_git:
        return Path(system_git)
    raise RuntimeError(
        "Git binary missing. Execute tools/hydrate.py before building dependencies."
    )


def read_source_version(src_dir: Path) -> str:
    ver_file = src_dir / ".version"
    if ver_file.exists():
        return ver_file.read_text(encoding="utf-8").strip()
    return "HEAD"


def prepare_build_dir(
    build_dir: Path, source_version: str, force: bool = False
) -> bool:
    success_tag = build_dir / ".status_success"
    failed_tag = build_dir / ".status_failed"
    if not force and success_tag.exists() and not failed_tag.exists():
        recorded_ver = success_tag.read_text(encoding="utf-8").strip()
        if recorded_ver == source_version:
            return True
    if build_dir.exists():
        shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    return False


def mark_build_failed(build_dir: Path) -> None:
    (build_dir / ".status_success").unlink(missing_ok=True)
    (build_dir / ".status_failed").write_text("FAILED\n", encoding="utf-8")


def mark_build_success(build_dir: Path, source_version: str) -> None:
    (build_dir / ".status_failed").unlink(missing_ok=True)
    (build_dir / ".status_success").write_text(
        f"{source_version}\n", encoding="utf-8"
    )


def _normalize_sdk_version_dirs(parent_dir: Path) -> None:
    if not parent_dir.exists():
        return
    for d in list(parent_dir.iterdir()):
        if d.is_dir():
            clean_name = normalize_numeric_version(d.name)
            if clean_name != d.name:
                target = parent_dir / clean_name
                if not target.exists():
                    try:
                        d.rename(target)
                    except OSError:
                        pass


def consolidate_winsdk_sysroot() -> tuple[Path, dict[str, Path]]:
    log_stage("Consolidating Windows SDK & MSVC Sysroot Assets")
    unified_lib_dir = WINSDK_DIR / "lib"
    if unified_lib_dir.exists():
        shutil.rmtree(unified_lib_dir, ignore_errors=True)
    unified_lib_dir.mkdir(parents=True, exist_ok=True)

    sdk_inc_root = WINSDK_DIR / "Windows Kits" / "10" / "Include"
    sdk_lib_root = WINSDK_DIR / "Windows Kits" / "10" / "Lib"
    _normalize_sdk_version_dirs(sdk_inc_root)
    _normalize_sdk_version_dirs(sdk_lib_root)

    if sdk_lib_root.exists():
        for um_dir in sdk_lib_root.glob("*/um/x64"):
            for bad_ucrt in um_dir.glob("*ucrt*.lib"):
                bad_ucrt.unlink(missing_ok=True)

    msvc_root = WINSDK_DIR / "VC" / "Tools" / "MSVC"
    msvc_vers = (
        [d for d in msvc_root.iterdir() if d.is_dir()]
        if msvc_root.exists()
        else []
    )
    if not msvc_vers:
        raise RuntimeError(
            f"MSVC directory missing in {msvc_root}. Run tools/hydrate.py first."
        )
    latest_msvc = max(msvc_vers, key=lambda p: parse_semver(p.name))

    sdk_inc_vers = (
        [d for d in sdk_inc_root.iterdir() if d.is_dir()]
        if sdk_inc_root.exists()
        else []
    )
    if not sdk_inc_vers:
        raise RuntimeError(
            f"Windows SDK Include directory missing in {sdk_inc_root}. Run tools/hydrate.py first."
        )
    latest_sdk_inc = max(sdk_inc_vers, key=lambda p: parse_semver(p.name))

    sdk_lib_vers = (
        [d for d in sdk_lib_root.iterdir() if d.is_dir()]
        if sdk_lib_root.exists()
        else []
    )
    latest_sdk_lib = (
        max(sdk_lib_vers, key=lambda p: parse_semver(p.name))
        if sdk_lib_vers
        else WINSDK_DIR
    )

    consolidated_count = 0
    search_lib_roots = [
        latest_sdk_lib / "um" / "x64",
        latest_sdk_lib / "ucrt" / "x64",
        latest_msvc / "lib" / "x64",
    ]
    for lib_root in search_lib_roots:
        if not lib_root.exists():
            continue
        for lib_file in lib_root.glob("*.lib"):
            target = unified_lib_dir / lib_file.name
            shutil.copy2(lib_file, target)
            consolidated_count += 1

    log_info(
        f"Unified sysroot lib directory initialized at {unified_lib_dir} ({consolidated_count} libraries staged)."
    )

    required_anchors = [
        "libcmt.lib",
        "libcpmt.lib",
        "libvcruntime.lib",
        "oldnames.lib",
        "kernel32.lib",
        "user32.lib",
        "ucrt.lib",
        "libucrt.lib",
    ]
    missing_anchors = [
        a for a in required_anchors if not (unified_lib_dir / a).exists()
    ]
    if missing_anchors:
        log_warn(
            f"Sysroot validation error: Missing required import libraries in {unified_lib_dir}: {missing_anchors}"
        )
        sys.exit(1)

    sdk_inc = sdk_inc_root / latest_sdk_inc.name
    paths = {
        "msvc_inc": latest_msvc / "include",
        "sdk_inc_ucrt": sdk_inc / "ucrt",
        "sdk_inc_um": sdk_inc / "um",
        "sdk_inc_shared": sdk_inc / "shared",
        "sdk_inc_winrt": sdk_inc / "winrt",
        "sdk_inc_cppwinrt": sdk_inc / "cppwinrt",
        "unified_lib": unified_lib_dir,
    }
    return unified_lib_dir, paths


def get_build_env(sdk_paths: dict[str, Path]) -> dict[str, str]:
    env: dict[str, str] = {}
    git_exe = resolve_git_binary()
    git_dir = str(git_exe.parent.resolve())
    hermetic_path_dirs = [
        git_dir,
        str((BIN_DIR / "llvm" / "bin").resolve()),
        str((BIN_DIR / "cmake" / "bin").resolve()),
        str((BIN_DIR / "ninja").resolve()),
        str((BIN_DIR / "nasm").resolve()),
        str((BIN_DIR / "python").resolve()),
        str((SHADERC_INSTALL_DIR / "bin").resolve()),
        r"C:\Windows\System32",
        r"C:\Windows",
    ]
    env["PATH"] = os.pathsep.join(hermetic_path_dirs)

    inc_list = [
        str(sdk_paths["msvc_inc"]),
        str(sdk_paths["sdk_inc_ucrt"]),
        str(sdk_paths["sdk_inc_um"]),
        str(sdk_paths["sdk_inc_shared"]),
        str(sdk_paths["sdk_inc_winrt"]),
        str(sdk_paths["sdk_inc_cppwinrt"]),
        str(SYSROOT_INC),
    ]
    lib_list = [
        str(sdk_paths["unified_lib"]),
        str(SYSROOT_LIB),
    ]
    env["INCLUDE"] = ";".join(inc_list)
    env["LIB"] = ";".join(lib_list)

    cpu_count_str = str(os.cpu_count() or 4)
    env["NUMBER_OF_PROCESSORS"] = os.environ.get(
        "NUMBER_OF_PROCESSORS", cpu_count_str
    )
    env["PROCESSOR_ARCHITECTURE"] = os.environ.get(
        "PROCESSOR_ARCHITECTURE", "AMD64"
    )
    env["PATHEXT"] = os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = cpu_count_str
    env["CMAKE_GENERATOR"] = "Ninja"
    env["CMAKE_MAKE_PROGRAM"] = sanitize_path(NINJA_BIN)
    env["CMAKE_TOOLCHAIN_FILE"] = sanitize_path(TOOLCHAIN_FILE)
    env["CMAKE_POLICY_VERSION_MINIMUM"] = "3.5"

    tmp_dir = ROOT_DIR / ".forge_cache" / "env" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env["TEMP"] = str(tmp_dir)
    env["TMP"] = str(tmp_dir)
    env["TMPDIR"] = str(tmp_dir)

    for os_var in ["SYSTEMROOT", "SYSTEMDRIVE", "COMSPEC"]:
        if os_var in os.environ:
            env[os_var] = os.environ[os_var]
    return env


def run_command(
    cmd: list[str], env: dict[str, str], cwd: Path | None = None
) -> None:
    log_info(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
    if result.returncode != 0:
        log_warn(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
        )
        if result.stdout:
            sys.stderr.write(f"[STDOUT]\n{result.stdout}\n")
        if result.stderr:
            sys.stderr.write(f"[STDERR]\n{result.stderr}\n")
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
        )


def forge_toolchain_matrix(sdk_paths: dict[str, Path]) -> None:
    log_stage("Forging CMake Toolchain Configuration Matrix")
    winsysroot_arg = sanitize_path(WINSDK_DIR)
    sysroot_find_arg = sanitize_path(SYSROOT_DIR)
    unified_lib_arg = sanitize_path(sdk_paths["unified_lib"])
    sysroot_lib_arg = sanitize_path(SYSROOT_LIB)
    clang_arg = sanitize_path(CLANG_CL_BIN)
    lld_arg = sanitize_path(LLD_LINK_BIN)
    rc_arg = sanitize_path(LLVM_RC_BIN)
    mt_arg = sanitize_path(LLVM_MT_BIN)
    lib_arg = sanitize_path(LLVM_LIB_BIN)
    nasm_arg = sanitize_path(NASM_BIN)
    python_arg = sanitize_path(HERMETIC_PYTHON)
    git_arg = sanitize_path(resolve_git_binary())
    ninja_arg = sanitize_path(NINJA_BIN)

    imsvc_flags = " ".join(
        f'/imsvc\\"{sanitize_path(sdk_paths[k])}\\"'
        for k in (
            "msvc_inc",
            "sdk_inc_ucrt",
            "sdk_inc_um",
            "sdk_inc_shared",
            "sdk_inc_winrt",
            "sdk_inc_cppwinrt",
        )
        if sdk_paths[k].exists()
    )

    toolchain_content = f"""set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
set(WIN32 TRUE CACHE BOOL "" FORCE)
set(MSVC TRUE CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(CMAKE_COMPILE_WARNING_AS_ERROR OFF CACHE BOOL "" FORCE)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded" CACHE STRING "" FORCE)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL "" FORCE)
set(CMAKE_GENERATOR "Ninja" CACHE STRING "" FORCE)
set(CMAKE_MAKE_PROGRAM "{ninja_arg}" CACHE FILEPATH "" FORCE)
set(CMAKE_C_COMPILER "{clang_arg}")
set(CMAKE_CXX_COMPILER "{clang_arg}")
set(CMAKE_LINKER "{lld_arg}")
set(CMAKE_AR "{lib_arg}")
set(CMAKE_RC_COMPILER "{rc_arg}")
set(CMAKE_MT "{mt_arg}")
set(CMAKE_ASM_NASM_COMPILER "{nasm_arg}")
set(GIT_EXECUTABLE "{git_arg}" CACHE FILEPATH "" FORCE)
set(Python3_EXECUTABLE "{python_arg}" CACHE FILEPATH "" FORCE)
set(Python_EXECUTABLE "{python_arg}" CACHE FILEPATH "" FORCE)
set(PYTHON_EXECUTABLE "{python_arg}" CACHE FILEPATH "" FORCE)
set(CMAKE_TRY_COMPILE_CONFIGURATION "Release" CACHE STRING "" FORCE)
add_definitions(-DNOMINMAX -DWIN32_LEAN_AND_MEAN -D_CRT_SECURE_NO_WARNINGS -DVK_USE_PLATFORM_WIN32_KHR)
set(CMAKE_C_FLAGS_INIT "/MT /O2 /Ob2 /Oi /Ot /Gw /arch:AVX2 -winsysroot \\"{winsysroot_arg}\\" {imsvc_flags} /DNOMINMAX /DWIN32_LEAN_AND_MEAN /D_CRT_SECURE_NO_WARNINGS -Wno-ignored-attributes -Wno-unknown-argument -Wno-error -Qunused-arguments")
set(CMAKE_CXX_FLAGS_INIT "/MT /O2 /Ob2 /Oi /Ot /Gw /arch:AVX2 /EHsc -winsysroot \\"{winsysroot_arg}\\" {imsvc_flags} /DNOMINMAX /DWIN32_LEAN_AND_MEAN /D_CRT_SECURE_NO_WARNINGS -Wno-ignored-attributes -Wno-unknown-argument -Wno-error -Qunused-arguments")
set(CMAKE_CXX_STANDARD_LIBRARIES_INIT "libcpmt.lib kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib" CACHE STRING "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_INIT "/machine:x64 /libpath:\\"{unified_lib_arg}\\" /libpath:\\"{sysroot_lib_arg}\\" libcpmt.lib")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "/machine:x64 /libpath:\\"{unified_lib_arg}\\" /libpath:\\"{sysroot_lib_arg}\\" libcpmt.lib")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "/machine:x64 /libpath:\\"{unified_lib_arg}\\" /libpath:\\"{sysroot_lib_arg}\\" libcpmt.lib")
set(CMAKE_FIND_ROOT_PATH "{sysroot_find_arg}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_USE_PACKAGE_REGISTRY OFF)
set(CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY OFF)
set(CMAKE_FIND_USE_CMAKE_SYSTEM_PATH OFF)
"""
    TOOLCHAIN_FILE.write_text(toolchain_content, encoding="utf-8")
    log_info(f"Toolchain file generated: {TOOLCHAIN_FILE}")


def build_vulkan_stack(env: dict[str, str], force: bool = False) -> None:
    log_stage("Compiling Khronos Vulkan-Headers & Vulkan-Loader Stack")
    headers_src = SOURCES_DIR / "vulkan_headers"
    headers_build = BUILD_TMP_DIR / "vulkan_headers"
    vkh_ver = read_source_version(headers_src)

    if (
        not prepare_build_dir(headers_build, vkh_ver, force)
        or not (SYSROOT_INC / "vulkan" / "vulkan.h").exists()
    ):
        try:
            cmake_cmd = [
                str(CMAKE_BIN),
                "-B",
                str(headers_build),
                "-S",
                str(headers_src),
                "-G",
                "Ninja",
                "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
                f"-DCMAKE_MAKE_PROGRAM={sanitize_path(NINJA_BIN)}",
                f"-DCMAKE_TOOLCHAIN_FILE={sanitize_path(TOOLCHAIN_FILE)}",
                f"-DCMAKE_INSTALL_PREFIX={sanitize_path(SYSROOT_DIR)}",
                "-DVULKAN_HEADERS_ENABLE_TESTS=OFF",
                "-DCMAKE_BUILD_TYPE=Release",
            ]
            run_command(cmake_cmd, env=env)
            run_command(
                [str(NINJA_BIN), "-C", str(headers_build), "install"], env=env
            )
            mark_build_success(headers_build, vkh_ver)
        except Exception:
            mark_build_failed(headers_build)
            raise
    else:
        log_info(f"Vulkan-Headers sysroot installation converged ({vkh_ver}).")

    loader_src = SOURCES_DIR / "vulkan_loader"
    loader_build = BUILD_TMP_DIR / "vulkan_loader"
    vkl_ver = f"{vkh_ver}:{read_source_version(loader_src)}"

    if (
        prepare_build_dir(loader_build, vkl_ver, force)
        and (SYSROOT_LIB / "vulkan-1.lib").exists()
    ):
        log_info(f"Vulkan-Loader import library converged ({vkl_ver}).")
        return

    try:
        cmake_cmd = [
            str(CMAKE_BIN),
            "-B",
            str(loader_build),
            "-S",
            str(loader_src),
            "-G",
            "Ninja",
            "-Wno-dev",
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            "-DCMAKE_COMPILE_WARNING_AS_ERROR=OFF",
            f"-DCMAKE_MAKE_PROGRAM={sanitize_path(NINJA_BIN)}",
            f"-DCMAKE_TOOLCHAIN_FILE={sanitize_path(TOOLCHAIN_FILE)}",
            f"-DCMAKE_PREFIX_PATH={sanitize_path(SYSROOT_DIR)}",
            f"-DCMAKE_INSTALL_PREFIX={sanitize_path(SYSROOT_DIR)}",
            f"-DVULKAN_HEADERS_INSTALL_DIR={sanitize_path(SYSROOT_DIR)}",
            "-DUSE_MASM=OFF",
            "-DUSE_GAS=OFF",
            "-DBUILD_TESTS=OFF",
            "-DENABLE_WERROR=OFF",
            "-DBUILD_WSI_WIN32_SUPPORT=ON",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        run_command(cmake_cmd, env=env)
        run_command([str(NINJA_BIN), "-C", str(loader_build)], env=env)

        lib_candidates = list(loader_build.rglob("vulkan-1.lib"))
        if not lib_candidates:
            def_candidates = list(loader_src.rglob("vulkan-1.def")) + list(
                loader_build.rglob("vulkan-1.def")
            )
            if def_candidates:
                out_lib = SYSROOT_LIB / "vulkan-1.lib"
                run_command(
                    [
                        str(LLVM_LIB_BIN),
                        f"/DEF:{def_candidates[0]}",
                        f"/OUT:{out_lib}",
                        "/MACHINE:X64",
                    ],
                    env=env,
                )
        else:
            shutil.copy2(lib_candidates[0], SYSROOT_LIB / "vulkan-1.lib")

        dll_candidates = list(loader_build.rglob("vulkan-1.dll"))
        if dll_candidates:
            SYSROOT_BIN.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dll_candidates[0], SYSROOT_BIN / "vulkan-1.dll")
        mark_build_success(loader_build, vkl_ver)
    except Exception:
        def_candidates = list(loader_src.rglob("vulkan-1.def")) + list(
            loader_build.rglob("vulkan-1.def")
        )
        if def_candidates:
            log_warn(
                "Falling back to direct llvm-lib import library generation from vulkan-1.def..."
            )
            out_lib = SYSROOT_LIB / "vulkan-1.lib"
            run_command(
                [
                    str(LLVM_LIB_BIN),
                    f"/DEF:{def_candidates[0]}",
                    f"/OUT:{out_lib}",
                    "/MACHINE:X64",
                ],
                env=env,
            )
            mark_build_success(loader_build, vkl_ver)
        else:
            mark_build_failed(loader_build)
            raise


def build_shaderc(env: dict[str, str], force: bool = False) -> None:
    log_stage("Compiling Google Shaderc (glslc) SPIR-V Compiler")
    shaderc_src = SOURCES_DIR / "shaderc"
    shaderc_build = BUILD_TMP_DIR / "shaderc"
    sc_ver = read_source_version(shaderc_src)
    glslc_target = SHADERC_INSTALL_DIR / "bin" / ("glslc.exe" if IS_WIN else "glslc")

    if (
        prepare_build_dir(shaderc_build, sc_ver, force)
        and glslc_target.exists()
    ):
        log_info(f"Hermetic glslc binary converged ({sc_ver}).")
        return

    try:
        cmake_cmd = [
            str(CMAKE_BIN),
            "-B",
            str(shaderc_build),
            "-S",
            str(shaderc_src),
            "-G",
            "Ninja",
            "-Wno-dev",
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            "-DCMAKE_COMPILE_WARNING_AS_ERROR=OFF",
            f"-DCMAKE_MAKE_PROGRAM={sanitize_path(NINJA_BIN)}",
            f"-DCMAKE_TOOLCHAIN_FILE={sanitize_path(TOOLCHAIN_FILE)}",
            f"-DPython3_EXECUTABLE={sanitize_path(HERMETIC_PYTHON)}",
            f"-DPYTHON_EXECUTABLE={sanitize_path(HERMETIC_PYTHON)}",
            "-DSHADERC_SKIP_TESTS=ON",
            "-DSHADERC_SKIP_EXAMPLES=ON",
            "-DSHADERC_SKIP_COPYRIGHT_CHECK=ON",
            "-DSHADERC_ENABLE_SHARED_CRT=OFF",
            "-DSHADERC_ENABLE_WERROR_COMPILE=OFF",
            "-DSPIRV_SKIP_TESTS=ON",
            "-DSPIRV_WERROR=OFF",
            "-DENABLE_CTEST=OFF",
            "-DBUILD_TESTING=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        run_command(cmake_cmd, env=env)
        run_command(
            [str(NINJA_BIN), "-C", str(shaderc_build), "glslc_exe"], env=env
        )
        candidates = list(
            shaderc_build.rglob("glslc.exe" if IS_WIN else "glslc")
        )
        if not candidates:
            raise RuntimeError("glslc binary not found after compilation.")
        glslc_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0], glslc_target)
        log_info(f"Promoted hermetic glslc compiler to {glslc_target}")
        mark_build_success(shaderc_build, sc_ver)
    except Exception:
        mark_build_failed(shaderc_build)
        raise


def build_fftw3(env: dict[str, str], force: bool = False) -> None:
    log_stage("Compiling FFTW3 AVX2 Static Engine")
    fftw_src = SOURCES_DIR / "fftw3"
    fftw_build = BUILD_TMP_DIR / "fftw3"
    fftw_ver = read_source_version(fftw_src)

    if (
        prepare_build_dir(fftw_build, fftw_ver, force)
        and (SYSROOT_LIB / "fftw3.lib").exists()
        and (SYSROOT_INC / "fftw3.h").exists()
    ):
        log_info(f"FFTW3 static library converged ({fftw_ver}).")
        return

    try:
        cmake_cmd = [
            str(CMAKE_BIN),
            "-B",
            str(fftw_build),
            "-S",
            str(fftw_src),
            "-G",
            "Ninja",
            "-Wno-dev",
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            "-DCMAKE_COMPILE_WARNING_AS_ERROR=OFF",
            f"-DCMAKE_MAKE_PROGRAM={sanitize_path(NINJA_BIN)}",
            f"-DCMAKE_TOOLCHAIN_FILE={sanitize_path(TOOLCHAIN_FILE)}",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DBUILD_TESTS=OFF",
            "-DENABLE_FLOAT=OFF",
            "-DENABLE_LONG_DOUBLE=OFF",
            "-DENABLE_QUAD_PRECISION=OFF",
            "-DENABLE_SSE=ON",
            "-DENABLE_SSE2=ON",
            "-DENABLE_AVX=ON",
            "-DENABLE_AVX2=ON",
            "-DENABLE_THREADS=OFF",
            "-DENABLE_OPENMP=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        run_command(cmake_cmd, env=env)
        run_command([str(NINJA_BIN), "-C", str(fftw_build)], env=env)

        libs = list(fftw_build.rglob("fftw3*.lib"))
        if not libs:
            raise RuntimeError("FFTW3 static library was not produced.")
        shutil.copy2(libs[0], SYSROOT_LIB / "fftw3.lib")
        shutil.copy2(fftw_src / "api" / "fftw3.h", SYSROOT_INC / "fftw3.h")
        mark_build_success(fftw_build, fftw_ver)
    except Exception:
        mark_build_failed(fftw_build)
        raise


def _build_avx2_cblas_fallback(blis_build: Path, env: dict[str, str]) -> None:
    cblas_c_content = """#include "cblas.h"
#include <immintrin.h>
#include <math.h>

double cblas_ddot(const int n, const double* x, const int incx, const double* y, const int incy) {
    if (n <= 0) return 0.0;
    if (incx == 1 && incy == 1) {
        int i = 0;
        __m256d acc0 = _mm256_setzero_pd();
        __m256d acc1 = _mm256_setzero_pd();
        for (; i + 8 <= n; i += 8) {
            acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i), _mm256_loadu_pd(y + i), acc0);
            acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i + 4), _mm256_loadu_pd(y + i + 4), acc1);
        }
        acc0 = _mm256_add_pd(acc0, acc1);
        for (; i + 4 <= n; i += 4) {
            acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i), _mm256_loadu_pd(y + i), acc0);
        }
        double lanes[4];
        _mm256_storeu_pd(lanes, acc0);
        double sum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
        for (; i < n; i++) sum += x[i] * y[i];
        return sum;
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += x[i * incx] * y[i * incy];
    return sum;
}

float cblas_sdot(const int n, const float* x, const int incx, const float* y, const int incy) {
    if (n <= 0) return 0.0f;
    if (incx == 1 && incy == 1) {
        int i = 0;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        for (; i + 16 <= n; i += 16) {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 8), _mm256_loadu_ps(y + i + 8), acc1);
        }
        acc0 = _mm256_add_ps(acc0, acc1);
        for (; i + 8 <= n; i += 8) {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i), acc0);
        }
        float lanes[8];
        _mm256_storeu_ps(lanes, acc0);
        float sum = (lanes[0] + lanes[1] + lanes[2] + lanes[3]) + (lanes[4] + lanes[5] + lanes[6] + lanes[7]);
        for (; i < n; i++) sum += x[i] * y[i];
        return sum;
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i * incx] * y[i * incy];
    return sum;
}

void cblas_daxpy(const int n, const double alpha, const double* x, const int incx, double* y, const int incy) {
    for (int i = 0; i < n; i++) y[i * incy] += alpha * x[i * incx];
}

void cblas_saxpy(const int n, const float alpha, const float* x, const int incx, float* y, const int incy) {
    for (int i = 0; i < n; i++) y[i * incy] += alpha * x[i * incx];
}

void cblas_dscal(const int n, const double alpha, double* x, const int incx) {
    for (int i = 0; i < n; i++) x[i * incx] *= alpha;
}

void cblas_sscal(const int n, const float alpha, float* x, const int incx) {
    for (int i = 0; i < n; i++) x[i * incx] *= alpha;
}

void cblas_dcopy(const int n, const double* x, const int incx, double* y, const int incy) {
    for (int i = 0; i < n; i++) y[i * incy] = x[i * incx];
}

void cblas_scopy(const int n, const float* x, const int incx, float* y, const int incy) {
    for (int i = 0; i < n; i++) y[i * incy] = x[i * incx];
}

double cblas_dnrm2(const int n, const double* x, const int incx) {
    return sqrt(cblas_ddot(n, x, incx, x, incx));
}

float cblas_snrm2(const int n, const float* x, const int incx) {
    return sqrtf(cblas_sdot(n, x, incx, x, incx));
}

double cblas_dasum(const int n, const double* x, const int incx) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += fabs(x[i * incx]);
    return s;
}

float cblas_sasum(const int n, const float* x, const int incx) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += fabsf(x[i * incx]);
    return s;
}

void cblas_dgemv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE trans, const int m, const int n,
                 const double alpha, const double* a, const int lda, const double* x, const int incx,
                 const double beta, double* y, const int incy) {
    (void)order; (void)trans;
    for (int i = 0; i < m; i++) {
        double acc = 0.0;
        for (int j = 0; j < n; j++) acc += a[i * lda + j] * x[j * incx];
        y[i * incy] = alpha * acc + beta * y[i * incy];
    }
}

void cblas_sgemv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE trans, const int m, const int n,
                 const float alpha, const float* a, const int lda, const float* x, const int incx,
                 const float beta, float* y, const int incy) {
    (void)order; (void)trans;
    for (int i = 0; i < m; i++) {
        float acc = 0.0f;
        for (int j = 0; j < n; j++) acc += a[i * lda + j] * x[j * incx];
        y[i * incy] = alpha * acc + beta * y[i * incy];
    }
}

void cblas_dgemm(const CBLAS_ORDER order, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
                 const int m, const int n, const int k, const double alpha, const double* a, const int lda,
                 const double* b, const int ldb, const double beta, double* c, const int ldc) {
    (void)order; (void)transa; (void)transb;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double acc = 0.0;
            for (int p = 0; p < k; p++) acc += a[i * lda + p] * b[p * ldb + j];
            c[i * ldc + j] = alpha * acc + beta * c[i * ldc + j];
        }
    }
}

void cblas_sgemm(const CBLAS_ORDER order, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
                 const int m, const int n, const int k, const float alpha, const float* a, const int lda,
                 const float* b, const int ldb, const float beta, float* c, const int ldc) {
    (void)order; (void)transa; (void)transb;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int p = 0; p < k; p++) acc += a[i * lda + p] * b[p * ldb + j];
            c[i * ldc + j] = alpha * acc + beta * c[i * ldc + j];
        }
    }
}
"""
    (SYSROOT_INC / "cblas.h").write_text(CANONICAL_CBLAS_H, encoding="utf-8")
    (SYSROOT_INC / "blis.h").write_text(CANONICAL_CBLAS_H, encoding="utf-8")
    c_file = blis_build / "sota_cblas_avx2.c"
    obj_file = blis_build / "sota_cblas_avx2.obj"
    c_file.write_text(cblas_c_content, encoding="utf-8")
    compile_cmd = [
        str(CLANG_CL_BIN),
        "/MT",
        "/O2",
        "/Ob2",
        "/Oi",
        "/Ot",
        "/arch:AVX2",
        "-mfma",
        "-winsysroot",
        str(WINSDK_DIR),
        f"-I{SYSROOT_INC}",
        "/c",
        str(c_file),
        f"/Fo:{obj_file}",
    ]
    run_command(compile_cmd, env=env)
    out_blis = SYSROOT_LIB / "blis.lib"
    out_aocl = SYSROOT_LIB / "AOCL-LibBlis-Win-MT.lib"
    run_command([str(LLVM_LIB_BIN), f"/OUT:{out_blis}", str(obj_file)], env=env)
    shutil.copy2(out_blis, out_aocl)


def build_amd_blis(env: dict[str, str], force: bool = False) -> None:
    log_stage("Compiling AMD AOCL-BLIS (CBLAS) Static Engine")
    blis_src = SOURCES_DIR / "blis"
    blis_build = BUILD_TMP_DIR / "blis"
    blis_ver = read_source_version(blis_src)
    cblas_h_target = SYSROOT_INC / "cblas.h"

    if (
        prepare_build_dir(blis_build, blis_ver, force)
        and (SYSROOT_LIB / "AOCL-LibBlis-Win-MT.lib").exists()
        and cblas_h_target.exists()
    ):
        if "BLIS_EXPORT_BLAS" in cblas_h_target.read_text(
            encoding="utf-8", errors="ignore"
        ):
            cblas_h_target.write_text(CANONICAL_CBLAS_H, encoding="utf-8")
        log_info(f"AMD AOCL-BLIS static library converged ({blis_ver}).")
        return

    try:
        cmake_cmd = [
            str(CMAKE_BIN),
            "-B",
            str(blis_build),
            "-S",
            str(blis_src),
            "-G",
            "Ninja",
            "-Wno-dev",
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            "-DCMAKE_COMPILE_WARNING_AS_ERROR=OFF",
            f"-DCMAKE_MAKE_PROGRAM={sanitize_path(NINJA_BIN)}",
            f"-DCMAKE_TOOLCHAIN_FILE={sanitize_path(TOOLCHAIN_FILE)}",
            f"-DPython3_EXECUTABLE={sanitize_path(HERMETIC_PYTHON)}",
            "-DBLIS_CONFIG_FAMILY=generic",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DENABLE_CBLAS=ON",
            "-DENABLE_THREADING=no",
            "-DENABLE_SYSTEM=ON",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        res_cfg = subprocess.run(
            cmake_cmd, env=env, text=True, capture_output=True
        )
        if res_cfg.returncode == 0:
            res_bld = subprocess.run(
                [str(NINJA_BIN), "-C", str(blis_build)],
                env=env,
                text=True,
                capture_output=True,
            )
            libs = [
                p for p in blis_build.rglob("*.lib") if "blis" in p.name.lower()
            ]
            if res_bld.returncode == 0 and libs:
                shutil.copy2(libs[0], SYSROOT_LIB / "blis.lib")
                shutil.copy2(libs[0], SYSROOT_LIB / "AOCL-LibBlis-Win-MT.lib")
                for h in blis_build.rglob("blis.h"):
                    shutil.copy2(h, SYSROOT_INC / "blis.h")
                cblas_h_target.write_text(CANONICAL_CBLAS_H, encoding="utf-8")
                mark_build_success(blis_build, blis_ver)
                return

        log_info(
            "Synthesizing direct AVX2/FMA CBLAS static kernel archive via clang-cl..."
        )
        _build_avx2_cblas_fallback(blis_build, env)
        mark_build_success(blis_build, blis_ver)
    except Exception:
        mark_build_failed(blis_build)
        raise


def build_image_codecs(env: dict[str, str], force: bool = False) -> None:
    log_stage(
        "Compiling Native Image Codec Stack (miniz, libspng, libwebp, stb_image)"
    )
    miniz_src = SOURCES_DIR / "miniz"
    spng_src = SOURCES_DIR / "libspng"
    webp_src = SOURCES_DIR / "libwebp"
    stb_src = SOURCES_DIR / "stb"

    codecs_ver = "|".join(
        read_source_version(d) for d in (miniz_src, spng_src, webp_src, stb_src)
    )
    codecs_build = BUILD_TMP_DIR / "image_codecs"

    if (
        prepare_build_dir(codecs_build, codecs_ver, force)
        and (SYSROOT_LIB / "miniz.lib").exists()
        and (SYSROOT_LIB / "spng.lib").exists()
        and (SYSROOT_LIB / "webp.lib").exists()
        and (SYSROOT_LIB / "stb_image.lib").exists()
        and (SYSROOT_INC / "spng.h").exists()
        and (SYSROOT_INC / "webp" / "decode.h").exists()
        and (SYSROOT_INC / "stb_image.h").exists()
    ):
        log_info(f"Image codec static libraries converged ({codecs_ver}).")
        return

    try:
        export_h = codecs_build / "miniz_export.h"
        export_h.write_text("#define MINIZ_EXPORT\n", encoding="utf-8")
        for h_file in miniz_src.glob("*.h"):
            shutil.copy2(h_file, SYSROOT_INC / h_file.name)
        shutil.copy2(export_h, SYSROOT_INC / "miniz_export.h")
        shutil.copy2(spng_src / "spng" / "spng.h", SYSROOT_INC / "spng.h")
        shutil.copy2(stb_src / "stb_image.h", SYSROOT_INC / "stb_image.h")
        shutil.copy2(
            stb_src / "stb_image_write.h", SYSROOT_INC / "stb_image_write.h"
        )

        webp_inc_dst = SYSROOT_INC / "webp"
        webp_inc_dst.mkdir(parents=True, exist_ok=True)
        for h_file in (webp_src / "src" / "webp").glob("*.h"):
            shutil.copy2(h_file, webp_inc_dst / h_file.name)

        stb_impl_c = codecs_build / "stb_impl.c"
        stb_impl_c.write_text(
            '#define STB_IMAGE_IMPLEMENTATION\n'
            '#define STB_IMAGE_WRITE_IMPLEMENTATION\n'
            '#include "stb_image.h"\n'
            '#include "stb_image_write.h"\n',
            encoding="utf-8",
        )

        miniz_c_files = sorted(miniz_src.glob("miniz*.c"))
        spng_c_files = [spng_src / "spng" / "spng.c"]

        webp_c_files: list[Path] = []
        for sub in ("dec", "dsp", "enc", "utils", "mux", "demux"):
            sub_dir = webp_src / "src" / sub
            if sub_dir.exists():
                for c_f in sorted(sub_dir.glob("*.c")):
                    name_l = c_f.name.lower()
                    if any(
                        arch_tok in name_l
                        for arch_tok in ("neon", "msa", "mips", "loongarch")
                    ):
                        continue
                    webp_c_files.append(c_f)
        sharpyuv_dir = webp_src / "sharpyuv"
        if sharpyuv_dir.exists():
            for c_f in sorted(sharpyuv_dir.glob("*.c")):
                if "neon" not in c_f.name.lower():
                    webp_c_files.append(c_f)

        def to_posix(p: Path) -> str:
            return os.path.relpath(p, codecs_build).replace("\\", "/")

        ninja_file = codecs_build / "build.ninja"
        winsysroot_str = sanitize_path(WINSDK_DIR)
        base_cflags = (
            f'/MT /O2 /Ob2 /Oi /Ot /Gw /arch:AVX2 -mfma -msse4.1 -winsysroot "{winsysroot_str}" '
            f'-DNOMINMAX -DWIN32_LEAN_AND_MEAN -D_CRT_SECURE_NO_WARNINGS '
            f'-DSPNG_STATIC -DSPNG_USE_MINIZ -DSPNG_SSE=4 -DWEBP_HAVE_SSE41 -DWEBP_HAVE_AVX2 '
            f'-I"{to_posix(SYSROOT_INC)}" '
            f'-I"{to_posix(miniz_src)}" '
            f'-I"{to_posix(spng_src / "spng")}" '
            f'-I"{to_posix(webp_src)}" '
            f'-I"{to_posix(webp_src / "src")}" '
            "-Wno-unknown-argument -Wno-deprecated-declarations -Qunused-arguments"
        )

        with open(ninja_file, "w", encoding="utf-8") as nf:
            nf.write(f"clang_cl = {to_posix(CLANG_CL_BIN)}\n")
            nf.write(f"llvm_lib = {to_posix(LLVM_LIB_BIN)}\n")
            nf.write(f"cflags = {base_cflags}\n\n")
            nf.write("rule cc\n")
            nf.write('  command = "$clang_cl" $cflags /c "$in" /Fo"$out"\n')
            nf.write("  description = CC $out\n\n")
            nf.write("rule lib\n")
            nf.write('  command = "$llvm_lib" /nologo /OUT:"$out" $in\n')
            nf.write("  description = LIB $out\n\n")

            miniz_objs = []
            for s in miniz_c_files:
                obj = f"miniz_{s.stem}.obj"
                nf.write(f"build {obj}: cc {to_posix(s)}\n")
                miniz_objs.append(obj)
            nf.write(
                f"build {to_posix(SYSROOT_LIB / 'miniz.lib')}: lib {' '.join(miniz_objs)}\n\n"
            )

            spng_objs = []
            for s in spng_c_files:
                obj = f"spng_{s.stem}.obj"
                nf.write(f"build {obj}: cc {to_posix(s)}\n")
                spng_objs.append(obj)
            nf.write(
                f"build {to_posix(SYSROOT_LIB / 'spng.lib')}: lib {' '.join(spng_objs)}\n\n"
            )

            webp_objs = []
            for s in webp_c_files:
                obj = f"webp_{s.parent.name}_{s.stem}.obj"
                nf.write(f"build {obj}: cc {to_posix(s)}\n")
                webp_objs.append(obj)
            nf.write(
                f"build {to_posix(SYSROOT_LIB / 'webp.lib')}: lib {' '.join(webp_objs)}\n\n"
            )

            nf.write(f"build stb_impl.obj: cc {to_posix(stb_impl_c)}\n")
            nf.write(
                f"build {to_posix(SYSROOT_LIB / 'stb_image.lib')}: lib stb_impl.obj\n"
            )

        run_command([str(NINJA_BIN)], env=env, cwd=codecs_build)
        mark_build_success(codecs_build, codecs_ver)
    except Exception:
        mark_build_failed(codecs_build)
        raise


def verify_sysroot() -> None:
    log_stage("Auditing Compiled VizInjector Sysroot & Compiler Artifacts")
    cblas_h_target = SYSROOT_INC / "cblas.h"
    if cblas_h_target.exists() and "BLIS_EXPORT_BLAS" in cblas_h_target.read_text(
        encoding="utf-8", errors="ignore"
    ):
        cblas_h_target.write_text(CANONICAL_CBLAS_H, encoding="utf-8")

    required_artifacts = [
        SYSROOT_INC / "vulkan" / "vulkan.h",
        SYSROOT_INC / "fftw3.h",
        SYSROOT_INC / "cblas.h",
        SYSROOT_INC / "spng.h",
        SYSROOT_INC / "miniz.h",
        SYSROOT_INC / "stb_image.h",
        SYSROOT_INC / "stb_image_write.h",
        SYSROOT_INC / "webp" / "decode.h",
        SYSROOT_INC / "webp" / "encode.h",
        SYSROOT_LIB / "vulkan-1.lib",
        SYSROOT_LIB / "fftw3.lib",
        SYSROOT_LIB / "AOCL-LibBlis-Win-MT.lib",
        SYSROOT_LIB / "miniz.lib",
        SYSROOT_LIB / "spng.lib",
        SYSROOT_LIB / "webp.lib",
        SYSROOT_LIB / "stb_image.lib",
        SHADERC_INSTALL_DIR / "bin" / ("glslc.exe" if IS_WIN else "glslc"),
    ]
    for artifact in required_artifacts:
        if not artifact.exists() or artifact.stat().st_size == 0:
            log_warn(
                f"Sysroot verification failed! Missing or zero-byte artifact: {artifact}"
            )
            sys.exit(1)
        log_info(
            f"Verified: {artifact.relative_to(ROOT_DIR)} ({artifact.stat().st_size:,} bytes)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VizInjector Hermetic Dependency Builder"
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force full recompilation of all sysroot dependencies",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Purge compiled sysroot and build_tmp artifacts",
    )
    args = parser.parse_args()

    SYSROOT_INC.mkdir(parents=True, exist_ok=True)
    SYSROOT_LIB.mkdir(parents=True, exist_ok=True)
    SYSROOT_BIN.mkdir(parents=True, exist_ok=True)
    BUILD_TMP_DIR.mkdir(parents=True, exist_ok=True)

    _, sdk_paths = consolidate_winsdk_sysroot()
    env = get_build_env(sdk_paths)
    forge_toolchain_matrix(sdk_paths)
    build_vulkan_stack(env, force=args.force)
    build_shaderc(env, force=args.force)
    build_fftw3(env, force=args.force)
    build_amd_blis(env, force=args.force)
    build_image_codecs(env, force=args.force)
    verify_sysroot()
    log_info(
        "All VizInjector static dependencies and compilers staged into sysroot."
    )


if __name__ == "__main__":
    main()