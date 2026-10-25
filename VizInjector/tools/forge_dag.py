#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
BIN_DIR = ROOT_DIR / "bin"
IS_WIN = sys.platform == "win32"
HERMETIC_PYTHON_EXE = BIN_DIR / "python" / ("python.exe" if IS_WIN else "bin/python")

if (
    HERMETIC_PYTHON_EXE.exists()
    and Path(sys.executable).resolve() != HERMETIC_PYTHON_EXE.resolve()
):
    respawn_cmd = [
        str(HERMETIC_PYTHON_EXE.resolve()),
        "-I",
        str(Path(__file__).resolve()),
    ] + sys.argv[1:]
    res = subprocess.run(respawn_cmd, check=False)
    sys.exit(res.returncode)

SRC_DIR = ROOT_DIR / "src"
PYTHON_PKG_DIR = ROOT_DIR / "python"
SHADERS_DIR = SRC_DIR / "shaders"
VULKAN_SRC_DIR = SRC_DIR / "vulkan"
CACHE_DIR = ROOT_DIR / ".forge_cache"
SPV_CACHE_DIR = CACHE_DIR / "spv"
DAG_DIR = CACHE_DIR / "dag-build"
DIST_DIR = ROOT_DIR / "dist"
DIST_BIN = DIST_DIR / "bin"
SYSROOT_DIR = BIN_DIR / "sysroot"
SYSROOT_INC = SYSROOT_DIR / "include"
SYSROOT_LIB = SYSROOT_DIR / "lib"
SYSROOT_BIN = SYSROOT_DIR / "bin"
WINSDK_DIR = BIN_DIR / "winsdk"
FFMPEG_DIR = BIN_DIR / "ffmpeg"

CLANG_EXE = BIN_DIR / "llvm" / "bin" / ("clang.exe" if IS_WIN else "clang")
CLANG_CL_EXE = BIN_DIR / "llvm" / "bin" / ("clang-cl.exe" if IS_WIN else "clang-cl")
LLD_LINK_EXE = BIN_DIR / "llvm" / "bin" / ("lld-link.exe" if IS_WIN else "lld-link")
NINJA_EXE = BIN_DIR / "ninja" / ("ninja.exe" if IS_WIN else "ninja")
GLSLC_EXE = BIN_DIR / "shaderc" / "bin" / ("glslc.exe" if IS_WIN else "glslc")

SHADER_SPECS = [
    ("ingest_linearize.comp", "g_spv_ingest_linearize"),
    ("kaiser_resampler.comp" if (SHADERS_DIR / "kaiser_resampler.comp").exists() else "kaiser_resample.comp", "g_spv_kaiser_resample"),
    ("branch_pack.comp", "g_spv_branch_pack"),
    ("telemetry_reduce.comp", "g_spv_telemetry_reduce"),
]


def ensure_writable(path: Path) -> None:
    if path.exists():
        try:
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        except Exception:
            pass


def compile_shaders_to_header() -> Path:
    print("[*] Compiling GLSL 450 Compute Shaders to SPIR-V via hermetic glslc...")
    SPV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    header_lines = [
        "#ifndef VIZINJECTOR_SHADERS_SPV_H",
        "#define VIZINJECTOR_SHADERS_SPV_H",
        "#include <stdint.h>",
        "",
    ]
    for filename, symbol_name in SHADER_SPECS:
        src_path = SHADERS_DIR / filename
        spv_path = SPV_CACHE_DIR / f"{src_path.stem}.spv"
        cmd = [
            str(GLSLC_EXE),
            "-O",
            "--target-env=vulkan1.2",
            "-fshader-stage=compute",
            str(src_path),
            "-o",
            str(spv_path),
        ]
        subprocess.run(cmd, check=True)
        raw_bytes = spv_path.read_bytes()
        if len(raw_bytes) % 4 != 0:
            raise RuntimeError(f"Unaligned SPIR-V output for {filename}")
        words = struct.unpack(f"<{len(raw_bytes) // 4}I", raw_bytes)
        header_lines.append(f"static const uint32_t {symbol_name}[] = {{")
        for i in range(0, len(words), 8):
            chunk = ", ".join(f"0x{w:08X}u" for w in words[i : i + 8])
            header_lines.append(f"    {chunk},")
        header_lines.append("};\n")

    header_lines.append("#endif\n")
    out_header = VULKAN_SRC_DIR / "shaders_spv.h"
    ensure_writable(out_header)
    out_header.write_text("\n".join(header_lines), encoding="utf-8")
    print(
        f"[+] Embedded SPIR-V compute shaders written to {out_header.relative_to(ROOT_DIR)}"
    )
    return out_header


def resolve_winsdk_libpaths(to_posix_fn) -> str:
    lib_dirs = []
    if (WINSDK_DIR / "lib").exists():
        lib_dirs.append(WINSDK_DIR / "lib")
    lib_dirs.extend(WINSDK_DIR.glob("VC/Tools/MSVC/*/lib/x64"))
    lib_dirs.extend(WINSDK_DIR.glob("Windows Kits/10/Lib/*/um/x64"))
    lib_dirs.extend(WINSDK_DIR.glob("Windows Kits/10/Lib/*/ucrt/x64"))
    return " ".join(f'/LIBPATH:"{to_posix_fn(d)}"' for d in lib_dirs)


def resolve_winsdk_imsvc_flags() -> str:
    inc_dirs = []
    inc_dirs.extend(WINSDK_DIR.glob("VC/Tools/MSVC/*/include"))
    for sub in ("ucrt", "um", "shared", "winrt"):
        inc_dirs.extend(WINSDK_DIR.glob(f"Windows Kits/10/Include/*/{sub}"))
    return " ".join(
        f'/imsvc"{str(d.resolve()).replace(chr(92), "/")}"' for d in inc_dirs
    )


def stage_python_bindings_to_dist() -> None:
    DIST_BIN.mkdir(parents=True, exist_ok=True)
    for stale_item in list(DIST_BIN.iterdir()):
        if stale_item.is_file() and stale_item.suffix.lower() in (
            ".html",
            ".lib",
            ".exp",
            ".pdb",
        ):
            ensure_writable(stale_item)
            stale_item.unlink(missing_ok=True)

    for py_name in ("schema.py", "bindings.py", "modeling_vae.py", "__init__.py"):
        src_py = PYTHON_PKG_DIR / py_name
        if src_py.is_file():
            dst_py = DIST_BIN / py_name
            ensure_writable(dst_py)
            shutil.copy2(src_py, dst_py)

    if SYSROOT_BIN.is_dir():
        for item in SYSROOT_BIN.iterdir():
            if item.is_file() and item.suffix.lower() in (".dll", ".so", ".dylib"):
                dst_sys = DIST_BIN / item.name
                ensure_writable(dst_sys)
                shutil.copy2(item, dst_sys)

    if IS_WIN and not (DIST_BIN / "vulkan-1.dll").is_file():
        vk_loader_tmp = BIN_DIR / "build_tmp" / "vulkan_loader"
        if vk_loader_tmp.is_dir():
            vk_candidates = list(vk_loader_tmp.rglob("vulkan-1.dll"))
            if vk_candidates:
                dst_vk = DIST_BIN / "vulkan-1.dll"
                ensure_writable(dst_vk)
                shutil.copy2(vk_candidates[0], dst_vk)

    ff_bins = {"ffmpeg.exe", "ffprobe.exe"} if IS_WIN else {"ffmpeg", "ffprobe"}
    if FFMPEG_DIR.is_dir():
        for item in FFMPEG_DIR.iterdir():
            if not item.is_file():
                continue
            if item.suffix.lower() == ".html":
                ensure_writable(item)
                item.unlink(missing_ok=True)
                continue
            if item.name.lower() in ff_bins or item.suffix.lower() in (
                ".dll",
                ".so",
                ".dylib",
            ):
                dst_ff = DIST_BIN / item.name
                ensure_writable(dst_ff)
                shutil.copy2(item, dst_ff)


def main() -> None:
    compile_shaders_to_header()
    print("[*] Synthesizing unified Ninja DAG for VizInjector DLL and CLI targets...")
    DAG_DIR.mkdir(parents=True, exist_ok=True)
    DIST_BIN.mkdir(parents=True, exist_ok=True)

    for target_bin in (
        DIST_BIN / ("vizinjector.dll" if IS_WIN else "libvizinjector.so"),
        DAG_DIR / "vizinjector.lib",
        DIST_BIN / "vizinjector.lib",
        DIST_BIN / ("vizinjector-cli.exe" if IS_WIN else "vizinjector-cli"),
    ):
        ensure_writable(target_bin)

    def to_posix(p: Path) -> str:
        return os.path.relpath(p, DAG_DIR).replace("\\", "/")

    sysroot_inc = to_posix(SYSROOT_INC)
    sysroot_lib = to_posix(SYSROOT_LIB)

    common_src = [
        SRC_DIR / "common" / "colorspace_srgb.c",
        SRC_DIR / "common" / "kaiser_resampler_2d.c",
        SRC_DIR / "common" / "patch_packer.c",
        SRC_DIR / "common" / "viz_container.c",
        SRC_DIR / "vulkan" / "vk_compute_engine.c",
        SRC_DIR / "bridge" / "vizinjector_core.c",
    ]
    dll_src = common_src + [SRC_DIR / "native" / "vizinjector_dll.c"]
    cli_src = common_src + [SRC_DIR / "native" / "main_cli.c"]

    ninja_file = DAG_DIR / "build.ninja"
    with open(ninja_file, "w", encoding="utf-8") as f:
        if IS_WIN:
            f.write(f"clang_cl = {to_posix(CLANG_CL_EXE)}\n")
            f.write(f"lld_link = {to_posix(LLD_LINK_EXE)}\n")
            ws_arg = str(WINSDK_DIR.resolve()).replace("\\", "/")
            imsvc_flags = resolve_winsdk_imsvc_flags()
            winsdk_libpaths = resolve_winsdk_libpaths(to_posix)
            cflags_nat = (
                f'/MT /O2 /Ob2 /Oi /Ot /Gw /arch:AVX2 -mfma -msse4.2 -winsysroot "{ws_arg}" {imsvc_flags} '
                f"/DNOMINMAX /DWIN32_LEAN_AND_MEAN /D_CRT_SECURE_NO_WARNINGS /DSPNG_STATIC "
                f'-DVIZINJECTOR_BUILD_DLL /I"{sysroot_inc}" /I"{to_posix(SRC_DIR)}" '
                "-Wno-unknown-argument -Wno-deprecated-declarations -Qunused-arguments"
            )
            win_sys_libs = (
                "vulkan-1.lib webp.lib spng.lib miniz.lib stb_image.lib fftw3.lib blis.lib "
                "libvcruntime.lib libucrt.lib libcmt.lib libcpmt.lib oldnames.lib "
                "kernel32.lib user32.lib advapi32.lib"
            )
            ld_dll = (
                f'/NOLOGO /DLL /OUT:"{to_posix(DIST_BIN / "vizinjector.dll")}" '
                f'/IMPLIB:"{to_posix(DAG_DIR / "vizinjector.lib")}" '
                f'/LIBPATH:"{sysroot_lib}" {winsdk_libpaths} {win_sys_libs}'
            )
            ld_cli = (
                f'/NOLOGO /OUT:"{to_posix(DIST_BIN / "vizinjector-cli.exe")}" '
                f'/LIBPATH:"{sysroot_lib}" {winsdk_libpaths} {win_sys_libs}'
            )
            f.write(f"cflags_nat = {cflags_nat}\n")
            f.write(
                'rule cc_nat\n  command = "$clang_cl" $cflags_nat /c "$in" /Fo"$out"\n\n'
            )
            f.write(f'rule link_dll\n  command = "$lld_link" $in {ld_dll}\n\n')
            f.write(f'rule link_cli\n  command = "$lld_link" $in {ld_cli}\n\n')
        else:
            f.write(f"clang = {to_posix(CLANG_EXE)}\n")
            cflags_nat = (
                f'-O3 -mavx2 -mfma -msse4.2 -flto -fPIC -DSPNG_STATIC -I"{sysroot_inc}" -I"{to_posix(SRC_DIR)}"'
            )
            f.write(f"cflags_nat = {cflags_nat}\n")
            f.write(
                'rule cc_nat\n  command = "$clang" $cflags_nat -c "$in" -o "$out"\n\n'
            )
            f.write(
                f'rule link_dll\n  command = "$clang" -shared -flto -o "{to_posix(DIST_BIN / "libvizinjector.so")}" '
                f'$in -L"{sysroot_lib}" -lvulkan -lwebp -lspng -lminiz -lstb_image -lfftw3 -lblis -lm -lpthread\n\n'
            )
            f.write(
                f'rule link_cli\n  command = "$clang" -flto -o "{to_posix(DIST_BIN / "vizinjector-cli")}" '
                f'$in -L"{sysroot_lib}" -lvulkan -lwebp -lspng -lminiz -lstb_image -lfftw3 -lblis -lm -lpthread\n\n'
            )

        dll_objs = []
        for s in dll_src:
            obj = f"d_{s.stem}.{'obj' if IS_WIN else 'o'}"
            f.write(f"build {obj}: cc_nat {to_posix(s)}\n")
            dll_objs.append(obj)
        f.write(f"build dll_target: link_dll {' '.join(dll_objs)}\n\n")

        cli_objs = []
        for s in cli_src:
            obj = f"c_{s.stem}.{'obj' if IS_WIN else 'o'}"
            f.write(f"build {obj}: cc_nat {to_posix(s)}\n")
            cli_objs.append(obj)
        f.write(f"build cli_target: link_cli {' '.join(cli_objs)}\n\n")
        f.write("default dll_target cli_target\n")

    subprocess.run([str(NINJA_EXE)], cwd=str(DAG_DIR), check=True)
    stage_python_bindings_to_dist()
    print(
        f"[+] VizInjector native deliverables and Python bindings staged in {DIST_BIN}"
    )


if __name__ == "__main__":
    main()