#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

ROOT_DIR = Path(__file__).parent.parent.resolve()
TOOLCHAIN_DIR = ROOT_DIR / "toolchain"
LIBRARY_DIR = ROOT_DIR / "library"
CACHE_DIR = ROOT_DIR / ".forge_cache"
NATIVE_SYSROOT = TOOLCHAIN_DIR / "native-sysroot"
WINSDK_DIR = TOOLCHAIN_DIR / "winsdk"

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

PCRE2_SRC = LIBRARY_DIR / "pcre2"


def sanitize_path(p: Path) -> str:
    return str(p.resolve()).replace("\\", "/")


def resolve_executable(name: str, preferred: Path) -> Path:
    if preferred.exists():
        return preferred.resolve()
    found = list(TOOLCHAIN_DIR.glob(f"**/{preferred.name}"))
    if found:
        return found[0].resolve()
    which_path = shutil.which(name)
    if which_path is not None:
        return Path(which_path).resolve()
    return preferred.resolve()


def ensure_writable(p: Path) -> None:
    if p.exists():
        try:
            os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
        except Exception:
            pass


def resolve_winsdk_imsvc_flags() -> str:
    inc_dirs = []
    inc_dirs.extend(WINSDK_DIR.glob("VC/Tools/MSVC/*/include"))
    for sub in ("ucrt", "um", "shared", "winrt"):
        inc_dirs.extend(WINSDK_DIR.glob(f"Windows Kits/10/Include/*/{sub}"))
    return " ".join(
        f'/imsvc"{str(d.resolve()).replace(chr(92), "/")}"' for d in inc_dirs
    )


def prepare_pcre2_headers(build_dir: Path) -> Path:
    pcre2_inc = build_dir / "pcre2_inc"
    pcre2_inc.mkdir(parents=True, exist_ok=True)
    src_pcre2_h = PCRE2_SRC / "src" / "pcre2.h.generic"
    if not src_pcre2_h.exists():
        src_pcre2_h = PCRE2_SRC / "src" / "pcre2.h"
    target_pcre2_h = pcre2_inc / "pcre2.h"
    ensure_writable(target_pcre2_h)
    shutil.copy2(src_pcre2_h, target_pcre2_h)

    src_config_h = PCRE2_SRC / "src" / "config.h.generic"
    config_content = ""
    if src_config_h.exists():
        config_content = src_config_h.read_text(encoding="utf-8")
    else:
        src_config_in = PCRE2_SRC / "config.h.in"
        if src_config_in.exists():
            config_content = src_config_in.read_text(encoding="utf-8")

    defines = [
        "#define PCRE2_CODE_UNIT_WIDTH 8",
        "#define SUPPORT_UNICODE 1",
        "#define SUPPORT_JIT 1",
        "#define HAVE_MEMMOVE 1",
        "#define HAVE_STRERROR 1",
        "#define MATCH_LIMIT 10000000",
        "#define MATCH_LIMIT_DEPTH MATCH_LIMIT",
        "#define HEAP_LIMIT 20000000",
        "#define MAX_NAME_SIZE 128",
        "#define MAX_NAME_COUNT 10000",
        "#define LINK_SIZE 2",
        "#define NEWLINE_DEFAULT 2",
        "#define PARENS_NEST_LIMIT 250",
        "#define PCRE2_STATIC 1",
    ]
    custom_config = "\n".join(defines) + "\n" + config_content
    target_config_h = pcre2_inc / "config.h"
    ensure_writable(target_config_h)
    target_config_h.write_text(custom_config, encoding="utf-8")

    src_chartables = PCRE2_SRC / "src" / "pcre2_chartables.c.dist"
    target_chartables = build_dir / "pcre2_chartables.c"
    ensure_writable(target_chartables)
    if src_chartables.exists():
        shutil.copy2(src_chartables, target_chartables)
    else:
        shutil.copy2(PCRE2_SRC / "src" / "pcre2_chartables.c", target_chartables)
    return pcre2_inc


def main() -> None:
    ninja_exe = resolve_executable(
        "ninja",
        TOOLCHAIN_DIR / "ninja" / ("ninja.exe" if IS_WIN else "ninja"),
    )
    clang_exe = resolve_executable(
        "clang",
        TOOLCHAIN_DIR / "llvm" / "bin" / ("clang.exe" if IS_WIN else "clang"),
    )
    clang_cl_exe = resolve_executable(
        "clang-cl" if IS_WIN else "clang",
        TOOLCHAIN_DIR / "llvm" / "bin" / ("clang-cl.exe" if IS_WIN else "clang-cl"),
    )
    llvm_lib_exe = resolve_executable(
        "llvm-lib" if IS_WIN else "llvm-ar",
        TOOLCHAIN_DIR / "llvm" / "bin" / ("llvm-lib.exe" if IS_WIN else "llvm-lib"),
    )
    llvm_ar_exe = resolve_executable(
        "llvm-ar",
        TOOLCHAIN_DIR / "llvm" / "bin" / ("llvm-ar.exe" if IS_WIN else "llvm-ar"),
    )

    NATIVE_SYSROOT.mkdir(parents=True, exist_ok=True)
    inc_sysroot = NATIVE_SYSROOT / "include"
    lib_sysroot = NATIVE_SYSROOT / "lib"
    inc_sysroot.mkdir(parents=True, exist_ok=True)
    lib_sysroot.mkdir(parents=True, exist_ok=True)

    build_dir = CACHE_DIR / "native-build"
    build_dir.mkdir(parents=True, exist_ok=True)
    pcre2_inc_dir = prepare_pcre2_headers(build_dir)

    def to_ninja_path(p: Path) -> str:
        rel = os.path.relpath(p, build_dir).replace("\\", "/")
        return rel.replace("$", "$$").replace(" ", "$ ").replace(":", "$:")

    def to_posix_flag(p: Path) -> str:
        return os.path.relpath(p, build_dir).replace("\\", "/")

    pcre2_candidate_sources = [
        PCRE2_SRC / "src" / "pcre2_auto_possess.c",
        PCRE2_SRC / "src" / "pcre2_chkdint.c",
        PCRE2_SRC / "src" / "pcre2_compile.c",
        PCRE2_SRC / "src" / "pcre2_compile_cgroup.c",
        PCRE2_SRC / "src" / "pcre2_compile_class.c",
        PCRE2_SRC / "src" / "pcre2_config.c",
        PCRE2_SRC / "src" / "pcre2_context.c",
        PCRE2_SRC / "src" / "pcre2_convert.c",
        PCRE2_SRC / "src" / "pcre2_dfa_match.c",
        PCRE2_SRC / "src" / "pcre2_error.c",
        PCRE2_SRC / "src" / "pcre2_extuni.c",
        PCRE2_SRC / "src" / "pcre2_find_bracket.c",
        PCRE2_SRC / "src" / "pcre2_jit_compile.c",
        PCRE2_SRC / "src" / "pcre2_maketables.c",
        PCRE2_SRC / "src" / "pcre2_match.c",
        PCRE2_SRC / "src" / "pcre2_match_data.c",
        PCRE2_SRC / "src" / "pcre2_match_next.c",
        PCRE2_SRC / "src" / "pcre2_newline.c",
        PCRE2_SRC / "src" / "pcre2_ord2utf.c",
        PCRE2_SRC / "src" / "pcre2_pattern_info.c",
        PCRE2_SRC / "src" / "pcre2_script_run.c",
        PCRE2_SRC / "src" / "pcre2_serialize.c",
        PCRE2_SRC / "src" / "pcre2_string_utils.c",
        PCRE2_SRC / "src" / "pcre2_study.c",
        PCRE2_SRC / "src" / "pcre2_substitute.c",
        PCRE2_SRC / "src" / "pcre2_substring.c",
        PCRE2_SRC / "src" / "pcre2_tables.c",
        PCRE2_SRC / "src" / "pcre2_ucd.c",
        PCRE2_SRC / "src" / "pcre2_valid_utf.c",
        PCRE2_SRC / "src" / "pcre2_xclass.c",
        build_dir / "pcre2_chartables.c",
    ]
    pcre2_core_sources = [s for s in pcre2_candidate_sources if s.exists()]

    ninja_file = build_dir / "build.ninja"
    with open(ninja_file, "w", encoding="utf-8") as f:
        f.write(f"clang = {to_posix_flag(clang_exe)}\n")
        if IS_WIN:
            f.write(f"clang_cl = {to_posix_flag(clang_cl_exe)}\n")
            f.write(f"llvm_lib = {to_posix_flag(llvm_lib_exe)}\n")
            winsysroot_arg = sanitize_path(WINSDK_DIR)
            imsvc_flags = resolve_winsdk_imsvc_flags()
            cflags_common = (
                f'/MT /O2 /Ob2 /Oi /Ot /Gw /arch:AVX2 -mfma -winsysroot "{winsysroot_arg}" {imsvc_flags} '
                "/DNOMINMAX /DWIN32_LEAN_AND_MEAN -DPCRE2_CODE_UNIT_WIDTH=8 -DPCRE2_STATIC "
                "-DHAVE_CONFIG_H "
                f'/I"{to_posix_flag(pcre2_inc_dir)}" '
                f'/I"{to_posix_flag(PCRE2_SRC / "src")}" -Wno-unknown-argument '
                "-Qunused-arguments"
            )
            f.write(f"cflags = {cflags_common}\n\n")
            f.write("rule cc\n")
            f.write('  command = "$clang_cl" $cflags /c "$in" /Fo:"$out"\n')
            f.write("  description = CC $out\n\n")
            f.write("rule lib\n")
            f.write('  command = "$llvm_lib" /nologo /OUT:"$out" $in\n')
            f.write("  description = LIB $out\n\n")
        else:
            f.write(f"llvm_ar = {to_posix_flag(llvm_ar_exe)}\n")
            cflags_common = (
                "-O3 -mavx2 -mfma -flto -fPIC -DPCRE2_CODE_UNIT_WIDTH=8 -DPCRE2_STATIC "
                "-DHAVE_CONFIG_H "
                f'-I"{to_posix_flag(pcre2_inc_dir)}" '
                f'-I"{to_posix_flag(PCRE2_SRC / "src")}"'
            )
            f.write(f"cflags = {cflags_common}\n\n")
            f.write("rule cc\n")
            f.write('  command = "$clang" $cflags -c "$in" -o "$out"\n')
            f.write("  description = CC $out\n\n")
            f.write("rule lib\n")
            f.write('  command = "$llvm_ar" rcs "$out" $in\n')
            f.write("  description = AR $out\n\n")

        pcre2_objs = []
        for src in pcre2_core_sources:
            obj = f"{src.stem}.obj" if IS_WIN else f"{src.stem}.o"
            f.write(f"build {obj}: cc {to_ninja_path(src)}\n")
            pcre2_objs.append(obj)

        out_pcre2_lib = lib_sysroot / ("pcre2-8.lib" if IS_WIN else "libpcre2-8.a")
        ensure_writable(out_pcre2_lib)
        f.write(
            f"build {to_ninja_path(out_pcre2_lib)}: lib {' '.join(pcre2_objs)}\n"
        )

    print("[*] Rebuilding PCRE2 static library with AVX2 SLJIT support...")
    subprocess.run([str(ninja_exe)], cwd=str(build_dir.resolve()), check=True)
    dest_inc = inc_sysroot / "pcre2.h"
    ensure_writable(dest_inc)
    shutil.copy2(pcre2_inc_dir / "pcre2.h", dest_inc)
    print(f"[+] Staged updated archive: {out_pcre2_lib}")


if __name__ == "__main__":
    main()