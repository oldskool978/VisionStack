#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
TOOLCHAIN_DIR = ROOT_DIR / "toolchain"
SRC_DIR = ROOT_DIR / "src"
CACHE_DIR = ROOT_DIR / ".forge_cache"
DIST_DIR = ROOT_DIR / "dist"
DIST_BIN_DIR = DIST_DIR / "bin"
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


def sanitize_path(p: Path) -> str:
    return str(p.resolve()).replace("\\", "/")


def resolve_executable(name: str, preferred: Path) -> Path:
    if preferred.exists():
        return preferred.resolve()
    found = list(TOOLCHAIN_DIR.glob(f"**/{preferred.name}"))
    if found:
        return found[0].resolve()
    which = shutil.which(name)
    if which is not None:
        return Path(which).resolve()
    return preferred.resolve()


def resolve_winsdk_imsvc_flags() -> str:
    inc_dirs = []
    inc_dirs.extend(WINSDK_DIR.glob("VC/Tools/MSVC/*/include"))
    for sub in ("ucrt", "um", "shared", "winrt"):
        inc_dirs.extend(WINSDK_DIR.glob(f"Windows Kits/10/Include/*/{sub}"))
    return " ".join(
        f'/imsvc"{str(d.resolve()).replace(chr(92), "/")}"' for d in inc_dirs
    )


def ensure_directory_writable(dir_path: Path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    probe_file = dir_path / f".write_test_{os.getpid()}"
    try:
        probe_file.write_text("probe", encoding="utf-8")
        probe_file.unlink()
    except (PermissionError, OSError) as e:
        raise PermissionError(
            f"Directory {dir_path} lacks verified write permissions: {e}"
        )


def remove_readonly_flag(path: Path) -> None:
    if path.exists():
        try:
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        except Exception:
            pass


def purge_non_runtime_artifacts(dist_bin: Path) -> None:
    if not dist_bin.is_dir():
        return
    for item in list(dist_bin.iterdir()):
        if item.is_file() and item.suffix.lower() in (
            ".lib",
            ".exp",
            ".pdb",
            ".json",
            ".html",
        ):
            remove_readonly_flag(item)
            item.unlink(missing_ok=True)


def main() -> None:
    print("[*] Synthesizing Ninja DAG for VizToken Monolith...")
    ensure_directory_writable(DIST_BIN_DIR)
    purge_non_runtime_artifacts(DIST_BIN_DIR)

    ninja_exe = resolve_executable(
        "ninja", TOOLCHAIN_DIR / "ninja" / ("ninja.exe" if IS_WIN else "ninja")
    )
    clang_exe = resolve_executable(
        "clang", TOOLCHAIN_DIR / "llvm" / "bin" / ("clang.exe" if IS_WIN else "clang")
    )
    clang_cl_exe = resolve_executable(
        "clang-cl" if IS_WIN else "clang",
        TOOLCHAIN_DIR / "llvm" / "bin" / ("clang-cl.exe" if IS_WIN else "clang-cl"),
    )
    lld_link_exe = resolve_executable(
        "lld-link" if IS_WIN else "clang",
        TOOLCHAIN_DIR / "llvm" / "bin" / ("lld-link.exe" if IS_WIN else "lld"),
    )

    dag_dir = CACHE_DIR / "dag-build"
    dag_dir.mkdir(parents=True, exist_ok=True)

    def to_posix(p: Path) -> str:
        return os.path.relpath(p, dag_dir).replace("\\", "/")

    c_source = SRC_DIR / "viztoken.c"
    main_source = SRC_DIR / "main.c"
    out_obj = "viztoken.obj" if IS_WIN else "viztoken.o"
    main_obj = "main.obj" if IS_WIN else "main.o"

    out_dll = DIST_BIN_DIR / ("viztoken.dll" if IS_WIN else "libviztoken.so")
    out_implib = dag_dir / "viztoken.lib"
    out_cli = DIST_BIN_DIR / ("viztoken-cli.exe" if IS_WIN else "viztoken-cli")
    out_bin = DIST_BIN_DIR / "qwen.ranks.bin"
    out_telem = dag_dir / "telemetry.json"

    for target in [out_dll, out_implib, out_cli, out_bin, out_telem]:
        remove_readonly_flag(target)

    inc_sysroot = NATIVE_SYSROOT / "include"
    lib_sysroot = NATIVE_SYSROOT / "lib"
    pcre2_lib = lib_sysroot / ("pcre2-8.lib" if IS_WIN else "libpcre2-8.a")
    if not pcre2_lib.exists():
        build_native_script = ROOT_DIR / "tools" / "build_native.py"
        subprocess.run([sys.executable, "-I", str(build_native_script)], check=True)

    qwen_tiktoken_candidate = ROOT_DIR / "qwen.tiktoken"

    ninja_file = dag_dir / "build.ninja"
    with open(ninja_file, "w", encoding="utf-8") as f:
        f.write(f"clang = {to_posix(clang_exe)}\n")
        if IS_WIN:
            f.write(f"clang_cl = {to_posix(clang_cl_exe)}\n")
            f.write(f"lld_link = {to_posix(lld_link_exe)}\n")
            winsysroot_arg = sanitize_path(WINSDK_DIR)
            imsvc_flags = resolve_winsdk_imsvc_flags()
            cflags = (
                f'/MT /O2 /Ob2 /Oi /Ot /Gw /arch:AVX2 -mfma -winsysroot "{winsysroot_arg}" {imsvc_flags} '
                f'/DNOMINMAX /DWIN32_LEAN_AND_MEAN /I"{to_posix(inc_sysroot)}" /I"{to_posix(SRC_DIR)}" '
                "-DVIZTOKEN_BUILD_DLL -DPCRE2_STATIC -DPCRE2_CODE_UNIT_WIDTH=8 "
                "-Wno-unknown-argument -Qunused-arguments"
            )
            f.write(f"cflags = {cflags}\n")

            msvc_libs = sorted(WINSDK_DIR.glob("VC/Tools/MSVC/*/lib/x64"))
            sdk_libs_um = sorted(WINSDK_DIR.glob("Windows Kits/10/Lib/*/um/x64"))
            sdk_libs_ucrt = sorted(WINSDK_DIR.glob("Windows Kits/10/Lib/*/ucrt/x64"))
            if not msvc_libs or not sdk_libs_um or not sdk_libs_ucrt:
                raise FileNotFoundError(
                    "Sysroot fakeroot libraries missing in toolchain/winsdk. Run tools/hydrate.py first."
                )
            msvc_lib = msvc_libs[-1]
            sdk_lib_um = sdk_libs_um[-1]
            sdk_lib_ucrt = sdk_libs_ucrt[-1]

            link_flags_dll = (
                f'/NOLOGO /DLL /OUT:"{to_posix(out_dll)}" /IMPLIB:"{to_posix(out_implib)}" '
                f'/LIBPATH:"{to_posix(lib_sysroot)}" /LIBPATH:"{to_posix(msvc_lib)}" '
                f'/LIBPATH:"{to_posix(sdk_lib_um)}" /LIBPATH:"{to_posix(sdk_lib_ucrt)}" '
                f'"{to_posix(pcre2_lib)}" libvcruntime.lib libucrt.lib libcmt.lib '
                "kernel32.lib user32.lib advapi32.lib"
            )
            link_flags_cli = (
                f'/NOLOGO /OUT:"{to_posix(out_cli)}" '
                f'/LIBPATH:"{to_posix(lib_sysroot)}" /LIBPATH:"{to_posix(msvc_lib)}" '
                f'/LIBPATH:"{to_posix(sdk_lib_um)}" /LIBPATH:"{to_posix(sdk_lib_ucrt)}" '
                f'"{to_posix(out_implib)}" "{to_posix(pcre2_lib)}" libvcruntime.lib libucrt.lib libcmt.lib '
                "kernel32.lib user32.lib advapi32.lib"
            )
            f.write(f"link_flags_dll = {link_flags_dll}\n")
            f.write(f"link_flags_cli = {link_flags_cli}\n\n")
            f.write("rule cc\n")
            f.write('  command = "$clang_cl" $cflags /c "$in" /Fo:"$out"\n')
            f.write("  description = CC $out\n\n")
            f.write("rule link_dll\n")
            f.write('  command = "$lld_link" $in $link_flags_dll\n')
            f.write("  description = LINK_DLL $out\n\n")
            f.write("rule link_cli\n")
            f.write('  command = "$lld_link" $in $link_flags_cli\n')
            f.write("  description = LINK_CLI $out\n\n")
        else:
            cflags = (
                f'-O3 -mavx2 -mfma -flto -fPIC -I"{to_posix(inc_sysroot)}" -I"{to_posix(SRC_DIR)}" '
                "-DVIZTOKEN_BUILD_DLL -DPCRE2_STATIC -DPCRE2_CODE_UNIT_WIDTH=8"
            )
            f.write(f"cflags = {cflags}\n\n")
            f.write("rule cc\n")
            f.write('  command = "$clang" $cflags -c "$in" -o "$out"\n')
            f.write("  description = CC $out\n\n")
            f.write("rule link_dll\n")
            f.write(
                f'  command = "$clang" -shared -flto -o "$out" $in "{to_posix(pcre2_lib)}" -lpthread\n'
            )
            f.write("  description = LINK_DLL $out\n\n")
            f.write("rule link_cli\n")
            f.write(
                f'  command = "$clang" -flto -o "$out" $in "{to_posix(out_dll)}" "{to_posix(pcre2_lib)}" -lpthread\n'
            )
            f.write("  description = LINK_CLI $out\n\n")

        f.write("rule compile_vocab\n")
        f.write('  command = "$in_cli" compile "$in_vocab" "$out_bin" "$out_telem"\n')
        f.write("  description = COMPILE_VOCAB $out_bin\n\n")

        f.write(f"build {out_obj}: cc {to_posix(c_source)}\n")
        if IS_WIN:
            f.write(
                f"build {to_posix(out_dll)} {to_posix(out_implib)}: link_dll {out_obj}\n"
            )
        else:
            f.write(f"build {to_posix(out_dll)}: link_dll {out_obj}\n")
        f.write(f"build {main_obj}: cc {to_posix(main_source)}\n")
        f.write(
            f"build {to_posix(out_cli)}: link_cli {main_obj} | {to_posix(out_dll)}\n"
        )

        if qwen_tiktoken_candidate.exists():
            f.write(
                f"build {to_posix(out_bin)} {to_posix(out_telem)}: compile_vocab | {to_posix(out_cli)}\n"
            )
            f.write(f"  in_cli = {to_posix(out_cli)}\n")
            f.write(f"  in_vocab = {to_posix(qwen_tiktoken_candidate)}\n")
            f.write(f"  out_bin = {to_posix(out_bin)}\n")
            f.write(f"  out_telem = {to_posix(out_telem)}\n")
            f.write(
                f"default {to_posix(out_dll)} {to_posix(out_cli)} {to_posix(out_bin)}\n"
            )
        else:
            f.write(f"default {to_posix(out_dll)} {to_posix(out_cli)}\n")

    print("  -> Compiling deliverables via Ninja...")
    subprocess.run([str(ninja_exe)], cwd=str(dag_dir.resolve()), check=True)

    root_dll = ROOT_DIR / out_dll.name
    remove_readonly_flag(root_dll)
    shutil.copy2(out_dll, root_dll)

    for py_name in ("viztoken.py", "__init__.py"):
        src_py = ROOT_DIR / py_name
        if src_py.is_file():
            dst_py = DIST_BIN_DIR / py_name
            remove_readonly_flag(dst_py)
            shutil.copy2(src_py, dst_py)

    print(f"[+] Build successful: {out_dll} and {out_cli}")
    if out_bin.exists():
        print(
            f"[+] Compiled SWISS5 vocabulary binary verified: {out_bin} ({out_bin.stat().st_size:,} bytes)"
        )

    qwen35_tiktoken = ROOT_DIR / "qwen35.tiktoken"
    if qwen35_tiktoken.is_file():
        if str(ROOT_DIR) not in sys.path:
            sys.path.insert(0, str(ROOT_DIR))
        import viztoken

        out_bin_35 = DIST_BIN_DIR / "qwen35.ranks.bin"
        out_telem_35 = dag_dir / "telemetry_qwen35.json"
        remove_readonly_flag(out_bin_35)
        remove_readonly_flag(out_telem_35)
        qwen35_pat = (
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|"
            r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        )
        telem35 = viztoken.compile_vocab(
            in_tiktoken_path=qwen35_tiktoken,
            out_bin_path=out_bin_35,
            out_telemetry_json=out_telem_35,
            regex_pattern=qwen35_pat,
            specials_list=viztoken.build_default_qwen35_specials(),
            eot_token_id=248044,
        )
        print(
            f"[+] Compiled Qwen3.5 SWISS5 binary verified: {out_bin_35} "
            f"({out_bin_35.stat().st_size:,} bytes, {telem35.total_tokens:,} tokens)"
        )

    purge_non_runtime_artifacts(DIST_BIN_DIR)


if __name__ == "__main__":
    main()