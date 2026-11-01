#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../bridge/vizinjector_core.h"
#include "../vulkan/vk_compute_engine.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>

static int utf8_to_wide(const char *utf8_path, wchar_t *out_wpath, size_t max_wchars) {
    if (!utf8_path || !out_wpath || max_wchars == 0) return -1;
    int n = MultiByteToWideChar(CP_UTF8, 0, utf8_path, -1, out_wpath, (int)max_wchars);
    if (n <= 0) {
        n = MultiByteToWideChar(CP_ACP, 0, utf8_path, -1, out_wpath, (int)max_wchars);
    }
    return (n > 0) ? 0 : -1;
}

static FILE *open_file_utf8(const char *path, const wchar_t *wmode) {
    wchar_t wpath[2048];
    if (utf8_to_wide(path, wpath, 2048) != 0) return NULL;
    return _wfopen(wpath, wmode);
}

static char *wide_slice_to_utf8(const wchar_t *wstr, size_t wlen) {
    if (wlen == 0) {
        char *empty = (char *)malloc(1);
        if (empty) empty[0] = '\0';
        return empty;
    }
    int needed = WideCharToMultiByte(CP_UTF8, 0, wstr, (int)wlen, NULL, 0, NULL, NULL);
    if (needed <= 0) return NULL;
    char *out = (char *)malloc((size_t)needed + 1);
    if (!out) return NULL;
    WideCharToMultiByte(CP_UTF8, 0, wstr, (int)wlen, out, needed, NULL, NULL);
    out[needed] = '\0';
    return out;
}

static int parse_win32_utf8_argv(int *out_argc, char ***out_argv) {
    const wchar_t *cmd = GetCommandLineW();
    if (!cmd) return -1;

    size_t cmd_len = wcslen(cmd);
    wchar_t *scratch = (wchar_t *)malloc((cmd_len + 1) * sizeof(wchar_t));
    if (!scratch) return -1;

    int cap = 16;
    int argc = 0;
    char **argv = (char **)malloc((size_t)cap * sizeof(char *));
    if (!argv) {
        free(scratch);
        return -1;
    }

    const wchar_t *p = cmd;
    while (*p) {
        while (*p == L' ' || *p == L'\t') p++;
        if (!*p) break;

        size_t out_idx = 0;
        int in_quotes = 0;

        while (*p) {
            if (!in_quotes && (*p == L' ' || *p == L'\t')) {
                break;
            }
            if (*p == L'\\') {
                size_t bs_count = 0;
                while (*p == L'\\') {
                    bs_count++;
                    p++;
                }
                if (*p == L'"') {
                    for (size_t i = 0; i < bs_count / 2; i++) {
                        scratch[out_idx++] = L'\\';
                    }
                    if (bs_count % 2 == 0) {
                        in_quotes = !in_quotes;
                    } else {
                        scratch[out_idx++] = L'"';
                    }
                    p++;
                } else {
                    for (size_t i = 0; i < bs_count; i++) {
                        scratch[out_idx++] = L'\\';
                    }
                }
            } else if (*p == L'"') {
                if (in_quotes && *(p + 1) == L'"') {
                    scratch[out_idx++] = L'"';
                    p += 2;
                } else {
                    in_quotes = !in_quotes;
                    p++;
                }
            } else {
                scratch[out_idx++] = *p++;
            }
        }

        if (argc >= cap) {
            cap *= 2;
            char **next_argv = (char **)realloc(argv, (size_t)cap * sizeof(char *));
            if (!next_argv) {
                for (int i = 0; i < argc; i++) free(argv[i]);
                free(argv);
                free(scratch);
                return -1;
            }
            argv = next_argv;
        }
        argv[argc++] = wide_slice_to_utf8(scratch, out_idx);
    }

    free(scratch);
    *out_argc = argc;
    *out_argv = argv;
    return 0;
}

static void free_win32_utf8_argv(int argc, char **argv) {
    if (!argv) return;
    for (int i = 0; i < argc; i++) {
        free(argv[i]);
    }
    free(argv);
}
#else
#include <unistd.h>

static FILE *open_file_utf8(const char *path, const char *mode) {
    return fopen(path, mode);
}
#endif

static uint8_t *read_entire_file(const char *path, size_t *out_len) {
#if defined(_WIN32)
    FILE *fp = open_file_utf8(path, L"rb");
#else
    FILE *fp = open_file_utf8(path, "rb");
#endif
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0) {
        fclose(fp);
        return NULL;
    }
    uint8_t *buf = (uint8_t *)malloc((size_t)sz);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    size_t r = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    *out_len = r;
    return buf;
}

static int write_entire_file(const char *path, const void *data, size_t len) {
#if defined(_WIN32)
    FILE *fp = open_file_utf8(path, L"wb");
#else
    FILE *fp = open_file_utf8(path, "wb");
#endif
    if (!fp) return -1;
    size_t w = fwrite(data, 1, len, fp);
    fflush(fp);
#if defined(_WIN32)
    int fd = _fileno(fp);
    if (fd >= 0) {
        HANDLE h = (HANDLE)_get_osfhandle(fd);
        if (h != INVALID_HANDLE_VALUE) {
            FlushFileBuffers(h);
        }
    }
#else
    int fd = fileno(fp);
    if (fd >= 0) {
        fsync(fd);
    }
#endif
    fclose(fp);
    return (w == len) ? 0 : -2;
}

int main(int argc, char **argv) {
#if defined(_WIN32)
    int w_argc = 0;
    char **w_argv = NULL;
    if (parse_win32_utf8_argv(&w_argc, &w_argv) == 0 && w_argv) {
        argc = w_argc;
        argv = w_argv;
    }
#endif

    int exit_code = 0;

    if (argc < 3) {
        fprintf(
            stdout,
            "Usage: vizinjector-cli <input_image> <output.vizj> [--width W] [--height H] "
            "[--role photo|color|depth|mask] [--framing inscribed|contain|stretch|never_crop] "
            "[--edit-384] [--cpu] [--export-debug]\n"
        );
        exit_code = 1;
        goto cleanup;
    }

    const char *in_path = argv[1];
    const char *out_path = argv[2];

    VizInjectorConfig cfg;
    vizinjector_core_default_config(&cfg);
    cfg.emit_wire_vizj = 1;
    int export_debug = 0;

    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            cfg.target_width = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            cfg.target_height = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--role") == 0 && i + 1 < argc) {
            const char *r = argv[++i];
            if (strcmp(r, "color") == 0) cfg.input_role = VIZ_ROLE_COLOR_MAP;
            else if (strcmp(r, "depth") == 0) cfg.input_role = VIZ_ROLE_DEPTH_MAP;
            else if (strcmp(r, "mask") == 0) cfg.input_role = VIZ_ROLE_INPAINT_MASK;
            else cfg.input_role = VIZ_ROLE_PHOTO_REF;
        } else if (strcmp(argv[i], "--framing") == 0 && i + 1 < argc) {
            const char *f = argv[++i];
            if (strcmp(f, "contain") == 0) cfg.framing_mode = VIZ_FRAMING_CONTAIN;
            else if (strcmp(f, "stretch") == 0) cfg.framing_mode = VIZ_FRAMING_STRETCH;
            else if (strcmp(f, "never_crop") == 0) cfg.framing_mode = VIZ_FRAMING_NEVER_CROP;
            else cfg.framing_mode = VIZ_FRAMING_INSCRIBED;
        } else if (strcmp(argv[i], "--edit-384") == 0) {
            cfg.edit_mode_384_vit = 1;
        } else if (strcmp(argv[i], "--cpu") == 0) {
            cfg.prefer_vulkan = 0;
        } else if (strcmp(argv[i], "--export-debug") == 0) {
            export_debug = 1;
        }
    }

    size_t in_len = 0;
    uint8_t *in_bytes = read_entire_file(in_path, &in_len);
    if (!in_bytes) {
        fprintf(stderr, "[!] Failed reading input image: %s\n", in_path);
        exit_code = 1;
        goto cleanup;
    }

    clock_t t0 = clock();
    VizInjectorOutput out;
    int32_t rc = vizinjector_core_process_memory(in_bytes, in_len, &cfg, &out);
    double elapsed_ms = ((double)(clock() - t0) / (double)CLOCKS_PER_SEC) * 1000.0;
    free(in_bytes);

    if (rc != 0) {
        fprintf(stderr, "[!] VizInjector conditioning failed with code %d\n", rc);
        exit_code = 1;
        goto cleanup;
    }

    if (out.wire_vizj_bytes && out.wire_vizj_len > 0) {
        if (write_entire_file(out_path, out.wire_vizj_bytes, out.wire_vizj_len) != 0) {
            fprintf(stderr, "[!] Failed writing output container: %s\n", out_path);
            vizinjector_core_free_output(&out);
            exit_code = 1;
            goto cleanup;
        }
    }

    fprintf(
        stdout,
        "[+] Conditioned %ux%u -> VAE %ux%u | ViT %ux%u (%u LLM tokens) in %.2f ms [%s]\n",
        out.orig_width,
        out.orig_height,
        out.vae_width,
        out.vae_height,
        out.vit_width,
        out.vit_height,
        out.llm_tokens,
        elapsed_ms,
        out.backend_used ? viz_vk_device_name() : "C11 AVX2"
    );
    fprintf(
        stdout,
        "[+] Telemetry: Luma=%.4f | Contrast=%.4f | Entropy=%.3f bits | Sharpness=%.4f\n",
        out.mean_luminance,
        out.rms_contrast,
        out.shannon_entropy_bits,
        out.laplacian_sharpness
    );

    if (export_debug && out.vae_planar_f32) {
        size_t px = (size_t)out.vae_width * (size_t)out.vae_height;
        uint8_t *rgb8 = (uint8_t *)malloc(px * 3u);
        if (rgb8) {
            viz_postprocess_vae_f32_to_rgb8(out.vae_planar_f32, out.vae_width, out.vae_height, 1, rgb8);
            uint8_t *png_buf = NULL;
            uint32_t png_len = 0;
            if (viz_encode_rgb8_to_png(rgb8, out.vae_width, out.vae_height, &png_buf, &png_len) == 0 && png_buf) {
                write_entire_file("debug_vae_16px.png", png_buf, png_len);
                free(png_buf);
            }
            free(rgb8);
        }
        if (out.latent_mask_f32 && out.mask_width > 0 && out.mask_height > 0) {
            size_t m_px = (size_t)out.mask_width * (size_t)out.mask_height;
            uint8_t *m_rgb = (uint8_t *)malloc(m_px * 3u);
            if (m_rgb) {
                for (size_t i = 0; i < m_px; ++i) {
                    uint8_t v = (uint8_t)lrintf(fminf(1.0f, fmaxf(0.0f, out.latent_mask_f32[i])) * 255.0f);
                    m_rgb[i * 3u + 0u] = v;
                    m_rgb[i * 3u + 1u] = v;
                    m_rgb[i * 3u + 2u] = v;
                }
                uint8_t *png_buf = NULL;
                uint32_t png_len = 0;
                if (viz_encode_rgb8_to_png(m_rgb, out.mask_width, out.mask_height, &png_buf, &png_len) == 0 && png_buf) {
                    write_entire_file("debug_mask_16x.png", png_buf, png_len);
                    free(png_buf);
                }
                free(m_rgb);
            }
        }
    }

    vizinjector_core_free_output(&out);
    viz_vk_shutdown();

cleanup:
#if defined(_WIN32)
    free_win32_utf8_argv(w_argc, w_argv);
#endif
    return exit_code;
}