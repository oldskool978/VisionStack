#include "viztoken.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

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
#endif

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

    if (argc < 2) {
        printf("VizToken Native Sovereign Tokenizer CLI (SWISS5 Zero-Init Protocol)\n");
        printf("Usage:\n");
        printf("  viztoken-cli compile <input.tiktoken> <output.ranks.bin> [telemetry.json] [pattern]\n");
        printf("  viztoken-cli bench <ranks.bin> <benchmark_text>\n");
        goto cleanup;
    }

    if (strcmp(argv[1], "compile") == 0) {
        if (argc < 4) {
            fprintf(stderr, "Error: Missing input/output container paths.\n");
            exit_code = 1;
            goto cleanup;
        }
        const char *in_tiktoken = argv[2];
        const char *out_bin = argv[3];
        const char *out_telem = (argc >= 5 && argv[4][0] != '\0') ? argv[4] : NULL;
        const char *pattern = (argc >= 6 && argv[5][0] != '\0') ? argv[5] : NULL;

        viztoken_telemetry_t telem;
        int rc = viztoken_compile_vocab_v5(
            in_tiktoken,
            out_bin,
            pattern,
            NULL,
            0,
            248044U,
            out_telem,
            &telem
        );
        if (rc != 0) {
            fprintf(stderr, "[!] Vocabulary compilation failed with error code: %d\n", rc);
            exit_code = 1;
            goto cleanup;
        }
        printf("[+] Serialization Succeeded (SWISS5 Zero-Init Container):\n");
        printf("    Tokens:         %u\n", telem.total_tokens);
        printf("    Capacity:       %u (Load: %.2f%%)\n", telem.table_capacity, telem.load_factor * 100.0);
        printf("    SSO Ratio:      %.2f%% (%u inline)\n", telem.sso_ratio * 100.0, telem.sso_tokens);
        printf("    Heap Blob:      %llu bytes\n", (unsigned long long)telem.string_blob_bytes);
        printf("    Mean PSL:       %.4f (Max: %u, Var: %.4f)\n", telem.mean_psl, telem.max_psl, telem.psl_variance);
        printf("    Duration:       %.4fs\n", telem.compile_seconds);
        goto cleanup;
    }

    if (strcmp(argv[1], "bench") == 0) {
        if (argc < 4) {
            fprintf(stderr, "Error: Missing container binary path and benchmark text argument.\n");
            exit_code = 1;
            goto cleanup;
        }
        viztoken_ctx_t *ctx = viztoken_init(argv[2]);
        if (!ctx) {
            fprintf(stderr, "[!] Failed to map ranks container: %s\n", argv[2]);
            exit_code = 1;
            goto cleanup;
        }

        const char *test_prompt = argv[3];
        uint32_t in_len = (uint32_t)strlen(test_prompt);
        uint32_t max_toks = in_len + 64U;
        int32_t *tokens = (int32_t *)malloc((size_t)max_toks * sizeof(int32_t));
        if (!tokens) {
            viztoken_destroy(ctx);
            exit_code = 1;
            goto cleanup;
        }

        clock_t t0 = clock();
        int32_t count = viztoken_encode(ctx, test_prompt, in_len, tokens, max_toks);
        clock_t t1 = clock();

        double ms = ((double)(t1 - t0) / CLOCKS_PER_SEC) * 1000.0;
        double denom_s = (ms > 0.0) ? (ms / 1000.0) : 1e-6;
        printf(
            "[+] Encoded %d tokens (%u bytes) in %.4f ms (%.2f MB/s)\n",
            count,
            in_len,
            ms,
            ((double)in_len / (1024.0 * 1024.0)) / denom_s
        );

        uint32_t dec_alloc = in_len * 4U + 1024U;
        char *decoded = (char *)malloc(dec_alloc);
        if (decoded) {
            int32_t needed = viztoken_decode(ctx, tokens, (uint32_t)count, decoded, dec_alloc);
            if ((uint32_t)needed >= dec_alloc) {
                free(decoded);
                dec_alloc = (uint32_t)needed + 1U;
                decoded = (char *)malloc(dec_alloc);
                if (decoded) {
                    viztoken_decode(ctx, tokens, (uint32_t)count, decoded, dec_alloc);
                }
            }
            if (decoded) {
                printf(
                    "[+] Roundtrip Parity: %s\n",
                    (strcmp(test_prompt, decoded) == 0) ? "EXACT MATCH" : "DIVERGENCE DETECTED"
                );
                free(decoded);
            }
        }

        free(tokens);
        viztoken_destroy(ctx);
        goto cleanup;
    }

cleanup:
#if defined(_WIN32)
    free_win32_utf8_argv(w_argc, w_argv);
#endif
    return exit_code;
}