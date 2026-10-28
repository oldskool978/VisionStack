#ifndef VIZTOKEN_H
#define VIZTOKEN_H

#include <stdint.h>
#include <stddef.h>

#if defined(_WIN32)
  #if defined(VIZTOKEN_BUILD_DLL)
    #define VIZTOKEN_API __declspec(dllexport)
  #else
    #define VIZTOKEN_API __declspec(dllimport)
  #endif
#else
  #define VIZTOKEN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct viztoken_ctx_t viztoken_ctx_t;

#pragma pack(push, 1)
typedef struct {
    int32_t  id;
    uint16_t length;
    char     literal[58];
} viztoken_special_disk_t;
#pragma pack(pop)

typedef struct {
    uint32_t total_tokens;
    uint32_t table_capacity;
    uint32_t sso_tokens;
    uint32_t max_psl;
    uint64_t string_blob_bytes;
    double   load_factor;
    double   sso_ratio;
    double   mean_psl;
    double   psl_variance;
    double   compile_seconds;
} viztoken_telemetry_t;

VIZTOKEN_API viztoken_ctx_t *viztoken_init(const char *ranks_bin_path);

VIZTOKEN_API int32_t viztoken_encode(
    viztoken_ctx_t *ctx,
    const char *text,
    uint32_t text_bytes,
    int32_t *out_token_ids,
    uint32_t max_tokens
);

VIZTOKEN_API int32_t viztoken_encode_ordinary(
    viztoken_ctx_t *ctx,
    const char *text,
    uint32_t text_bytes,
    int32_t *out_token_ids,
    uint32_t max_tokens
);

VIZTOKEN_API int32_t viztoken_decode(
    viztoken_ctx_t *ctx,
    const int32_t *token_ids,
    uint32_t id_count,
    char *out_text,
    uint32_t max_bytes
);

VIZTOKEN_API uint32_t viztoken_find_common_prefix(
    const int32_t *tokens_a,
    uint32_t len_a,
    const int32_t *tokens_b,
    uint32_t len_b
);

VIZTOKEN_API int32_t viztoken_compile_vocab(
    const char *in_tiktoken_path,
    const char *out_bin_path
);

VIZTOKEN_API int32_t viztoken_compile_vocab_with_telemetry(
    const char *in_tiktoken_path,
    const char *out_bin_path,
    const char *out_telemetry_json,
    viztoken_telemetry_t *out_telemetry
);

VIZTOKEN_API int32_t viztoken_compile_vocab_v5(
    const char *in_tiktoken_path,
    const char *out_bin_path,
    const char *regex_pattern,
    const viztoken_special_disk_t *specials,
    uint32_t specials_count,
    uint32_t eot_token_id,
    const char *out_telemetry_json,
    viztoken_telemetry_t *out_telemetry
);

VIZTOKEN_API void viztoken_destroy(viztoken_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif