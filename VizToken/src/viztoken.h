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

#define QWEN35_BASE_VOCAB_SIZE          248044U
#define QWEN35_TOTAL_VOCAB_SIZE         248320U

#define QWEN35_EOT_TOKEN_ID             248044U
#define QWEN35_IM_START_TOKEN_ID        248045U
#define QWEN35_IM_END_TOKEN_ID          248046U
#define QWEN35_OBJECT_REF_START_ID      248047U
#define QWEN35_OBJECT_REF_END_ID        248048U
#define QWEN35_BOX_START_ID             248049U
#define QWEN35_BOX_END_ID               248050U
#define QWEN35_QUAD_START_ID            248051U
#define QWEN35_QUAD_END_ID              248052U
#define QWEN35_VISION_START_ID          248053U
#define QWEN35_VISION_END_ID            248054U
#define QWEN35_VISION_PAD_ID            248055U
#define QWEN35_IMAGE_PAD_ID             248056U
#define QWEN35_VIDEO_PAD_ID             248057U
#define QWEN35_TOOL_CALL_START_ID       248058U
#define QWEN35_TOOL_CALL_END_ID         248059U
#define QWEN35_FIM_PREFIX_ID            248060U
#define QWEN35_FIM_MIDDLE_ID            248061U
#define QWEN35_FIM_SUFFIX_ID            248062U
#define QWEN35_FIM_PAD_ID               248063U
#define QWEN35_REPO_NAME_ID             248064U
#define QWEN35_FILE_SEP_ID              248065U
#define QWEN35_TOOL_RESPONSE_START_ID   248066U
#define QWEN35_TOOL_RESPONSE_END_ID     248067U
#define QWEN35_THINK_START_ID           248068U
#define QWEN35_THINK_END_ID             248069U
#define QWEN35_AUDIO_START_ID           248070U
#define QWEN35_AUDIO_END_ID             248071U
#define QWEN35_TTS_PAD_ID               248072U
#define QWEN35_TTS_TEXT_BOS_ID          248073U
#define QWEN35_TTS_TEXT_EOD_ID          248074U
#define QWEN35_TTS_TEXT_BOS_SINGLE_ID   248075U
#define QWEN35_AUDIO_PAD_ID             248076U

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