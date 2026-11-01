#ifndef VIZINJECTOR_DLL_H
#define VIZINJECTOR_DLL_H

#include "../bridge/vizinjector_core.h"

#if defined(_WIN32)
  #if defined(VIZINJECTOR_BUILD_DLL)
    #define VIZINJECTOR_API __declspec(dllexport)
  #else
    #define VIZINJECTOR_API __declspec(dllimport)
  #endif
#else
  #define VIZINJECTOR_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

VIZINJECTOR_API int32_t vizinjector_init_backend(int32_t prefer_vulkan);
VIZINJECTOR_API const char *vizinjector_get_backend_name(void);
VIZINJECTOR_API void vizinjector_shutdown_backend(void);

VIZINJECTOR_API void vizinjector_get_default_config(VizInjectorConfig *out_cfg);

VIZINJECTOR_API int32_t vizinjector_process_image_bytes(
    const uint8_t *file_bytes,
    uint32_t file_len,
    const VizInjectorConfig *cfg,
    VizInjectorOutput *out_result
);

VIZINJECTOR_API int32_t vizinjector_process_rgba8(
    const uint8_t *rgba8,
    uint32_t width,
    uint32_t height,
    const VizInjectorConfig *cfg,
    VizInjectorOutput *out_result
);

VIZINJECTOR_API int32_t vizinjector_postprocess_vae_to_rgb8(
    const float *planar_vae_rgb,
    uint32_t width,
    uint32_t height,
    int32_t apply_bayer_dither,
    uint8_t *out_rgb8,
    VizPerceptualTelemetry *out_telemetry
);

VIZINJECTOR_API int32_t vizinjector_pack_vizj_buffer(
    const VizContainerHeader *hdr_template,
    const void *vae_payload,
    uint64_t vae_bytes,
    const void *vit_payload,
    uint64_t vit_bytes,
    const void *mask_payload,
    uint64_t mask_bytes,
    uint8_t **out_bytes,
    uint64_t *out_len
);

VIZINJECTOR_API int32_t vizinjector_unpack_vizj_buffer(
    const uint8_t *container_bytes,
    uint64_t container_len,
    VizContainerHeader *out_hdr,
    const void **out_vae_ptr,
    const void **out_vit_ptr,
    const void **out_mask_ptr
);

VIZINJECTOR_API void vizinjector_free_output(VizInjectorOutput *out_result);
VIZINJECTOR_API void vizinjector_free_buffer(void *ptr);

#ifdef __cplusplus
}
#endif

#endif