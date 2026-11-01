#define VIZINJECTOR_BUILD_DLL 1
#include "vizinjector_dll.h"
#include "../common/colorspace_srgb.h"
#include "../vulkan/vk_compute_engine.h"
#include <immintrin.h>
#include <stdlib.h>

#if defined(_MSC_VER)
#define VIZ_DLL_TLS __declspec(thread)
#else
#define VIZ_DLL_TLS __thread
#endif

static VIZ_DLL_TLS float *g_tls_dll_lin = NULL;
static VIZ_DLL_TLS size_t g_tls_dll_lin_cap = 0;

static float *acquire_dll_tls_f32(size_t needed_floats) {
    if (needed_floats <= g_tls_dll_lin_cap && g_tls_dll_lin != NULL) {
        return g_tls_dll_lin;
    }
    size_t alloc_floats = (needed_floats + 4095u) & ~(size_t)4095u;
    float *next_buf = (float *)realloc(g_tls_dll_lin, alloc_floats * sizeof(float));
    if (!next_buf) {
        return NULL;
    }
    g_tls_dll_lin = next_buf;
    g_tls_dll_lin_cap = alloc_floats;
    return next_buf;
}

static void clamp_f32_01_avx2(const float *src, size_t count, float *dst) {
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_one = _mm256_set1_ps(1.0f);
    size_t i = 0;
    for (; i + 8u <= count; i += 8u) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m256 valid = _mm256_cmp_ps(v, v_zero, _CMP_GT_OQ);
        __m256 clamped = _mm256_min_ps(v_one, _mm256_and_ps(v, valid));
        _mm256_storeu_ps(dst + i, clamped);
    }
    for (; i < count; ++i) {
        float v = src[i];
        if (!(v > 0.0f)) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        dst[i] = v;
    }
}

VIZINJECTOR_API int32_t vizinjector_init_backend(int32_t prefer_vulkan) {
    viz_colorspace_init_tables();
    if (prefer_vulkan) {
        if (viz_vk_init() == 0) return 1;
    }
    return 0;
}

VIZINJECTOR_API const char *vizinjector_get_backend_name(void) {
    return viz_vk_device_name();
}

VIZINJECTOR_API void vizinjector_shutdown_backend(void) {
    viz_vk_shutdown();
}

VIZINJECTOR_API void vizinjector_get_default_config(VizInjectorConfig *out_cfg) {
    vizinjector_core_default_config(out_cfg);
}

VIZINJECTOR_API int32_t vizinjector_process_image_bytes(
    const uint8_t *file_bytes,
    uint32_t file_len,
    const VizInjectorConfig *cfg,
    VizInjectorOutput *out_result
) {
    return vizinjector_core_process_memory(file_bytes, (size_t)file_len, cfg, out_result);
}

VIZINJECTOR_API int32_t vizinjector_process_rgba8(
    const uint8_t *rgba8,
    uint32_t width,
    uint32_t height,
    const VizInjectorConfig *cfg,
    VizInjectorOutput *out_result
) {
    return vizinjector_core_process_rgba8(rgba8, width, height, cfg, out_result);
}

VIZINJECTOR_API int32_t vizinjector_process_rgba16(
    const uint16_t *rgba16,
    uint32_t width,
    uint32_t height,
    const VizInjectorConfig *cfg,
    VizInjectorOutput *out_result
) {
    if (!rgba16 || width == 0u || height == 0u || !out_result) return -1;
    size_t needed = (size_t)width * (size_t)height * 4u;
    float *lin_buf = acquire_dll_tls_f32(needed);
    if (!lin_buf) return -2;

    int32_t role = cfg ? cfg->input_role : VIZ_ROLE_PHOTO_REF;
    if (role == VIZ_ROLE_DEPTH_MAP || role == VIZ_ROLE_INPAINT_MASK) {
        viz_ingest_scalar_rgba16_to_f32(rgba16, width, height, lin_buf);
    } else {
        viz_ingest_rgba16_to_linear_f32(rgba16, width, height, lin_buf);
    }
    return vizinjector_core_process_linear_f32(lin_buf, width, height, cfg, out_result);
}

VIZINJECTOR_API int32_t vizinjector_process_rgbaf32(
    const float *rgbaf32,
    uint32_t width,
    uint32_t height,
    int32_t input_is_linear,
    const VizInjectorConfig *cfg,
    VizInjectorOutput *out_result
) {
    if (!rgbaf32 || width == 0u || height == 0u || !out_result) return -1;
    size_t needed = (size_t)width * (size_t)height * 4u;
    float *lin_buf = acquire_dll_tls_f32(needed);
    if (!lin_buf) return -2;

    int32_t role = cfg ? cfg->input_role : VIZ_ROLE_PHOTO_REF;
    if (role == VIZ_ROLE_DEPTH_MAP || role == VIZ_ROLE_INPAINT_MASK) {
        clamp_f32_01_avx2(rgbaf32, needed, lin_buf);
        return vizinjector_core_process_linear_f32(lin_buf, width, height, cfg, out_result);
    }

    int lin_mode = 0;
    if (input_is_linear == 1) {
        lin_mode = 2;
    } else if (input_is_linear >= 2) {
        lin_mode = 1;
    }
    viz_ingest_rgbaf32_to_linear_f32(rgbaf32, width, height, lin_mode, lin_buf);
    return vizinjector_core_process_linear_f32(lin_buf, width, height, cfg, out_result);
}

VIZINJECTOR_API int32_t vizinjector_process_linear_f32(
    const float *linear_rgba,
    uint32_t width,
    uint32_t height,
    const VizInjectorConfig *cfg,
    VizInjectorOutput *out_result
) {
    return vizinjector_core_process_linear_f32(linear_rgba, width, height, cfg, out_result);
}

VIZINJECTOR_API int32_t vizinjector_postprocess_vae_to_rgb8(
    const float *planar_vae_rgb,
    uint32_t width,
    uint32_t height,
    int32_t apply_bayer_dither,
    uint8_t *out_rgb8,
    VizPerceptualTelemetry *out_telemetry
) {
    if (!planar_vae_rgb || width == 0 || height == 0) return -1;
    if (out_rgb8) {
        viz_postprocess_vae_f32_to_rgb8(planar_vae_rgb, width, height, apply_bayer_dither, out_rgb8);
    }
    if (out_telemetry) {
        viz_compute_perceptual_telemetry(planar_vae_rgb, width, height, out_telemetry);
    }
    return 0;
}

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
) {
    size_t sz = 0;
    int rc = viz_pack_vizj_container(
        hdr_template,
        vae_payload,
        (size_t)vae_bytes,
        vit_payload,
        (size_t)vit_bytes,
        mask_payload,
        (size_t)mask_bytes,
        out_bytes,
        &sz
    );
    if (out_len) *out_len = (uint64_t)sz;
    return (int32_t)rc;
}

VIZINJECTOR_API int32_t vizinjector_unpack_vizj_buffer(
    const uint8_t *container_bytes,
    uint64_t container_len,
    VizContainerHeader *out_hdr,
    const void **out_vae_ptr,
    const void **out_vit_ptr,
    const void **out_mask_ptr
) {
    return (int32_t)viz_unpack_vizj_container(
        container_bytes,
        (size_t)container_len,
        out_hdr,
        out_vae_ptr,
        out_vit_ptr,
        out_mask_ptr
    );
}

VIZINJECTOR_API void vizinjector_free_output(VizInjectorOutput *out_result) {
    vizinjector_core_free_output(out_result);
}

VIZINJECTOR_API void vizinjector_free_buffer(void *ptr) {
    if (ptr) free(ptr);
}