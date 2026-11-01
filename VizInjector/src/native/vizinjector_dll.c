#define VIZINJECTOR_BUILD_DLL 1
#include "vizinjector_dll.h"
#include "../common/colorspace_srgb.h"
#include "../vulkan/vk_compute_engine.h"
#include <stdlib.h>

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