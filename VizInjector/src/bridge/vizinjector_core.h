#ifndef VIZINJECTOR_CORE_H
#define VIZINJECTOR_CORE_H

#include <stdint.h>
#include <stddef.h>
#include "../common/kaiser_resampler_2d.h"
#include "../common/patch_packer.h"
#include "../common/viz_container.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VIZ_ROLE_PHOTO_REF    = 0,
    VIZ_ROLE_COLOR_MAP    = 1,
    VIZ_ROLE_DEPTH_MAP    = 2,
    VIZ_ROLE_INPAINT_MASK = 3
} VizInputRole;

typedef struct {
    uint32_t target_width;
    uint32_t target_height;
    uint32_t max_vit_patches;
    int32_t framing_mode;
    int32_t input_role;
    int32_t edit_mode_384_vit;
    int32_t emit_vae_branch;
    int32_t emit_vit_branch;
    int32_t emit_mask_branch;
    int32_t emit_wire_vizj;
    int32_t deband_8bit;
    int32_t prefer_vulkan;
    uint32_t seed;
    float mtf_boost;
    float soft_knee_tau;
    float mask_feather_radius;
    float webp_quality;
    int32_t webp_lossless;
} VizInjectorConfig;

typedef struct {
    float *vae_planar_f32;
    uint32_t vae_width;
    uint32_t vae_height;
    uint32_t vae_Floats;

    float *vit_patches_f32;
    uint32_t vit_width;
    uint32_t vit_height;
    uint32_t vit_num_patches;
    uint32_t vit_Floats;
    uint32_t grid_thw[3];
    uint32_t llm_tokens;

    float *latent_mask_f32;
    uint32_t mask_width;
    uint32_t mask_height;
    uint32_t mask_Floats;

    uint8_t *wire_vizj_bytes;
    uint32_t wire_vizj_len;

    uint32_t orig_width;
    uint32_t orig_height;

    float mean_luminance;
    float rms_contrast;
    float shannon_entropy_bits;
    float laplacian_sharpness;

    int32_t backend_used;
    int32_t status_code;
} VizInjectorOutput;

void vizinjector_core_default_config(VizInjectorConfig *cfg);

int32_t vizinjector_core_process_linear_f32(
    const float *src_linear_rgba,
    uint32_t src_w,
    uint32_t src_h,
    const VizInjectorConfig *cfg_in,
    VizInjectorOutput *out
);

int32_t vizinjector_core_process_rgba8(
    const uint8_t *src_rgba8,
    uint32_t src_w,
    uint32_t src_h,
    const VizInjectorConfig *cfg_in,
    VizInjectorOutput *out
);

int32_t vizinjector_core_process_memory(
    const uint8_t *file_bytes,
    size_t file_len,
    const VizInjectorConfig *cfg_in,
    VizInjectorOutput *out
);

void vizinjector_core_free_output(VizInjectorOutput *out);

#ifdef __cplusplus
}
#endif

#endif