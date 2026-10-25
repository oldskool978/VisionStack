#ifndef VIZ_VK_COMPUTE_ENGINE_H
#define VIZ_VK_COMPUTE_ENGINE_H

#include <stdint.h>
#include <stddef.h>
#include "../common/kaiser_resampler_2d.h"

#ifdef __cplusplus
extern "C" {
#endif

int viz_vk_init(void);
int viz_vk_is_available(void);
const char *viz_vk_device_name(void);
void viz_vk_shutdown(void);

int viz_vk_condition_image_rgba8(
    const uint8_t *src_rgba8,
    uint32_t src_w,
    uint32_t src_h,
    uint32_t vae_w,
    uint32_t vae_h,
    uint32_t vit_w,
    uint32_t vit_h,
    VizFramingMode framing,
    int deband_8bit,
    uint32_t seed,
    float mtf_boost,
    float soft_knee_tau,
    float mask_feather_radius,
    float *out_vae_planar_f32,
    float *out_vit_patches_f32,
    uint32_t out_grid_thw[3],
    uint32_t *out_llm_tokens,
    float *out_latent_mask_f32
);

#ifdef __cplusplus
}
#endif

#endif