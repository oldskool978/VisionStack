#ifndef VIZINJECTOR_PATCH_PACKER_H
#define VIZINJECTOR_PATCH_PACKER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float mean_luminance;
    float rms_contrast;
    float shannon_entropy_bits;
    float laplacian_sharpness;
} VizPerceptualTelemetry;

void viz_pack_vae_planar_f32(
    const float *linear_rgba,
    uint32_t width,
    uint32_t height,
    float *out_planar_rgb
);

void viz_pack_qwen35_vit_patches_f32(
    const float *linear_rgba,
    uint32_t vis_w,
    uint32_t vis_h,
    float *out_patches_5d,
    uint32_t out_grid_thw[3],
    uint32_t *out_llm_tokens
);

void viz_downsample_latent_mask_16x(
    const float *linear_rgba_or_mask,
    int input_stride_floats,
    int channel_offset,
    uint32_t vae_w,
    uint32_t vae_h,
    float feather_radius_cells,
    float *out_latent_mask
);

void viz_compute_perceptual_telemetry(
    const float *planar_vae_rgb,
    uint32_t width,
    uint32_t height,
    VizPerceptualTelemetry *out_telemetry
);

void viz_postprocess_vae_f32_to_rgb8(
    const float *planar_vae_rgb,
    uint32_t width,
    uint32_t height,
    int apply_bayer_dither,
    uint8_t *out_interleaved_rgb8
);

#ifdef __cplusplus
}
#endif

#endif