#ifndef VIZ_COLORSPACE_SRGB_H
#define VIZ_COLORSPACE_SRGB_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void viz_colorspace_init_tables(void);

float viz_srgb_to_linear_f32(float srgb_val);
float viz_linear_to_srgb_f32(float linear_val);

const float *viz_get_srgb_to_lin_lut(void);
const float *viz_get_lin_to_srgb_lut(void);

void viz_ingest_scalar_rgba8_to_f32(
    const uint8_t *src_rgba8,
    uint32_t width,
    uint32_t height,
    float *out_linear_rgba
);

void viz_ingest_rgba8_to_linear_f32(
    const uint8_t *src_rgba8,
    uint32_t width,
    uint32_t height,
    int deband_8bit,
    uint32_t seed,
    float *out_linear_rgba
);

void viz_ingest_rgba16_to_linear_f32(
    const uint16_t *src_rgba16,
    uint32_t width,
    uint32_t height,
    float *out_linear_rgba
);

void viz_ingest_rgbaf32_to_linear_f32(
    const float *src_rgbaf32,
    uint32_t width,
    uint32_t height,
    int input_is_linear,
    float *out_linear_rgba
);

int viz_parse_exif_orientation(const uint8_t *bytes, size_t len);

void viz_apply_exif_orientation_rgba8(
    const uint8_t *src_rgba,
    uint32_t width,
    uint32_t height,
    int orientation,
    uint8_t **out_rgba,
    uint32_t *out_w,
    uint32_t *out_h
);

#ifdef __cplusplus
}
#endif

#endif