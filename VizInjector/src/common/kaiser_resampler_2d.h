#ifndef VIZ_KAISER_RESAMPLER_2D_H
#define VIZ_KAISER_RESAMPLER_2D_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VIZ_FRAMING_INSCRIBED  = 0,
    VIZ_FRAMING_CONTAIN    = 1,
    VIZ_FRAMING_STRETCH    = 2,
    VIZ_FRAMING_NEVER_CROP = 3
} VizFramingMode;

typedef struct {
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t max_taps;
    float scale;
    int is_identity;
    int32_t *start_idx;
    float *weights;
    float *src_center;
} VizPolyphaseSchedule1D;

double viz_bessel_i0(double x);

void viz_solve_coupled_grid(
    uint32_t src_w,
    uint32_t src_h,
    uint32_t req_w,
    uint32_t req_h,
    uint32_t max_vit_patches,
    int edit_mode_384_vit,
    VizFramingMode framing,
    uint32_t *out_vae_w,
    uint32_t *out_vae_h,
    uint32_t *out_vit_w,
    uint32_t *out_vit_h
);

int viz_build_polyphase_schedule_1d(
    uint32_t in_dim,
    uint32_t out_dim,
    float scale,
    float in_offset,
    float out_offset,
    float active_out_span,
    float mtf_boost,
    VizPolyphaseSchedule1D *out_sched
);

int viz_build_coupled_schedules_2d(
    uint32_t src_w,
    uint32_t src_h,
    uint32_t dst_w,
    uint32_t dst_h,
    VizFramingMode framing,
    float mtf_boost,
    VizPolyphaseSchedule1D *out_sched_x,
    VizPolyphaseSchedule1D *out_sched_y
);

void viz_free_polyphase_schedule_1d(VizPolyphaseSchedule1D *sched);

int viz_resample_2d_linear_rgba(
    const float *src_linear_rgba,
    uint32_t src_w,
    uint32_t src_h,
    uint32_t dst_w,
    uint32_t dst_h,
    VizFramingMode framing,
    float mtf_boost,
    float soft_knee_tau,
    float *out_linear_rgba
);

void viz_resample_majority_color_map_rgba8(
    const uint8_t *src_rgba8,
    uint32_t src_w,
    uint32_t src_h,
    uint32_t dst_w,
    uint32_t dst_h,
    VizFramingMode framing,
    float *out_planar_vae_rgb
);

#ifdef __cplusplus
}
#endif

#endif