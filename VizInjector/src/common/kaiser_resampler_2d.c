#include "kaiser_resampler_2d.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if defined(_MSC_VER)
#define VIZ_TLS __declspec(thread)
#else
#define VIZ_TLS __thread
#endif

static VIZ_TLS float *g_tls_resample_tmp = NULL;
static VIZ_TLS size_t g_tls_resample_cap_floats = 0;

static float *acquire_tls_scratch(size_t needed_floats) {
    if (needed_floats <= g_tls_resample_cap_floats && g_tls_resample_tmp != NULL) {
        return g_tls_resample_tmp;
    }
    size_t alloc_floats = (needed_floats + 4095u) & ~(size_t)4095u;
    float *next_buf = (float *)realloc(g_tls_resample_tmp, alloc_floats * sizeof(float));
    if (!next_buf) {
        return NULL;
    }
    g_tls_resample_tmp = next_buf;
    g_tls_resample_cap_floats = alloc_floats;
    return g_tls_resample_tmp;
}

double viz_bessel_i0(double x) {
    double sum = 1.0;
    double term = 1.0;
    double half_x = 0.5 * x;
    double half_x_sq = half_x * half_x;
    for (int k = 1; k < 64; ++k) {
        term *= half_x_sq / ((double)k * (double)k);
        sum += term;
        if (term < sum * 1e-15) {
            break;
        }
    }
    return sum;
}

static inline double clean_sinc(double x) {
    double ax = fabs(x);
    if (ax < 1e-8) return 1.0;
    double n = floor(x + 0.5);
    if (fabs(x - n) < 1e-8) return 0.0;
    double px = M_PI * x;
    return sin(px) / px;
}

static inline double mitchell_netravali(double x) {
    double ax = fabs(x);
    if (ax < 1.0) {
        return ((7.0 * ax - 12.0) * ax * ax + (16.0 / 3.0)) * (1.0 / 6.0);
    }
    if (ax < 2.0) {
        return (((-7.0 / 3.0) * ax + 12.0) * ax - 20.0) * ax * (1.0 / 6.0) + (16.0 / 9.0);
    }
    return 0.0;
}

static uint32_t snap_multiple(float val, uint32_t mult, uint32_t min_v, uint32_t max_v) {
    uint32_t snapped = ((uint32_t)floorf(val / (float)mult + 0.5f)) * mult;
    if (snapped < min_v) snapped = min_v;
    if (snapped > max_v) snapped = (max_v / mult) * mult;
    return snapped;
}

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
) {
    if (src_w == 0) src_w = 1024u;
    if (src_h == 0) src_h = 1024u;
    float aspect = (float)src_w / (float)src_h;

    uint32_t vae_w = 1024u;
    uint32_t vae_h = 1024u;

    if (framing == VIZ_FRAMING_NEVER_CROP) {
        uint32_t target_area = (req_w > 0 && req_h > 0) ? (req_w * req_h) : (1024u * 1024u);
        float raw_h = sqrtf((float)target_area / aspect);
        float raw_w = raw_h * aspect;
        vae_w = snap_multiple(raw_w, 16u, 64u, 2048u);
        vae_h = snap_multiple(raw_h, 16u, 64u, 2048u);
    } else {
        vae_w = snap_multiple((float)(req_w > 0 ? req_w : src_w), 16u, 16u, 4096u);
        vae_h = snap_multiple((float)(req_h > 0 ? req_h : src_h), 16u, 16u, 4096u);
    }

    uint32_t vit_w = 384u;
    uint32_t vit_h = 384u;

    if (edit_mode_384_vit) {
        if (src_w >= src_h) {
            vit_w = 384u;
            vit_h = snap_multiple(384.0f / aspect, 32u, 64u, 384u);
        } else {
            vit_h = 384u;
            vit_w = snap_multiple(384.0f * aspect, 32u, 64u, 384u);
        }
    } else {
        uint32_t max_px = (max_vit_patches > 0 ? max_vit_patches : 2304u) * 256u;
        float area_in = (float)src_w * (float)src_h;
        float scale = 1.0f;
        if (area_in > (float)max_px) {
            scale = sqrtf((float)max_px / area_in);
        }
        vit_w = snap_multiple((float)src_w * scale, 32u, 32u, 2048u);
        vit_h = snap_multiple((float)src_h * scale, 32u, 32u, 2048u);
        while ((vit_w / 16u) * (vit_h / 16u) > max_vit_patches && vit_w > 32u && vit_h > 32u) {
            if (vit_w >= vit_h) vit_w -= 32u;
            else vit_h -= 32u;
        }
    }

    if (out_vae_w) *out_vae_w = vae_w;
    if (out_vae_h) *out_vae_h = vae_h;
    if (out_vit_w) *out_vit_w = vit_w;
    if (out_vit_h) *out_vit_h = vit_h;
}

int viz_build_polyphase_schedule_1d(
    uint32_t in_dim,
    uint32_t out_dim,
    float scale,
    float in_offset,
    float out_offset,
    float active_out_span,
    float mtf_boost,
    VizPolyphaseSchedule1D *out_sched
) {
    if (!out_sched || in_dim == 0u || out_dim == 0u) return -1;
    memset(out_sched, 0, sizeof(*out_sched));

    int is_ident = (in_dim == out_dim &&
                    fabsf(scale - 1.0f) < 1e-5f &&
                    fabsf(in_offset) < 1e-5f &&
                    fabsf(out_offset) < 1e-5f &&
                    fabsf(active_out_span - (float)out_dim) < 1e-3f &&
                    fabsf(mtf_boost) < 1e-5f);

    uint32_t max_taps = 1u;
    double support = 0.5;
    double delta = (double)scale;
    double cutoff = 1.0;
    const double kaiser_a = 4.0;
    const double kaiser_beta = 8.5;
    double inv_i0_beta = 1.0 / viz_bessel_i0(kaiser_beta);

    if (!is_ident) {
        if (delta >= 1.0) {
            double bw = (mtf_boost > 0.0f) ? fmin(1.32, 0.92 + 3.8 * (double)mtf_boost) : 0.92;
            cutoff = fmin(1.0, bw / delta);
            support = kaiser_a * delta;
        } else {
            support = 2.0;
        }
        uint32_t raw_taps = (uint32_t)ceil(2.0 * support) + 2u;
        max_taps = (raw_taps + 3u) & ~3u;
        if (max_taps > 64u) max_taps = 64u;
    }

    int32_t *start_idx = (int32_t *)calloc(out_dim, sizeof(int32_t));
    float *weights = (float *)calloc((size_t)out_dim * max_taps, sizeof(float));
    float *src_center = (float *)calloc(out_dim, sizeof(float));
    if (!start_idx || !weights || !src_center) {
        free(start_idx);
        free(weights);
        free(src_center);
        return -2;
    }

    if (is_ident) {
        for (uint32_t d = 0; d < out_dim; ++d) {
            start_idx[d] = (int32_t)d;
            weights[d] = 1.0f;
            src_center[d] = (float)d;
        }
        out_sched->in_dim = in_dim;
        out_sched->out_dim = out_dim;
        out_sched->max_taps = 1u;
        out_sched->scale = 1.0f;
        out_sched->is_identity = 1;
        out_sched->start_idx = start_idx;
        out_sched->weights = weights;
        out_sched->src_center = src_center;
        return 0;
    }

    for (uint32_t d = 0; d < out_dim; ++d) {
        float d_center = (float)d + 0.5f;
        float *w_row = weights + (size_t)d * max_taps;
        if (d_center < out_offset || d_center > (out_offset + active_out_span)) {
            start_idx[d] = 0;
            src_center[d] = -1.0f;
            continue;
        }

        double s_center = (double)in_offset + ((double)d_center - (double)out_offset) * delta - 0.5;
        src_center[d] = (float)s_center;

        int32_t ideal_start = (int32_t)floor(s_center - support) + 1;
        int32_t clamped_start = ideal_start;
        if (in_dim >= max_taps) {
            if (clamped_start < 0) clamped_start = 0;
            if (clamped_start + (int32_t)max_taps > (int32_t)in_dim) {
                clamped_start = (int32_t)in_dim - (int32_t)max_taps;
            }
        } else {
            clamped_start = 0;
        }
        start_idx[d] = clamped_start;

        double accum[64];
        memset(accum, 0, sizeof(accum));
        double w_sum = 0.0;

        for (uint32_t t = 0; t < max_taps; ++t) {
            int32_t s = ideal_start + (int32_t)t;
            double dist = (double)s - s_center;
            if (fabs(dist) >= support) continue;

            double w = 0.0;
            if (delta >= 1.0) {
                double ratio = dist / support;
                double win = viz_bessel_i0(kaiser_beta * sqrt(fmax(0.0, 1.0 - ratio * ratio))) * inv_i0_beta;
                double x_c = dist * cutoff;
                double sinc_v = clean_sinc(x_c);
                if (mtf_boost > 0.0f) {
                    double axc = fabs(x_c);
                    if (axc > 0.0 && axc < 1.0) {
                        sinc_v *= fmax(0.15, 1.0 - 2.2 * (double)mtf_boost * (axc * axc));
                    } else if (axc >= 1.0 && axc < 2.2) {
                        sinc_v *= (1.0 + 5.5 * (double)mtf_boost * (axc - 1.0));
                    }
                }
                w = sinc_v * win;
            } else {
                w = mitchell_netravali(dist);
            }

            int32_t eff_s = s;
            if (eff_s < 0) eff_s = 0;
            if (eff_s >= (int32_t)in_dim) eff_s = (int32_t)in_dim - 1;
            int32_t slot = eff_s - clamped_start;
            if (slot < 0) slot = 0;
            if (slot >= (int32_t)max_taps) slot = (int32_t)max_taps - 1;
            accum[slot] += w;
            w_sum += w;
        }

        if (fabs(w_sum) > 1e-12) {
            double inv_sum = 1.0 / w_sum;
            for (uint32_t k = 0; k < max_taps; ++k) {
                w_row[k] = (float)(accum[k] * inv_sum);
            }
        }
    }

    out_sched->in_dim = in_dim;
    out_sched->out_dim = out_dim;
    out_sched->max_taps = max_taps;
    out_sched->scale = scale;
    out_sched->is_identity = 0;
    out_sched->start_idx = start_idx;
    out_sched->weights = weights;
    out_sched->src_center = src_center;
    return 0;
}

int viz_build_coupled_schedules_2d(
    uint32_t src_w,
    uint32_t src_h,
    uint32_t dst_w,
    uint32_t dst_h,
    VizFramingMode framing,
    float mtf_boost,
    VizPolyphaseSchedule1D *out_sched_x,
    VizPolyphaseSchedule1D *out_sched_y
) {
    float scale_x = (float)src_w / (float)dst_w;
    float scale_y = (float)src_h / (float)dst_h;
    float in_off_x = 0.0f;
    float in_off_y = 0.0f;
    float out_off_x = 0.0f;
    float out_off_y = 0.0f;
    float span_x = (float)dst_w;
    float span_y = (float)dst_h;

    if (framing == VIZ_FRAMING_INSCRIBED || framing == VIZ_FRAMING_NEVER_CROP) {
        float fill_scale = fmaxf((float)dst_w / (float)src_w, (float)dst_h / (float)src_h);
        scale_x = 1.0f / fill_scale;
        scale_y = 1.0f / fill_scale;
        float vis_w = (float)dst_w * scale_x;
        float vis_h = (float)dst_h * scale_y;
        in_off_x = 0.5f * ((float)src_w - vis_w);
        in_off_y = 0.5f * ((float)src_h - vis_h);
    } else if (framing == VIZ_FRAMING_CONTAIN) {
        float fit_scale = fminf((float)dst_w / (float)src_w, (float)dst_h / (float)src_h);
        scale_x = 1.0f / fit_scale;
        scale_y = 1.0f / fit_scale;
        span_x = (float)src_w * fit_scale;
        span_y = (float)src_h * fit_scale;
        out_off_x = 0.5f * ((float)dst_w - span_x);
        out_off_y = 0.5f * ((float)dst_h - span_y);
    }

    if (viz_build_polyphase_schedule_1d(src_w, dst_w, scale_x, in_off_x, out_off_x, span_x, mtf_boost, out_sched_x) != 0) {
        return -1;
    }
    if (viz_build_polyphase_schedule_1d(src_h, dst_h, scale_y, in_off_y, out_off_y, span_y, mtf_boost, out_sched_y) != 0) {
        viz_free_polyphase_schedule_1d(out_sched_x);
        return -1;
    }
    return 0;
}

void viz_free_polyphase_schedule_1d(VizPolyphaseSchedule1D *sched) {
    if (!sched) return;
    free(sched->start_idx);
    free(sched->weights);
    free(sched->src_center);
    memset(sched, 0, sizeof(*sched));
}

static inline float soft_knee_bound_1d(float v, float lo, float hi, float tau) {
    float span = hi - lo;
    float eff_tau = tau * span;
    if (v > hi) {
        if (eff_tau <= 1e-7f) return hi;
        float z = (v - hi) / eff_tau;
        float t = (z >= 3.0f) ? 1.0f : (z * (27.0f + z * z) / (27.0f + 9.0f * z * z));
        return hi + eff_tau * t;
    }
    if (v < lo) {
        if (eff_tau <= 1e-7f) return lo;
        float z = (lo - v) / eff_tau;
        float t = (z >= 3.0f) ? 1.0f : (z * (27.0f + z * z) / (27.0f + 9.0f * z * z));
        return lo - eff_tau * t;
    }
    return v;
}

static inline __m128 finalize_rgba_128(__m128 v_acc, __m128 v_raw, __m128 p0, __m128 p1, __m128 v_tau) {
    __m128 a_bcast = _mm_shuffle_ps(v_acc, v_acc, _MM_SHUFFLE(3, 3, 3, 3));
    __m128 has_a = _mm_cmpgt_ps(a_bcast, _mm_set1_ps(1e-5f));
    __m128 rgb_div = _mm_div_ps(v_acc, _mm_max_ps(a_bcast, _mm_set1_ps(1e-5f)));
    __m128 v = _mm_blendv_ps(v_raw, rgb_div, has_a);

    __m128 lo = _mm_min_ps(p0, p1);
    __m128 hi = _mm_max_ps(p0, p1);
    __m128 span = _mm_sub_ps(hi, lo);
    __m128 eff_tau = _mm_mul_ps(v_tau, span);
    __m128 valid_tau = _mm_cmpgt_ps(eff_tau, _mm_set1_ps(1e-7f));
    __m128 eff_tau_masked = _mm_and_ps(eff_tau, valid_tau);
    __m128 safe_tau = _mm_max_ps(eff_tau, _mm_set1_ps(1e-7f));

    __m128 c27 = _mm_set1_ps(27.0f);
    __m128 c9 = _mm_set1_ps(9.0f);
    __m128 c3 = _mm_set1_ps(3.0f);

    __m128 z_hi = _mm_min_ps(c3, _mm_div_ps(_mm_sub_ps(v, hi), safe_tau));
    __m128 z2_hi = _mm_mul_ps(z_hi, z_hi);
    __m128 t_hi = _mm_div_ps(_mm_mul_ps(z_hi, _mm_add_ps(c27, z2_hi)), _mm_fmadd_ps(c9, z2_hi, c27));
    __m128 res_hi = _mm_fmadd_ps(eff_tau_masked, t_hi, hi);

    __m128 z_lo = _mm_min_ps(c3, _mm_div_ps(_mm_sub_ps(lo, v), safe_tau));
    __m128 z2_lo = _mm_mul_ps(z_lo, z_lo);
    __m128 t_lo = _mm_div_ps(_mm_mul_ps(z_lo, _mm_add_ps(c27, z2_lo)), _mm_fmadd_ps(c9, z2_lo, c27));
    __m128 res_lo = _mm_fnmadd_ps(eff_tau_masked, t_lo, lo);

    __m128 out = _mm_blendv_ps(v, res_hi, _mm_cmpgt_ps(v, hi));
    out = _mm_blendv_ps(out, res_lo, _mm_cmplt_ps(v, lo));
    out = _mm_blend_ps(out, v_acc, 0x8);
    return _mm_min_ps(_mm_set1_ps(1.0f), _mm_max_ps(_mm_setzero_ps(), out));
}

static inline __m256 finalize_rgba_256(__m256 v_acc, __m256 v_raw, __m256 p0, __m256 p1, __m256 v_tau) {
    __m256 a_bcast = _mm256_permute_ps(v_acc, _MM_SHUFFLE(3, 3, 3, 3));
    __m256 has_a = _mm256_cmp_ps(a_bcast, _mm256_set1_ps(1e-5f), _CMP_GT_OQ);
    __m256 rgb_div = _mm256_div_ps(v_acc, _mm256_max_ps(a_bcast, _mm256_set1_ps(1e-5f)));
    __m256 v = _mm256_blendv_ps(v_raw, rgb_div, has_a);

    __m256 lo = _mm256_min_ps(p0, p1);
    __m256 hi = _mm256_max_ps(p0, p1);
    __m256 span = _mm256_sub_ps(hi, lo);
    __m256 eff_tau = _mm256_mul_ps(v_tau, span);
    __m256 valid_tau = _mm256_cmp_ps(eff_tau, _mm256_set1_ps(1e-7f), _CMP_GT_OQ);
    __m256 eff_tau_masked = _mm256_and_ps(eff_tau, valid_tau);
    __m256 safe_tau = _mm256_max_ps(eff_tau, _mm256_set1_ps(1e-7f));

    __m256 c27 = _mm256_set1_ps(27.0f);
    __m256 c9 = _mm256_set1_ps(9.0f);
    __m256 c3 = _mm256_set1_ps(3.0f);

    __m256 z_hi = _mm256_min_ps(c3, _mm256_div_ps(_mm256_sub_ps(v, hi), safe_tau));
    __m256 z2_hi = _mm256_mul_ps(z_hi, z_hi);
    __m256 t_hi = _mm256_div_ps(_mm256_mul_ps(z_hi, _mm256_add_ps(c27, z2_hi)), _mm256_fmadd_ps(c9, z2_hi, c27));
    __m256 res_hi = _mm256_fmadd_ps(eff_tau_masked, t_hi, hi);

    __m256 z_lo = _mm256_min_ps(c3, _mm256_div_ps(_mm256_sub_ps(lo, v), safe_tau));
    __m256 z2_lo = _mm256_mul_ps(z_lo, z_lo);
    __m256 t_lo = _mm256_div_ps(_mm256_mul_ps(z_lo, _mm256_add_ps(c27, z2_lo)), _mm256_fmadd_ps(c9, z2_lo, c27));
    __m256 res_lo = _mm256_fnmadd_ps(eff_tau_masked, t_lo, lo);

    __m256 out = _mm256_blendv_ps(v, res_hi, _mm256_cmp_ps(v, hi, _CMP_GT_OQ));
    out = _mm256_blendv_ps(out, res_lo, _mm256_cmp_ps(v, lo, _CMP_LT_OQ));
    out = _mm256_blend_ps(out, v_acc, 0x88);
    return _mm256_min_ps(_mm256_set1_ps(1.0f), _mm256_max_ps(_mm256_setzero_ps(), out));
}

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
) {
    if (!src_linear_rgba || !out_linear_rgba || src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0) {
        return -1;
    }

    VizPolyphaseSchedule1D sx, sy;
    if (viz_build_coupled_schedules_2d(src_w, src_h, dst_w, dst_h, framing, mtf_boost, &sx, &sy) != 0) {
        return -2;
    }

    if (sx.is_identity && sy.is_identity) {
        memcpy(out_linear_rgba, src_linear_rgba, (size_t)dst_w * (size_t)dst_h * 4u * sizeof(float));
        viz_free_polyphase_schedule_1d(&sx);
        viz_free_polyphase_schedule_1d(&sy);
        return 0;
    }

    size_t tmp_floats = (size_t)src_h * (size_t)dst_w * 4u;
    int owns_tmp = 0;
    float *tmp = acquire_tls_scratch(tmp_floats);
    if (!tmp) {
        tmp = (float *)malloc(tmp_floats * sizeof(float));
        if (!tmp) {
            viz_free_polyphase_schedule_1d(&sx);
            viz_free_polyphase_schedule_1d(&sy);
            return -3;
        }
        owns_tmp = 1;
    }

    const uint32_t taps_x = sx.max_taps;
    const uint32_t taps_y = sy.max_taps;
    const __m128 v_one128 = _mm_set1_ps(1.0f);
    const __m256 v_one256 = _mm256_set1_ps(1.0f);
    const __m128 v_tau128 = _mm_set1_ps(soft_knee_tau);
    const __m256 v_tau256 = _mm256_set1_ps(soft_knee_tau);
    const int fast_contig_x = (src_w >= taps_x && taps_x >= 4u && (taps_x & 3u) == 0u);

    if (sx.is_identity) {
        memcpy(tmp, src_linear_rgba, tmp_floats * sizeof(float));
    } else {
        for (uint32_t y = 0; y < src_h; ++y) {
            const float *src_row = src_linear_rgba + (size_t)y * (size_t)src_w * 4u;
            float *tmp_row = tmp + (size_t)y * (size_t)dst_w * 4u;

            for (uint32_t dx = 0; dx < dst_w; ++dx) {
                float sc = sx.src_center[dx];
                float *dst_px = tmp_row + (size_t)dx * 4u;
                if (sc < -0.5f) {
                    _mm_storeu_ps(dst_px, _mm_setzero_ps());
                    continue;
                }

                int32_t s0 = sx.start_idx[dx];
                const float *w = sx.weights + (size_t)dx * taps_x;
                __m128 v_acc;
                __m128 v_raw;

                if (fast_contig_x) {
                    const float *s_ptr = src_row + (size_t)s0 * 4u;
                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 raw0 = _mm256_setzero_ps();
                    __m256 raw1 = _mm256_setzero_ps();

                    for (uint32_t k = 0; k < taps_x; k += 4u) {
                        __m256 p01 = _mm256_loadu_ps(s_ptr + (size_t)k * 4u);
                        __m256 p23 = _mm256_loadu_ps(s_ptr + (size_t)(k + 2u) * 4u);

                        __m256 wk01 = _mm256_set_m128(_mm_set1_ps(w[k + 1u]), _mm_set1_ps(w[k]));
                        __m256 wk23 = _mm256_set_m128(_mm_set1_ps(w[k + 3u]), _mm_set1_ps(w[k + 2u]));

                        __m256 a01 = _mm256_permute_ps(p01, _MM_SHUFFLE(3, 3, 3, 3));
                        __m256 a23 = _mm256_permute_ps(p23, _MM_SHUFFLE(3, 3, 3, 3));

                        __m256 wa01 = _mm256_mul_ps(wk01, a01);
                        __m256 wa23 = _mm256_mul_ps(wk23, a23);

                        __m256 u01 = _mm256_blend_ps(p01, v_one256, 0x88);
                        __m256 u23 = _mm256_blend_ps(p23, v_one256, 0x88);

                        acc0 = _mm256_fmadd_ps(u01, wa01, acc0);
                        acc1 = _mm256_fmadd_ps(u23, wa23, acc1);
                        raw0 = _mm256_fmadd_ps(p01, wk01, raw0);
                        raw1 = _mm256_fmadd_ps(p23, wk23, raw1);
                    }

                    acc0 = _mm256_add_ps(acc0, acc1);
                    raw0 = _mm256_add_ps(raw0, raw1);
                    v_acc = _mm_add_ps(_mm256_castps256_ps128(acc0), _mm256_extractf128_ps(acc0, 1));
                    v_raw = _mm_add_ps(_mm256_castps256_ps128(raw0), _mm256_extractf128_ps(raw0, 1));
                } else {
                    v_acc = _mm_setzero_ps();
                    v_raw = _mm_setzero_ps();
                    for (uint32_t k = 0; k < taps_x; ++k) {
                        int32_t sx_idx = s0 + (int32_t)k;
                        if (sx_idx < 0) sx_idx = 0;
                        if (sx_idx >= (int32_t)src_w) sx_idx = (int32_t)src_w - 1;
                        __m128 p = _mm_loadu_ps(src_row + (size_t)sx_idx * 4u);
                        __m128 wk = _mm_set1_ps(w[k]);
                        __m128 a_bcast = _mm_shuffle_ps(p, p, _MM_SHUFFLE(3, 3, 3, 3));
                        __m128 wa = _mm_mul_ps(wk, a_bcast);
                        __m128 p_unit_a = _mm_blend_ps(p, v_one128, 0x8);
                        v_acc = _mm_fmadd_ps(p_unit_a, wa, v_acc);
                        v_raw = _mm_fmadd_ps(p, wk, v_raw);
                    }
                }

                int32_t n0 = (int32_t)floorf(sc);
                int32_t n1 = n0 + 1;
                if (n0 < 0) n0 = 0;
                if (n0 >= (int32_t)src_w) n0 = (int32_t)src_w - 1;
                if (n1 < 0) n1 = 0;
                if (n1 >= (int32_t)src_w) n1 = (int32_t)src_w - 1;

                __m128 p0 = _mm_loadu_ps(src_row + (size_t)n0 * 4u);
                __m128 p1 = _mm_loadu_ps(src_row + (size_t)n1 * 4u);
                _mm_storeu_ps(dst_px, finalize_rgba_128(v_acc, v_raw, p0, p1, v_tau128));
            }
        }
    }

    uint32_t active_dx_start = 0u;
    uint32_t active_dx_end = dst_w;
    while (active_dx_start < dst_w && sx.src_center[active_dx_start] < -0.5f) {
        active_dx_start++;
    }
    while (active_dx_end > active_dx_start && sx.src_center[active_dx_end - 1u] < -0.5f) {
        active_dx_end--;
    }

    const float *tap_rows[64];

    for (uint32_t dy = 0; dy < dst_h; ++dy) {
        float sc = sy.src_center[dy];
        float *dst_row = out_linear_rgba + (size_t)dy * (size_t)dst_w * 4u;
        if (sc < -0.5f) {
            memset(dst_row, 0, (size_t)dst_w * 4u * sizeof(float));
            continue;
        }

        if (active_dx_start > 0u) {
            memset(dst_row, 0, (size_t)active_dx_start * 4u * sizeof(float));
        }
        if (active_dx_end < dst_w) {
            memset(dst_row + (size_t)active_dx_end * 4u, 0, (size_t)(dst_w - active_dx_end) * 4u * sizeof(float));
        }

        int32_t s0 = sy.start_idx[dy];
        const float *w = sy.weights + (size_t)dy * taps_y;
        for (uint32_t k = 0; k < taps_y; ++k) {
            int32_t sy_idx = s0 + (int32_t)k;
            if (sy_idx < 0) sy_idx = 0;
            if (sy_idx >= (int32_t)src_h) sy_idx = (int32_t)src_h - 1;
            tap_rows[k] = tmp + (size_t)sy_idx * (size_t)dst_w * 4u;
        }

        int32_t n0 = (int32_t)floorf(sc);
        int32_t n1 = n0 + 1;
        if (n0 < 0) n0 = 0;
        if (n0 >= (int32_t)src_h) n0 = (int32_t)src_h - 1;
        if (n1 < 0) n1 = 0;
        if (n1 >= (int32_t)src_h) n1 = (int32_t)src_h - 1;
        const float *row_n0 = tmp + (size_t)n0 * (size_t)dst_w * 4u;
        const float *row_n1 = tmp + (size_t)n1 * (size_t)dst_w * 4u;

        uint32_t dx = active_dx_start;
        for (; dx + 4u <= active_dx_end; dx += 4u) {
            size_t off0 = (size_t)dx * 4u;
            size_t off1 = off0 + 8u;

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 raw0 = _mm256_setzero_ps();
            __m256 raw1 = _mm256_setzero_ps();

            for (uint32_t k = 0; k < taps_y; ++k) {
                const float *r_ptr = tap_rows[k];
                __m256 wk = _mm256_set1_ps(w[k]);

                __m256 p0 = _mm256_loadu_ps(r_ptr + off0);
                __m256 p1 = _mm256_loadu_ps(r_ptr + off1);

                __m256 a0 = _mm256_permute_ps(p0, _MM_SHUFFLE(3, 3, 3, 3));
                __m256 a1 = _mm256_permute_ps(p1, _MM_SHUFFLE(3, 3, 3, 3));

                __m256 wa0 = _mm256_mul_ps(wk, a0);
                __m256 wa1 = _mm256_mul_ps(wk, a1);

                __m256 u0 = _mm256_blend_ps(p0, v_one256, 0x88);
                __m256 u1 = _mm256_blend_ps(p1, v_one256, 0x88);

                acc0 = _mm256_fmadd_ps(u0, wa0, acc0);
                acc1 = _mm256_fmadd_ps(u1, wa1, acc1);
                raw0 = _mm256_fmadd_ps(p0, wk, raw0);
                raw1 = _mm256_fmadd_ps(p1, wk, raw1);
            }

            __m256 p0_0 = _mm256_loadu_ps(row_n0 + off0);
            __m256 p1_0 = _mm256_loadu_ps(row_n1 + off0);
            __m256 p0_1 = _mm256_loadu_ps(row_n0 + off1);
            __m256 p1_1 = _mm256_loadu_ps(row_n1 + off1);

            _mm256_storeu_ps(dst_row + off0, finalize_rgba_256(acc0, raw0, p0_0, p1_0, v_tau256));
            _mm256_storeu_ps(dst_row + off1, finalize_rgba_256(acc1, raw1, p0_1, p1_1, v_tau256));
        }

        for (; dx < active_dx_end; ++dx) {
            size_t off = (size_t)dx * 4u;
            if (sx.src_center[dx] < -0.5f) {
                _mm_storeu_ps(dst_row + off, _mm_setzero_ps());
                continue;
            }
            __m128 v_acc = _mm_setzero_ps();
            __m128 v_raw = _mm_setzero_ps();
            for (uint32_t k = 0; k < taps_y; ++k) {
                __m128 p = _mm_loadu_ps(tap_rows[k] + off);
                __m128 wk = _mm_set1_ps(w[k]);
                __m128 a_bcast = _mm_shuffle_ps(p, p, _MM_SHUFFLE(3, 3, 3, 3));
                __m128 wa = _mm_mul_ps(wk, a_bcast);
                __m128 p_unit_a = _mm_blend_ps(p, v_one128, 0x8);
                v_acc = _mm_fmadd_ps(p_unit_a, wa, v_acc);
                v_raw = _mm_fmadd_ps(p, wk, v_raw);
            }
            __m128 p0 = _mm_loadu_ps(row_n0 + off);
            __m128 p1 = _mm_loadu_ps(row_n1 + off);
            _mm_storeu_ps(dst_row + off, finalize_rgba_128(v_acc, v_raw, p0, p1, v_tau128));
        }
    }

    if (owns_tmp) {
        free(tmp);
    }
    viz_free_polyphase_schedule_1d(&sx);
    viz_free_polyphase_schedule_1d(&sy);
    return 0;
}

void viz_resample_majority_color_map_rgba8(
    const uint8_t *src_rgba8,
    uint32_t src_w,
    uint32_t src_h,
    uint32_t dst_w,
    uint32_t dst_h,
    VizFramingMode framing,
    float *out_planar_vae_rgb
) {
    (void)framing;
    size_t plane = (size_t)dst_w * (size_t)dst_h;
    float *r_p = out_planar_vae_rgb;
    float *g_p = out_planar_vae_rgb + plane;
    float *b_p = out_planar_vae_rgb + plane * 2u;

    float scale_x = (float)src_w / (float)dst_w;
    float scale_y = (float)src_h / (float)dst_h;

    for (uint32_t dy = 0; dy < dst_h; ++dy) {
        int y0 = (int)floorf((float)dy * scale_y);
        int y1 = (int)ceilf((float)(dy + 1u) * scale_y);
        if (y0 < 0) y0 = 0;
        if (y1 <= y0) y1 = y0 + 1;
        if (y1 > (int)src_h) y1 = (int)src_h;

        for (uint32_t dx = 0; dx < dst_w; ++dx) {
            int x0 = (int)floorf((float)dx * scale_x);
            int x1 = (int)ceilf((float)(dx + 1u) * scale_x);
            if (x0 < 0) x0 = 0;
            if (x1 <= x0) x1 = x0 + 1;
            if (x1 > (int)src_w) x1 = (int)src_w;

            uint32_t cand_rgb[16];
            uint32_t cand_cnt[16];
            int n_cand = 0;

            for (int sy = y0; sy < y1; ++sy) {
                for (int sx = x0; sx < x1; ++sx) {
                    const uint8_t *p = src_rgba8 + ((size_t)sy * src_w + (size_t)sx) * 4u;
                    uint32_t packed = ((uint32_t)p[0] << 16) | ((uint32_t)p[1] << 8) | (uint32_t)p[2];
                    int found = -1;
                    for (int c = 0; c < n_cand; ++c) {
                        if (cand_rgb[c] == packed) {
                            found = c;
                            break;
                        }
                    }
                    if (found >= 0) {
                        cand_cnt[found]++;
                    } else if (n_cand < 16) {
                        cand_rgb[n_cand] = packed;
                        cand_cnt[n_cand] = 1u;
                        n_cand++;
                    }
                }
            }

            uint32_t best_packed = cand_rgb[0];
            uint32_t best_count = cand_cnt[0];
            for (int c = 1; c < n_cand; ++c) {
                if (cand_cnt[c] > best_count) {
                    best_count = cand_cnt[c];
                    best_packed = cand_rgb[c];
                }
            }

            size_t idx = (size_t)dy * dst_w + dx;
            float r = (float)((best_packed >> 16) & 0xFFu) * (1.0f / 255.0f);
            float g = (float)((best_packed >> 8) & 0xFFu) * (1.0f / 255.0f);
            float b = (float)(best_packed & 0xFFu) * (1.0f / 255.0f);

            r_p[idx] = r * 2.0f - 1.0f;
            g_p[idx] = g * 2.0f - 1.0f;
            b_p[idx] = b * 2.0f - 1.0f;
        }
    }
}