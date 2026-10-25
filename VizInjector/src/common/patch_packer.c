#include "patch_packer.h"
#include "colorspace_srgb.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#define VIZ_PACK_AVX2 1
#else
#define VIZ_PACK_AVX2 0
#endif

#if defined(_MSC_VER)
#define VIZ_TLS __declspec(thread)
#else
#define VIZ_TLS __thread
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define VIZ_TELEM_MAX_STATIC_WIDTH 4096u
static VIZ_TLS float g_tls_luma_ring[3u * VIZ_TELEM_MAX_STATIC_WIDTH];

#if VIZ_PACK_AVX2
static inline void viz_deinterleave_rgba8_avx2(
    const float *base,
    __m256 *out_r,
    __m256 *out_g,
    __m256 *out_b
) {
    __m256 v0 = _mm256_loadu_ps(base + 0);
    __m256 v1 = _mm256_loadu_ps(base + 8);
    __m256 v2 = _mm256_loadu_ps(base + 16);
    __m256 v3 = _mm256_loadu_ps(base + 24);

    __m256 a0 = _mm256_permute2f128_ps(v0, v2, 0x20);
    __m256 a1 = _mm256_permute2f128_ps(v0, v2, 0x31);
    __m256 a2 = _mm256_permute2f128_ps(v1, v3, 0x20);
    __m256 a3 = _mm256_permute2f128_ps(v1, v3, 0x31);

    __m256 u0 = _mm256_unpacklo_ps(a0, a1);
    __m256 u1 = _mm256_unpacklo_ps(a2, a3);
    __m256 u2 = _mm256_unpackhi_ps(a0, a1);
    __m256 u3 = _mm256_unpackhi_ps(a2, a3);

    *out_r = _mm256_shuffle_ps(u0, u1, _MM_SHUFFLE(1, 0, 1, 0));
    *out_g = _mm256_shuffle_ps(u0, u1, _MM_SHUFFLE(3, 2, 3, 2));
    *out_b = _mm256_shuffle_ps(u2, u3, _MM_SHUFFLE(1, 0, 1, 0));
}

static inline __m256 viz_lin_to_srgb_avx2(__m256 v_lin, const float *lut) {
    __m256 v_zero = _mm256_setzero_ps();
    __m256 v_one = _mm256_set1_ps(1.0f);
    __m256 v_clamped = _mm256_min_ps(v_one, _mm256_max_ps(v_zero, v_lin));

    __m256 v_toe_thresh = _mm256_set1_ps(0.0031308f);
    __m256 v_toe = _mm256_mul_ps(v_clamped, _mm256_set1_ps(12.92f));

    __m256 v_scaled = _mm256_mul_ps(v_clamped, _mm256_set1_ps(65536.0f));
    __m256i v_idx = _mm256_cvttps_epi32(v_scaled);
    __m256 v_idx_f = _mm256_cvtepi32_ps(v_idx);
    __m256 v_frac = _mm256_sub_ps(v_scaled, v_idx_f);

    __m256i v_idx1 = _mm256_add_epi32(v_idx, _mm256_set1_epi32(1));
    __m256 v_y0 = _mm256_i32gather_ps(lut, v_idx, 4);
    __m256 v_y1 = _mm256_i32gather_ps(lut, v_idx1, 4);
    __m256 v_lut_val = _mm256_fmadd_ps(v_frac, _mm256_sub_ps(v_y1, v_y0), v_y0);

    __m256 v_is_toe = _mm256_cmp_ps(v_clamped, v_toe_thresh, _CMP_LE_OQ);
    return _mm256_blendv_ps(v_lut_val, v_toe, v_is_toe);
}
#endif

void viz_pack_vae_planar_f32(
    const float *linear_rgba,
    uint32_t width,
    uint32_t height,
    float *out_planar_rgb
) {
    size_t plane = (size_t)width * (size_t)height;
    float *r_dst = out_planar_rgb;
    float *g_dst = out_planar_rgb + plane;
    float *b_dst = out_planar_rgb + plane * 2u;

    size_t i = 0;
#if VIZ_PACK_AVX2
    const float *lut = viz_get_lin_to_srgb_lut();
    __m256 v_two = _mm256_set1_ps(2.0f);
    __m256 v_neg_one = _mm256_set1_ps(-1.0f);

    for (; i + 8u <= plane; i += 8u) {
        __m256 v_r_lin, v_g_lin, v_b_lin;
        viz_deinterleave_rgba8_avx2(linear_rgba + i * 4u, &v_r_lin, &v_g_lin, &v_b_lin);

        __m256 v_r_srgb = viz_lin_to_srgb_avx2(v_r_lin, lut);
        __m256 v_g_srgb = viz_lin_to_srgb_avx2(v_g_lin, lut);
        __m256 v_b_srgb = viz_lin_to_srgb_avx2(v_b_lin, lut);

        _mm256_storeu_ps(r_dst + i, _mm256_fmadd_ps(v_r_srgb, v_two, v_neg_one));
        _mm256_storeu_ps(g_dst + i, _mm256_fmadd_ps(v_g_srgb, v_two, v_neg_one));
        _mm256_storeu_ps(b_dst + i, _mm256_fmadd_ps(v_b_srgb, v_two, v_neg_one));
    }
#endif
    for (; i < plane; ++i) {
        float r_srgb = viz_linear_to_srgb_f32(linear_rgba[i * 4u + 0u]);
        float g_srgb = viz_linear_to_srgb_f32(linear_rgba[i * 4u + 1u]);
        float b_srgb = viz_linear_to_srgb_f32(linear_rgba[i * 4u + 2u]);
        r_dst[i] = r_srgb * 2.0f - 1.0f;
        g_dst[i] = g_srgb * 2.0f - 1.0f;
        b_dst[i] = b_srgb * 2.0f - 1.0f;
    }
}

void viz_pack_qwen35_vit_patches_f32(
    const float *linear_rgba,
    uint32_t vis_w,
    uint32_t vis_h,
    float *out_patches_5d,
    uint32_t out_grid_thw[3],
    uint32_t *out_llm_tokens
) {
    const float mean_r = 0.48145466f;
    const float mean_g = 0.4578275f;
    const float mean_b = 0.40821073f;
    const float inv_std_r = 1.0f / 0.26862954f;
    const float inv_std_g = 1.0f / 0.26130258f;
    const float inv_std_b = 1.0f / 0.27577711f;

    uint32_t grid_h = vis_h / 16u;
    uint32_t grid_w = vis_w / 16u;
    const uint32_t merge = 2u;

    out_grid_thw[0] = 1u;
    out_grid_thw[1] = grid_h;
    out_grid_thw[2] = grid_w;
    *out_llm_tokens = (grid_h * grid_w) / (merge * merge);

    size_t patch_stride = 3u * 2u * 256u;

#if VIZ_PACK_AVX2
    const float *lut = viz_get_lin_to_srgb_lut();
    __m256 v_mean_r = _mm256_set1_ps(mean_r);
    __m256 v_mean_g = _mm256_set1_ps(mean_g);
    __m256 v_mean_b = _mm256_set1_ps(mean_b);
    __m256 v_istd_r = _mm256_set1_ps(inv_std_r);
    __m256 v_istd_g = _mm256_set1_ps(inv_std_g);
    __m256 v_istd_b = _mm256_set1_ps(inv_std_b);
#endif

    for (uint32_t gh = 0; gh < grid_h; ++gh) {
        uint32_t gh_m = gh / merge;
        uint32_t m_y = gh % merge;
        for (uint32_t gw = 0; gw < grid_w; ++gw) {
            uint32_t gw_m = gw / merge;
            uint32_t m_x = gw % merge;
            uint32_t patch_idx = ((gh_m * (grid_w / merge) + gw_m) * merge + m_y) * merge + m_x;
            float *p_out = out_patches_5d + (size_t)patch_idx * patch_stride;

            float *r_t0 = p_out + (0u * 2u + 0u) * 256u;
            float *r_t1 = p_out + (0u * 2u + 1u) * 256u;
            float *g_t0 = p_out + (1u * 2u + 0u) * 256u;
            float *g_t1 = p_out + (1u * 2u + 1u) * 256u;
            float *b_t0 = p_out + (2u * 2u + 0u) * 256u;
            float *b_t1 = p_out + (2u * 2u + 1u) * 256u;

            for (uint32_t py = 0; py < 16u; ++py) {
                uint32_t y = gh * 16u + py;
                uint32_t sp_base = py * 16u;
                const float *row_in = linear_rgba + ((size_t)y * vis_w + (size_t)gw * 16u) * 4u;

#if VIZ_PACK_AVX2
                for (uint32_t px = 0; px < 16u; px += 8u) {
                    __m256 r_lin, g_lin, b_lin;
                    viz_deinterleave_rgba8_avx2(row_in + px * 4u, &r_lin, &g_lin, &b_lin);

                    __m256 vr = viz_lin_to_srgb_avx2(r_lin, lut);
                    __m256 vg = viz_lin_to_srgb_avx2(g_lin, lut);
                    __m256 vb = viz_lin_to_srgb_avx2(b_lin, lut);

                    vr = _mm256_mul_ps(_mm256_sub_ps(vr, v_mean_r), v_istd_r);
                    vg = _mm256_mul_ps(_mm256_sub_ps(vg, v_mean_g), v_istd_g);
                    vb = _mm256_mul_ps(_mm256_sub_ps(vb, v_mean_b), v_istd_b);

                    uint32_t off = sp_base + px;
                    _mm256_storeu_ps(r_t0 + off, vr);
                    _mm256_storeu_ps(r_t1 + off, vr);
                    _mm256_storeu_ps(g_t0 + off, vg);
                    _mm256_storeu_ps(g_t1 + off, vg);
                    _mm256_storeu_ps(b_t0 + off, vb);
                    _mm256_storeu_ps(b_t1 + off, vb);
                }
#else
                for (uint32_t px = 0; px < 16u; ++px) {
                    const float *src_p = row_in + px * 4u;
                    float r = (viz_linear_to_srgb_f32(src_p[0]) - mean_r) * inv_std_r;
                    float g = (viz_linear_to_srgb_f32(src_p[1]) - mean_g) * inv_std_g;
                    float b = (viz_linear_to_srgb_f32(src_p[2]) - mean_b) * inv_std_b;
                    uint32_t off = sp_base + px;
                    r_t0[off] = r;
                    r_t1[off] = r;
                    g_t0[off] = g;
                    g_t1[off] = g;
                    b_t0[off] = b;
                    b_t1[off] = b;
                }
#endif
            }
        }
    }
}

void viz_downsample_latent_mask_16x(
    const float *linear_rgba_or_mask,
    int input_stride_floats,
    int channel_offset,
    uint32_t vae_w,
    uint32_t vae_h,
    float feather_radius_cells,
    float *out_latent_mask
) {
    uint32_t lat_w = vae_w / 16u;
    uint32_t lat_h = vae_h / 16u;
    float *raw_mask = (feather_radius_cells > 0.05f)
        ? (float *)malloc((size_t)lat_w * (size_t)lat_h * sizeof(float))
        : NULL;
    float *target_buf = raw_mask ? raw_mask : out_latent_mask;

    for (uint32_t ly = 0; ly < lat_h; ++ly) {
        for (uint32_t lx = 0; lx < lat_w; ++lx) {
            float sum = 0.0f;
            float max_val = 0.0f;
            for (uint32_t dy = 0; dy < 16u; ++dy) {
                uint32_t py = ly * 16u + dy;
                const float *row_ptr = linear_rgba_or_mask + ((size_t)py * vae_w + (size_t)lx * 16u) * (size_t)input_stride_floats + (size_t)channel_offset;
                for (uint32_t dx = 0; dx < 16u; ++dx) {
                    float v = fminf(1.0f, fmaxf(0.0f, row_ptr[(size_t)dx * (size_t)input_stride_floats]));
                    sum += v;
                    if (v > max_val) max_val = v;
                }
            }
            float avg = sum * (1.0f / 256.0f);
            if (avg > 0.01f && avg < 0.25f && max_val > 0.75f) {
                avg = 0.65f * avg + 0.35f * max_val;
            }
            target_buf[(size_t)ly * lat_w + lx] = fminf(1.0f, fmaxf(0.0f, avg));
        }
    }

    if (raw_mask) {
        int rad = (int)ceilf(feather_radius_cells);
        if (rad > 6) rad = 6;
        for (uint32_t ly = 0; ly < lat_h; ++ly) {
            for (uint32_t lx = 0; lx < lat_w; ++lx) {
                float acc = 0.0f;
                float w_sum = 0.0f;
                for (int ky = -rad; ky <= rad; ++ky) {
                    int ny = (int)ly + ky;
                    if (ny < 0) ny = 0;
                    if (ny >= (int)lat_h) ny = (int)lat_h - 1;
                    for (int kx = -rad; kx <= rad; ++kx) {
                        int nx = (int)lx + kx;
                        if (nx < 0) nx = 0;
                        if (nx >= (int)lat_w) nx = (int)lat_w - 1;
                        float dist = sqrtf((float)(kx * kx + ky * ky));
                        if (dist <= feather_radius_cells) {
                            float w = 0.5f * (1.0f + cosf((float)M_PI * dist / feather_radius_cells));
                            acc += raw_mask[(size_t)ny * lat_w + (size_t)nx] * w;
                            w_sum += w;
                        }
                    }
                }
                out_latent_mask[(size_t)ly * lat_w + lx] = (w_sum > 0.0f) ? (acc / w_sum) : raw_mask[(size_t)ly * lat_w + lx];
            }
        }
        free(raw_mask);
    }
}

void viz_compute_perceptual_telemetry(
    const float *planar_vae_rgb,
    uint32_t width,
    uint32_t height,
    VizPerceptualTelemetry *out_telemetry
) {
    if (!planar_vae_rgb || !out_telemetry || width == 0 || height == 0) return;

    size_t plane = (size_t)width * (size_t)height;
    const float *r_p = planar_vae_rgb;
    const float *g_p = planar_vae_rgb + plane;
    const float *b_p = planar_vae_rgb + plane * 2u;

    int owns_ring = 0;
    float *luma_ring = g_tls_luma_ring;
    if (width > VIZ_TELEM_MAX_STATIC_WIDTH) {
        luma_ring = (float *)malloc((size_t)width * 3u * sizeof(float));
        if (!luma_ring) return;
        owns_ring = 1;
    }

    uint32_t hist[256];
    memset(hist, 0, sizeof(hist));

    double sum_luma = 0.0;
    double sum_luma_sq = 0.0;
    double sum_lap = 0.0;
    double sum_lap_sq = 0.0;

#if VIZ_PACK_AVX2
    const __m256 v_half = _mm256_set1_ps(0.5f);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_wr = _mm256_set1_ps(0.2126f);
    const __m256 v_wg = _mm256_set1_ps(0.7152f);
    const __m256 v_wb = _mm256_set1_ps(0.0722f);
    const __m256 v_255 = _mm256_set1_ps(255.0f);
    const __m256 v_four = _mm256_set1_ps(4.0f);
#endif

    for (uint32_t y = 0; y < height; ++y) {
        float *row_cur = luma_ring + (size_t)(y % 3u) * width;
        size_t row_off = (size_t)y * width;
        uint32_t x = 0;

#if VIZ_PACK_AVX2
        __m256 v_row_sum = _mm256_setzero_ps();
        __m256 v_row_sq = _mm256_setzero_ps();
        int32_t bins[8];

        for (; x + 8u <= width; x += 8u) {
            size_t idx = row_off + x;
            __m256 vr = _mm256_min_ps(v_one, _mm256_max_ps(v_zero, _mm256_fmadd_ps(_mm256_loadu_ps(r_p + idx), v_half, v_half)));
            __m256 vg = _mm256_min_ps(v_one, _mm256_max_ps(v_zero, _mm256_fmadd_ps(_mm256_loadu_ps(g_p + idx), v_half, v_half)));
            __m256 vb = _mm256_min_ps(v_one, _mm256_max_ps(v_zero, _mm256_fmadd_ps(_mm256_loadu_ps(b_p + idx), v_half, v_half)));
            __m256 vy = _mm256_fmadd_ps(vr, v_wr, _mm256_fmadd_ps(vg, v_wg, _mm256_mul_ps(vb, v_wb)));
            _mm256_storeu_ps(row_cur + x, vy);

            v_row_sum = _mm256_add_ps(v_row_sum, vy);
            v_row_sq = _mm256_fmadd_ps(vy, vy, v_row_sq);

            __m256i vbins = _mm256_cvtps_epi32(_mm256_mul_ps(vy, v_255));
            _mm256_storeu_si256((__m256i *)bins, vbins);
            hist[(uint32_t)bins[0] & 0xFFu]++;
            hist[(uint32_t)bins[1] & 0xFFu]++;
            hist[(uint32_t)bins[2] & 0xFFu]++;
            hist[(uint32_t)bins[3] & 0xFFu]++;
            hist[(uint32_t)bins[4] & 0xFFu]++;
            hist[(uint32_t)bins[5] & 0xFFu]++;
            hist[(uint32_t)bins[6] & 0xFFu]++;
            hist[(uint32_t)bins[7] & 0xFFu]++;
        }

        float s_lanes[8], sq_lanes[8];
        _mm256_storeu_ps(s_lanes, v_row_sum);
        _mm256_storeu_ps(sq_lanes, v_row_sq);
        for (int k = 0; k < 8; ++k) {
            sum_luma += (double)s_lanes[k];
            sum_luma_sq += (double)sq_lanes[k];
        }
#endif
        for (; x < width; ++x) {
            size_t idx = row_off + x;
            float r = fminf(1.0f, fmaxf(0.0f, r_p[idx] * 0.5f + 0.5f));
            float g = fminf(1.0f, fmaxf(0.0f, g_p[idx] * 0.5f + 0.5f));
            float b = fminf(1.0f, fmaxf(0.0f, b_p[idx] * 0.5f + 0.5f));
            float y_val = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            row_cur[x] = y_val;
            sum_luma += (double)y_val;
            sum_luma_sq += (double)y_val * (double)y_val;
            int bin = (int)lrintf(y_val * 255.0f);
            if (bin < 0) bin = 0;
            if (bin > 255) bin = 255;
            hist[bin]++;
        }

        if (y >= 2u && width >= 3u) {
            const float *row_up = luma_ring + (size_t)((y - 2u) % 3u) * width;
            const float *row_ct = luma_ring + (size_t)((y - 1u) % 3u) * width;
            const float *row_dn = row_cur;
            uint32_t lx = 1u;

#if VIZ_PACK_AVX2
            __m256 v_l_sum = _mm256_setzero_ps();
            __m256 v_l_sq = _mm256_setzero_ps();
            for (; lx + 8u < width; lx += 8u) {
                __m256 v_up = _mm256_loadu_ps(row_up + lx);
                __m256 v_dn = _mm256_loadu_ps(row_dn + lx);
                __m256 v_lf = _mm256_loadu_ps(row_ct + lx - 1u);
                __m256 v_rt = _mm256_loadu_ps(row_ct + lx + 1u);
                __m256 v_ct = _mm256_loadu_ps(row_ct + lx);
                __m256 v_lap = _mm256_fnmadd_ps(v_four, v_ct, _mm256_add_ps(_mm256_add_ps(v_up, v_dn), _mm256_add_ps(v_lf, v_rt)));
                v_l_sum = _mm256_add_ps(v_l_sum, v_lap);
                v_l_sq = _mm256_fmadd_ps(v_lap, v_lap, v_l_sq);
            }
            float ls[8], lsq[8];
            _mm256_storeu_ps(ls, v_l_sum);
            _mm256_storeu_ps(lsq, v_l_sq);
            for (int k = 0; k < 8; ++k) {
                sum_lap += (double)ls[k];
                sum_lap_sq += (double)lsq[k];
            }
#endif
            for (; lx + 1u < width; ++lx) {
                double lap = (double)row_up[lx]
                           + (double)row_dn[lx]
                           + (double)row_ct[lx - 1u]
                           + (double)row_ct[lx + 1u]
                           - 4.0 * (double)row_ct[lx];
                sum_lap += lap;
                sum_lap_sq += lap * lap;
            }
        }
    }

    if (owns_ring) {
        free(luma_ring);
    }

    double inv_n = 1.0 / (double)plane;
    double mean_l = sum_luma * inv_n;
    double var_l = fmax(0.0, (sum_luma_sq * inv_n) - (mean_l * mean_l));

    double entropy = 0.0;
    const double inv_log2 = 1.0 / log(2.0);
    for (int b = 0; b < 256; ++b) {
        if (hist[b] > 0) {
            double p = (double)hist[b] * inv_n;
            entropy -= p * (log(p) * inv_log2);
        }
    }

    double sharpness = 0.0;
    if (width >= 3u && height >= 3u) {
        size_t inner_cnt = (size_t)(width - 2u) * (size_t)(height - 2u);
        double mean_lap = sum_lap / (double)inner_cnt;
        double var_lap = fmax(0.0, (sum_lap_sq / (double)inner_cnt) - (mean_lap * mean_lap));
        sharpness = var_lap * 1000.0;
    }

    out_telemetry->mean_luminance = (float)mean_l;
    out_telemetry->rms_contrast = (float)sqrt(var_l);
    out_telemetry->shannon_entropy_bits = (float)entropy;
    out_telemetry->laplacian_sharpness = (float)sharpness;
}

static const float g_bayer_8x8[64] = {
    -31.5f/64.0f,  16.5f/64.0f, -19.5f/64.0f,  28.5f/64.0f, -28.5f/64.0f,  19.5f/64.0f, -16.5f/64.0f,  31.5f/64.0f,
      0.5f/64.0f, -15.5f/64.0f,  12.5f/64.0f,  -3.5f/64.0f,   3.5f/64.0f, -12.5f/64.0f,  15.5f/64.0f,  -0.5f/64.0f,
    -23.5f/64.0f,  24.5f/64.0f, -27.5f/64.0f,  20.5f/64.0f, -20.5f/64.0f,  27.5f/64.0f, -24.5f/64.0f,  23.5f/64.0f,
      8.5f/64.0f,  -7.5f/64.0f,   4.5f/64.0f, -11.5f/64.0f,  11.5f/64.0f,  -4.5f/64.0f,   7.5f/64.0f,  -8.5f/64.0f,
    -29.5f/64.0f,  18.5f/64.0f, -17.5f/64.0f,  30.5f/64.0f, -30.5f/64.0f,  17.5f/64.0f, -18.5f/64.0f,  29.5f/64.0f,
      2.5f/64.0f, -13.5f/64.0f,  14.5f/64.0f,  -1.5f/64.0f,   1.5f/64.0f, -14.5f/64.0f,  13.5f/64.0f,  -2.5f/64.0f,
    -21.5f/64.0f,  26.5f/64.0f, -25.5f/64.0f,  22.5f/64.0f, -22.5f/64.0f,  25.5f/64.0f, -26.5f/64.0f,  21.5f/64.0f,
     10.5f/64.0f,  -5.5f/64.0f,   6.5f/64.0f,  -9.5f/64.0f,   9.5f/64.0f,  -6.5f/64.0f,   5.5f/64.0f, -10.5f/64.0f
};

void viz_postprocess_vae_f32_to_rgb8(
    const float *planar_vae_rgb,
    uint32_t width,
    uint32_t height,
    int apply_bayer_dither,
    uint8_t *out_interleaved_rgb8
) {
    size_t plane = (size_t)width * (size_t)height;
    const float *r_p = planar_vae_rgb;
    const float *g_p = planar_vae_rgb + plane;
    const float *b_p = planar_vae_rgb + plane * 2u;

#if VIZ_PACK_AVX2
    const __m256 v_half = _mm256_set1_ps(0.5f);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_255 = _mm256_set1_ps(255.0f);
    const __m256i v_i0 = _mm256_setzero_si256();
    const __m256i v_i255 = _mm256_set1_epi32(255);
#endif

    for (uint32_t y = 0; y < height; ++y) {
        const float *bayer_row = g_bayer_8x8 + ((y & 7u) << 3u);
        size_t row_off = (size_t)y * width;
        uint32_t x = 0;

#if VIZ_PACK_AVX2
        __m256 v_dither = apply_bayer_dither ? _mm256_loadu_ps(bayer_row) : v_zero;
        int32_t r_i[8], g_i[8], b_i[8];

        for (; x + 8u <= width; x += 8u) {
            size_t idx = row_off + x;
            __m256 vr = _mm256_min_ps(v_one, _mm256_max_ps(v_zero, _mm256_fmadd_ps(_mm256_loadu_ps(r_p + idx), v_half, v_half)));
            __m256 vg = _mm256_min_ps(v_one, _mm256_max_ps(v_zero, _mm256_fmadd_ps(_mm256_loadu_ps(g_p + idx), v_half, v_half)));
            __m256 vb = _mm256_min_ps(v_one, _mm256_max_ps(v_zero, _mm256_fmadd_ps(_mm256_loadu_ps(b_p + idx), v_half, v_half)));

            __m256i vri = _mm256_min_epi32(v_i255, _mm256_max_epi32(v_i0, _mm256_cvtps_epi32(_mm256_fmadd_ps(vr, v_255, v_dither))));
            __m256i vgi = _mm256_min_epi32(v_i255, _mm256_max_epi32(v_i0, _mm256_cvtps_epi32(_mm256_fmadd_ps(vg, v_255, v_dither))));
            __m256i vbi = _mm256_min_epi32(v_i255, _mm256_max_epi32(v_i0, _mm256_cvtps_epi32(_mm256_fmadd_ps(vb, v_255, v_dither))));

            _mm256_storeu_si256((__m256i *)r_i, vri);
            _mm256_storeu_si256((__m256i *)g_i, vgi);
            _mm256_storeu_si256((__m256i *)b_i, vbi);

            uint8_t *dst = out_interleaved_rgb8 + idx * 3u;
            for (int k = 0; k < 8; ++k) {
                dst[k * 3 + 0] = (uint8_t)r_i[k];
                dst[k * 3 + 1] = (uint8_t)g_i[k];
                dst[k * 3 + 2] = (uint8_t)b_i[k];
            }
        }
#endif
        for (; x < width; ++x) {
            size_t idx = row_off + x;
            float dither = apply_bayer_dither ? bayer_row[x & 7u] : 0.0f;
            float r = fminf(1.0f, fmaxf(0.0f, r_p[idx] * 0.5f + 0.5f)) * 255.0f + dither;
            float g = fminf(1.0f, fmaxf(0.0f, g_p[idx] * 0.5f + 0.5f)) * 255.0f + dither;
            float b = fminf(1.0f, fmaxf(0.0f, b_p[idx] * 0.5f + 0.5f)) * 255.0f + dither;
            int ri = (int)lrintf(r);
            int gi = (int)lrintf(g);
            int bi = (int)lrintf(b);
            out_interleaved_rgb8[idx * 3u + 0u] = (uint8_t)(ri < 0 ? 0 : (ri > 255 ? 255 : ri));
            out_interleaved_rgb8[idx * 3u + 1u] = (uint8_t)(gi < 0 ? 0 : (gi > 255 ? 255 : gi));
            out_interleaved_rgb8[idx * 3u + 2u] = (uint8_t)(bi < 0 ? 0 : (bi > 255 ? 255 : bi));
        }
    }
}