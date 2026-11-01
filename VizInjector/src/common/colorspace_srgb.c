#include "colorspace_srgb.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#define VIZ_SRGB_INTERP_LUT_SCALE 8192

static float g_srgb_u8_to_lin_lut[256];
static float g_srgb_to_lin_interp_lut[VIZ_SRGB_INTERP_LUT_SCALE + 2];
static float g_lin_to_srgb_lut[65538];
static volatile long g_tables_state = 0;

float viz_srgb_to_linear_f32(float srgb_val) {
    if (!(srgb_val > 0.0f)) return 0.0f;
    if (srgb_val >= 1.0f) return 1.0f;
    if (srgb_val <= 0.04045f) {
        return srgb_val * (1.0f / 12.92f);
    }
    return (float)pow(((double)srgb_val + 0.055) * (1.0 / 1.055), 2.4);
}

float viz_linear_to_srgb_f32(float linear_val) {
    if (!(linear_val > 0.0f)) return 0.0f;
    if (linear_val >= 1.0f) return 1.0f;
    if (linear_val <= 0.0031308f) {
        return linear_val * 12.92f;
    }
    return (float)(1.055 * pow((double)linear_val, 1.0 / 2.4) - 0.055);
}

void viz_colorspace_init_tables(void) {
#if defined(_WIN32)
    if (InterlockedCompareExchange(&g_tables_state, 0, 0) == 2) {
        return;
    }
    if (InterlockedCompareExchange(&g_tables_state, 1, 0) == 0) {
        for (int i = 0; i < 256; ++i) {
            g_srgb_u8_to_lin_lut[i] = viz_srgb_to_linear_f32((float)i * (1.0f / 255.0f));
        }
        for (int i = 0; i <= VIZ_SRGB_INTERP_LUT_SCALE; ++i) {
            double x = (double)i * (1.0 / (double)VIZ_SRGB_INTERP_LUT_SCALE);
            if (x <= 0.04045) {
                g_srgb_to_lin_interp_lut[i] = (float)(x * (1.0 / 12.92));
            } else {
                g_srgb_to_lin_interp_lut[i] = (float)pow((x + 0.055) * (1.0 / 1.055), 2.4);
            }
        }
        g_srgb_to_lin_interp_lut[VIZ_SRGB_INTERP_LUT_SCALE + 1] = 1.0f;
        for (int i = 0; i <= 65536; ++i) {
            g_lin_to_srgb_lut[i] = viz_linear_to_srgb_f32((float)i * (1.0f / 65536.0f));
        }
        g_lin_to_srgb_lut[65537] = 1.0f;
        InterlockedExchange(&g_tables_state, 2);
        return;
    }
    while (InterlockedCompareExchange(&g_tables_state, 0, 0) != 2) {
        SwitchToThread();
    }
#else
    if (__atomic_load_n(&g_tables_state, __ATOMIC_ACQUIRE) == 2) {
        return;
    }
    long expected = 0;
    if (__atomic_compare_exchange_n(&g_tables_state, &expected, 1, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        for (int i = 0; i < 256; ++i) {
            g_srgb_u8_to_lin_lut[i] = viz_srgb_to_linear_f32((float)i * (1.0f / 255.0f));
        }
        for (int i = 0; i <= VIZ_SRGB_INTERP_LUT_SCALE; ++i) {
            double x = (double)i * (1.0 / (double)VIZ_SRGB_INTERP_LUT_SCALE);
            if (x <= 0.04045) {
                g_srgb_to_lin_interp_lut[i] = (float)(x * (1.0 / 12.92));
            } else {
                g_srgb_to_lin_interp_lut[i] = (float)pow((x + 0.055) * (1.0 / 1.055), 2.4);
            }
        }
        g_srgb_to_lin_interp_lut[VIZ_SRGB_INTERP_LUT_SCALE + 1] = 1.0f;
        for (int i = 0; i <= 65536; ++i) {
            g_lin_to_srgb_lut[i] = viz_linear_to_srgb_f32((float)i * (1.0f / 65536.0f));
        }
        g_lin_to_srgb_lut[65537] = 1.0f;
        __atomic_store_n(&g_tables_state, 2, __ATOMIC_RELEASE);
        return;
    }
    while (__atomic_load_n(&g_tables_state, __ATOMIC_ACQUIRE) != 2) {
        _mm_pause();
    }
#endif
}

const float *viz_get_srgb_to_lin_lut(void) {
    viz_colorspace_init_tables();
    return g_srgb_u8_to_lin_lut;
}

const float *viz_get_srgb_to_lin_interp_lut(void) {
    viz_colorspace_init_tables();
    return g_srgb_to_lin_interp_lut;
}

const float *viz_get_lin_to_srgb_lut(void) {
    viz_colorspace_init_tables();
    return g_lin_to_srgb_lut;
}

static inline __m256 viz_srgb_to_lin_avx2(__m256 v_srgb, const float *lut) {
    __m256 v_zero = _mm256_setzero_ps();
    __m256 v_one = _mm256_set1_ps(1.0f);
    __m256 v_valid = _mm256_cmp_ps(v_srgb, v_zero, _CMP_GT_OQ);
    __m256 v_pos = _mm256_and_ps(v_srgb, v_valid);
    __m256 v_clamped = _mm256_min_ps(v_one, v_pos);

    __m256 v_toe_thresh = _mm256_set1_ps(0.04045f);
    __m256 v_toe = _mm256_mul_ps(v_clamped, _mm256_set1_ps(1.0f / 12.92f));

    __m256 v_scaled = _mm256_mul_ps(v_clamped, _mm256_set1_ps((float)VIZ_SRGB_INTERP_LUT_SCALE));
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

static inline uint32_t hash_u32(uint32_t x) {
    x = (x ^ 61u) ^ (x >> 16u);
    x *= 9u;
    x = x ^ (x >> 4u);
    x *= 0x27D4EB2Du;
    x = x ^ (x >> 15u);
    return x;
}

static inline float hash_tpdf_dither(uint32_t x, uint32_t y, uint32_t ch, uint32_t seed) {
    uint32_t n1 = hash_u32(x * 0x1F123BB5u + y * 0x05491333u + ch * 0x9E3779B9u + seed);
    uint32_t n2 = hash_u32(n1 * 0x85EBCA6Bu + 0xC2B2AE35u);
    float u1 = (float)(n1 & 0xFFFFu) * (1.0f / 65535.0f);
    float u2 = (float)(n2 & 0xFFFFu) * (1.0f / 65535.0f);
    return (u1 - u2) * (0.5f / 255.0f);
}

static inline __m256i hash_u32_avx2(__m256i x) {
    x = _mm256_xor_si256(_mm256_xor_si256(x, _mm256_set1_epi32(61)), _mm256_srli_epi32(x, 16));
    x = _mm256_mullo_epi32(x, _mm256_set1_epi32(9));
    x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 4));
    x = _mm256_mullo_epi32(x, _mm256_set1_epi32((int)0x27D4EB2Du));
    x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 15));
    return x;
}

static inline __m256 hash_tpdf_dither_2px_avx2(uint32_t x, uint32_t y, uint32_t seed) {
    __m256i vx = _mm256_setr_epi32(
        (int)x, (int)x, (int)x, (int)x,
        (int)(x + 1u), (int)(x + 1u), (int)(x + 1u), (int)(x + 1u)
    );
    __m256i vy = _mm256_set1_epi32((int)y);
    __m256i vch = _mm256_setr_epi32(0, 1, 2, 0, 0, 1, 2, 0);
    __m256i vseed = _mm256_set1_epi32((int)seed);

    __m256i term_x = _mm256_mullo_epi32(vx, _mm256_set1_epi32((int)0x1F123BB5u));
    __m256i term_y = _mm256_mullo_epi32(vy, _mm256_set1_epi32((int)0x05491333u));
    __m256i term_c = _mm256_mullo_epi32(vch, _mm256_set1_epi32((int)0x9E3779B9u));
    __m256i in1 = _mm256_add_epi32(_mm256_add_epi32(term_x, term_y), _mm256_add_epi32(term_c, vseed));

    __m256i n1 = hash_u32_avx2(in1);
    __m256i in2 = _mm256_add_epi32(
        _mm256_mullo_epi32(n1, _mm256_set1_epi32((int)0x85EBCA6Bu)),
        _mm256_set1_epi32((int)0xC2B2AE35u)
    );
    __m256i n2 = hash_u32_avx2(in2);

    __m256i mask16 = _mm256_set1_epi32(0xFFFF);
    __m256 inv65535 = _mm256_set1_ps(1.0f / 65535.0f);
    __m256 u1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(n1, mask16)), inv65535);
    __m256 u2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(n2, mask16)), inv65535);
    return _mm256_mul_ps(_mm256_sub_ps(u1, u2), _mm256_set1_ps(0.5f / 255.0f));
}

void viz_ingest_scalar_rgba8_to_f32(
    const uint8_t *src_rgba8,
    uint32_t width,
    uint32_t height,
    float *out_linear_rgba
) {
    const size_t num_pixels = (size_t)width * (size_t)height;
    const __m256 inv255 = _mm256_set1_ps(1.0f / 255.0f);
    size_t i = 0;
    for (; i + 2u <= num_pixels; i += 2u) {
        __m128i bytes = _mm_loadl_epi64((const __m128i *)(src_rgba8 + i * 4u));
        __m256i ints = _mm256_cvtepu8_epi32(bytes);
        __m256 scaled = _mm256_mul_ps(_mm256_cvtepi32_ps(ints), inv255);
        _mm256_storeu_ps(out_linear_rgba + i * 4u, scaled);
    }
    for (; i < num_pixels; ++i) {
        const uint8_t *s = src_rgba8 + i * 4u;
        float *d = out_linear_rgba + i * 4u;
        d[0] = (float)s[0] * (1.0f / 255.0f);
        d[1] = (float)s[1] * (1.0f / 255.0f);
        d[2] = (float)s[2] * (1.0f / 255.0f);
        d[3] = (float)s[3] * (1.0f / 255.0f);
    }
}

void viz_ingest_scalar_rgba16_to_f32(
    const uint16_t *src_rgba16,
    uint32_t width,
    uint32_t height,
    float *out_linear_rgba
) {
    const size_t num_pixels = (size_t)width * (size_t)height;
    const __m256 inv65535 = _mm256_set1_ps(1.0f / 65535.0f);
    size_t i = 0;
    for (; i + 2u <= num_pixels; i += 2u) {
        __m128i words = _mm_loadu_si128((const __m128i *)(src_rgba16 + i * 4u));
        __m256i ints = _mm256_cvtepu16_epi32(words);
        __m256 scaled = _mm256_mul_ps(_mm256_cvtepi32_ps(ints), inv65535);
        _mm256_storeu_ps(out_linear_rgba + i * 4u, scaled);
    }
    for (; i < num_pixels; ++i) {
        const uint16_t *s = src_rgba16 + i * 4u;
        float *d = out_linear_rgba + i * 4u;
        d[0] = (float)s[0] * (1.0f / 65535.0f);
        d[1] = (float)s[1] * (1.0f / 65535.0f);
        d[2] = (float)s[2] * (1.0f / 65535.0f);
        d[3] = (float)s[3] * (1.0f / 65535.0f);
    }
}

static inline void ingest_pixel_scalar(
    const uint8_t *src_rgba8,
    const uint8_t *row_src,
    float *row_dst,
    uint32_t x,
    uint32_t y,
    uint32_t width,
    uint32_t height,
    int deband_8bit,
    uint32_t seed
) {
    const uint8_t *p = row_src + (size_t)x * 4u;
    float *dst = row_dst + (size_t)x * 4u;

    if (p[3] <= 1u) {
        float sum_r = 0.0f;
        float sum_g = 0.0f;
        float sum_b = 0.0f;
        float sum_w = 0.0f;
        int ix = (int)x;
        int iy = (int)y;
        int max_x = (int)width - 1;
        int max_y = (int)height - 1;

        int y0 = (iy >= 2) ? (iy - 2) : 0;
        int y1 = (iy + 2 <= max_y) ? (iy + 2) : max_y;
        int x0 = (ix >= 2) ? (ix - 2) : 0;
        int x1 = (ix + 2 <= max_x) ? (ix + 2) : max_x;

        for (int ny = y0; ny <= y1; ++ny) {
            int dy = ny - iy;
            int ady = dy < 0 ? -dy : dy;
            const uint8_t *nrow = src_rgba8 + (size_t)ny * (size_t)width * 4u;
            for (int nx = x0; nx <= x1; ++nx) {
                const uint8_t *np = nrow + (size_t)nx * 4u;
                if (np[3] > 1u) {
                    int dx = nx - ix;
                    int dist = (dx < 0 ? -dx : dx) + ady;
                    if (dist < 1) dist = 1;
                    float w = ((float)np[3] * (1.0f / 255.0f)) / (float)dist;
                    sum_r += g_srgb_u8_to_lin_lut[np[0]] * w;
                    sum_g += g_srgb_u8_to_lin_lut[np[1]] * w;
                    sum_b += g_srgb_u8_to_lin_lut[np[2]] * w;
                    sum_w += w;
                }
            }
        }

        if (sum_w <= 1e-6f) {
            int oy0 = (iy >= 8) ? (iy - 8) : 0;
            int oy1 = (iy + 8 <= max_y) ? (iy + 8) : max_y;
            int ox0 = (ix >= 8) ? (ix - 8) : 0;
            int ox1 = (ix + 8 <= max_x) ? (ix + 8) : max_x;

            for (int ny = oy0; ny <= oy1; ++ny) {
                int dy = ny - iy;
                int ady = dy < 0 ? -dy : dy;
                const uint8_t *nrow = src_rgba8 + (size_t)ny * (size_t)width * 4u;
                for (int nx = ox0; nx <= ox1; ++nx) {
                    int dx = nx - ix;
                    int adx = dx < 0 ? -dx : dx;
                    if (adx <= 2 && ady <= 2) {
                        continue;
                    }
                    const uint8_t *np = nrow + (size_t)nx * 4u;
                    if (np[3] > 1u) {
                        int dist = adx + ady;
                        float w = ((float)np[3] * (1.0f / 255.0f)) / (float)dist;
                        sum_r += g_srgb_u8_to_lin_lut[np[0]] * w;
                        sum_g += g_srgb_u8_to_lin_lut[np[1]] * w;
                        sum_b += g_srgb_u8_to_lin_lut[np[2]] * w;
                        sum_w += w;
                    }
                }
            }
        }

        if (sum_w > 1e-6f) {
            float inv_w = 1.0f / sum_w;
            dst[0] = sum_r * inv_w;
            dst[1] = sum_g * inv_w;
            dst[2] = sum_b * inv_w;
        } else {
            dst[0] = 0.0f;
            dst[1] = 0.0f;
            dst[2] = 0.0f;
        }
        dst[3] = 0.0f;
        return;
    }

    float a = (float)p[3] * (1.0f / 255.0f);
    if (deband_8bit) {
        uint32_t rx = (x + 1u < width) ? (x + 1u) : x;
        uint32_t by = (y + 1u < height) ? (y + 1u) : y;
        const uint8_t *pr = row_src + (size_t)rx * 4u;
        const uint8_t *pb = src_rgba8 + ((size_t)by * (size_t)width + (size_t)x) * 4u;

        float dr = (float)((int)p[0] - (int)pr[0]) * (1.0f / 255.0f);
        float dg = (float)((int)p[1] - (int)pr[1]) * (1.0f / 255.0f);
        float db = (float)((int)p[2] - (int)pr[2]) * (1.0f / 255.0f);
        float grad_r = sqrtf(dr * dr + dg * dg + db * db);

        float dbr = (float)((int)p[0] - (int)pb[0]) * (1.0f / 255.0f);
        float dbg = (float)((int)p[1] - (int)pb[1]) * (1.0f / 255.0f);
        float dbb = (float)((int)p[2] - (int)pb[2]) * (1.0f / 255.0f);
        float grad_b = sqrtf(dbr * dbr + dbg * dbg + dbb * dbb);

        float grad = (grad_r > grad_b) ? grad_r : grad_b;
        float gate = 1.0f - grad * 16.0f;
        if (gate > 0.0f) {
            if (gate > 1.0f) gate = 1.0f;
            for (uint32_t c = 0; c < 3u; ++c) {
                float s = (float)p[c] * (1.0f / 255.0f) + hash_tpdf_dither(x, y, c, seed) * gate;
                if (s <= 0.0f) {
                    dst[c] = 0.0f;
                } else if (s >= 1.0f) {
                    dst[c] = 1.0f;
                } else if (s <= 0.04045f) {
                    dst[c] = s * (1.0f / 12.92f);
                } else {
                    float scaled = s * (float)VIZ_SRGB_INTERP_LUT_SCALE;
                    int idx = (int)scaled;
                    float frac = scaled - (float)idx;
                    float y0 = g_srgb_to_lin_interp_lut[idx];
                    float y1 = g_srgb_to_lin_interp_lut[idx + 1];
                    dst[c] = y0 + frac * (y1 - y0);
                }
            }
            dst[3] = a;
            return;
        }
    }

    dst[0] = g_srgb_u8_to_lin_lut[p[0]];
    dst[1] = g_srgb_u8_to_lin_lut[p[1]];
    dst[2] = g_srgb_u8_to_lin_lut[p[2]];
    dst[3] = a;
}

void viz_ingest_rgba8_to_linear_f32(
    const uint8_t *src_rgba8,
    uint32_t width,
    uint32_t height,
    int deband_8bit,
    uint32_t seed,
    float *out_linear_rgba
) {
    viz_colorspace_init_tables();
    const __m256 inv255 = _mm256_set1_ps(1.0f / 255.0f);
    const float *interp_lut = g_srgb_to_lin_interp_lut;

    for (uint32_t y = 0; y < height; ++y) {
        const uint8_t *row_src = src_rgba8 + (size_t)y * (size_t)width * 4u;
        float *row_dst = out_linear_rgba + (size_t)y * (size_t)width * 4u;
        uint32_t x = 0;

        if (!deband_8bit) {
            while (x + 2u <= width) {
                const uint8_t *p = row_src + (size_t)x * 4u;
                if (__builtin_expect(p[3] > 1u && p[7] > 1u, 1)) {
                    __m128i b8 = _mm_loadl_epi64((const __m128i *)p);
                    __m256i idx = _mm256_cvtepu8_epi32(b8);
                    __m256 lin_rgb = _mm256_i32gather_ps(g_srgb_u8_to_lin_lut, idx, 4);
                    __m256 lin_a = _mm256_mul_ps(_mm256_cvtepi32_ps(idx), inv255);
                    __m256 res = _mm256_blend_ps(lin_rgb, lin_a, 0x88);
                    _mm256_storeu_ps(row_dst + (size_t)x * 4u, res);
                    x += 2u;
                } else {
                    ingest_pixel_scalar(src_rgba8, row_src, row_dst, x, y, width, height, 0, seed);
                    x += 1u;
                }
            }
        } else if (width >= 3u) {
            uint32_t by = (y + 1u < height) ? (y + 1u) : y;
            const uint8_t *row_bot = src_rgba8 + (size_t)by * (size_t)width * 4u;
            while (x + 2u < width) {
                const uint8_t *p = row_src + (size_t)x * 4u;
                if (__builtin_expect(p[3] > 1u && p[7] > 1u, 1)) {
                    const uint8_t *pr = p + 4u;
                    const uint8_t *pb = row_bot + (size_t)x * 4u;

                    __m256i idx_p = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)p));
                    __m256i idx_r = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)pr));
                    __m256i idx_b = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)pb));

                    __m256 vf_p = _mm256_mul_ps(_mm256_cvtepi32_ps(idx_p), inv255);
                    __m256 vf_r = _mm256_mul_ps(_mm256_cvtepi32_ps(idx_r), inv255);
                    __m256 vf_b = _mm256_mul_ps(_mm256_cvtepi32_ps(idx_b), inv255);

                    __m256 diff_r = _mm256_blend_ps(_mm256_sub_ps(vf_p, vf_r), _mm256_setzero_ps(), 0x88);
                    __m256 diff_b = _mm256_blend_ps(_mm256_sub_ps(vf_p, vf_b), _mm256_setzero_ps(), 0x88);
                    __m256 sq_r = _mm256_mul_ps(diff_r, diff_r);
                    __m256 sq_b = _mm256_mul_ps(diff_b, diff_b);

                    __m256 sum_r = _mm256_add_ps(
                        sq_r,
                        _mm256_add_ps(
                            _mm256_permute_ps(sq_r, _MM_SHUFFLE(3, 0, 2, 1)),
                            _mm256_permute_ps(sq_r, _MM_SHUFFLE(3, 1, 0, 2))
                        )
                    );
                    __m256 sum_b = _mm256_add_ps(
                        sq_b,
                        _mm256_add_ps(
                            _mm256_permute_ps(sq_b, _MM_SHUFFLE(3, 0, 2, 1)),
                            _mm256_permute_ps(sq_b, _MM_SHUFFLE(3, 1, 0, 2))
                        )
                    );

                    __m256 grad = _mm256_max_ps(_mm256_sqrt_ps(sum_r), _mm256_sqrt_ps(sum_b));
                    __m256 gate = _mm256_min_ps(
                        _mm256_set1_ps(1.0f),
                        _mm256_max_ps(_mm256_setzero_ps(), _mm256_fnmadd_ps(grad, _mm256_set1_ps(16.0f), _mm256_set1_ps(1.0f)))
                    );

                    __m256 lut_exact = _mm256_i32gather_ps(g_srgb_u8_to_lin_lut, idx_p, 4);
                    __m256 has_gate = _mm256_cmp_ps(gate, _mm256_setzero_ps(), _CMP_GT_OQ);
                    if (_mm256_movemask_ps(has_gate) != 0) {
                        __m256 dither = hash_tpdf_dither_2px_avx2(x, y, seed);
                        __m256 s_dithered = _mm256_fmadd_ps(dither, gate, vf_p);
                        __m256 lin_dithered = viz_srgb_to_lin_avx2(s_dithered, interp_lut);
                        lut_exact = _mm256_blendv_ps(lut_exact, lin_dithered, has_gate);
                    }

                    __m256 res = _mm256_blend_ps(lut_exact, vf_p, 0x88);
                    _mm256_storeu_ps(row_dst + (size_t)x * 4u, res);
                    x += 2u;
                } else {
                    ingest_pixel_scalar(src_rgba8, row_src, row_dst, x, y, width, height, 1, seed);
                    x += 1u;
                }
            }
        }

        for (; x < width; ++x) {
            ingest_pixel_scalar(src_rgba8, row_src, row_dst, x, y, width, height, deband_8bit, seed);
        }
    }
}

static inline void ingest_pixel_rgba16_scalar(
    const uint16_t *src_rgba16,
    const uint16_t *row_src,
    float *row_dst,
    uint32_t x,
    uint32_t y,
    uint32_t width,
    uint32_t height
) {
    const float inv65535 = 1.0f / 65535.0f;
    const uint16_t *p = row_src + (size_t)x * 4u;
    float *dst = row_dst + (size_t)x * 4u;

    if (p[3] <= 257u) {
        float sum_r = 0.0f;
        float sum_g = 0.0f;
        float sum_b = 0.0f;
        float sum_w = 0.0f;
        int ix = (int)x;
        int iy = (int)y;
        int max_x = (int)width - 1;
        int max_y = (int)height - 1;

        int y0 = (iy >= 2) ? (iy - 2) : 0;
        int y1 = (iy + 2 <= max_y) ? (iy + 2) : max_y;
        int x0 = (ix >= 2) ? (ix - 2) : 0;
        int x1 = (ix + 2 <= max_x) ? (ix + 2) : max_x;

        for (int ny = y0; ny <= y1; ++ny) {
            int dy = ny - iy;
            int ady = dy < 0 ? -dy : dy;
            const uint16_t *nrow = src_rgba16 + (size_t)ny * (size_t)width * 4u;
            for (int nx = x0; nx <= x1; ++nx) {
                const uint16_t *np = nrow + (size_t)nx * 4u;
                if (np[3] > 257u) {
                    int dx = nx - ix;
                    int dist = (dx < 0 ? -dx : dx) + ady;
                    if (dist < 1) dist = 1;
                    float w = ((float)np[3] * inv65535) / (float)dist;
                    sum_r += viz_srgb_to_linear_f32((float)np[0] * inv65535) * w;
                    sum_g += viz_srgb_to_linear_f32((float)np[1] * inv65535) * w;
                    sum_b += viz_srgb_to_linear_f32((float)np[2] * inv65535) * w;
                    sum_w += w;
                }
            }
        }

        if (sum_w <= 1e-6f) {
            int oy0 = (iy >= 8) ? (iy - 8) : 0;
            int oy1 = (iy + 8 <= max_y) ? (iy + 8) : max_y;
            int ox0 = (ix >= 8) ? (ix - 8) : 0;
            int ox1 = (ix + 8 <= max_x) ? (ix + 8) : max_x;

            for (int ny = oy0; ny <= oy1; ++ny) {
                int dy = ny - iy;
                int ady = dy < 0 ? -dy : dy;
                const uint16_t *nrow = src_rgba16 + (size_t)ny * (size_t)width * 4u;
                for (int nx = ox0; nx <= ox1; ++nx) {
                    int dx = nx - ix;
                    int adx = dx < 0 ? -dx : dx;
                    if (adx <= 2 && ady <= 2) {
                        continue;
                    }
                    const uint16_t *np = nrow + (size_t)nx * 4u;
                    if (np[3] > 257u) {
                        int dist = adx + ady;
                        float w = ((float)np[3] * inv65535) / (float)dist;
                        sum_r += viz_srgb_to_linear_f32((float)np[0] * inv65535) * w;
                        sum_g += viz_srgb_to_linear_f32((float)np[1] * inv65535) * w;
                        sum_b += viz_srgb_to_linear_f32((float)np[2] * inv65535) * w;
                        sum_w += w;
                    }
                }
            }
        }

        if (sum_w > 1e-6f) {
            float inv_w = 1.0f / sum_w;
            dst[0] = sum_r * inv_w;
            dst[1] = sum_g * inv_w;
            dst[2] = sum_b * inv_w;
        } else {
            dst[0] = 0.0f;
            dst[1] = 0.0f;
            dst[2] = 0.0f;
        }
        dst[3] = 0.0f;
        return;
    }

    dst[0] = viz_srgb_to_linear_f32((float)p[0] * inv65535);
    dst[1] = viz_srgb_to_linear_f32((float)p[1] * inv65535);
    dst[2] = viz_srgb_to_linear_f32((float)p[2] * inv65535);
    dst[3] = (float)p[3] * inv65535;
}

void viz_ingest_rgba16_to_linear_f32(
    const uint16_t *src_rgba16,
    uint32_t width,
    uint32_t height,
    float *out_linear_rgba
) {
    viz_colorspace_init_tables();
    const float *interp_lut = g_srgb_to_lin_interp_lut;
    const __m256 inv65535 = _mm256_set1_ps(1.0f / 65535.0f);

    for (uint32_t y = 0; y < height; ++y) {
        const uint16_t *row_src = src_rgba16 + (size_t)y * (size_t)width * 4u;
        float *row_dst = out_linear_rgba + (size_t)y * (size_t)width * 4u;
        uint32_t x = 0;

        while (x + 2u <= width) {
            const uint16_t *p = row_src + (size_t)x * 4u;
            if (__builtin_expect(p[3] > 257u && p[7] > 257u, 1)) {
                __m128i w16 = _mm_loadu_si128((const __m128i *)p);
                __m256i i32 = _mm256_cvtepu16_epi32(w16);
                __m256 norm = _mm256_mul_ps(_mm256_cvtepi32_ps(i32), inv65535);
                __m256 lin_rgb = viz_srgb_to_lin_avx2(norm, interp_lut);
                __m256 res = _mm256_blend_ps(lin_rgb, norm, 0x88);
                _mm256_storeu_ps(row_dst + (size_t)x * 4u, res);
                x += 2u;
            } else {
                ingest_pixel_rgba16_scalar(src_rgba16, row_src, row_dst, x, y, width, height);
                x += 1u;
            }
        }
        for (; x < width; ++x) {
            ingest_pixel_rgba16_scalar(src_rgba16, row_src, row_dst, x, y, width, height);
        }
    }
}

static inline float aces_tonemap_channel(float x) {
    if (!(x > 0.0f)) return 0.0f;
    if (x > 65504.0f) x = 65504.0f;
    const float a = 1.80f;
    const float b = 1.00f;
    const float c = 2.00f;
    float mapped = (x * (a * x + b)) / (x * (a * x + c) + b);
    return (mapped > 1.0f) ? 1.0f : mapped;
}

static inline float clamp_unit_channel(float x) {
    if (!(x > 0.0f)) return 0.0f;
    return (x > 1.0f) ? 1.0f : x;
}

static inline float convert_f32_channel_to_linear(float x, int input_is_linear) {
    if (input_is_linear == 0) {
        return viz_srgb_to_linear_f32(x);
    }
    if (input_is_linear == 1) {
        return aces_tonemap_channel(x);
    }
    return clamp_unit_channel(x);
}

static inline __m256 aces_tonemap_avx2(__m256 x) {
    __m256 v_zero = _mm256_setzero_ps();
    __m256 v_one = _mm256_set1_ps(1.0f);
    __m256 v_max_hdr = _mm256_set1_ps(65504.0f);
    __m256 valid = _mm256_cmp_ps(x, v_zero, _CMP_GT_OQ);
    __m256 xp = _mm256_min_ps(v_max_hdr, _mm256_and_ps(x, valid));

    __m256 va = _mm256_set1_ps(1.80f);
    __m256 vb = _mm256_set1_ps(1.00f);
    __m256 vc = _mm256_set1_ps(2.00f);

    __m256 num = _mm256_mul_ps(xp, _mm256_fmadd_ps(va, xp, vb));
    __m256 den = _mm256_fmadd_ps(xp, _mm256_fmadd_ps(va, xp, vc), vb);
    return _mm256_min_ps(v_one, _mm256_div_ps(num, den));
}

static inline void ingest_pixel_rgbaf32_scalar(
    const float *src_rgbaf32,
    const float *row_src,
    float *row_dst,
    uint32_t x,
    uint32_t y,
    uint32_t width,
    uint32_t height,
    int input_is_linear
) {
    const float alpha_thresh = 1.0f / 255.0f;
    const float *p = row_src + (size_t)x * 4u;
    float *dst = row_dst + (size_t)x * 4u;

    if (!(p[3] > alpha_thresh)) {
        float sum_r = 0.0f;
        float sum_g = 0.0f;
        float sum_b = 0.0f;
        float sum_w = 0.0f;
        int ix = (int)x;
        int iy = (int)y;
        int max_x = (int)width - 1;
        int max_y = (int)height - 1;

        int y0 = (iy >= 2) ? (iy - 2) : 0;
        int y1 = (iy + 2 <= max_y) ? (iy + 2) : max_y;
        int x0 = (ix >= 2) ? (ix - 2) : 0;
        int x1 = (ix + 2 <= max_x) ? (ix + 2) : max_x;

        for (int ny = y0; ny <= y1; ++ny) {
            int dy = ny - iy;
            int ady = dy < 0 ? -dy : dy;
            const float *nrow = src_rgbaf32 + (size_t)ny * (size_t)width * 4u;
            for (int nx = x0; nx <= x1; ++nx) {
                const float *np = nrow + (size_t)nx * 4u;
                if (np[3] > alpha_thresh) {
                    int dx = nx - ix;
                    int dist = (dx < 0 ? -dx : dx) + ady;
                    if (dist < 1) dist = 1;
                    float np_a = (np[3] > 1.0f) ? 1.0f : np[3];
                    float w = np_a / (float)dist;
                    sum_r += convert_f32_channel_to_linear(np[0], input_is_linear) * w;
                    sum_g += convert_f32_channel_to_linear(np[1], input_is_linear) * w;
                    sum_b += convert_f32_channel_to_linear(np[2], input_is_linear) * w;
                    sum_w += w;
                }
            }
        }

        if (sum_w <= 1e-6f) {
            int oy0 = (iy >= 8) ? (iy - 8) : 0;
            int oy1 = (iy + 8 <= max_y) ? (iy + 8) : max_y;
            int ox0 = (ix >= 8) ? (ix - 8) : 0;
            int ox1 = (ix + 8 <= max_x) ? (ix + 8) : max_x;

            for (int ny = oy0; ny <= oy1; ++ny) {
                int dy = ny - iy;
                int ady = dy < 0 ? -dy : dy;
                const float *nrow = src_rgbaf32 + (size_t)ny * (size_t)width * 4u;
                for (int nx = ox0; nx <= ox1; ++nx) {
                    int dx = nx - ix;
                    int adx = dx < 0 ? -dx : dx;
                    if (adx <= 2 && ady <= 2) {
                        continue;
                    }
                    const float *np = nrow + (size_t)nx * 4u;
                    if (np[3] > alpha_thresh) {
                        int dist = adx + ady;
                        float np_a = (np[3] > 1.0f) ? 1.0f : np[3];
                        float w = np_a / (float)dist;
                        sum_r += convert_f32_channel_to_linear(np[0], input_is_linear) * w;
                        sum_g += convert_f32_channel_to_linear(np[1], input_is_linear) * w;
                        sum_b += convert_f32_channel_to_linear(np[2], input_is_linear) * w;
                        sum_w += w;
                    }
                }
            }
        }

        if (sum_w > 1e-6f) {
            float inv_w = 1.0f / sum_w;
            dst[0] = sum_r * inv_w;
            dst[1] = sum_g * inv_w;
            dst[2] = sum_b * inv_w;
        } else {
            dst[0] = 0.0f;
            dst[1] = 0.0f;
            dst[2] = 0.0f;
        }
        dst[3] = 0.0f;
        return;
    }

    dst[0] = convert_f32_channel_to_linear(p[0], input_is_linear);
    dst[1] = convert_f32_channel_to_linear(p[1], input_is_linear);
    dst[2] = convert_f32_channel_to_linear(p[2], input_is_linear);
    dst[3] = clamp_unit_channel(p[3]);
}

void viz_ingest_rgbaf32_to_linear_f32(
    const float *src_rgbaf32,
    uint32_t width,
    uint32_t height,
    int input_is_linear,
    float *out_linear_rgba
) {
    viz_colorspace_init_tables();
    const float *interp_lut = g_srgb_to_lin_interp_lut;
    const float alpha_thresh = 1.0f / 255.0f;
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_one = _mm256_set1_ps(1.0f);

    for (uint32_t y = 0; y < height; ++y) {
        const float *row_src = src_rgbaf32 + (size_t)y * (size_t)width * 4u;
        float *row_dst = out_linear_rgba + (size_t)y * (size_t)width * 4u;
        uint32_t x = 0;

        while (x + 2u <= width) {
            const float *p = row_src + (size_t)x * 4u;
            if (__builtin_expect(p[3] > alpha_thresh && p[7] > alpha_thresh, 1)) {
                __m256 v_in = _mm256_loadu_ps(p);
                __m256 v_pos = _mm256_and_ps(v_in, _mm256_cmp_ps(v_in, v_zero, _CMP_GT_OQ));
                __m256 v_clamped = _mm256_min_ps(v_one, v_pos);
                __m256 v_rgb;
                if (input_is_linear == 0) {
                    v_rgb = viz_srgb_to_lin_avx2(v_in, interp_lut);
                } else if (input_is_linear == 1) {
                    v_rgb = aces_tonemap_avx2(v_in);
                } else {
                    v_rgb = v_clamped;
                }
                _mm256_storeu_ps(row_dst + (size_t)x * 4u, _mm256_blend_ps(v_rgb, v_clamped, 0x88));
                x += 2u;
            } else {
                ingest_pixel_rgbaf32_scalar(src_rgbaf32, row_src, row_dst, x, y, width, height, input_is_linear);
                x += 1u;
            }
        }
        for (; x < width; ++x) {
            ingest_pixel_rgbaf32_scalar(src_rgbaf32, row_src, row_dst, x, y, width, height, input_is_linear);
        }
    }
}

static uint16_t read_u16_end(const uint8_t *p, int little_endian) {
    return little_endian
        ? (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8))
        : (uint16_t)(((uint16_t)p[0] << 8) | (uint16_t)p[1]);
}

static uint32_t read_u32_end(const uint8_t *p, int little_endian) {
    return little_endian
        ? ((uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24))
        : (((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3]);
}

int viz_parse_exif_orientation(const uint8_t *bytes, size_t len) {
    if (!bytes || len < 14 || bytes[0] != 0xFF || bytes[1] != 0xD8) {
        return 1;
    }
    size_t pos = 2;
    while (pos + 4 <= len) {
        if (bytes[pos] != 0xFF) break;
        uint8_t marker = bytes[pos + 1];
        if (marker == 0xDA || marker == 0xD9) break;
        uint16_t seg_len = (uint16_t)(((uint16_t)bytes[pos + 2] << 8) | bytes[pos + 3]);
        if (seg_len < 2 || pos + 2 + seg_len > len) break;
        if (marker == 0xE1 && seg_len >= 14 && memcmp(bytes + pos + 4, "Exif\0\0", 6) == 0) {
            const uint8_t *tiff = bytes + pos + 10;
            size_t tiff_len = seg_len - 8;
            int le = 0;
            if (tiff[0] == 'I' && tiff[1] == 'I') le = 1;
            else if (tiff[0] == 'M' && tiff[1] == 'M') le = 0;
            else return 1;
            uint32_t ifd_off = read_u32_end(tiff + 4, le);
            if (ifd_off + 2 > tiff_len) return 1;
            uint16_t entries = read_u16_end(tiff + ifd_off, le);
            for (uint16_t i = 0; i < entries; ++i) {
                size_t e_off = ifd_off + 2u + (size_t)i * 12u;
                if (e_off + 12u > tiff_len) break;
                uint16_t tag = read_u16_end(tiff + e_off, le);
                if (tag == 0x0112u) {
                    uint16_t val = read_u16_end(tiff + e_off + 8u, le);
                    if (val >= 1u && val <= 8u) return (int)val;
                    return 1;
                }
            }
            return 1;
        }
        pos += 2u + seg_len;
    }
    return 1;
}

void viz_apply_exif_orientation_rgba8(
    const uint8_t *src_rgba,
    uint32_t width,
    uint32_t height,
    int orientation,
    uint8_t **out_rgba,
    uint32_t *out_w,
    uint32_t *out_h
) {
    if (!src_rgba || !out_rgba || !out_w || !out_h) return;
    uint32_t dst_w = (orientation >= 5 && orientation <= 8) ? height : width;
    uint32_t dst_h = (orientation >= 5 && orientation <= 8) ? width : height;
    size_t bytes = (size_t)dst_w * (size_t)dst_h * 4u;
    uint8_t *dst = (uint8_t *)malloc(bytes);
    if (!dst) {
        *out_rgba = NULL;
        return;
    }
    if (orientation <= 1 || orientation > 8) {
        memcpy(dst, src_rgba, bytes);
        *out_rgba = dst;
        *out_w = width;
        *out_h = height;
        return;
    }
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            uint32_t dx = x, dy = y;
            switch (orientation) {
                case 2: dx = width - 1u - x; dy = y; break;
                case 3: dx = width - 1u - x; dy = height - 1u - y; break;
                case 4: dx = x; dy = height - 1u - y; break;
                case 5: dx = y; dy = x; break;
                case 6: dx = height - 1u - y; dy = x; break;
                case 7: dx = height - 1u - y; dy = width - 1u - x; break;
                case 8: dx = y; dy = width - 1u - x; break;
                default: break;
            }
            memcpy(
                dst + ((size_t)dy * dst_w + dx) * 4u,
                src_rgba + ((size_t)y * width + x) * 4u,
                4u
            );
        }
    }
    *out_rgba = dst;
    *out_w = dst_w;
    *out_h = dst_h;
}