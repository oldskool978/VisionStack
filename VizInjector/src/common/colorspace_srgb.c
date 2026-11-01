#include "colorspace_srgb.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

static float g_srgb_u8_to_lin_lut[256];
static float g_lin_to_srgb_lut[65538];
static int g_tables_initialized = 0;

float viz_srgb_to_linear_f32(float srgb_val) {
    if (srgb_val <= 0.0f) return 0.0f;
    if (srgb_val >= 1.0f) return 1.0f;
    if (srgb_val <= 0.04045f) {
        return srgb_val * (1.0f / 12.92f);
    }
    return powf((srgb_val + 0.055f) * (1.0f / 1.055f), 2.4f);
}

float viz_linear_to_srgb_f32(float linear_val) {
    if (linear_val <= 0.0f) return 0.0f;
    if (linear_val >= 1.0f) return 1.0f;
    if (linear_val <= 0.0031308f) {
        return linear_val * 12.92f;
    }
    return 1.055f * powf(linear_val, 1.0f / 2.4f) - 0.055f;
}

void viz_colorspace_init_tables(void) {
    if (__builtin_expect(g_tables_initialized, 1)) {
        return;
    }
    for (int i = 0; i < 256; ++i) {
        g_srgb_u8_to_lin_lut[i] = viz_srgb_to_linear_f32((float)i * (1.0f / 255.0f));
    }
    for (int i = 0; i <= 65536; ++i) {
        g_lin_to_srgb_lut[i] = viz_linear_to_srgb_f32((float)i * (1.0f / 65536.0f));
    }
    g_lin_to_srgb_lut[65537] = 1.0f;
    g_tables_initialized = 1;
}

const float *viz_get_srgb_to_lin_lut(void) {
    viz_colorspace_init_tables();
    return g_srgb_u8_to_lin_lut;
}

const float *viz_get_lin_to_srgb_lut(void) {
    viz_colorspace_init_tables();
    return g_lin_to_srgb_lut;
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

        int y0 = (y >= 2u) ? ((int)y - 2) : 0;
        int y1 = (y + 2u < height) ? ((int)y + 2) : ((int)height - 1);
        int x0 = (x >= 2u) ? ((int)x - 2) : 0;
        int x1 = (x + 2u < width) ? ((int)x + 2) : ((int)width - 1);

        for (int ny = y0; ny <= y1; ++ny) {
            int dy = ny - (int)y;
            int ady = dy < 0 ? -dy : dy;
            const uint8_t *nrow = src_rgba8 + (size_t)ny * (size_t)width * 4u;
            for (int nx = x0; nx <= x1; ++nx) {
                const uint8_t *np = nrow + (size_t)nx * 4u;
                if (np[3] > 1u) {
                    int dx = nx - (int)x;
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
                dst[c] = viz_srgb_to_linear_f32(s);
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
        }

        for (; x < width; ++x) {
            ingest_pixel_scalar(src_rgba8, row_src, row_dst, x, y, width, height, deband_8bit, seed);
        }
    }
}

void viz_ingest_rgba16_to_linear_f32(
    const uint16_t *src_rgba16,
    uint32_t width,
    uint32_t height,
    float *out_linear_rgba
) {
    size_t count = (size_t)width * (size_t)height;
    for (size_t i = 0; i < count; ++i) {
        float r = (float)src_rgba16[i * 4u + 0u] * (1.0f / 65535.0f);
        float g = (float)src_rgba16[i * 4u + 1u] * (1.0f / 65535.0f);
        float b = (float)src_rgba16[i * 4u + 2u] * (1.0f / 65535.0f);
        float a = (float)src_rgba16[i * 4u + 3u] * (1.0f / 65535.0f);
        out_linear_rgba[i * 4u + 0u] = viz_srgb_to_linear_f32(r);
        out_linear_rgba[i * 4u + 1u] = viz_srgb_to_linear_f32(g);
        out_linear_rgba[i * 4u + 2u] = viz_srgb_to_linear_f32(b);
        out_linear_rgba[i * 4u + 3u] = a;
    }
}

static inline float aces_tonemap_channel(float x) {
    if (x <= 0.0f) return 0.0f;
    if (x <= 1.0f) return x;
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    float mapped = (x * (a * x + b)) / (x * (c * x + d) + e);
    return (mapped > 1.0f) ? 1.0f : mapped;
}

void viz_ingest_rgbaf32_to_linear_f32(
    const float *src_rgbaf32,
    uint32_t width,
    uint32_t height,
    int input_is_linear,
    float *out_linear_rgba
) {
    size_t count = (size_t)width * (size_t)height;
    for (size_t i = 0; i < count; ++i) {
        float r = src_rgbaf32[i * 4u + 0u];
        float g = src_rgbaf32[i * 4u + 1u];
        float b = src_rgbaf32[i * 4u + 2u];
        float a = src_rgbaf32[i * 4u + 3u];
        if (!input_is_linear) {
            r = viz_srgb_to_linear_f32(r);
            g = viz_srgb_to_linear_f32(g);
            b = viz_srgb_to_linear_f32(b);
        } else {
            r = aces_tonemap_channel(r);
            g = aces_tonemap_channel(g);
            b = aces_tonemap_channel(b);
        }
        if (a < 0.0f) a = 0.0f;
        if (a > 1.0f) a = 1.0f;
        out_linear_rgba[i * 4u + 0u] = r;
        out_linear_rgba[i * 4u + 1u] = g;
        out_linear_rgba[i * 4u + 2u] = b;
        out_linear_rgba[i * 4u + 3u] = a;
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