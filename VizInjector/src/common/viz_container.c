#include "viz_container.h"
#include "colorspace_srgb.h"

#include <spng.h>
#include <webp/decode.h>
#include <webp/encode.h>
#include <stb_image.h>
#include <stdlib.h>
#include <string.h>

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__AVX2__) || defined(__SSE4_2__))
#include <immintrin.h>
#define VIZ_HAS_HW_CRC32C 1
#else
#define VIZ_HAS_HW_CRC32C 0
#endif

typedef char viz_header_size_assertion[(sizeof(VizContainerHeader) == VIZJ_HEADER_BYTES) ? 1 : -1];

#if !VIZ_HAS_HW_CRC32C
static uint32_t g_crc32c_table[256];
static int g_crc_init = 0;

static void init_crc32c_table(void) {
    if (g_crc_init) return;
    for (uint32_t i = 0; i < 256u; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k) {
            c = (c & 1u) ? (0x82F63B78u ^ (c >> 1u)) : (c >> 1u);
        }
        g_crc32c_table[i] = c;
    }
    g_crc_init = 1;
}
#endif

uint32_t viz_crc32c(uint32_t crc, const void *buf, size_t len) {
    const uint8_t *p = (const uint8_t *)buf;
    uint32_t c = ~crc;
#if VIZ_HAS_HW_CRC32C
    uint64_t c64 = (uint64_t)c;
    while (len >= 32u) {
        uint64_t w0, w1, w2, w3;
        memcpy(&w0, p + 0, 8);
        memcpy(&w1, p + 8, 8);
        memcpy(&w2, p + 16, 8);
        memcpy(&w3, p + 24, 8);
        c64 = _mm_crc32_u64(c64, w0);
        c64 = _mm_crc32_u64(c64, w1);
        c64 = _mm_crc32_u64(c64, w2);
        c64 = _mm_crc32_u64(c64, w3);
        p += 32u;
        len -= 32u;
    }
    while (len >= 8u) {
        uint64_t w;
        memcpy(&w, p, 8);
        c64 = _mm_crc32_u64(c64, w);
        p += 8u;
        len -= 8u;
    }
    c = (uint32_t)c64;
    while (len > 0u) {
        c = _mm_crc32_u8(c, *p++);
        len--;
    }
#else
    init_crc32c_table();
    for (size_t i = 0; i < len; ++i) {
        c = g_crc32c_table[(c ^ p[i]) & 0xFFu] ^ (c >> 8u);
    }
#endif
    return ~c;
}

static int is_png_signature(const uint8_t *b, size_t n) {
    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    return (n >= 8 && memcmp(b, sig, 8) == 0);
}

static int is_webp_signature(const uint8_t *b, size_t n) {
    return (n >= 12 && memcmp(b, "RIFF", 4) == 0 && memcmp(b + 8, "WEBP", 4) == 0);
}

int viz_decode_image_memory_to_rgba8(
    const uint8_t *file_bytes,
    size_t file_len,
    uint8_t **out_rgba8,
    uint32_t *out_w,
    uint32_t *out_h
) {
    if (!file_bytes || file_len < 8 || !out_rgba8 || !out_w || !out_h) return -1;
    *out_rgba8 = NULL;
    *out_w = 0;
    *out_h = 0;

    if (is_png_signature(file_bytes, file_len)) {
        spng_ctx *ctx = spng_ctx_new(0);
        if (!ctx) return -2;
        spng_set_crc_action(ctx, SPNG_CRC_USE, SPNG_CRC_USE);
        if (spng_set_png_buffer(ctx, file_bytes, file_len) != 0) {
            spng_ctx_free(ctx);
            return -3;
        }
        struct spng_ihdr ihdr;
        if (spng_get_ihdr(ctx, &ihdr) != 0) {
            spng_ctx_free(ctx);
            return -4;
        }
        size_t out_sz = 0;
        if (spng_decoded_image_size(ctx, SPNG_FMT_RGBA8, &out_sz) != 0) {
            spng_ctx_free(ctx);
            return -5;
        }
        uint8_t *rgba = (uint8_t *)malloc(out_sz);
        if (!rgba) {
            spng_ctx_free(ctx);
            return -6;
        }
        if (spng_decode_image(ctx, rgba, out_sz, SPNG_FMT_RGBA8, SPNG_DECODE_TRNS) != 0) {
            free(rgba);
            spng_ctx_free(ctx);
            return -7;
        }
        spng_ctx_free(ctx);
        *out_rgba8 = rgba;
        *out_w = ihdr.width;
        *out_h = ihdr.height;
        return 0;
    }

    if (is_webp_signature(file_bytes, file_len)) {
        int w = 0, h = 0;
        uint8_t *webp_rgba = WebPDecodeRGBA(file_bytes, file_len, &w, &h);
        if (!webp_rgba || w <= 0 || h <= 0) {
            if (webp_rgba) WebPFree(webp_rgba);
            return -8;
        }
        size_t sz = (size_t)w * (size_t)h * 4u;
        uint8_t *rgba = (uint8_t *)malloc(sz);
        if (!rgba) {
            WebPFree(webp_rgba);
            return -9;
        }
        memcpy(rgba, webp_rgba, sz);
        WebPFree(webp_rgba);
        *out_rgba8 = rgba;
        *out_w = (uint32_t)w;
        *out_h = (uint32_t)h;
        return 0;
    }

    int w = 0, h = 0, comp = 0;
    stbi_uc *stb_px = stbi_load_from_memory(file_bytes, (int)file_len, &w, &h, &comp, 4);
    if (!stb_px || w <= 0 || h <= 0) {
        if (stb_px) stbi_image_free(stb_px);
        return -10;
    }
    int orient = viz_parse_exif_orientation(file_bytes, file_len);
    uint8_t *oriented_rgba = NULL;
    uint32_t ow = 0, oh = 0;
    viz_apply_exif_orientation_rgba8(stb_px, (uint32_t)w, (uint32_t)h, orient, &oriented_rgba, &ow, &oh);
    stbi_image_free(stb_px);
    if (!oriented_rgba) return -11;
    *out_rgba8 = oriented_rgba;
    *out_w = ow;
    *out_h = oh;
    return 0;
}

int viz_decode_image_memory_to_linear_f32(
    const uint8_t *file_bytes,
    size_t file_len,
    int deband_8bit,
    uint32_t seed,
    float **out_linear_rgba,
    uint32_t *out_w,
    uint32_t *out_h
) {
    if (!file_bytes || file_len < 8 || !out_linear_rgba || !out_w || !out_h) return -1;
    *out_linear_rgba = NULL;
    *out_w = 0;
    *out_h = 0;

    if (is_png_signature(file_bytes, file_len)) {
        spng_ctx *ctx = spng_ctx_new(0);
        if (!ctx) return -2;
        spng_set_crc_action(ctx, SPNG_CRC_USE, SPNG_CRC_USE);
        if (spng_set_png_buffer(ctx, file_bytes, file_len) == 0) {
            struct spng_ihdr ihdr;
            if (spng_get_ihdr(ctx, &ihdr) == 0 && ihdr.bit_depth == 16) {
                size_t sz16 = 0;
                if (spng_decoded_image_size(ctx, SPNG_FMT_RGBA16, &sz16) == 0) {
                    uint16_t *rgba16 = (uint16_t *)malloc(sz16);
                    if (rgba16 && spng_decode_image(ctx, rgba16, sz16, SPNG_FMT_RGBA16, SPNG_DECODE_TRNS) == 0) {
                        float *lin = (float *)malloc((size_t)ihdr.width * (size_t)ihdr.height * 4u * sizeof(float));
                        if (lin) {
                            viz_ingest_rgba16_to_linear_f32(rgba16, ihdr.width, ihdr.height, lin);
                            free(rgba16);
                            spng_ctx_free(ctx);
                            *out_linear_rgba = lin;
                            *out_w = ihdr.width;
                            *out_h = ihdr.height;
                            return 0;
                        }
                    }
                    free(rgba16);
                }
            }
        }
        spng_ctx_free(ctx);
    }

    if (stbi_is_hdr_from_memory(file_bytes, (int)file_len)) {
        int w = 0, h = 0, comp = 0;
        float *hdr_px = stbi_loadf_from_memory(file_bytes, (int)file_len, &w, &h, &comp, 4);
        if (hdr_px && w > 0 && h > 0) {
            float *lin = (float *)malloc((size_t)w * (size_t)h * 4u * sizeof(float));
            if (lin) {
                viz_ingest_rgbaf32_to_linear_f32(hdr_px, (uint32_t)w, (uint32_t)h, 1, lin);
                stbi_image_free(hdr_px);
                *out_linear_rgba = lin;
                *out_w = (uint32_t)w;
                *out_h = (uint32_t)h;
                return 0;
            }
        }
        if (hdr_px) stbi_image_free(hdr_px);
    }

    uint8_t *rgba8 = NULL;
    uint32_t w = 0, h = 0;
    int rc = viz_decode_image_memory_to_rgba8(file_bytes, file_len, &rgba8, &w, &h);
    if (rc != 0 || !rgba8) return rc;
    float *lin = (float *)malloc((size_t)w * (size_t)h * 4u * sizeof(float));
    if (!lin) {
        free(rgba8);
        return -12;
    }
    viz_ingest_rgba8_to_linear_f32(rgba8, w, h, deband_8bit, seed, lin);
    free(rgba8);
    *out_linear_rgba = lin;
    *out_w = w;
    *out_h = h;
    return 0;
}

int viz_encode_rgba8_to_webp(
    const uint8_t *rgba8,
    uint32_t width,
    uint32_t height,
    float quality,
    int lossless,
    uint8_t **out_webp_bytes,
    uint32_t *out_webp_len
) {
    if (!rgba8 || width == 0 || height == 0 || !out_webp_bytes || !out_webp_len) return -1;
    uint8_t *webp_buf = NULL;
    size_t sz = lossless
        ? WebPEncodeLosslessRGBA(rgba8, (int)width, (int)height, (int)(width * 4u), &webp_buf)
        : WebPEncodeRGBA(rgba8, (int)width, (int)height, (int)(width * 4u), quality, &webp_buf);
    if (sz == 0 || !webp_buf) return -2;
    uint8_t *owned = (uint8_t *)malloc(sz);
    if (!owned) {
        WebPFree(webp_buf);
        return -3;
    }
    memcpy(owned, webp_buf, sz);
    WebPFree(webp_buf);
    *out_webp_bytes = owned;
    *out_webp_len = (uint32_t)sz;
    return 0;
}

int viz_encode_rgb8_to_png(
    const uint8_t *rgb8,
    uint32_t width,
    uint32_t height,
    uint8_t **out_png_bytes,
    uint32_t *out_png_len
) {
    if (!rgb8 || width == 0 || height == 0 || !out_png_bytes || !out_png_len) return -1;
    spng_ctx *ctx = spng_ctx_new(SPNG_CTX_ENCODER);
    if (!ctx) return -2;
    spng_set_option(ctx, SPNG_ENCODE_TO_BUFFER, 1);
    struct spng_ihdr ihdr;
    memset(&ihdr, 0, sizeof(ihdr));
    ihdr.width = width;
    ihdr.height = height;
    ihdr.bit_depth = 8;
    ihdr.color_type = SPNG_COLOR_TYPE_TRUECOLOR;
    spng_set_ihdr(ctx, &ihdr);
    int rc = spng_encode_image(ctx, rgb8, (size_t)width * (size_t)height * 3u, SPNG_FMT_PNG, SPNG_ENCODE_FINALIZE);
    if (rc != 0) {
        spng_ctx_free(ctx);
        return -3;
    }
    size_t png_sz = 0;
    int err = 0;
    void *png_buf = spng_get_png_buffer(ctx, &png_sz, &err);
    spng_ctx_free(ctx);
    if (!png_buf || png_sz == 0) {
        free(png_buf);
        return -4;
    }
    *out_png_bytes = (uint8_t *)png_buf;
    *out_png_len = (uint32_t)png_sz;
    return 0;
}

static size_t align_up_64(size_t v) {
    return (v + 63u) & ~(size_t)63u;
}

int viz_pack_vizj_container(
    const VizContainerHeader *hdr_template,
    const void *vae_payload,
    size_t vae_bytes,
    const void *vit_payload,
    size_t vit_bytes,
    const void *mask_payload,
    size_t mask_bytes,
    uint8_t **out_container_bytes,
    size_t *out_container_len
) {
    if (!hdr_template || !out_container_bytes || !out_container_len) return -1;
    size_t hdr_end = sizeof(VizContainerHeader);
    size_t cursor = align_up_64(hdr_end);
    uint64_t vae_off = vae_bytes > 0 ? cursor : 0;
    size_t vae_end = cursor + vae_bytes;
    cursor = align_up_64(vae_end);
    uint64_t vit_off = vit_bytes > 0 ? cursor : 0;
    size_t vit_end = cursor + vit_bytes;
    cursor = align_up_64(vit_end);
    uint64_t mask_off = mask_bytes > 0 ? cursor : 0;
    size_t mask_end = cursor + mask_bytes;
    size_t total_len = align_up_64(mask_end);

    uint8_t *buf = (uint8_t *)malloc(total_len);
    if (!buf) return -2;

    if (vae_off > hdr_end) memset(buf + hdr_end, 0, (size_t)vae_off - hdr_end);
    else if (cursor > hdr_end && vae_bytes == 0) memset(buf + hdr_end, 0, cursor - hdr_end);

    if (vae_bytes > 0 && vae_payload) {
        memcpy(buf + vae_off, vae_payload, vae_bytes);
    }
    size_t after_vae_align = align_up_64(vae_end);
    if (after_vae_align > vae_end) {
        memset(buf + vae_end, 0, after_vae_align - vae_end);
    }

    if (vit_bytes > 0 && vit_payload) {
        memcpy(buf + vit_off, vit_payload, vit_bytes);
    }
    size_t after_vit_align = align_up_64(vit_end);
    if (after_vit_align > vit_end) {
        memset(buf + vit_end, 0, after_vit_align - vit_end);
    }

    if (mask_bytes > 0 && mask_payload) {
        memcpy(buf + mask_off, mask_payload, mask_bytes);
    }
    if (total_len > mask_end) {
        memset(buf + mask_end, 0, total_len - mask_end);
    }

    VizContainerHeader hdr = *hdr_template;
    memcpy(hdr.magic, VIZJ_MAGIC_STR, 16);
    hdr.version = 1u;
    hdr.vae_offset = vae_off;
    hdr.vae_bytes = vae_bytes;
    hdr.vit_offset = vit_off;
    hdr.vit_bytes = vit_bytes;
    hdr.mask_offset = mask_off;
    hdr.mask_bytes = mask_bytes;
    hdr.payload_crc32c = viz_crc32c(0u, buf + sizeof(VizContainerHeader), total_len - sizeof(VizContainerHeader));
    memcpy(buf, &hdr, sizeof(VizContainerHeader));

    *out_container_bytes = buf;
    *out_container_len = total_len;
    return 0;
}

int viz_unpack_vizj_container(
    const uint8_t *container_bytes,
    size_t container_len,
    VizContainerHeader *out_hdr,
    const void **out_vae_ptr,
    const void **out_vit_ptr,
    const void **out_mask_ptr
) {
    if (!container_bytes || container_len < sizeof(VizContainerHeader) || !out_hdr) return -1;
    memcpy(out_hdr, container_bytes, sizeof(VizContainerHeader));
    if (memcmp(out_hdr->magic, VIZJ_MAGIC_STR, 16) != 0) return -2;
    uint32_t calc_crc = viz_crc32c(
        0u,
        container_bytes + sizeof(VizContainerHeader),
        container_len - sizeof(VizContainerHeader)
    );
    if (calc_crc != out_hdr->payload_crc32c) return -3;
    if (out_hdr->vae_offset + out_hdr->vae_bytes > container_len) return -4;
    if (out_hdr->vit_offset + out_hdr->vit_bytes > container_len) return -5;
    if (out_hdr->mask_offset + out_hdr->mask_bytes > container_len) return -6;
    if (out_vae_ptr) *out_vae_ptr = out_hdr->vae_bytes ? (container_bytes + out_hdr->vae_offset) : NULL;
    if (out_vit_ptr) *out_vit_ptr = out_hdr->vit_bytes ? (container_bytes + out_hdr->vit_offset) : NULL;
    if (out_mask_ptr) *out_mask_ptr = out_hdr->mask_bytes ? (container_bytes + out_hdr->mask_offset) : NULL;
    return 0;
}