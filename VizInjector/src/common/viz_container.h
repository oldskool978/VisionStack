#ifndef VIZINJECTOR_VIZ_CONTAINER_H
#define VIZINJECTOR_VIZ_CONTAINER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VIZJ_MAGIC_STR "VIZJ_CONTAINER01"
#define VIZJ_HEADER_BYTES 144

typedef enum {
    VIZJ_TIER_WIRE_WEBP = 0,
    VIZJ_TIER_VAE_LATENT = 1,
    VIZJ_TIER_DUAL_LATENT = 2
} VizContainerTier;

typedef enum {
    VIZJ_DTYPE_FP32 = 0,
    VIZJ_DTYPE_BF16 = 1,
    VIZJ_DTYPE_FP16 = 2,
    VIZJ_DTYPE_UINT8_WEBP = 3
} VizContainerDtype;

#pragma pack(push, 1)
typedef struct {
    char magic[16];
    uint32_t version;
    uint32_t tier;
    uint32_t vae_dtype;
    uint32_t vit_dtype;
    uint32_t orig_width;
    uint32_t orig_height;
    uint32_t vae_width;
    uint32_t vae_height;
    uint32_t vit_width;
    uint32_t vit_height;
    uint32_t grid_thw[3];
    uint32_t llm_tokens;
    uint64_t vae_offset;
    uint64_t vae_bytes;
    uint64_t vit_offset;
    uint64_t vit_bytes;
    uint64_t mask_offset;
    uint64_t mask_bytes;
    float mean_luminance;
    float rms_contrast;
    float shannon_entropy_bits;
    float laplacian_sharpness;
    uint32_t payload_crc32c;
    uint8_t reserved[4];
} VizContainerHeader;
#pragma pack(pop)

uint32_t viz_crc32c(uint32_t crc, const void *buf, size_t len);

int viz_decode_image_memory_to_linear_f32(
    const uint8_t *file_bytes,
    size_t file_len,
    int deband_8bit,
    uint32_t seed,
    float **out_linear_rgba,
    uint32_t *out_w,
    uint32_t *out_h
);

int viz_decode_image_memory_to_rgba8(
    const uint8_t *file_bytes,
    size_t file_len,
    uint8_t **out_rgba8,
    uint32_t *out_w,
    uint32_t *out_h
);

int viz_encode_rgba8_to_webp(
    const uint8_t *rgba8,
    uint32_t width,
    uint32_t height,
    float quality,
    int lossless,
    uint8_t **out_webp_bytes,
    uint32_t *out_webp_len
);

int viz_encode_rgb8_to_png(
    const uint8_t *rgb8,
    uint32_t width,
    uint32_t height,
    uint8_t **out_png_bytes,
    uint32_t *out_png_len
);

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
);

int viz_unpack_vizj_container(
    const uint8_t *container_bytes,
    size_t container_len,
    VizContainerHeader *out_hdr,
    const void **out_vae_ptr,
    const void **out_vit_ptr,
    const void **out_mask_ptr
);

#ifdef __cplusplus
}
#endif

#endif