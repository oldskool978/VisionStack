#include "vizinjector_core.h"
#include "../common/colorspace_srgb.h"
#include "../vulkan/vk_compute_engine.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(_MSC_VER)
#define VIZ_TLS __declspec(thread)
#else
#define VIZ_TLS __thread
#endif

static VIZ_TLS float *g_tls_src_lin = NULL;
static VIZ_TLS size_t g_tls_src_lin_cap = 0;

static VIZ_TLS float *g_tls_vae_lin = NULL;
static VIZ_TLS size_t g_tls_vae_lin_cap = 0;

static VIZ_TLS float *g_tls_vit_lin = NULL;
static VIZ_TLS size_t g_tls_vit_lin_cap = 0;

static VIZ_TLS float *g_tls_out_vae = NULL;
static VIZ_TLS size_t g_tls_out_vae_cap = 0;

static VIZ_TLS float *g_tls_out_vit = NULL;
static VIZ_TLS size_t g_tls_out_vit_cap = 0;

static VIZ_TLS float *g_tls_out_mask = NULL;
static VIZ_TLS size_t g_tls_out_mask_cap = 0;

static float *acquire_tls_f32(float **slot, size_t *cap, size_t needed_floats) {
    if (needed_floats <= *cap && *slot != NULL) {
        return *slot;
    }
    size_t alloc_floats = (needed_floats + 4095u) & ~(size_t)4095u;
    float *next_buf = (float *)realloc(*slot, alloc_floats * sizeof(float));
    if (!next_buf) {
        return NULL;
    }
    *slot = next_buf;
    *cap = alloc_floats;
    return next_buf;
}

static int is_high_bit_depth_or_hdr_memory(const uint8_t *file_bytes, size_t file_len) {
    static const uint8_t png_sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    if (file_len >= 29u &&
        memcmp(file_bytes, png_sig, 8u) == 0 &&
        memcmp(file_bytes + 12u, "IHDR", 4u) == 0 &&
        file_bytes[24] == 16u) {
        return 1;
    }
    if ((file_len >= 10u && memcmp(file_bytes, "#?RADIANCE", 10u) == 0) ||
        (file_len >= 6u && memcmp(file_bytes, "#?RGBE", 6u) == 0)) {
        return 1;
    }
    return 0;
}

void vizinjector_core_default_config(VizInjectorConfig *cfg) {
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->target_width = 1024u;
    cfg->target_height = 1024u;
    cfg->max_vit_patches = 2304u;
    cfg->framing_mode = VIZ_FRAMING_INSCRIBED;
    cfg->input_role = VIZ_ROLE_PHOTO_REF;
    cfg->edit_mode_384_vit = 0;
    cfg->emit_vae_branch = 1;
    cfg->emit_vit_branch = 1;
    cfg->emit_mask_branch = 1;
    cfg->emit_wire_vizj = 0;
    cfg->deband_8bit = 1;
    cfg->prefer_vulkan = 1;
    cfg->seed = 42u;
    cfg->mtf_boost = 0.08f;
    cfg->soft_knee_tau = 0.015f;
    cfg->mask_feather_radius = 1.25f;
    cfg->webp_quality = 98.0f;
    cfg->webp_lossless = 1;
}

static void enforce_depth_monotonic_planar(
    const float *resampled_lin_rgba,
    uint32_t width,
    uint32_t height,
    float *out_planar_vae
) {
    size_t plane = (size_t)width * (size_t)height;
    float *r_p = out_planar_vae;
    float *g_p = out_planar_vae + plane;
    float *b_p = out_planar_vae + plane * 2u;
    for (size_t i = 0; i < plane; ++i) {
        float r_s = resampled_lin_rgba[i * 4u + 0u];
        float g_s = resampled_lin_rgba[i * 4u + 1u];
        float b_s = resampled_lin_rgba[i * 4u + 2u];
        float d_val = (0.2126f * r_s + 0.7152f * g_s + 0.0722f * b_s) * 2.0f - 1.0f;
        d_val = fminf(1.0f, fmaxf(-1.0f, d_val));
        r_p[i] = d_val;
        g_p[i] = d_val;
        b_p[i] = d_val;
    }
}

static int build_wire_vizj_from_branches(
    const VizInjectorConfig *cfg,
    VizInjectorOutput *out
) {
    if (!out->vae_planar_f32 || out->vae_width == 0 || out->vae_height == 0) return -1;
    size_t px_count = (size_t)out->vae_width * (size_t)out->vae_height;
    uint8_t *rgb8 = (uint8_t *)malloc(px_count * 3u);
    uint8_t *rgba8 = (uint8_t *)malloc(px_count * 4u);
    if (!rgb8 || !rgba8) {
        free(rgb8);
        free(rgba8);
        return -2;
    }
    viz_postprocess_vae_f32_to_rgb8(out->vae_planar_f32, out->vae_width, out->vae_height, 0, rgb8);
    for (size_t i = 0; i < px_count; ++i) {
        rgba8[i * 4u + 0u] = rgb8[i * 3u + 0u];
        rgba8[i * 4u + 1u] = rgb8[i * 3u + 1u];
        rgba8[i * 4u + 2u] = rgb8[i * 3u + 2u];
        rgba8[i * 4u + 3u] = 255u;
    }
    free(rgb8);

    uint8_t *webp_bytes = NULL;
    uint32_t webp_len = 0;
    int rc = viz_encode_rgba8_to_webp(
        rgba8,
        out->vae_width,
        out->vae_height,
        cfg->webp_quality,
        cfg->webp_lossless,
        &webp_bytes,
        &webp_len
    );
    free(rgba8);
    if (rc != 0 || !webp_bytes) return -3;

    VizContainerHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.tier = VIZJ_TIER_WIRE_WEBP;
    hdr.vae_dtype = VIZJ_DTYPE_UINT8_WEBP;
    hdr.vit_dtype = VIZJ_DTYPE_FP32;
    hdr.orig_width = out->orig_width;
    hdr.orig_height = out->orig_height;
    hdr.vae_width = out->vae_width;
    hdr.vae_height = out->vae_height;
    hdr.vit_width = out->vit_width;
    hdr.vit_height = out->vit_height;
    hdr.grid_thw[0] = out->grid_thw[0];
    hdr.grid_thw[1] = out->grid_thw[1];
    hdr.grid_thw[2] = out->grid_thw[2];
    hdr.llm_tokens = out->llm_tokens;
    hdr.mean_luminance = out->mean_luminance;
    hdr.rms_contrast = out->rms_contrast;
    hdr.shannon_entropy_bits = out->shannon_entropy_bits;
    hdr.laplacian_sharpness = out->laplacian_sharpness;

    uint8_t *container_buf = NULL;
    size_t container_len = 0;
    rc = viz_pack_vizj_container(
        &hdr,
        webp_bytes,
        webp_len,
        NULL,
        0,
        out->latent_mask_f32,
        (size_t)out->mask_Floats * sizeof(float),
        &container_buf,
        &container_len
    );
    free(webp_bytes);
    if (rc != 0 || !container_buf) return -4;

    out->wire_vizj_bytes = container_buf;
    out->wire_vizj_len = (uint32_t)container_len;
    return 0;
}

int32_t vizinjector_core_process_linear_f32(
    const float *src_linear_rgba,
    uint32_t src_w,
    uint32_t src_h,
    const VizInjectorConfig *cfg_in,
    VizInjectorOutput *out
) {
    if (!src_linear_rgba || src_w == 0 || src_h == 0 || !out) return -1;
    memset(out, 0, sizeof(*out));

    VizInjectorConfig cfg;
    if (cfg_in) cfg = *cfg_in;
    else vizinjector_core_default_config(&cfg);

    out->orig_width = src_w;
    out->orig_height = src_h;

    uint32_t vae_w = 0, vae_h = 0, vit_w = 0, vit_h = 0;
    viz_solve_coupled_grid(
        src_w,
        src_h,
        cfg.target_width,
        cfg.target_height,
        cfg.max_vit_patches,
        cfg.edit_mode_384_vit,
        (VizFramingMode)cfg.framing_mode,
        &vae_w,
        &vae_h,
        &vit_w,
        &vit_h
    );

    out->vae_width = vae_w;
    out->vae_height = vae_h;
    out->vit_width = vit_w;
    out->vit_height = vit_h;
    out->mask_width = vae_w / 16u;
    out->mask_height = vae_h / 16u;

    float mtf = (cfg.input_role == VIZ_ROLE_PHOTO_REF) ? cfg.mtf_boost : 0.0f;
    float knee = (cfg.input_role == VIZ_ROLE_DEPTH_MAP) ? 0.005f : cfg.soft_knee_tau;

    float *vae_lin = NULL;
    if (cfg.emit_vae_branch || cfg.emit_mask_branch) {
        vae_lin = acquire_tls_f32(&g_tls_vae_lin, &g_tls_vae_lin_cap, (size_t)vae_w * (size_t)vae_h * 4u);
        if (!vae_lin) return -2;
        if (viz_resample_2d_linear_rgba(src_linear_rgba, src_w, src_h, vae_w, vae_h, (VizFramingMode)cfg.framing_mode, mtf, knee, vae_lin) != 0) {
            return -3;
        }
    }

    if (cfg.emit_vae_branch && vae_lin) {
        size_t vae_floats = (size_t)vae_w * (size_t)vae_h * 3u;
        out->vae_planar_f32 = acquire_tls_f32(&g_tls_out_vae, &g_tls_out_vae_cap, vae_floats);
        if (!out->vae_planar_f32) {
            return -4;
        }
        out->vae_Floats = (uint32_t)vae_floats;

        if (cfg.input_role == VIZ_ROLE_DEPTH_MAP) {
            enforce_depth_monotonic_planar(vae_lin, vae_w, vae_h, out->vae_planar_f32);
        } else {
            viz_pack_vae_planar_f32(vae_lin, vae_w, vae_h, out->vae_planar_f32);
        }

        VizPerceptualTelemetry telem;
        viz_compute_perceptual_telemetry(out->vae_planar_f32, vae_w, vae_h, &telem);
        out->mean_luminance = telem.mean_luminance;
        out->rms_contrast = telem.rms_contrast;
        out->shannon_entropy_bits = telem.shannon_entropy_bits;
        out->laplacian_sharpness = telem.laplacian_sharpness;
    }

    if (cfg.emit_mask_branch && vae_lin) {
        uint32_t mw = vae_w / 16u;
        uint32_t mh = vae_h / 16u;
        size_t m_floats = (size_t)mw * (size_t)mh;
        out->latent_mask_f32 = acquire_tls_f32(&g_tls_out_mask, &g_tls_out_mask_cap, m_floats);
        if (out->latent_mask_f32) {
            int ch_off = (cfg.input_role == VIZ_ROLE_INPAINT_MASK) ? 0 : 3;
            viz_downsample_latent_mask_16x(vae_lin, 4, ch_off, vae_w, vae_h, cfg.mask_feather_radius, out->latent_mask_f32);
            out->mask_Floats = (uint32_t)m_floats;
        }
    }

    if (cfg.emit_vit_branch && cfg.input_role == VIZ_ROLE_PHOTO_REF) {
        uint32_t grid_h = vit_h / 16u;
        uint32_t grid_w = vit_w / 16u;
        uint32_t num_patches = grid_h * grid_w;
        size_t vit_floats = (size_t)num_patches * 3u * 2u * 256u;
        out->vit_patches_f32 = acquire_tls_f32(&g_tls_out_vit, &g_tls_out_vit_cap, vit_floats);
        if (!out->vit_patches_f32) {
            vizinjector_core_free_output(out);
            return -5;
        }
        out->vit_num_patches = num_patches;
        out->vit_Floats = (uint32_t)vit_floats;

        if (vit_w == vae_w && vit_h == vae_h && vae_lin) {
            viz_pack_qwen35_vit_patches_f32(vae_lin, vit_w, vit_h, out->vit_patches_f32, out->grid_thw, &out->llm_tokens);
        } else {
            float *vit_lin = acquire_tls_f32(&g_tls_vit_lin, &g_tls_vit_lin_cap, (size_t)vit_w * (size_t)vit_h * 4u);
            if (!vit_lin) {
                vizinjector_core_free_output(out);
                return -6;
            }
            viz_resample_2d_linear_rgba(src_linear_rgba, src_w, src_h, vit_w, vit_h, (VizFramingMode)cfg.framing_mode, mtf, knee, vit_lin);
            viz_pack_qwen35_vit_patches_f32(vit_lin, vit_w, vit_h, out->vit_patches_f32, out->grid_thw, &out->llm_tokens);
        }
    }

    if (cfg.emit_wire_vizj && out->vae_planar_f32) {
        build_wire_vizj_from_branches(&cfg, out);
    }

    out->backend_used = 0;
    out->status_code = 0;
    return 0;
}

int32_t vizinjector_core_process_rgba8(
    const uint8_t *src_rgba8,
    uint32_t src_w,
    uint32_t src_h,
    const VizInjectorConfig *cfg_in,
    VizInjectorOutput *out
) {
    if (!src_rgba8 || src_w == 0 || src_h == 0 || !out) return -1;
    memset(out, 0, sizeof(*out));

    VizInjectorConfig cfg;
    if (cfg_in) cfg = *cfg_in;
    else vizinjector_core_default_config(&cfg);

    out->orig_width = src_w;
    out->orig_height = src_h;

    uint32_t vae_w = 0, vae_h = 0, vit_w = 0, vit_h = 0;
    viz_solve_coupled_grid(
        src_w,
        src_h,
        cfg.target_width,
        cfg.target_height,
        cfg.max_vit_patches,
        cfg.edit_mode_384_vit,
        (VizFramingMode)cfg.framing_mode,
        &vae_w,
        &vae_h,
        &vit_w,
        &vit_h
    );
    out->vae_width = vae_w;
    out->vae_height = vae_h;
    out->vit_width = vit_w;
    out->vit_height = vit_h;
    out->mask_width = vae_w / 16u;
    out->mask_height = vae_h / 16u;

    if (cfg.input_role == VIZ_ROLE_COLOR_MAP) {
        size_t vae_floats = (size_t)vae_w * (size_t)vae_h * 3u;
        out->vae_planar_f32 = acquire_tls_f32(&g_tls_out_vae, &g_tls_out_vae_cap, vae_floats);
        if (!out->vae_planar_f32) return -2;
        out->vae_Floats = (uint32_t)vae_floats;
        viz_resample_majority_color_map_rgba8(src_rgba8, src_w, src_h, vae_w, vae_h, (VizFramingMode)cfg.framing_mode, out->vae_planar_f32);
        VizPerceptualTelemetry telem;
        viz_compute_perceptual_telemetry(out->vae_planar_f32, vae_w, vae_h, &telem);
        out->mean_luminance = telem.mean_luminance;
        out->rms_contrast = telem.rms_contrast;
        out->shannon_entropy_bits = telem.shannon_entropy_bits;
        out->laplacian_sharpness = telem.laplacian_sharpness;
        out->backend_used = 0;
        out->status_code = 0;
        return 0;
    }

    if (cfg.prefer_vulkan && cfg.input_role == VIZ_ROLE_PHOTO_REF && viz_vk_is_available()) {
        size_t vae_floats = cfg.emit_vae_branch ? ((size_t)vae_w * (size_t)vae_h * 3u) : 0u;
        uint32_t grid_h = vit_h / 16u;
        uint32_t grid_w = vit_w / 16u;
        uint32_t num_patches = grid_h * grid_w;
        size_t vit_floats = cfg.emit_vit_branch ? ((size_t)num_patches * 3u * 2u * 256u) : 0u;
        uint32_t mw = vae_w / 16u;
        uint32_t mh = vae_h / 16u;
        size_t m_floats = cfg.emit_mask_branch ? ((size_t)mw * (size_t)mh) : 0u;

        if (vae_floats > 0u) out->vae_planar_f32 = acquire_tls_f32(&g_tls_out_vae, &g_tls_out_vae_cap, vae_floats);
        if (vit_floats > 0u) out->vit_patches_f32 = acquire_tls_f32(&g_tls_out_vit, &g_tls_out_vit_cap, vit_floats);
        if (m_floats > 0u) out->latent_mask_f32 = acquire_tls_f32(&g_tls_out_mask, &g_tls_out_mask_cap, m_floats);

        int vk_rc = viz_vk_condition_image_rgba8(
            src_rgba8,
            src_w,
            src_h,
            vae_w,
            vae_h,
            vit_w,
            vit_h,
            (VizFramingMode)cfg.framing_mode,
            cfg.deband_8bit,
            cfg.seed,
            cfg.mtf_boost,
            cfg.soft_knee_tau,
            cfg.mask_feather_radius,
            out->vae_planar_f32,
            out->vit_patches_f32,
            out->grid_thw,
            &out->llm_tokens,
            out->latent_mask_f32
        );

        if (vk_rc == 0) {
            if (cfg.emit_vae_branch && out->vae_planar_f32) {
                out->vae_Floats = (uint32_t)vae_floats;
                VizPerceptualTelemetry telem;
                viz_compute_perceptual_telemetry(out->vae_planar_f32, vae_w, vae_h, &telem);
                out->mean_luminance = telem.mean_luminance;
                out->rms_contrast = telem.rms_contrast;
                out->shannon_entropy_bits = telem.shannon_entropy_bits;
                out->laplacian_sharpness = telem.laplacian_sharpness;
            }
            if (cfg.emit_vit_branch && out->vit_patches_f32) {
                out->vit_num_patches = num_patches;
                out->vit_Floats = (uint32_t)vit_floats;
            }
            if (cfg.emit_mask_branch && out->latent_mask_f32) {
                out->mask_Floats = (uint32_t)m_floats;
            }
            if (cfg.emit_wire_vizj && out->vae_planar_f32) {
                build_wire_vizj_from_branches(&cfg, out);
            }
            out->backend_used = 1;
            out->status_code = 0;
            return 0;
        }
        vizinjector_core_free_output(out);
    }

    float *src_lin = acquire_tls_f32(&g_tls_src_lin, &g_tls_src_lin_cap, (size_t)src_w * (size_t)src_h * 4u);
    if (!src_lin) return -3;
    if (cfg.input_role == VIZ_ROLE_INPAINT_MASK || cfg.input_role == VIZ_ROLE_DEPTH_MAP) {
        viz_ingest_scalar_rgba8_to_f32(src_rgba8, src_w, src_h, src_lin);
    } else {
        viz_ingest_rgba8_to_linear_f32(src_rgba8, src_w, src_h, cfg.deband_8bit, cfg.seed, src_lin);
    }
    return vizinjector_core_process_linear_f32(src_lin, src_w, src_h, &cfg, out);
}

int32_t vizinjector_core_process_memory(
    const uint8_t *file_bytes,
    size_t file_len,
    const VizInjectorConfig *cfg_in,
    VizInjectorOutput *out
) {
    if (!file_bytes || file_len < 8 || !out) return -1;
    VizInjectorConfig cfg;
    if (cfg_in) cfg = *cfg_in;
    else vizinjector_core_default_config(&cfg);

    if (file_len >= sizeof(VizContainerHeader) && memcmp(file_bytes, VIZJ_MAGIC_STR, 16) == 0) {
        VizContainerHeader hdr;
        const void *vae_ptr = NULL;
        const void *vit_ptr = NULL;
        const void *mask_ptr = NULL;
        if (viz_unpack_vizj_container(file_bytes, file_len, &hdr, &vae_ptr, &vit_ptr, &mask_ptr) != 0) {
            return -2;
        }
        if (hdr.tier == VIZJ_TIER_WIRE_WEBP && vae_ptr && hdr.vae_bytes > 0) {
            VizInjectorConfig wire_cfg = cfg;
            if (hdr.vae_width > 0u && hdr.vae_height > 0u) {
                wire_cfg.target_width = hdr.vae_width;
                wire_cfg.target_height = hdr.vae_height;
            }
            wire_cfg.deband_8bit = 0;
            wire_cfg.mtf_boost = 0.0f;
            int32_t rc = vizinjector_core_process_memory((const uint8_t *)vae_ptr, (size_t)hdr.vae_bytes, &wire_cfg, out);
            if (rc == 0) {
                if (hdr.orig_width > 0u && hdr.orig_height > 0u) {
                    out->orig_width = hdr.orig_width;
                    out->orig_height = hdr.orig_height;
                }
                if (mask_ptr && hdr.mask_bytes == (size_t)(out->vae_width / 16u) * (out->vae_height / 16u) * sizeof(float)) {
                    size_t m_floats = (size_t)(out->vae_width / 16u) * (out->vae_height / 16u);
                    if (!out->latent_mask_f32) {
                        out->latent_mask_f32 = acquire_tls_f32(&g_tls_out_mask, &g_tls_out_mask_cap, m_floats);
                    }
                    if (out->latent_mask_f32) {
                        memcpy(out->latent_mask_f32, mask_ptr, (size_t)hdr.mask_bytes);
                        out->mask_width = out->vae_width / 16u;
                        out->mask_height = out->vae_height / 16u;
                        out->mask_Floats = out->mask_width * out->mask_height;
                    }
                }
            }
            return rc;
        }
    }

    if (cfg.input_role != VIZ_ROLE_PHOTO_REF ||
        (cfg.prefer_vulkan && viz_vk_is_available() && !is_high_bit_depth_or_hdr_memory(file_bytes, file_len))) {
        uint8_t *rgba8 = NULL;
        uint32_t w = 0, h = 0;
        if (viz_decode_image_memory_to_rgba8(file_bytes, file_len, &rgba8, &w, &h) == 0 && rgba8) {
            int32_t rc = vizinjector_core_process_rgba8(rgba8, w, h, &cfg, out);
            free(rgba8);
            return rc;
        }
    }

    float *lin_rgba = NULL;
    uint32_t w = 0, h = 0;
    int dec_rc = viz_decode_image_memory_to_linear_f32(file_bytes, file_len, cfg.deband_8bit, cfg.seed, &lin_rgba, &w, &h);
    if (dec_rc != 0 || !lin_rgba) return dec_rc;
    int32_t rc = vizinjector_core_process_linear_f32(lin_rgba, w, h, &cfg, out);
    free(lin_rgba);
    return rc;
}

void vizinjector_core_free_output(VizInjectorOutput *out) {
    if (!out) return;
    if (out->vae_planar_f32 && out->vae_planar_f32 != g_tls_out_vae) {
        free(out->vae_planar_f32);
    }
    if (out->vit_patches_f32 && out->vit_patches_f32 != g_tls_out_vit) {
        free(out->vit_patches_f32);
    }
    if (out->latent_mask_f32 && out->latent_mask_f32 != g_tls_out_mask) {
        free(out->latent_mask_f32);
    }
    free(out->wire_vizj_bytes);
    memset(out, 0, sizeof(*out));
}