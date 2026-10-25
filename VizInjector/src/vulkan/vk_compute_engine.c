#include "vk_compute_engine.h"
#include "shaders_spv.h"
#include <vulkan/vulkan.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define VIZ_VK_MAX_DIM 2048u
#define VIZ_VK_MAX_PIXELS ((size_t)VIZ_VK_MAX_DIM * (size_t)VIZ_VK_MAX_DIM)
#define VIZ_VK_SCHED_MAX_LEN (VIZ_VK_MAX_DIM * 4u)
#define VIZ_VK_SCHED_MAX_WEIGHTS (VIZ_VK_SCHED_MAX_LEN * 64u)

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t deband_8bit;
    uint32_t seed;
} VizPushIngest;

typedef struct {
    uint32_t in_w;
    uint32_t in_h;
    uint32_t out_w;
    uint32_t out_h;
    uint32_t max_taps;
    uint32_t pass_dir;
    uint32_t is_identity;
    float soft_knee_tau;
    uint32_t idx_offset;
    uint32_t weight_offset;
} VizPushResample;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t mode;
} VizPushPack;

typedef struct {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void *mapped;
    VkDeviceSize size;
} VizVkBuffer;

typedef struct {
    VkInstance instance;
    VkPhysicalDevice phys_dev;
    VkDevice device;
    VkQueue compute_queue;
    uint32_t queue_family_idx;
    char device_name[256];

    VkDescriptorSetLayout dsl_two;
    VkDescriptorSetLayout dsl_five;
    VkPipelineLayout pl_ingest;
    VkPipelineLayout pl_resample;
    VkPipelineLayout pl_pack;

    VkShaderModule sm_ingest;
    VkShaderModule sm_resample;
    VkShaderModule sm_pack;

    VkPipeline pipe_ingest;
    VkPipeline pipe_resample;
    VkPipeline pipe_pack;

    VkDescriptorPool dpool;
    VkDescriptorSet ds_ingest;
    VkDescriptorSet ds_vae_x;
    VkDescriptorSet ds_vae_y;
    VkDescriptorSet ds_vit_x;
    VkDescriptorSet ds_vit_y;
    VkDescriptorSet ds_pack_vae;
    VkDescriptorSet ds_pack_vit;
    VkDescriptorSet ds_pack_mask;

    VkCommandPool cmd_pool;
    VkCommandBuffer cmd_buf;
    VkFence fence;

    VizVkBuffer buf_raw_rgba;
    VizVkBuffer buf_src_lin;
    VizVkBuffer buf_tmp_lin;
    VizVkBuffer buf_vae_lin;
    VizVkBuffer buf_vit_lin;

    VizVkBuffer buf_sched_idx;
    VizVkBuffer buf_sched_weights;
    VizVkBuffer buf_sched_center;

    VizVkBuffer buf_vae_out;
    VizVkBuffer buf_vit_out;
    VizVkBuffer buf_mask_out;

    int cache_valid;
    uint32_t cached_src_w;
    uint32_t cached_src_h;
    uint32_t cached_vae_w;
    uint32_t cached_vae_h;
    uint32_t cached_vit_w;
    uint32_t cached_vit_h;
    VizFramingMode cached_framing;
    float cached_mtf_boost;

    VizPolyphaseSchedule1D vae_sx;
    VizPolyphaseSchedule1D vae_sy;
    VizPolyphaseSchedule1D vit_sx;
    VizPolyphaseSchedule1D vit_sy;

    uint32_t off_idx_vae_x;
    uint32_t off_idx_vae_y;
    uint32_t off_idx_vit_x;
    uint32_t off_idx_vit_y;

    uint32_t off_w_vae_x;
    uint32_t off_w_vae_y;
    uint32_t off_w_vit_x;
    uint32_t off_w_vit_y;

    int initialized;
} VizVkEngineState;

static VizVkEngineState g_vk = {0};

static int find_memory_type(
    VkPhysicalDevice phys_dev,
    uint32_t type_filter,
    VkMemoryPropertyFlags props,
    uint32_t *out_idx
) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(phys_dev, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((type_filter & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & props) == props) {
            *out_idx = i;
            return 0;
        }
    }
    return -1;
}

static int create_vk_buffer(
    VkDeviceSize size,
    VkMemoryPropertyFlags preferred_props,
    VkMemoryPropertyFlags fallback_props,
    int map_host,
    VizVkBuffer *out_buf
) {
    memset(out_buf, 0, sizeof(*out_buf));
    VkBufferCreateInfo binfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };
    if (vkCreateBuffer(g_vk.device, &binfo, NULL, &out_buf->buffer) != VK_SUCCESS) {
        return -1;
    }

    VkMemoryRequirements mem_req;
    vkGetBufferMemoryRequirements(g_vk.device, out_buf->buffer, &mem_req);

    uint32_t mem_idx = 0;
    if (find_memory_type(g_vk.phys_dev, mem_req.memoryTypeBits, preferred_props, &mem_idx) != 0) {
        if (find_memory_type(g_vk.phys_dev, mem_req.memoryTypeBits, fallback_props, &mem_idx) != 0) {
            vkDestroyBuffer(g_vk.device, out_buf->buffer, NULL);
            out_buf->buffer = VK_NULL_HANDLE;
            return -1;
        }
    }

    VkMemoryAllocateInfo ainfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_req.size,
        .memoryTypeIndex = mem_idx
    };
    if (vkAllocateMemory(g_vk.device, &ainfo, NULL, &out_buf->memory) != VK_SUCCESS) {
        vkDestroyBuffer(g_vk.device, out_buf->buffer, NULL);
        out_buf->buffer = VK_NULL_HANDLE;
        return -1;
    }
    if (vkBindBufferMemory(g_vk.device, out_buf->buffer, out_buf->memory, 0) != VK_SUCCESS) {
        vkFreeMemory(g_vk.device, out_buf->memory, NULL);
        vkDestroyBuffer(g_vk.device, out_buf->buffer, NULL);
        memset(out_buf, 0, sizeof(*out_buf));
        return -1;
    }
    if (map_host) {
        if (vkMapMemory(g_vk.device, out_buf->memory, 0, size, 0, &out_buf->mapped) != VK_SUCCESS) {
            vkFreeMemory(g_vk.device, out_buf->memory, NULL);
            vkDestroyBuffer(g_vk.device, out_buf->buffer, NULL);
            memset(out_buf, 0, sizeof(*out_buf));
            return -1;
        }
    }
    out_buf->size = size;
    return 0;
}

static void destroy_vk_buffer(VizVkBuffer *buf) {
    if (!g_vk.device || !buf) return;
    if (buf->mapped) {
        vkUnmapMemory(g_vk.device, buf->memory);
        buf->mapped = NULL;
    }
    if (buf->buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(g_vk.device, buf->buffer, NULL);
        buf->buffer = VK_NULL_HANDLE;
    }
    if (buf->memory != VK_NULL_HANDLE) {
        vkFreeMemory(g_vk.device, buf->memory, NULL);
        buf->memory = VK_NULL_HANDLE;
    }
}

static void bind_dset_two(VkDescriptorSet dset, VizVkBuffer *b0, VizVkBuffer *b1) {
    VkDescriptorBufferInfo binfos[2] = {
        {b0->buffer, 0, VK_WHOLE_SIZE},
        {b1->buffer, 0, VK_WHOLE_SIZE}
    };
    VkWriteDescriptorSet writes[2];
    memset(writes, 0, sizeof(writes));
    for (uint32_t i = 0; i < 2u; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = dset;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1u;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &binfos[i];
    }
    vkUpdateDescriptorSets(g_vk.device, 2u, writes, 0u, NULL);
}

static void bind_dset_five(
    VkDescriptorSet dset,
    VizVkBuffer *b_in,
    VizVkBuffer *b_out,
    VizVkBuffer *b_idx,
    VizVkBuffer *b_weights,
    VizVkBuffer *b_center
) {
    VkDescriptorBufferInfo binfos[5] = {
        {b_in->buffer, 0, VK_WHOLE_SIZE},
        {b_out->buffer, 0, VK_WHOLE_SIZE},
        {b_idx->buffer, 0, VK_WHOLE_SIZE},
        {b_weights->buffer, 0, VK_WHOLE_SIZE},
        {b_center->buffer, 0, VK_WHOLE_SIZE}
    };
    VkWriteDescriptorSet writes[5];
    memset(writes, 0, sizeof(writes));
    for (uint32_t i = 0; i < 5u; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = dset;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1u;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &binfos[i];
    }
    vkUpdateDescriptorSets(g_vk.device, 5u, writes, 0u, NULL);
}

static int create_shader_and_pipeline(
    const uint32_t *spv_code,
    size_t spv_bytes,
    VkDescriptorSetLayout dsl,
    uint32_t push_bytes,
    VkShaderModule *out_sm,
    VkPipelineLayout *out_pl,
    VkPipeline *out_pipe
) {
    VkShaderModuleCreateInfo sinfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spv_bytes,
        .pCode = spv_code
    };
    if (vkCreateShaderModule(g_vk.device, &sinfo, NULL, out_sm) != VK_SUCCESS) {
        return -1;
    }
    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0u,
        .size = push_bytes
    };
    VkPipelineLayoutCreateInfo pl_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1u,
        .pSetLayouts = &dsl,
        .pushConstantRangeCount = 1u,
        .pPushConstantRanges = &pcr
    };
    if (vkCreatePipelineLayout(g_vk.device, &pl_info, NULL, out_pl) != VK_SUCCESS) {
        return -1;
    }
    VkComputePipelineCreateInfo cp_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = *out_sm,
            .pName = "main"
        },
        .layout = *out_pl
    };
    if (vkCreateComputePipelines(g_vk.device, VK_NULL_HANDLE, 1u, &cp_info, NULL, out_pipe) != VK_SUCCESS) {
        return -1;
    }
    return 0;
}

int viz_vk_init(void) {
    if (g_vk.initialized) return 0;
    memset(&g_vk, 0, sizeof(g_vk));

    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "VizInjector",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "VizInjectorVulkan",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_2
    };
    VkInstanceCreateInfo icreate = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info
    };
    if (vkCreateInstance(&icreate, NULL, &g_vk.instance) != VK_SUCCESS) {
        return -1;
    }

    uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(g_vk.instance, &dev_count, NULL);
    if (dev_count == 0u) {
        viz_vk_shutdown();
        return -2;
    }

    VkPhysicalDevice *devs = (VkPhysicalDevice *)calloc(dev_count, sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(g_vk.instance, &dev_count, devs);
    g_vk.phys_dev = devs[0];
    for (uint32_t i = 0; i < dev_count; ++i) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devs[i], &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            g_vk.phys_dev = devs[i];
            break;
        }
    }
    free(devs);

    VkPhysicalDeviceProperties chosen_props;
    vkGetPhysicalDeviceProperties(g_vk.phys_dev, &chosen_props);
    snprintf(g_vk.device_name, sizeof(g_vk.device_name), "%s", chosen_props.deviceName);

    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_vk.phys_dev, &qf_count, NULL);
    VkQueueFamilyProperties *qprops = (VkQueueFamilyProperties *)calloc(qf_count, sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(g_vk.phys_dev, &qf_count, qprops);
    int found_qf = 0;
    for (uint32_t i = 0; i < qf_count; ++i) {
        if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            g_vk.queue_family_idx = i;
            found_qf = 1;
            break;
        }
    }
    free(qprops);
    if (!found_qf) {
        viz_vk_shutdown();
        return -3;
    }

    float q_prio = 1.0f;
    VkDeviceQueueCreateInfo qinfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = g_vk.queue_family_idx,
        .queueCount = 1u,
        .pQueuePriorities = &q_prio
    };
    VkDeviceCreateInfo dinfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1u,
        .pQueueCreateInfos = &qinfo
    };
    if (vkCreateDevice(g_vk.phys_dev, &dinfo, NULL, &g_vk.device) != VK_SUCCESS) {
        viz_vk_shutdown();
        return -4;
    }
    vkGetDeviceQueue(g_vk.device, g_vk.queue_family_idx, 0u, &g_vk.compute_queue);

    VkDescriptorSetLayoutBinding b_two[2];
    memset(b_two, 0, sizeof(b_two));
    for (uint32_t i = 0; i < 2u; ++i) {
        b_two[i].binding = i;
        b_two[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b_two[i].descriptorCount = 1u;
        b_two[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsl2_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2u,
        .pBindings = b_two
    };
    if (vkCreateDescriptorSetLayout(g_vk.device, &dsl2_info, NULL, &g_vk.dsl_two) != VK_SUCCESS) {
        viz_vk_shutdown();
        return -5;
    }

    VkDescriptorSetLayoutBinding b_five[5];
    memset(b_five, 0, sizeof(b_five));
    for (uint32_t i = 0; i < 5u; ++i) {
        b_five[i].binding = i;
        b_five[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b_five[i].descriptorCount = 1u;
        b_five[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsl5_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 5u,
        .pBindings = b_five
    };
    if (vkCreateDescriptorSetLayout(g_vk.device, &dsl5_info, NULL, &g_vk.dsl_five) != VK_SUCCESS) {
        viz_vk_shutdown();
        return -6;
    }

    if (create_shader_and_pipeline(
            g_spv_ingest_linearize, sizeof(g_spv_ingest_linearize),
            g_vk.dsl_two, sizeof(VizPushIngest),
            &g_vk.sm_ingest, &g_vk.pl_ingest, &g_vk.pipe_ingest) != 0 ||
        create_shader_and_pipeline(
            g_spv_kaiser_resample, sizeof(g_spv_kaiser_resample),
            g_vk.dsl_five, sizeof(VizPushResample),
            &g_vk.sm_resample, &g_vk.pl_resample, &g_vk.pipe_resample) != 0 ||
        create_shader_and_pipeline(
            g_spv_branch_pack, sizeof(g_spv_branch_pack),
            g_vk.dsl_two, sizeof(VizPushPack),
            &g_vk.sm_pack, &g_vk.pl_pack, &g_vk.pipe_pack) != 0) {
        viz_vk_shutdown();
        return -7;
    }

    VkDescriptorPoolSize pool_size = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 32u
    };
    VkDescriptorPoolCreateInfo dp_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 8u,
        .poolSizeCount = 1u,
        .pPoolSizes = &pool_size
    };
    if (vkCreateDescriptorPool(g_vk.device, &dp_info, NULL, &g_vk.dpool) != VK_SUCCESS) {
        viz_vk_shutdown();
        return -8;
    }

    VkDescriptorSetLayout layouts[8] = {
        g_vk.dsl_two,
        g_vk.dsl_five,
        g_vk.dsl_five,
        g_vk.dsl_five,
        g_vk.dsl_five,
        g_vk.dsl_two,
        g_vk.dsl_two,
        g_vk.dsl_two
    };
    VkDescriptorSet dsets[8] = {VK_NULL_HANDLE};
    VkDescriptorSetAllocateInfo ds_alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = g_vk.dpool,
        .descriptorSetCount = 8u,
        .pSetLayouts = layouts
    };
    if (vkAllocateDescriptorSets(g_vk.device, &ds_alloc, dsets) != VK_SUCCESS) {
        viz_vk_shutdown();
        return -9;
    }
    g_vk.ds_ingest = dsets[0];
    g_vk.ds_vae_x = dsets[1];
    g_vk.ds_vae_y = dsets[2];
    g_vk.ds_vit_x = dsets[3];
    g_vk.ds_vit_y = dsets[4];
    g_vk.ds_pack_vae = dsets[5];
    g_vk.ds_pack_vit = dsets[6];
    g_vk.ds_pack_mask = dsets[7];

    VkMemoryPropertyFlags wc_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkMemoryPropertyFlags cached_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    VkMemoryPropertyFlags dev_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    size_t max_vit_floats = (size_t)(VIZ_VK_MAX_DIM / 16u) * (size_t)(VIZ_VK_MAX_DIM / 16u) * 3u * 2u * 256u;
    size_t max_mask_floats = (size_t)(VIZ_VK_MAX_DIM / 16u) * (size_t)(VIZ_VK_MAX_DIM / 16u);

    if (create_vk_buffer(VIZ_VK_MAX_PIXELS * sizeof(uint32_t), wc_flags, wc_flags, 1, &g_vk.buf_raw_rgba) != 0 ||
        create_vk_buffer(VIZ_VK_MAX_PIXELS * sizeof(float) * 4u, dev_flags, wc_flags, 0, &g_vk.buf_src_lin) != 0 ||
        create_vk_buffer(VIZ_VK_MAX_PIXELS * sizeof(float) * 4u, dev_flags, wc_flags, 0, &g_vk.buf_tmp_lin) != 0 ||
        create_vk_buffer(VIZ_VK_MAX_PIXELS * sizeof(float) * 4u, dev_flags, wc_flags, 0, &g_vk.buf_vae_lin) != 0 ||
        create_vk_buffer(VIZ_VK_MAX_PIXELS * sizeof(float) * 4u, dev_flags, wc_flags, 0, &g_vk.buf_vit_lin) != 0 ||
        create_vk_buffer(VIZ_VK_SCHED_MAX_LEN * sizeof(int32_t), wc_flags, wc_flags, 1, &g_vk.buf_sched_idx) != 0 ||
        create_vk_buffer(VIZ_VK_SCHED_MAX_WEIGHTS * sizeof(float), wc_flags, wc_flags, 1, &g_vk.buf_sched_weights) != 0 ||
        create_vk_buffer(VIZ_VK_SCHED_MAX_LEN * sizeof(float), wc_flags, wc_flags, 1, &g_vk.buf_sched_center) != 0 ||
        create_vk_buffer(VIZ_VK_MAX_PIXELS * sizeof(float) * 3u, cached_flags, wc_flags, 1, &g_vk.buf_vae_out) != 0 ||
        create_vk_buffer(max_vit_floats * sizeof(float), cached_flags, wc_flags, 1, &g_vk.buf_vit_out) != 0 ||
        create_vk_buffer(max_mask_floats * sizeof(float), cached_flags, wc_flags, 1, &g_vk.buf_mask_out) != 0) {
        viz_vk_shutdown();
        return -10;
    }

    bind_dset_two(g_vk.ds_ingest, &g_vk.buf_raw_rgba, &g_vk.buf_src_lin);
    bind_dset_five(g_vk.ds_vae_x, &g_vk.buf_src_lin, &g_vk.buf_tmp_lin, &g_vk.buf_sched_idx, &g_vk.buf_sched_weights, &g_vk.buf_sched_center);
    bind_dset_five(g_vk.ds_vae_y, &g_vk.buf_tmp_lin, &g_vk.buf_vae_lin, &g_vk.buf_sched_idx, &g_vk.buf_sched_weights, &g_vk.buf_sched_center);
    bind_dset_five(g_vk.ds_vit_x, &g_vk.buf_src_lin, &g_vk.buf_tmp_lin, &g_vk.buf_sched_idx, &g_vk.buf_sched_weights, &g_vk.buf_sched_center);
    bind_dset_five(g_vk.ds_vit_y, &g_vk.buf_tmp_lin, &g_vk.buf_vit_lin, &g_vk.buf_sched_idx, &g_vk.buf_sched_weights, &g_vk.buf_sched_center);
    bind_dset_two(g_vk.ds_pack_vae, &g_vk.buf_vae_lin, &g_vk.buf_vae_out);
    bind_dset_two(g_vk.ds_pack_vit, &g_vk.buf_vit_lin, &g_vk.buf_vit_out);
    bind_dset_two(g_vk.ds_pack_mask, &g_vk.buf_vae_lin, &g_vk.buf_mask_out);

    VkCommandPoolCreateInfo cp_create = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = g_vk.queue_family_idx
    };
    if (vkCreateCommandPool(g_vk.device, &cp_create, NULL, &g_vk.cmd_pool) != VK_SUCCESS) {
        viz_vk_shutdown();
        return -11;
    }

    VkCommandBufferAllocateInfo cb_alloc = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = g_vk.cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1u
    };
    if (vkAllocateCommandBuffers(g_vk.device, &cb_alloc, &g_vk.cmd_buf) != VK_SUCCESS) {
        viz_vk_shutdown();
        return -12;
    }

    VkFenceCreateInfo finfo = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO
    };
    if (vkCreateFence(g_vk.device, &finfo, NULL, &g_vk.fence) != VK_SUCCESS) {
        viz_vk_shutdown();
        return -13;
    }

    g_vk.initialized = 1;
    return 0;
}

int viz_vk_is_available(void) {
    if (!g_vk.initialized) {
        viz_vk_init();
    }
    return g_vk.initialized ? 1 : 0;
}

const char *viz_vk_device_name(void) {
    return g_vk.initialized ? g_vk.device_name : "C11 AVX2";
}

void viz_vk_shutdown(void) {
    if (g_vk.device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(g_vk.device);
    }
    viz_free_polyphase_schedule_1d(&g_vk.vae_sx);
    viz_free_polyphase_schedule_1d(&g_vk.vae_sy);
    viz_free_polyphase_schedule_1d(&g_vk.vit_sx);
    viz_free_polyphase_schedule_1d(&g_vk.vit_sy);

    destroy_vk_buffer(&g_vk.buf_raw_rgba);
    destroy_vk_buffer(&g_vk.buf_src_lin);
    destroy_vk_buffer(&g_vk.buf_tmp_lin);
    destroy_vk_buffer(&g_vk.buf_vae_lin);
    destroy_vk_buffer(&g_vk.buf_vit_lin);
    destroy_vk_buffer(&g_vk.buf_sched_idx);
    destroy_vk_buffer(&g_vk.buf_sched_weights);
    destroy_vk_buffer(&g_vk.buf_sched_center);
    destroy_vk_buffer(&g_vk.buf_vae_out);
    destroy_vk_buffer(&g_vk.buf_vit_out);
    destroy_vk_buffer(&g_vk.buf_mask_out);

    if (g_vk.fence != VK_NULL_HANDLE) vkDestroyFence(g_vk.device, g_vk.fence, NULL);
    if (g_vk.cmd_pool != VK_NULL_HANDLE) vkDestroyCommandPool(g_vk.device, g_vk.cmd_pool, NULL);
    if (g_vk.dpool != VK_NULL_HANDLE) vkDestroyDescriptorPool(g_vk.device, g_vk.dpool, NULL);

    if (g_vk.pipe_ingest != VK_NULL_HANDLE) vkDestroyPipeline(g_vk.device, g_vk.pipe_ingest, NULL);
    if (g_vk.pipe_resample != VK_NULL_HANDLE) vkDestroyPipeline(g_vk.device, g_vk.pipe_resample, NULL);
    if (g_vk.pipe_pack != VK_NULL_HANDLE) vkDestroyPipeline(g_vk.device, g_vk.pipe_pack, NULL);

    if (g_vk.pl_ingest != VK_NULL_HANDLE) vkDestroyPipelineLayout(g_vk.device, g_vk.pl_ingest, NULL);
    if (g_vk.pl_resample != VK_NULL_HANDLE) vkDestroyPipelineLayout(g_vk.device, g_vk.pl_resample, NULL);
    if (g_vk.pl_pack != VK_NULL_HANDLE) vkDestroyPipelineLayout(g_vk.device, g_vk.pl_pack, NULL);

    if (g_vk.sm_ingest != VK_NULL_HANDLE) vkDestroyShaderModule(g_vk.device, g_vk.sm_ingest, NULL);
    if (g_vk.sm_resample != VK_NULL_HANDLE) vkDestroyShaderModule(g_vk.device, g_vk.sm_resample, NULL);
    if (g_vk.sm_pack != VK_NULL_HANDLE) vkDestroyShaderModule(g_vk.device, g_vk.sm_pack, NULL);

    if (g_vk.dsl_two != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(g_vk.device, g_vk.dsl_two, NULL);
    if (g_vk.dsl_five != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(g_vk.device, g_vk.dsl_five, NULL);

    if (g_vk.device != VK_NULL_HANDLE) vkDestroyDevice(g_vk.device, NULL);
    if (g_vk.instance != VK_NULL_HANDLE) vkDestroyInstance(g_vk.instance, NULL);
    memset(&g_vk, 0, sizeof(g_vk));
}

static int ensure_cached_schedules(
    uint32_t src_w,
    uint32_t src_h,
    uint32_t vae_w,
    uint32_t vae_h,
    uint32_t vit_w,
    uint32_t vit_h,
    VizFramingMode framing,
    float mtf_boost
) {
    uint32_t eff_vae_w = vae_w > 0 ? vae_w : 16u;
    uint32_t eff_vae_h = vae_h > 0 ? vae_h : 16u;
    uint32_t eff_vit_w = vit_w > 0 ? vit_w : 32u;
    uint32_t eff_vit_h = vit_h > 0 ? vit_h : 32u;

    if (g_vk.cache_valid &&
        g_vk.cached_src_w == src_w &&
        g_vk.cached_src_h == src_h &&
        g_vk.cached_vae_w == eff_vae_w &&
        g_vk.cached_vae_h == eff_vae_h &&
        g_vk.cached_vit_w == eff_vit_w &&
        g_vk.cached_vit_h == eff_vit_h &&
        g_vk.cached_framing == framing &&
        fabsf(g_vk.cached_mtf_boost - mtf_boost) < 1e-6f) {
        return 0;
    }

    viz_free_polyphase_schedule_1d(&g_vk.vae_sx);
    viz_free_polyphase_schedule_1d(&g_vk.vae_sy);
    viz_free_polyphase_schedule_1d(&g_vk.vit_sx);
    viz_free_polyphase_schedule_1d(&g_vk.vit_sy);

    if (viz_build_coupled_schedules_2d(src_w, src_h, eff_vae_w, eff_vae_h, framing, mtf_boost, &g_vk.vae_sx, &g_vk.vae_sy) != 0) {
        g_vk.cache_valid = 0;
        return -1;
    }
    if (viz_build_coupled_schedules_2d(src_w, src_h, eff_vit_w, eff_vit_h, framing, mtf_boost, &g_vk.vit_sx, &g_vk.vit_sy) != 0) {
        viz_free_polyphase_schedule_1d(&g_vk.vae_sx);
        viz_free_polyphase_schedule_1d(&g_vk.vae_sy);
        g_vk.cache_valid = 0;
        return -1;
    }

    g_vk.off_idx_vae_x = 0u;
    g_vk.off_idx_vae_y = g_vk.off_idx_vae_x + g_vk.vae_sx.out_dim;
    g_vk.off_idx_vit_x = g_vk.off_idx_vae_y + g_vk.vae_sy.out_dim;
    g_vk.off_idx_vit_y = g_vk.off_idx_vit_x + g_vk.vit_sx.out_dim;

    uint32_t w_vae_x = g_vk.vae_sx.out_dim * g_vk.vae_sx.max_taps;
    uint32_t w_vae_y = g_vk.vae_sy.out_dim * g_vk.vae_sy.max_taps;
    uint32_t w_vit_x = g_vk.vit_sx.out_dim * g_vk.vit_sx.max_taps;
    uint32_t w_vit_y = g_vk.vit_sy.out_dim * g_vk.vit_sy.max_taps;

    g_vk.off_w_vae_x = 0u;
    g_vk.off_w_vae_y = g_vk.off_w_vae_x + w_vae_x;
    g_vk.off_w_vit_x = g_vk.off_w_vae_y + w_vae_y;
    g_vk.off_w_vit_y = g_vk.off_w_vit_x + w_vit_x;

    int32_t *idx_dst = (int32_t *)g_vk.buf_sched_idx.mapped;
    memcpy(idx_dst + g_vk.off_idx_vae_x, g_vk.vae_sx.start_idx, (size_t)g_vk.vae_sx.out_dim * sizeof(int32_t));
    memcpy(idx_dst + g_vk.off_idx_vae_y, g_vk.vae_sy.start_idx, (size_t)g_vk.vae_sy.out_dim * sizeof(int32_t));
    memcpy(idx_dst + g_vk.off_idx_vit_x, g_vk.vit_sx.start_idx, (size_t)g_vk.vit_sx.out_dim * sizeof(int32_t));
    memcpy(idx_dst + g_vk.off_idx_vit_y, g_vk.vit_sy.start_idx, (size_t)g_vk.vit_sy.out_dim * sizeof(int32_t));

    float *c_dst = (float *)g_vk.buf_sched_center.mapped;
    memcpy(c_dst + g_vk.off_idx_vae_x, g_vk.vae_sx.src_center, (size_t)g_vk.vae_sx.out_dim * sizeof(float));
    memcpy(c_dst + g_vk.off_idx_vae_y, g_vk.vae_sy.src_center, (size_t)g_vk.vae_sy.out_dim * sizeof(float));
    memcpy(c_dst + g_vk.off_idx_vit_x, g_vk.vit_sx.src_center, (size_t)g_vk.vit_sx.out_dim * sizeof(float));
    memcpy(c_dst + g_vk.off_idx_vit_y, g_vk.vit_sy.src_center, (size_t)g_vk.vit_sy.out_dim * sizeof(float));

    float *w_dst = (float *)g_vk.buf_sched_weights.mapped;
    memcpy(w_dst + g_vk.off_w_vae_x, g_vk.vae_sx.weights, (size_t)w_vae_x * sizeof(float));
    memcpy(w_dst + g_vk.off_w_vae_y, g_vk.vae_sy.weights, (size_t)w_vae_y * sizeof(float));
    memcpy(w_dst + g_vk.off_w_vit_x, g_vk.vit_sx.weights, (size_t)w_vit_x * sizeof(float));
    memcpy(w_dst + g_vk.off_w_vit_y, g_vk.vit_sy.weights, (size_t)w_vit_y * sizeof(float));

    g_vk.cached_src_w = src_w;
    g_vk.cached_src_h = src_h;
    g_vk.cached_vae_w = eff_vae_w;
    g_vk.cached_vae_h = eff_vae_h;
    g_vk.cached_vit_w = eff_vit_w;
    g_vk.cached_vit_h = eff_vit_h;
    g_vk.cached_framing = framing;
    g_vk.cached_mtf_boost = mtf_boost;
    g_vk.cache_valid = 1;
    return 0;
}

static void insert_compute_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_HOST_READ_BIT
    };
    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1u, &mb,
        0u, NULL,
        0u, NULL
    );
}

static void feather_mask_grid_2d(
    const float *raw_mask,
    uint32_t lat_w,
    uint32_t lat_h,
    float feather_radius_cells,
    float *out_latent_mask
) {
    if (feather_radius_cells <= 0.05f) {
        memcpy(out_latent_mask, raw_mask, (size_t)lat_w * (size_t)lat_h * sizeof(float));
        return;
    }
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
}

int viz_vk_condition_image_rgba8(
    const uint8_t *src_rgba8,
    uint32_t src_w,
    uint32_t src_h,
    uint32_t vae_w,
    uint32_t vae_h,
    uint32_t vit_w,
    uint32_t vit_h,
    VizFramingMode framing,
    int deband_8bit,
    uint32_t seed,
    float mtf_boost,
    float soft_knee_tau,
    float mask_feather_radius,
    float *out_vae_planar_f32,
    float *out_vit_patches_f32,
    uint32_t out_grid_thw[3],
    uint32_t *out_llm_tokens,
    float *out_latent_mask_f32
) {
    if (!viz_vk_is_available() || !src_rgba8) return -1;
    if (src_w > VIZ_VK_MAX_DIM || src_h > VIZ_VK_MAX_DIM ||
        vae_w > VIZ_VK_MAX_DIM || vae_h > VIZ_VK_MAX_DIM ||
        vit_w > VIZ_VK_MAX_DIM || vit_h > VIZ_VK_MAX_DIM) {
        return -2;
    }

    if (ensure_cached_schedules(src_w, src_h, vae_w, vae_h, vit_w, vit_h, framing, mtf_boost) != 0) {
        return -3;
    }

    memcpy(g_vk.buf_raw_rgba.mapped, src_rgba8, (size_t)src_w * (size_t)src_h * 4u);

    vkResetCommandBuffer(g_vk.cmd_buf, 0);
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    vkBeginCommandBuffer(g_vk.cmd_buf, &begin_info);

    vkCmdBindPipeline(g_vk.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.pipe_ingest);
    vkCmdBindDescriptorSets(g_vk.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.pl_ingest, 0u, 1u, &g_vk.ds_ingest, 0u, NULL);
    VizPushIngest pc_ing = {
        .width = src_w,
        .height = src_h,
        .deband_8bit = (uint32_t)deband_8bit,
        .seed = seed
    };
    vkCmdPushConstants(g_vk.cmd_buf, g_vk.pl_ingest, VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(pc_ing), &pc_ing);
    vkCmdDispatch(g_vk.cmd_buf, (src_w + 15u) / 16u, (src_h + 15u) / 16u, 1u);
    insert_compute_barrier(g_vk.cmd_buf);

    int need_vae_lin = ((out_vae_planar_f32 != NULL || out_latent_mask_f32 != NULL) && vae_w > 0u && vae_h > 0u);
    if (need_vae_lin) {
        vkCmdBindPipeline(g_vk.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.pipe_resample);
        vkCmdBindDescriptorSets(g_vk.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.pl_resample, 0u, 1u, &g_vk.ds_vae_x, 0u, NULL);
        VizPushResample pc_vx = {
            .in_w = src_w,
            .in_h = src_h,
            .out_w = vae_w,
            .out_h = src_h,
            .max_taps = g_vk.vae_sx.max_taps,
            .pass_dir = 0u,
            .is_identity = (uint32_t)g_vk.vae_sx.is_identity,
            .soft_knee_tau = soft_knee_tau,
            .idx_offset = g_vk.off_idx_vae_x,
            .weight_offset = g_vk.off_w_vae_x
        };
        vkCmdPushConstants(g_vk.cmd_buf, g_vk.pl_resample, VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(pc_vx), &pc_vx);
        vkCmdDispatch(g_vk.cmd_buf, (vae_w + 15u) / 16u, (src_h + 15u) / 16u, 1u);
        insert_compute_barrier(g_vk.cmd_buf);

        vkCmdBindDescriptorSets(g_vk.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.pl_resample, 0u, 1u, &g_vk.ds_vae_y, 0u, NULL);
        VizPushResample pc_vy = {
            .in_w = vae_w,
            .in_h = src_h,
            .out_w = vae_w,
            .out_h = vae_h,
            .max_taps = g_vk.vae_sy.max_taps,
            .pass_dir = 1u,
            .is_identity = (uint32_t)g_vk.vae_sy.is_identity,
            .soft_knee_tau = soft_knee_tau,
            .idx_offset = g_vk.off_idx_vae_y,
            .weight_offset = g_vk.off_w_vae_y
        };
        vkCmdPushConstants(g_vk.cmd_buf, g_vk.pl_resample, VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(pc_vy), &pc_vy);
        vkCmdDispatch(g_vk.cmd_buf, (vae_w + 15u) / 16u, (vae_h + 15u) / 16u, 1u);
        insert_compute_barrier(g_vk.cmd_buf);

        vkCmdBindPipeline(g_vk.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.pipe_pack);
        if (out_vae_planar_f32) {
            vkCmdBindDescriptorSets(g_vk.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.pl_pack, 0u, 1u, &g_vk.ds_pack_vae, 0u, NULL);
            VizPushPack pc_pv = {
                .width = vae_w,
                .height = vae_h,
                .mode = 0u
            };
            vkCmdPushConstants(g_vk.cmd_buf, g_vk.pl_pack, VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(pc_pv), &pc_pv);
            vkCmdDispatch(g_vk.cmd_buf, (vae_w + 15u) / 16u, (vae_h + 15u) / 16u, 1u);
        }
        if (out_latent_mask_f32) {
            uint32_t lat_w = vae_w / 16u;
            uint32_t lat_h = vae_h / 16u;
            vkCmdBindDescriptorSets(g_vk.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.pl_pack, 0u, 1u, &g_vk.ds_pack_mask, 0u, NULL);
            VizPushPack pc_pm = {
                .width = vae_w,
                .height = vae_h,
                .mode = 2u
            };
            vkCmdPushConstants(g_vk.cmd_buf, g_vk.pl_pack, VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(pc_pm), &pc_pm);
            vkCmdDispatch(g_vk.cmd_buf, (lat_w + 15u) / 16u, (lat_h + 15u) / 16u, 1u);
        }
        insert_compute_barrier(g_vk.cmd_buf);
    }

    if (out_vit_patches_f32 && vit_w > 0u && vit_h > 0u) {
        if (need_vae_lin && vit_w == vae_w && vit_h == vae_h) {
            VkDescriptorBufferInfo binfos[2] = {
                {g_vk.buf_vae_lin.buffer, 0, VK_WHOLE_SIZE},
                {g_vk.buf_vit_out.buffer, 0, VK_WHOLE_SIZE}
            };
            VkWriteDescriptorSet writes[2];
            memset(writes, 0, sizeof(writes));
            for (uint32_t i = 0; i < 2u; ++i) {
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet = g_vk.ds_pack_vit;
                writes[i].dstBinding = i;
                writes[i].descriptorCount = 1u;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[i].pBufferInfo = &binfos[i];
            }
            vkUpdateDescriptorSets(g_vk.device, 2u, writes, 0u, NULL);
        } else {
            bind_dset_two(g_vk.ds_pack_vit, &g_vk.buf_vit_lin, &g_vk.buf_vit_out);
            vkCmdBindPipeline(g_vk.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.pipe_resample);
            vkCmdBindDescriptorSets(g_vk.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.pl_resample, 0u, 1u, &g_vk.ds_vit_x, 0u, NULL);
            VizPushResample pc_wx = {
                .in_w = src_w,
                .in_h = src_h,
                .out_w = vit_w,
                .out_h = src_h,
                .max_taps = g_vk.vit_sx.max_taps,
                .pass_dir = 0u,
                .is_identity = (uint32_t)g_vk.vit_sx.is_identity,
                .soft_knee_tau = soft_knee_tau,
                .idx_offset = g_vk.off_idx_vit_x,
                .weight_offset = g_vk.off_w_vit_x
            };
            vkCmdPushConstants(g_vk.cmd_buf, g_vk.pl_resample, VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(pc_wx), &pc_wx);
            vkCmdDispatch(g_vk.cmd_buf, (vit_w + 15u) / 16u, (src_h + 15u) / 16u, 1u);
            insert_compute_barrier(g_vk.cmd_buf);

            vkCmdBindDescriptorSets(g_vk.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.pl_resample, 0u, 1u, &g_vk.ds_vit_y, 0u, NULL);
            VizPushResample pc_wy = {
                .in_w = vit_w,
                .in_h = src_h,
                .out_w = vit_w,
                .out_h = vit_h,
                .max_taps = g_vk.vit_sy.max_taps,
                .pass_dir = 1u,
                .is_identity = (uint32_t)g_vk.vit_sy.is_identity,
                .soft_knee_tau = soft_knee_tau,
                .idx_offset = g_vk.off_idx_vit_y,
                .weight_offset = g_vk.off_w_vit_y
            };
            vkCmdPushConstants(g_vk.cmd_buf, g_vk.pl_resample, VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(pc_wy), &pc_wy);
            vkCmdDispatch(g_vk.cmd_buf, (vit_w + 15u) / 16u, (vit_h + 15u) / 16u, 1u);
            insert_compute_barrier(g_vk.cmd_buf);
        }

        vkCmdBindPipeline(g_vk.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.pipe_pack);
        vkCmdBindDescriptorSets(g_vk.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.pl_pack, 0u, 1u, &g_vk.ds_pack_vit, 0u, NULL);
        VizPushPack pc_pw = {
            .width = vit_w,
            .height = vit_h,
            .mode = 1u
        };
        vkCmdPushConstants(g_vk.cmd_buf, g_vk.pl_pack, VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(pc_pw), &pc_pw);
        vkCmdDispatch(g_vk.cmd_buf, (vit_w + 15u) / 16u, (vit_h + 15u) / 16u, 1u);
        insert_compute_barrier(g_vk.cmd_buf);
    }

    vkEndCommandBuffer(g_vk.cmd_buf);
    vkResetFences(g_vk.device, 1u, &g_vk.fence);

    VkSubmitInfo sinfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1u,
        .pCommandBuffers = &g_vk.cmd_buf
    };
    if (vkQueueSubmit(g_vk.compute_queue, 1u, &sinfo, g_vk.fence) != VK_SUCCESS) {
        return -4;
    }
    if (vkWaitForFences(g_vk.device, 1u, &g_vk.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        return -5;
    }

    if (out_vae_planar_f32 && vae_w > 0u && vae_h > 0u) {
        memcpy(out_vae_planar_f32, g_vk.buf_vae_out.mapped, (size_t)vae_w * (size_t)vae_h * 3u * sizeof(float));
    }
    if (out_vit_patches_f32 && vit_w > 0u && vit_h > 0u) {
        uint32_t gh = vit_h / 16u;
        uint32_t gw = vit_w / 16u;
        size_t vit_floats = (size_t)gh * (size_t)gw * 3u * 2u * 256u;
        memcpy(out_vit_patches_f32, g_vk.buf_vit_out.mapped, vit_floats * sizeof(float));
        if (out_grid_thw) {
            out_grid_thw[0] = 1u;
            out_grid_thw[1] = gh;
            out_grid_thw[2] = gw;
        }
        if (out_llm_tokens) {
            *out_llm_tokens = (gh * gw) / 4u;
        }
    }
    if (out_latent_mask_f32 && vae_w > 0u && vae_h > 0u) {
        uint32_t lat_w = vae_w / 16u;
        uint32_t lat_h = vae_h / 16u;
        feather_mask_grid_2d((const float *)g_vk.buf_mask_out.mapped, lat_w, lat_h, mask_feather_radius, out_latent_mask_f32);
    }

    return 0;
}