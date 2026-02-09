#ifndef __GGML_EXTEND_HPP__
#define __GGML_EXTEND_HPP__

#include <assert.h>
#include <inttypes.h>
#include <stdarg.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include "model.h"

#ifdef SD_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef SD_USE_METAL
#include "ggml-metal.h"
#endif

#ifdef SD_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#ifdef SD_USE_OPENCL
#include "ggml-opencl.h"
#endif

#ifdef SD_USE_SYCL
#include "ggml-sycl.h"
#endif

#include "rng.hpp"
#include "util.h"

#define EPS 1e-05f

#ifndef __STATIC_INLINE__
#define __STATIC_INLINE__ static inline
#endif

#ifndef SD_UNUSED
#define SD_UNUSED(x) (void)(x)
#endif

// ---------------------------------------------------------------------------
// Vulkan pinned host buffer helpers
//
// When the runtime backend is Vulkan, allocating work context memory from a
// Vulkan host buffer ensures that CPU→GPU data transfers (via
// ggml_backend_tensor_set) bypass the staging buffer and use direct DMA.
// The ggml-vulkan backend detects source pointers in registered pinned
// memory ranges and emits vkCmdCopyBuffer directly from the host buffer.
//
// Usage:
//   ggml_backend_buffer_t pinned = sd_alloc_pinned_host_buffer(backend, size);
//   params.mem_buffer = pinned ? ggml_backend_buffer_get_base(pinned) : nullptr;
//   ctx = ggml_init(params);
//   ...
//   ggml_free(ctx);                          // does NOT free mem_buffer
//   if (pinned) ggml_backend_buffer_free(pinned);  // frees pinned memory
// ---------------------------------------------------------------------------

// Try to allocate a Vulkan pinned host buffer of the given size.
// Returns nullptr when not on Vulkan, or if the allocation fails (caller
// should fall back to normal malloc via ggml_init with mem_buffer=nullptr).
__STATIC_INLINE__ ggml_backend_buffer_t sd_alloc_pinned_host_buffer(
        ggml_backend_t backend, size_t size) {
#ifdef SD_USE_VULKAN
    if (backend != nullptr && !ggml_backend_is_cpu(backend)) {
        ggml_backend_buffer_type_t host_buft = ggml_backend_vk_host_buffer_type();
        if (host_buft) {
            ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(host_buft, size);
            if (buf) {
                LOG_DEBUG("pinned host buffer: %.1f MB allocated for DMA transfers",
                          size / (1024.0 * 1024.0));
                return buf;
            }
        }
    }
#endif
    SD_UNUSED(backend);
    SD_UNUSED(size);
    return nullptr;
}

__STATIC_INLINE__ int align_up_offset(int n, int multiple) {
    return (multiple - n % multiple) % multiple;
}

__STATIC_INLINE__ void ggml_log_callback_default(ggml_log_level level, const char* text, void*) {
    switch (level) {
        case GGML_LOG_LEVEL_DEBUG:
            LOG_DEBUG(text);
            break;
        case GGML_LOG_LEVEL_INFO:
            LOG_INFO(text);
            break;
        case GGML_LOG_LEVEL_WARN:
            LOG_WARN(text);
            break;
        case GGML_LOG_LEVEL_ERROR:
            LOG_ERROR(text);
            break;
        default:
            LOG_DEBUG(text);
    }
}


__STATIC_INLINE__ void ggml_ext_im_set_randn_f32(struct ggml_tensor* tensor, std::shared_ptr<RNG> rng) {
    uint32_t n                        = (uint32_t)ggml_nelements(tensor);
    std::vector<float> random_numbers = rng->randn(n);
    if (tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) && tensor->buffer == nullptr) {
        memcpy(tensor->data, random_numbers.data(), n * sizeof(float));
    } else {
        for (uint32_t i = 0; i < n; i++) {
            ggml_set_f32_1d(tensor, i, random_numbers[i]);
        }
    }
}

__STATIC_INLINE__ void ggml_ext_tensor_set_f32(struct ggml_tensor* tensor, float value, int64_t i0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) {
    GGML_ASSERT(tensor->nb[0] == sizeof(float));
    *(float*)((char*)(tensor->data) + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 * tensor->nb[0]) = value;
}

__STATIC_INLINE__ float ggml_ext_tensor_get_f32(const ggml_tensor* tensor, int64_t i0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) {
    if (tensor->buffer != nullptr) {
        float value;
        ggml_backend_tensor_get(tensor, &value, i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 * tensor->nb[0], sizeof(float));
        return value;
    }
    GGML_ASSERT(tensor->nb[0] == sizeof(float));
    return *(float*)((char*)(tensor->data) + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 * tensor->nb[0]);
}

__STATIC_INLINE__ int ggml_ext_tensor_get_i32(const ggml_tensor* tensor, int64_t i0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) {
    if (tensor->buffer != nullptr) {
        int value;
        ggml_backend_tensor_get(tensor, &value, i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 * tensor->nb[0], sizeof(int));
        return value;
    }
    GGML_ASSERT(tensor->nb[0] == sizeof(int));
    return *(int*)((char*)(tensor->data) + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 * tensor->nb[0]);
}

__STATIC_INLINE__ ggml_fp16_t ggml_ext_tensor_get_f16(const ggml_tensor* tensor, int64_t i0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) {
    GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
    return *(ggml_fp16_t*)((char*)(tensor->data) + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 * tensor->nb[0]);
}

__STATIC_INLINE__ void copy_ggml_tensor(struct ggml_tensor* dst, struct ggml_tensor* src) {
    if (dst->type == src->type) {
        dst->nb[0] = src->nb[0];
        dst->nb[1] = src->nb[1];
        dst->nb[2] = src->nb[2];
        dst->nb[3] = src->nb[3];

        memcpy(((char*)dst->data), ((char*)src->data), ggml_nbytes(dst));
        return;
    }
    struct ggml_init_params params;
    params.mem_size          = 10 * 1024 * 1024;  // for padding
    params.mem_buffer        = nullptr;
    params.no_alloc          = false;
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        LOG_ERROR("ggml_init() failed");
        return;
    }
    ggml_tensor* final = ggml_cpy(ctx, src, dst);

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, final);
    ggml_graph_compute_with_ctx(ctx, graph, 1);
    ggml_free(ctx);
}

// SPECIAL OPERATIONS WITH TENSORS

// V2-5: Direct pointer arithmetic — avoids ~3M per-element accessor calls for
// 1024×1024×3 output. Mirrors the sd_image_to_ggml_tensor optimization (I2).
__STATIC_INLINE__ uint8_t* ggml_tensor_to_sd_image(struct ggml_tensor* input, uint8_t* image_data = nullptr) {
    int64_t width    = input->ne[0];
    int64_t height   = input->ne[1];
    int64_t channels = input->ne[2];
    GGML_ASSERT(channels == 3 && input->type == GGML_TYPE_F32);
    if (image_data == nullptr) {
        image_data = (uint8_t*)malloc(width * height * channels);
    }

    // Tensor layout: planar CHW — ne[0]=W innermost, then H, then C
    // Output layout: interleaved HWC
    const float* src = (const float*)input->data;
    for (int64_t c = 0; c < channels; c++) {
        const float* src_plane = src + c * height * width;
        for (int64_t h = 0; h < height; h++) {
            const float* src_row = src_plane + h * width;
            uint8_t* dst_row = image_data + h * width * channels + c;
            for (int64_t w = 0; w < width; w++) {
                dst_row[w * channels] = (uint8_t)(src_row[w] * 255.0f);
            }
        }
    }
    return image_data;
}

__STATIC_INLINE__ void sd_image_to_ggml_tensor(sd_image_t image,
                                               ggml_tensor* tensor,
                                               bool scale = true) {
    GGML_ASSERT(image.width == tensor->ne[0]);
    GGML_ASSERT(image.height == tensor->ne[1]);
    GGML_ASSERT(image.channel == tensor->ne[2]);
    GGML_ASSERT(1 == tensor->ne[3]);
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);

    // Direct pointer arithmetic — avoids ~3M std::function dispatches + per-element
    // accessor overhead for a 1024x1024x3 image (I2 optimization)
    float* dst           = (float*)tensor->data;
    const uint8_t* src   = image.data;
    const float inv_255  = 1.0f / 255.0f;
    const int64_t W      = (int64_t)image.width;
    const int64_t H      = (int64_t)image.height;
    const int64_t C      = (int64_t)image.channel;

    // sd_image layout: row-major [H][W][C] interleaved (HWC)
    // ggml tensor layout: ne[0]=W innermost, then H, then C (planar CHW)
    for (int64_t c = 0; c < C; c++) {
        for (int64_t h = 0; h < H; h++) {
            const uint8_t* src_row = src + h * W * C + c;
            float* dst_row         = dst + c * H * W + h * W;
            if (scale) {
                for (int64_t w = 0; w < W; w++) {
                    dst_row[w] = (float)src_row[w * C] * inv_255;
                }
            } else {
                for (int64_t w = 0; w < W; w++) {
                    dst_row[w] = (float)src_row[w * C];
                }
            }
        }
    }
}

__STATIC_INLINE__ void ggml_ext_tensor_split_2d(struct ggml_tensor* input,
                                                struct ggml_tensor* output,
                                                int x,
                                                int y) {
    int64_t width    = output->ne[0];
    int64_t height   = output->ne[1];
    int64_t channels = output->ne[2];
    int64_t ne3      = output->ne[3];
    GGML_ASSERT(input->type == GGML_TYPE_F32 && output->type == GGML_TYPE_F32);
    for (int iy = 0; iy < height; iy++) {
        for (int ix = 0; ix < width; ix++) {
            for (int k = 0; k < channels; k++) {
                for (int l = 0; l < ne3; l++) {
                    float value = ggml_ext_tensor_get_f32(input, ix + x, iy + y, k, l);
                    ggml_ext_tensor_set_f32(output, value, ix, iy, k, l);
                }
            }
        }
    }
}

// unclamped -> expects x in the range [0-1]
__STATIC_INLINE__ float smootherstep_f32(const float x) {
    GGML_ASSERT(x >= 0.f && x <= 1.f);
    return x * x * x * (x * (6.0f * x - 15.0f) + 10.0f);
}

__STATIC_INLINE__ void ggml_ext_tensor_merge_2d(struct ggml_tensor* input,
                                                struct ggml_tensor* output,
                                                int x,
                                                int y,
                                                int overlap_x,
                                                int overlap_y,
                                                int x_skip = 0,
                                                int y_skip = 0) {
    int64_t width    = input->ne[0];
    int64_t height   = input->ne[1];
    int64_t channels = input->ne[2];
    int64_t ne3      = input->ne[3];

    int64_t img_width  = output->ne[0];
    int64_t img_height = output->ne[1];

    GGML_ASSERT(input->type == GGML_TYPE_F32 && output->type == GGML_TYPE_F32);
    for (int iy = y_skip; iy < height; iy++) {
        for (int ix = x_skip; ix < width; ix++) {
            for (int k = 0; k < channels; k++) {
                for (int l = 0; l < ne3; l++) {
                    float new_value = ggml_ext_tensor_get_f32(input, ix, iy, k, l);
                    if (overlap_x > 0 || overlap_y > 0) {  // blend colors in overlapped area
                        float old_value = ggml_ext_tensor_get_f32(output, x + ix, y + iy, k, l);

                        const float x_f_0 = (overlap_x > 0 && x > 0) ? (ix - x_skip) / float(overlap_x) : 1;
                        const float x_f_1 = (overlap_x > 0 && x < (img_width - width)) ? (width - ix) / float(overlap_x) : 1;
                        const float y_f_0 = (overlap_y > 0 && y > 0) ? (iy - y_skip) / float(overlap_y) : 1;
                        const float y_f_1 = (overlap_y > 0 && y < (img_height - height)) ? (height - iy) / float(overlap_y) : 1;

                        const float x_f = std::min(std::min(x_f_0, x_f_1), 1.f);
                        const float y_f = std::min(std::min(y_f_0, y_f_1), 1.f);

                        ggml_ext_tensor_set_f32(
                            output,
                            old_value + new_value * smootherstep_f32(y_f) * smootherstep_f32(x_f),
                            x + ix, y + iy, k, l);
                    } else {
                        ggml_ext_tensor_set_f32(output, new_value, x + ix, y + iy, k, l);
                    }
                }
            }
        }
    }
}

__STATIC_INLINE__ float ggml_ext_tensor_mean(struct ggml_tensor* src) {
    float mean        = 0.0f;
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        mean += data[i] / nelements * 1.0f;
    }
    return mean;
}

// a = a+b
__STATIC_INLINE__ void ggml_ext_tensor_add_inplace(struct ggml_tensor* a, struct ggml_tensor* b) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));
    int64_t nelements = ggml_nelements(a);
    float* vec_a      = (float*)a->data;
    float* vec_b      = (float*)b->data;
    for (int i = 0; i < nelements; i++) {
        vec_a[i] = vec_a[i] + vec_b[i];
    }
}

__STATIC_INLINE__ void ggml_ext_tensor_scale_inplace(struct ggml_tensor* src, float scale) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        data[i] = data[i] * scale;
    }
}

__STATIC_INLINE__ void ggml_ext_tensor_clamp_inplace(struct ggml_tensor* src, float min, float max) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        float val = data[i];
        data[i]   = val < min ? min : (val > max ? max : val);
    }
}

// convert values from [0, 1] to [-1, 1]
__STATIC_INLINE__ void process_vae_input_tensor(struct ggml_tensor* src) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        float val = data[i];
        data[i]   = val * 2.0f - 1.0f;
    }
}

// convert values from [-1, 1] to [0, 1]
__STATIC_INLINE__ void process_vae_output_tensor(struct ggml_tensor* src) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        float val = data[i];
        data[i]   = (val + 1.0f) * 0.5f;
    }
}

__STATIC_INLINE__ struct ggml_tensor* ggml_ext_cont(struct ggml_context* ctx,
                                                    struct ggml_tensor* x) {
    if (ggml_is_contiguous(x)) {
        return x;
    }
    return ggml_cont(ctx, x);
}

// torch like permute
__STATIC_INLINE__ struct ggml_tensor* ggml_ext_torch_permute(struct ggml_context* ctx,
                                                             struct ggml_tensor* x,
                                                             int axis0,
                                                             int axis1,
                                                             int axis2,
                                                             int axis3) {
    int torch_axes[4] = {axis0, axis1, axis2, axis3};

    int ggml_axes[4] = {0};
    for (int i = 0; i < 4; ++i) {
        int found = 0;
        for (int j = 0; j < 4; ++j) {
            if (torch_axes[j] == i) {
                ggml_axes[i] = j;
                found        = 1;
                break;
            }
        }
        GGML_ASSERT(found && "Invalid permute input: must be a permutation of 0-3");
    }

    return ggml_permute(ctx, x, ggml_axes[0], ggml_axes[1], ggml_axes[2], ggml_axes[3]);
}

// example: [N, 3*C, H, W] => ([N, C, H, W], [N, C, H, W], [N, C, H, W])
__STATIC_INLINE__ std::vector<struct ggml_tensor*> ggml_ext_chunk(struct ggml_context* ctx,
                                                                  struct ggml_tensor* x,
                                                                  int num,
                                                                  int64_t dim,
                                                                  bool cont = true) {
    GGML_ASSERT(dim >= 0 && dim < 4);
    GGML_ASSERT(x->ne[dim] % num == 0);

    std::vector<struct ggml_tensor*> chunks;
    int64_t chunk_size  = x->ne[dim] / num;
    int64_t stride      = chunk_size * x->nb[dim];
    int64_t chunk_ne[4] = {x->ne[0], x->ne[1], x->ne[2], x->ne[3]};
    chunk_ne[dim]       = chunk_size;
    for (int i = 0; i < num; i++) {
        auto chunk = ggml_view_4d(
            ctx, x,
            chunk_ne[0], chunk_ne[1], chunk_ne[2], chunk_ne[3],
            x->nb[1], x->nb[2], x->nb[3], stride * i);
        if (cont) {
            chunk = ggml_cont(ctx, chunk);
        }
        chunks.push_back(chunk);
    }

    return chunks;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_silu_act(ggml_context* ctx, ggml_tensor* x, bool gate_first = true) {
    // x: [ne3, ne2, ne1, ne0]
    // return: [ne3, ne2, ne1, ne0/2]

    auto x_vec = ggml_ext_chunk(ctx, x, 2, 0, false);
    ggml_tensor* gate;
    if (gate_first) {
        gate = x_vec[0];
        x    = x_vec[1];
    } else {
        x    = x_vec[0];
        gate = x_vec[1];
    }
    gate = ggml_cont(ctx, gate);
    gate = ggml_silu_inplace(ctx, gate);

    x = ggml_mul(ctx, x, gate);  // [ne3, ne2, ne1, ne0/2]

    return x;
}

typedef std::function<void(ggml_tensor*, ggml_tensor*, bool)> on_tile_process;

__STATIC_INLINE__ void sd_tiling_calc_tiles(int& num_tiles_dim,
                                            float& tile_overlap_factor_dim,
                                            int small_dim,
                                            int tile_size,
                                            const float tile_overlap_factor) {
    int tile_overlap     = static_cast<int>(tile_size * tile_overlap_factor);
    int non_tile_overlap = tile_size - tile_overlap;

    num_tiles_dim     = (small_dim - tile_overlap) / non_tile_overlap;
    int overshoot_dim = ((num_tiles_dim + 1) * non_tile_overlap + tile_overlap) % small_dim;

    if ((overshoot_dim != non_tile_overlap) && (overshoot_dim <= num_tiles_dim * (tile_size / 2 - tile_overlap))) {
        // if tiles don't fit perfectly using the desired overlap
        // and there is enough room to squeeze an extra tile without overlap becoming >0.5
        num_tiles_dim++;
    }

    tile_overlap_factor_dim = (float)(tile_size * num_tiles_dim - small_dim) / (float)(tile_size * (num_tiles_dim - 1));
    if (num_tiles_dim <= 2) {
        if (small_dim <= tile_size) {
            num_tiles_dim           = 1;
            tile_overlap_factor_dim = 0;
        } else {
            num_tiles_dim           = 2;
            tile_overlap_factor_dim = (2 * tile_size - small_dim) / (float)tile_size;
        }
    }
}

// Tiling
__STATIC_INLINE__ void sd_tiling_non_square(ggml_tensor* input,
                                            ggml_tensor* output,
                                            const int scale,
                                            const int p_tile_size_x,
                                            const int p_tile_size_y,
                                            const float tile_overlap_factor,
                                            on_tile_process on_processing) {
    output = ggml_set_f32(output, 0);

    int input_width   = (int)input->ne[0];
    int input_height  = (int)input->ne[1];
    int output_width  = (int)output->ne[0];
    int output_height = (int)output->ne[1];

    GGML_ASSERT(((input_width / output_width) == (input_height / output_height)) &&
                ((output_width / input_width) == (output_height / input_height)));
    GGML_ASSERT(((input_width / output_width) == scale) ||
                ((output_width / input_width) == scale));

    int small_width  = output_width;
    int small_height = output_height;

    bool decode = output_width > input_width;
    if (decode) {
        small_width  = input_width;
        small_height = input_height;
    }

    int num_tiles_x;
    float tile_overlap_factor_x;
    sd_tiling_calc_tiles(num_tiles_x, tile_overlap_factor_x, small_width, p_tile_size_x, tile_overlap_factor);

    int num_tiles_y;
    float tile_overlap_factor_y;
    sd_tiling_calc_tiles(num_tiles_y, tile_overlap_factor_y, small_height, p_tile_size_y, tile_overlap_factor);

    LOG_DEBUG("num tiles : %d, %d ", num_tiles_x, num_tiles_y);
    LOG_DEBUG("optimal overlap : %f, %f (targeting %f)", tile_overlap_factor_x, tile_overlap_factor_y, tile_overlap_factor);

    int tile_overlap_x     = (int32_t)(p_tile_size_x * tile_overlap_factor_x);
    int non_tile_overlap_x = p_tile_size_x - tile_overlap_x;

    int tile_overlap_y     = (int32_t)(p_tile_size_y * tile_overlap_factor_y);
    int non_tile_overlap_y = p_tile_size_y - tile_overlap_y;

    int tile_size_x = p_tile_size_x < small_width ? p_tile_size_x : small_width;
    int tile_size_y = p_tile_size_y < small_height ? p_tile_size_y : small_height;

    int input_tile_size_x  = tile_size_x;
    int input_tile_size_y  = tile_size_y;
    int output_tile_size_x = tile_size_x;
    int output_tile_size_y = tile_size_y;

    if (decode) {
        output_tile_size_x *= scale;
        output_tile_size_y *= scale;
    } else {
        input_tile_size_x *= scale;
        input_tile_size_y *= scale;
    }

    struct ggml_init_params params = {};
    params.mem_size += input_tile_size_x * input_tile_size_y * input->ne[2] * input->ne[3] * sizeof(float);      // input chunk
    params.mem_size += output_tile_size_x * output_tile_size_y * output->ne[2] * output->ne[3] * sizeof(float);  // output chunk
    params.mem_size += 3 * ggml_tensor_overhead();
    params.mem_buffer = nullptr;
    params.no_alloc   = false;

    LOG_DEBUG("tile work buffer size: %.2f MB", params.mem_size / 1024.f / 1024.f);

    // draft context
    struct ggml_context* tiles_ctx = ggml_init(params);
    if (!tiles_ctx) {
        LOG_ERROR("ggml_init() failed");
        return;
    }

    // tiling
    ggml_tensor* input_tile  = ggml_new_tensor_4d(tiles_ctx, GGML_TYPE_F32, input_tile_size_x, input_tile_size_y, input->ne[2], input->ne[3]);
    ggml_tensor* output_tile = ggml_new_tensor_4d(tiles_ctx, GGML_TYPE_F32, output_tile_size_x, output_tile_size_y, output->ne[2], output->ne[3]);
    int num_tiles            = num_tiles_x * num_tiles_y;
    LOG_DEBUG("processing %i tiles", num_tiles);
    pretty_progress(0, num_tiles, 0.0f);
    int tile_count = 1;
    bool last_y = false, last_x = false;
    float last_time = 0.0f;
    for (int y = 0; y < small_height && !last_y; y += non_tile_overlap_y) {
        int dy = 0;
        if (y + tile_size_y >= small_height) {
            int _y = y;
            y      = small_height - tile_size_y;
            dy     = _y - y;
            if (decode) {
                dy *= scale;
            }
            last_y = true;
        }
        for (int x = 0; x < small_width && !last_x; x += non_tile_overlap_x) {
            int dx = 0;
            if (x + tile_size_x >= small_width) {
                int _x = x;
                x      = small_width - tile_size_x;
                dx     = _x - x;
                if (decode) {
                    dx *= scale;
                }
                last_x = true;
            }

            int x_in  = decode ? x : scale * x;
            int y_in  = decode ? y : scale * y;
            int x_out = decode ? x * scale : x;
            int y_out = decode ? y * scale : y;

            int overlap_x_out = decode ? tile_overlap_x * scale : tile_overlap_x;
            int overlap_y_out = decode ? tile_overlap_y * scale : tile_overlap_y;

            int64_t t1 = ggml_time_ms();
            LOG_DEBUG("sd_tiling: About to split input tile at x=%d, y=%d", x_in, y_in);
            ggml_ext_tensor_split_2d(input, input_tile, x_in, y_in);
            LOG_DEBUG("sd_tiling: About to call on_processing callback");
            on_processing(input_tile, output_tile, false);
            LOG_DEBUG("sd_tiling: on_processing returned, merging output tile");
            ggml_ext_tensor_merge_2d(output_tile, output, x_out, y_out, overlap_x_out, overlap_y_out, dx, dy);
            LOG_DEBUG("sd_tiling: Tile %d/%d completed", tile_count, num_tiles);

            int64_t t2 = ggml_time_ms();
            last_time  = (t2 - t1) / 1000.0f;
            pretty_progress(tile_count, num_tiles, last_time);
            tile_count++;
        }
        last_x = false;
    }
    if (tile_count < num_tiles) {
        pretty_progress(num_tiles, num_tiles, last_time);
    }
    ggml_free(tiles_ctx);
}

__STATIC_INLINE__ void sd_tiling(ggml_tensor* input,
                                 ggml_tensor* output,
                                 const int scale,
                                 const int tile_size,
                                 const float tile_overlap_factor,
                                 on_tile_process on_processing) {
    sd_tiling_non_square(input, output, scale, tile_size, tile_size, tile_overlap_factor, on_processing);
}

__STATIC_INLINE__ struct ggml_tensor* ggml_ext_scale(struct ggml_context* ctx,
                                                     struct ggml_tensor* x,
                                                     float factor,
                                                     bool inplace = false) {
    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(ctx, x);
    }
    if (inplace) {
        x = ggml_scale_inplace(ctx, x, factor);
    } else {
        x = ggml_scale(ctx, x, factor);
    }
    return x;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_ext_linear(struct ggml_context* ctx,
                                                      struct ggml_tensor* x,
                                                      struct ggml_tensor* w,
                                                      struct ggml_tensor* b,
                                                      bool force_prec_f32 = false,
                                                      float scale         = 1.f) {
    if (scale != 1.f) {
        x = ggml_ext_scale(ctx, x, scale);
    }
    if (x->ne[2] * x->ne[3] > 1024) {
        // workaround: avoid ggml cuda error
        int64_t ne2 = x->ne[2];
        int64_t ne3 = x->ne[3];
        x           = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1] * x->ne[2] * x->ne[3]);
        x           = ggml_mul_mat(ctx, w, x);
        if (force_prec_f32) {
            ggml_mul_mat_set_prec(x, GGML_PREC_F32);
        }
        x = ggml_reshape_4d(ctx, x, x->ne[0], x->ne[1] / ne2 / ne3, ne2, ne3);
    } else {
        x = ggml_mul_mat(ctx, w, x);
        if (force_prec_f32) {
            ggml_mul_mat_set_prec(x, GGML_PREC_F32);
        }
    }
    if (scale != 1.f) {
        x = ggml_ext_scale(ctx, x, 1.f / scale);
    }
    if (b != nullptr) {
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_ext_pad_ext(struct ggml_context* ctx,
                                                       struct ggml_tensor* x,
                                                       int lp0,
                                                       int rp0,
                                                       int lp1,
                                                       int rp1,
                                                       int lp2,
                                                       int rp2,
                                                       int lp3,
                                                       int rp3,
                                                       bool circular_x = false,
                                                       bool circular_y = false) {
    if (circular_x && circular_y) {
        return ggml_pad_ext_circular(ctx, x, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
    }

    if (circular_x && (lp0 != 0 || rp0 != 0)) {
        x   = ggml_pad_ext_circular(ctx, x, lp0, rp0, 0, 0, 0, 0, 0, 0);
        lp0 = rp0 = 0;
    }
    if (circular_y && (lp1 != 0 || rp1 != 0)) {
        x   = ggml_pad_ext_circular(ctx, x, 0, 0, lp1, rp1, 0, 0, 0, 0);
        lp1 = rp1 = 0;
    }

    if (lp0 != 0 || rp0 != 0 || lp1 != 0 || rp1 != 0 || lp2 != 0 || rp2 != 0 || lp3 != 0 || rp3 != 0) {
        x = ggml_pad_ext(ctx, x, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
    }
    return x;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_ext_pad(struct ggml_context* ctx,
                                                   struct ggml_tensor* x,
                                                   int p0,
                                                   int p1,
                                                   int p2          = 0,
                                                   int p3          = 0,
                                                   bool circular_x = false,
                                                   bool circular_y = false) {
    return ggml_ext_pad_ext(ctx, x, 0, p0, 0, p1, 0, p2, 0, p3, circular_x, circular_y);
}

// w: [OC，IC, KH, KW]
// x: [N, IC, IH, IW]
// b: [OC,]
// result: [N, OC, OH, OW]
__STATIC_INLINE__ struct ggml_tensor* ggml_ext_conv_2d(struct ggml_context* ctx,
                                                       struct ggml_tensor* x,
                                                       struct ggml_tensor* w,
                                                       struct ggml_tensor* b,
                                                       int s0          = 1,
                                                       int s1          = 1,
                                                       int p0          = 0,
                                                       int p1          = 0,
                                                       int d0          = 1,
                                                       int d1          = 1,
                                                       bool direct     = false,
                                                       bool circular_x = false,
                                                       bool circular_y = false,
                                                       float scale     = 1.f) {
    if (scale != 1.f) {
        x = ggml_ext_scale(ctx, x, scale);
    }
    if (w->ne[2] != x->ne[2] && ggml_n_dims(w) == 2) {
        w = ggml_reshape_4d(ctx, w, 1, 1, w->ne[0], w->ne[1]);
    }

    if ((p0 != 0 || p1 != 0) && (circular_x || circular_y)) {
        x  = ggml_ext_pad_ext(ctx, x, p0, p0, p1, p1, 0, 0, 0, 0, circular_x, circular_y);
        p0 = 0;
        p1 = 0;
    }

    if (direct) {
        x = ggml_conv_2d_direct(ctx, w, x, s0, s1, p0, p1, d0, d1);
    } else {
        x = ggml_conv_2d(ctx, w, x, s0, s1, p0, p1, d0, d1);
    }
    if (scale != 1.f) {
        x = ggml_ext_scale(ctx, x, 1.f / scale);
    }
    if (b != nullptr) {
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_ext_full(struct ggml_context* ctx,
                                                    float value,
                                                    int64_t ne0,
                                                    int64_t ne1,
                                                    int64_t ne2,
                                                    int64_t ne3) {
    auto one = ggml_get_tensor(ctx, "ggml_runner_build_in_tensor:one");
    auto t   = ggml_ext_scale(ctx, one, value);             // [1,]
    t        = ggml_repeat_4d(ctx, t, ne0, ne1, ne2, ne3);  // [ne0, ne1, ne2, ne3]
    return t;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_ext_zeros(struct ggml_context* ctx,
                                                     int64_t ne0,
                                                     int64_t ne1,
                                                     int64_t ne2,
                                                     int64_t ne3) {
    return ggml_ext_full(ctx, 0.f, ne0, ne1, ne2, ne3);
}

// q: [N, L_q, C(n_head*d_head)] or [N*n_head, L_q, d_head]
// k: [N, L_k, n_kv_head*d_head] or [N*n_kv_head, L_k, d_head]
// v: [N, L_k, n_kv_head*d_head] or [N, L_k, n_kv_head, d_head]
// mask: [N, L_q, L_k]
// return: [N, L_q, C]
__STATIC_INLINE__ struct ggml_tensor* ggml_ext_attention_ext(struct ggml_context* ctx,
                                                             ggml_backend_t backend,
                                                             struct ggml_tensor* q,
                                                             struct ggml_tensor* k,
                                                             struct ggml_tensor* v,
                                                             int64_t n_head,
                                                             struct ggml_tensor* mask = nullptr,
                                                             bool skip_reshape        = false,
                                                             bool flash_attn          = false,
                                                             float kv_scale           = 1.0f) {  // avoid overflow
    int64_t L_q;
    int64_t L_k;
    int64_t C;
    int64_t N;
    int64_t d_head;
    int64_t n_kv_head;
    if (!skip_reshape) {
        L_q       = q->ne[1];
        L_k       = k->ne[1];
        C         = q->ne[0];
        N         = q->ne[2];
        d_head    = C / n_head;
        n_kv_head = k->ne[0] / d_head;

        q = ggml_reshape_4d(ctx, q, d_head, n_head, L_q, N);       // [N, L_q, n_head, d_head]
        q = ggml_ext_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [N, n_head, L_q, d_head]
        q = ggml_reshape_3d(ctx, q, d_head, L_q, n_head * N);      // [N * n_head, L_q, d_head]

        k = ggml_reshape_4d(ctx, k, d_head, n_kv_head, L_k, N);    // [N, L_k, n_kv_head, d_head]
        k = ggml_ext_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [N, n_kv_head, L_k, d_head]
        k = ggml_reshape_3d(ctx, k, d_head, L_k, n_kv_head * N);   // [N * n_kv_head, L_k, d_head]

        v = ggml_reshape_4d(ctx, v, d_head, n_kv_head, L_k, N);  // [N, L_k, n_kv_head, d_head]
    } else {
        L_q       = q->ne[1];
        L_k       = k->ne[1];
        d_head    = v->ne[0];
        N         = v->ne[3];
        n_kv_head = k->ne[2] / N;
        C         = d_head * n_head;
    }

    float scale = (1.0f / sqrt((float)d_head));

    int kv_pad       = 0;
    ggml_tensor* kqv = nullptr;

    auto build_kqv = [&](ggml_tensor* q_in, ggml_tensor* k_in, ggml_tensor* v_in, ggml_tensor* mask_in) -> ggml_tensor* {
        if (kv_pad != 0) {
            k_in = ggml_pad(ctx, k_in, 0, kv_pad, 0, 0);
        }
        if (kv_scale != 1.0f) {
            k_in = ggml_ext_scale(ctx, k_in, kv_scale);
        }
        k_in = ggml_cast(ctx, k_in, GGML_TYPE_F16);

        v_in = ggml_ext_cont(ctx, ggml_permute(ctx, v_in, 0, 2, 1, 3));
        v_in = ggml_reshape_3d(ctx, v_in, d_head, L_k, n_kv_head * N);
        if (kv_pad != 0) {
            v_in = ggml_pad(ctx, v_in, 0, kv_pad, 0, 0);
        }
        if (kv_scale != 1.0f) {
            v_in = ggml_ext_scale(ctx, v_in, kv_scale);
        }
        v_in = ggml_cast(ctx, v_in, GGML_TYPE_F16);

        if (mask_in != nullptr) {
            mask_in = ggml_transpose(ctx, mask_in);
        } else {
            if (kv_pad > 0) {
                mask_in         = ggml_ext_zeros(ctx, L_k, L_q, 1, 1);
                auto pad_tensor = ggml_ext_full(ctx, -INFINITY, kv_pad, L_q, 1, 1);
                mask_in         = ggml_concat(ctx, mask_in, pad_tensor, 0);
            }
        }

        if (mask_in != nullptr) {
            // the need for padding got removed in ggml 4767bda
            // ensure we can still use the old version for now
#ifdef GGML_KQ_MASK_PAD
            int mask_pad = 0;
            if (mask_in->ne[1] % GGML_KQ_MASK_PAD != 0) {
                mask_pad = GGML_PAD(L_q, GGML_KQ_MASK_PAD) - mask_in->ne[1];
            }
            if (mask_pad > 0) {
                mask_in = ggml_pad(ctx, mask_in, 0, mask_pad, 0, 0);
            }
#endif
            mask_in = ggml_cast(ctx, mask_in, GGML_TYPE_F16);
        }

        auto out = ggml_flash_attn_ext(ctx, q_in, k_in, v_in, mask_in, scale / kv_scale, 0, 0);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        if (kv_scale != 1.0f) {
            out = ggml_ext_scale(ctx, out, 1.0f / kv_scale);
        }
        return out;
    };

    if (flash_attn) {
        // LOG_DEBUG("attention_ext L_q:%d L_k:%d n_head:%d C:%d d_head:%d N:%d", L_q, L_k, n_head, C, d_head, N);
        bool can_use_flash_attn = true;
        if (can_use_flash_attn && L_k % 256 != 0) {
            kv_pad = GGML_PAD(L_k, 256) - static_cast<int>(L_k);
        }

        if (mask != nullptr) {
            // TODO: figure out if we can bend t5 to work too
            can_use_flash_attn = can_use_flash_attn && mask->ne[3] == 1;
        }

        if (can_use_flash_attn) {
            kqv = build_kqv(q, k, v, mask);
            if (!ggml_backend_supports_op(backend, kqv)) {
                kqv = nullptr;
            } else {
                kqv = ggml_view_3d(ctx, kqv, d_head, n_head, L_q, kqv->nb[1], kqv->nb[2], 0);
            }
        }
    }

    if (kqv == nullptr) {
        // if (flash_attn) {
        //     LOG_DEBUG("fallback to default attention, L_q:%d L_k:%d n_head:%d C:%d d_head:%d N:%d", L_q, L_k, n_head, C, d_head, N);
        // }
        v = ggml_ext_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));  // [N, n_kv_head, d_head, L_k]
        v = ggml_reshape_3d(ctx, v, L_k, d_head, n_kv_head * N);   // [N * n_kv_head, d_head, L_k]

        auto kq = ggml_mul_mat(ctx, k, q);  // [N * n_head, L_q, L_k]
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = ggml_scale_inplace(ctx, kq, scale);
        if (mask) {
            kq = ggml_add_inplace(ctx, kq, mask);
        }
        kq = ggml_soft_max_inplace(ctx, kq);

        kqv = ggml_mul_mat(ctx, v, kq);  // [N * n_head, L_q, d_head]

        kqv = ggml_reshape_4d(ctx, kqv, d_head, L_q, n_head, N);  // [N, n_head, L_q, d_head]
        kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);                 // [N, L_q, n_head, d_head]
    }

    kqv = ggml_ext_cont(ctx, kqv);
    kqv = ggml_reshape_3d(ctx, kqv, d_head * n_head, L_q, N);  // [N, L_q, C]

    return kqv;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_ext_layer_norm(struct ggml_context* ctx,
                                                          struct ggml_tensor* x,
                                                          struct ggml_tensor* w,
                                                          struct ggml_tensor* b,
                                                          float eps = EPS) {
    x = ggml_norm(ctx, x, eps);
    if (w != nullptr) {
        x = ggml_mul_inplace(ctx, x, w);
        if (b != nullptr) {
            x = ggml_add_inplace(ctx, x, b);
        }
    }
    return x;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_ext_group_norm(struct ggml_context* ctx,
                                                          struct ggml_tensor* x,
                                                          struct ggml_tensor* w,
                                                          struct ggml_tensor* b,
                                                          int num_groups = 32) {
    if (ggml_n_dims(x) >= 3 && w != nullptr && b != nullptr) {
        w = ggml_reshape_4d(ctx, w, 1, 1, w->ne[0], 1);
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
    }

    const float eps = 1e-6f;  // default eps parameter
    x               = ggml_group_norm(ctx, x, num_groups, eps);
    if (w != nullptr && b != nullptr) {
        x = ggml_mul_inplace(ctx, x, w);
        // b = ggml_repeat(ctx, b, x);
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

__STATIC_INLINE__ void ggml_ext_backend_tensor_get_and_sync(ggml_backend_t backend, const struct ggml_tensor* tensor, void* data, size_t offset, size_t size) {
#if defined(SD_USE_CUDA) || defined(SD_USE_SYCL)
    if (!ggml_backend_is_cpu(backend)) {
        ggml_backend_tensor_get_async(backend, tensor, data, offset, size);
        ggml_backend_synchronize(backend);
    } else {
        ggml_backend_tensor_get(tensor, data, offset, size);
    }
#else
    ggml_backend_tensor_get(tensor, data, offset, size);
#endif
}

__STATIC_INLINE__ float ggml_ext_backend_tensor_get_f32(ggml_tensor* tensor) {
    GGML_ASSERT(tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_I32 || tensor->type == GGML_TYPE_BF16);
    float value;
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, &value, 0, sizeof(value));
    } else if (tensor->type == GGML_TYPE_BF16) {
        ggml_bf16_t bf16_value;
        ggml_backend_tensor_get(tensor, &bf16_value, 0, sizeof(bf16_value));
        value = ggml_bf16_to_fp32(bf16_value);
    } else if (tensor->type == GGML_TYPE_F16) {
        ggml_fp16_t f16_value;
        ggml_backend_tensor_get(tensor, &f16_value, 0, sizeof(f16_value));
        value = ggml_fp16_to_fp32(f16_value);
    } else {  // GGML_TYPE_I32
        int int32_value;
        ggml_backend_tensor_get(tensor, &int32_value, 0, sizeof(int32_value));
        value = (float)int32_value;
    }
    return value;
}

__STATIC_INLINE__ struct ggml_tensor* vector_to_ggml_tensor(struct ggml_context* ctx,
                                                            const std::vector<float>& vec) {
    struct ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, vec.size());
    memcpy(t->data, (const void*)vec.data(), ggml_nbytes(t));
    return t;
}

__STATIC_INLINE__ struct ggml_tensor* vector_to_ggml_tensor_i32(struct ggml_context* ctx,
                                                                const std::vector<int>& vec) {
    struct ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, vec.size());
    memcpy(t->data, (const void*)vec.data(), ggml_nbytes(t));
    return t;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_ext_timestep_embedding(
    struct ggml_context* ctx,
    struct ggml_tensor* timesteps,
    int dim,
    int max_period    = 10000,
    float time_factor = 1.0f) {
    timesteps = ggml_ext_scale(ctx, timesteps, time_factor);
    return ggml_timestep_embedding(ctx, timesteps, dim, max_period);
}

__STATIC_INLINE__ size_t ggml_tensor_num(ggml_context* ctx) {
    size_t num = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        num++;
    }
    return num;
}

/* SDXL with LoRA requires more space */
#define MAX_PARAMS_TENSOR_NUM 32768
#define MAX_GRAPH_SIZE 327680


struct GGMLRunnerContext {
    ggml_backend_t backend                        = nullptr;
    ggml_context* ggml_ctx                        = nullptr;
    bool flash_attn_enabled                       = false;
    bool conv2d_direct_enabled                    = false;
    bool circular_x_enabled                       = false;
    bool circular_y_enabled                       = false;
};

struct GGMLRunner {
protected:
    typedef std::function<struct ggml_cgraph*()> get_graph_cb_t;

    ggml_backend_t params_backend  = nullptr;
    ggml_backend_t runtime_backend = nullptr;

    struct ggml_context* params_ctx             = nullptr;
    ggml_backend_buffer_t params_buffer         = nullptr;
    struct ggml_context* offload_ctx            = nullptr;
    ggml_backend_buffer_t runtime_params_buffer = nullptr;
    bool params_on_runtime_backend              = false;

    struct ggml_context* cache_ctx     = nullptr;
    ggml_backend_buffer_t cache_buffer = nullptr;

    struct ggml_context* compute_ctx    = nullptr;
    struct ggml_gallocr* compute_allocr = nullptr;

    std::vector<float> one_vec = {1.f};
    ggml_tensor* one_tensor    = nullptr;

    std::vector<int> zero_int_vec = {0};
    ggml_tensor* zero_int_tensor  = nullptr;

    std::map<struct ggml_tensor*, const void*> backend_tensor_data_map;
    std::map<std::string, struct ggml_tensor*> cache_tensor_map;  // name -> tensor
    const std::string final_result_name = "ggml_runner_final_result_tensor";

    bool flash_attn_enabled    = false;
    bool conv2d_direct_enabled = false;
    bool circular_x_enabled    = false;
    bool circular_y_enabled    = false;

    void alloc_params_ctx() {
        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(MAX_PARAMS_TENSOR_NUM * ggml_tensor_overhead());
        params.mem_buffer = nullptr;
        params.no_alloc   = true;

        params_ctx = ggml_init(params);
        GGML_ASSERT(params_ctx != nullptr);
        if (params_backend != runtime_backend) {
            offload_ctx = ggml_init(params);
            GGML_ASSERT(offload_ctx != nullptr);
        }
    }

    void free_params_ctx() {
        if (params_ctx != nullptr) {
            ggml_free(params_ctx);
            params_ctx = nullptr;
        }
        if (offload_ctx != nullptr) {
            ggml_free(offload_ctx);
            offload_ctx = nullptr;
        }
    }

    void alloc_cache_ctx() {
        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(MAX_PARAMS_TENSOR_NUM * ggml_tensor_overhead());
        params.mem_buffer = nullptr;
        params.no_alloc   = true;

        cache_ctx = ggml_init(params);
        GGML_ASSERT(cache_ctx != nullptr);
    }

    void free_cache_ctx() {
        if (cache_ctx != nullptr) {
            ggml_free(cache_ctx);
            cache_ctx = nullptr;
        }
    }

    void alloc_compute_ctx() {
        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(ggml_tensor_overhead() * MAX_GRAPH_SIZE + ggml_graph_overhead());
        params.mem_buffer = nullptr;
        params.no_alloc   = true;

        compute_ctx = ggml_init(params);
        GGML_ASSERT(compute_ctx != nullptr);
    }

    void free_compute_ctx() {
        if (compute_ctx != nullptr) {
            ggml_free(compute_ctx);
            compute_ctx = nullptr;
        }
    }

    void prepare_build_in_tensor_before() {
        one_tensor = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_F32, 1);
        ggml_set_name(one_tensor, "ggml_runner_build_in_tensor:one");
        set_backend_tensor_data(one_tensor, one_vec.data());

        zero_int_tensor = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, 1);
        ggml_set_name(zero_int_tensor, "ggml_runner_build_in_tensor:zero_int");
        set_backend_tensor_data(zero_int_tensor, zero_int_vec.data());
    }

    void prepare_build_in_tensor_after(struct ggml_cgraph* gf) {
        ggml_build_forward_expand(gf, one_tensor);
        ggml_build_forward_expand(gf, zero_int_tensor);
    }

    struct ggml_cgraph* new_graph_custom(size_t graph_size) {
        return ggml_new_graph_custom(compute_ctx, graph_size, false);
    }

    struct ggml_cgraph* get_compute_graph(get_graph_cb_t get_graph) {
        prepare_build_in_tensor_before();
        struct ggml_cgraph* gf = get_graph();
        if (ggml_graph_n_nodes(gf) > 0) {
            auto result = ggml_graph_node(gf, -1);
            ggml_set_name(result, final_result_name.c_str());
        }
        prepare_build_in_tensor_after(gf);
        return gf;
    }

    bool alloc_compute_buffer(get_graph_cb_t get_graph) {
        if (compute_allocr != nullptr) {
            return true;
        }
        reset_compute_ctx();
        struct ggml_cgraph* gf = get_compute_graph(get_graph);
        backend_tensor_data_map.clear();
        compute_allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_backend));

        if (!ggml_gallocr_reserve(compute_allocr, gf)) {
            // failed to allocate the compute buffer
            LOG_ERROR("%s: failed to allocate the compute buffer\n", get_desc().c_str());
            free_compute_buffer();
            return false;
        }

        // compute the required memory
        size_t compute_buffer_size = ggml_gallocr_get_buffer_size(compute_allocr, 0);
        LOG_DEBUG("%s compute buffer size: %.2f MB(%s)",
                  get_desc().c_str(),
                  compute_buffer_size / 1024.0 / 1024.0,
                  ggml_backend_is_cpu(runtime_backend) ? "RAM" : "VRAM");
        return true;
    }

    void free_cache_buffer() {
        if (cache_buffer != nullptr) {
            ggml_backend_buffer_free(cache_buffer);
            cache_buffer = nullptr;
        }
    }

    void copy_cache_tensors_to_cache_buffer() {
        if (cache_tensor_map.size() == 0) {
            return;
        }
        free_cache_ctx_and_buffer();
        alloc_cache_ctx();
        GGML_ASSERT(cache_buffer == nullptr);
        std::map<ggml_tensor*, ggml_tensor*> runtime_tensor_to_cache_tensor;
        for (auto kv : cache_tensor_map) {
            auto cache_tensor = ggml_dup_tensor(cache_ctx, kv.second);
            ggml_set_name(cache_tensor, kv.first.c_str());
            runtime_tensor_to_cache_tensor[kv.second] = cache_tensor;
        }
        size_t num_tensors = ggml_tensor_num(cache_ctx);
        cache_buffer       = ggml_backend_alloc_ctx_tensors(cache_ctx, runtime_backend);
        GGML_ASSERT(cache_buffer != nullptr);
        for (auto kv : runtime_tensor_to_cache_tensor) {
            ggml_backend_tensor_copy(kv.first, kv.second);
        }
        ggml_backend_synchronize(runtime_backend);
        cache_tensor_map.clear();
        size_t cache_buffer_size = ggml_backend_buffer_get_size(cache_buffer);
        LOG_DEBUG("%s cache backend buffer size = % 6.2f MB(%s) (%i tensors)",
                  get_desc().c_str(),
                  cache_buffer_size / (1024.f * 1024.f),
                  ggml_backend_is_cpu(runtime_backend) ? "RAM" : "VRAM",
                  num_tensors);
    }

    void copy_data_to_backend_tensor() {
        for (auto& kv : backend_tensor_data_map) {
            auto tensor = kv.first;
            auto data   = kv.second;

            ggml_backend_tensor_set(tensor, data, 0, ggml_nbytes(tensor));
        }

        backend_tensor_data_map.clear();
    }

    bool offload_params_to_runtime_backend() {
        if (params_backend == runtime_backend) {
            return true;
        }
        if (params_on_runtime_backend) {
            return true;
        }
        GGML_ASSERT(runtime_params_buffer == nullptr);
        int64_t t0         = ggml_time_ms();
        size_t num_tensors = ggml_tensor_num(offload_ctx);
        if (num_tensors == 0) {
            for (ggml_tensor* t = ggml_get_first_tensor(params_ctx); t != nullptr; t = ggml_get_next_tensor(params_ctx, t)) {
                GGML_ASSERT(t->view_src == nullptr);
                ggml_dup_tensor(offload_ctx, t);
            }
        }
        num_tensors = ggml_tensor_num(offload_ctx);
        GGML_ASSERT(num_tensors == ggml_tensor_num(params_ctx));

        runtime_params_buffer = ggml_backend_alloc_ctx_tensors(offload_ctx, runtime_backend);

        if (runtime_params_buffer == nullptr) {
            LOG_ERROR("%s alloc runtime params backend buffer failed, num_tensors = %i",
                      get_desc().c_str(),
                      num_tensors);
            return false;
        }

        ggml_tensor* t         = ggml_get_first_tensor(params_ctx);
        ggml_tensor* offload_t = ggml_get_first_tensor(offload_ctx);

        while (t != nullptr && offload_t != nullptr) {
            ggml_backend_tensor_copy(t, offload_t);
            std::swap(t->buffer, offload_t->buffer);
            std::swap(t->data, offload_t->data);
            std::swap(t->extra, offload_t->extra);

            t         = ggml_get_next_tensor(params_ctx, t);
            offload_t = ggml_get_next_tensor(offload_ctx, offload_t);
        }

        int64_t t1 = ggml_time_ms();

        size_t params_buffer_size = ggml_backend_buffer_get_size(runtime_params_buffer);
        LOG_INFO("%s offload params (%6.2f MB, %i tensors) to runtime backend (%s), taking %.2fs",
                 get_desc().c_str(),
                 params_buffer_size / (1024.f * 1024.f),
                 num_tensors,
                 ggml_backend_name(runtime_backend),
                 (t1 - t0) * 1.0f / 1000);

        params_on_runtime_backend = true;

        return true;
    }

    void offload_params_to_params_backend() {
        if (!params_on_runtime_backend) {
            return;
        }
        ggml_tensor* t         = ggml_get_first_tensor(params_ctx);
        ggml_tensor* offload_t = ggml_get_first_tensor(offload_ctx);

        while (t != nullptr && offload_t != nullptr) {
            t->buffer         = offload_t->buffer;
            t->data           = offload_t->data;
            t->extra          = offload_t->extra;
            offload_t->buffer = nullptr;
            offload_t->data   = nullptr;
            offload_t->extra  = nullptr;

            t         = ggml_get_next_tensor(params_ctx, t);
            offload_t = ggml_get_next_tensor(offload_ctx, offload_t);
        }

        if (runtime_params_buffer != nullptr) {
            ggml_backend_buffer_free(runtime_params_buffer);
            runtime_params_buffer = nullptr;
        }
        params_on_runtime_backend = false;
    }

public:
    virtual std::string get_desc() = 0;

    GGMLRunner(ggml_backend_t backend, bool offload_params_to_cpu = false)
        : runtime_backend(backend) {
        alloc_params_ctx();
        if (!ggml_backend_is_cpu(runtime_backend) && offload_params_to_cpu) {
            params_backend = ggml_backend_cpu_init();
        } else {
            params_backend = runtime_backend;
        }
    }

    virtual ~GGMLRunner() {
        free_params_buffer();
        free_compute_buffer();
        free_params_ctx();
        free_compute_ctx();
        if (params_backend != runtime_backend) {
            ggml_backend_free(params_backend);
        }
        free_cache_ctx_and_buffer();
    }

    virtual GGMLRunnerContext get_context() {
        GGMLRunnerContext runner_ctx;
        runner_ctx.ggml_ctx              = compute_ctx;
        runner_ctx.backend               = runtime_backend;
        runner_ctx.flash_attn_enabled    = flash_attn_enabled;
        runner_ctx.conv2d_direct_enabled = conv2d_direct_enabled;
        runner_ctx.circular_x_enabled    = circular_x_enabled;
        runner_ctx.circular_y_enabled    = circular_y_enabled;
        return runner_ctx;
    }

    void reset_compute_ctx() {
        free_compute_ctx();
        alloc_compute_ctx();
    }

    bool alloc_params_buffer() {
        size_t num_tensors = ggml_tensor_num(params_ctx);
        params_buffer      = ggml_backend_alloc_ctx_tensors(params_ctx, params_backend);
        if (params_buffer == nullptr) {
            LOG_ERROR("%s alloc params backend buffer failed, num_tensors = %i",
                      get_desc().c_str(),
                      num_tensors);
            return false;
        }
        size_t params_buffer_size = ggml_backend_buffer_get_size(params_buffer);
        LOG_DEBUG("%s params backend buffer size = % 6.2f MB(%s) (%i tensors)",
                  get_desc().c_str(),
                  params_buffer_size / (1024.f * 1024.f),
                  ggml_backend_is_cpu(params_backend) ? "RAM" : "VRAM",
                  num_tensors);
        return true;
    }

    void free_params_buffer() {
        if (params_buffer != nullptr) {
            ggml_backend_buffer_free(params_buffer);
            params_buffer = nullptr;
        }
    }

    size_t get_params_buffer_size() {
        if (params_buffer != nullptr) {
            return ggml_backend_buffer_get_size(params_buffer);
        }
        return 0;
    }

    void free_cache_ctx_and_buffer() {
        free_cache_buffer();
        free_cache_ctx();
    }

    void free_compute_buffer() {
        if (compute_allocr != nullptr) {
            ggml_gallocr_free(compute_allocr);
            compute_allocr = nullptr;
        }
        offload_params_to_params_backend();
    }

    // do copy after alloc graph
    void set_backend_tensor_data(struct ggml_tensor* tensor, const void* data) {
        backend_tensor_data_map[tensor] = data;
    }

    struct ggml_tensor* to_backend(struct ggml_tensor* tensor) {
        GGML_ASSERT(compute_ctx != nullptr);
        if (tensor == nullptr) {
            return nullptr;
        }
        // it's performing a compute, check if backend isn't cpu
        if (!ggml_backend_is_cpu(runtime_backend) && (tensor->buffer == nullptr || ggml_backend_buffer_is_host(tensor->buffer))) {
            // pass input tensors to gpu memory
            auto backend_tensor = ggml_dup_tensor(compute_ctx, tensor);

            set_backend_tensor_data(backend_tensor, tensor->data);
            return backend_tensor;
        } else {
            return tensor;
        }
    }

    void cache(const std::string name, struct ggml_tensor* tensor) {
        cache_tensor_map[name] = tensor;
    }

    struct ggml_tensor* get_cache_tensor_by_name(const std::string& name) {
        if (cache_ctx == nullptr) {
            return nullptr;
        }
        return ggml_get_tensor(cache_ctx, name.c_str());
    }

    bool compute(get_graph_cb_t get_graph,
                 int n_threads,
                 bool free_compute_buffer_immediately = true,
                 struct ggml_tensor** output          = nullptr,
                 struct ggml_context* output_ctx      = nullptr) {

        if (!offload_params_to_runtime_backend()) {
            LOG_ERROR("offload params to runtime backend failed");
            return false;
        }
        if (!alloc_compute_buffer(get_graph)) {
            LOG_ERROR("%s alloc compute buffer failed", get_desc().c_str());
            return false;
        }
        reset_compute_ctx();
        struct ggml_cgraph* gf = get_compute_graph(get_graph);
        if (!ggml_gallocr_alloc_graph(compute_allocr, gf)) {
            LOG_ERROR("%s alloc compute graph failed", get_desc().c_str());
            return false;
        }
        copy_data_to_backend_tensor();
        if (ggml_backend_is_cpu(runtime_backend)) {
            ggml_backend_cpu_set_n_threads(runtime_backend, n_threads);
        }
        ggml_status status = ggml_backend_graph_compute(runtime_backend, gf);
        if (status != GGML_STATUS_SUCCESS) {
            LOG_ERROR("%s compute failed: %s", get_desc().c_str(), ggml_status_to_string(status));
            return false;
        }
#ifdef GGML_PERF
        ggml_graph_print(gf);
#endif 
        copy_cache_tensors_to_cache_buffer();
        if (output != nullptr) {
            auto result = ggml_get_tensor(compute_ctx, final_result_name.c_str());
            if (*output == nullptr && output_ctx != nullptr) {
                *output = ggml_dup_tensor(output_ctx, result);
            }
            if (*output != nullptr) {
                ggml_ext_backend_tensor_get_and_sync(runtime_backend, result, (*output)->data, 0, ggml_nbytes(*output));
            }
        }
        if (free_compute_buffer_immediately) {
            free_compute_buffer();
        }
        return true;
    }

    void set_flash_attention_enabled(bool enabled) {
        flash_attn_enabled = enabled;
    }

    void set_conv2d_direct_enabled(bool enabled) {
        conv2d_direct_enabled = enabled;
    }

    void set_circular_axes(bool circular_x, bool circular_y) {
        circular_x_enabled = circular_x;
        circular_y_enabled = circular_y;
    }
};

class GGMLBlock {
protected:
    typedef std::unordered_map<std::string, struct ggml_tensor*> ParameterMap;
    typedef std::unordered_map<std::string, std::shared_ptr<GGMLBlock>> GGMLBlockMap;
    GGMLBlockMap blocks;
    ParameterMap params;

    ggml_type get_type(const std::string& name, const String2TensorStorage& tensor_storage_map, ggml_type default_type) {
        ggml_type wtype = default_type;
        auto iter       = tensor_storage_map.find(name);
        if (iter != tensor_storage_map.end()) {
            const TensorStorage& tensor_storage = iter->second;
            if (tensor_storage.expected_type != GGML_TYPE_COUNT) {
                wtype = tensor_storage.expected_type;
            } else {
                wtype = tensor_storage.type;
            }
        }
        return wtype;
    }

    void init_blocks(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") {
        for (auto& pair : blocks) {
            auto& block = pair.second;
            block->init(ctx, tensor_storage_map, prefix + pair.first);
        }
    }

    virtual void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") {}

public:
    void init(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, std::string prefix = "") {
        if (prefix.size() > 0) {
            prefix = prefix + ".";
        }
        init_params(ctx, tensor_storage_map, prefix);
        init_blocks(ctx, tensor_storage_map, prefix);
    }

    size_t get_params_num() {
        size_t num_tensors = params.size();
        for (auto& pair : blocks) {
            auto& block = pair.second;

            num_tensors += block->get_params_num();
        }
        return num_tensors;
    };

    size_t get_params_mem_size() {
        size_t mem_size = 0;
        for (auto& pair : blocks) {
            auto& block = pair.second;

            mem_size += block->get_params_mem_size();
        }

        for (auto& pair : params) {
            mem_size += ggml_nbytes(pair.second);
        }

        return mem_size;
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, std::string prefix = "") {
        if (prefix.size() > 0) {
            prefix = prefix + ".";
        }
        for (auto& pair : blocks) {
            auto& block = pair.second;
            block->get_param_tensors(tensors, prefix + pair.first);
        }

        for (auto& pair : params) {
            struct ggml_tensor* param    = pair.second;
            tensors[prefix + pair.first] = pair.second;
        }
    }

    virtual std::string get_desc() {
        return "GGMLBlock";
    }

    void get_all_blocks(std::vector<GGMLBlock*>& result) {
        result.push_back(this);
        for (auto& block_iter : blocks) {
            if (block_iter.second) {
                block_iter.second->get_all_blocks(result);
            }
        }
    }
};

class UnaryBlock : public GGMLBlock {
public:
    virtual struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) = 0;
};

class Linear : public UnaryBlock {
protected:
    int64_t in_features;
    int64_t out_features;
    bool bias;
    bool force_f32;
    bool force_prec_f32;
    float scale;
    std::string prefix;

    void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        this->prefix         = prefix;
        enum ggml_type wtype = get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_F32);
        if (in_features % ggml_blck_size(wtype) != 0 || force_f32) {
            wtype = GGML_TYPE_F32;
        }
        params["weight"] = ggml_new_tensor_2d(ctx, wtype, in_features, out_features);
        if (bias) {
            enum ggml_type wtype = GGML_TYPE_F32;
            params["bias"]       = ggml_new_tensor_1d(ctx, wtype, out_features);
        }
    }

public:
    Linear(int64_t in_features,
           int64_t out_features,
           bool bias           = true,
           bool force_f32      = false,
           bool force_prec_f32 = false,
           float scale         = 1.f)
        : in_features(in_features),
          out_features(out_features),
          bias(bias),
          force_f32(force_f32),
          force_prec_f32(force_prec_f32),
          scale(scale) {}

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        struct ggml_tensor* w = params["weight"];
        struct ggml_tensor* b = nullptr;
        if (bias) {
            b = params["bias"];
        }

        return ggml_ext_linear(ctx->ggml_ctx, x, w, b, force_prec_f32, scale);
    }
};

__STATIC_INLINE__ bool support_get_rows(ggml_type wtype) {
    std::set<ggml_type> allow_types = {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0};
    if (allow_types.find(wtype) != allow_types.end()) {
        return true;
    }
    return false;
}

class Embedding : public UnaryBlock {
protected:
    int64_t embedding_dim;
    int64_t num_embeddings;
    void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map, const std::string prefix = "") override {
        enum ggml_type wtype = get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_F32);
        if (!support_get_rows(wtype)) {
            wtype = GGML_TYPE_F32;
        }
        params["weight"] = ggml_new_tensor_2d(ctx, wtype, embedding_dim, num_embeddings);
    }

public:
    Embedding(int64_t num_embeddings, int64_t embedding_dim)
        : embedding_dim(embedding_dim),
          num_embeddings(num_embeddings) {
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                struct ggml_tensor* input_ids) {
        // input_ids: [N, n_token]
        auto weight = params["weight"];

        // There are issues with ggml batch inference, so we are expanding it here first.
        // TODO: fix ggml batch inference
        int64_t n = input_ids->ne[1];
        input_ids = ggml_reshape_1d(ctx->ggml_ctx, input_ids, input_ids->ne[0] * input_ids->ne[1]);

        input_ids      = ggml_reshape_3d(ctx->ggml_ctx, input_ids, input_ids->ne[0], 1, input_ids->ne[1]);
        auto embedding = ggml_get_rows(ctx->ggml_ctx, weight, input_ids);
        embedding      = ggml_reshape_3d(ctx->ggml_ctx, embedding, embedding->ne[0], embedding->ne[1] / n, n);

        // [N, n_token, embedding_dim]
        return embedding;
    }
};

class Conv2d : public UnaryBlock {
protected:
    int64_t in_channels;
    int64_t out_channels;
    std::pair<int, int> kernel_size;
    std::pair<int, int> stride;
    std::pair<int, int> padding;
    std::pair<int, int> dilation;
    bool bias;
    float scale = 1.f;
    std::string prefix;

    void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map, const std::string prefix = "") override {
        this->prefix         = prefix;
        enum ggml_type wtype = GGML_TYPE_F16;
        params["weight"]     = ggml_new_tensor_4d(ctx, wtype, kernel_size.second, kernel_size.first, in_channels, out_channels);
        if (bias) {
            enum ggml_type wtype = GGML_TYPE_F32;
            params["bias"]       = ggml_new_tensor_1d(ctx, wtype, out_channels);
        }
    }

public:
    Conv2d(int64_t in_channels,
           int64_t out_channels,
           std::pair<int, int> kernel_size,
           std::pair<int, int> stride   = {1, 1},
           std::pair<int, int> padding  = {0, 0},
           std::pair<int, int> dilation = {1, 1},
           bool bias                    = true)
        : in_channels(in_channels),
          out_channels(out_channels),
          kernel_size(kernel_size),
          stride(stride),
          padding(padding),
          dilation(dilation),
          bias(bias) {}

    void set_scale(float scale_value) {
        scale = scale_value;
    }

    std::string get_desc() {
        return "Conv2d";
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        struct ggml_tensor* w = params["weight"];
        struct ggml_tensor* b = nullptr;
        if (bias) {
            b = params["bias"];
        }

        return ggml_ext_conv_2d(ctx->ggml_ctx,
                                x,
                                w,
                                b,
                                stride.second,
                                stride.first,
                                padding.second,
                                padding.first,
                                dilation.second,
                                dilation.first,
                                ctx->conv2d_direct_enabled,
                                ctx->circular_x_enabled,
                                ctx->circular_y_enabled,
                                scale);
    }
};

class LayerNorm : public UnaryBlock {
protected:
    int64_t normalized_shape;
    float eps;
    bool elementwise_affine;
    bool bias;
    std::string prefix;

    void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        this->prefix = prefix;
        if (elementwise_affine) {
            enum ggml_type wtype = GGML_TYPE_F32;
            params["weight"]     = ggml_new_tensor_1d(ctx, wtype, normalized_shape);
            if (bias) {
                enum ggml_type wtype = GGML_TYPE_F32;
                params["bias"]       = ggml_new_tensor_1d(ctx, wtype, normalized_shape);
            }
        }
    }

public:
    LayerNorm(int64_t normalized_shape,
              float eps               = 1e-05f,
              bool elementwise_affine = true,
              bool bias               = true)
        : normalized_shape(normalized_shape),
          eps(eps),
          elementwise_affine(elementwise_affine),
          bias(bias) {}

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        struct ggml_tensor* w = nullptr;
        struct ggml_tensor* b = nullptr;

        if (elementwise_affine) {
            w = params["weight"];
            if (bias) {
                b = params["bias"];
            }
        }
        return ggml_ext_layer_norm(ctx->ggml_ctx, x, w, b, eps);
    }
};

class GroupNorm : public GGMLBlock {
protected:
    int num_groups;
    int64_t num_channels;
    float eps;
    bool affine;
    std::string prefix;

    void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        this->prefix = prefix;
        if (affine) {
            enum ggml_type wtype      = GGML_TYPE_F32;
            enum ggml_type bias_wtype = GGML_TYPE_F32;
            params["weight"]          = ggml_new_tensor_1d(ctx, wtype, num_channels);
            params["bias"]            = ggml_new_tensor_1d(ctx, bias_wtype, num_channels);
        }
    }

public:
    GroupNorm(int num_groups,
              int64_t num_channels,
              float eps   = 1e-05f,
              bool affine = true)
        : num_groups(num_groups),
          num_channels(num_channels),
          eps(eps),
          affine(affine) {}

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        struct ggml_tensor* w = nullptr;
        struct ggml_tensor* b = nullptr;
        if (affine) {
            w = params["weight"];
            b = params["bias"];
        }
        return ggml_ext_group_norm(ctx->ggml_ctx, x, w, b, num_groups);
    }
};

class GroupNorm32 : public GroupNorm {
public:
    GroupNorm32(int64_t num_channels)
        : GroupNorm(32, num_channels, 1e-06f) {}
};

class RMSNorm : public UnaryBlock {
protected:
    int64_t hidden_size;
    float eps;
    std::string prefix;

    void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, std::string prefix = "") override {
        this->prefix         = prefix;
        enum ggml_type wtype = GGML_TYPE_F32;
        params["weight"]     = ggml_new_tensor_1d(ctx, wtype, hidden_size);
    }

public:
    RMSNorm(int64_t hidden_size,
            float eps = 1e-06f)
        : hidden_size(hidden_size),
          eps(eps) {}

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        struct ggml_tensor* w = params["weight"];

        x = ggml_rms_norm(ctx->ggml_ctx, x, eps);
        x = ggml_mul_inplace(ctx->ggml_ctx, x, w);
        return x;
    }
};

#endif  // __GGML_EXTEND__HPP__
