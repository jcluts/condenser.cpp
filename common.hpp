#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include "ggml_extend.hpp"

class DownSampleBlock : public GGMLBlock {
protected:
    int channels;
    int out_channels;
    bool vae_downsample;

public:
    DownSampleBlock(int channels,
                    int out_channels,
                    bool vae_downsample = false)
        : channels(channels),
          out_channels(out_channels),
          vae_downsample(vae_downsample) {
        if (vae_downsample) {
            blocks["conv"] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, out_channels, {3, 3}, {2, 2}, {0, 0}));
        } else {
            blocks["op"] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, out_channels, {3, 3}, {2, 2}, {1, 1}));
        }
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        // x: [N, channels, h, w]
        if (vae_downsample) {
            auto conv = std::dynamic_pointer_cast<Conv2d>(blocks["conv"]);

            x = ggml_ext_pad(ctx->ggml_ctx, x, 1, 1, 0, 0, ctx->circular_x_enabled, ctx->circular_y_enabled);
            x = conv->forward(ctx, x);
        } else {
            auto conv = std::dynamic_pointer_cast<Conv2d>(blocks["op"]);

            x = conv->forward(ctx, x);
        }
        return x;  // [N, out_channels, h/2, w/2]
    }
};

class UpSampleBlock : public GGMLBlock {
protected:
    int channels;
    int out_channels;

public:
    UpSampleBlock(int channels,
                  int out_channels)
        : channels(channels),
          out_channels(out_channels) {
        blocks["conv"] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, out_channels, {3, 3}, {1, 1}, {1, 1}));
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        // x: [N, channels, h, w]
        auto conv = std::dynamic_pointer_cast<Conv2d>(blocks["conv"]);

        x = ggml_upscale(ctx->ggml_ctx, x, 2, GGML_SCALE_MODE_NEAREST);  // [N, channels, h*2, w*2]
        x = conv->forward(ctx, x);                                       // [N, out_channels, h*2, w*2]
        return x;
    }
};

#endif  // __COMMON_HPP__
