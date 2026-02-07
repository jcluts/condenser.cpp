#include <cstddef>
#include <cstdint>
#include "ggml.h"

const float flux2_latent_rgb_proj[32][3] = {
    {0.000736f, -0.008385f, -0.019710f},
    {-0.001352f, -0.016392f, 0.020693f},
    {-0.006376f, 0.002428f, 0.036736f},
    {0.039384f, 0.074167f, 0.119789f},
    {0.007464f, -0.005705f, -0.004734f},
    {-0.004086f, 0.005287f, -0.000409f},
    {-0.032835f, 0.050802f, -0.028120f},
    {-0.003158f, -0.000835f, 0.000406f},
    {-0.112840f, -0.084337f, -0.023083f},
    {0.001462f, -0.006656f, 0.000549f},
    {-0.009980f, -0.007480f, 0.009702f},
    {0.032540f, 0.000214f, -0.061388f},
    {0.011023f, 0.000694f, 0.007143f},
    {-0.001468f, -0.006723f, -0.001678f},
    {-0.005921f, -0.010320f, -0.003907f},
    {-0.028434f, 0.027584f, 0.018457f},
    {0.014349f, 0.011523f, 0.000441f},
    {0.009874f, 0.003081f, 0.001507f},
    {0.002218f, 0.005712f, 0.001563f},
    {0.053010f, -0.019844f, 0.008683f},
    {-0.002507f, 0.005384f, 0.000938f},
    {-0.002177f, -0.011366f, 0.003559f},
    {-0.000261f, 0.015121f, -0.003240f},
    {-0.003944f, -0.002083f, 0.005043f},
    {-0.009138f, 0.011336f, 0.003781f},
    {0.011429f, 0.003985f, -0.003855f},
    {0.010518f, -0.005586f, 0.010131f},
    {0.007883f, 0.002912f, -0.001473f},
    {-0.003318f, -0.003160f, 0.003684f},
    {-0.034560f, -0.008740f, 0.012996f},
    {0.000166f, 0.001079f, -0.012153f},
    {0.017772f, 0.000937f, -0.011953f}};
float flux2_latent_rgb_bias[3] = {-0.028738f, -0.098463f, -0.107619f};

void preview_latent_video(uint8_t* buffer, struct ggml_tensor* latents, const float (*latent_rgb_proj)[3], const float latent_rgb_bias[3], int patch_size) {
    size_t buffer_head = 0;

    uint32_t latent_width  = static_cast<uint32_t>(latents->ne[0]);
    uint32_t latent_height = static_cast<uint32_t>(latents->ne[1]);
    uint32_t dim           = static_cast<uint32_t>(latents->ne[ggml_n_dims(latents) - 1]);
    uint32_t frames        = 1;
    if (ggml_n_dims(latents) == 4) {
        frames = static_cast<uint32_t>(latents->ne[2]);
    }

    uint32_t rgb_width  = latent_width * patch_size;
    uint32_t rgb_height = latent_height * patch_size;

    uint32_t unpatched_dim = dim / (patch_size * patch_size);

    for (uint32_t k = 0; k < frames; k++) {
        for (uint32_t rgb_x = 0; rgb_x < rgb_width; rgb_x++) {
            for (uint32_t rgb_y = 0; rgb_y < rgb_height; rgb_y++) {
                int latent_x = rgb_x / patch_size;
                int latent_y = rgb_y / patch_size;

                int channel_offset = 0;
                if (patch_size > 1) {
                    channel_offset = ((rgb_y % patch_size) * patch_size + (rgb_x % patch_size));
                }

                size_t latent_id = (latent_x * latents->nb[0] + latent_y * latents->nb[1] + k * latents->nb[2]);

                // should be incremented by 1 for each pixel
                size_t pixel_id = k * rgb_width * rgb_height + rgb_y * rgb_width + rgb_x;

                float r = 0, g = 0, b = 0;
                if (latent_rgb_proj != nullptr) {
                    for (uint32_t d = 0; d < unpatched_dim; d++) {
                        float value = *(float*)((char*)latents->data + latent_id + (d * patch_size * patch_size + channel_offset) * latents->nb[ggml_n_dims(latents) - 1]);
                        r += value * latent_rgb_proj[d][0];
                        g += value * latent_rgb_proj[d][1];
                        b += value * latent_rgb_proj[d][2];
                    }
                } else {
                    // interpret first 3 channels as RGB
                    r = *(float*)((char*)latents->data + latent_id + 0 * latents->nb[ggml_n_dims(latents) - 1]);
                    g = *(float*)((char*)latents->data + latent_id + 1 * latents->nb[ggml_n_dims(latents) - 1]);
                    b = *(float*)((char*)latents->data + latent_id + 2 * latents->nb[ggml_n_dims(latents) - 1]);
                }
                if (latent_rgb_bias != nullptr) {
                    // bias
                    r += latent_rgb_bias[0];
                    g += latent_rgb_bias[1];
                    b += latent_rgb_bias[2];
                }
                // change range
                r = r * .5f + .5f;
                g = g * .5f + .5f;
                b = b * .5f + .5f;

                // clamp rgb values to [0,1] range
                r = r >= 0 ? r <= 1 ? r : 1 : 0;
                g = g >= 0 ? g <= 1 ? g : 1 : 0;
                b = b >= 0 ? b <= 1 ? b : 1 : 0;

                buffer[pixel_id * 3 + 0] = (uint8_t)(r * 255);
                buffer[pixel_id * 3 + 1] = (uint8_t)(g * 255);
                buffer[pixel_id * 3 + 2] = (uint8_t)(b * 255);
            }
        }
    }
}
