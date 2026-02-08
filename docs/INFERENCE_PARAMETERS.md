# Condenser Inference Parameters Guide

This document explains the hardware and optimization parameters available when running inference with Condenser (Flux 2 Klein image generation). Understanding these flags will help you get the best balance of speed, memory usage, and output quality for your specific hardware.

## Architecture Overview

Before diving into individual parameters, it helps to understand the three main components of the inference pipeline:

1. **LLM Text Encoder** (Qwen3-4B) -- Converts your text prompt into conditioning tensors. Runs once per unique prompt. ~4-8 GB depending on quantization.
2. **Diffusion Model** (Flux DiT) -- The core denoising model. Runs once per sampling step (default: 4 steps for Klein). This is the most computationally expensive component. ~4-8 GB depending on quantization.
3. **VAE** (AutoEncoderKL) -- Decodes the final latent into a pixel image (and encodes reference images for img2img). Runs once per generation. ~150-300 MB.

Each component can have its weights placed on different backends (GPU or CPU), and each has its own compute buffer that is allocated during execution.

---

## Parameter Reference

### `--offload-to-cpu`

**What it does:** Stores model weights in system RAM instead of VRAM. When a component needs to run, its weights are automatically copied from RAM to VRAM on demand, then moved back to RAM when that component finishes. This applies to all three components (LLM, diffusion model, VAE).

**How it works internally:** Each `GGMLRunner` (the base class for all model components) maintains two backends: a `params_backend` (where weights live at rest) and a `runtime_backend` (where computation happens). With this flag, `params_backend` is set to CPU while `runtime_backend` stays on GPU. Before each `compute()` call, `offload_params_to_runtime_backend()` copies all tensors to VRAM. After computation, `offload_params_to_params_backend()` moves them back to RAM.

**When to use it:**
- When your GPU VRAM is too small to hold all three model components simultaneously
- Practically mandatory for GPUs with less than ~16 GB VRAM at higher quantization levels
- Works well on GPUs with 8-12 GB VRAM

**Tradeoffs:**
- **Speed:** Adds ~1-3 seconds per component transition (LLM -> Diffusion -> VAE) due to RAM-to-VRAM data transfer over PCIe/system bus
- **Memory:** Dramatically reduces peak VRAM usage -- only one component's weights need to fit in VRAM at a time, plus the compute buffer
- **Quality:** No impact on output quality

**Interaction with other flags:** This flag works independently of `--llm-on-cpu` and `--vae-on-cpu`. If you use `--offload-to-cpu` together with `--llm-on-cpu`, the LLM stays fully on CPU (no offloading needed since it never goes to GPU), while the diffusion model and VAE still offload between RAM and VRAM.

---

### `--fa` (Flash Attention)

**What it does:** Enables flash attention for **all** model components: the LLM text encoder, the diffusion model, and the VAE (if it has attention layers).

**How it works internally:** Sets `flash_attn_enabled = true` on all GGMLRunner instances. During graph construction, attention operations use `ggml_flash_attn_ext()` instead of the standard Q*K^T -> softmax -> *V sequence. Flash attention computes attention in a single fused kernel with O(N) memory instead of O(N^2), and avoids materializing the full attention matrix.

**When to use it:** Almost always. This is one of the most impactful performance flags.

**Tradeoffs:**
- **Speed:** Significant speedup, especially at higher resolutions. Can reduce total generation time by 30-50%
- **Memory:** Reduces peak VRAM during attention computation (no full N x N attention matrix materialized)
- **Quality:** No quality loss -- mathematically equivalent output (computed at F32 precision)

**Backend support:** Requires backend support. Works well on Vulkan and CUDA. The code includes a fallback: if the backend doesn't support the flash attention op, it silently falls back to standard attention.

**Alignment requirement:** Key/value sequence lengths are padded to multiples of 256 for flash attention.

---

### `--diffusion-fa` (Diffusion-Only Flash Attention)

**What it does:** Enables flash attention **only** for the diffusion model, leaving the LLM and VAE with standard attention.

**How it works internally:** Only calls `set_flash_attention_enabled(true)` on the `diffusion_model`, not on `cond_stage_model` or `first_stage_model`.

**When to use it:** If you encounter issues with flash attention on the LLM or VAE but still want the speedup for the most expensive component (the diffusion model). In practice, `--fa` is preferred since the LLM and VAE attention are relatively cheap.

**Note:** `--fa` implies `--diffusion-fa`. If `--fa` is set, the diffusion model always gets flash attention regardless of `--diffusion-fa`.

---

### `--vae-on-cpu`

**What it does:** Forces the VAE (AutoEncoderKL) to use a dedicated CPU backend for both weight storage and computation. The VAE's entire forward pass runs on CPU.

**How it works internally:** Creates a separate `ggml_backend_cpu_init()` for `vae_backend`, which is then passed to the VAE model constructor. Both the VAE's parameters and its compute graph execute entirely on CPU.

**When to use it:**
- When the VAE on GPU produces quality issues (see Known Issues below)
- When VRAM is extremely tight and you can't spare even the VAE's compute buffer on GPU
- The VAE is relatively small (~150-300 MB), so CPU execution is feasible

**Tradeoffs:**
- **Speed:** VAE decode becomes slower (CPU vs GPU). Adds ~2-5 seconds to generation time depending on resolution and CPU
- **Memory:** Frees up VRAM that would be used by the VAE's compute buffer
- **Quality:** Can actually *improve* quality in some cases -- see Known Issues

**Known Issues:** On CUDA backends, the VAE decode has been observed to produce degraded image quality (color shifts, artifacts). Running with `--vae-on-cpu` is a workaround that avoids this issue entirely. This appears to be a numerical precision issue in the CUDA VAE path that needs further investigation.

---

### `--llm-on-cpu`

**What it does:** Forces the LLM text encoder (Qwen3-4B) to use a dedicated CPU backend. The entire text encoding forward pass runs on CPU.

**How it works internally:** Creates a separate `ggml_backend_cpu_init()` for `llm_backend`. The LLM model (`LLMEmbedder`) is constructed with this CPU backend for both parameters and compute.

**When to use it:**
- When VRAM is too tight to hold the LLM weights even temporarily
- When using `--offload-to-cpu` but you want to avoid the cost of copying LLM weights to GPU and back (the LLM only runs once per unique prompt, so it may not be worth the transfer overhead)

**Tradeoffs:**
- **Speed:** Text encoding becomes slower. Adds ~1-5 seconds to the first generation with a new prompt. However, when using the engine with prompt caching, this cost is amortized -- the LLM only runs once per unique prompt
- **Memory:** Saves significant VRAM (~4-8 GB of LLM weights never touch the GPU)
- **Quality:** No impact on output quality

**Interaction with `--offload-to-cpu`:** If both flags are set, the LLM stays entirely on CPU (no offload cycle needed). The offload flag still applies to the diffusion model and VAE.

---

### `--vae-tiling`

**What it does:** Processes the VAE encode/decode in tiles instead of processing the entire image at once. Reduces peak VRAM usage during VAE operations at the cost of potential tile-boundary artifacts.

**How it works internally:** The function `sd_tiling_non_square()` splits the latent/image into overlapping tiles, processes each tile independently through the VAE, and blends the results at tile boundaries using overlap regions.

**When to use it:**
- When generating at high resolutions where the VAE compute buffer exceeds available VRAM
- Most useful for resolutions above 1024x1024

**Related parameters:**
- `--vae-tile-size <WxH>` -- Tile size in latent space (default: 32x32). Smaller tiles = less VRAM but more artifacts
- `--vae-relative-tile-size <WxH>` -- Tile size as a fraction of image dimensions. Overrides `--vae-tile-size`
- `--vae-tile-overlap <float>` -- Overlap between tiles as a fraction of tile size (default: 0.5). Higher overlap = better blending but slower

**Tradeoffs:**
- **Speed:** Slower due to processing multiple overlapping tiles. The overhead depends on tile count and overlap
- **Memory:** Significantly reduces peak VRAM during VAE operations
- **Quality:** Can introduce visible artifacts at tile boundaries, including color inconsistencies and seam lines. The overlap blending mitigates this but doesn't eliminate it entirely

**Known Issues:** VAE tiling has been observed to produce strange coloring artifacts that can make results unusable. This is a known issue under investigation. For standard 1024x1024 generation, tiling is generally not needed.

---

### `--mmap` (Memory-Mapped I/O)

**What it does:** Uses memory-mapped file I/O instead of standard file reads when loading model weights from disk.

**How it works internally:** During `ModelLoader::load_tensors()`, creates a `MmapWrapper` around the model file. Instead of reading tensor data into a buffer with `ifstream::read()`, the OS maps the file into the process's virtual address space. Data is loaded from disk on demand as pages are accessed.

**When to use it:**
- Can speed up initial model loading, especially for large model files
- Most beneficial when system RAM is large and the OS can keep the mapped pages cached
- Can reduce peak RAM usage during loading since the OS manages page-in/page-out

**Tradeoffs:**
- **Speed:** May speed up or slow down model loading depending on OS, filesystem, and available RAM. SSDs benefit less than HDDs
- **Memory:** The OS manages memory more efficiently -- pages that aren't needed can be evicted without explicit deallocation
- **Quality:** No impact on output quality

**Platform notes:** Falls back gracefully if memory mapping fails (e.g., on certain filesystems or with insufficient virtual address space).

---

### `--vae-conv-direct`

**What it does:** Uses `ggml_conv_2d_direct()` instead of `ggml_conv_2d()` for convolution operations inside the VAE model.

**How it works internally:** Sets `conv2d_direct_enabled = true` on the VAE's GGMLRunner. During graph construction, the `Conv2d` layer checks this flag and uses the direct convolution implementation.

**When to use it:**
- The direct implementation may be faster on some backends/hardware combinations
- Experimental -- try it and benchmark to see if it helps

**Tradeoffs:**
- **Speed:** May be faster or slower depending on backend, GPU, and kernel implementation
- **Memory:** May use different amounts of working memory depending on the implementation
- **Quality:** Should produce identical results, but as with any alternative code path, verify

**Note:** This flag only affects the VAE, not the LLM or diffusion model (which don't use 2D convolutions).

---

### `--threads <N>`

**What it does:** Sets the number of CPU threads used during computation. Default: auto-detected number of physical CPU cores.

**How it works internally:** Passed as `n_threads` to all model compute calls. Used by ggml for CPU-side tensor operations and by the model loader for parallel tensor loading.

**When to use it:**
- The default (physical core count) is usually optimal
- You might lower it if running alongside other CPU-intensive tasks
- You might raise it on systems with SMT/Hyperthreading where logical cores > physical cores

**Tradeoffs:**
- More threads = faster CPU operations, but diminishing returns and may cause contention
- On GPU-heavy workloads, thread count mainly affects CPU-side pre/post-processing and model loading

---

### `--type <format>`

**What it does:** Overrides the weight quantization type for all model weights. Options include `f32`, `f16`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q2_K`, `q3_K`, `q4_K`, etc.

**When to use it:**
- Most users should use pre-quantized model files instead of runtime conversion
- Useful for testing different quantization levels without re-converting models

**Tradeoffs:**
- Lower quantization = less memory, faster inference, but reduced quality
- `f16` is a good balance of quality and performance
- `q4_K` and `q5_K` offer significant size reduction with acceptable quality loss
- `q8_0` is nearly lossless while halving memory vs `f16`

---

### `--tensor-type-rules`

**What it does:** Allows fine-grained control over weight types per tensor name pattern. Format: `"^pattern=type,pattern2=type2"`.

**Example:** `"^vae\.=f16,model\.=q8_0"` -- keep VAE in f16 precision, quantize diffusion model to q8_0.

**When to use it:**
- To keep the VAE at higher precision (important for output quality) while quantizing the larger diffusion model more aggressively
- To experiment with mixed-precision configurations

---

## Parameter Interaction Matrix

| Flag | Reduces VRAM | Affects Speed | Quality Risk | Works With |
|------|:---:|:---:|:---:|---|
| `--offload-to-cpu` | Yes (major) | Slower | None | All flags |
| `--fa` | Yes (minor) | Faster | None | All flags |
| `--diffusion-fa` | Yes (minor) | Faster | None | Redundant with `--fa` |
| `--vae-on-cpu` | Yes (minor) | Slower | Can improve (CUDA) | All flags |
| `--llm-on-cpu` | Yes (major) | Slower | None | All flags |
| `--vae-tiling` | Yes | Slower | Artifacts risk | All flags |
| `--mmap` | Neutral | Varies | None | All flags |
| `--vae-conv-direct` | Neutral | Varies | None | VAE-related flags |

### Combinations to consider:

**Low VRAM (6-8 GB):**
```
--offload-to-cpu --fa --llm-on-cpu
```
Keeps weights in RAM, only loads one component at a time to GPU. LLM never touches GPU. Flash attention reduces peak VRAM during diffusion.

**Medium VRAM (8-12 GB):**
```
--offload-to-cpu --fa
```
The most common configuration. Weights shuttle between RAM and VRAM as needed. Flash attention for speed.

**High VRAM (16+ GB):**
```
--fa
```
Everything fits in VRAM. No offloading needed. Flash attention for speed.

**CUDA with VAE quality issues:**
```
--offload-to-cpu --fa --vae-on-cpu
```
Add `--vae-on-cpu` to any CUDA configuration to work around the VAE quality bug.

### Combinations to avoid:

- **`--vae-on-cpu` + `--vae-tiling`**: Tiling is a VRAM optimization; if the VAE is already on CPU, tiling adds overhead with no benefit.
- **`--llm-on-cpu` + `--offload-to-cpu`**: Not harmful, but redundant for the LLM component. The LLM weights already stay on CPU, so offloading only applies to diffusion and VAE.

---

## Typical VRAM Budgets (Flux 2 Klein, 1024x1024)

These are approximate figures and vary by quantization:

| Component | Weights (Q5_K) | Weights (F16) | Compute Buffer |
|-----------|:-:|:-:|:-:|
| LLM (Qwen3-4B) | ~3 GB | ~8 GB | ~0.5-1 GB |
| Diffusion (Flux DiT) | ~4 GB | ~8 GB | ~1-2 GB |
| VAE | ~150 MB | ~300 MB | ~0.5-1 GB |
| **Peak (all in VRAM)** | **~9 GB** | **~18 GB** | |
| **Peak (with offload)** | **~5 GB** | **~10 GB** | |

*With `--offload-to-cpu`, peak VRAM = largest single component (weights + compute buffer).*

---

## Quick Reference

| Goal | Recommended Flags |
|------|---|
| Fastest generation | `--fa` |
| Fit in 8 GB VRAM | `--offload-to-cpu --fa --llm-on-cpu` |
| Fit in 12 GB VRAM | `--offload-to-cpu --fa` |
| Fix CUDA VAE quality | Add `--vae-on-cpu` |
| Very high resolution | Add `--vae-tiling` |
| Faster model loading | `--mmap` |
