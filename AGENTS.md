# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

condenser.cpp is a fork of stable-diffusion.cpp focused exclusively on FLUX.2 Klein image generation. It provides a C API (`condenser.h`) consumed by two frontends: a CLI (`cn-cli`) and a persistent inference engine (`cn-engine`).

## Build Commands

```shell
# Clone with submodules (ggml is a submodule)
git clone --recursive https://github.com/jcluts/condenser.cpp

# Configure (from repo root)
cmake -B build -DSD_VULKAN=ON          # or -DSD_CUDA=ON, -DSD_METAL=ON
cmake --build build --config Release

# Binaries land in build/bin/
# cn-cli   - single-shot generation
# cn-engine - persistent JSON-over-stdio server
```

Key CMake options: `SD_VULKAN`, `SD_CUDA`, `SD_METAL`, `SD_HIPBLAS`, `SD_OPENCL`, `SD_SYCL`, `SD_FAST_SOFTMAX` (CUDA only, indeterministic).

No test suite exists. Validation is done by running cn-cli or cn-engine and inspecting output images.

## Code Style

- `.clang-format`: Chromium-based, 4-space indent, no column limit, aligned assignments
- `.clang-tidy`: modernize checks (nullptr, override, pass-by-value, deprecated-headers)
- C++17 required (`cxx_std_17`)

## Architecture

### Three-Stage Pipeline

All inference flows through `StableDiffusionGGML` in `condenser.cpp`:

1. **LLM** (Qwen3-4B text encoder) — `conditioner.hpp` / `llm.hpp` — converts prompt → conditioning tensors
2. **Diffusion** (Flux DiT) — `flux.hpp` — iterative denoising (default 4 steps for Klein)
3. **VAE** (AutoEncoderKL) — `vae.hpp` — decodes latents → pixel image

### GGMLRunner Pattern

All model components inherit `GGMLRunner` (`ggml_extend.hpp`). Key concept: **dual backends**.

- `params_backend` — where weights live at rest (CPU when `--offload-to-cpu`)
- `runtime_backend` — where compute happens (GPU)

Components are offloaded to GPU one at a time via `offload_params_to_runtime_backend()` then back via `offload_params_to_params_backend()`. This lets a 12GB GPU run models that total ~8GB+ by keeping only one component in VRAM at a time.

### Public C API (`condenser.h`)

- `new_sd_ctx()` / `free_sd_ctx()` — create/destroy context with model loaded
- `generate_image()` — full pipeline (encode prompt → diffuse → decode)
- `sd_compute_condition()` / `generate_image_with_condition()` — split API for caching prompt encodings
- `sd_encode_ref_image()` / `generate_image_with_condition_and_latents()` — split API for caching VAE-encoded reference images

### Engine (`tools/engine/main.cpp`)

JSON-over-stdio protocol (NDJSON). Reads commands from stdin, writes responses to stdout, logs to stderr. Keeps `sd_ctx_t*` alive between generations for fast re-use. Includes LRU caches for prompt conditioning and VAE latents.

### CLI (`tools/cli/main.cpp`)

Single-shot: parse args → load model → generate → save PNG → exit. Uses shared param structs from `tools/common/common.hpp`.

### Model Loading (`model.h` / `model.cpp`)

Supports GGUF and safetensors formats with memory-mapped I/O. Handles tensor name conversion (PyTorch → GGML) and type conversion on load.

## Key Runtime Flags

- `--offload-to-cpu` — weights on CPU, compute on GPU (essential for VRAM-limited setups)
- `--fa` — flash attention (requires backend support)
- `--vae-on-cpu` — run VAE decode on CPU (workaround for CUDA quality degradation)
- `--llm-on-cpu` — keep LLM entirely on CPU
- `--vae-tiling` — tile-based VAE decode for high-res (may have boundary artifacts)

## Known Issues

- CUDA VAE produces quality degradation vs CPU (use `--vae-on-cpu`)
- VAE tiling has color artifacts at tile boundaries
- Only FLUX.2 and FLUX.2 Klein model variants are supported
