# condenser.cpp

Focused FLUX.2 Klein inference engine. Fork of [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) stripped to a single model family, with a persistent JSON-over-stdio engine that keeps models in VRAM between generations.

stable-diffusion.cpp supports dozens of model architectures and sampling strategies. condenser.cpp trades that breadth for depth: one model family (FLUX.2 Klein), two frontends (CLI and engine), and a C API designed for embedding into desktop applications. The engine (`cn-engine`) runs as a child process, accepts NDJSON commands on stdin, and streams progress and results back on stdout — no HTTP server, no dependencies beyond the GPU driver.

## Features

- **FLUX.2 Klein** text-to-image and image-to-image (4GB and 9GB GGUF variants)
- **Persistent engine** — load once, generate many. 3-5x faster repeat generations vs cold-start CLI
- **Prompt conditioning cache** — same prompt with different seeds skips the text encoder entirely
- **Reference image latent cache** — same reference image across img2img runs skips the VAE encoder
- **Multi-backend** — Vulkan, CUDA, Metal, CPU (and experimental ROCm, SYCL, OpenCL)
- **VRAM offloading** — run on 8-12GB GPUs by keeping idle model components on system RAM
- **Flash attention** — reduced memory footprint and faster inference where supported
- **C API** — clean C interface (`condenser.h`) for embedding into any language

## Build

```shell
git clone --recursive https://github.com/jcluts/condenser.cpp
cd condenser.cpp
```

### Vulkan

```shell
cmake -B build -DSD_VULKAN=ON
cmake --build build --config Release
```

### CUDA

```shell
cmake -B build -DSD_CUDA=ON
cmake --build build --config Release
```

### Metal (macOS)

```shell
cmake -B build -DSD_METAL=ON
cmake --build build --config Release
```

### CPU only

```shell
cmake -B build
cmake --build build --config Release
```

Binaries are output to `build/bin/`. See [docs/build.md](docs/build.md) for advanced options and platform-specific notes.

## Usage

### CLI — Single-shot generation

```shell
./build/bin/cn-cli \
  --diffusion-model model.gguf \
  --vae ae.safetensors \
  --llm qwen.gguf \
  --prompt "a cat on a windowsill" \
  -W 1024 -H 1024 \
  --steps 4 \
  --seed 42 \
  --offload-to-cpu --fa \
  -o output.png
```

### Engine — Persistent inference

`cn-engine` is designed to be spawned as a child process by a parent application. It reads JSON commands from stdin and writes JSON responses to stdout. All log output goes to stderr.

```shell
# Quick test
echo '{"cmd":"ping","id":"1"}' | ./build/bin/cn-engine
# → {"id":"1","type":"ok","data":{"status":"pong"}}
```

```shell
# Interactive session
./build/bin/cn-engine
{"cmd":"load","id":"1","params":{"diffusion_model":"model.gguf","vae":"ae.safetensors","llm":"qwen.gguf","offload_to_cpu":true,"flash_attn":true}}
{"cmd":"generate","id":"2","params":{"prompt":"a sunset over mountains","width":1024,"height":1024,"seed":42,"steps":4,"output":"output.png"}}
{"cmd":"generate","id":"3","params":{"prompt":"a sunset over mountains","width":1024,"height":1024,"seed":99,"steps":4,"output":"output2.png"}}
{"cmd":"quit","id":"4"}
```

The second generate is fast — the model stays loaded and the prompt conditioning is cached from the first run.

See [tools/engine/README.md](tools/engine/README.md) for the full protocol reference, caching behavior, and integration examples (Python, Node.js).

## Performance

Measured on FLUX.2 Klein Q5_K, 1024x1024, 4 steps:

| Workflow | cn-cli | cn-engine |
|----------|--------|-----------|
| First generation (cold start) | ~12s | ~12s |
| Same model, new seed | ~12s | **3-4s** |
| Same prompt + new seed | ~12s | **3-4s** (prompt cache hit) |

## Key Runtime Flags

| Flag | Effect |
|------|--------|
| `--offload-to-cpu` | Keep model weights on system RAM, move to VRAM only during compute |
| `--fa` | Enable flash attention (Vulkan, CUDA) |
| `--vae-on-cpu` | Run VAE on CPU (workaround for CUDA quality issues) |
| `--llm-on-cpu` | Keep text encoder on CPU entirely |
| `--vae-tiling` | Tile-based VAE decode for high-resolution output |

See [docs/INFERENCE_PARAMETERS.md](docs/INFERENCE_PARAMETERS.md) for the complete parameter reference.

## Supported Models

- FLUX.2 Klein (4GB and 9GB GGUF variants)
- FLUX.2

## Credits

Based on [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) by leejet.
