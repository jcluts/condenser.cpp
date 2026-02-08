# cn-engine — Persistent Inference Engine

`cn-engine` is a persistent inference process that reads JSON commands from stdin and writes JSON responses to stdout. Unlike the CLI (`cn-cli`), which loads and unloads the model for every invocation, `cn-engine` keeps the model resident in VRAM between generations — delivering **3-5× faster** repeat generations with the same model.

## Quick Start

```bash
# Build (from repo root)
mkdir build && cd build
cmake .. -DSD_VULKAN=ON    # or -DSD_CUBLAS=ON for CUDA
cmake --build . --config Release

# Test with a ping
echo '{"cmd":"ping","id":"1"}' | ./bin/cn-engine
# → {"id":"1","type":"ok","data":{"status":"pong"}}
```

## Usage

`cn-engine` is designed to be spawned as a child process. The parent writes JSON commands to the engine's stdin (one per line) and reads JSON responses from stdout (one per line). All human-readable log output goes to stderr.

### Interactive Example

```bash
./bin/cn-engine
# Then type commands, one JSON object per line:
{"cmd":"ping","id":"1"}
{"cmd":"load","id":"2","params":{"diffusion_model":"/path/to/model.gguf","vae":"/path/to/ae.safetensors","llm":"/path/to/qwen.gguf"}}
{"cmd":"generate","id":"3","params":{"prompt":"a cat on a windowsill","width":1024,"height":1024,"seed":42,"steps":4,"output":"./output.png"}}
{"cmd":"generate","id":"4","params":{"prompt":"a cat on a windowsill","width":1024,"height":1024,"seed":99,"steps":4,"output":"./output2.png"}}
{"cmd":"quit","id":"5"}
```

The second `generate` command will be **much faster** because the model is already loaded.

### Piping Commands from a File

```bash
cat commands.jsonl | ./bin/cn-engine 2>engine.log
```

## Protocol Reference

Every message is a single JSON object terminated by `\n` (newline-delimited JSON / NDJSON).

### Commands (Client → Engine)

#### `ping` — Health Check

```json
{"cmd": "ping", "id": "1"}
```

Response:
```json
{"id": "1", "type": "ok", "data": {"status": "pong"}}
```

#### `load` — Load a Model

Creates or replaces the inference context. Frees any previously loaded model first.

```json
{
  "cmd": "load",
  "id": "req-1",
  "params": {
    "diffusion_model": "/path/to/model.gguf",
    "vae": "/path/to/ae.safetensors",
    "llm": "/path/to/qwen.gguf",
    "n_threads": 8,
    "flash_attn": true,
    "diffusion_flash_attn": false,
    "vae_conv_direct": false,
    "offload_to_cpu": false,
    "llm_on_cpu": false,
    "vae_on_cpu": false,
    "vae_decode_only": true,
    "free_params_immediately": true
  }
}
```

All `params` fields except model paths are optional with sensible defaults.

Response:
```json
{"id": "req-1", "type": "ok", "data": {"status": "model_loaded", "model_info": "model.gguf", "load_time_ms": 8500}}
```

#### `generate` — Run Inference

Generates images using the currently loaded model. Fails if no model is loaded.

```json
{
  "cmd": "generate",
  "id": "req-2",
  "params": {
    "prompt": "a cat sitting on a windowsill",
    "width": 1024,
    "height": 1024,
    "seed": 42,
    "steps": 4,
    "sampling_method": "euler",
    "guidance": 3.5,
    "batch_count": 1,
    "output": "/path/to/output.png",
    "ref_images": ["/path/to/ref1.png"],
    "increase_ref_index": false,
    "vae_tiling": {
      "enabled": true,
      "tile_size_x": 256,
      "tile_size_y": 256,
      "target_overlap": 0.5
    },
    "use_prompt_cache": true,
    "use_ref_latent_cache": true
  }
}
```

**`use_prompt_cache`** (default: `true`): When enabled, the engine caches the text encoder (LLM) output keyed by prompt string. On subsequent generations with the same prompt but a different seed, dimensions, or sampling settings, the text encoder is skipped entirely — saving ~0.5-2s per generation. The cache is automatically cleared on model load/unload. Up to 16 prompt conditions are cached with LRU eviction.

**`use_ref_latent_cache`** (default: `true`): When enabled, the engine caches the VAE-encoded latent representation of reference images, keyed by file path + modification time + file size. On subsequent img2img generations with the same reference image but different seeds or prompts, the VAE encode step is skipped — saving ~1-3s per generation. Up to 8 latents are cached with LRU eviction.

During generation, the engine emits streaming progress messages:

```json
{"id": "req-2", "type": "progress", "data": {"phase": "conditioning", "message": "Prompt cache hit — skipping text encoder", "cache_hit": true}}
{"id": "req-2", "type": "progress", "data": {"phase": "encoding", "message": "Reference image cache hit — skipping VAE encode", "cache_hit": true}}
{"id": "req-2", "type": "progress", "data": {"phase": "sampling", "step": 1, "total_steps": 4, "step_time_s": 1.2}}
{"id": "req-2", "type": "progress", "data": {"phase": "sampling", "step": 2, "total_steps": 4, "step_time_s": 0.9}}
{"id": "req-2", "type": "progress", "data": {"phase": "sampling", "step": 3, "total_steps": 4, "step_time_s": 0.9}}
{"id": "req-2", "type": "progress", "data": {"phase": "sampling", "step": 4, "total_steps": 4, "step_time_s": 0.8}}
{"id": "req-2", "type": "progress", "data": {"phase": "saving", "message": "Saving output..."}}
```

On the first generation with a new prompt, the conditioning progress will show `"cache_hit": false`:
```json
{"id": "req-2", "type": "progress", "data": {"phase": "conditioning", "message": "Running text encoder (will cache result)...", "cache_hit": false}}
```

Final result (includes `prompt_cache_hit` and `ref_latent_cache_hit` fields):
```json
{"id": "req-2", "type": "result", "data": {"success": true, "output": "/path/to/output.png", "seed": 42, "total_time_ms": 4200, "images_saved": 1, "prompt_cache_hit": false, "ref_latent_cache_hit": false}}
```

#### `unload` — Free VRAM

Releases the model and frees VRAM without shutting down the engine.

```json
{"cmd": "unload", "id": "req-3"}
```

Response:
```json
{"id": "req-3", "type": "ok", "data": {"status": "model_unloaded"}}
```

#### `status` — Query Engine State

```json
{"cmd": "status", "id": "req-4"}
```

Response (model loaded):
```json
{"id": "req-4", "type": "ok", "data": {"model_loaded": true, "model_info": "flux2-klein-Q5_K.gguf", "uptime_s": 342}}
```

Response (no model):
```json
{"id": "req-4", "type": "ok", "data": {"model_loaded": false}}
```

#### `quit` — Graceful Shutdown

```json
{"cmd": "quit", "id": "req-5"}
```

Response:
```json
{"id": "req-5", "type": "ok", "data": {"status": "quitting"}}
```

The engine also shuts down cleanly if stdin is closed (e.g., parent process exits).

### Response Types

| Type | Meaning |
|------|---------|
| `ok` | Command completed successfully |
| `progress` | Streaming update during generation |
| `result` | Generation completed with output files |
| `error` | Command failed |

### Error Response

```json
{"id": "req-2", "type": "error", "data": {"message": "No model loaded — send a 'load' command first", "code": "NO_MODEL"}}
```

Error codes: `PARSE_ERROR`, `UNKNOWN_CMD`, `NO_MODEL`, `CTX_CREATION_FAILED`, `REF_IMAGE_LOAD_FAILED`, `OUTPUT_DIR_FAILED`, `GENERATION_FAILED`, `NO_OUTPUT`.

## Design Decisions

- **NDJSON** — Standard newline-delimited JSON. Easy to parse, easy to pipe, easy to debug.
- **Sequential processing** — One command at a time. No concurrency needed for a single-parent sidecar.
- **Unbuffered stdout** — `setvbuf(stdout, NULL, _IONBF, 0)` ensures progress lines arrive immediately, which is critical on Windows where pipe buffering can cause long delays.
- **Stderr for logs** — All `LOG_INFO`/`LOG_WARN`/`LOG_ERROR` output from the library goes to stderr via `sd_set_log_callback`. This keeps stdout exclusively for the JSON protocol.
- **File-based image I/O** — Images are passed as file paths, not base64. Zero overhead for local use.

## Performance

| Workflow | cn-cli | cn-engine |
|----------|--------|-----------|
| First generation (cold start) | 12s | 12s |
| Same model, new seed | 12s | **3-4s** |
| Same model + prompt, new seed | 12s | **3-4s** |

The speedup comes from keeping the model in VRAM. With `cn-cli`, every invocation loads ~4-8 GB of model weights from disk into VRAM. With `cn-engine`, this happens once and subsequent generations go straight to inference.

## Integration Examples

### Python

```python
import subprocess, json

engine = subprocess.Popen(
    ["./cn-engine"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True, bufsize=1
)

def send(cmd):
    engine.stdin.write(json.dumps(cmd) + "\n")
    engine.stdin.flush()

def read_until_done():
    while True:
        line = engine.stdout.readline().strip()
        if not line: continue
        msg = json.loads(line)
        print(f"  [{msg['type']}] {msg.get('data', {})}")
        if msg["type"] in ("ok", "result", "error"):
            return msg

# Ping
send({"cmd": "ping", "id": "1"})
read_until_done()

# Load model
send({"cmd": "load", "id": "2", "params": {"diffusion_model": "model.gguf"}})
read_until_done()

# Generate (fast — model already loaded)
send({"cmd": "generate", "id": "3", "params": {
    "prompt": "a sunset over mountains",
    "width": 1024, "height": 1024,
    "seed": 42, "steps": 4,
    "output": "output.png"
}})
read_until_done()

# Quit
send({"cmd": "quit", "id": "4"})
engine.wait()
```

### Node.js

```javascript
const { spawn } = require('child_process');
const engine = spawn('./cn-engine', [], { stdio: ['pipe', 'pipe', 'pipe'] });

let buffer = '';
engine.stdout.on('data', (chunk) => {
    buffer += chunk.toString();
    let idx;
    while ((idx = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (line) {
            const msg = JSON.parse(line);
            console.log(`[${msg.type}]`, msg.data);
        }
    }
});

function send(cmd) {
    engine.stdin.write(JSON.stringify(cmd) + '\n');
}

send({ cmd: 'ping', id: '1' });
```

## Build Options

The engine is built by default when `SD_BUILD_EXAMPLES` is ON (the default). To disable:

```bash
cmake .. -DSD_BUILD_EXAMPLES=OFF
```

## Prompt Conditioning Cache

The engine includes a built-in LRU cache for text encoder (LLM) conditioning results. When you generate multiple images with the same prompt but different seeds, the expensive Qwen text encoder forward pass (~0.5-2s) is skipped on subsequent generations.

**How it works:**
1. First generation with a prompt: runs the full text encoder, caches the result
2. Subsequent generations with the same prompt: skips the text encoder, uses the cached condition
3. Generation with a different prompt: runs the text encoder, caches the new result

**Cache management:**
- Up to 16 prompt conditions are cached (LRU eviction when full)
- Cache is automatically cleared on `load` (model change) or `unload`
- Disable per-request with `"use_prompt_cache": false` in generate params

**Library API:** The cache is built on top of three new C API functions exposed by the library:
- `sd_compute_condition()` — compute conditioning from a prompt (standalone)
- `generate_image_with_condition()` — generate using pre-computed conditioning
- `sd_free_condition()` — free a cached condition

These are available to any application linking against the library, not just the engine.

| Workflow | Without Cache | With Cache |
|----------|---------------|------------|
| Same prompt, new seed | ~0.5-2s conditioning | **~1ms deserialize** |
| Different prompt | ~0.5-2s conditioning | ~0.5-2s conditioning (new cache entry) |

## Reference Image Latent Cache

The engine also caches VAE-encoded latent representations of reference images. When doing img2img with the same reference image across multiple generations (different seeds, prompts), the expensive VAE encode step (~1-3s) is skipped.

**How it works:**
1. First img2img generation with a reference image: runs the VAE encoder, caches the latent
2. Subsequent generations with the same reference: skips the VAE encoder, uses the cached latent
3. Modified reference image (different mtime/size): detects the change, re-encodes and caches

**Cache key:** file path + filesystem modification time + file size. This catches re-edits without needing content hashing.

**Cache management:**
- Up to 8 latents are cached (LRU eviction when full, ~2 MB each)
- Cache is automatically cleared on `load` (model change) or `unload`
- Disable per-request with `"use_ref_latent_cache": false` in generate params

**Library API:** The cache is built on three new C API functions:
- `sd_encode_ref_image()` — encode a reference image to its latent representation (standalone)
- `generate_image_with_condition_and_latents()` — generate using pre-computed condition and/or pre-encoded latents
- `sd_free_latent()` — free a cached latent

**Combined with prompt cache:** When both caches hit (same prompt + same reference image), the engine skips both the text encoder AND the VAE encoder — going directly to the denoising loop.

| Workflow | Without Cache | With Both Caches |
|----------|---------------|------------------|
| Same ref + same prompt, new seed | ~2-5s encode+condition | **~2ms deserialize** |
| Same ref, different prompt | ~1-3s encode | **~1ms deserialize** (latent cached) |
| Different ref, same prompt | ~0.5-2s condition | **~1ms deserialize** (prompt cached) |

## Version

```bash
./cn-engine --version
# cn-engine 0.1.0 (abc1234)
```
