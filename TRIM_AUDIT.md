# Condenser.cpp — Fat-Trimming Audit Report

**Files audited:** `stable-diffusion.h`, `stable-diffusion.cpp`, `tokenize_util.cpp`, plus
`denoiser.hpp`, `model.h`, `conditioner.hpp`, `diffusion_model.hpp`, `vae.hpp`,
`examples/cli/main.cpp`, `examples/common/common.hpp`

**Reference implementations checked:**
- `D:\flux2` (BFL official)
- `D:\diffusers\src\diffusers\pipelines\flux2`

---

## 1. PREDICTION TYPES — 5 of 6 are dead

**File:** `stable-diffusion.h` lines 72–79, `stable-diffusion.cpp` lines 1231–1237

Flux 2 Klein uses exactly **one** prediction type: `FLUX2_FLOW_PRED` (flow matching, velocity prediction). The remaining five are legacy baggage from SD1.x / SDXL / SD3 / Flux 1:

| Enum value | Used by | Verdict |
|---|---|---|
| `EPS_PRED` | SD 1.x, SDXL | **CUT** |
| `V_PRED` | SD 2.x | **CUT** |
| `EDM_V_PRED` | EDM models | **CUT** |
| `FLOW_PRED` | SD3 | **CUT** |
| `FLUX_FLOW_PRED` | Flux 1 | **CUT** |
| `FLUX2_FLOW_PRED` | Flux 2 Klein | KEEP |

The string array `prediction_to_str[]` and the `str_to_prediction()` / `sd_prediction_name()` functions would shrink accordingly. The `--prediction` CLI arg in `common.hpp` still accepts all 6 values.

**Impact:** Low risk. The prediction type is auto-detected for Flux 2 Klein; the user almost never sets it manually.

---

## 2. SAMPLING METHODS — 12 of 14 are unnecessary

**File:** `stable-diffusion.h` lines 39–54, `denoiser.hpp` lines 150–1280

The BFL reference implementation uses **Euler only**. All other samplers are generic k-diffusion methods ported from the upstream SD ecosystem. They'll technically work (they operate on the same `(x - denoised)/sigma` ODE) but were never designed or tested for Flux 2's flow-matching sigma schedule.

| Method | Lines in denoiser.hpp | Verdict |
|---|---|---|
| `EULER_SAMPLE_METHOD` | ~210–247 | **KEEP** (canonical for Flux 2) |
| `EULER_A_SAMPLE_METHOD` | ~152–210 | Consider keeping (cheap, popular) |
| `HEUN_SAMPLE_METHOD` | ~249–312 | Consider keeping (higher-order, useful) |
| `DPM2_SAMPLE_METHOD` | ~313–380 | CUT candidate |
| `DPMPP2S_A_SAMPLE_METHOD` | ~381–460 | CUT candidate |
| `DPMPP2M_SAMPLE_METHOD` | ~461–510 | CUT candidate |
| `DPMPP2Mv2_SAMPLE_METHOD` | ~511–560 | CUT candidate |
| `IPNDM_SAMPLE_METHOD` | ~561–597 | CUT candidate |
| `IPNDM_V_SAMPLE_METHOD` | ~598–670 | CUT candidate |
| `LCM_SAMPLE_METHOD` | ~671–710 | CUT candidate |
| `DDIM_TRAILING_SAMPLE_METHOD` | ~711–860 | **CUT** (CompVis-based, doesn't apply to flow matching) |
| `TCD_SAMPLE_METHOD` | ~861–1010 | **CUT** (CompVis-based, doesn't apply to flow matching) |
| `RES_MULTISTEP_SAMPLE_METHOD` | ~1011–1110 | Consider keeping (efficient multi-step) |
| `RES_2S_SAMPLE_METHOD` | ~1111–1220 | Consider keeping (second-order) |

**DDIM Trailing** and **TCD** are the strongest cut candidates — they use `alphas_cumprod` / `compvis_sigmas` which are SD1.x/SDXL DDPM concepts that don't apply to Flux 2's flow-matching formulation. They'd produce incorrect results with Flux 2's sigma schedule.

**Recommendation:** Keep Euler (mandatory), optionally keep Euler A, Heun, and Res Multistep if you want variety. Cut the rest. That removes ~800 lines from `denoiser.hpp`.

---

## 3. SCHEDULERS — most are ignored

**File:** `stable-diffusion.h` lines 58–70, `stable-diffusion.cpp` lines 1200–1230

`Flux2FlowDenoiser::get_sigmas()` **ignores the `scheduler_t` parameter entirely** — it always computes the empirical mu-shifted schedule from the BFL reference. The scheduler enum exists for API compatibility but does nothing for Flux 2 Klein.

| Scheduler | Used by Flux 2? | Verdict |
|---|---|---|
| `DISCRETE_SCHEDULER` | Ignored | Could cut, but harmless |
| `KARRAS_SCHEDULER` | Ignored | Could cut |
| `EXPONENTIAL_SCHEDULER` | Ignored | Could cut |
| `AYS_SCHEDULER` | Ignored | Could cut |
| `GITS_SCHEDULER` | Ignored | Could cut |
| `SGM_UNIFORM_SCHEDULER` | Ignored | Could cut |
| `SIMPLE_SCHEDULER` | Ignored | Could cut |
| `SMOOTHSTEP_SCHEDULER` | Ignored | Could cut |
| `KL_OPTIMAL_SCHEDULER` | Ignored | Could cut |
| `LCM_SCHEDULER` | Only for LCM sampler (also a cut candidate) | Could cut |
| `BONG_TANGENT_SCHEDULER` | Ignored | Could cut |

**Recommendation:** Since all schedulers are ignored by Flux 2, you could collapse this to a single `DEFAULT_SCHEDULER` or keep the enum for `--scheduler` CLI compatibility but remove the string arrays. Low priority — these are just a few string arrays.

---

## 4. DENOISER BASE CLASS — dead fields

**File:** `denoiser.hpp` lines 31–55

`FluxFlowDenoiser` (the Flux 1 base class inherited by `Flux2FlowDenoiser`) has dead members:

| Member | Lines | Verdict |
|---|---|---|
| `float sigmas[TIMESTEPS]` | L32 | **CUT** — 4KB array, never read by Flux2FlowDenoiser |
| `float sigma_data = 1.0f` | L35 | **CUT** — never read anywhere |
| `void set_parameters(float shift)` | L43–47 | **CUT** — populates the dead sigmas[] array |
| `float sigma_min()` | L50–52 | **CUT** — reads from dead array |
| `float sigma_max()` | L54–56 | **CUT** — reads from dead array |
| `FluxFlowDenoiser(shift=1.15f)` constructor | L39–41 | Simplify — remove `set_parameters()` call |

**Keep:** `set_shift()`, `sigma_to_t()`, `t_to_sigma()`, `get_scalings()`, `noise_scaling()`, `inverse_noise_scaling()` — all actively used.

---

## 5. ADM CHANNELS — fully dead plumbing

Two places maintain ADM (Adaptive Denoising Model) conditioning channels, an SD1.x/SDXL concept:

1. **`diffusion_model.hpp`**: `get_adm_in_channels()` returns hardcoded `768`
2. **`conditioner.hpp`**: `ConditionerParams::adm_in_channels` field
3. **`stable-diffusion.cpp`** L1581: `condition_params.adm_in_channels = ...`

`LLMEmbedder::get_learned_condition()` never reads `adm_in_channels`. The entire chain is dead.

**Recommendation:** Remove `adm_in_channels` from `ConditionerParams`, remove `get_adm_in_channels()` from `DiffusionModel`/`FluxModel`, and remove the assignment in `generate_image_internal()`.

---

## 6. VIDEO / FRAMES ARTIFACTS

| Location | Item | Verdict |
|---|---|---|
| `stable-diffusion.cpp` L861–862 | `generate_init_latent(... int frames = 1, bool video = false)` | **CUT** dead params |
| `stable-diffusion.cpp` L866 | `int T = frames;` (unused variable) | **CUT** |
| `stable-diffusion.cpp` L611 | Comment "assuming video mode" | **CUT** dead comment |
| `stable-diffusion.h` L246 | `sd_preview_cb_t` has `int frame_count` | Keep (used in callback, always 1) |
| `examples/common/common.hpp` L821 | `int fps = 16` in SDGenerationParams | Low priority — only feeds preview FPS |

---

## 7. MISLEADING NAMING — `clip` → `llm`

Throughout the codebase, "clip" is used where "LLM" would be accurate:

| Location | Current name | Should be |
|---|---|---|
| `stable-diffusion.h` L168 | `keep_clip_on_cpu` | `keep_llm_on_cpu` |
| `stable-diffusion.cpp` L63 | `clip_backend` | `llm_backend` |
| Common.hpp CLI | `--clip-on-cpu` | `--llm-on-cpu` |

**Recommendation:** Rename for clarity. This is cosmetic but reduces confusion for anyone reading the code.

---

## 8. tokenize_util.cpp — KEEP AS-IS

This file (988 lines) implements:
- Unicode-aware BPE tokenization for Qwen2/Mistral-style patterns
- UTF-8 ↔ codepoint conversion
- `token_split()` and `split_with_special_tokens()`

The large Unicode `is_letter()` table (lines 13–695) is bulky but **necessary** — the LLM tokenizer needs it. No dead code here. The file is as lean as it can be without breaking tokenization.

---

## 9. CONVERT MODE — decision point

The `CONVERT` mode in `SDMode` / `main.cpp` converts safetensors models to GGUF format. It's useful as a utility but is **not** image generation. If you want the slimmest possible binary, it could be a separate tool. But it's small and harmless.

---

## SUMMARY — Priority-ordered cut list

### High Priority (significant dead code)
1. **Sampling methods:** Remove DDIM Trailing (~150 lines) and TCD (~150 lines) — mathematically wrong for flow matching. Consider removing DPM2, DPM++2s_a, DPM++2M, DPM++2Mv2, iPNDM, iPNDM_v, LCM (~600 more lines)
2. **Prediction types:** Remove 5 of 6 enum values + string arrays (~30 lines)
3. **Denoiser dead fields:** Remove `sigmas[]` array, `sigma_data`, `set_parameters()`, `sigma_min()`, `sigma_max()` (~25 lines, 4KB memory)

### Medium Priority (dead plumbing)
4. **ADM channels:** Remove `get_adm_in_channels()`, `adm_in_channels` field, assignment line (~10 lines)
5. **Video params:** Remove `frames`/`video`/`T` from `generate_init_latent()` (~5 lines)
6. **Scheduler enum:** Could collapse but low impact (just string arrays)

### Low Priority (cosmetic)
7. **Rename clip → llm** in variable names, CLI args
8. **Remove dead comments** referencing video mode
