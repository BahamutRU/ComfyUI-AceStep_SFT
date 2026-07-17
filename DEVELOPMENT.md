# DEVELOPMENT.md — Architecture, Differences vs Official Repo & Roadmap

> **Purpose:** a self-reference for future development sessions. Captures how the
> official ACE-Step repo (`/home/bahamut/ACE-Step-1.5`) differs from this ComfyUI
> node, where each feature lives in both codebases (file:line), what's already
> ported, and what remains. Read this before re-investigating the codebase.

## 0. Repositories

| Repo | Path | Role |
|------|------|------|
| **This node** | `/home/bahamut/ComfyUI/custom_nodes/ComfyUI-AceStep_SFT` | ComfyUI custom node. Single-file: `nodes.py`. Self-contained, reimplements sampling on ComfyUI primitives. |
| **Official ACE-Step** | `/home/bahamut/ACE-Step-1.5` | Developer repo. Full Gradio pipeline, mixin-based handler, model ships its own `generate_audio()`. |
| **ComfyUI core** | `/home/bahamut/ComfyUI/comfy`, `comfy_extras` | Native ACE-Step 1.5 support: model loader, sampler, conditioning plumbing, basic cover/reference-audio nodes. |

## 1. The fundamental architectural split (READ FIRST)

This is the single most important fact for any future work:

**The model object loaded by ComfyUI is NOT the official repo's model class.**

- Official: `acestep/models/turbo/modeling_acestep_v15_turbo.py` → class
  `AceStepConditionGenerationModel` (a HF `PreTrainedModel`) with a full
  `generate_audio()` sampler (~370 lines incl. cover/repaint/heun/DCW).
- ComfyUI core: `comfy/ldm/ace/ace_step15.py` → a *separate* reimplementation of
  the same architecture (plain `nn.Module`) with **only** `forward()`,
  `prepare_condition()`, `encode()`. **No `generate_audio()`. No repaint logic.**
- The node loads via `comfy.sd.load_diffusion_model()` → `ModelPatcher` wrapping
  `comfy/model_base.py:ACEStep15` → `.diffusion_model` = the ComfyUI
  `AceStepConditionGenerationModel`. Weights are numerically compatible (same
  arch) but the Python class & forward contract differ.

**Consequence:** editing-task logic that in the official repo lives inside the
model's *sampler loop* (repaint step-injection, cover noise init, condition
swap) has **no home** in ComfyUI's stack. It must be re-implemented via ComfyUI
hooks (`set_model_sampler_cfg_function`, `calc_cond_batch_function`) or by
rebuilding the sampler. The node already uses these hooks for APG/ADG guidance —
that is the established extension pattern.

## 2. Task types (official) vs node support

Defined in `acestep/constants.py:76-111`:

| Task | UI mode | What it does | Official | ComfyUI core | **This node** |
|------|---------|--------------|----------|--------------|---------------|
| `text2music` | Simple/Custom | Generate from scratch | ✅ | ✅ | ✅ |
| `cover` / `cover-nofsq` | Remix | Re-style source structure | ✅ | ✅ (partial) | ✅ **ported** |
| `repaint` | Repaint | Inpaint a time region | ✅ | ❌ | ❌ **missing** |
| `extract` | Extract | Isolate a stem (vocals/drums/...) | ✅ | ❌ | ❌ **missing** |
| `lego` | Lego | Add an instrument track | ✅ | ❌ | ❌ **missing** |
| `complete` | Complete | Vocal→BGM (add accompaniment) | ✅ | ❌ | ❌ **missing** |

**Model-variant constraints** (Model Zoo, `README.md` of official repo; enforced
in `acestep/ui/gradio/events/generation/model_config.py:94-145`):

| Variant | Steps | CFG | Extra tasks beyond cover/repaint |
|---------|-------|-----|----------------------------------|
| `turbo` | 8 | OFF (baked-in, `guidance_scale=1.0`) | none |
| `sft` | 50 | ON | none |
| `base` | 50 | ON | **extract, lego, complete** ← base-only |

> The user primarily runs the **merged turbo+SFT** model → only `text2music`,
> `cover`, `repaint` are reachable. `extract`/`lego`/`complete` require the
> **base** model (worse generation quality, 50 steps).

Orthogonal features (not task types, params that apply across tasks):
- `reference_audio` — timbre/style reference (3×10s segments, ≤750 latent frames)
- `retake` — variation control (noise blending via `retake_seed`/`retake_variance`)
- flow-edit morph (issue #1156 overlay)

## 3. Conditioning data flow (the critical plumbing)

How text/source/timbre conditioning reaches the model. Three layers:

### 3a. Conditioning keys → `model_conds` (`ACEStep15.extra_conds`)
`comfy/model_base.py:2170-2211`. Reads from the pooled conditioning dict (`**kwargs`):

| Conditioning key | → model_conds | Notes |
|------------------|---------------|-------|
| `cross_attn` | `c_crossattn` | text embed (caption) |
| `conditioning_lyrics` | `lyric_embed` | lyric embed |
| `reference_audio_timbre_latents` | `refer_audio` + `is_covers` | **cover trigger**: if present & non-empty → `is_covers=CONDConstant(True)`, `pass_audio_codes=False`. Else → silence latent + `is_covers=False`. |
| `audio_codes` | `audio_codes` | LM-generated codes (only when NOT cover) |

> **Key:** `audio_codes` and `reference_audio_timbre_latents` are mutually
> exclusive. Cover suppresses LM code generation.

### 3b. `refer_audio` format
- Shape **`[B, 64, T]`** (64 = `audio_acoustic_hidden_dim`, T = latent frames @ 25 Hz).
- `get_silence_latent(length, device)` (`comfy/ldm/ace/ace_step15.py:10`) returns `[1, 64, T]`.
- VAE-encode output (`_vae_encode_with_optional_tiling`) is also `[B, 64, T]` — **no movedim needed** when feeding cover source.
- In `extra_conds` line 2194: `refer_audio[-1][:, :, :noise.shape[2]]` — takes last list element, slices T. So the conditioning value is a **list wrapping one tensor**: `[cover_latent]`.

### 3c. How cover conditioning is built inside the model
`comfy/ldm/ace/ace_step15.py:1077-1112` (`prepare_condition`):
- `refer_audio` → FSQ tokenize+detokenize → `lm_hints` @ 25 Hz
- when `is_covers=True`: `src_latents = lm_hints` (replaces the noisy input as the context source)
- `context_latents = cat([src_latents, chunk_masks], dim=-1)` → fed to the DiT decoder

### 3d. Divergence from official repo (known, accepted)
The ComfyUI wrapper tokenizes **`refer_audio`** (the timbre/cover source). The
official repo tokenizes **`src_latents`** (a separate VAE latent of the song).
For cover, the node feeds the source as `refer_audio`, so the tokenization input
matches the cover intent — this works in practice (cover is ported & functional).
For pure timbre-reference (distinct from cover), this conflation could matter,
but that path is not currently exposed.

## 4. What's PORTED (Cover) — exact locations

Committed in `e0f473c`. Three mechanisms mapped from official → node:

| Official mechanic | Official file:line | Node implementation | Node file:line |
|------------------|--------------------|---------------------|----------------|
| `is_covers=True` + source conditioning | `modeling_acestep_v15_turbo.py:1654` (`prepare_condition`) + `:1996-2011` | Inject `reference_audio_timbre_latents` via `conditioning_set_values` | `nodes.py:2881-2916` |
| `cover_noise_strength` (partial re-noise of source) | `modeling_acestep_v15_turbo.py:2050-2064` + `renoise` `:1822` | `latent_image = cover_latent; denoise = 1 - strength` (ComfyUI's `noise_scaling` ≡ `renoise`) | `nodes.py:2946-2955` |
| `audio_cover_strength` (mid-sample condition swap) | `modeling_acestep_v15_turbo.py:2091-2102` (cover_steps switch) | `calc_cond_batch_function` hook: after `cover_steps`, swap to non-cover conditioning | `nodes.py:3008-3026, 3091-3097` + helper `_build_non_cover_conditioning` `:281` |

Node inputs (`nodes.py:2718-2751`): `cover_source` (AUDIO/LATENT),
`cover_noise_strength` (0-1), `audio_cover_strength` (0-1).

### How the cover noise-strength mapping works
Official `renoise(x, t) = t*noise + (1-t)*x` with `t = 1 - cover_noise_strength`.
ComfyUI `KSampler.set_steps` (`comfy/samplers.py:1412-1422`): for `denoise<1`,
`new_steps = steps/denoise`, then takes last `steps+1` sigmas. Plus
`noise_scaling(sigmas[0], noise, latent_image)` ≡ the same flow-matching blend.
So `denoise = 1 - cover_noise_strength` reproduces the start point. The official
finds the *nearest discrete timestep* (`nearest_t`); ComfyUI takes a denser
schedule — mathematically equivalent, not bit-identical.

### The non-cover branch (`_build_non_cover_conditioning`, `nodes.py:281-307`)
Clones *processed* conditioning (dict with `model_conds`), flips `is_covers`→False,
replaces `refer_audio` with a silence latent. Built lazily inside the hook from
`get_silence_latent`. Matches official's non-cover branch
(`modeling_acestep_v15_turbo.py:2014-2033`) which uses silence + `is_covers=False`.

## 5. What's MISSING — detailed

### 5a. Repaint (inpaint a time region) — NEXT PRIORITY
The hardest port. Three masking layers in the official repo, all living in the
*sampler loop* (no ComfyUI equivalent):

1. **Latent step-injection** — `apply_repaint_step_injection()`
   (`acestep/core/generation/handler/repaint_step_injection.py:6-30`):
   ```python
   zt_src = t_next * noise + (1.0 - t_next) * clean_src_latents
   xt = torch.where(mask, xt, zt_src)  # True=generate, False=preserve source
   ```
   Runs every step up to `injection_cutoff = round(repaint_injection_ratio * num_steps)`.
   In official: `modeling_acestep_v15_turbo.py:2207-2217`.

2. **Latent boundary blend** — `apply_repaint_boundary_blend()`
   (`repaint_step_injection.py:84-103`): soft float mask, `crossfade_frames` blend.

3. **Waveform splice** — `apply_repaint_waveform_splice()`
   (`acestep/core/generation/handler/repaint_waveform_splice.py:54-115`): post-VAE,
   replace non-repaint waveform regions with original (kills VAE artifacts).

**Repaint mode resolution** (`generate_music.py:26-48`): mode+strength →
`(injection_ratio, crossfade_frames, wav_crossfade_sec)`:
- `aggressive` → (0.0, 0, 0.0)
- `conservative` → (1.0, 25, 0.05)
- `balanced` → `(1.0-strength, round(25*inv), 0.05*inv)`

**Chunk mask construction** — `conditioning_masks.py:14-107`
(`_build_chunk_masks_and_src_latents`): for repaint builds boolean `[T]` mask True
only in `[start_latent:end_latent]`; `src_latents` = source with repaint region
replaced by silence. Latent frame conversion: `start_latent = int(start_sec * sr // 1920)`.

**Implementation plan:**
- Hook point: **`sampler_post_cfg_function`** (via `model.set_model_sampler_cfg_function`).
  After each step's denoised prediction, blend `zt = sigma*noise + (1-sigma)*clean_src`
  into the non-masked region per `repaint_mask`. The node already uses
  `set_model_sampler_cfg_function` for APG (`nodes.py:3191`).
- Carry `repaint_mask` + `clean_src_latents` via closure (same pattern as
  `branch_state`/`schedule_state`, `nodes.py:2997-3002`).
- Final boundary blend + waveform splice run on `samples_raw` post-sample
  (`nodes.py` ~3136) and post-VAE-decode (`nodes.py` ~3151) respectively.
- Repaint is NOT expressible through conditioning alone (unlike cover) — needs
  the sampler hook. This is why it wasn't trivial.
- Inputs needed: `repaint_source` (AUDIO), `repainting_start`/`repainting_end`
  (seconds), `repaint_mode` (aggressive/balanced/conservative), `repaint_strength`.

### 5b. Extract / Lego / Complete (base-model only)
- `extract`: isolate a stem. `TRACK_NAMES` (`constants.py:153-156`):
  `["woodwinds","brass","fx","synth","strings","percussion","keyboard","guitar","bass","drums","backing_vocals","vocals"]`.
- `lego`: add a new instrument track (chunk mask, no silence replacement,
  `conditioning_masks.py:88-89`).
- `complete`: add accompaniment to a single track (vocal→BGM). Special lego
  conditioning format `"Global: {}\nLocal: {}\nMask Control: true"`
  (`conditioning_text.py:104-114`).

All three: mostly prompt-instruction differences (`TASK_INSTRUCTIONS`,
`constants.py:128-139`) + track-name conditioning + `complete_track_classes`.
Requires **base** model (50 steps) — not reachable on turbo/sft/merge.
**Lower priority** unless user switches to base model.

### 5c. reference_audio (timbre reference) — distinct from cover
Official: `reference_audio` (≤750 frames, 3×10s) is a *separate* optional input
from `src_audio`. Used by all tasks for timbre/style guidance. Was in this node's
history (`e3b014a "reference_audio bugfix"`) but **removed** in a later commit.
Could be restored as a standalone conditioning input alongside cover. The core
node `comfy_extras/nodes_ace.py:ReferenceAudio` already exists for this.

### 5d. Retake (variation) — easy, not ported
`modeling_acestep_v15_turbo.py:2042-2045`:
```python
if retake_variance > 0.0:
    retake_noise = self.prepare_noise(context_latents, retake_seed)
    v_rad = retake_variance * (pi/2)
    noise = cos(v_rad)*noise + sin(v_rad)*retake_noise
```
Pure noise-space blend at sample start. Trivial to add: blend the initial noise
with a second noise seeded by `retake_seed`. Inputs: `retake_seed`, `retake_variance`.

## 6. Other notable differences (non-task)

### 6a. APG/ADG/ERG guidance (node-only enhancement)
The official repo uses standard CFG (turbo) or simple CFG (sft). This node adds:
- **APG** (Adaptive Projected Guidance) — `nodes.py:63` (`apg_guidance`), ported.
- **ADG** (Angle Dynamic Guidance) — `nodes.py:101` (`adg_guidance`).
- ERG, omega-scale, split text/lyric guidance, guidance interval decay.
All via `set_model_sampler_calc_cond_batch_function` (`nodes.py:3184`) and
`set_model_sampler_cfg_function` (`nodes.py:3188-3191`). These are *node
enhancements*, not official features.

### 6b. img2img refinement (node) vs cover (official)
Node has `latent_or_audio` + `denoise<1` (`nodes.py:2603`, `_build_source_latent`
`:401`). This is **generic latent re-noise** — the model is NOT told about the
source, so structure drifts toward the prompt. Cover (`cover_source`) is the
*correct* way to remix. They're mutually exclusive in the node (`nodes.py:2824`).

### 6c. LM audio-code generation
`AceStepSFTTextEncode` (`nodes.py:2290`) runs a **real autoregressive LM loop**
(`ace15.py:10` `sample_manual_loop_no_classes`) over the Qwen3 LM to produce
audio codes when `generate_audio_codes=True`. Skipped automatically in cover mode.
Official does the same but it's mandatory for text2music.

### 6d. Analysis model storage (recently fixed)
Analysis models (Transcriber etc.) now resolve via `_get_model_dir`
(`nodes.py:697-717`): primary `ComfyUI/models/<key>/`, legacy node-folder fallback.
See commit `92529ff`. Transcriber lives at `ComfyUI/models/ACE-Step-Transcriber/` (~22 GB).

### 6e. Analysis model CPU offload (fixed — was OOM on 16 GB)
The Transcriber (Qwen2.5-Omni, ~21 GB) does not fit in 16 GB VRAM. Previously
`_get_analysis_device_map()` returned `{"": "cuda:0"}` (whole model on GPU) → OOM
during load (BPM/Key still worked because they run on librosa/CPU, only tag/lyric
extraction loads the model).

**Fix:** a `cpu_offload` toggle on the Get Music Infos node drives
`_analysis_offload_mode` ("full" | "offload"). In offload mode:
- `_get_analysis_device_map()` → `"auto"`; `_get_analysis_max_memory()` →
  `{<gpu_index>: ~90% free VRAM, "cpu": available RAM}` (both in bytes;
  accelerate needs concrete sizes + integer GPU index, NOT `"auto"`/`"cuda:0"`).
- accelerate splits the model GPU↔CPU (Transcriber: 20 modules GPU, 16 CPU,
  ~11.9 GB VRAM). Slower generation, no OOM.

Gotchas hit during implementation (don't repeat):
- accelerate's `get_max_memory` rejects `"cpu": "auto"` and `"cuda:0"` keys — use
  bytes and integer index.
- Under dispatch, `model.device` is unreliable for inputs → use
  `_model_input_device(model)` (`nodes.py`) which reads `hf_device_map` first,
  falls back to `model.device`. All `inputs.to(model.device)` replaced.
- Models that fit (Whisper, MERT, AST) keep using `device_map` without max_memory
  — accelerate places them fully on GPU, so offload toggle is a no-op for them.

## 7. Key constants (for porting)

| Constant | Value | Source |
|----------|-------|--------|
| Sample rate | 48000 Hz | `handler.py:121` |
| Latent frame rate | 25 Hz | `sr // 1920` |
| PCM samples per latent frame | 1920 | `48000/25` |
| Latent channels | 64 | `audio_acoustic_hidden_dim` |
| Reference audio | 3×10s = 30s, ≤750 latent frames | `io_audio.py:148`, `conditioning_embed.py:47` |
| Silence latent frames (default empty ref) | 750 | `conditioning_embed.py:47` |
| Max text tokens | 256 | `conditioning_text.py:131` |
| Max lyrics tokens | 2048 | `conditioning_text.py:142` |
| Valid turbo shifts | 1.0, 2.0, 3.0 | `modeling_acestep_v15_turbo.py:1915` |
| Default repaint crossfade | 10 latent frames ≈ 0.4s | `inference.py:164` |
| `chunk_mask_mode="auto"` sentinel | 2.0 | `conditioning_masks.py:74-77` |

## 8. Extension points in this node (all proven, reuse for new tasks)

| Hook | Registration | Used for |
|------|--------------|----------|
| `model.add_object_patch("model_sampling", obj)` | `nodes.py:2958` | shift (ModelSamplingDiscreteFlow+CONST) |
| `model.set_model_sampler_calc_cond_batch_function(fn)` | `nodes.py:3184` | split/ERG/cover-swap conditioning |
| `model.set_model_sampler_cfg_function(fn, disable_cfg1_optimization=True)` | `nodes.py:3188` | APG/ADG/omega; **also the hook for repaint step-injection** |
| Closure-captured state | `nodes.py:2992-3002` (`schedule_state`/`branch_state`) | carry custom tensors (masks, src latents) into the sampler loop |
| `node_helpers.conditioning_set_values(cond, {key: [tensor]})` | `nodes.py:2911` | inject cover conditioning; same pattern for reference_audio |

`transformer_options` is **NOT used** in this node — closures are the established
way to carry per-step data.

## 9. Roadmap (priority order)

1. **Repaint** (inpaint) — highest value, hardest. See §5a. Needs
   `sampler_post_cfg_function` + waveform splice. ~2-3 sessions.
2. **reference_audio** (timbre) — restore removed feature, moderate. See §5c.
   Standalone conditioning input. ~1 session.
3. **Retake** (variations) — easy, quick win. See §5d. Noise blend. ~30 min.
4. **extract/lego/complete** — only if switching to base model. See §5b.

## 10. Quick-reference: official repo file map

```
acestep/
├── constants.py                      # TASK_TYPES, TASK_INSTRUCTIONS, TRACK_NAMES, MODE_TO_TASK_TYPE
├── inference.py                      # GenerationParams dataclass (:45-222), config, result
├── core/generation/handler/
│   ├── generate_music.py             # main orchestration (:180), repaint config (:26)
│   ├── conditioning_batch.py         # _prepare_batch (:21)
│   ├── conditioning_target.py        # _prepare_target_latents_and_wavs (:35) — VAE-encode source
│   ├── conditioning_masks.py         # _build_chunk_masks_and_src_latents (:14) — repaint mask + src
│   ├── conditioning_text.py          # _prepare_text_conditioning_inputs (:57)
│   ├── conditioning_embed.py         # timbre refer latent (≤750)
│   ├── repaint_step_injection.py     # apply_repaint_step_injection (:6), boundary_blend (:84)
│   ├── repaint_waveform_splice.py    # apply_repaint_waveform_splice (:54)
│   ├── task_utils.py                 # determine_task_type (:101), generate_instruction (:65)
│   ├── io_audio.py                   # process_src_audio (:178), process_reference_audio (:117)
│   └── audio_codes.py                # _decode_audio_codes_to_latents (:47)
├── models/turbo/modeling_acestep_v15_turbo.py
│   ├── prepare_condition (:1654)     # is_covers lm_hints swap (:1696)
│   ├── generate_audio (:1852)        # cover noise (:2050), cover_steps swap (:2091), repaint loop (:2207)
│   ├── renoise (:1822)               # t*noise + (1-t)*x
│   └── flowedit_generate_audio (:1830)
└── ui/gradio/events/generation/
    ├── mode_ui.py                    # compute_mode_ui_updates (:16) — per-mode visibility
    └── model_config.py               # per-variant steps/cfg/tasks (:94-145)
```

## 11. Changelog of this node's recent work

| Commit | What |
|--------|------|
| `cd1a1d0` | **Analysis model CPU offload** (fix Transcriber OOM on 16 GB) |
| `92529ff` | Analysis models → `ComfyUI/models/`, `.gitignore` |
| `e0f473c` | **Cover (remix) task** ported |
| `c2cfe8e` | (upstream) bugfix Loras |
| `bb2b019` | (upstream) modular Loader/Lora/TextEncode/Generate split |
