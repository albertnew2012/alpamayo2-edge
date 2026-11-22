# Concrete model structure — Alpamayo 2 Edge (2.65 B)

> Every number below was dumped from the **loaded checkpoint**
> (`nvidia/Alpamayo2-Edge` @ `f8951d09`, 2 shards, 907 tensors) or measured on the README
> example (clip `ea7bbd31…`, t0 = 4.0 s, *"Turn right in 30m"*) on an RTX 3090. None of it
> comes from the model card or from memory.
> Reproduce: [`01_anatomy.py`](scripts/01_anatomy.py) (parameters, shapes, tokenizer),
> [`02_prompt_and_tokens.py`](scripts/02_prompt_and_tokens.py) (token layout, M-RoPE),
> [`04_walkthrough.py`](scripts/04_walkthrough.py) (shapes and timings of one call).

Code: [`Alpamayo2Edge`](../src/alpamayo2_edge/models/alpamayo2_edge.py) owns a
`Cosmos3EdgeForConditionalGeneration` (from `transformers==5.17.0`, the Cosmos3-Edge
backbone) as `self.vlm` and an [`ExpertModel`](../src/alpamayo2_edge/models/expert.py) as
`self.expert`. Dimensions come from the checkpoint's `config.json`; the action space,
diffusion and projections are Hydra `_target_`s inside it.

The diagram version of this page: [`diagrams/alpamayo2_edge_architecture.svg`](diagrams/alpamayo2_edge_architecture.svg).
The 6-candidate data flow of pipeline 3-1, traced tensor by tensor: [12 — pipeline 3-1](12_PIPELINE_3-1.md).

---

## 0. End-to-end data flow of one `sample_trajectories_from_data()`

```
┌── CAMERAS (6 cams, 22 frames) ───────────────────┐ ┌── EGO HISTORY ──────────┐ ┌── ROUTE ────────────┐
  cam 1 front wide 120° ┐ 5 frames each             20 poses @10 Hz            "Turn right in 30m"
  cam 6 front tele  30° ┘ −2, −1, −0.5, −0.1, 0 s    −1.9 … 0 s, t0 ego frame   plain text, supplied
  cam 0/2 cross L/R 120° ┐ 3 frames each             xyz [20,3] rot [20,3,3]    by the caller (the
  cam 3/5 rear L/R   70° ┘ −1, −0.1, 0 s                     │                  dataset has none)
  1080×1920 uint8, actual capture times kept         DeltaTrajectoryTokenizer            │
                │                                    19 Δ × (dx,dy,dz) → 57 bins          │
   sort camera→time, bicubic resize IN uint8:        <i0>…<i999> (0.8 cm xy step)          │
   front/tele → 320×576 (184 320 px)                          │                           │
   others     → 224×416 ( 93 184 px)                          │                           │
   normalize (x/255−0.5)/0.5, patchify 16×16                  │                           │
                ▼                                             │                           │
   pixel_values [11 568, 768]   grid_thw [22, 3]              │                           │
                │                                             │                           │
   SigLIP2 ViT   27 blocks × 1152, attention within an image  │                           │
   412.65 M      learned 16×16 position grid resized per image│                           │
                ▼  [11 568, 1152]                             │                           │
   projector     LN → 2×2 merge (4608) → 11520 → GELU → 2048  │                           │
   76.69 M       ▼  [2 892, 2048]  (180 per front/tele frame, │                           │
                     91 per side/rear frame)                  │                           │
                └──────────────────────┬──────────────────────┴───────────────────────────┘
                                       ▼
     PROMPT [1, 3 261] = 2 892 image + 57 history + 312 text  ← measured
     system · per camera "Front camera: " then per frame "<t=-2.0s><|vision_start|>[pads]<|vision_end|>"
     · <|traj_history_start|> 57 × <|traj_history|> <|traj_history_end|> · "Navigation input is: …"
     · task sentence · <|im_end|> · <|im_start|>assistant\n<think></think>
                                       │
       masked_scatter: image embeddings over <|image_pad|>; fuse_traj_tokens: the 57
       <|traj_history|> ids are overwritten with history bin ids (vocabulary, not embeddings)
                                       │
╔══════════════════════════════════════▼══════════════════════════════════════════════════╗
║ VLM text decoder  —  Cosmos3-Edge (Nemotron-style, Llama-derived)          self.vlm      ║
║ 28 layers · width 2048 · GQA 16 q / 8 kv heads · head_dim 128 · MLP 9216 relu² (no gate) ║
║ interleaved M-RoPE [24,20,20] θ=1e8 · vocab 135 168 · lm_head untied       2.45 B (92.5%) ║
║ ► prefill prompt[:-1] ONCE (138 ms), cache replicated per CoT (shared prefill)           ║
║ ► sample: top_p 0.98, T 0.6; 4000 trajectory bins and <|im_end|> forced to −inf         ║
║ ► StopAfterEOS: stop one step after <|traj_future_start|> so the anchor is in the cache  ║
╚══════════════════════════════════════╤══════════════════════════════════════════════════╝
                                       │ 23 tokens generated (incl. anchor), 17 ms/token
          CoT: "Slow down and turn right at the intersection to follow our route,
                as the lead vehicle has cleared our path."
                                       │
                ┌──────────────────────┴───────────────────────┐
                │ KV cache: 28 × (K, V), each [1, 8, 3 284, 128] │  read 10×, written once;
                │ 112 KiB per token → 359 MiB                    │  crop()ped back after
                └──────────────────────┬───────────────────────┘  every Euler step
                                       │
   ┌───────────────────────────────────┴────────────────────────────────────────────────┐
   │   FLOW MATCHING (FlowMatching._euler, 10 steps, t: 0 → 1)              self.expert │
   │                                                                                    │
   │   x₀ ~ N(0, I) [1, 64, 2]   own torch.Generator(seed)                              │
   │      │                                                                             │
   │   action_in_proj  0.56 M: per channel 10 log-freq sin/cos (+√2), + t → 60 feats    │
   │      │            MLP 60→512→512→512 (SiLU, RMSNorm) → LayerNorm   [1, 64, 512]     │
   │      ▼                                                                             │
   │   expert decoder 198.2 M: 28 layers × width 512, MLP 3840 relu²                    │
   │      q 512→2048 (16×128), k/v 512→1024 (8×128) — the VLM's KV geometry             │
   │      layer i attends [VLM layer i cache (3 284) ; own 64 K/V], NON-causal          │
   │      positions 728…791 (= right after the anchor, + rope_delta)                    │
   │      ▼                                                                             │
   │   action_out_proj Linear 512→2 → v [1, 64, 2];   x ← x + 0.1·v   (×10)              │
   └───────────────────────────────────┬────────────────────────────────────────────────┘
                                       ▼
   x₁ [1, 64, 2] = normalized (accel, curvature) per 0.1 s step
                                       │
   UnicycleAccelCurvatureCourseActionSpace.action_to_traj:
   a = 0.680·x[...,0] + 0.029 m/s²,  κ = 0.0261·x[...,1] + 0.0003 1/m   (bf16-held constants)
   v0, course θ0 from the history → integrate v, θ (κ·v·dt), x/y by trapezoid along the course
   yaw = course − θ0 + 2.5e-3·(a_lat − a_lat,0)   (rear-axle sideslip model)
                                       ▼
   OUTPUT ► pred_xyz [1, sets, samples, 64, 3]   pred_rot [1, sets, samples, 64, 3, 3]
            64 poses × 0.1 s = 6.4 s, ego frame at t0;  extra["cot"] per candidate
```

Three load-bearing facts:

1. **The trajectory is decoded from the KV cache that wrote the reasoning.** The expert has
   no input of its own besides noise and time: everything it knows about the scene, the
   history and the route comes through attention to the VLM's cached keys/values, *including
   the CoT tokens the VLM just sampled*. Reasoning and action share one representation.
2. **The expert is a narrow transformer with wide attention.** Hidden size 512 (¼ of the
   VLM), but its q/k/v projections land in exactly the VLM's 16 q × 8 kv × 128 geometry, so
   expert layer *i* can attend the VLM's layer-*i* cache with no adapter. 28 layers ↔ 28
   layers. The constructor enforces this (`model_type`, `num_hidden_layers`,
   `num_key_value_heads`, `head_dim` must match).
3. **The route is just prompt text.** `"Navigation input is: Turn right in 30m"` sits in
   the user turn. No route tokens, no classifier-free guidance (the config carries
   `inference_guidance_weight: 3.0` but `use_classifier_free_guidance: false`, and the model
   refuses CFG at construction).

---

## 0b. Parameter tree (from `01_anatomy.py`)

```
Alpamayo2Edge ·························································· 2,651,163,378  (2.6512 B)
│
├── vlm  Cosmos3EdgeForConditionalGeneration ·········· 2,452,393,712   92.50 %
│   ├── model.visual  SigLIP2 ViT ·························   412,649,712   15.56 %
│   │   ├── embeddings.patch_embedding  Linear 768→1152 ·······   885,888
│   │   ├── embeddings.position_embedding  256 × 1152 ··········   294,912   (16×16 grid)
│   │   ├── encoder.layers × 27  @ 15,239,504 ················ 411,466,608
│   │   └── post_layernorm ···································     2,304
│   ├── model.projector  2×2 merger + MLP ··················    76,692,992    2.89 %
│   ├── model.language_model ································ 1,686,226,944   63.60 %
│   │   ├── embed_tokens  135,168 × 2048 ·····················   276,824,064   10.44 %
│   │   ├── layers × 28  @ 50,335,744 ······················· 1,409,400,832   53.16 %
│   │   └── norm ···········································      2,048
│   └── lm_head  135,168 × 2048 (untied) ·····················   276,824,064   10.44 %
│
└── expert  ExpertModel ································    198,769,666    7.50 %
    ├── expert  Cosmos3EdgeTextModel, embed_tokens deleted ···   198,210,048
    │   ├── layers × 28  @ 7,078,912 ························   198,209,536
    │   └── norm ···········································        512
    ├── action_in_proj  PerWaypointActionInProjV2 ············       558,592    0.021 %
    ├── action_out_proj  Linear 512→2 ························         1,026
    └── action_space / diffusion ·····························             0   (constants only)
```

Matches the model card exactly (2,651,163,378 total; expert incl. projections 198,769,666).

What is *driving-specific* by construction: the expert stack and its two projections
(**198.8 M, 7.5 %**), plus whatever post-training did to the VLM. The VLM itself is a
general Cosmos3-Edge reasoner whose vocabulary was extended with 4,000 trajectory bins and
30 driving special tokens. The genuinely new action-interface parameters (in/out
projections) are **0.56 M — 0.021 %**.

### One layer of each stack

| stack | tensor | shape | params |
|---|---|---|---|
| **vision layer** (×27, 15.24 M) | `self_attn.{q,k,v,out}_proj` | `[1152, 1152]` + bias | 4 × 1.33 M |
| | `mlp.fc1` / `mlp.fc2` | `[4304, 1152]` / `[1152, 4304]` + bias | 9.92 M |
| | `layer_norm1/2` | `[1152]` w + b | |
| **projector** (76.69 M) | `norm` | LayerNorm(1152) | |
| | `linear_fc1` | `[11520, 4608]` + bias | 53.10 M |
| | `linear_fc2` | `[2048, 11520]` + bias | 23.60 M |
| **VLM decoder layer** (×28, 50.34 M) | `q_proj` / `o_proj` | `[2048, 2048]` / `[2048, 2048]` | 8.39 M |
| | `k_proj`, `v_proj` | `[1024, 2048]` | 4.19 M |
| | `mlp.fc1` / `mlp.fc2` | `[9216, 2048]` / `[2048, 9216]` | 37.75 M |
| | `input_layernorm`, `post_attention_layernorm` | RMSNorm `[2048]` | |
| **expert layer** (×28, 7.08 M) | `q_proj` / `o_proj` | `[2048, 512]` / `[512, 2048]` | 2.10 M |
| | `k_proj`, `v_proj` | `[1024, 512]` | 1.05 M |
| | `mlp.fc1` / `mlp.fc2` | `[3840, 512]` / `[512, 3840]` | 3.93 M |
| **action_in_proj** (0.56 M) | `encoder.trunk.0` | Linear 60→512 | |
| | `trunk.2/5` RMSNorm, `trunk.3/6` | Linear 512→512 ×2 | |
| | `norm` | LayerNorm(512) | |
| **action_out_proj** | `weight`, `bias` | `[2, 512]`, `[2]` | 1,026 |

No biases anywhere in either text stack (`attention_bias=False`, `mlp_bias=False`); the
vision tower and projector do have biases. 44 % of an expert layer is attention
projections (q/o are 2048-wide), against 25 % in a VLM layer — that is the cost of reading
the VLM's cache with full-width heads.

---

## 1. Vision tower — SigLIP2 with per-image resolution

[`Cosmos3EdgeVisionModel`](../.venv/lib/python3.12/site-packages/transformers/models/cosmos3_edge/modeling_cosmos3_edge.py)
(read the HF source; it is short).

- **Patch embedding is a Linear, not a Conv**: the processor already patchifies
  (16×16×3 = 768 values per row of `pixel_values`). `temporal_patch_size = 1`: every frame
  is its own image. Unlike Qwen2/3-VL there is no temporal merge of two frames into one patch.
- **Position embedding is a learned 16×16 grid** (`num_patches=256`) that is bilinearly
  resized (antialiased) to every image's own patch grid — 20×36 for front/tele frames, 14×26
  for the others — then reordered into the processor's 2×2 block-major layout.
- **Attention never crosses images.** The 11,568 patches of all 22 images are packed into
  one sequence and attention is split by `cu_seqlens` (FA2 varlen, or a loop of SDPA calls
  per image). So all cross-camera and cross-time fusion happens in the LLM, not in the ViT.
- 27 pre-LN blocks × width 1152, 16 heads, MLP 4304 with `gelu_pytorch_tanh`, final
  `post_layernorm`. This is the SigLIP2-so400m shape.
- Output dtype is FP32 under BF16 autocast (LayerNorm upcasts), then the projector casts back.

**Measured:** 143 ms for 22 images on the 3090 (eager, SDPA).

## 2. Projector — a 2×2 merger with a very wide MLP

```
[11 568, 1152] --LayerNorm(1152)--> --view(-1, 4·1152)--> [2 892, 4608]
               --Linear 4608→11520--> GELU --Linear 11520→2048--> [2 892, 2048]
```

The LayerNorm is applied **per patch before** grouping (`use_postshuffle_norm=False`). The
2×2 groups are adjacent patches because the processor emits patches block-major. 76.7 M
parameters is large for a projector (19 % of the vision tower) — the hidden width 11,520
is 5.6× the LLM width.

One LLM token therefore covers a 32×32 pixel block of the *resized* image.

## 3. Token layout of the driving prompt (from `02_prompt_and_tokens.py`)

| # | camera | label | resized | grid (t,h,w) | patches | LLM tokens |
|---|---|---|---|---|---|---|
| 0–2 | Front left (0) | −1.0 / −0.1 / 0.0 s | 224 × 416 | (1, 14, 26) | 364 | **91** each |
| 3–7 | Front (1) | −2.0 / −1.0 / −0.5 / −0.1 / 0.0 s | 320 × 576 | (1, 20, 36) | 720 | **180** each |
| 8–10 | Front right (2) | −1.0 / −0.1 / 0.0 s | 224 × 416 | (1, 14, 26) | 364 | 91 |
| 11–13 | Rear left (3) | −1.0 / −0.1 / 0.0 s | 224 × 416 | (1, 14, 26) | 364 | 91 |
| 14–16 | Rear right (5) | −1.0 / −0.1 / 0.0 s | 224 × 416 | (1, 14, 26) | 364 | 91 |
| 17–21 | Front tele (6) | −2.0 / −1.0 / −0.5 / −0.1 / 0.0 s | 320 × 576 | (1, 20, 36) | 720 | 180 |
| | **total** | | | | **11 568** | **2 892** |

| token class | count |
|---|---|
| image (`<|image_pad|>`, id 19) | 2 892 |
| history placeholders → bins | 57 |
| other text and specials | 312 |
| **prompt** | **3 261** |

The prompt, with image pads collapsed (exact, from `study/outputs/prompt_collapsed.txt`):

```
<|im_start|>system
You are a driving assistant that generates safe and accurate actions.<|im_end|>
<|im_start|>user
Front left camera: <t=-1.0s><|vision_start|>[91 x <|image_pad|>]<|vision_end|><t=-0.1s>…<t=0.0s>…
Front camera: <t=-2.0s><|vision_start|>[180 x <|image_pad|>]<|vision_end|><t=-1.0s>… (5 frames)
Front right camera: … Rear left camera: … Rear right camera: … Front telephoto camera: … (5 frames)
<|traj_history_start|>[57 x <|traj_history|>]<|traj_history_end|>
Navigation input is: Turn right in 30m
Your task: output the chain-of-thought reasoning of the driving process, then output the future trajectory.<|im_end|>
<|im_start|>assistant
<think></think>
```

Design notes worth seeing:

- **Time is text, not position.** Each frame is preceded by a relative-time label computed
  from the *actual* capture timestamp relative to the newest frame (`<t=-0.1s>`; a `z`
  format flag turns `-0.0` into `0.0`). Frames are sorted camera-major, then by time.
- **Per-camera pixel budgets.** Front and tele get 163,840–196,608 px (→ 320×576); the four
  side/rear cameras get half (81,920–98,304 px → 224×416). The 6-camera, 22-frame input
  costs 2,892 image tokens — about what Alpamayo 1.5 spends on 4 cameras × 4 frames (2,880).
- **The empty `<think></think>`** is what the chat template emits with
  `enable_thinking=False`. The model then writes its driving reasoning as the answer text.
  `prepare_model_inputs` refuses a template that does not end that way.

### Ego history → 57 tokens

`DeltaTrajectoryTokenizer(pad_origin_at_beginning=False)`: 20 poses → 19 consecutive
deltas → `(dx, dy, dz)` each quantized into 1000 bins over `[−4, 4]` m (xy) and
`[−10, 10]` m (z), i.e. **0.80 cm per 0.1 s = 0.08 m/s speed resolution** in xy. Ids are
`<i0>`…`<i999>` = 131,072…132,071. On the example (7.6 m/s, straight): bins
`[583, 499, 500]` → `[595, 499, 500]`. `decode(encode(history))` re-integrates the deltas
backwards from the origin; rounding accumulates to **6.8 cm** at the oldest pose.

Rotations are not tokenized (`predict_yaw=False`); the action space reads them separately
(see §8).

## 4. Multimodal RoPE (interleaved, θ = 1e8)

`head_dim 128` → 64 rotary frequency pairs, split `mrope_section = [24, 20, 20]` across
(t, h, w). Cosmos3-Edge **interleaves** them (`recomposition_frequencies`): pairs
1, 4, 7, …, 58 take the h position, 2, 5, …, 59 take w, and the rest (0, 3, …, 57 and the
four lowest frequencies 60–63) take t.

Positions from `get_rope_index` (measured):

| token | seq idx | (t, h, w) |
|---|---|---|
| last token before image 0 | 33 | (33, 33, 33) |
| image 0 first token | 34 | (34, 34, 34) |
| image 0 end of row 0 | 46 | (34, 34, 46) |
| image 0 start of row 1 | 47 | (34, 35, 34) |
| image 0 last token | 124 | (34, 40, 46) |
| first token after image 0 | 125 | **(47, 47, 47)** |
| last prompt token | 3 260 | (704, 704, 704) |

An image of h×w LLM tokens advances the text position by **max(h, w)**, not h·w (13
for a 7×13 image, 18 for 10×18). So the 3,261-token prompt ends at position 704 and
`rope_delta = 704 + 1 − 3261 = −2556`. Every later token (CoT, anchor, the expert's 64
action tokens) uses `cache_length + rope_delta`. Each image's t component is its own start
position; inter-frame time lives in the `<t=…>` text, not in RoPE.

## 5. VLM text decoder (Cosmos3-Edge, Nemotron-style)

Per layer: `RMSNorm → GQA attention → residual → RMSNorm → MLP → residual`.

- **Attention**: 16 query heads, 8 KV heads (GQA 2:1), head_dim 128, no QK-norm, no bias,
  causal, M-RoPE as above.
- **MLP**: `fc1 2048→9216 → relu² → fc2 9216→2048`. Two matrices, no gate — the
  Nemotron/Minitron squared-ReLU design, not SwiGLU.
- **Vocabulary 135,168** rows (embedding and untied lm_head, 276.8 M each = 20.9 % of the
  model between them):

  | ids | content |
  |---|---|
  | 0–131,071 | base tokenizer (Mistral-Tekken style: `<unk>`=0 pad, `<s>`, `[INST]`, `<|im_start|>`=10, `<|im_end|>`=11 eos, `<think>`=12, `</think>`=13, `<|image_pad|>`=19, `<|vision_start|>`=20 …) |
  | 131,072–132,071 | `<i0>`…`<i999>` history bins |
  | 132,072–135,071 | `<i1000>`…`<i3999>` future bins (3000) |
  | 135,072–135,101 | 30 driving specials: `<|traj_history_start|>`=135,077, `<|traj_history_end|>`=135,079, `<|traj_future_start|>`=135,084 (**the expert anchor**), `<|traj_history|>`=135,087, `<|traj_future|>`=135,088, `<|route_*|>`, `<|question_*|>`, `<|cot_*|>`, `<|meta_action_*|>`, `<|box_*|>` … |
  | 135,102–135,167 | padding rows (tokenizer length is 135,102) |

- **KV cache**: 28 layers × 2 (K, V) × 8 heads × 128 × 2 bytes = **112 KiB per token**.
  The example's 3,284-token cache (prompt + CoT + anchor) is **359 MiB per CoT**.

The HF class also lists `_COSMOS3_EDGE_DROPPED_GENERATOR_KEYS` (`time_embedder`,
`mlp_moe_gen`, `action_modality_embed`, …): the Cosmos3 family has generation weights;
Edge loads only the understanding path, and `load_model` refuses any missing or unexpected
key.

## 6. The decode loop (what turns the prompt into a CoT)

[`sample_trajectories_from_data`](../src/alpamayo2_edge/models/alpamayo2_edge.py) →
[`_generate_with_shared_prefill`](../src/alpamayo2_edge/models/alpamayo2_edge.py):

1. **Shared prefill.** `self.vlm.model(input_ids[:, :-1], …)` once at batch 1 (vision
   included). Then `prompt_cache.batch_repeat_interleave(n_samples_total)` and the rope
   deltas are repeated, and `vlm.generate()` is called with the full ids and that cache, so it
   forwards only the last prompt token. Vision and prefill cost do **not** scale with the
   number of candidates; only decode and the expert do. (Alpamayo 1.5 used
   `num_return_sequences`, which expands `pixel_values` before prefill.)
2. **Logits processors**, in HF order: mask `<i0>…<i999>` and `<i1000>…<i3999>` to −∞
   (the CoT can never emit trajectory tokens), mask `<|im_end|>` (the CoT cannot end before
   the anchor), then temperature 0.6, then top-p 0.98. `generation_config.eos_token_id` is
   set to `None`.
3. **`StopAfterEOS(<|traj_future_start|>)`**: generation runs **one step past** the anchor
   in every row, because `generate` caches the token it *feeds*, not the one it samples. That
   extra step puts the anchor's K/V in the cache; the extra sampled token is discarded.
4. If `max_generation_length` lands exactly on the anchor, the anchor is forwarded by
   hand through `language_model` (reusing the vision cache). A row with no anchor at all
   raises `No <traj_future_start> token found`.
5. `cache.crop(prefix_length)`; tokens after each row's anchor are replaced by padding.

## 7. The action expert

[`ExpertModel`](../src/alpamayo2_edge/models/expert.py) is `AutoModel.from_config(llm_config)`
— a full `Cosmos3EdgeTextModel` at width 512 — with `embed_tokens` deleted. Its only input is
`inputs_embeds` from `action_in_proj`.

**Attention per Euler step** ([`_sample_actions`](../src/alpamayo2_edge/models/alpamayo2_edge.py)):

```
queries:  64 action tokens                    [B*, 16 heads, 64, 128]
keys:     VLM layer-i cache [prompt + CoT]   ┐ [B*, 8, 3 284, 128]   (frozen, from the VLM)
          ⊕ the expert's own 64 keys         ┘ [B*, 8,    64, 128]   (written, then cropped)
mask:     4-D float [B*, 1, 64, 3 284 + 64]: 0 = visible; −inf for left padding and for cache
          columns beyond a row's own anchor (other rows' longer CoTs)
causal:   no — every action token sees every other one (is_causal=False)
positions: 3 × [B*, 64]: offset + 0…63 + rope_delta, all three M-RoPE components equal
```

After each step `cache.crop(prefix_length)` removes the 64 action K/V again, so every step
conditions only on the prompt and CoT. Under SDPA this needs the explicit 4-D mask;
with `flash_attention_2` it is reduced to a 2-D padding mask and the kernel runs non-causal.

**Precision:** `action_in_proj` ends in a LayerNorm, which autocast runs in FP32, so the
expert's residual stream is FP32 under BF16 autocast (the walkthrough prints
`torch.float32` for the step tokens). Matmuls still run in BF16.

**`action_in_proj`** (`PerWaypointActionInProjV2`, 0.56 M):

```
x_t [B, 64, 2]   t [B, 1, 1]
per channel c∈{accel, κ}: FourierEncoderV2(20): 10 freqs logspace(1…100), sin & cos, ×√2 → 20
t (the step's flow time):  FourierEncoderV2(20)                                      → 20, repeated ×64
concat → 60 → Linear 60→512, SiLU, RMSNorm, Linear 512→512, SiLU, RMSNorm, Linear 512→512 → LayerNorm
→ [B, 64, 512]  (one token per future waypoint)
```

`action_out_proj` = `Linear(512, 2)`: one velocity per waypoint and channel.

**Non-persistent constants.** The Fourier frequencies and the four action-space
normalization constants are *not* in the checkpoint; `ExpertModel._init_weights` rebuilds
them from config. They are then cast with the model, so at inference they are **bf16**:
`accel_std` 0.6810 → **0.6797**, `curvature_std` 0.026148 → 0.026123, frequencies
`[1, 1.6719, 2.7813, …]`. Decoding the same action with bf16 instead of fp32 constants
moves the 6.4 s endpoint by **3.1 cm** ([`03_action_space.py`](scripts/03_action_space.py) §4).
Whether training saw the same rounding is not knowable from the release.

## 8. Flow matching and the action space (summary; details in [04](04_ACTION_SPACE_AND_FLOW.md))

- **Flow**: rectified flow with Euler, `x_t = t·x₁ + (1−t)·ε`, target velocity `x₁ − ε`,
  10 uniform steps from t=0 (noise) to t=1 (action). Training time sampler
  `t = 0.999·(1 − Beta(1.5, 1))`. No CFG.
- **Action**: `[64, 2]` normalized `(acceleration, curvature)`, one per 0.1 s step.
- **Integration**: `UnicycleAccelCurvatureCourseActionSpace` — the GA checkpoint's
  *course-heading* space: the rig origin moves along its course, not its body yaw. Speed
  and initial course are estimated from the 20 history poses by regularized least squares.
  Output rotations add a sideslip model (`slip_gradient` 2.5e-3 rad per m/s²).

## 9. Config fields that hint at training

From `config.json` (inference does not use them):
`cotrain_expert_vlm: true`, `cotrain_detach_expert_kv: true` (expert gradients do not flow
into the VLM's K/V), `loss_weights: {future_traj: 1.0, others: 1.0}`,
`tokens_per_future_traj: 128` with a 3000-bin `DiscreteTrajectoryTokenizer` over the same
course action space (64 × 2 tokens). The vocabulary carries those 3000 future bins, but
inference masks them: the trajectory always comes from the continuous expert.

## 10. Numbers worth memorising

| | |
|---|---|
| total | **2.651 B**, 5.30 GB bf16 on disk and resident |
| VLM | 2.452 B: ViT 413 M + projector 77 M + decoder 1.41 B + embed/lm_head 2 × 277 M |
| expert | **198.8 M**, 28 × 512, attention in VLM geometry |
| prompt | **3,261** tokens (2,892 image) for 22 frames from 6 cameras |
| history | 20 poses → **57** tokens (19 deltas × 3) |
| CoT | ~20–35 tokens, then `<|traj_future_start|>` |
| KV cache | 112 KiB/token, **359 MiB** per CoT on the example |
| action | **64 × (accel, curvature)**, 10 Euler steps, 6.4 s at 10 Hz |
| peak VRAM | 9.2 GB with 6 candidates (eager), weights 5.3 GB |
