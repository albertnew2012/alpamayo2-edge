# Pipeline 3-1, end to end — model structure and data flow (6 candidates)

> Every shape, size and time below was recorded from **the exact call config 3-1 makes**
> (`06_demo_bev.py` §2: `predict(model, inputs, num_traj_sets=6)`, README clip `ea7bbd31…`,
> t0 = 4.0 s, *"Turn right in 30m"*, seed 43, SDPA, bf16) by
> [`19_pipeline_trace.py`](scripts/19_pipeline_trace.py) → `study/outputs/pipeline_trace.json`.
> The script wraps the model's own methods (it observes, it does not reimplement), and its six
> CoTs equal the ones in 3-1's `outputs/demo.json` (checked: `True`). Parameter counts come from
> [`01_anatomy.py`](scripts/01_anatomy.py). RTX 3090, eager, second (warm) call; the trace
> wrappers synchronize CUDA around every watched call, so decode times are slightly high.
> Reproduce: launch config **3-1b**, or `python study/scripts/19_pipeline_trace.py`.

Code: [`Alpamayo2Edge`](../src/alpamayo2_edge/models/alpamayo2_edge.py) owns the Cosmos3-Edge VLM
(`self.vlm`, HF `transformers==5.17.0`) and an [`ExpertModel`](../src/alpamayo2_edge/models/expert.py)
(`self.expert`). This page is the **6-candidate, call-level** companion to
[01 — model structure](01_MODEL_STRUCTURE.md) (one candidate, every module and parameter) and
[03 — one call, unrolled](03_INFERENCE_WALKTHROUGH.md) (bit-exact re-implementation of one call).

## 0. End-to-end data flow of pipeline 3-1

```
 F5 "3-1" ─► study/scripts/06_demo_bev.py
   load_sample(ea7bbd31…, t0 = 4 000 000 µs, "Turn right in 30m")    cached in tmp/sample_cache
   load_edge()   Alpamayo2Edge · bf16 · attn "sdpa" · cuda:0
   prepare()     prepare_model_inputs on CPU (1 675 ms) → to_device
   predict(model, inputs, num_traj_sets=6)     ◄── everything below is this ONE call: 1 493 ms

┌── CAMERAS: 6 cams, 22 frames ──────────────┐ ┌── EGO HISTORY ───────────┐ ┌── ROUTE ───────────┐
  image_frames [22, 1, 3, 1080, 1920] uint8     ego_history_xyz [1,1,20,3]   nav_text (a string)
  front 120° + tele 30°: 5 frames, −2 … 0 s     ego_history_rot [1,1,20,3,3] "Turn right in 30m"
  cross L/R + rear L/R:  3 frames, −1 … 0 s     20 poses @ 10 Hz, t0 frame          │
        │ sort camera → time; bicubic resize        │ 19 Δ × (dx, dy, dz)           │
        │ front/tele → 320×576, others → 224×416    │ → 57 bins <i0>…<i999>         │
        │ normalize, cut 16×16 patches (2×2 blocks)  │                               │
        ▼                                            │                               │
  pixel_values [11 568, 768]   image_grid_thw [22, 3]│                               │
        └──────────────── chat template + tokenizer ─┴───────────────────────────────┘
                                         ▼
  input_ids [1, 3 261] = 2 892 <|image_pad|> + 57 history + 312 text · attention_mask · mm_token_type_ids
════════════════════════════ GPU · inference.predict: manual_seed(43) + cuda Generator(43) ═════════
  fuse_traj_tokens: the 57 <|traj_history|> ids → history bin ids               (alpamayo2_edge.py:234)
                                         │
 ╔═══════════════════════════════════════▼═══════════════════════════════════════════════════════╗
 ║ ① SHARED PREFILL   self.vlm.model(input_ids[:, :-1])   batch 1, runs ONCE         298 ms (:139) ║
 ║   SigLIP2 ViT  [11 568, 768] → [11 568, 1152] fp32   27 blocks, attention only within    146 ms ║
 ║                an image: cu_seqlens [0, 364, 728, 1 092, 1 812, …, 11 568] (22 segments)        ║
 ║   projector    2×2 merge → MLP → [2 892, 2048] bf16   (91 per side frame, 180 per front)  4 ms ║
 ║   embed_tokens [1, 3 260] → [1, 3 260, 2048];  masked_scatter writes the 2 892 image rows       ║
 ║   M-RoPE       position_ids [3, 1, 3 260], max 703  →  rope_deltas = 704 − 3 260 = −2 556   7 ms ║
 ║   28 decoder layers, 2048 wide, CAUSAL  →  KV cache 28 × (K, V) [1, 8, 3 260, 128]   357 MiB   ║
 ╚═══════════════════════════════════════╤═══════════════════════════════════════════════════════╝
                                         │ cache.batch_repeat_interleave(6) → [6, 8, 3 260, 128] (real copies)
                                         │ rope_deltas → [6, 1]                                 (:147–152)
 ╔═══════════════════════════════════════▼═══════════════════════════════════════════════════════╗
 ║ ② CoT DECODE   self.vlm.generate(input_ids ×6 = [6, 3 261], cache)               650 ms (:153) ║
 ║   34 steps × [6, 1] token, 17.1 ms/step   (first step forwards the 3 261st prompt token)       ║
 ║   per step, per layer: q,k,v → RoPE (position = index − 2 556, first 704) → append K/V → attend ║
 ║   lm_head → [6, 135 168] → −∞ on 4 000 trajectory bins and <|im_end|> → T 0.6 → top-p 0.98    ║
 ║   → multinomial (one token per row) → StopAfterEOS: stop one step after the LAST row's anchor   ║
 ║   → sequences [6, 3 295] = 3 261 + 34      cache [6, 8, 3 294, 128]  2 162 MiB                 ║
 ╚═══════════════════════════════════════╤═══════════════════════════════════════════════════════╝
                                         │ find_eos_offset: first <|traj_future_start|> + 1 per row
                                         │   [24, 23, 33, 22, 23, 23] → offsets [3 285, 3 284, 3 294, 3 283, 3 284, 3 284]
                                         │ prefix_length = 3 294; cache.crop (no-op here); tokens after anchor → 0
                                         │ build_expert_pos_ids_and_attn_mask                     (:295–345)
                                         │   position_ids [3, 6, 64]: 729…792 | 728…791 | 738…801 | 727… | 728… | 728…
                                         │   attention_mask [6, 1, 64, 3 358] fp32, blocks [9, 10, 0, 11, 10, 10] junk cols
   ┌─────────────────────────────────────┴───────────────────────────────────────────────────────┐
   │ ③ FLOW MATCHING   _sample_actions → FlowMatching._euler, 10 steps, t: 0 → 1      530 ms (:350) │
   │   x ~ N(0, I) [6, 64, 2]   (from the seeded cuda Generator; diffusion temperature 1.0)          │
   │   for t in 0.0, 0.1, …, 0.9:                                                                  │
   │     action_in_proj(x, t)    per channel 20 Fourier feats + 20 for t → 60 → MLP → LN            │
   │                             → [6, 64, 512] fp32: one token per waypoint                         │
   │     expert.expert           28 layers × 512 wide, NON-causal, 4-D mask                         │
   │                             layer i: K,V = [VLM layer-i cache (3 294) ; own 64] = 3 358       │
   │     action_out_proj         512 → 2  →  v [6, 64, 2]                         ≈ 51.6 ms / step  │
   │     x ← x + 0.1·v;  cache.crop(3 294) drops the 64 action K/V before the next step             │
   └─────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                         ▼  x [6, 64, 2] = normalized (acceleration, curvature) per 0.1 s
   ④ action_to_traj(x, history_xyz [6, 20, 3], history_rot [6, 20, 3, 3])              2.4 ms (:359)
        un-normalize · speed v0 and course θ0 fitted on the 20 history poses · unicycle integration
                                         ▼
   OUTPUT ►  pred_xyz [1, 6, 1, 64, 3]   pred_rot [1, 6, 1, 64, 3, 3]   = [B, sets, samples, 64 poses, …]
             extra["cot"] [1, 6, 1] strings · generated_ids [6, 34] (−1 after each row's anchor)
                                         ▼
   back in 06_demo_bev.py: ADE/FDE vs the logged future, BEV plot, §3 diversity split
```

`traj_data["ego_history_xyz"][:, -1]` at line 361 picks the last **trajectory group** (the size-1
axis of `[1, 1, 20, 3]`), not the last pose. The action space gets all 20 history poses, because it
fits the starting speed and course to them ([01 §8](01_MODEL_STRUCTURE.md)).

**Cost, measured** (the traced call, warm):

| stage | time | grows with the number of candidates? |
|---|---|---|
| CPU `prepare_model_inputs` (resize 22 × 1080p, tokenize) | 1,675 ms | no, and it is outside the model call |
| ① vision + prefill | 298 ms (ViT 146, projector 4) | **no**: batch 1, shared by all 6 |
| ② CoT decode, 34 steps | 650 ms (581 ms in the 34 forwards, 17.1 ms/step) | barely per step (6 rows still read the same weights); the step count is the **longest** CoT |
| ③ expert, 10 Euler steps | 530 ms (51.6 ms/step) | **yes**: 6 × 64 queries over a 6-row cache |
| ④ and the rest (masks, crop, text decode) | ≈ 15 ms | |
| **model call** | **1,493 ms**, peak **8.58 GiB** (9.21 GB) | |

With one candidate, [05](05_ACCELERATION.md) measured 295 ms prefill, 18.5 ms per decode token and
19.6 ms per expert step (eager, no wrappers). Going to six barely changes the cost of a decode step,
because decode is launch-bound. The expert gets 2.6× slower per step. The prefill does not change.

### Why one KV cache can serve all 10 Euler steps (the invariant)

```
                     keys →   prompt (3 261)   CoT_k + anchor    junk_k          actions (64)
         queries ↓          ┌────────────────┬────────────────┬───────────────┬───────────────┐
         prompt   (①)       │ causal         │       ·        │       ·       │       ·       │
         CoT_k    (②)       │ all            │ causal         │       ·       │       ·       │
         junk_k   (②)       │ all            │ all            │ causal        │       ·       │ ← sampled while waiting
         actions  (③)       │ all            │ all            │   BLOCKED     │ all, both ways│   for the longest row
                            └────────────────┴────────────────┴───────────────┴───────────────┘
   k = 0 … 5:  CoT_k + anchor = 24 / 23 / 33 / 22 / 23 / 23     junk_k = 9 / 10 / 0 / 11 / 10 / 10
```

No prompt or CoT query ever looks at an action token (the right column is empty above the action
row). So the cached K/V do not depend on `x`, and the same cache is read by all 10 steps; only the
64 action K/V are appended and cropped each time. Measured on the expert's mask: visible cache
columns per row `[3285, 3284, 3294, 3283, 3284, 3284]`, junk columns blocked
`[9, 10, 0, 11, 10, 10]`, and the 64 × 64 action block fully visible. How the expert's attention
is spread over these blocks (prompt text, images, CoT, its own tokens) is measured in
[08](08_EXPERT_ATTENTION.md).

There are two differences from π₀.₅ ([openpi-pytorch 07](../../openpi-pytorch/study/07_MODEL_STRUCTURE.md)):
the prompt block is **causal** here (bidirectional there), and there is a middle block that the
model **writes by sampling**, the CoT. Because each row's CoT differs, each candidate needs its own
cache rows: real copies, 2,162 MiB for six. π₀.₅'s noise draws can share one prefix with `expand`.

### Inside layer *i*: two weight sets, one attention

```
  VLM layer i  (runs in ① and ②)                     expert layer i  (runs in ③, 10×)
  x⁰ [·, ·, 2048]                                     x¹ [6, 64, 512]
     │ input_layernorm  RMSNorm(2048)                    │ input_layernorm  RMSNorm(512)
     │ q_proj  2048 → 16 × 128                           │ q_proj  512 → 16 × 128
     │ k_proj  2048 →  8 × 128 ─┐                        │ k_proj  512 →  8 × 128 ─┐
     │ v_proj  2048 →  8 × 128 ─┤ RoPE, write            │ v_proj  512 →  8 × 128 ─┤ RoPE at 729…
     │                          ▼                        │                         ▼
     │            cache layer i [6, 8, 3 294, 128] ──────────────► cat → [6, 8, 3 358, 128]
     │ attend: causal                                    │ attend: 4-D mask, non-causal   ← the stacks meet here
     │ o_proj  16 × 128 → 2048                           │ o_proj  16 × 128 → 512
     │ fc1 2048 → 9216, relu², fc2 9216 → 2048           │ fc1 512 → 3840, relu², fc2 3840 → 512
     ▼                                                   ▼
```

Both stacks are the **same class**, `Cosmos3EdgeTextModel`, with **separate weights**
(`vlm.model.language_model.*` vs `expert.expert.*`). Only the residual width and the MLP width
differ. The K/V geometry (8 heads × 128) is identical, so expert layer *i* reads VLM layer *i*'s
cache with no adapter. The constructor refuses anything else
([alpamayo2_edge.py:99-103](../src/alpamayo2_edge/models/alpamayo2_edge.py#L99-L103)).
Unlike π₀.₅ there is no adaRMS gate: time enters only through `action_in_proj`.

### Where the call goes in code

```
06_demo_bev.py:89        predict(model, inputs, num_traj_sets=6)
_common.py:164           → inference.predict with DEFAULT_SETTINGS + num_traj_sets=6
inference.py:38          manual_seed(43); Generator(cuda).manual_seed(43); inference_mode + bf16 autocast
alpamayo2_edge.py:203    sample_trajectories_from_data
  :234                     fuse_traj_tokens (history placeholders → bins)
  :255-285                 GenerationConfig, logits processors, eos_token_id = None
  :286 → :115              _generate_with_shared_prefill
    :139                     ① prefill  self.vlm.model(input_ids[:, :-1], …)
    :147-152                 cache ×6, rope_deltas ×6
    :153                     ② self.vlm.generate(…, past_key_values=prompt_cache)
  :295-297                 offsets = first anchor + 1, in full-sequence indices
  :307-335                 rope_deltas padding fix, prefix_length, crop, pad after each anchor
  :337-345                 expert position_ids and 4-D mask
  :350 → :162              ③ _sample_actions → expert.diffusion.sample → _euler
    :177                     step_fn: action_in_proj → expert.expert → cache.crop → action_out_proj
  :359                     ④ action_space.action_to_traj
  :364-377                 reshape to [B, sets, samples, …]; decode the CoT text; extra
```

To step into the backbone from here, 3-1 includes `transformers.models.cosmos3_edge` as
project code (its `rules` entry), so F11 on line 139 or inside `step_fn` lands in the HF model.

---

## 0b. Parameter tree (from `01_anatomy.py`)

```
Alpamayo2Edge ··············································· 2,651,163,378  (2.651 B)
├── vlm  Cosmos3EdgeForConditionalGeneration ················ 2.452 B   92.5 %
│   ├── model.visual  SigLIP2 ViT, 27 × 1152 ················   412.6 M   15.6 %
│   ├── model.projector  2×2 merge + MLP ····················    76.7 M    2.9 %
│   ├── model.language_model  embed 276.8 M + 28 × 50.3 M ··· 1,686.2 M   63.6 %
│   └── lm_head  135,168 × 2048, untied ······················   276.8 M   10.4 %
└── expert  ExpertModel ······································   198.8 M    7.5 %
    ├── expert  Cosmos3EdgeTextModel, 28 × 7.08 M, no embed ·   198.2 M
    ├── action_in_proj  PerWaypointActionInProjV2 ···········     0.56 M
    ├── action_out_proj  Linear 512 → 2 ·····················     1,026
    └── action_space, diffusion ··································     0   (constants)
```

What runs in each stage of 3-1: ① uses `visual`, `projector`, `embed_tokens` and the 28 VLM layers.
② uses the 28 VLM layers plus `lm_head`. ③ uses only the expert's three parts and *reads* the VLM's
cache. ④ has no parameters. Full tree and per-layer shapes: [01 §0b](01_MODEL_STRUCTURE.md).

---

## 1. Config of pipeline 3-1

| field | value | set in |
|---|---|---|
| sample | clip `ea7bbd31-b7a5-4972-8dbd-7089e6b53de4`, t0 = 4.0 s, `"Turn right in 30m"` | `examples/nav_samples.json[0]` via `_common.example()` |
| dataset revision | PhysicalAI-AV `2ac2e9c` | `_common.DATASET_REVISION` |
| dtype / attention | bf16 / `sdpa` | `_common.load_edge()` |
| `num_traj_sets × num_traj_samples` | **6 × 1** (the split is only a reshape; 1 × 6 gives the same numbers) | `06_demo_bev.py --candidates 6` |
| seed | 43, for the text RNG and for the diffusion generator | `DEFAULT_SETTINGS`, `inference.predict` |
| text sampling | temperature 0.6, top-p 0.98, top-k off, ≤ 256 new tokens | `DEFAULT_SETTINGS` |
| flow matching | 10 Euler steps, noise temperature 1.0 | `DEFAULT_SETTINGS`, `FlowMatching.sample` default |
| action space | `UnicycleAccelCurvatureCourseActionSpace`: 64 × (accel, curvature), dt 0.1 s | checkpoint `config.json` |
| acceleration | off (eager) | 3-1 does not pass `--accel` |

---

## 2. The token sequence of one candidate (row k)

```
cache columns 0 ………………………………………… 3 260 │ 3 261 ……… offset_k − 1 │ offset_k … 3 293 │ (1 more sampled, never fed)
              prompt, shared by all rows       │ CoT_k + anchor            │ junk_k            │
              2 892 image · 57 history · 312 text                                            │
              positions 0 … 704                │ 705 …  (index − 2 556)                        │

expert suffix (③): 64 action tokens, positions offset_k − 2 556 + 0 … 63, the same on t, h and w
```

| row | CoT + anchor | junk in cache | expert positions | CoT (from `demo.json`) | ADE / FDE vs logged |
|---|---|---|---|---|---|
| 0 | 24 | 9 | 729 … 792 | Slow down and turn right at the intersection to follow our route, as the priority road is clear of conflicting traffic. | 0.70 / 2.55 m |
| 1 | 23 | 10 | 728 … 791 | Decelerate approaching the intersection to turn right to follow our route, yielding as required by the yield sign. | 0.33 / 1.99 m |
| 2 | **33** | 0 | 738 … 801 | Decelerate to maintain a safe following distance as the lead vehicle slows to turn left at the intersection, while preparing to turn right to follow our route. | 0.31 / 1.66 m |
| 3 | 22 | 11 | 727 … 790 | Decelerate and turn right at the intersection to follow our route, as the roadway is clear. | 1.42 / 5.15 m |
| 4 | 23 | 10 | 728 … 791 | Decelerate and turn right at the intersection to follow our route, yielding as required by the yield sign. | 0.39 / 1.02 m |
| 5 | 23 | 10 | 728 … 791 | (same text as row 4) | 0.80 / 2.81 m |

Row 2 has the longest CoT, so it sets the decode length (33 + 1 extra step = 34) and
`prefix_length` = 3,261 + 33 = 3,294. Rows 4 and 5 have identical text but different
trajectories. Their diffusion noise differs, which is the diversity that 3-1 §3 measures
(spread 1.34 m with six CoTs, 1.70 m with one CoT and six noises).

**Positions in the prompt** (M-RoPE, measured): text tokens get `(n, n, n)`, and image tokens get
`(s, s + row, s + col)`. Tokens 30–51 run `(30,30,30) … (33,33,33)`, then image 0 as
`(34,34,34), (34,34,35) … (34,34,46)`, `(34,35,34) …`. Each image moves the counter by
`max(rows, cols)` (13 or 18), not by its token count. That gives 390 distinct values on the t axis,
and 2,870 of the 3,260 prefill tokens whose three axes differ. Explained in
[01 §4](01_MODEL_STRUCTURE.md).

---

## 3. Stage by stage: every tensor (measured)

| # | stage | code | in | out |
|---|---|---|---|---|
| 0 | load clip | `_common.load_sample` → `load_physical_aiavdataset` | clip, t0 | `image_frames [22,1,3,1080,1920]` u8, `camera_indices [22]`, `absolute_timestamps [22]`, `ego_history_xyz [1,1,20,3]`, `ego_history_rot [1,1,20,3,3]`, `nav_text` |
| 1 | CPU preprocessing | [`helper.prepare_model_inputs`](../src/alpamayo2_edge/helper.py) | ↑ | `input_ids [1,3261]`, `attention_mask [1,3261]`, `mm_token_type_ids [1,3261]`, `pixel_values [11568,768]` fp32, `image_grid_thw [22,3]` |
| 2 | history fuse | `fuse_traj_tokens` (:234) | 57 placeholders | 57 bin ids in `input_ids` |
| 3 | vision tower | `vlm.model.visual` | `[11568,768]` → bf16 | `[11568,1152]` fp32 (LayerNorm upcast) |
| 4 | projector | `vlm.model.projector` | `[11568,1152]` | `[2892,2048]` bf16, split 91 / 180 per image |
| 5 | scatter | `Cosmos3EdgeModel.forward` | `embed_tokens([1,3260])` = `[1,3260,2048]` | same shape, 2,892 rows replaced |
| 6 | M-RoPE | `get_rope_index` | `input_ids`, `mm_token_type_ids`, grid | `position_ids [3,1,3260]` (max 703), `rope_deltas [[-2556]]` |
| 7 | prefill | 28 VLM layers (:139) | `[1,3260,2048]` | `last_hidden_state [1,3260,2048]`, cache 28 × `[1,8,3260,128]` (357 MiB) |
| 8 | replicate | `batch_repeat_interleave(6)` (:147) | cache, `rope_deltas` | 28 × `[6,8,3260,128]` (≈ 2.09 GiB, computed), `rope_deltas [6,1]` |
| 9 | decode | `vlm.generate` (:153) | `input_ids [6,3261]` + cache | `sequences [6,3295]`, cache 28 × `[6,8,3294,128]` (2,162 MiB) |
| 10 | anchors | `find_eos_offset` (:295) | `sequences[:, 3261:]` `[6,34]` | `[24,23,33,22,23,23]` + 3,261 = offsets |
| 11 | expert geometry | `build_expert_pos_ids_and_attn_mask` (:337) | offsets, `rope_deltas`, 3,294, 64 | `position_ids [3,6,64]`, `attention_mask [6,1,64,3358]` fp32 |
| 12 | noise | `FlowMatching._euler` | generator | `x [6,64,2]` |
| 13 | action tokens | `action_in_proj` (×10) | `x [6,64,2]`, `t [6,1,1]` | `[6,64,512]` fp32 |
| 14 | expert | `expert.expert` (×10) | `inputs_embeds [6,64,512]`, cache 3,294 | `[6,64,512]`; cache 3,358 during the call, cropped to 3,294 after |
| 15 | velocity | `action_out_proj` (×10) | `[6,64,512]` | `v [6,64,2]` bf16; `x ← x + 0.1·v` |
| 16 | trajectory | `action_to_traj` (:359) | `x [6,64,2]`, history `[6,20,3]`, `[6,20,3,3]` | `xyz [6,64,3]`, `rot [6,64,3,3]` |
| 17 | outputs | reshape (:364) | | `pred_xyz [1,6,1,64,3]`, `pred_rot [1,6,1,64,3,3]`, `extra` keys `answer, box, cot, cot_auto_labeling, generated_ids, meta_action, raw_outputs` |

**What `x` looks like across the 10 Euler steps** (std over all 6 × 64 × 2 values, measured at the
input of each step):

| t | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|---|---|---|
| std(x) | 0.98 | 0.91 | 0.86 | 0.84 | 0.86 | 0.91 | 1.00 | 1.10 | 1.23 | 1.36 |

At t = 0, `x` is unit noise. Training defines `x_t = t·plan + (1 − t)·noise`
([flow_matching.py:172](../src/alpamayo2_edge/diffusion/flow_matching.py#L172)), and the variance
of a blend of two independent signals dips in the middle. By the end, `x` carries the plan's own
spread, which is larger than 1 in normalized units for this braking right turn.

---

## 4. What is new compared with a plain VLM

| piece | params | role in 3-1 | code |
|---|---|---|---|
| expert transformer | 198.2 M | ③: reads the VLM cache and denoises 64 waypoint tokens | `ExpertModel.expert` |
| `action_in_proj` | 0.56 M | noisy (accel, curvature) + t → one 512-d token per waypoint | `PerWaypointActionInProjV2` |
| `action_out_proj` | 1,026 | 512 → velocity (2) | `Linear` |
| action space, flow matching | 0 | Euler loop, unicycle integration | constants rebuilt from config |
| driving vocabulary | 4,000 trajectory bins + 30 specials (in the 276.8 M embed / lm_head) | 57 history bins in the prompt; `<|traj_future_start|>` ends the CoT | tokenizer |
| logits processors + `StopAfterEOS` | 0 | ②: the CoT can only end with the anchor, and the anchor is cached | `alpamayo2_edge.py`, `expert_utils.py` |

The action interface itself (in/out projections) is **0.56 M parameters, 0.021 %** of the model.
The rest of the driving behaviour lives in the expert stack and in the VLM weights.

---

## 5. Contrast with π₀.₅ (openpi-pytorch)

| | π₀.₅-LIBERO ([openpi 07](../../openpi-pytorch/study/07_MODEL_STRUCTURE.md)) | Edge, pipeline 3-1 |
|---|---|---|
| shape of one call | prefix → 10 denoising steps | prefix → **CoT decode** → 10 denoising steps |
| prefix attention | bidirectional | causal |
| prefix | 3 × 256 image tokens (224², 1 view masked) + ≤ 200 text | 22 frames, 2,892 image tokens, 2 resolutions + 57 history + 312 text |
| positions | 1D RoPE | M-RoPE (t, h, w) |
| VLM / expert | Gemma-2B 2048 / Gemma-300M 1024, 18 / 18 layers | 2048 / 512, 28 / 28 layers |
| head geometry | 8 q / 1 kv × 256 | 16 q / 8 kv × 128 |
| suffix | 10 action tokens × 32-d | 64 waypoint tokens × 2-d |
| time conditioning | adaRMS on every norm | Fourier features concatenated in `action_in_proj` |
| flow time | t = 1 → 0, `dt = −0.1`, target `noise − actions` | t = 0 → 1, `dt = +0.1`, target `plan − noise` |
| several samples | noise draws share the prefix via `expand` | each candidate has its own CoT → cache rows copied |
| where the time goes | "the VLM prefix is the whole bill" (69 ms + n × 16 ms) | decode 650 + expert 530 vs prefill 298 ms |

The skeleton is the same: a VLM fills a per-layer KV cache once, and a narrower expert with the same
head geometry reads it layer by layer while it integrates a flow. Edge adds a stage in the middle
where the VLM writes its own reasoning into that cache first.

---

## 6. Reproduce

```bash
python study/scripts/19_pipeline_trace.py      # this page's numbers → study/outputs/pipeline_trace.json  (config 3-1b)
python study/scripts/06_demo_bev.py            # pipeline 3-1 itself: BEV plot, ADE/FDE, diversity       (config 3-1)
python study/scripts/04_walkthrough.py         # one call re-implemented stage by stage, bit-exact         (config 3-4)
python study/scripts/14_expert_attention.py    # where the expert's attention goes                         (config 3-5g)
```

### Check yourself

1. Which of ①, ② and ③ would get slower if you asked for 12 candidates instead of 6, and why
   does ① not?
2. Why does `generate` run 34 steps when row 0's CoT is only 24 tokens, and what stops row 0's
   extra tokens from reaching its trajectory?
3. The cache holds 3,294 tokens per row but `sequences` has 3,295 columns. Where is the missing one?
4. Why can the expert reuse the same cache for all 10 Euler steps, and what would break if
   `cache.crop(prefix_length)` were removed from `step_fn`?
