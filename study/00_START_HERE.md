# START HERE — Alpamayo 2 Edge study course

**Alpamayo 2 Edge** is NVIDIA's 2.65 B-parameter reasoning VLA for in-vehicle use. It looks
at **six cameras (22 frames)**, 2 s of ego motion and a **route sentence**, writes a short
explanation of what it is about to do (or just `<NOMINAL>` when nothing needs explaining),
and then a flow-matching expert decodes **64 (acceleration, curvature) controls → a 6.4 s
trajectory** from the very KV cache that wrote the explanation.

This repository is **inference only**: load the released checkpoint, run it on PhysicalAI-AV
clips, optionally accelerate it with `torch.compile` + CUDA graphs.

> Coming from your other two repos: it is **Alpamayo 1.5's design at a quarter of the size**
> (same "VLM writes CoT → expert reads the cache" split, same unicycle action space), with
> six cameras including the rear, the route as plain text, a sideslip-aware "course" action
> space, adaptive reasoning, and a full CUDA-graph path. Unlike **Qwen-Drive**, the VLM is not
> kept general and the planner has no direct conditioning on ego state or route: everything
> reaches the expert through the VLM's cache. → [02](02_VS_ALPAMAYO_1_5_AND_QWEN_DRIVE.md)

## The 60-second mental model

```
6 cams × (5 or 3) frames ──► SigLIP2 ViT (413 M) ─► projector (77 M) ──┐ 2 892 image tokens
20 ego poses ──────────────► 57 delta tokens <i0..i999> ───────────────┤
"Navigation input is: Turn right in 30m" ──────────────────────────────┘ 3 261-token prompt
                                    │
                    Cosmos3-Edge decoder, 28 × 2048 (1.96 B with embeddings)
                    prefill once, then sample the CoT …  "<|traj_future_start|>"
                                    │
                         KV cache 28 × [8, 3 284, 128]  ── frozen, read 10×
                                    │
noise [64, 2] ──► action_in_proj ──► expert 28 × 512, layer i attends cache i ──► v [64, 2]
                         ▲                                                          │
                         └──────────── x ← x + 0.1·v, 10 Euler steps ◄──────────────┘
                                    │
               (accel, curvature) × 64 ──► unicycle along the course ──► 64 poses, 6.4 s
```

Picture version: [`diagrams/alpamayo2_edge_architecture.svg`](diagrams/alpamayo2_edge_architecture.svg)
(open it in VS Code, or Ctrl+Shift+V on [FIGURES.md](FIGURES.md)).

---

## Read in this order

| # | doc | what you get |
|---|---|---|
| 00 | this page | the plan |
| **01** | [Model structure](01_MODEL_STRUCTURE.md) | every parameter located, the token layout, M-RoPE, the KV contract, the expert — all measured |
| 02 | [vs Alpamayo 1.5 and Qwen-Drive](02_VS_ALPAMAYO_1_5_AND_QWEN_DRIVE.md) | what is inherited, what changed, what each is good for |
| **03** | [One call, unrolled](03_INFERENCE_WALKTHROUGH.md) | nine stages, bit-exact against the library, with timings |
| 04 | [Action space and flow matching](04_ACTION_SPACE_AND_FLOW.md) | course vs yaw heading, why 1 Euler step is almost enough, where diversity comes from |
| 05 | [Acceleration](05_ACCELERATION.md) | why eager decode is launch-bound, what the CUDA graphs replace, 2.2× measured |
| 06 | [Local setup and results](06_LOCAL_SETUP_AND_RESULTS.md) | what was installed and run on this machine, every result |
| **07** | [Head to head vs Alpamayo 1.5](07_HEAD_TO_HEAD_VS_A15.md) | both models on the same 21 clips, paired: indistinguishable with the right route; Edge follows a wrong route aggressively, A1.5 barely reacts |
| 08 | [Where the expert looks](08_EXPERT_ATTENTION.md) | the flow-matching expert attends to the whole cache; how the attention mass is measured (hooks, labels, averaging), sinks, by-layer split |
| 09 | [Reasoning-tampering study](09_REASONING_ACTION_STUDY.md) | force the reasoning by hand, keep everything else fixed: direction follows the route, "stop" in the reasoning overrides speed, other words do nothing |
| 10 | [Data workflow](10_DATA_WORKFLOW.md) | clip id → streamed videos and poses → chat prompt → resized, patchified tensors; what `pixel_values [11568, 768]` is and how it is ordered |
| 11 | [RL explained](11_RL_EXPLAINED.md) | what reinforcement learning is, shown on a 3-weight toy planner, and how Qwen-Drive's RL stage maps onto it |
| **12** | [Pipeline 3-1, end to end](12_PIPELINE_3-1.md) | the 6-candidate call of 3-1 traced: every tensor from camera frames to 6 trajectories, the KV-cache invariant, one layer of each stack, timings, contrast with π₀.₅ |
| — | [FIGURES.md](FIGURES.md) | all figures inline (Ctrl+Shift+V) |

## Run in this order (VS Code: F5, pick the config; each stops on its first line)

| config | what it shows | ≈ time |
|---|---|---|
| 0a | environment, GPUs, FlashAttention, checkpoint, HF access | 10 s |
| 1a / 1b / 1c | the README CLI: 1 CoT, 6 CoTs, accelerated | 40 s / 45 s / 100 s |
| **3-1** | 22 cameras, six candidates in BEV vs the logged path, diversity split | 1 min |
| 3-1b | the same call traced stage by stage: shapes, KV cache, masks, timings ([12](12_PIPELINE_3-1.md)) | 1 min |
| 3-2 | the parameter tree | 20 s |
| 3-3 | the 3 261-token prompt, history bins, M-RoPE positions | 30 s |
| 3-3b | the data workflow, clip id → model inputs, stage by stage ([10](10_DATA_WORKFLOW.md)) | 1–2 min |
| **3-4** | **the call unrolled, bit-exact** — the one to step through | 30 s |
| 3-5a / 3-5b | action space round trips / flow-matching steps | 10 s / 1 min |
| 3-5c … 3-5f | forced reasoning: does the trajectory follow the text? ([09](09_REASONING_ACTION_STUDY.md)) | 1 min each |
| 3-5g | where the expert's attention goes ([08](08_EXPERT_ATTENTION.md)) | 1 min |
| 3-5h | how much the reasoning helps: own CoT vs `<NOMINAL>` vs nothing on 21 clips ([09 §7](09_REASONING_ACTION_STUDY.md); needs 3-7a first) | 3 min |
| 3-6a / 3-6b | route ablation / freeform VQA | 1 min / 30 s |
| 3-7a / 3-7b | 21-clip eval with the right / a wrong route | 2 min (first run ~25 min: streaming) |
| 3-7c | paired comparison with Alpamayo 1.5 (run configs **29** and **29b** in `~/Desktop/alpamayo1.5` first, ~1 h each) | 10 s |
| 3-8 | latency and memory: eager vs accelerated vs FA2 | 5 min |
| 3-10 / 3-10b | RL in miniature: imitation vs reward training / the math of one update, CPU ([11](11_RL_EXPLAINED.md)) | 10 s / 1 s |
| 2a | the upstream notebook, executed headless on local weights | 3 min |
| 4a / 4b / 4c | demo videos: drive-along (6 cameras + BEV + live reasoning), route split-screen, flow-matching animation → [FIGURES.md](FIGURES.md) | 13 / 10 / 2 min |

---

## The study plan

Nine sessions, about 9 hours in total. Each has a goal, what to run, what to read, where to
break, and a question to answer before moving on. Answers are at the bottom; try first.

### Session 1 — Make it run and look at it (45 min)

**Goal:** a trajectory and a sentence you can trust, from the command line and as a picture.

- Run **1a**, then **1b**. Open `outputs/local-model/*/output.json` and `prompt.txt`.
- Run **3-1** → [`outputs/demo_cameras.png`](outputs/demo_cameras.png), [`outputs/demo_bev.png`](outputs/demo_bev.png).
- Read [`inference.py`](../src/alpamayo2_edge/inference.py) top to bottom (353 lines):
  `main → load_physical_aiavdataset → load_model → predict → run_data`.

> **Q1.** `output.json` reports `elapsed_seconds ≈ 3.1 s`, but the model call is 0.93 s.
> Where did the other two seconds go?

### Session 2 — Where the 2.65 B parameters live (45 min)

**Goal:** draw the model from memory, with sizes.

- Run **3-2**; read [01 §0b, §5, §7](01_MODEL_STRUCTURE.md) next to its output.
- Read [`checkpoint.py`](../src/alpamayo2_edge/checkpoint.py) (strict loading),
  [`config.py`](../src/alpamayo2_edge/config.py) (`normalize_checkpoint_config`, the
  `_TARGETS` allow-list for Hydra), [`models/expert.py`](../src/alpamayo2_edge/models/expert.py).

> **Q2.** The expert is 512 wide but its `q_proj` is `[2048, 512]`. Why is that the
> minimum it can be, and what does the constructor check to make it safe?
>
> **Q3.** `accel_std` is 0.6810 in `config.json` but 0.6797 in the loaded model. Why, and
> what does it cost?

### Session 3 — From pixels and poses to 3 261 tokens (60 min)

**Goal:** be able to predict the prompt length of an input before running it.

- Run **3-3**; read [01 §3–§4](01_MODEL_STRUCTURE.md).
- Read [`load_physical_aiavdataset.py`](../src/alpamayo2_edge/load_physical_aiavdataset.py),
  [`sample.py`](../src/alpamayo2_edge/sample.py) (validation and camera-major sorting),
  [`chat_template.py`](../src/alpamayo2_edge/chat_template.py),
  [`helper.py`](../src/alpamayo2_edge/helper.py) (`_tokenize_messages`: resize in uint8, then
  normalize).
- Breakpoint in `_tokenize_messages` after `smart_resize`; inspect `height, width` per camera.

> **Q4.** Why does a side camera frame cost 91 tokens and a front frame 180?
>
> **Q5.** The prompt is 3 261 tokens long, yet its last token sits at RoPE position 704. Why?

### Session 4 — The call, unrolled (90 min) ★ the central session

**Goal:** know every tensor between the prompt and the 64 poses.

- Run **3-4** and step through it; read [03](03_INFERENCE_WALKTHROUGH.md) alongside.
- Then read the real thing:
  [`sample_trajectories_from_data`](../src/alpamayo2_edge/models/alpamayo2_edge.py),
  `_generate_with_shared_prefill`, `_sample_actions`, and
  [`expert_utils.py`](../src/alpamayo2_edge/models/expert_utils.py).
- Breakpoints: the decode loop ([7]), the Euler loop ([8]); watch `cache.get_seq_length()`
  before and after `crop`.

> **Q6.** Why does generation run one step *past* `<|traj_future_start|>`?
>
> **Q7.** Why is `cache.crop(prefix_length)` called after every Euler step, and why is reusing
> the cache 10 times valid at all?
>
> **Q8.** With `num_traj_sets=6`, which stages run once and which run six times?

### Session 5 — What the expert actually predicts (60 min)

**Goal:** explain the action space and the flow to someone else.

- Run **3-5a** and **3-5b**; read [04](04_ACTION_SPACE_AND_FLOW.md).
- Read [`unicycle_accel_curvature.py`](../src/alpamayo2_edge/action_space/unicycle_accel_curvature.py)
  (`action_to_traj` first, `traj_to_action` second),
  [`unicycle_accel_curvature_course.py`](../src/alpamayo2_edge/action_space/unicycle_accel_curvature_course.py),
  [`flow_matching.py`](../src/alpamayo2_edge/diffusion/flow_matching.py),
  [`action_in_proj.py`](../src/alpamayo2_edge/models/action_in_proj.py).

> **Q9.** What is the difference between "course" and "yaw", and why does integrating along
> yaw cut inside a turn?
>
> **Q10.** One Euler step lands within 0.10 m of ten. What property of the learned flow
> makes that possible, and why does that *not* make the noise irrelevant?

### Session 6 — How the route and the question get in (45 min)

**Goal:** know what the model does with the text you give it.

- Run **3-6a** (four routes, same scene) and **3-6b** (VQA); optionally **1d** with
  `"Turn left in 30m"`.
- Read [`text_tasks.py`](../src/alpamayo2_edge/text_tasks.py) and `build_vqa_conversation`.

> **Q11.** The route is text in the prompt. By which path can it change the trajectory?
>
> **Q12.** Asked "How should the ego vehicle drive?" without a route, the model says it will
> turn left "to follow our route". The logged car turned right. What does that tell you?

### Session 7 — Measuring it (60 min)

**Goal:** a number you can defend, and knowing what it does not mean.

- Run **3-7a** and **3-7b**; look at [`outputs/eval_oracle_k6.png`](outputs/eval_oracle_k6.png)
  and [`outputs/eval_fixed_k6.png`](outputs/eval_fixed_k6.png).
- Then **3-7c** and read [07](07_HEAD_TO_HEAD_VS_A15.md): the same clips through Alpamayo 1.5
  (already run; its config **29** lives in the 1.5 repo).
- Read [`10_eval_multiclip.py`](scripts/10_eval_multiclip.py) and `derive_nav_text` in
  [`_common.py`](scripts/_common.py).

> **Q13.** Mean ADE@6.4 s is 1.50 m, median 0.67 m. Which do you report, and why are they
> so different?
>
> **Q14.** The route was derived from the logged future. Is that cheating?
>
> **Q14b.** With the same route, Edge's median ADE@6.4 s is 0.67 m and Alpamayo 1.5's is
> 1.02 m. Is Edge the better planner?

### Session 8 — Speed (60 min)

**Goal:** know where the milliseconds go and what each optimization buys.

- Run **3-8** and **1c**; read [05](05_ACCELERATION.md) and
  [`docs/acceleration.md`](../docs/acceleration.md).
- Read [`accel/__init__.py`](../src/alpamayo2_edge/accel/__init__.py),
  [`accel/graphs.py`](../src/alpamayo2_edge/accel/graphs.py),
  [`accel/expert.py`](../src/alpamayo2_edge/accel/expert.py); skim
  [`accel/decode.py`](../src/alpamayo2_edge/accel/decode.py) (`FlashKVCache`, `DecodeStep`,
  `GraphSampler`).

> **Q15.** Reading all decoder weights once takes ~4 ms on a 3090, yet an eager decode step
> takes 18.5 ms. Where does the rest go?
>
> **Q16.** Sampling happens inside a CUDA graph. Why does a replay draw the same token as
> eager execution would?

### Session 9 — The tests as a specification (optional, 45 min)

Run **0c**. Read [`tests/test_expert_attention.py`](../tests/test_expert_attention.py)
(the mask and position contract), [`tests/test_course_action_space.py`](../tests/test_course_action_space.py)
(a float64 reference round trip), [`tests/test_accel.py`](../tests/test_accel.py) (graphs
vs eager). 158 tests pass on this machine.

---

## Numbers worth memorising

| | |
|---|---|
| parameters | **2.651 B** = VLM 2.452 B (ViT 413 M, projector 77 M, decoder 1.409 B, embed + lm_head 2 × 277 M) + expert **199 M** |
| weights | 5.30 GB bf16 |
| inputs | 6 cameras, **22 frames**; 20 poses; one route sentence |
| prompt | **3 261** tokens = 2 892 image + 57 history + 312 text |
| decoder | 28 × 2048, GQA 16 / 8 × 128, relu² MLP 9216, interleaved M-RoPE θ = 1e8 |
| expert | 28 × 512, attention in the VLM's 16 / 8 × 128 geometry, non-causal |
| action | 64 × (accel, curvature), 10 Euler steps → **6.4 s** at 10 Hz |
| KV cache | 112 KiB / token, 359 MiB on the example |
| model call, 3090 | eager **934 ms**, `--accel tuned` **420 ms**; CPU preprocessing 1.4 s |
| peak memory | 6.2 GB (1 CoT), 9.2 GB (6 CoTs) |
| CoT | ~20–35 tokens, or `<NOMINAL>` (13 / 21 random clips) |
| eval, 21 clips, oracle route | ADE@6.4 s median **0.67 m** (candidate 1), minADE@6 median **0.39 m** |
| vs Alpamayo 1.5, same clips | statistically indistinguishable (Edge better on 12/21, p = 0.89; vs A1.5 + CFG 11/21, p = 0.92); a wrong route costs Edge +1.43 m median, A1.5 −0.02 m |

## Things that will bite you

1. **The model card links `ga-release`; use `main`.** That tag is the Qwen3-VL era and
   rejects this Cosmos3-Edge checkpoint.
2. **`uv sync` removes FlashAttention** (it is not in `uv.lock`). Use the task
   "env: uv sync" (adds `--inexact`) or reinstall the wheel afterwards.
3. **The notebook downloads the model again** unless `MODEL_ID` points at
   `.work/Alpamayo2-Edge`. Config 2a does that for you without editing the notebook.
4. **The dataset has no route.** Every metric depends on how you invent `nav_text`; a wrong
   route quadruples median ADE (0.67 → 2.63 m).
5. **`elapsed_seconds` in `output.json` includes CPU preprocessing** (1.4 s) and first-call
   warm-up. Time the model call yourself.
6. **The third return value is a placeholder** (`zeros`); do not rank candidates with it.
7. **Action-space constants are bf16** after loading (3 cm at 6.4 s on the example).
8. **Use `CUDA_VISIBLE_DEVICES=0`.** `cuda:1` drives the desktop. All launch configs set it.
9. **`tmp/sample_cache` holds 137 MB per clip** (22 raw 1080p frames). Delete it freely; it
   is rebuilt by streaming.

## Open questions (the next experiments)

- ~~A fair head-to-head with Alpamayo 1.5~~ — done on 21 clips, see [07](07_HEAD_TO_HEAD_VS_A15.md).
  Next: a few hundred clips (≈400–1 500 to resolve 0.1 m) and a route source that is not
  derived from the future.
- **Steps vs accuracy over many clips**: does 2–3 Euler steps hold across the 21 clips?
  (`--diffusion-steps` in the CLI; `inference_step` in `predict`.)
- **A perception probe on Edge**: your 1.5 head read 720 image tokens from decoder layer 24
  of 36. Edge's equivalent is ~layer 18 of 28, 2 892 image tokens, 2048-d, and it sees the
  rear.
- **Is `<NOMINAL>` safe?** On which clips does the model skip reasoning, and is its ADE
  there better or worse than when it reasons?
- **Route phrasing**: the training phrasing is `Turn right in 30m`; how does the model react
  to street names, distances, `"Take the second exit"`?

---

## Answers

<details><summary>Q1</summary>
<code>prepare_model_inputs</code> on the CPU (≈1.4 s: bicubic resize of 22 × 1080p frames in
uint8, the processor, tokenization), first-call CUDA/cuBLAS initialization, and
<code>fuse_traj_tokens</code> / packing. The model call itself is 0.93 s eager.
</details>

<details><summary>Q2</summary>
The expert's queries must match the VLM's cached keys: 16 query heads × 128 = 2048 for q,
8 KV heads × 128 = 1024 for its own k and v (which are appended to the VLM's cache). The
constructor requires the same <code>model_type</code>, <code>num_hidden_layers</code>,
<code>num_key_value_heads</code> and <code>head_dim</code> in the VLM and expert configs.
</details>

<details><summary>Q3</summary>
The constants are non-persistent buffers rebuilt from config by <code>_init_weights</code> and
then cast to the model dtype (bf16). The same normalized action decodes to an endpoint 3.1 cm
away from the fp32-constant result on the example.
</details>

<details><summary>Q4</summary>
Per-camera pixel budgets: front/tele frames are resized to 320×576 (163 840–196 608 px),
side/rear to 224×416 (81 920–98 304 px). Patch 16 → 20×36 or 14×26 patches; the 2×2 merger
gives 10×18 = 180 or 7×13 = 91 tokens.
</details>

<details><summary>Q5</summary>
M-RoPE: an image of h×w tokens advances the position by max(h, w), not h·w. 22 images
contribute 2 892 tokens but only 336 positions (10 frames × 18 + 12 frames × 13), so
<code>rope_delta</code> = 336 − 2 892 = −2556.
</details>

<details><summary>Q6</summary>
<code>generate</code> caches the token it feeds into the model, not the one it samples. The
anchor must be in the cache (the expert's tokens are positioned right after it), so one more
step is needed to feed it; the token sampled in that step is discarded.
</details>

<details><summary>Q7</summary>
<code>DynamicCache.update</code> appends the expert's 64 K/V; cropping restores the prompt +
CoT prefix so every step conditions only on that prefix. Reuse is valid because the prefix
never depends on the action tokens: the expert reads the VLM's K/V but never writes into the
VLM's part, and each step's action tokens are dropped.
</details>

<details><summary>Q8</summary>
Once: CPU preprocessing, the vision tower, the prompt prefill (the cache is then replicated
per row). Six times (batched): the CoT decode and the 10 expert steps, each row with its own
CoT and its own noise.
</details>

<details><summary>Q9</summary>
Yaw is where the car's body points; course is the direction the rear-axle centre actually
moves. They differ by the sideslip angle, which grows with lateral acceleration. A model that
integrates speed along yaw systematically places the path inside the driven one in turns;
the course space round-trips the example's right turn with FDE 0.24 cm vs 1.84 cm.
</details>

<details><summary>Q10</summary>
The learned velocity field is nearly constant along each path (rectified flow): the decoded
plan moves linearly from noise to its final value, so one Euler step already lands close.
But the end point depends on the starting noise: with the CoT fixed, six noise seeds spread
the 6.4 s endpoint by 1.7 m on the example.
</details>

<details><summary>Q11</summary>
Only through attention: the route tokens are in the prompt, so they shape the VLM's hidden
states and KV cache (prompt and CoT, which often restates the route); the expert attends
those caches. There is no route input to the expert itself.
</details>

<details><summary>Q12</summary>
The VQA answers keep the driving-CoT format ("to follow our route") — post-training bias —
and without a route the model guesses one from the scene (the lead car turning left). The
route is not inferable from images alone; that is why the driving prompt requires it.
</details>

<details><summary>Q13</summary>
Report both, and the median first: a few failures (two roundabouts, a 30 m/s clip) dominate
the mean. 21 clips is a sanity check, not a benchmark.
</details>

<details><summary>Q14</summary>
It leaks the direction (left / right / straight) but not the geometry, speed or timing — the
same information a navigation system provides in a car. It is fair for a route-conditioned
planner, but the numbers are not comparable to a model evaluated without a route.
</details>

<details><summary>Q14b</summary>
Not on this evidence. Paired per clip, Edge is better on 12 of 21 clips with a median
difference of −0.06 m and Wilcoxon p = 0.89: the medians of two columns hide that the
per-clip differences go both ways. What <i>is</i> clear is that Edge matches a 4× larger model
and that it follows the route text far more readily than Alpamayo 1.5, which turns only where
the scene makes the manoeuvre plausible (see 07).
</details>

<details><summary>Q15</summary>
Launch overhead: eager PyTorch issues hundreds of small kernels per token from Python, plus
HF <code>generate</code> bookkeeping and cache concatenation. One CUDA-graph replay per token
removes almost all of it (420 ms vs 934 ms for the whole call).
</details>

<details><summary>Q16</summary>
CUDA RNG is Philox counter-based: a draw depends on (seed, offset). Graph capture records the
draws with offsets relative to the generator state at replay time, and <code>GraphedCall</code>
restores the RNG state after warm-up and capture, so a replay consumes the same offsets eager
code would.
</details>
