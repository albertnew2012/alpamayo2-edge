# Alpamayo 2 Edge vs Alpamayo 1.5 vs Qwen-Drive 1.0

> All three are "VLM + flow-matching action expert" driving VLAs, and all three run on your
> 3090. Edge numbers here are measured in this repo; Alpamayo 1.5 and Qwen-Drive numbers come
> from your own study notes in
> [`alpamayo1.5/study`](/home/albert/Desktop/alpamayo1.5/study/00_START_HERE.md) and
> [`Qwen-Drive-1.0/study`](/home/albert/Desktop/Qwen-Drive-1.0/study/02_VS_ALPAMAYO_1_5.md)
> (read for reference, nothing from those repos was executed here).

---

## 0. The one-paragraph answer

**Alpamayo 2 Edge is Alpamayo 1.5's recipe, shrunk ~4× and re-plumbed for a car.** The core
idea is unchanged: a VLM writes its reasoning, stops at `<|traj_future_start|>`, and a
separate expert transformer denoises 64 (accel, curvature) controls while attending the
VLM's per-layer KV cache. What changed is everything around it: a 2.45 B Cosmos3-Edge
backbone instead of an 8.8 B Cosmos-Reason2 (Qwen3-VL) one, a 199 M expert instead of
2.28 B, **six cameras including the rear** with 2 s of front-camera history, the route as
plain prompt text, a course-heading action space, *adaptive* reasoning (`<NOMINAL>` when
nothing needs explaining), and a complete compile/CUDA-graph path for Thor. **Qwen-Drive is a
different philosophy:** keep a strong general VLM intact and hang heads off it (an official
BEV perception head; a planner conditioned by adaLN on nav one-hot + ego status, reading 8
KV caches, predicting `(x, y, heading)` waypoints). Edge and 1.5 are *reasoning-first*
planners; Qwen-Drive is a *VLM-first platform* where reasoning is one of three modes.

**Measured head to head** ([07](07_HEAD_TO_HEAD_VS_A15.md)): on the same 21 clips with the same
route, Edge and Alpamayo 1.5 are statistically indistinguishable (Edge better on 12/21,
p = 0.89) — at a quarter of the size. The clear difference is the route: a wrong one moves
Edge's plans by metres (+1.43 m median ADE, right turns on 10/21 clips) and Alpamayo 1.5's by
centimetres (−0.02 m), with or without its classifier-free guidance.

---

## 1. Side by side

| | **Alpamayo 2 Edge** (this repo) | **Alpamayo 1.5** | **Qwen-Drive 1.0** |
|---|---|---|---|
| total params | **2.65 B** | 11.08 B | 5.70 B (incl. 0.13 B perception) |
| bf16 weights | **5.30 GB** | 22.16 GB | 11.66 GB |
| on your 3090 | fits; **9.2 GB peak** with 6 candidates | needs CPU offload (21.2 GB peak) | fits, 12 GB spare |
| VLM backbone | Cosmos3-Edge: SigLIP2 ViT 27×1152 + Nemotron-style decoder **28 × 2048**, relu² MLP, GQA 16/8 | Cosmos-Reason2-8B (Qwen3-VL): ViT 27 blocks + decoder **36 × 4096**, GQA 32/8 | Qwen3.5-4B, **hybrid** 24 Gated-DeltaNet + 8 full-attention layers |
| vocab | 135,168 (131k base + 4000 traj bins + 30 specials), untied | 155,697 (+ 4000 bins), untied | 248,320, tied |
| cameras | **6: front wide, front tele, cross L/R, rear L/R** | 4: front wide, front tele, cross L/R | 3 front views (planning); 6–8 for perception |
| frames | 5 front/tele (−2…0 s), 3 others (−1…0 s) = **22** | 4 per camera at 0.1 s (−0.3…0 s) = 16 | 4 timesteps × 3 = 12 |
| image tokens | **2,892** (180 front/tele, 91 side/rear) | 2,880 (180 each) | 3,054 |
| time encoding | text labels `<t=-0.5s>` from actual capture times | frame numbers | view tags (`<FRONT>` …); 3 older timesteps at low resolution (156 tok), current at high (550 tok) |
| ego history | 20 poses (1.9 s) → **57** delta tokens | 16 poses (1.5 s) → 48 delta tokens | ego-status vector into the planner (adaLN) |
| route / nav | **plain prompt text** `Navigation input is: …` | `<|route_start|>…<|route_end|>` tokens; optional CFG (two denoiser passes) | nav one-hot → MLP → adaLN |
| reasoning | always generated, but **adaptive**: `<NOMINAL>` on 13/21 eval clips | always (chain of causation) | optional (1 of 3 planning modes) |
| expert | **28 × 512**, q/k/v widened to the VLM's 16×128 / 8×128, **198.8 M** | 36 × 2048, **2.28 B** | 32 × 1024, 1.04 B, 4 expert layers per VLM cache |
| expert reads | all 28 VLM layers' KV (layer i ↔ layer i) | all 36 VLM layers' KV | only the 8 full-attention layers' KV |
| expert attention backend | SDPA (4-D mask) **or FA2** (2-D padding mask) | forced SDPA | — |
| flow matching | velocity, 10 Euler steps, no CFG | velocity, 10 Euler steps (+ CFG variant for nav) | clean-endpoint (x₁), 10 Euler steps |
| action space | (accel, κ) × 64 → unicycle, **course heading** (sideslip-aware) | (accel, κ) × 64 → unicycle, body-yaw heading | (x, y, heading) × 50 |
| horizon | **6.4 s** @ 10 Hz | 6.4 s @ 10 Hz | 5.0 s @ 10 Hz |
| multi-candidate | **shared prefill**, cache replicated per CoT | `num_return_sequences` (vision re-run per sample) | per mode |
| acceleration in repo | compile + CUDA graphs for **prefill, decode and expert**; FA2 / FA3 | CUDA graph of the diffusion expert only | none upstream (your fork adds ONNX export) |
| transformers | 5.17 | 4.57 | 5.14 |
| perception output | none | none (your 28 M bolt-on head) | official BEV head: 3D det + occupancy + map |
| training code | none here (README: fine-tuning via NVIDIA recipes) | `NVlabs/alpamayo-recipes` | none released |
| target | **in-vehicle, NVIDIA DRIVE AGX Thor** | research | research / platform |

## 2. What Edge inherited unchanged from Alpamayo 1.5

You already know these from the 1.5 study; they carry over 1:1 and the code is visibly the
same lineage (same module names, SPDX headers, Hydra `_target_`s):

- the **two-transformer split with a frozen cache between them**: the expert has no
  embedding table and no scene input except attention to the VLM cache
- **`StopAfterEOS`** one step past `<|traj_future_start|>`, so the anchor is cached
- **`cache.crop(prefix)` after every Euler step**: every step sees only prompt + CoT
- the **non-causal** expert with a 4-D mask hiding padding and other rows' CoT tails
- **`DeltaTrajectoryTokenizer`** for history, injected as vocabulary via `masked_scatter`
- 4000 `<i*>` trajectory bins in the vocabulary, **masked to −∞ during CoT sampling**
- `PerWaypointActionInProjV2` (Fourier features per channel + time → MLP), `Linear` out
- `FlowMatching` Euler sampler and the unicycle accel/curvature integration
- sampling defaults: top-p 0.98, temperature 0.6, 256 max new tokens

## 3. What Edge changed, and why it matters

| change | effect you can measure here |
|---|---|
| **4× smaller VLM, 11× smaller expert** | 5.3 GB weights; 18.5 ms/token and ~19 ms/Euler step eager on a 3090; 0.93 s per call eager (0.42 s accelerated) vs 5.7 s for 1.5 with offload |
| **expert 512 wide, attention 2048 wide** | 7.08 M per expert layer, of which 44 % are q/k/v/o — the price of reading full-width KV heads without an adapter |
| **6 cameras incl. rear, per-camera pixel budget** | 360° coverage for the same ~2,900 image tokens: side/rear frames at half resolution and 3 frames instead of 5 |
| **2 s of front video instead of 0.3 s** | motion of other agents over 2 s is visible to the VLM; time is given as text from actual capture timestamps |
| **20-pose history (57 tokens)** | 0.4 s more ego history; `pad_origin_at_beginning=False` (history ends at the origin) |
| **route as text, no CFG** | one expert pass per step instead of two; the route steers through the CoT and the cache. Measured: a wrong route ("Turn right in 30m" on every clip) raises median ADE@6.4 s from 0.67 m to 2.63 m, and the model turns right wherever the scene offers a right turn (even at 30 m/s). Alpamayo 1.5 with route tokens in the same test: 1.02 → 0.84 m, CoT unchanged on 20/21 clips |
| **course-heading action space (GA)** | encode→decode of a logged right turn: FDE 0.24 cm vs 1.84 cm with the yaw space |
| **adaptive reasoning** | `<NOMINAL>` (5 ordinary text tokens) on 13 of 21 random clips; those calls take 0.95 s vs 1.27 s median with 6 candidates (fewer decode steps) |
| **shared prefill** | vision + prefill cost is independent of the number of candidates; 6 candidates peak at 9.2 GB |
| **FA2 for the expert** | 1.5 had to force SDPA for the expert; Edge reduces the mask to a padding mask and runs FA2 non-causally |
| **`accel/` package** | whole-pipeline CUDA graphs over a static flash-layout KV cache (see [05](05_ACCELERATION.md)) |
| **Cosmos3-Edge + transformers 5** | `mm_token_type_ids`, interleaved M-RoPE with θ=1e8, an empty `<think></think>` before the answer |

### Repository history (why tags and the model card disagree)

`ea-release` (EA checkpoint, body-yaw heading) → `ga-release` (GA checkpoint, course
heading) were both **Qwen3-VL** based (`vlm_class = Qwen3VLForConditionalGeneration`). The
latest commit `02f55a7` (Nov 9) moved to **Cosmos3-Edge** and pinned HF revision
`f8951d09`. The HF model card still links `tree/ga-release`, but that code rejects this
checkpoint's `vlm_class`. Use `main`. A remote branch `toyin/streaming-prefill-simulation`
(Qwen3-VL era) simulates a 10 Hz streaming runtime that reuses older frames' K/V; it is not
merged and was not used here.

## 4. Edge vs Qwen-Drive: two different bets

| question | Edge | Qwen-Drive |
|---|---|---|
| Where does the scene enter the planner? | only through the VLM's KV cache (all 28 layers) | 8 KV caches **plus** adaLN conditioning on nav + ego status + time |
| Is the VLM still a general VLM? | it answers freeform driving questions (VQA path below), but it was post-trained for driving; general benchmarks are not published | yes by design; the paper shows it keeps general VQA scores |
| Can the plan be infeasible? | no: bounded controls integrated by a unicycle model | in principle yes: waypoints are predicted directly |
| Does reasoning gate the plan? | yes: the expert always reads the CoT (or `<NOMINAL>`) | only in reasoning mode |
| Plan without reasoning? | not an official mode; forcing `<NOMINAL>` works ([09 §7](09_REASONING_ACTION_STUDY.md)) | `DIRECT_PLANNING`: the assistant turn is closed empty, the expert reads the scene cache (with `planner-sft`; `planner-rl` was tuned only with reasoning) |
| How much reasoning helps (open loop) | 21 clips: 1.770 vs 1.776 m mean ADE; 2–5 % on the 8 clips where it reasons | published: PDMS 87.8 → 88.2, ADE@3 s 0.38 → 0.37 m |
| 3D structure | implicit (no perception output) | explicit BEV head (3D boxes, occupancy, map) |
| Horizon | 6.4 s | 5.0 s |
| Cameras for planning | 6 incl. rear | 3 front |
| Deployment story | DRIVE AGX Thor, CUDA graphs in-repo, small | research release |

## 5. Numbers side by side, and how far to trust them

Same 21 clips, t0 = 5.1 s, 6 candidates, same ground truth; Alpamayo 1.5 run in its own
repo with its own code ([07](07_HEAD_TO_HEAD_VS_A15.md)):

| metric (median) | Edge, oracle route | Alpamayo 1.5, oracle route | Alpamayo 1.5, no route | Qwen-Drive (paper) |
|---|---|---|---|---|
| ADE@3 s, candidate 1 | **0.152 m** | 0.196 m | 0.194 m | minADE@3 s 0.34 m (SFT), own protocol |
| minADE@3 s over 6 | 0.101 m | **0.082 m** | 0.082 m | |
| ADE@6.4 s, candidate 1 | **0.671 m** | 1.019 m | 1.029 m | — |
| minADE@6.4 s over 6 | **0.388 m** | 0.494 m | 0.471 m | — |
| paired, ADE@6.4 s cand. 1 | Edge better on 12/21, p = 0.89 | | | |
| ADE@6.4 s with a **wrong** route | 2.626 m | 0.839 m | — | — |
| model call latency, 1 candidate | eager **0.93 s**, `--accel tuned` **0.42 s** (+1.4 s CPU preprocessing) | 5.7 s (offload + FA2) | ~2 s planning |

**The medians are not a ranking.** Paired per clip, the Edge–A1.5 difference goes both
ways (p = 0.89), and 21 clips can only resolve differences of roughly 0.4 m. Qwen-Drive's
number uses a different horizon, subset and candidate count and is not comparable. The
earlier single-clip figure in your 1.5 notes (0.373 m, n = 1, seed 42) is one draw; with 6
candidates on the same clip A1.5's candidate 1 scores 1.03 m and its best 0.22 m.

## 6. Which one for what

| if you want to … | use |
|---|---|
| study the reasoning→action coupling with fast iteration, or deploy | **Edge**: 5 GB, 1–2 s per call, bit-exact walkthrough available, everything graphable |
| the strongest reasoning VLM and the longest-studied checkpoint | Alpamayo 1.5 (your fork already has an unrolled reimplementation and a perception head) |
| explicit 3D perception, general VQA, a planner you can condition directly | Qwen-Drive |
| port your perception-probe idea to a 360° model | **Edge**: the only one of the three whose planning prompt includes the rear cameras; a forward hook on a mid decoder layer over the 2,892 image tokens would mirror your 1.5 head at a quarter of the feature size (2048-d instead of 4096-d) |
