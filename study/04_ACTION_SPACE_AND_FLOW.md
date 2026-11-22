# 04 — The action space and the flow-matching expert

What the expert actually produces is not a trajectory but **64 pairs of normalized
(acceleration, curvature)**, one per 0.1 s. A unicycle model integrates them into poses.
Launch configs **3-5a** ([`03_action_space.py`](scripts/03_action_space.py)) and **3-5b**
([`05_flow_matching.py`](scripts/05_flow_matching.py)) produce every number on this page.

![action space](outputs/action_space.png)

## 1. Why controls instead of waypoints

| | predicting `(x, y, heading)` (e.g. Qwen-Drive) | predicting `(a, κ)` + integration (Alpamayo) |
|---|---|---|
| feasibility | any shape, including physically impossible ones | bounded `a ∈ [−9.8, 9.8]` m/s², `κ ∈ [−0.33, 0.33]` 1/m → drivable by construction |
| smoothness | must be learned | integration of piecewise-constant controls is smooth |
| conditioning on the present | the model must reproduce the current speed | `v0` and heading come from the history; the model predicts only *changes* |
| failure mode | jitter, kinks | an error in `a` early on compounds into position error late |

The normalization constants come from training data statistics:
`accel ~ 0.029 ± 0.681 m/s²`, `curvature ~ 0.00027 ± 0.0261 1/m`. A normalized curvature
of −6.0 (the example's right turn) is −0.157 1/m, a 6.4 m radius.

## 2. Encoding (training labels) and decoding (inference)

**Decode** (`action_to_traj`, used at inference):

```
a_k = x[k,0]·σ_a + μ_a              κ_k = x[k,1]·σ_κ + μ_κ
v_0 = estimated from the 20 history poses (regularized least squares, dxy_theta_to_v_without_v0)
v_{k+1} = v_k + a_k·dt
θ_{k+1} = θ_k + κ_k·v_k·dt + κ_k·a_k·dt²/2          θ_0 = t0 course (course space) or 0 (yaw space)
x, y    = trapezoid integration of v·(cos θ, sin θ)
```

**Encode** (`traj_to_action`, used to make training targets and discrete tokens):
smooth the heading targets (`theta_smooth`, Tikhonov), solve for `v` from the position
deltas, `a` from `v` (2nd-order smoothness), `κ = Δθ / s` (2nd-order smoothness), zero `κ`
at very low speed, clamp to bounds, normalize.

## 3. The GA change: course heading instead of body yaw

The released GA checkpoint uses `UnicycleAccelCurvatureCourseActionSpace` (the earlier EA
checkpoint used the yaw space; commit `20d1134`). The rig origin is the rear-axle centre,
which travels along its **course**; the body **yaw** differs from it by the rear-axle
sideslip (~0.4 mrad + ~2.7 mrad per m/s² of lateral acceleration). Integrating along yaw
puts decoded paths *inside* the driven one in turns.

Changes relative to the yaw space ([`unicycle_accel_curvature_course.py`](../src/alpamayo2_edge/action_space/unicycle_accel_curvature_course.py)):

- heading targets = yaw + chord-direction offset (central chords; backward chord at the end)
- both encode and decode start from `θ0`, the course at t0 estimated from the last history chord
- the offset fades in between 0.5 and 1.0 m/s and is dropped if > 50 mrad (pose noise)
- curvature zeroed only below 0.1 m/s (yaw space: 0.6 m/s), so pull-away turns keep their heading change
- decoded rotations = course − θ0 + `slip_gradient`·Δ(lateral accel), so yaw is 0 at t0

**Measured round trip on the example's logged right turn** (encode → decode, float64):

| action space | ADE | FDE | max error | max yaw error |
|---|---|---|---|---|
| **course (GA, this checkpoint)** | **0.12 cm** | **0.24 cm** | 0.50 cm | 0.23° |
| yaw (EA) | 1.17 cm | 1.84 cm | 2.00 cm | 0.12° |

The course space represents the driven path ~8× more faithfully; the yaw space is slightly
better on heading, which is expected because it fits yaw directly. On this clip
`θ0 = −1.0 mrad` and `v0 = 7.62 m/s`.

The **controls the expert has to produce** for this turn: acceleration −1.30 … +0.19 m/s²
(braking into the turn), curvature 0 … −0.157 1/m; speed 7.62 → 3.38 → 3.43 m/s.

## 4. The discrete trajectory tokens (training-time only)

`future_traj_tokenizer_cfg` = `DiscreteTrajectoryTokenizer` over the same course space,
3000 bins on normalized `[−10, 10]` (0.0067 σ per bin), 64 × 2 = **128 tokens**
(`<i1000>`…`<i3999>`). Round trip on the example: **ADE 0.13 cm, FDE 0.15 cm**. First codes:
`[1482, 1500, 1430, 1499, …]` (interleaved accel, curvature).

These tokens are in the vocabulary and the config has a `future_traj` loss weight, so the
VLM was presumably also trained to *emit* them; at inference
`MaskDiscreteTrajectoryLogitsProcessor` forbids them and the continuous expert produces
the trajectory instead.

## 5. The bf16 constants

The four normalization constants are rebuilt from config and then cast with the model:

| constant | config (fp64) | held by `load_model` (bf16) |
|---|---|---|
| `accel_mean` | 0.029027 | 0.029053 |
| `accel_std` | 0.681043 | **0.679688** (−0.20 %) |
| `curvature_mean` | 0.000269 | 0.000269 |
| `curvature_std` | 0.026148 | 0.026123 (−0.10 %) |

Same normalized action, bf16 vs fp32 constants: endpoint moves **3.1 cm** at 6.4 s on the
example. Small against a 0.3–0.7 m ADE, but it is a systematic scale error, and it is the
kind of thing that changes when you port the action space to another runtime.

## 6. Flow matching

[`diffusion/flow_matching.py`](../src/alpamayo2_edge/diffusion/flow_matching.py):

```
training:  t ~ 0.999·(1 − Beta(1.5, 1))      noisy = t·x + (1−t)·ε      loss = ‖v_θ(noisy, t) − (x − ε)‖²
inference: x ← ε ~ N(0, I);  for t = 0, 0.1, …, 0.9:  x ← x + 0.1 · v_θ(x, t)
```

Beta(1.5, 1) has more mass near 1, so after the flip training sees more samples near
**t = 0 (the noisy end)**. There is no classifier-free guidance at inference
(`use_classifier_free_guidance=false`; `inference_guidance_weight: 3.0` is a leftover).

### The path is almost a straight line

The 10 intermediate `x_t`, each decoded through the action space ([`outputs/flow_matching.png`](outputs/flow_matching.png)):

| t | rms(x_t) | endpoint (x, y) m | mean distance to final plan |
|---|---|---|---|
| 0.0 (noise) | 1.10 | (49.3, 0.0) | 5.50 m |
| 0.2 | 0.94 | (46.1, −2.6) | 4.35 m |
| 0.5 | 1.01 | (41.0, −4.7) | 2.68 m |
| 0.8 | 1.38 | (36.2, −5.0) | 1.06 m |
| 1.0 | 1.71 | (33.3, −4.7) | 0 |

The distance to the final plan falls **linearly** in t — the learned velocity field is
nearly constant along the path, which is what rectified flow aims for. Consequence:

### Step-count sweep (same CoT, same initial noise; only the integrator changes)

| steps | ADE vs 10-step | ADE vs logged | FDE vs logged |
|---|---|---|---|
| 1 | 0.096 m | 0.544 m | 2.33 m |
| 2 | 0.084 m | 0.571 m | 2.27 m |
| 3 | 0.051 m | 0.592 m | 2.30 m |
| 5 | 0.024 m | 0.615 m | 2.33 m |
| **10 (default)** | 0 | 0.625 m | 2.32 m |
| 20 | 0.012 m | 0.629 m | 2.32 m |
| 50 | 0.018 m | 0.628 m | 2.30 m |

A single Euler step lands within 10 cm (ADE) of the 10-step result, far inside the
model's own error against the logged path. Single sample, single clip — not a benchmark,
but a strong hint that the expert could run in 2–3 steps on a latency budget. (On this
sample the 1-step plan is even closer to the logged path; do not read anything into that.)

### Where candidate diversity comes from

[`06_demo_bev.py`](scripts/06_demo_bev.py) §3, six candidates each:

| source of randomness | mean endpoint distance to the centroid |
|---|---|
| (a) six independent CoTs, each with its own noise (what `num_traj_sets=6` gives) | 1.34 m |
| (b) **one fixed CoT**, six diffusion noise seeds | **1.70 m** |

Even with the reasoning text fixed, the noise moves the 6.4 s endpoint by metres: the
expert is a genuinely multimodal sampler, not a deterministic decoder of the CoT. (The
CoTs in (a) mostly say the same thing — "decelerate and turn right … yield sign" — so
they add little spread here.) Before you use the CoT as an *explanation* of a particular
trajectory, keep this in mind: several different trajectories are consistent with one CoT.

![demo](outputs/demo_bev.png)

### Does the trajectory follow the reasoning? (configs 3-5c … 3-5f)

Moved to its own study: [09 — reasoning-tampering study](09_REASONING_ACTION_STUDY.md).
In short: with the reasoning forced by hand and images, route and noise fixed, the route
sentence decides the direction (the reasoning only when the route names none), an explicit
"stop" in the reasoning overrides the speed, and "slow down", "yield" or hazard descriptions
have no measurable effect.

### Where does the expert look? (config 3-5g)

Full write-up, including exactly how this is measured: [08 — expert attention](08_EXPERT_ATTENTION.md).

The 64 action tokens attend to the **whole** cache — all 3,261 prompt tokens, the 22
reasoning tokens and the anchor — plus each other (mask shape `[1, 1, 64, 3348]`, nothing
hidden for a single candidate). [`14_expert_attention.py`](scripts/14_expert_attention.py)
loads the model with eager attention, hooks the 28 expert layers during the 10 Euler steps,
and sums the attention by input segment (mean over heads, queries and steps):

| segment | tokens | share of keys | attention mass | per token vs uniform |
|---|---|---|---|---|
| first 4 tokens `\n<|im_start|>system\n` (sinks) | 4 | 0.1 % | ~28 % | — |
| other prompt text (system prompt, camera names, time labels, markup, task) | 299 | 8.9 % | 46.6 % (incl. the sinks above; top 10 tokens hold 65 %) | 5.2× |
| images, front + telephoto | 1,800 | 53.8 % | 13.1 % | 0.2–0.3× |
| images, the 4 side/rear cameras | 1,092 | 32.6 % | 3.2 % | 0.1× |
| ego history | 57 | 1.7 % | 3.2 % | 1.9× |
| route sentence | 12 | 0.4 % | 2.1 % | 5.8× |
| **reasoning (CoT)** | **22** | **0.7 %** | **4.5 %** | **6.8×** |
| anchor | 1 | 0.03 % | 0.9 % | 30× |
| own 64 action tokens | 64 | 1.9 % | 19.2 % | 10× |

By depth: layers 0–8 park ~72 % on prompt text, mostly sinks; layers 9–18 read the images
(front and telephoto ~19 %) and the reasoning (5.6 %); layers 19–27 attend mostly to the
other action tokens (39 %), with ego history (5.2 %), reasoning (5.7 %) and route (3.0 %).

Attention mass is not influence (sink tokens absorb mass but contribute little), but it
does show the reasoning is read — per token, more than anything except sinks, the anchor
and the action tokens themselves — while the scene still supplies most of the content.

![expert attention](outputs/expert_attention.png)

![cot intervention](outputs/cot_intervention.png)

Next: [05 — acceleration](05_ACCELERATION.md).
