# 08 — Where the flow-matching expert looks: attention by input segment

**Question.** When the expert denoises the trajectory, does it attend only to the reasoning
(CoT) tokens, or to everything in the KV cache? And how is its attention split?

**Answer in one line.** It attends to *everything*: all 3,261 prompt tokens, the 22
reasoning tokens, the anchor and its own 64 action tokens, at every one of its 28 layers.
The reasoning gets ~7× its share of attention per token, but most attention goes to attention
sinks, the action tokens themselves and the scene.

| | |
|---|---|
| script | [`scripts/14_expert_attention.py`](scripts/14_expert_attention.py) |
| debug config | **3-5g. Expert attention: where the flow-matching expert looks** (`.vscode/launch.json`, `justMyCode` + `stopOnEntry`) |
| outputs | [`outputs/expert_attention.png`](outputs/expert_attention.png), [`outputs/expert_attention.json`](outputs/expert_attention.json) |
| sample | README clip `ea7bbd31`, t0 = 4.0 s, route "Turn right in 30m", 1 candidate, seed 43 |
| runtime | ~1 min on the RTX 3090 |

---

## 1. What the expert *can* see (from the code)

[`build_expert_pos_ids_and_attn_mask`](../src/alpamayo2_edge/models/expert_utils.py) builds the
mask the expert's attention uses:

```python
attention_mask = torch.zeros((b_star, 1, n_diffusion_tokens, kv_cache_seq_len + n_diffusion_tokens))
# then -inf only for: cache columns beyond this row's own anchor (other candidates' longer CoTs)
#                     and left-padding of the prompt
```

Zeros mean visible. For one candidate nothing is hidden: each of the 64 action tokens sees
all `3,284` cached tokens (prompt 3,261 + CoT 22 + anchor 1) and all 64 action tokens, a
`[1, 1, 64, 3348]` mask. Expert layer *i* reads VLM layer *i*'s cache
([01 §7](01_MODEL_STRUCTURE.md)), so this holds at all 28 layers.

## 2. How the attention is measured

```
load model with attn="eager"              eager returns softmax(QKᵀ/√d); SDPA never materialises it
        │
hook the 28 expert self_attn modules      keep output[1]: [1, 16 heads, 64 queries, 3348 keys]
        │
run one normal predict()                   10 Euler steps × 28 layers = 280 attention maps
        │
label every key position                   image (per camera) / ego history / route / other text /
        │                                  reasoning / anchor / own action tokens
        │
average over heads and queries             one probability per key (each query row sums to 1)
sum inside each segment                    attention mass of that segment
average over steps (and layers)            the tables below
```

Step by step, with the lines in [`14_expert_attention.py`](scripts/14_expert_attention.py):

| step | line | what happens |
|---|---|---|
| 1 | 50 | `load_edge(attn="eager")`: the eager attention path returns the probability matrix. The default SDPA path computes the same attention without ever storing it. |
| 2 | 60–66 | A forward hook on each `model.expert.expert.layers[i].self_attn` keeps `output[1]`, the attention probabilities `[16, 64, 3348]`. Only expert layers are hooked, not the VLM. |
| 3 | 68 | One ordinary `predict()` (the CLI's function, seed 43). The hooks fire 280 times. Eager attention sampled the same CoT as SDPA: *"Slow down and turn right at the intersection to follow our route, as the lead vehicle has cleared our path."* |
| 4 | 91–117 | Each key position gets a label. `<|image_pad|>` positions (id 19) → images, split per camera from each image's token count (`image_grid_thw`) and the camera-major order; ids `<i0>`…`<i999>` → ego history; the span from `Navigation input is:` to `Your task:` → route sentence; positions after the prompt → reasoning, then the anchor; the last 64 → own action tokens; everything else → other prompt text. |
| 5 | 129–132 | For each map: mean over heads and queries → one probability per key; sum the keys of each segment → `mass[layer, segment]`; average over the 10 steps. |
| 6 | 139 | "Share of keys" = segment size / 3,348: what attention would be if spread uniformly. "Per token vs uniform" = mass / share. |
| 7 | 154– | Ranks single keys by attention to find attention sinks. |

Because every query row is a probability distribution over the 3,348 keys, the segment
masses add up to 100 % for every layer.

## 3. Results

### Overall (mean over 28 layers, 16 heads, 64 queries, 10 steps)

| segment | tokens | share of keys | attention mass | per token vs uniform |
|---|---|---|---|---|
| first token `\n` | 1 | 0.03 % | 7.3 % | 244× |
| other prompt text (system prompt, camera names, time labels, markup, task) | 299 | 8.9 % | 46.6 % | 5.2× |
| images: front camera | 900 | 26.9 % | 6.0 % | 0.2× |
| images: front telephoto | 900 | 26.9 % | 7.1 % | 0.3× |
| images: front left / front right | 273 + 273 | 16.3 % | 0.9 % + 0.6 % | 0.1× |
| images: rear left / rear right | 273 + 273 | 16.3 % | 0.9 % + 0.8 % | 0.1× |
| ego history | 57 | 1.7 % | 3.2 % | 1.9× |
| route sentence | 12 | 0.4 % | 2.1 % | 5.8× |
| **reasoning (CoT)** | **22** | **0.7 %** | **4.5 %** | **6.8×** |
| anchor `<|traj_future_start|>` | 1 | 0.03 % | 0.9 % | 30× |
| own 64 action tokens | 64 | 1.9 % | 19.2 % | 10× |

All six cameras together: 2,892 tokens (86 % of the keys) and 16.3 % of the attention.

### Attention sinks

The single keys with the most attention:

| key | token | attention |
|---|---|---|
| 2 | `system` | 9.2 % |
| 0 | `\n` | 7.3 % |
| 1 | `<|im_start|>` | 7.2 % |
| 3 | `\n` | 4.8 % |
| 6, 9, 12, 15 | ` a`, ` that`, ` and`, `.` (system prompt) | 1.4–1.9 % each |
| 3282 | `.\n` (end of the CoT) | 1.3 % |
| 3283 | `<|traj_future_start|>` | 0.9 % |

The first four tokens `\n<|im_start|>system\n` take **~28 %** of all attention, and the top 10
of the 299 "other prompt text" tokens hold 65 % of that segment's mass. These are
**attention sinks**: positions where heads park probability mass they do not need. They
carry little information, so most of the 47 % "other prompt text" is not reading the
system prompt.

### By layer depth (attention mass, %)

| segment | layers 0–8 | layers 9–18 | layers 19–27 |
|---|---|---|---|
| first token + other prompt text (mostly sinks) | 84.4 | 46.3 | 31.9 |
| images, front + telephoto | 6.9 | 19.3 | 12.3 |
| images, 4 side/rear | 2.4 | 5.5 | 1.5 |
| ego history | 1.2 | 3.1 | 5.2 |
| route sentence | 0.6 | 2.6 | 3.0 |
| reasoning (CoT) | 2.0 | 5.6 | 5.7 |
| anchor | 0.2 | 1.1 | 1.3 |
| own action tokens | 2.3 | 16.6 | 39.1 |

![expert attention](outputs/expert_attention.png)

Reading the figure from left to right: early layers mostly park attention on the sinks;
middle layers read the cameras (front and telephoto far more than the side and rear ones)
and the reasoning; late layers mostly attend among the 64 action tokens — making the 6.4 s of
waypoints consistent with each other — while still checking history, route and reasoning.

## 4. What this does and does not tell you

- **It shows what is *read*, not what *matters*.** Attention mass is not influence: sinks get a
  lot and contribute little; a token with little attention can still carry a decisive value
  vector. The forced-reasoning experiments in [04](04_ACTION_SPACE_AND_FLOW.md) (configs
  3-5c to 3-5f) measure influence directly; the two agree in spirit — the reasoning is read,
  but the scene and route supply most of the plan, and only strong words such as "stop" move it.
- **One clip, one seed, one candidate.** With several candidates the mask also hides other
  candidates' longer CoTs; the per-segment picture can differ by scene (e.g. more side-camera
  attention in a turn across traffic).
- **Eager attention is only for measuring.** It produced the same CoT here, but it is slower
  and uses more memory than SDPA; do not use it for normal inference.

## 5. Debugging it (config 3-5g)

| line | breakpoint / inspect |
|---|---|
| 63 (inside `hook`) | `weights.shape` → `[1, 16, 64, 3348]`; `weights[0, 0, 0].sum()` → 1.0 |
| 72 (after `predict`) | `len(captured)` → 280; `extra["cot"]` |
| 118 (after labelling) | `labels[3280:3290]` — where reasoning, anchor and action tokens meet; `labels[:6]` — the sinks |
| 132 | `mass` → `[28 layers, 13 segments]`, the numbers behind the figure |

To try another scene, change `spec = example()` to any clip, e.g.
`spec = {"clip_id": "030c760c-ae38-49aa-9ad8-f5650a545d26", "t0_us": 5_100_000, "nav_text": "Continue straight"}`.
