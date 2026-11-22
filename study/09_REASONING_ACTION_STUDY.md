# 09 — Does the action follow the reasoning? A reasoning-tampering study

**Question.** Edge writes a reasoning text (CoT) and then the flow-matching expert produces
the trajectory. Does the trajectory actually *follow* what the reasoning says?

**Method.** Tamper with the reasoning: write the CoT by hand, force it into the model in
place of the one it would have written, keep **everything else identical** (images, ego
history, route sentence, diffusion noise), and see what the trajectory does. Any change is
then caused by the text alone.

**Answer, on the README intersection.**

| what the reasoning says | does the trajectory follow? |
|---|---|
| a direction ("turn left", "go straight", "Turn left in 20m") | **only if the route sentence in the prompt names no direction**; otherwise the route wins |
| "stop" (even just "Stop.") | **yes**, whatever the route says: the car brakes to ~0.5 m/s |
| "slow down", "slow down significantly", "speed up", "yield …", a hazard ("crossing traffic has not cleared our path") | **no** measurable effect |
| nothing (`<NOMINAL>` or empty) | the plan is almost the same as with the model's own reasoning |

**And over 21 clips** (§7): removing the reasoning leaves the mean error unchanged
(1.770 m with the model's own reasoning, 1.776 m with `<NOMINAL>` or nothing). On the 8 clips
where the model chose to reason, its reasoning helps by about 2 % against `<NOMINAL>`
(within noise) and 5 % against nothing at all.

| | |
|---|---|
| script | [`scripts/13_cot_intervention.py`](scripts/13_cot_intervention.py); §7: [`scripts/16_reasoning_ablation.py`](scripts/16_reasoning_ablation.py) |
| debug configs | **3-5c** (default: sections 0–2), **3-5d** (route-style reasoning × 3 routes), **3-5e** (speed words × 3 routes), **3-5f** (stop vs hazard vs yield), **3-5h** (§7, 21 clips) — `.vscode/launch.json`, all `justMyCode` + `stopOnEntry` |
| outputs | `outputs/cot_intervention*.png`, `outputs/cot_intervention*.json`; §7: `outputs/reasoning_ablation*.png`, `outputs/reasoning_ablation.json` |
| sample | README clip `ea7bbd31`, t0 = 4.0 s (approaching the intersection where the car turns right), 6 noise seeds per cell |
| runtime | ~1 min per config on the RTX 3090 |

Related: [03](03_INFERENCE_WALKTHROUGH.md) (the call unrolled), [08](08_EXPERT_ATTENTION.md)
(where the expert's attention goes), [04 §6](04_ACTION_SPACE_AND_FLOW.md) (flow matching).

---

## 0. Background: how reasoning and action are connected

```
cameras + ego history + route ──► VLM writes the reasoning, word by word        🎲 text sampling
                                  … "<|traj_future_start|>"   (generation stops here)
                                              │
                         KV cache = prompt (3,261) + reasoning (22) + anchor (1)
                                              │
noise x₀ ~ N(0, I) ───────────► 10 Euler steps: x ← x + 0.1·v(x, t | whole cache)   🎲 noise
                                              │
                                    64 × (accel, curvature) → 6.4 s trajectory
```

Four facts established earlier, which frame this study:

1. **The action waits for the reasoning.** `sample_trajectories_from_data` runs the whole
   generation first (`_generate_with_shared_prefill`, line 286) and only then the expert
   (`_sample_actions`, line 350); with several candidates it waits for all of them.
2. **The noise is not conditioned on the reasoning.** `x₀ = torch.randn(...)` is drawn blind
   ([flow_matching.py:126](../src/alpamayo2_edge/diffusion/flow_matching.py#L126)). The
   reasoning enters at every Euler step, through attention.
3. **The expert attends to the whole cache**, not only the reasoning: images, history, route,
   all text, the reasoning, the anchor and its own 64 tokens. The reasoning is 0.7 % of the
   keys and gets 4.5 % of the attention ([08](08_EXPERT_ATTENTION.md)).
4. **One reasoning ≠ one trajectory.** With the reasoning fixed, six noise seeds spread the
   6.4 s endpoint by 1.70 m, slightly more than six different reasonings (1.34 m)
   ([04, "Where candidate diversity comes from"](04_ACTION_SPACE_AND_FLOW.md), config 3-1).

So the reasoning *can* influence the action (it is in the cache the expert reads). Whether it
*does* is what tampering measures.

## 1. How the tampering works

Normally step 1 is sampled. Here it is replaced by **teacher forcing**: the CoT text is
tokenized and fed through the model in one forward pass, so the cache holds exactly that text.
Step 2 is then the library's own code, unchanged.

| step | line in `13_cot_intervention.py` | what happens |
|---|---|---|
| 1 | 77 | Reference: one normal `predict()` (seed 43). Its CoT is the model's own: *"Slow down and turn right at the intersection to follow our route, as the lead vehicle has cleared our path."* |
| 2 | 84–93 `prompt_for(route)` | The same sample re-tokenized with any route sentence (for the route × reasoning grids). |
| 3 | 99 | `cot_ids = tokenizer(cot + "\n") + [<|traj_future_start|>]` — exactly the format the model produces (re-tokenizing the model's own CoT reproduces its sampled ids). |
| 4 | 103–107 | `vlm.model(prompt + cot_ids, pixel_values, ...)` in one forward → cache of prompt + forced CoT + anchor; `rope_deltas` from the same call. |
| 5 | 108 | `build_expert_pos_ids_and_attn_mask` — the library function the normal path uses. |
| 6 | 114–117 | For each noise seed (1000–1005): `model._sample_actions(...)` (the library's 10-step Euler loop) and `action_space.action_to_traj` → 64 poses. |
| 7 | 126 | **Sanity check**: forcing the model's own CoT with the CLI seed reproduces `predict()` to **1.0 cm** (bf16 rounding: the CoT is cached in one forward instead of token by token). |

What is held fixed in every comparison: the 22 camera frames, the 20 history poses, the route
sentence (unless it is the variable), the noise seeds, the model. Only the forced text changes.

Metrics, means over the 6 noise seeds: endpoint (x forward, y left; y < 0 is right) at 6.4 s,
distance covered in 6.4 s, speed at 6.4 s, and the speed profile over the horizon.

## 2. Experiment A — six reasonings, route "Turn right in 30m" (config 3-5c, §1)

| forced reasoning | endpoint (x, y) | distance | speed at 6.4 s |
|---|---|---|---|
| model's own: "Slow down and turn right … lead vehicle has cleared our path." | (32.5, −5.4) | 34.7 m | 4.7 m/s |
| **"Stop at the intersection and wait, as crossing traffic has not cleared our path."** | **(25.0, −0.5)** | **25.1 m** | **0.5 m/s** |
| "Slow down and turn left at the intersection …" | (32.5, −5.2) | 34.6 m | 4.6 m/s |
| "Accelerate and proceed straight through the intersection …" | (32.7, −5.4) | 34.9 m | 4.7 m/s |
| `<NOMINAL>` | (32.6, −5.0) | 34.6 m | 4.5 m/s |
| (empty) | (33.4, −4.1) | 34.8 m | 4.3 m/s |

"Stop" stops the car short of the intersection; "turn left" and "accelerate … straight" still
turn right at the same speed; no reasoning at all gives nearly the same plan.

![experiment A](outputs/cot_intervention.png)

## 3. Experiment B — route × reasoning (config 3-5c, §2)

| | reasoning: model's own ("… turn right …") | reasoning "… turn left …" | reasoning "stop …" |
|---|---|---|---|
| route "Turn right in 30m" | (32.5, −5.4) 4.7 m/s — right | (32.5, −5.2) 4.6 m/s — **right** | (25.0, −0.5) 0.5 m/s — stops |
| route "Turn left in 30m" | (29.6, −0.5) 2.4 m/s — **not right** | (30.4, 0.1) 2.8 m/s — slow, edges left | (24.8, −0.1) 0.2 m/s — stops |

The rows differ; the first two columns do not. **The route decides the direction; the
reasoning's direction words are not used; "stop" wins in both rows.**

![experiment B](outputs/cot_intervention_grid.png)

## 4. Experiment C — reasoning phrased like a route, plus a route with no direction (config 3-5d)

Maybe the expert ignores "turn left" only because it is buried in a sentence. Here the
reasoning is phrased exactly like a route, and a third route names no direction.

| | reasoning "Turn right in 20m." | reasoning "Turn left in 20m." | reasoning "stop …" |
|---|---|---|---|
| route "Turn right in 30m" | (32.8, −5.6) 4.7 m/s — right | (32.8, −5.3) 4.6 m/s — **right** | (25.0, −0.5) 0.5 m/s — stops |
| route "Turn left in 30m" | (29.3, −0.3) 2.1 m/s — **not right** | (30.5, 0.2) 2.7 m/s — slow, edges left | (24.8, −0.1) 0.2 m/s — stops |
| route **"Continue straight"** | (32.0, −3.5) 3.6 m/s — **turns right** | (30.4, 0.3) 2.6 m/s — **slow, edges left** | (24.8, −0.1) 0.2 m/s — stops |

Phrasing does not matter when the route names a direction. But with "Continue straight" the
reasoning's direction **is** followed. So the expert does read direction from the reasoning;
it gives the route sentence priority when both speak.

"Left" always looks like "slow, edges left": this left turn crosses traffic behind a yield
sign, and within 6.4 s the model only creeps into it — the same as with a left route in the
route ablation (config 3-6a).

![experiment C](outputs/cot_intervention_grid_route_style.png)

## 5. Experiment D — speed words (config 3-5e)

| | `<NOMINAL>` (baseline) | "Slow down." | "Slow down significantly." | "Speed up." | "Stop at the intersection and wait, …" |
|---|---|---|---|---|---|
| route right | 34.6 m, 4.5 m/s | 34.3 m, 4.4 m/s | 34.3 m, 4.4 m/s | 34.9 m, 4.5 m/s | **25.1 m, 0.5 m/s** |
| route left | 29.3 m, 2.1 m/s | 29.6 m, 2.2 m/s | 29.5 m, 2.2 m/s | 29.7 m, 2.1 m/s | **24.8 m, 0.2 m/s** |
| route straight | 29.9 m, 2.3 m/s | 30.2 m, 2.4 m/s | 29.9 m, 2.4 m/s | 30.4 m, 2.4 m/s | **24.8 m, 0.2 m/s** |

(distance covered in 6.4 s, speed at 6.4 s)

"Slow down", "slow down significantly" and even "speed up" stay within 0.3 m of the baseline:
their speed curves lie on top of each other. The **route** sets the speed: left and straight
yield at the crossing and brake to ~2 m/s; right keeps ~4.5 m/s.

![experiment D](outputs/cot_intervention_speed_slow_down.png)

## 6. Experiment E — which part of "stop" matters (config 3-5f, route right)

The stop sentence contains a command *and* a hazard. Split them:

| forced reasoning | distance | speed at 6.4 s |
|---|---|---|
| `<NOMINAL>` | 34.6 m | 4.5 m/s |
| **"Stop."** | **28.0 m** | **1.0 m/s** |
| **"Stop at the intersection and wait."** | **24.6 m** | **0.4 m/s** |
| "Crossing traffic has not cleared our path." | 34.7 m | 4.6 m/s |
| "Yield to the crossing vehicle from the right before turning right to follow our route." | 34.6 m | 4.6 m/s |
| "Stop at the intersection and wait, as crossing traffic has not cleared our path." | 25.1 m | 0.5 m/s |

The trigger is the **word "stop"**: alone it brakes the car hard, with "and wait" almost to a
standstill. The hazard on its own, and "yield", do nothing.

![experiment E](outputs/cot_intervention_speed_stop_split.png)

## 7. Experiment F — how much does the reasoning help? 21 clips (config 3-5h)

Experiments A–E tamper with one scene. This one asks the practical question: **if the
reasoning is removed, does the trajectory get worse?** It uses the 21 clips of the
evaluation (config 3-7a, t0 = 5.1 s, oracle route) and scores against the logged future.

**Setup** ([`scripts/16_reasoning_ablation.py`](scripts/16_reasoning_ablation.py)). Per clip,
three conditions share the images, history, route and the noise seed of each candidate k
(1000 + k), so only the forced text differs:

- **own**: the six CoTs the model samples itself (CLI settings, seed 43), forced back in
  with the same teacher forcing as §1.
- **`<NOMINAL>`**: the model's own "nothing to explain" output, forced for all six.
- **empty**: just the newline and the anchor right after the prompt. This is the closest
  thing to Qwen-Drive's `DIRECT_PLANNING`, but Edge was never trained on it.

**Results** (ADE over 6.4 s in metres, averaged over the 6 candidates of each clip, then over
clips):

| clips | own | `<NOMINAL>` | empty |
|---|---|---|---|
| all 21 | **1.770** | 1.776 | 1.776 |
| the 8 where candidate 1 reasoned | **2.434** | 2.484 | 2.556 |
| those 8, ADE over the first 3 s, candidate 1 | **0.860** | 0.958 | 1.002 |
| those 8, best of 6 (minADE) | 1.342 | **1.223** | 1.283 |

Paired per clip (own minus the other condition; negative = the reasoning helped; bootstrap
95 % interval over clips):

- **All 21 clips:** −0.006 m against `<NOMINAL>` [−0.07, +0.06] and −0.006 m against empty
  [−0.09, +0.07]. **No effect.** On 13 of the 21 clips candidate 1 wrote `<NOMINAL>` itself
  (76 of the 126 candidates), and there own and `<NOMINAL>` are the same input by
  construction.
- **The 8 clips where the model reasoned:** against `<NOMINAL>` −0.05 m [−0.20, +0.12],
  better on 6 of 8 (about 2 %, within noise). Against empty −0.12 m [−0.26, −0.02],
  better on 6 of 8 (about 5 %). Over the first 3 s against empty: −0.14 m [−0.25, −0.06],
  better on **8 of 8**.
- **Plan shift:** on the 50 candidates where the model reasoned, swapping its CoT for
  `<NOMINAL>` moves the plan by 0.34 m on average (0.41 m for empty). Changing only the
  noise seed, with the text fixed, moves it by **1.17 m**. The noise seed matters about
  3× more than the reasoning.
- **Cost:** 22 generated tokens on average when it reasons, 6 for `<NOMINAL>`.

The clearest single win is the stop-sign clip `cf1fabe6` ("Continue decelerating to stop at
the stop line for the stop sign …"): 1.74 m with its reasoning, 2.06 m with `<NOMINAL>` and
1.93 m with empty. That fits Experiment E: "stop" is the word the expert listens to.

![reasoning ablation](outputs/reasoning_ablation.png)
![reasoning ablation, clips that reasoned](outputs/reasoning_ablation_bev.png)

**What it means.**

1. Over a typical mix of clips the reasoning changes nothing measurable, mostly because the
   model already skips it (`<NOMINAL>`) where nothing is happening.
2. Where it does reason, the gain is small (2–5 %) and mostly in the first 3 s.
3. Best-of-6 does not improve (minADE +0.12 m, interval spans zero): the reasoning does not
   make the candidates more diverse.
4. If you want to skip the reasoning in Edge, **force `<NOMINAL>`, not empty**. Empty is
   slightly worse on the clips that needed reasoning (2.556 vs 2.484 m) because the model
   never saw it in training.

**Why so small.** The expert reads the whole KV cache: 2,892 image tokens, 57 history
tokens, the route sentence and the CoT. The CoT is a ~22-token summary of things already in
the cache. [08](08_EXPERT_ATTENTION.md) measured it: 0.7 % of the keys, 4.5 % of the
attention.

**A pitfall found on the way.** Candidate k gets the same noise seed on every clip, so one
unlucky draw shows up everywhere. Seed 1000 (candidate 1) has a mean ADE of 3.0 m in all
three conditions, against 1.3–1.7 m for the other five. That is why the headline numbers
average the six candidates, and why "candidate 1" here (3.0 m) is worse than in 3-7a
(1.50 m, different noise). The script prints the per-seed means.

**Caveats.** 21 clips at one moment each, open loop, only 8 with reasoning. ADE measures
how close the plan stays to one logged drive, not safety. The gains NVIDIA reports for
reasoning (Alpamayo-R1: up to 12 % on curated hard cases, 35 % fewer off-road events and
25 % fewer close encounters in closed-loop simulation) come from hard cases and closed-loop
driving, which this setup cannot see. Qwen-Drive's own tables show the same small open-loop
effect: NAVSIM PDMS 87.8 → 88.2, Waymo ADE@3 s 1.20 → 1.19 m, PhysicalAI ADE@3 s
0.38 → 0.37 m (`~/Desktop/Qwen-Drive-1.0/docs/evaluation.md`).

## 8. What it means

**The rule, on this scene:**

1. **Direction** comes from the route sentence in the prompt; the reasoning decides direction
   only when the route names none.
2. **Speed** comes from the route and the images, **except** that an explicit "stop" in the
   reasoning overrides it.
3. **Everything else in the reasoning** — "slow down", "yield", "speed up", hazard
   descriptions — has no measurable effect on the trajectory.

**This is not a bug.**

- The code path is the library's own (`_sample_actions`, `action_to_traj`), and forcing the
  model's own reasoning reproduces normal inference to 1 cm.
- The conflicts are artificial. In normal use the model writes its reasoning *after* reading
  the route, so they agree: with route "Turn left in 30m" the model writes "turn left …"
  itself (config 3-6a). Tampering only reveals which input the expert relies on when they
  disagree.

**A plausible explanation** (an inference; the training code is not in this repo): during
training, reasoning and route always agreed, so for direction the expert could learn from
either — and the route is the more reliable signal (always present, explicit, while the
reasoning is often `<NOMINAL>` or silent on direction). For speed the route says nothing.
Words like "slow down", "decelerate" and "yield" appear in most reasoning near
intersections — the model's own reasoning here starts with "Slow down…" and its plan matches
`<NOMINAL>` — so they carry little that the images do not already show; "stop" goes with a
distinctly different trajectory, so it carries real information.

**Why it matters.** If you treat the reasoning as a *faithful explanation* of the trajectory,
it is only partly one: it tells you whether the model intends to stop, but not where the path
goes, and several different trajectories fit the same sentence (§0, fact 4). In the
literature this is called partial *faithfulness* of chain-of-thought. For driving the
behaviour is sensible: like a driver who follows the GPS over a passenger saying "go left",
but brakes when the passenger shouts "stop".

**Caveats.** One intersection, one moment, one route family. The rule should be checked on
other clips (straight roads, lane changes, highway) before it is relied on.

## 9. How to run, change and debug it

```bash
# defaults: sanity check, experiment A, experiment B          (config 3-5c)
python study/scripts/13_cot_intervention.py

# any grid: routes are rows, forced reasoning texts are columns (config 3-5d / 3-5e / 3-5f)
python study/scripts/13_cot_intervention.py --grid-only --tag my_test \
    --grid-routes "Turn right in 30m" "Continue straight" \
    --grid-cots "<NOMINAL>" "Stop." "Change to the left lane."
```

`--grid-cots` accepts `own` (the model's own CoT), a key of `COTS` in the script
(`stop`, `turn left`, `go straight`, `<NOMINAL>`, `empty`), or any literal text.
Each run writes `cot_intervention_grid_<tag>.png` (BEV per route),
`cot_intervention_speed_<tag>.png` (speed over time) and `cot_intervention_<tag>.json`.

| line | breakpoint / inspect |
|---|---|
| 99 | `tok.decode(cot_ids)` — the exact text forced in, ending with `.\n<|traj_future_start|>` |
| 107 | `cache.get_seq_length()` → prompt + CoT + anchor; `rope_delta` → −2556 for this prompt |
| 115 | inside the seed loop: `action` `[1, 64, 2]`, the normalized (accel, curvature) |
| 126 | `own` vs `ref_xyz` — the 1 cm sanity check |
| 191 | `profile` `[seeds, 64]` — the speed curves in the plots |

For §7 (`16_reasoning_ablation.py`, config 3-5h):

| line | breakpoint / inspect |
|---|---|
| 82 | `tok.decode(forced)`: `<NOMINAL>\n` + anchor, or just `\n` + anchor for empty |
| 121 | one forward per distinct CoT; identical CoTs share it |
| 128 | `cots`: the six sampled CoTs; `reasoned` marks the ones that are not `<NOMINAL>` |
| 139 | `row["shift_own_vs_nominal_m"]` vs `row["noise_floor_m"]`, per candidate |

To try another clip, change `spec = example()` near the top to any
`{"clip_id": ..., "t0_us": ..., "nav_text": ...}` (the reasoning texts in `COTS` are written for
an intersection; adapt them to the scene).
