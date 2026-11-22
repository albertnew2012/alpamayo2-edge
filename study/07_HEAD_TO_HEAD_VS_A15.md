# 07 — Head to head: Alpamayo 2 Edge vs Alpamayo 1.5 on the same 21 clips

The comparison [02 §5](02_VS_ALPAMAYO_1_5_AND_QWEN_DRIVE.md) said was missing: both models,
the same clips, the same t0, the same number of candidates, the same route text, scored
against the same ground truth.

| | |
|---|---|
| clips | the 21 of [`eval_oracle_k6.json`](outputs/eval_oracle_k6.json): Alpamayo 1.5's own test clip `030c760c` + 20 drawn with seed 0 from all clips with six cameras and egomotion |
| t0 | 5.1 s |
| candidates | 6 per call (Edge `num_traj_sets=6`; A1.5 `num_traj_samples=6`), top-p 0.98, T 0.6, ≤ 256 new tokens |
| inputs | each model's **native** profile: Edge 6 cameras / 22 frames / 20 poses; A1.5 4 cameras / 16 frames / 16 poses |
| ground truth | egomotion at PhysicalAI-AV revision `2ac2e9c`, pinned in both runs; endpoints agree to **0.0000 m** on all 21 clips |
| routes | **oracle**: the same derived text for both (`derive_nav_text`); **wrong**: `"Turn right in 30m"` everywhere; **none**: A1.5 only (Edge requires a route) |
| routes, A1.5 only | **+ CFG**: the same oracle and wrong routes through `sample_trajectories_from_data_with_vlm_rollout_cfg_nav`, guidance weight 1.5 (the 1.5 nav notebook's setting) |
| where it ran | Edge in this repo (configs 3-7a/b). A1.5 in `~/Desktop/alpamayo1.5` with its own venv and code: [`study/scripts/21_eval_vs_edge.py`](/home/albert/Desktop/alpamayo1.5/study/scripts/21_eval_vs_edge.py), launch configs **29** (none/oracle/wrong) and **29b** (CFG) there, loaded with that repo's `run_inference_local.load_model` (FA2; 14.3 GiB of weights on cuda:0 — 11 GiB for CFG, which OOMed otherwise — rest offloaded to CPU; peak 18.5 / 18.6 GiB) |
| compare | [`11_compare_with_a15.py`](scripts/11_compare_with_a15.py), config **3-7c** |

## 1. Results (median over clips; mean in brackets; metres)

| run | ADE@3 s cand. 1 | minADE@3 s | ADE@6.4 s cand. 1 | minADE@6.4 s | FDE@6.4 s cand. 1 | minFDE@6.4 s |
|---|---|---|---|---|---|---|
| **Edge, oracle route** | **0.152** (0.33) | 0.101 (0.19) | **0.671** (1.50) | **0.388** (0.83) | **2.279** (4.21) | 1.088 (2.17) |
| A1.5, oracle route | 0.196 (0.34) | **0.082** (0.17) | 1.019 (1.59) | 0.494 (0.92) | 2.611 (4.77) | **0.793** (2.70) |
| A1.5, no route | 0.194 (0.34) | 0.082 (0.17) | 1.029 (1.64) | 0.471 (0.93) | 2.803 (4.84) | 0.991 (2.67) |
| Edge, wrong route | 0.704 (1.14) | 0.180 (0.60) | 2.626 (6.31) | 0.988 (3.37) | 7.040 (18.56) | 3.316 (10.57) |
| A1.5, wrong route | 0.191 (0.33) | 0.095 (0.18) | 0.839 (1.56) | 0.546 (0.97) | 2.405 (4.74) | 0.812 (2.79) |
| A1.5, oracle route + CFG | 0.196 (0.34) | 0.082 (0.17) | 1.017 (1.58) | 0.483 (0.91) | 2.581 (4.76) | 0.767 (2.67) |
| A1.5, wrong route + CFG | 0.185 (0.34) | 0.098 (0.18) | 0.866 (1.56) | 0.535 (0.98) | 2.277 (4.74) | 0.735 (2.84) |

![comparison](outputs/compare_a15.png)

## 2. Is Edge better? Paired, per clip

Medians favour Edge with the oracle route (0.67 vs 1.02 m), but the medians of two columns
are not a paired comparison. Per clip (Wilcoxon signed-rank, two-sided):

| comparison | metric | Edge better on | median per-clip difference | p |
|---|---|---|---|---|
| Edge vs A1.5, both oracle route | ADE@6.4 s cand. 1 | 12 / 21 | −0.058 m | 0.89 |
| | minADE@6.4 s | 11 / 21 | −0.035 m | 0.56 |
| | ADE@3 s cand. 1 | 11 / 21 | −0.001 m | 0.89 |
| Edge (oracle) vs A1.5 (no route, its default) | ADE@6.4 s cand. 1 | 13 / 21 | −0.189 m | 0.61 |
| Edge vs A1.5 + CFG, both oracle route | ADE@6.4 s cand. 1 | 11 / 21 | −0.049 m | 0.92 |

**On these 21 clips the two models are statistically indistinguishable.** Edge does as well
as Alpamayo 1.5 at a quarter of the parameters (2.65 B vs 11.08 B) and, on this machine,
25× less wall time (1.19 s vs 30.2 s per 6-candidate call — A1.5's number is dominated by
CPU offload, so this is a statement about fitting on a 3090, not about the models' FLOPs).
With 21 clips only a large difference would show up; "no detectable difference" is the
honest summary, not "Edge wins".

## 3. The real difference: how hard each model follows the route

| | Edge | A1.5 |
|---|---|---|
| median ADE@6.4 s increase, wrong route vs oracle route | **+1.434 m** | −0.016 m |
| clips worse by more than 0.5 m with the wrong route | **13 / 21** | 1 / 21 |
| clips whose candidate-1 CoT text changes with the route | 7 / 21 | 1 / 21 |
| clips whose mean 6.4 s endpoint moves > 1 m right under "Turn right in 30m" | **10 / 21** | 1 / 21 (CFG: 1 / 21) |
| A1.5 + CFG: median ADE@6.4 s increase, wrong vs oracle | — | −0.017 m (worse by > 0.5 m on 2/21) |
| A1.5: oracle route vs no route, ADE@6.4 s | — | better on 15/21, median −0.014 m (p = 0.055) |

Lateral endpoint shift (mean over 6 candidates, + = left) when the oracle route is replaced
by "Turn right in 30m", selected clips:

| clip | speed | Edge | A1.5 | A1.5 + CFG |
|---|---|---|---|---|
| `2bb07f05` highway | 30.1 m/s | **−15.0 m** | −0.05 m | −0.11 m |
| `a4625ea7` intersection | 14.4 m/s | **−9.2 m** | −0.03 m | −0.06 m |
| `030c760c` (1.5's test clip) | 8.7 m/s | **−5.2 m** | −0.05 m | −0.11 m |
| `f85dd14e` intersection | 8.3 m/s | **−5.2 m** | −0.02 m | −0.03 m |
| `7f1d660f` left turn | 14.3 m/s | −22.0 m (turns right instead) | −2.3 m | −2.3 m |
| `eeed08e5` highway | 30.0 m/s | −1.0 m | −0.2 m | −0.3 m |

On these clips Alpamayo 1.5 writes the same chain of causation ("Keep lane since the lane
is clear ahead", "Nudge to the left to clear the construction equipment…") whatever the
`<|route_start|>…<|route_end|>` says, and its plans move by centimetres — with or without
classifier-free guidance at 1.5. It is **not** deaf to routes: your own 1.5 notes
([`12_LOCAL_SETUP.md`](/home/albert/Desktop/alpamayo1.5/study/12_LOCAL_SETUP.md), nav
comparison) show "Turn right in 30m" pulling its endpoint to y = −3.7 m at the intersection
in clip `ea7bbd31` (Edge on the same clip and route: −4.7 m, config 3-6a,
[`outputs/nav_ablation.png`](outputs/nav_ablation.png)). It follows a route where the scene makes the manoeuvre plausible and
otherwise keeps to the road. Edge follows it far more readily: its CoT restates the route
("turn right … to follow our route") and its plan turns right on 10 of 21 clips, including a
15 m swerve at 30 m/s.

That is a trade-off, not a bug in either:

- Edge **is easy to steer** and is at least as good when the route is right, but it
  **depends on the route being right**: with a wrong route its median ADE is 3× A1.5's, and
  some of what it does (a right turn at 30 m/s) is unsafe.
- A1.5 is **robust to a bad route** and resists instructions the scene does not support, at
  the price of being hard to steer where several manoeuvres are possible.

## 4. Clip-level observations ([`outputs/compare_a15_bev.png`](outputs/compare_a15_bev.png))

![overlay](outputs/compare_a15_bev.png)

- On most clips the two models choose nearly the same path (the bold lines overlap).
- **Shared failure**: `e92fce02`, a roundabout at 2.9 m/s — both swing right while the car
  went round (ADE 8.9 m Edge, 7.0 m A1.5). The oracle route ("Continue straight", from a
  crude heuristic) is wrong for a roundabout, but A1.5, which barely reacts to the route,
  fails too, so the scene itself is hard.
- **Edge much better**: `a1436457` at 24 m/s (0.49 vs 4.84 m) — A1.5's first candidate
  misjudges speed behind a lead vehicle; `4d0e8218` creeping at 0.2 m/s (0.29 vs 1.24 m).
- **A1.5 much better**: `d8db2176` curve at 17 m/s (0.60 vs 2.62 m), `8d82d3c7` roundabout
  with "Turn left in 10m" (0.62 vs 1.90 m), `12fe079a` highway (0.17 vs 0.59 m).
- **Reasoning style**: A1.5 always writes a short causal sentence ("Keep lane since the lane
  is clear ahead"); Edge writes `<NOMINAL>` on 13 of these clips and a longer,
  route-referencing sentence on the rest.

## 5. What would make this stronger

- More clips (the dataset has 306 k). The per-clip difference in ADE@6.4 s has a robust
  spread of ~0.73 m (MAD-based; 1.36 m standard deviation), so resolving a 0.1 m difference
  at 80 % power takes roughly **400–1 500 paired clips**; 21 can only detect ~0.4 m or more.
- A route source that is not derived from the future (map-matched navigation); the
  roundabout clip shows what a crude heuristic does.
- A1.5 fully on a GPU that holds it, for a timing comparison that is about the models.
- Each model at its own best candidate-selection policy (neither repo ships a scorer;
  "candidate 1" and "min over 6" bracket what a selector could achieve).
