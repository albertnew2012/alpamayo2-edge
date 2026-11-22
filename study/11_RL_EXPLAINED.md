# 11 — RL explained with a toy, then mapped onto Qwen-Drive

**Question.** Qwen-Drive's `planner-rl` scores 90.7 PDMS against 88.2 for `planner-sft`, the
same network before RL. What is RL, and why does it help a planner already trained on human
driving?

| | |
|---|---|
| script | [`scripts/17_rl_toy.py`](scripts/17_rl_toy.py), CPU only, ~10 s |
| debug config | **3-10** in `.vscode/launch.json` (`justMyCode` + `stopOnEntry`) |
| outputs | `outputs/rl_toy.png`, `outputs/rl_toy.json` |

## 1. The toy

A parked car sticks `c` metres into our lane (0–0.8 m, different in every scene). The planner
outputs **one number**: how far left we pass it, the lateral offset `y`.

**The judge** has the same shape as NAVSIM's PDMS: two yes/no gates times a graded part.

- `y < c + 1.0` → **collision**, score 0
- `y > 2.6` → **off the road**, score 0
- otherwise the score rises with extra clearance up to 0.6 m (like PDMS's time-to-collision),
  then falls slowly if you swerve wider than needed (like PDMS's progress)

**The humans** pass at `y = c + 1.3 ± 0.2`. Usually that's fine. Sometimes it's too close:
6.8 % would count as collisions by the judge's strict rule.

**The planner** is a tiny generative model, like the flow-matching expert. Given `c` it
outputs a *distribution* of offsets, `N(a·c + b, σ²)`, and a plan is a sample from it. The
whole model is three weights: `a`, `b`, `σ`.

## 2. Step 1, SFT: copy the humans

Training maximises the probability of the human offsets. The planner learns
`y ~ N(1.03·c + 1.29, 0.20²)`, an almost perfect copy of the humans, **including their spread**.

That is the point: a model trained to imitate reproduces the human *distribution*, the
too-close tail included. It has no way to know that the left tail crashes and the right tail
is fine, because both are equally "human".

## 3. Step 2, RL: try, get graded, push toward the better tries

One RL step, printed by the script (section 3) for one scene with `c = 0.4`. Collision is below
1.40 m and the score peaks at 2.00 m:

| plan | offset y | score | advantage = (score − group mean) / spread |
|---|---|---|---|
| 3, 4, 5 | 1.30, 1.34, 1.37 m | 0 (crash) | **−1.19** |
| 1 | 1.66 m | 0.72 | +0.45 |
| 7, 2, 8 | 1.75–1.85 m | 0.79–0.87 | +0.63 … +0.82 |
| 6 | 1.94 m | 0.95 | **+0.98** |

Group mean 0.52. The update then:

```python
loss = -(advantage * log_prob(plan)).mean()   # line 158
```

It raises the probability of plans with positive advantage, lowers it for negative ones, and
does nothing to the judge: the judge only supplied the numbers. After this single step the
planner's average offset for `c = 0.4` moves from 1.707 to 1.720 m, toward the good plans.
RL repeats this over many scenes (here 800 steps × 64 scenes × 8 plans).

**Result (50 000 fresh scenes, one sampled plan each):**

| | score | collisions | off-road | distance to the human's offset |
|---|---|---|---|---|
| human drivers | 0.713 | 6.83 % | 0.05 % | 0.227 m |
| SFT planner | 0.723 | 6.05 % | 0.07 % | 0.224 m |
| **after RL** | **0.985** | **0.00 %** | 0.00 % | **0.342 m** (worse) |

The RL planner is `y ~ N(1.00·c + 1.63, 0.05²)`. It learned two things:

- **pass wider:** +1.63 instead of +1.29, where the judge's score peaks;
- **be more decisive:** spread 0.05 instead of 0.20, because the spread only produced crashes.

It also moved **away** from how the humans drove, so the distance to the human's offset grew.

![RL toy](outputs/rl_toy.png)

Left: one scene. The SFT planner (blue) sits on top of the humans (dashed) and leaks into the
collision zone. After RL (green) it is a narrow peak where the judge's score (purple) is
highest. Middle: the learned rule over all scenes. Right: RL training.

## 4. Why not just use the judge as a loss?

Section 2 of the script runs `score.backward()` on 263 crashed plans. The gradient is
**exactly 0** for every one of them. The score is flat at 0 on the wrong side of the gate,
so it gives no direction at all, and those are the plans most worth learning from. RL never
differentiates the judge. A crashed plan gets a large negative advantage, and the gradient
flows through `log_prob(plan)`, which belongs to the model and is always differentiable.
(Your Qwen-Drive stage 3b hit the same wall with a sharp occupancy map and had to blur it;
see `~/Desktop/Qwen-Drive-1.0/study/11_GROUNDING_PLAN_IN_PERCEPTION.md` §2.6.)

## 5. The same thing in Qwen-Drive

| toy | Qwen-Drive stage 4 (technical report, arXiv 2609.00111) |
|---|---|
| planner = 3 weights | planning expert, 1.04 B weights (VLM and vision encoder frozen) |
| a plan = one number `y` | a plan = 50 waypoints |
| randomness = sampling from `N(μ, σ²)` | noise added in the last 3 of the 10 denoising steps, in 6 smooth cosine shapes |
| judge = gates × graded part | NAVSIM PDMS, Waymo RFS (human ratings), plus a displacement term |
| 8 plans per scene | 8 plans per scene, each after its own sampled reasoning |
| `-(advantage * log_prob).mean()` | the same, summed over the 3 noisy steps with weights 0.6² / 0.6 / 1 |
| SFT 0.723 → RL 0.985, collisions 6 % → 0 % | PDMS 88.2 → 90.7: drivable area 96.6 → 98.2, progress 82.4 → 84.8, no-collision 98.4 → 98.6 |
| distance to human 0.224 → 0.342 m | ADE@3 s 0.37 → 0.42 m (PhysicalAI) |

**The benefit, in one sentence:** imitation copies the human distribution, mistakes included,
and RL reshapes that distribution toward what the judge rewards: fewer plans off the road or
too close, more confident progress. The price is drifting from the logged human path.

**What the toy leaves out:** the real expert's plans depend on images through 1 B weights, so
RL has to *generalise* to scenes it never trained on. The toy's rule `a·c + b` generalises
trivially. Your stage 3b found that this is the hard part: its collision term worked on the
120 training frames and was not resolved on 80 held-out frames.

## 6. The math of one update (config 3-10b)

Script: [`scripts/18_advantage_math.py`](scripts/18_advantage_math.py). The planner outputs a
Gaussian `N(μ, σ²)`, so

```
log p(y) = −(y − μ)² / (2σ²) − log σ − const
∂ log p(y) / ∂μ = (y − μ) / σ²
loss = −A · log p(y)            →   ∂ loss / ∂μ = −A · (y − μ) / σ²
gradient descent: μ ← μ − lr · ∂loss/∂μ = μ + lr · A · (y − μ) / σ²
```

- **A > 0:** μ moves toward `y`, so `p(y)` goes up.
- **A < 0:** μ moves away from `y`, so `p(y)` goes down.
- **A = 0:** nothing moves.

Measured, with `y` = 1.90, μ = 1.70, σ = 0.2 and one step of lr 0.01:

| A | ∂loss/∂μ | μ after | p(y) before → after |
|---|---|---|---|
| +1 | −5.00 | 1.750 | 1.210 → **1.506** |
| 0 | 0.00 | 1.700 | 1.210 → 1.210 |
| −1 | +5.00 | 1.650 | 1.210 → **0.913** |

**A group** sums the pulls, `Σ A_i (y_i − μ) / σ²`. With the 8 plans of §3, the three crashes
sit below μ with A = −1.19. Negative times negative gives a pull *up*, away from them
(+1.49, +1.34, +1.23). The good plans sit above μ with A > 0, so they also pull up, toward
themselves. Net: μ goes from 1.700 to 1.754. The crashes lose probability (0.270 → 0.151 …) and
the best plans gain it (0.971 → 1.296).

**A refinement of the "better than average goes up" rule.** Plan 4 (`y` = 1.66, A = +0.46)
*lost* probability (1.955 → 1.785). Its own pull is toward itself, but the other plans pull
harder the other way, and the step carries μ past it. Each plan's term pushes its own
probability in the direction of its advantage. The net change is the sum over the group, so
a good-but-not-best plan can still lose probability.

**Any model, not only a Gaussian.** For weights θ, the step on `−A · log p(y)` is
`Δθ = lr · A · ∇θ log p(y)`. To first order:

```
Δ log p(y) ≈ ∇θ log p(y) · Δθ = lr · A · |∇θ log p(y)|²
```

`|…|²` is never negative, so the change has the sign of `A`. The script checks it on a tiny
MLP: +0.152 measured vs +0.166 predicted for A = +1, and −0.180 vs −0.166 for A = −1.

In Qwen-Drive `log p` is the log-probability of each of the 3 noisy denoising steps (each one
a Gaussian jump, as here), and θ is the 1.04 B weights of the planning expert.

## 7. Breakpoints (configs 3-10 and 3-10b)

| line | inspect |
|---|---|
| 105 | SFT loss: `-log p(human offset)`, plain imitation |
| 118 | `y.grad[crashed]`: all zeros, the judge has no gradient where it matters |
| 143 | the single slow-motion RL step: `adv`, `y8`, `s8`; step over it and `model.b` moves |
| 155 | `plans` `[64, 8]`: 8 sampled plans per scene |
| 156 | `scores`: judge output, under `no_grad` |
| 157 | `adv`: score minus the group mean, over the group's spread |
| 158 | the policy-gradient loss |

`18_advantage_math.py` (config 3-10b):

| line | inspect |
|---|---|
| 28 | `loss = -A * log_p(mu, y)` for A = +1, 0, −1 |
| 32 | `mu.grad` against the formula `-A * (y - mu) / sigma^2`, then `mu` after the step |
| 42 | the group loss: `adv`, `plans`, `scores` |
| 44 | `pull`: each plan's share of the step on μ |
| 75 | the MLP step: `grads`, `sq_norm` |
