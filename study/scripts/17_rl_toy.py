"""RL in miniature: what reward training adds to a planner trained by imitation (CPU, seconds).

    python study/scripts/17_rl_toy.py

The scene: a parked car sticks c metres into our lane (c differs per scene, 0..0.8 m). The
planner picks ONE number, how far left we pass it: the lateral offset y (m). The judge,
shaped like NAVSIM's PDMS (two yes/no gates times a graded part):
    y < c + 1.0        collision        -> score 0
    y > 2.6            off the road     -> score 0
    otherwise          0.5 * margin + 0.5 * progress
        margin   = min(1, (y - c - 1.0) / 0.6)        wants 0.6 m of extra clearance (like TTC)
        progress = 1 - 0.4 * max(0, y - c - 1.6)      swerving wider than that costs progress
Human drivers pass at y = c + 1.3 +- 0.2: usually fine, sometimes too close, sometimes wide.

The planner is a tiny generative model, like the flow-matching expert: given c it outputs a
distribution over y, here a Gaussian N(a*c + b, sigma^2) with three weights (a, b, sigma).
  1. SFT  : fit the weights to the human examples (imitation). It learns the human spread too,
            tails included, exactly as a flow-matching expert learns the spread of human paths.
  2. RL   : per scene sample G = 8 plans, score them with the judge, advantage = (score - group
            mean) / group std, loss = -(advantage * log p(plan)).mean(). The judge is only
            evaluated, never differentiated.
Same loop as Qwen-Drive stage 4, with 3 weights instead of 1.04 B and one number instead of
50 waypoints. Writes study/outputs/rl_toy.png and rl_toy.json.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import OUT, banner, save_json  # noqa: E402

torch.manual_seed(0)
CLEAR, EDGE, WANT, HUMAN_MEAN, HUMAN_STD = 1.0, 2.6, 0.6, 1.3, 0.2
G, SCENES_PER_STEP, STEPS, LR = 8, 64, 800, 0.01
MIN_SIGMA = 0.05  # exploration floor: RL needs some spread to compare plans


def judge(y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """PDMS-shaped score in [0, 1]: gates (collision, off-road) times a graded part."""
    gap = y - c - CLEAR
    margin = torch.clamp(gap / WANT, max=1.0)
    progress = 1 - 0.4 * torch.clamp(gap - WANT, min=0.0)
    ok = (gap >= 0) & (y <= EDGE)
    return torch.where(ok, 0.5 * margin + 0.5 * progress, torch.zeros_like(y))


def human(c: torch.Tensor) -> torch.Tensor:
    return c + HUMAN_MEAN + HUMAN_STD * torch.randn_like(c)


class Planner(torch.nn.Module):
    """p(y | c) = N(a*c + b, sigma^2): the whole 'model' is three numbers."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(0.0))
        self.b = torch.nn.Parameter(torch.tensor(0.0))
        self.log_sigma = torch.nn.Parameter(torch.tensor(0.0))

    def dist(self, c: torch.Tensor) -> torch.distributions.Normal:
        sigma = self.log_sigma.clamp(min=np.log(MIN_SIGMA)).exp()
        return torch.distributions.Normal(self.a * c + self.b, sigma)

    def describe(self) -> str:
        return (f"y ~ N({self.a.item():.2f}*c + {self.b.item():.2f}, "
                f"{self.log_sigma.clamp(min=np.log(MIN_SIGMA)).exp().item():.2f}^2)")


@torch.no_grad()
def evaluate(model: Planner, n: int = 50_000) -> dict:
    """One sampled plan per fresh scene, as at inference."""
    gen = torch.Generator().manual_seed(123)
    c = torch.rand(n, generator=gen) * 0.8
    y = model.dist(c).sample() if model is not None else human(c)
    y_human = c + HUMAN_MEAN + HUMAN_STD * torch.randn(n, generator=gen)  # what the human did
    gap = y - c - CLEAR
    return {"score": judge(y, c).mean().item(),
            "collision_pct": 100 * (gap < 0).float().mean().item(),
            "offroad_pct": 100 * (y > EDGE).float().mean().item(),
            "dist_to_human_m": (y - y_human).abs().mean().item()}


def report(name: str, m: dict) -> None:
    print(f"  {name:22s} score {m['score']:.3f}   collisions {m['collision_pct']:5.2f} %   "
          f"off-road {m['offroad_pct']:5.2f} %   distance to the human's offset "
          f"{m['dist_to_human_m']:.3f} m")


banner("0. The human drivers, scored by the judge")
results = {"human": evaluate(None)}
report("human drivers", results["human"])

banner("1. SFT: imitate the humans (maximise log p(human offset))")
model = Planner()
opt = torch.optim.Adam(model.parameters(), lr=0.05)
for _ in range(2000):
    c = torch.rand(256) * 0.8
    loss = -model.dist(c).log_prob(human(c)).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
print(f"  learned {model.describe()}   (humans: y = 1.00*c + 1.30 +- 0.20)")
results["sft"] = evaluate(model)
report("SFT planner", results["sft"])
sft_state = {k: v.clone() for k, v in model.state_dict().items()}

banner("2. Why the judge cannot simply be a loss term")
c = torch.rand(4096) * 0.8
y = model.dist(c).sample().requires_grad_(True)
score = judge(y, c)
score.sum().backward()
crashed = (y - c - CLEAR < 0).detach()
print(f"  d score / d y over {crashed.sum().item()} crashed plans: "
      f"max |grad| = {y.grad[crashed].abs().max().item():.1f}")
print("  The crashed plans are the ones most worth learning from, and the judge's gradient "
      "there is exactly 0\n  (the score is flat at 0 on the wrong side of the gate). RL "
      "uses their score instead: far below\n  the group mean, so a large negative advantage.")

banner("3. One RL step in slow motion (one scene, G = 8)")
c1 = torch.tensor([0.4])
with torch.no_grad():
    y8 = model.dist(c1.expand(G)).sample()
    s8 = judge(y8, c1.expand(G))
adv = (s8 - s8.mean()) / (s8.std() + 1e-6)
print(f"  parked car sticks out c = 0.40 m -> collision below y = {0.4 + CLEAR:.2f}, "
      f"best score at y = {0.4 + CLEAR + WANT:.2f}, road edge at {EDGE}")
for k in np.argsort(y8.numpy()):
    tag = "CRASH" if y8[k] < 0.4 + CLEAR else ""
    print(f"   plan {k + 1}: y = {y8[k]:.2f} m  score {s8[k]:.2f}  advantage {adv[k]:+.2f}  {tag}")
print(f"  group mean score {s8.mean():.2f}")
mean_before = model.dist(c1).mean.item()
probe = torch.optim.SGD(model.parameters(), lr=0.002)
loss = -(adv * model.dist(c1.expand(G)).log_prob(y8)).mean()
probe.zero_grad()
loss.backward()
probe.step()
print(f"  after this one update the planner's mean offset for c = 0.40 moves "
      f"{mean_before:.3f} -> {model.dist(c1).mean.item():.3f} m (a small step toward the high scorers; RL repeats it thousands of times)")
model.load_state_dict(sft_state)

banner(f"4. RL: {STEPS} steps x {SCENES_PER_STEP} scenes x G = {G} plans, starting from SFT")
opt = torch.optim.Adam(model.parameters(), lr=LR)
curve = []
for step in range(STEPS + 1):
    c = (torch.rand(SCENES_PER_STEP) * 0.8)[:, None].expand(-1, G)  # each scene repeated G times
    dist = model.dist(c)
    with torch.no_grad():
        plans = dist.sample()                                       # [scenes, G]
        scores = judge(plans, c)                                    # black box: no gradient
    adv = (scores - scores.mean(1, keepdim=True)) / (scores.std(1, keepdim=True) + 1e-6)
    loss = -(adv * dist.log_prob(plans)).mean()                     # policy gradient
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0:
        m = evaluate(model, 20_000)
        curve.append({"step": step, **m})
        print(f"  step {step:5d}  score {m['score']:.3f}  collisions {m['collision_pct']:5.2f} %  "
              f"off-road {m['offroad_pct']:5.2f} %  {model.describe()}")
results["rl"] = evaluate(model)

banner("5. Before vs after")
for name in ("human", "sft", "rl"):
    report({"human": "human drivers", "sft": "SFT planner", "rl": "after RL"}[name], results[name])
print(f"  RL planner: {model.describe()}")
print("  RL raised the score and removed the crashes by moving AWAY from how the humans drove,\n"
      "  so the distance to the human's offset grew: the same trade as Qwen-Drive's PDMS 88.2 -> 90.7\n"
      "  with ADE@3 s 0.37 -> 0.42 m.")
save_json("rl_toy.json", {"results": results, "curve": curve,
                          "rl_planner": {"a": model.a.item(), "b": model.b.item(),
                                         "sigma": model.dist(torch.zeros(1)).stddev.item()}})

# Figure: the three distributions on one scene, the learned rule over all scenes, the curve.
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
sft = Planner()
sft.load_state_dict(sft_state)
ax = axes[0]
c0 = 0.4
ys = torch.linspace(0.8, 3.0, 800)
with torch.no_grad():
    for label, d, color in (("human drivers", torch.distributions.Normal(c0 + HUMAN_MEAN, HUMAN_STD), "#52514e"),
                            ("SFT planner", sft.dist(torch.tensor(c0)), "#2a78d6"),
                            ("after RL", model.dist(torch.tensor(c0)), "#1baf7a")):
        ax.plot(ys, d.log_prob(ys).exp(), color=color, lw=2, ls="--" if label.startswith("human") else "-",
                label=label)
    ax.axvspan(0.8, c0 + CLEAR, color="#d62728", alpha=0.12, label="collision (score 0)")
    ax.axvspan(EDGE, 3.0, color="#eda100", alpha=0.15, label="off road (score 0)")
    twin = ax.twinx()
    twin.plot(ys, judge(ys, torch.full_like(ys, c0)), color="#8a4fbf", lw=1.2)
    twin.set_ylabel("judge's score", color="#8a4fbf")
    twin.set_ylim(0, 1.05)
ax.set_xlabel("lateral offset y (m)")
ax.set_ylabel("probability density")
ax.set_title(f"one scene (car sticks out {c0} m): where plans land")
ax.legend(fontsize=8, loc="upper left")
ax = axes[1]
cs = np.linspace(0, 0.8, 50)
ax.fill_between(cs, 0.8, cs + CLEAR, color="#d62728", alpha=0.12, label="collision")
ax.fill_between(cs, EDGE, 3.0, color="#eda100", alpha=0.15, label="off road")
ax.plot(cs, cs + CLEAR + WANT, color="#8a4fbf", lw=1, ls=":", label="where the judge's score peaks")
for label, m, color in (("SFT planner", sft, "#2a78d6"), ("after RL", model, "#1baf7a")):
    with torch.no_grad():
        d = m.dist(torch.tensor(cs, dtype=torch.float32))
        ax.plot(cs, d.mean, color=color, lw=2, label=f"{label}: mean ± 2 sigma")
        ax.fill_between(cs, d.mean - 2 * d.stddev, d.mean + 2 * d.stddev, color=color, alpha=0.15)
ax.set_xlim(0, 0.8)
ax.set_ylim(0.8, 3.0)
ax.set_xlabel("how far the parked car sticks out, c (m)")
ax.set_ylabel("lateral offset y (m)")
ax.set_title("the rule each planner learned, over all scenes")
ax.legend(fontsize=8, loc="upper left")
ax = axes[2]
steps = [p["step"] for p in curve]
ax.plot(steps, [p["score"] for p in curve], color="#1baf7a", lw=2, label="mean score")
ax.set_xlabel("RL step")
ax.set_ylabel("mean score", color="#1baf7a")
twin = ax.twinx()
twin.plot(steps, [p["collision_pct"] for p in curve], color="#d62728", lw=2, label="collisions %")
twin.plot(steps, [p["dist_to_human_m"] * 100 for p in curve], color="#52514e", lw=1.5, ls="--",
          label="distance to human (cm)")
twin.set_ylabel("collisions (%) / distance to human (cm)")
ax.set_title("during RL")
lines = ax.get_lines() + twin.get_lines()
ax.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="center right")
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "rl_toy.png", dpi=100)
print("wrote study/outputs/rl_toy.png")
