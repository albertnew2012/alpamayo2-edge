"""Session 5b: watch noise become a trajectory, and how many Euler steps it needs.

    python study/scripts/05_flow_matching.py

1. Captures every intermediate x_t of the 10-step Euler loop (by wrapping
   ``expert.diffusion.sample`` with ``return_all_steps=True``) and decodes each one
   through the action space, so you can see the plan sharpen.
2. Re-runs the same seed with 1, 2, 3, 5, 10, 20, 50 steps. The CoT and the initial
   noise are identical across runs (text sampling and the diffusion generator are seeded
   separately), so every difference comes from the integrator alone.
Writes study/outputs/flow_matching.png and study/outputs/flow_matching.json.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    OUT,
    ade,
    banner,
    example,
    fde,
    load_edge,
    load_logged_future,
    load_sample,
    predict,
    prepare,
    save_json,
)

model, _ = load_edge()
spec = example()
data = load_sample(**spec)
inputs = prepare(model, data)
logged = load_logged_future(spec["clip_id"], spec["t0_us"])
hist = data["ego_history_xyz"][0].numpy()

banner("1. The 10 Euler steps of one sample")
diffusion = model.expert.diffusion
original_sample = diffusion.sample
captured = {}


def sample_all_steps(*args, **kwargs):
    kwargs["return_all_steps"] = True
    steps, times = original_sample(*args, **kwargs)
    captured["x"], captured["t"] = steps, times
    return steps[:, -1]


diffusion.sample = sample_all_steps
try:
    xyz, _, extra = predict(model, inputs)
finally:
    diffusion.sample = original_sample
print("  CoT:", extra["cot"].reshape(-1)[0])
all_x = captured["x"]  # [1, steps + 1, 64, 2]
times = captured["t"].tolist()
history_xyz = inputs["ego_history_xyz"][:, -1]
history_rot = inputs["ego_history_rot"][:, -1]
decoded = []
for k in range(all_x.shape[1]):
    step_xyz, _ = model.expert.action_space.action_to_traj(all_x[:, k].float(), history_xyz,
                                                           history_rot)
    decoded.append(step_xyz[0].float().cpu().numpy())
final = decoded[-1]
print(f"  {'t':>5} {'|x| rms':>8} {'endpoint x,y (m)':>20} {'dist to final (m)':>18}")
for k, t in enumerate(times):
    rms = all_x[0, k].float().pow(2).mean().sqrt().item()
    gap = np.linalg.norm(decoded[k][:, :2] - final[:, :2], axis=-1).mean()
    print(f"  {t:5.1f} {rms:8.3f} {decoded[k][-1, 0]:9.2f},{decoded[k][-1, 1]:8.2f}   {gap:14.3f}")

banner("2. Step-count sweep (same CoT, same initial noise)")
sweep = {}
reference = None
for steps in (1, 2, 3, 5, 10, 20, 50):
    torch.cuda.synchronize()
    xyz, _, extra = predict(model, inputs, diffusion_kwargs={"inference_step": steps})
    traj = xyz.float().cpu().numpy().reshape(64, 3)
    sweep[steps] = {"xyz": traj, "cot": extra["cot"].reshape(-1)[0]}
reference = sweep[10]["xyz"]
assert len({v["cot"] for v in sweep.values()}) == 1, "CoT changed across the sweep"
print(f"  {'steps':>5} {'ADE vs 10-step':>15} {'ADE vs logged':>14} {'FDE vs logged':>14}")
rows = {}
for steps, item in sweep.items():
    vs_ref = np.linalg.norm(item["xyz"][:, :2] - reference[:, :2], axis=-1).mean()
    rows[steps] = {
        "ade_vs_10": float(vs_ref),
        "ade_logged": float(ade(item["xyz"], logged)[0]) if logged is not None else None,
        "fde_logged": float(fde(item["xyz"], logged)[0]) if logged is not None else None,
    }
    print(f"  {steps:5d} {vs_ref:15.3f} {rows[steps]['ade_logged']:14.3f} "
          f"{rows[steps]['fde_logged']:14.3f}")
print("  One CoT, one noise draw: these are single-sample numbers, not a benchmark.")

fig, axes = plt.subplots(1, 3, figsize=(17, 7))
cmap = plt.get_cmap("viridis")
ax = axes[0]
ax.plot(hist[:, 1], hist[:, 0], "-.", color="#52514e", label="history")
for k, step_xyz in enumerate(decoded):
    ax.plot(step_xyz[:, 1], step_xyz[:, 0], color=cmap(k / (len(decoded) - 1)), lw=1.5,
            label=f"t={times[k]:.1f}" if k in (0, 1, 2, 5, len(decoded) - 1) else None)
if logged is not None:
    ax.plot(logged[:, 1], logged[:, 0], "k--", lw=2, label="logged future")
ax.invert_xaxis()
ax.set_aspect("equal", adjustable="datalim")
ax.set_title("decoded x_t along the 10 Euler steps\n(t=0 is pure noise)")
ax.set_xlabel("y (m, left +)")
ax.set_ylabel("x (m, forward)")
ax.legend(fontsize=8)
seconds = np.arange(1, 65) * 0.1
for index, (ax, label) in enumerate(zip(axes[1:], ("normalized accel", "normalized curvature"),
                                        strict=True)):
    for k in range(all_x.shape[1]):
        ax.plot(seconds, all_x[0, k, :, index].float().cpu(), color=cmap(k / (all_x.shape[1] - 1)),
                lw=1.2)
    ax.set_title(f"{label} across steps (purple = noise, yellow = final)")
    ax.set_xlabel("time after t0 (s)")
    ax.grid(color="#e1e0d9")
fig.suptitle(f"Flow matching: clip {spec['clip_id'][:8]}, nav {spec['nav_text']!r}")
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "flow_matching.png", dpi=110)
print(f"wrote {(OUT / 'flow_matching.png').relative_to(OUT.parents[1])}")
save_json("flow_matching.json", {"times": times, "sweep": rows, "cot": sweep[10]["cot"]})
