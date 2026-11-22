"""Session 6: how much does the navigation sentence steer the plan?

    python study/scripts/07_nav_ablation.py [--clip-id ... --t0-us ...] [--navs "A" "B" ...]

Same images, same ego history, same seed; only the ``Navigation input is: ...`` line
changes. In Edge the route is plain prompt text (no classifier-free guidance, no route
tokens), so any effect has to flow through the CoT and the KV cache the expert reads.
Writes study/outputs/nav_ablation.png and nav_ablation.json.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    OUT,
    banner,
    example,
    load_edge,
    load_logged_future,
    load_sample,
    predict,
    prepare,
    save_json,
)

parser = argparse.ArgumentParser()
parser.add_argument("--clip-id")
parser.add_argument("--t0-us", type=int, default=5_100_000)
parser.add_argument("--navs", nargs="+",
                    default=["Turn right in 30m", "Turn left in 30m", "Continue straight", "Stop"])
parser.add_argument("--candidates", type=int, default=3)
parser.add_argument("--tag", default="nav_ablation")
args = parser.parse_args()
spec = example()
clip_id, t0_us = (args.clip_id, args.t0_us) if args.clip_id else (spec["clip_id"], spec["t0_us"])

model, _ = load_edge()
logged = load_logged_future(clip_id, t0_us)
results = {}
for nav in args.navs:
    banner(f"nav = {nav!r}")
    data = load_sample(clip_id, t0_us, nav)
    xyz, _, extra = predict(model, prepare(model, data), num_traj_sets=args.candidates)
    trajs = xyz[0].reshape(-1, 64, 3).float().cpu().numpy()
    cots = list(extra["cot"].reshape(-1))
    for k, (traj, cot) in enumerate(zip(trajs, cots, strict=True)):
        print(f"  [{k + 1}] end ({traj[-1, 0]:5.1f}, {traj[-1, 1]:5.1f}) m  {cot}")
    results[nav] = {"trajs": trajs, "cots": cots}

hist = load_sample(clip_id, t0_us, args.navs[0])["ego_history_xyz"][0].numpy()
fig, axes = plt.subplots(1, len(args.navs), figsize=(4.8 * len(args.navs), 7.5))
axes = np.atleast_1d(axes)
shown = np.concatenate([r["trajs"][..., :2].reshape(-1, 2) for r in results.values()]
                       + [hist[:, :2]] + ([] if logged is None else [logged[:, :2]]))
lo, hi = shown.min(0) - 3, shown.max(0) + 3
# Keep at least 12 m either side so left and right turns are both readable.
lo[1], hi[1] = min(lo[1], -12.0), max(hi[1], 12.0)
palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
for ax, (nav, res) in zip(axes, results.items(), strict=True):
    ax.plot(hist[:, 1], hist[:, 0], "-.", color="#52514e", lw=2)
    if logged is not None:
        ax.plot(logged[:, 1], logged[:, 0], "k--", lw=1.5, label="logged future")
    for k, traj in enumerate(res["trajs"]):
        ax.plot(traj[:, 1], traj[:, 0], color=palette[k % 6], lw=2, label=f"candidate {k + 1}")
    ax.plot(0, 0, "k^", ms=9)
    ax.set_xlim(hi[1], lo[1])
    ax.set_ylim(lo[0], hi[0])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#e1e0d9")
    ax.set_title(f"nav: {nav!r}\n" + "\n".join(
        f"{k + 1}: {c[:46]}{'...' if len(c) > 46 else ''}" for k, c in enumerate(res["cots"])),
        fontsize=7.5, loc="left")
    ax.set_xlabel("y (m, left +)")
axes[0].set_ylabel("x (m, forward)")
axes[0].legend(fontsize=8, loc="lower left")
fig.suptitle(f"Navigation ablation - clip {clip_id[:8]} t0={t0_us / 1e6:.1f}s "
             "(same images, history and seed)")
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / f"{args.tag}.png", dpi=100)
print(f"wrote study/outputs/{args.tag}.png")
save_json(f"{args.tag}.json", {
    "clip_id": clip_id, "t0_us": t0_us,
    "results": {nav: {"cots": r["cots"], "end_xy": r["trajs"][:, -1, :2].tolist()}
                for nav, r in results.items()},
})
