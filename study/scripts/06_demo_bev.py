"""Session 1 / 6: the notebook demo as a script, plus where candidate diversity comes from.

    python study/scripts/06_demo_bev.py [--clip-id ... --t0-us ... --nav-text ...]

1. Camera grid: the 22 frames in model order (study/outputs/demo_cameras.png).
2. Six candidates (num_traj_sets=6, the notebook setting): six CoTs, six trajectories,
   BEV against the logged future, ADE/FDE per candidate and minADE.
3. Diversity, split by source:
     a. six independent CoTs, each with its own noise   (what the model offers)
     b. ONE fixed CoT, six different diffusion noises     (what the expert alone adds)
   If (b) is tight and (a) is wide, the reasoning text, not the noise, carries the choice.
Writes study/outputs/demo_bev.png, demo_cameras.png, demo.json.
"""

import argparse
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

from alpamayo2_edge.chat_template import CAMERA_DISPLAY_NAMES  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--clip-id")
parser.add_argument("--t0-us", type=int)
parser.add_argument("--nav-text")
parser.add_argument("--candidates", type=int, default=6)
parser.add_argument("--tag", default="demo", help="output file prefix")
args = parser.parse_args()
spec = example()
if args.clip_id:
    spec = {"clip_id": args.clip_id, "t0_us": args.t0_us or 5_100_000,
            "nav_text": args.nav_text or spec["nav_text"]}
elif args.nav_text:
    spec["nav_text"] = args.nav_text
N = args.candidates

data = load_sample(**spec)
logged = load_logged_future(spec["clip_id"], spec["t0_us"])
hist = data["ego_history_xyz"][0].numpy()
print(f"clip {spec['clip_id']}  t0 {spec['t0_us'] / 1e6:.1f} s  nav {spec['nav_text']!r}")

banner("1. Camera inputs")
seconds = (data["absolute_timestamps"].reshape(-1) - spec["t0_us"]).numpy() / 1e6
offsets = np.array([-2.0, -1.0, -0.5, -0.1, 0.0])
fig, axes = plt.subplots(6, 5, figsize=(16, 11))
for ax in axes.flat:
    ax.set_axis_off()
for row, (camera, name) in enumerate(CAMERA_DISPLAY_NAMES.items()):
    axes[row, 0].text(-0.05, 0.5, name, transform=axes[row, 0].transAxes, ha="right",
                      va="center", fontsize=9)
    for index in np.nonzero(data["camera_indices"].numpy() == camera)[0]:
        ax = axes[row, np.abs(offsets - seconds[index]).argmin()]
        ax.imshow(data["image_frames"][index, 0].permute(1, 2, 0).numpy())
        ax.set_title(f"{seconds[index]:+.2f} s", fontsize=8)
fig.suptitle(f"22 camera inputs, nav {spec['nav_text']!r}")
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / f"{args.tag}_cameras.png", dpi=80)
plt.close(fig)
print(f"wrote study/outputs/{args.tag}_cameras.png")

model, _ = load_edge()
inputs = prepare(model, data)
print("prompt length", inputs["tokenized_data"]["input_ids"].shape[1])

banner(f"2. {N} candidates (num_traj_sets={N}): independent CoT + trajectory each")
torch.cuda.reset_peak_memory_stats()
xyz, rot, extra = predict(model, inputs, num_traj_sets=N)  # [1, N, 1, 64, 3], [1, N, 1, 64, 3, 3]
peak = torch.cuda.max_memory_allocated() / 1e9
cands = xyz[0].reshape(-1, 64, 3).float().cpu().numpy()
cots = list(extra["cot"].reshape(-1))
records = []
ades = ade(cands, logged) if logged is not None else [None] * N
fdes = fde(cands, logged) if logged is not None else [None] * N
for k, (traj, cot) in enumerate(zip(cands, cots, strict=True)):
    records.append({"cot": cot, "end_xy": traj[-1, :2].tolist(),
                    "ade": None if ades[k] is None else float(ades[k]),
                    "fde": None if fdes[k] is None else float(fdes[k])})
    metric = "" if ades[k] is None else f"ADE {ades[k]:.2f} m  FDE {fdes[k]:.2f} m"
    print(f"[{k + 1}] {cot}\n     end ({traj[-1, 0]:.1f}, {traj[-1, 1]:.1f}) m   {metric}")
if logged is not None:
    print(f"minADE@{N} {min(ades):.3f} m   minFDE@{N} {min(fdes):.3f} m   "
          f"ADE of candidate 1 {ades[0]:.3f} m")
print(f"peak GPU memory {peak:.2f} GB")

banner("3. Diversity: CoT sampling vs diffusion noise")
spread_cot = np.linalg.norm(cands[:, -1, :2] - cands[:, -1, :2].mean(0), axis=-1).mean()
fixed = []
for noise_seed in range(N):
    torch.manual_seed(43)  # same text RNG -> same CoT
    generator = torch.Generator(device="cuda").manual_seed(1000 + noise_seed)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out, _, _, ex = model.sample_trajectories_from_data(
            inputs, top_p=0.98, temperature=0.6, max_generation_length=256,
            diffusion_kwargs={"inference_step": 10, "generator": generator}, return_extra=True)
    fixed.append(out.reshape(64, 3).float().cpu().numpy())
fixed_cot = ex["cot"].reshape(-1)[0]
fixed = np.stack(fixed)
spread_noise = np.linalg.norm(fixed[:, -1, :2] - fixed[:, -1, :2].mean(0), axis=-1).mean()
print(f"  (a) {N} CoTs:        mean endpoint distance to centroid {spread_cot:.2f} m")
print(f"  (b) 1 CoT, {N} noises: mean endpoint distance to centroid {spread_noise:.2f} m")
print(f"      fixed CoT: {fixed_cot}")

palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
fig, axes = plt.subplots(1, 2, figsize=(13, 8.5))
shown = np.concatenate([cands[..., :2].reshape(-1, 2), fixed[..., :2].reshape(-1, 2), hist[:, :2]]
                       + ([] if logged is None else [logged[:, :2]]))
lo, hi = shown.min(0) - 3, shown.max(0) + 3
for ax, trajs, title in ((axes[0], cands, f"(a) {N} independent CoTs"),
                         (axes[1], fixed, f"(b) one CoT, {N} noise seeds")):
    ax.plot(hist[:, 1], hist[:, 0], "-.", color="#52514e", lw=2, label="history")
    for k, traj in enumerate(trajs):
        ax.plot(traj[:, 1], traj[:, 0], color=palette[k % len(palette)], lw=2,
                label=f"candidate {k + 1}" + ("" if ades[0] is None or ax is axes[1]
                                               else f" ADE {ades[k]:.2f}"))
    if logged is not None:
        ax.plot(logged[:, 1], logged[:, 0], "k--", lw=2, label="logged future")
    ax.plot(0, 0, "k^", ms=10)
    ax.set_title(title)
    ax.set_xlabel("y (m, left +)")
    ax.grid(color="#e1e0d9")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim(hi[1], lo[1])  # left (+y) on the left
    ax.set_ylim(lo[0], hi[0])
    ax.set_aspect("equal", adjustable="box")
axes[0].set_ylabel("x (m, forward)")
fig.suptitle(f"clip {spec['clip_id'][:8]} t0={spec['t0_us'] / 1e6:.1f}s  nav {spec['nav_text']!r}")
fig.tight_layout()
fig.savefig(OUT / f"{args.tag}_bev.png", dpi=100)
print(f"wrote study/outputs/{args.tag}_bev.png")
save_json(f"{args.tag}.json", {
    "sample": spec, "candidates": records, "peak_gpu_gb": peak,
    "min_ade": None if logged is None else float(min(ades)),
    "spread_cot_m": float(spread_cot), "spread_noise_m": float(spread_noise),
    "fixed_cot": fixed_cot,
})
