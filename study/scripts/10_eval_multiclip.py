"""Session 7: a small open-loop evaluation over many PhysicalAI-AV clips.

    python study/scripts/10_eval_multiclip.py [--n-clips 20] [--candidates 6]
                                              [--nav-mode oracle|fixed --fixed-nav "..."]

Clips: a seeded random draw from the SDK's ``clip_index`` among valid clips with all six
Edge cameras and egomotion, plus the Alpamayo 1.5 repo's test clip (030c760c..., t0 5.1 s)
first so its number lines up with the one in alpamayo1.5/study.

Navigation: the dataset has no route text, so ``oracle`` derives one from the logged
future (``_common.derive_nav_text``: "Turn left/right in Nm", "Continue straight",
"Stop"). That tells the model *which way* the car went, never *where*; treat the result as
route-conditioned, as a deployed stack's navigation would be. ``fixed`` gives every clip
the same sentence (default "Turn right in 30m", i.e. mostly a wrong route), to see how much
the route text moves the plan.

Metrics (xy, ego frame at t0): ADE/FDE at 3 s and 6.4 s for candidate 1, and the
min over candidates. Writes study/outputs/eval_<tag>.json.
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    A15_GOLDEN_CLIP,
    OUT,
    ade,
    avdi,
    banner,
    derive_nav_text,
    fde,
    load_edge,
    load_logged_future,
    load_sample,
    predict,
    prepare,
    save_json,
)

parser = argparse.ArgumentParser()
parser.add_argument("--n-clips", type=int, default=20)
parser.add_argument("--seed", type=int, default=0, help="clip draw seed")
parser.add_argument("--t0-us", type=int, default=5_100_000)
parser.add_argument("--candidates", type=int, default=6)
parser.add_argument("--nav-mode", choices=["oracle", "fixed"], default="oracle")
parser.add_argument("--fixed-nav", default="Turn right in 30m")
parser.add_argument("--tag", default=None)
args = parser.parse_args()
tag = args.tag or f"{args.nav_mode}_k{args.candidates}"

CAMERAS = ["camera_cross_left_120fov", "camera_front_wide_120fov", "camera_cross_right_120fov",
           "camera_rear_left_70fov", "camera_rear_right_70fov", "camera_front_tele_30fov"]
interface = avdi()
presence = interface.feature_presence
valid = interface.clip_index["clip_is_valid"] & presence[CAMERAS + ["egomotion"]].all(axis=1)
pool = sorted(valid[valid].index.tolist())
rng = np.random.default_rng(args.seed)
clips = [A15_GOLDEN_CLIP[0]] + [pool[i] for i in rng.choice(len(pool), args.n_clips, replace=False)]
print(f"{len(pool):,} eligible clips; evaluating {len(clips)} at t0 = {args.t0_us / 1e6} s")

model, _ = load_edge()
rows, plots = [], []
for index, clip_id in enumerate(clips):
    banner(f"[{index + 1}/{len(clips)}] {clip_id}")
    start = time.perf_counter()
    try:
        logged = load_logged_future(clip_id, args.t0_us)
        if logged is None:
            print("  skipped: clip too short for a 6.4 s future")
            continue
        nav = derive_nav_text(logged) if args.nav_mode == "oracle" else args.fixed_nav
        data = load_sample(clip_id, args.t0_us, nav)
    except Exception as exc:  # noqa: BLE001 - a broken or short clip should not stop the sweep
        print(f"  skipped: {type(exc).__name__}: {exc}")
        continue
    load_s = time.perf_counter() - start
    inputs = prepare(model, data)
    torch.cuda.synchronize()
    start = time.perf_counter()
    xyz, _, extra = predict(model, inputs, num_traj_sets=args.candidates)
    torch.cuda.synchronize()
    infer_s = time.perf_counter() - start
    cands = xyz[0].reshape(-1, 64, 3).float().cpu().numpy()
    a64, f64 = ade(cands, logged), fde(cands, logged)
    a30, f30 = ade(cands, logged, 30), fde(cands, logged, 30)
    speed = float(np.linalg.norm(data["ego_history_xyz"][0, -1] - data["ego_history_xyz"][0, -2]) * 10)
    row = {
        "clip_id": clip_id, "nav": nav, "speed_t0": speed,
        "logged_end_xy": logged[-1, :2].tolist(),
        "ade_6.4_first": float(a64[0]), "fde_6.4_first": float(f64[0]),
        "min_ade_6.4": float(a64.min()), "min_fde_6.4": float(f64.min()),
        "ade_3_first": float(a30[0]), "min_ade_3": float(a30.min()),
        "cot_first": extra["cot"].reshape(-1)[0], "load_s": load_s, "infer_s": infer_s,
    }
    rows.append(row)
    plots.append((clip_id, nav, cands, logged, data["ego_history_xyz"][0].numpy(), row))
    print(f"  nav {nav!r:24s} v0 {speed:5.2f} m/s   ADE6.4 first {a64[0]:.3f}  "
          f"min@{args.candidates} {a64.min():.3f}   ADE3 first {a30[0]:.3f}  ({infer_s:.1f} s)")
    print(f"  CoT: {row['cot_first']}")

banner(f"Summary ({tag}, {len(rows)} clips)")
summary = {}
for key in ("ade_3_first", "min_ade_3", "ade_6.4_first", "min_ade_6.4", "fde_6.4_first",
            "min_fde_6.4"):
    values = np.array([r[key] for r in rows])
    summary[key] = {"mean": float(values.mean()), "median": float(np.median(values))}
    print(f"  {key:16s} mean {values.mean():6.3f} m   median {np.median(values):6.3f} m")
nominal = sum(r["cot_first"].strip() == "<NOMINAL>" for r in rows)
print(f"  CoT of candidate 1 is '<NOMINAL>' (adaptive reasoning skipped) on {nominal}/{len(rows)} clips")
summary["nominal_fraction"] = nominal / max(len(rows), 1)
golden = next((r for r in rows if r["clip_id"] == A15_GOLDEN_CLIP[0]), None)
if golden:
    print(f"  A1.5 golden clip: Edge ADE6.4 first {golden['ade_6.4_first']:.3f} m "
          f"(nav {golden['nav']!r}); alpamayo1.5/study reports 0.373 m for A1.5, n=1, no route")
save_json(f"eval_{tag}.json", {"args": vars(args), "summary": summary, "rows": rows})
# Per-clip candidates, logged future and history, for 11_compare_with_a15.py.
np.savez_compressed(OUT / f"eval_{tag}_trajs.npz", **{
    f"{clip_id}/{name}": value
    for clip_id, _, cands, logged, hist, _ in plots
    for name, value in (("cands", cands), ("gt", logged), ("hist", hist))
})

columns = 7
fig, axes = plt.subplots(-(-len(plots) // columns), columns, figsize=(3.1 * columns, 4.2 * -(-len(plots) // columns)))
for ax in axes.flat:
    ax.set_axis_off()
for ax, (clip_id, nav, cands, logged, hist, row) in zip(axes.flat, plots):
    ax.set_axis_on()
    ax.plot(hist[:, 1], hist[:, 0], "-.", color="#52514e", lw=1.2)
    for k, traj in enumerate(cands):
        ax.plot(traj[:, 1], traj[:, 0], color="#2a78d6" if k == 0 else "#9dbfe8", lw=1.6 if k == 0 else 1)
    ax.plot(logged[:, 1], logged[:, 0], "k--", lw=1.4)
    span = max(np.ptp(np.concatenate([cands[..., 0].ravel(), logged[:, 0], hist[:, 0]])), 20) / 2 + 3
    cx = (max(cands[..., 0].max(), logged[:, 0].max()) + min(hist[:, 0].min(), 0)) / 2
    ax.set_ylim(cx - span, cx + span)
    ax.set_xlim(span * 0.6, -span * 0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=6)
    cot = row["cot_first"]
    ax.set_title(f"{clip_id[:8]} v0={row['speed_t0']:.0f} m/s\nnav: {nav}\n"
                 f"ADE {row['ade_6.4_first']:.2f} / min {row['min_ade_6.4']:.2f} m\n"
                 f"{cot[:38] + ('...' if len(cot) > 38 else '')}", fontsize=6.5, loc="left")
fig.suptitle(f"Open-loop eval ({tag}): candidate 1 dark blue, others light, logged future dashed", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / f"eval_{tag}.png", dpi=90)
print(f"wrote study/outputs/eval_{tag}.png")
