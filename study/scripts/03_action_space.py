"""Session 5: the action space - what the expert actually denoises, and why "course".

    python study/scripts/03_action_space.py

No model weights needed: the action spaces are built from the checkpoint's config.json.
On the README clip's logged future (a right turn) this measures:
  1. encode -> decode round trip of the GA course-heading space vs the yaw-heading
     space it replaced (positions and headings)
  2. the (accel, curvature) controls and the speed profile the expert sees
  3. the 3000-bin discrete tokenizer used for training-time trajectory tokens
  4. what the bf16 cast of the normalization constants does to a decoded plan
Writes study/outputs/action_space.png and study/outputs/action_space.json.
"""

import copy
import json
import sys
from pathlib import Path

import hydra.utils as hyu
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    MODEL_DIR,
    OUT,
    banner,
    example,
    load_logged_future_pose,
    load_sample,
    save_json,
)

from alpamayo2_edge.config import normalize_checkpoint_config  # noqa: E402
from alpamayo2_edge.geometry.rotation import so3_to_yaw_torch  # noqa: E402

raw = normalize_checkpoint_config(json.loads((MODEL_DIR / "config.json").read_text()))
course_cfg = raw["expert_config"]["action_space_cfg"]
yaw_cfg = {k: v for k, v in course_cfg.items() if k != "slip_gradient"}
yaw_cfg["_target_"] = "alpamayo2_edge.action_space.UnicycleAccelCurvatureActionSpace"
course_space = hyu.instantiate(course_cfg)
yaw_space = hyu.instantiate(yaw_cfg)

spec = example()
data = load_sample(**spec)
hist_xyz = data["ego_history_xyz"].double()  # [1, 20, 3], t0 frame
hist_rot = data["ego_history_rot"].double()
fut_xyz, fut_rot = load_logged_future_pose(spec["clip_id"], spec["t0_us"])
fut_xyz = torch.from_numpy(fut_xyz)[None].double()
fut_rot = torch.from_numpy(fut_rot)[None].double()
for space in (course_space, yaw_space):
    space.double()

banner("1. Round trip: logged future -> (accel, curvature) -> trajectory")
results = {}
for name, space in (("course (GA)", course_space), ("yaw (EA)", yaw_space)):
    t0_states = space.estimate_t0_states(hist_xyz, hist_rot)
    action, states = space.traj_to_action(hist_xyz, hist_rot, fut_xyz, fut_rot,
                                          t0_states=t0_states, output_all_states=True)
    xyz, rot = space.action_to_traj(action, hist_xyz, hist_rot, t0_states=t0_states)
    err = (xyz - fut_xyz)[..., :2].norm(dim=-1)[0]
    yaw_err = torch.rad2deg((so3_to_yaw_torch(rot) - so3_to_yaw_torch(fut_rot)).abs())[0]
    results[name] = {
        "action": action[0].numpy(),
        "xyz": xyz[0].numpy(),
        "v": states[0, :, 0].numpy(),
        "ade_m": err.mean().item(),
        "fde_m": err[-1].item(),
        "max_err_m": err.max().item(),
        "max_yaw_err_deg": yaw_err.max().item(),
        "t0": {k: v.item() for k, v in t0_states.items()},
    }
    print(f"  {name:12s} ADE {err.mean() * 100:6.2f} cm  FDE {err[-1] * 100:6.2f} cm  "
          f"max {err.max() * 100:6.2f} cm   max |yaw err| {yaw_err.max():.3f} deg   "
          f"t0 states {results[name]['t0']}")
print("  The course space integrates along the direction the rear axle actually travels, so the")
print("  decoded path stays on the driven one; integrating along body yaw cuts inside the turn.")

banner("2. What the expert denoises (normalized controls, course space)")
action = results["course (GA)"]["action"]
accel = action[:, 0] * course_cfg["accel_std"] + course_cfg["accel_mean"]
kappa = action[:, 1] * course_cfg["curvature_std"] + course_cfg["curvature_mean"]
print(f"  action shape {action.shape}  normalized range accel [{action[:, 0].min():.2f}, "
      f"{action[:, 0].max():.2f}]  kappa [{action[:, 1].min():.2f}, {action[:, 1].max():.2f}]")
print(f"  physical accel  [{accel.min():.2f}, {accel.max():.2f}] m/s^2    curvature "
      f"[{kappa.min():.4f}, {kappa.max():.4f}] 1/m  (min radius {1 / np.abs(kappa).max():.1f} m)")
v = results["course (GA)"]["v"]
print(f"  speed at t0 {v[0]:.2f} m/s, min {v.min():.2f}, at 6.4 s {v[-1]:.2f} m/s")

banner("3. Discrete future tokenizer (training-time <i1000>..<i3999> tokens)")
tokenizer = hyu.instantiate(copy.deepcopy(raw["future_traj_tokenizer_cfg"]), _recursive_=False)
tokenizer.action_space.double()
codes = tokenizer.encode(hist_xyz, hist_rot, fut_xyz, fut_rot)
dec_xyz, _, _ = tokenizer.decode(hist_xyz, hist_rot, codes)
tok_err = (dec_xyz - fut_xyz)[..., :2].norm(dim=-1)[0]
print(f"  {codes.shape[1]} tokens (64 waypoints x 2 controls), bins 0..{tokenizer.num_bins - 1} "
      f"over normalized [-10, 10]: step {20 / (tokenizer.num_bins - 1):.4f} sigma")
print(f"  first 8 codes {codes[0, :8].tolist()}")
print(f"  decode(encode) ADE {tok_err.mean() * 100:.2f} cm  FDE {tok_err[-1] * 100:.2f} cm")
print("  These tokens exist in the vocabulary (tokens_per_future_traj=128) but inference masks")
print("  them: the continuous expert, not the LM head, produces the trajectory.")

banner("4. bf16 normalization constants (what load_model actually holds)")
space32 = hyu.instantiate(course_cfg).double()
space16 = hyu.instantiate(course_cfg).to(torch.bfloat16)
a = torch.from_numpy(action)[None]
xyz32, _ = space32.action_to_traj(a.double(), hist_xyz, hist_rot)
space16_constants = {n: b.item() for n, b in space16.named_buffers()}
space16.double()  # keep the bf16-rounded values, compute in float64
xyz16, _ = space16.action_to_traj(a.double(), hist_xyz, hist_rot)
bf16_shift = (xyz16 - xyz32)[..., :2].norm(dim=-1)[0]
print("  config  :", {k: round(course_cfg[k], 6) for k in
                      ("accel_mean", "accel_std", "curvature_mean", "curvature_std")})
print("  in bf16 :", {k: round(v, 6) for k, v in space16_constants.items()})
print(f"  same normalized action decoded with bf16 vs fp32 constants: max shift "
      f"{bf16_shift.max() * 100:.2f} cm, at 6.4 s {bf16_shift[-1] * 100:.2f} cm")

fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
ax = axes[0]
ax.plot(hist_xyz[0, :, 1], hist_xyz[0, :, 0], "-.", color="#52514e", label="history (-1.9..0 s)")
ax.plot(fut_xyz[0, :, 1], fut_xyz[0, :, 0], "k--", lw=2, label="logged future")
for (name, res), color in zip(results.items(), ("#2a78d6", "#eb6834"), strict=True):
    ax.plot(res["xyz"][:, 1], res["xyz"][:, 0], color=color,
            label=f"{name} round trip (FDE {res['fde_m'] * 100:.1f} cm)")
ax.invert_xaxis()
ax.set_aspect("equal", adjustable="datalim")
ax.set_xlabel("y (m, left +)")
ax.set_ylabel("x (m, forward)")
ax.set_title("encode -> decode of the logged future")
ax.legend(fontsize=8)
t = np.arange(1, 65) * 0.1
axes[1].plot(t, accel, color="#2a78d6")
axes[1].set_title("acceleration (m/s$^2$) - action[:, 0]")
axes[2].plot(t, kappa, color="#1baf7a")
axes[2].set_title("curvature (1/m) - action[:, 1]")
axes[3].plot(np.arange(0, 64) * 0.1, v, color="#eda100")
axes[3].set_title("speed (m/s), integrated")
for ax in axes[1:]:
    ax.set_xlabel("time after t0 (s)")
    ax.grid(color="#e1e0d9")
fig.suptitle(f"Action space on clip {spec['clip_id'][:8]} t0={spec['t0_us'] / 1e6:.1f}s "
             f"(nav: {spec['nav_text']!r})")
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "action_space.png", dpi=110)
print(f"wrote {(OUT / 'action_space.png').relative_to(OUT.parents[1])}")

save_json("action_space.json", {
    "roundtrip": {k: {kk: vv for kk, vv in v.items() if kk not in ("action", "xyz", "v")}
                  for k, v in results.items()},
    "physical_ranges": {"accel": [accel.min(), accel.max()], "kappa": [kappa.min(), kappa.max()]},
    "discrete_tokenizer": {"ade_m": tok_err.mean().item(), "fde_m": tok_err[-1].item()},
    "bf16_constants": space16_constants,
    "bf16_shift_max_m": bf16_shift.max().item(),
})
