"""Session 5c: does the trajectory follow the reasoning? Swap the CoT, keep everything else.

    python study/scripts/13_cot_intervention.py [--seeds 6]

Normally step 1 (the VLM samples a CoT) and step 2 (the expert denoises a trajectory while
attending to prompt + CoT) run back to back. Here step 1 is replaced: the CoT text is
written by hand and forced into the cache (teacher forcing), then step 2 runs exactly as in
``sample_trajectories_from_data`` (``_sample_actions``, ``action_to_traj``).

Held fixed for every CoT: the 22 images, the ego history, the route sentence
("Turn right in 30m"), and the diffusion noise seeds. Only the reasoning text changes, so
any change in the trajectory is caused by the text.

Sanity check: with the model's own CoT and the CLI's seed, the result must match
``inference.predict`` (up to bf16 rounding, since the CoT is cached in one forward here
instead of token by token).

Writes study/outputs/cot_intervention.png and cot_intervention.json.
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
    banner,
    example,
    load_edge,
    load_logged_future,
    load_sample,
    predict,
    prepare,
    save_json,
)

from alpamayo2_edge.models.expert_utils import build_expert_pos_ids_and_attn_mask  # noqa: E402
from alpamayo2_edge.models.utils import fuse_traj_tokens  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, default=6, help="diffusion noise seeds per CoT")
parser.add_argument("--grid-only", action="store_true", help="skip sections 0-1, run the grid")
parser.add_argument("--grid-routes", nargs="+", default=["Turn right in 30m", "Turn left in 30m"],
                    help="route sentences (rows of the grid)")
parser.add_argument("--grid-cots", nargs="+", default=["own", "turn left", "stop"],
                    help="forced reasoning (columns): 'own', a key of COTS, or literal text")
parser.add_argument("--tag", default="", help="suffix for the grid's output files")
args = parser.parse_args()

OWN = "Slow down and turn right at the intersection to follow our route, as the lead vehicle has cleared our path."
COTS = {
    "model's own CoT": OWN,
    "stop": "Stop at the intersection and wait, as crossing traffic has not cleared our path.",
    "turn left": "Slow down and turn left at the intersection to follow our route, as the lead vehicle has cleared our path.",
    "go straight": "Accelerate and proceed straight through the intersection, as the road ahead is clear.",
    "<NOMINAL>": "<NOMINAL>",
    "empty": "",
}

model, _ = load_edge()
spec = example()  # README clip, t0 = 4.0 s, "Turn right in 30m"
data = load_sample(**spec)
inputs = prepare(model, data)
logged = load_logged_future(spec["clip_id"], spec["t0_us"])
cfg, tok = model.config, model.tokenizer
anchor = cfg.traj_ids["future_start"]

# Reference: the normal pipeline with the CLI seed (its CoT is OWN, see 04_walkthrough).
ref_xyz, _, ref_extra = predict(model, inputs)
assert ref_extra["cot"].reshape(-1)[0] == OWN, ref_extra["cot"].reshape(-1)[0]

history_xyz = inputs["ego_history_xyz"][:, -1]
history_rot = inputs["ego_history_rot"][:, -1]


def prompt_for(route: str):
    """Tokenized prompt (history tokens fused) for the same sample with another route."""
    prepared = inputs if route == spec["nav_text"] else prepare(model, {**data, "nav_text": [route]})
    tokenized = prepared["tokenized_data"]
    ids = fuse_traj_tokens(model.history_traj_tokenizer, model.future_traj_tokenizer,
                           tokenized["input_ids"], {k: prepared[k] for k in ("ego_history_xyz",
                                                                             "ego_history_rot")},
                           cfg.traj_ids)
    return tokenized, ids


@torch.inference_mode()
def trajectories(cot: str, seeds: list[int], route: str = spec["nav_text"]) -> np.ndarray:
    """Force ``cot`` + newline + anchor after the prompt, then run the expert per seed."""
    td, prompt_ids = prompt_for(route)
    cot_ids = tok(cot + "\n", add_special_tokens=False)["input_ids"] + [anchor]
    ids = torch.cat([prompt_ids, torch.tensor([cot_ids], device=prompt_ids.device)], dim=1)
    types = torch.cat([td["mm_token_type_ids"], torch.zeros_like(ids[:, prompt_ids.shape[1]:])], 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True,
                              pixel_values=td["pixel_values"], image_grid_thw=td["image_grid_thw"],
                              mm_token_type_ids=types)
        cache, length = out.past_key_values, ids.shape[1]
        rope_delta = model.vlm.model.rope_deltas
        positions, mask = build_expert_pos_ids_and_attn_mask(
            offset=torch.tensor([length], device=ids.device), rope_deltas=rope_delta,
            kv_cache_seq_len=length, n_diffusion_tokens=64, b_star=1, device=ids.device,
            prefix_mask=torch.ones_like(prompt_ids))
        results = []
        for seed in seeds:
            generator = torch.Generator(device="cuda").manual_seed(seed)
            action = model._sample_actions(cache, length, positions, mask, 1, ids.device,
                                           {"inference_step": 10, "generator": generator})
            xyz, _ = model.expert.action_space.action_to_traj(action, history_xyz, history_rot)
            results.append(xyz[0].float().cpu().numpy())
    return np.stack(results)


seeds = list(range(1000, 1000 + args.seeds))
rows, diff = {}, None
if not args.grid_only:
    banner("0. Sanity check: forced own CoT vs the normal pipeline (seed 43)")
    own = trajectories(OWN, [43])[0]
    diff = float(np.abs(own - ref_xyz.reshape(64, 3).float().cpu().numpy()).max())
    print(f"  max |dxyz| = {diff * 100:.2f} cm  (bf16: CoT cached in one forward vs token by token)")

    banner(f"1. Same images, history, route and noise; only the reasoning text differs "
           f"({args.seeds} noise seeds each)")
    trajs = {}
    print(f"  {'reasoning':16s} {'endpoint x':>10s} {'endpoint y':>10s} {'distance':>9s} "
          f"{'speed@6.4s':>10s} {'spread':>7s}")
    for name, cot in COTS.items():
        t = trajectories(cot, seeds)
        trajs[name] = t
        end = t[:, -1, :2]
        dist = np.linalg.norm(np.diff(np.concatenate([np.zeros((len(t), 1, 3)), t], 1)[..., :2],
                                      axis=1), axis=-1).sum(-1)
        speed = np.linalg.norm(t[:, -1, :2] - t[:, -2, :2], axis=-1) * 10
        spread = np.linalg.norm(end - end.mean(0), axis=-1).mean()
        rows[name] = {"cot": cot, "end_x": float(end[:, 0].mean()),
                      "end_y": float(end[:, 1].mean()), "distance_m": float(dist.mean()),
                      "end_speed_mps": float(speed.mean()), "spread_m": float(spread)}
        r = rows[name]
        print(f"  {name:16s} {r['end_x']:10.1f} {r['end_y']:10.1f} {r['distance_m']:9.1f} "
              f"{r['end_speed_mps']:10.2f} {r['spread_m']:7.2f}")
    print("  (means over the noise seeds; y < 0 is right; spread = mean endpoint distance to the "
          "centroid within one CoT)")

    hist = data["ego_history_xyz"][0].numpy()
    colors = {"model's own CoT": "#2a78d6", "stop": "#d62728", "turn left": "#1baf7a",
              "go straight": "#eda100", "<NOMINAL>": "#8a4fbf", "empty": "#7a7a7a"}
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(hist[:, 1], hist[:, 0], "-.", color="#52514e", lw=2, label="history")
    ax.plot(logged[:, 1], logged[:, 0], "k--", lw=2, label="logged future (what the car did)")
    for name, t in trajs.items():
        for k, traj in enumerate(t):
            ax.plot(traj[:, 1], traj[:, 0], color=colors[name], lw=1.6, alpha=0.85,
                    label=f'"{COTS[name][:60]}{"..." if len(COTS[name]) > 60 else ""}"'
                    if k == 0 else None)
    ax.plot(0, 0, "k^", ms=10)
    ax.invert_xaxis()
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("y (m), left +")
    ax.set_ylabel("x (m), forward")
    ax.grid(color="#e1e0d9")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title(f"Forced reasoning, everything else fixed (route {spec['nav_text']!r}, "
                 f"{args.seeds} noise seeds per text)", fontsize=10)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "cot_intervention.png", dpi=110)
    print("wrote study/outputs/cot_intervention.png")

banner("2. Who decides what: route sentence x forced reasoning")
grid_cots = {}
for item in args.grid_cots:
    grid_cots["own: 'Slow down and turn right ...'" if item == "own" else f"'{item}'"] = (
        OWN if item == "own" else COTS.get(item, item))
grid, grid_trajs = {}, {}
print(f"  {'':28s}" + "".join(f"{name[:30]:>32s}" for name in grid_cots))
for route in args.grid_routes:
    cells = []
    for name, cot in grid_cots.items():
        t = trajectories(cot, seeds, route)
        grid_trajs[(route, name)] = t
        end = t[:, -1, :2].mean(0)
        steps = np.diff(np.concatenate([np.zeros((len(t), 1, 3)), t], 1)[..., :2], axis=1)
        profile = np.linalg.norm(steps, axis=-1) * 10  # [seeds, 64] speed in m/s
        speed, distance = float(profile[:, -1].mean()), float(profile.sum(1).mean() / 10)
        grid[f"{route} | {name}"] = {"cot": cot, "end_x": float(end[0]), "end_y": float(end[1]),
                                     "end_speed_mps": speed, "distance_m": distance,
                                     "speed_profile_mps": profile.mean(0).tolist()}
        cells.append(f"({end[0]:5.1f},{end[1]:5.1f}) {distance:4.1f}m {speed:4.1f}m/s")
    print(f"  route {route!r:22s}" + "".join(f"{c:>32s}" for c in cells))
print("  cells: mean endpoint (x forward, y left; y < 0 = right), distance covered in 6.4 s and "
      "speed at 6.4 s, over the noise seeds")

palette = ["#2a78d6", "#1baf7a", "#d62728", "#eda100", "#8a4fbf", "#7a7a7a"]
hist = data["ego_history_xyz"][0].numpy()
fig, axes = plt.subplots(1, len(args.grid_routes), figsize=(max(6 * len(args.grid_routes), 10), 7.5),
                         squeeze=False)
for ax, route in zip(axes[0], args.grid_routes, strict=True):
    ax.plot(hist[:, 1], hist[:, 0], "-.", color="#52514e", lw=2)
    for i, name in enumerate(grid_cots):
        for k, traj in enumerate(grid_trajs[(route, name)]):
            ax.plot(traj[:, 1], traj[:, 0], color=palette[i % 6], lw=1.5, alpha=0.85,
                    label=f"reasoning {name[:42]}" if k == 0 else None)
    ax.plot(0, 0, "k^", ms=9)
    ax.set_xlim(14, -14)
    ax.set_ylim(-15, 40)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#e1e0d9")
    ax.set_title(f"route in the prompt: {route!r}", fontsize=11)
    ax.set_xlabel("y (m), left +")
axes[0, 0].set_ylabel("x (m), forward")
axes[0, 0].legend(fontsize=8, loc="lower left")
fig.suptitle(f"Same images, history and noise ({args.seeds} seeds per cell); "
             "route sentence (panels) x forced reasoning (colours)", fontsize=11)
fig.tight_layout()
suffix = f"_{args.tag}" if args.tag else ""
fig.savefig(OUT / f"cot_intervention_grid{suffix}.png", dpi=105)
print(f"wrote study/outputs/cot_intervention_grid{suffix}.png")

v0 = float(np.linalg.norm(hist[-1, :2] - hist[-2, :2]) * 10)
seconds = np.arange(0, 65) * 0.1
fig, axes = plt.subplots(1, len(args.grid_routes), figsize=(max(6 * len(args.grid_routes), 10), 4.8),
                         squeeze=False, sharey=True)
for ax, route in zip(axes[0], args.grid_routes, strict=True):
    for i, name in enumerate(grid_cots):
        profile = grid[f"{route} | {name}"]["speed_profile_mps"]
        ax.plot(seconds, [v0, *profile], color=palette[i % 6], lw=2,
                label=f"reasoning {name[:42]}")
    ax.set_title(f"route in the prompt: {route!r}", fontsize=11)
    ax.set_xlabel("seconds after t0")
    ax.grid(color="#e1e0d9")
axes[0, 0].set_ylabel("speed (m/s), mean over noise seeds")
axes[0, 0].legend(fontsize=8)
fig.suptitle("Planned speed over the 6.4 s horizon: route sentence (panels) x forced reasoning "
             "(colours)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / f"cot_intervention_speed{suffix}.png", dpi=105)
print(f"wrote study/outputs/cot_intervention_speed{suffix}.png")
save_json(f"cot_intervention{suffix}.json", {"sample": spec, "seeds": seeds,
                                             "sanity_max_diff_m": diff, "rows": rows,
                                             "grid": grid})
