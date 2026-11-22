"""Session 5h: how much does the reasoning help the trajectory? Remove it and compare.

    python study/scripts/16_reasoning_ablation.py [--candidates 6]

Clips: the 21 of 10_eval_multiclip (read from study/outputs/eval_oracle_k6.json), each
with its oracle route sentence. Run 3-7a first.

Three conditions per clip, sharing the images, history, route and the diffusion noise
seed of each candidate k (1000 + k). Only the text forced between the prompt and
<|traj_future_start|> differs (teacher forcing, as in 13_cot_intervention.py):
  own        the K CoTs the model samples itself (CLI settings, seed 43), forced back in
  <NOMINAL>  the model's own "nothing to explain" output, for every candidate
  empty      no reasoning at all: newline + anchor right after the prompt
Edge has no official "no reasoning" mode; "empty" is the closest thing to Qwen-Drive's
DIRECT_PLANNING, but this checkpoint was never trained on it.

Reported:
  * ADE/FDE against the logged future: candidate 1, mean over K, min over K
  * plan shift: mean distance between the own-CoT plan and the <NOMINAL> plan of the SAME
    noise seed, next to the noise floor (two seeds, same text). Shift near the floor means
    the reasoning did not move the plan.
  * reasoning cost in tokens
Writes study/outputs/reasoning_ablation.json, reasoning_ablation.png, reasoning_ablation_bev.png.
"""

import argparse
import json
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
    fde,
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
parser.add_argument("--candidates", type=int, default=6)
parser.add_argument("--eval-json", default=str(OUT / "eval_oracle_k6.json"),
                    help="clip list + routes from 10_eval_multiclip")
args = parser.parse_args()
K = args.candidates
SEEDS = [1000 + k for k in range(K)]
NOMINAL = "<NOMINAL>"

source = json.loads(Path(args.eval_json).read_text())
t0_us = source["args"]["t0_us"]
clips = [(row["clip_id"], row["nav"]) for row in source["rows"]]

model, _ = load_edge()
cfg, tok = model.config, model.tokenizer
anchor = cfg.traj_ids["future_start"]


@torch.inference_mode()
def plans(inputs: dict, cot: str, seeds: list[int]) -> np.ndarray:
    """Force ``cot`` + newline + anchor after the prompt, then run the expert once per seed."""
    td = inputs["tokenized_data"]
    prompt = fuse_traj_tokens(model.history_traj_tokenizer, model.future_traj_tokenizer,
                              td["input_ids"], {k: inputs[k] for k in ("ego_history_xyz",
                                                                       "ego_history_rot")},
                              cfg.traj_ids)
    forced = tok(cot + "\n", add_special_tokens=False)["input_ids"] + [anchor]
    ids = torch.cat([prompt, torch.tensor([forced], device=prompt.device)], dim=1)
    types = torch.cat([td["mm_token_type_ids"], torch.zeros_like(ids[:, prompt.shape[1]:])], 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True,
                              pixel_values=td["pixel_values"], image_grid_thw=td["image_grid_thw"],
                              mm_token_type_ids=types)
        cache, length = out.past_key_values, ids.shape[1]
        positions, mask = build_expert_pos_ids_and_attn_mask(
            offset=torch.tensor([length], device=ids.device),
            rope_deltas=model.vlm.model.rope_deltas, kv_cache_seq_len=length,
            n_diffusion_tokens=64, b_star=1, device=ids.device,
            prefix_mask=torch.ones_like(prompt))
        result = []
        for seed in seeds:
            generator = torch.Generator(device="cuda").manual_seed(seed)
            action = model._sample_actions(cache, length, positions, mask, 1, ids.device,
                                           {"inference_step": 10, "generator": generator})
            xyz, _ = model.expert.action_space.action_to_traj(
                action, inputs["ego_history_xyz"][:, -1], inputs["ego_history_rot"][:, -1])
            result.append(xyz[0].reshape(64, 3).float().cpu().numpy())
    return np.stack(result)


def path_gap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean xy distance over the 64 steps between paired plans [N, 64, 3] -> [N]."""
    return np.linalg.norm(a[..., :2] - b[..., :2], axis=-1).mean(-1)


CONDITIONS = ("own", NOMINAL, "empty")
rows, kept = [], []
for index, (clip_id, nav) in enumerate(clips):
    banner(f"[{index + 1}/{len(clips)}] {clip_id}  route {nav!r}")
    data = load_sample(clip_id, t0_us, nav)
    logged = load_logged_future(clip_id, t0_us)
    inputs = prepare(model, data)
    _, _, extra = predict(model, inputs, num_traj_sets=K)  # only the K sampled CoTs are used
    cots = [str(c) for c in extra["cot"].reshape(-1)]

    trajs = {"own": np.zeros((K, 64, 3), np.float32)}
    for cot in dict.fromkeys(cots):  # one forward per distinct CoT, its candidates' seeds
        ks = [k for k, c in enumerate(cots) if c == cot]
        trajs["own"][ks] = plans(inputs, cot, [SEEDS[k] for k in ks])
    trajs[NOMINAL] = plans(inputs, NOMINAL, SEEDS)
    trajs["empty"] = plans(inputs, "", SEEDS)

    reasoned = [c.strip() != NOMINAL for c in cots]
    row = {"clip_id": clip_id, "nav": nav, "cots": cots, "reasoned": reasoned,
           "cot_tokens": [len(tok(c + "\n", add_special_tokens=False)["input_ids"]) + 1
                          for c in cots]}
    for name in CONDITIONS:
        a64, a30, f64 = ade(trajs[name], logged), ade(trajs[name], logged, 30), fde(trajs[name], logged)
        row[name] = {"ade_first": float(a64[0]), "ade_mean": float(a64.mean()),
                     "ade_min": float(a64.min()), "ade3_first": float(a30[0]),
                     "fde_first": float(f64[0]), "ade_per_candidate": a64.tolist()}
    row["shift_own_vs_nominal_m"] = path_gap(trajs["own"], trajs[NOMINAL]).tolist()
    row["shift_own_vs_empty_m"] = path_gap(trajs["own"], trajs["empty"]).tolist()
    row["noise_floor_m"] = path_gap(trajs[NOMINAL], np.roll(trajs[NOMINAL], 1, axis=0)).tolist()
    rows.append(row)
    kept.append((clip_id, nav, trajs, logged, data["ego_history_xyz"][0].numpy(), row))

    print(f"  CoTs: {sum(reasoned)}/{K} reasoned, {K - sum(reasoned)}/{K} {NOMINAL}; "
          f"candidate 1: {cots[0][:90]}")
    print("  " + "   ".join(f"{name}: ADE1 {row[name]['ade_first']:.2f} mean {row[name]['ade_mean']:.2f} "
                            f"min {row[name]['ade_min']:.2f}" for name in CONDITIONS))
    gap = np.array(row["shift_own_vs_nominal_m"])[reasoned]
    if len(gap):
        print(f"  reasoned candidates: own vs {NOMINAL} plan shift {gap.mean():.2f} m, "
              f"noise floor {np.mean(row['noise_floor_m']):.2f} m")


def paired(metric: str, other: str, subset=None) -> dict:
    """own minus ``other`` per clip; negative = the reasoning helped. Bootstrap 95 % CI."""
    picked = [r for r in rows if subset is None or subset(r)]
    if not picked:
        return {"n": 0}
    diff = np.array([r["own"][metric] - r[other][metric] for r in picked])
    rng = np.random.default_rng(0)
    boot = rng.choice(diff, (10_000, len(diff))).mean(1)
    return {"n": len(diff), "own_mean": float(np.mean([r["own"][metric] for r in picked])),
            "other_mean": float(np.mean([r[other][metric] for r in picked])),
            "mean_diff": float(diff.mean()), "ci95": [float(np.percentile(boot, 2.5)),
                                                      float(np.percentile(boot, 97.5))],
            "own_better": int((diff < -1e-3).sum()), "other_better": int((diff > 1e-3).sum()),
            "ties": int((np.abs(diff) <= 1e-3).sum())}


banner(f"Summary over {len(rows)} clips (negative diff = reasoning helped)")
subsets = {"all clips": None, "clips where candidate 1 reasoned": lambda r: r["reasoned"][0]}
summary = {}
for label, subset in subsets.items():
    print(f"  {label}")
    for other in (NOMINAL, "empty"):
        for metric in ("ade_first", "ade_mean", "ade_min", "ade3_first", "fde_first"):
            s = paired(metric, other, subset)
            summary[f"{label} | own vs {other} | {metric}"] = s
            if s["n"]:
                print(f"    own vs {other:9s} {metric:10s} n={s['n']:2d}  own {s['own_mean']:.3f}  "
                      f"{other} {s['other_mean']:.3f}  diff {s['mean_diff']:+.3f} m "
                      f"[{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]  "
                      f"own better {s['own_better']}, worse {s['other_better']}, tie {s['ties']}")

reasoned_mask = np.concatenate([r["reasoned"] for r in rows])
shift_n = np.concatenate([r["shift_own_vs_nominal_m"] for r in rows])
shift_e = np.concatenate([r["shift_own_vs_empty_m"] for r in rows])
floor = np.concatenate([r["noise_floor_m"] for r in rows])
tokens = np.concatenate([r["cot_tokens"] for r in rows])
summary["candidates"] = {"total": int(reasoned_mask.size), "reasoned": int(reasoned_mask.sum())}
summary["plan_shift_m"] = {
    "own_vs_nominal_reasoned": float(shift_n[reasoned_mask].mean()),
    "own_vs_empty_reasoned": float(shift_e[reasoned_mask].mean()),
    "own_vs_empty_nominal": float(shift_e[~reasoned_mask].mean()) if (~reasoned_mask).any() else None,
    "noise_floor": float(floor.mean()),
}
summary["cot_tokens"] = {"reasoned_mean": float(tokens[reasoned_mask].mean()),
                         "nominal_mean": float(tokens[~reasoned_mask].mean())
                         if (~reasoned_mask).any() else None, "empty": 2}
per_seed = {name: np.array([r[name]["ade_per_candidate"] for r in rows]).mean(0) for name in CONDITIONS}
summary["ade_per_seed_mean"] = {name: v.tolist() for name, v in per_seed.items()}
print("  mean ADE per noise seed (every clip gets the same noise for candidate k, so one unlucky")
print("  draw shows up in every clip; prefer ade_mean over ade_first):")
for name, v in per_seed.items():
    print(f"    {name:9s} " + "  ".join(f"{seed}: {x:.2f}" for seed, x in zip(SEEDS, v, strict=True)))
ade_c = {name: np.concatenate([r[name]["ade_per_candidate"] for r in rows]) for name in CONDITIONS}
for other in (NOMINAL, "empty"):
    diff = (ade_c["own"] - ade_c[other])[reasoned_mask]
    summary[f"reasoned candidates | own vs {other}"] = {
        "n": int(diff.size), "mean_diff": float(diff.mean()), "median_diff": float(np.median(diff)),
        "own_better": int((diff < -1e-3).sum()), "other_better": int((diff > 1e-3).sum())}
    print(f"  reasoned candidates only (n={diff.size}), own vs {other}: mean diff {diff.mean():+.3f} m, "
          f"median {np.median(diff):+.3f} m, own better {(diff < -1e-3).sum()}, "
          f"worse {(diff > 1e-3).sum()}")
print(f"  candidates that reasoned: {reasoned_mask.sum()}/{reasoned_mask.size}")
print(f"  plan shift on reasoned candidates: own vs {NOMINAL} {shift_n[reasoned_mask].mean():.2f} m, "
      f"own vs empty {shift_e[reasoned_mask].mean():.2f} m; noise floor (same text, two seeds) "
      f"{floor.mean():.2f} m")
print(f"  reasoning cost: {tokens[reasoned_mask].mean():.0f} tokens when it reasons, "
      f"{summary['cot_tokens']['nominal_mean']} for {NOMINAL}, 2 for empty")
save_json("reasoning_ablation.json", {"args": vars(args), "seeds": SEEDS, "summary": summary,
                                      "rows": rows})

# Paired scatter: one point per clip, mean ADE over the K candidates.
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, other in zip(axes, (NOMINAL, "empty"), strict=True):
    for r in rows:
        x, y = r[other]["ade_mean"], r["own"]["ade_mean"]
        ax.scatter(x, y, s=40, color="#2a78d6" if r["reasoned"][0] else "#9a9a9a",
                   edgecolor="k", linewidth=0.4, zorder=3)
        if abs(x - y) > 0.25:
            ax.annotate(r["clip_id"][:8], (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    top = max(max(r[c]["ade_mean"] for c in CONDITIONS) for r in rows) * 1.08
    ax.plot([0, top], [0, top], color="#52514e", lw=1, ls="--")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"ADE 6.4 s with forced {other!r} reasoning (m)")
    ax.set_ylabel("ADE 6.4 s with the model's own reasoning (m)")
    s = summary[f"all clips | own vs {other} | ade_mean"]
    ax.set_title(f"own vs {other}: mean diff {s['mean_diff']:+.2f} m "
                 f"[{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}]\nbelow the line = reasoning helped",
                 fontsize=10)
    ax.grid(color="#e1e0d9")
axes[0].scatter([], [], color="#2a78d6", edgecolor="k", label="candidate 1 reasoned")
axes[0].scatter([], [], color="#9a9a9a", edgecolor="k", label=f"candidate 1 wrote {NOMINAL}")
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle(f"Reasoning ablation, {len(rows)} clips, oracle route, mean over {K} candidates "
             "with matched noise", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "reasoning_ablation.png", dpi=100, bbox_inches="tight")
print("wrote study/outputs/reasoning_ablation.png")

# BEV of the clips where candidate 1 reasoned: own vs <NOMINAL> vs empty, candidate 1 bold.
shown = [item for item in kept if item[5]["reasoned"][0]]
if shown:
    columns = min(4, len(shown))
    nrows = -(-len(shown) // columns)
    fig, axes = plt.subplots(nrows, columns, figsize=(4.2 * columns, 5.2 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    colors = {"own": "#2a78d6", NOMINAL: "#8a4fbf", "empty": "#eb6834"}
    styles = {"own": "-", NOMINAL: "--", "empty": ":"}
    for ax, (clip_id, nav, trajs, logged, hist, row) in zip(axes.flat, shown):
        ax.set_axis_on()
        ax.plot(hist[:, 1], hist[:, 0], "-.", color="#52514e", lw=1.2)
        for name in CONDITIONS:  # thin: candidates 2..K; bold on top: candidate 1
            for traj in trajs[name][1:]:
                ax.plot(traj[:, 1], traj[:, 0], color=colors[name], lw=0.8, alpha=0.4)
        for name in CONDITIONS:
            traj = trajs[name][0]
            ax.plot(traj[:, 1], traj[:, 0], color=colors[name], lw=2.2, ls=styles[name],
                    label=f"{name} (ADE mean {row[name]['ade_mean']:.2f})")
        ax.plot(logged[:, 1], logged[:, 0], "k--", lw=1.4, label="logged")
        everything = np.concatenate([np.concatenate([t[..., :2].reshape(-1, 2) for t in trajs.values()]),
                                     logged[:, :2], hist[:, :2]])
        lo, hi = everything.min(0) - 3, everything.max(0) + 3
        half = max(hi[0] - lo[0], hi[1] - lo[1]) / 2
        cx, cy = (lo + hi) / 2
        ax.set_ylim(cx - half, cx + half)
        ax.set_xlim(cy + half, cy - half)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=6)
        ax.grid(color="#e1e0d9")
        ax.legend(fontsize=6.5, loc="lower left")
        cot = row["cots"][0]
        ax.set_title(f"{clip_id[:8]}  route: {nav}\n{cot[:48]}{'...' if len(cot) > 48 else ''}",
                     fontsize=7.5, loc="left")
    fig.suptitle("Clips where candidate 1 reasoned: own CoT vs forced <NOMINAL> vs forced empty "
                 "(bold = candidate 1, thin = the other candidates; same noise seeds in every colour)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "reasoning_ablation_bev.png", dpi=100)
    print("wrote study/outputs/reasoning_ablation_bev.png")
