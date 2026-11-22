"""Session 7b: Alpamayo 2 Edge vs Alpamayo 1.5 on the same 21 clips, paired.

    python study/scripts/10_eval_multiclip.py                     # Edge, oracle route
    python study/scripts/10_eval_multiclip.py --nav-mode fixed    # Edge, wrong route
    # in ~/Desktop/alpamayo1.5 (its own venv and code):
    python study/scripts/21_eval_vs_edge.py                       # A1.5: none / oracle / wrong
    python study/scripts/11_compare_with_a15.py

Both models see the same clips at the same t0 with 6 candidates each, scored against the
same egomotion (same dataset revision). Each model gets its *native* input profile (Edge:
6 cameras x 22 frames, 20 poses; A1.5: 4 cameras x 4 frames, 16 poses). Route conditions:
oracle (derived from the logged future, identical text), wrong ("Turn right in 30m"),
and for A1.5 also none (its default prompt; Edge requires a route).

Reports paired per-clip differences with a Wilcoxon signed-rank test, and writes
study/outputs/compare_a15.{json,png} and compare_a15_bev.png.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import OUT, banner, save_json  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--a15-dir", type=Path,
                    default=Path.home() / "Desktop/alpamayo1.5/study/outputs")
parser.add_argument("--a15-tag", default="eval_vs_edge")
parser.add_argument("--a15-cfg-tag", default="eval_vs_edge_cfg",
                    help="optional second A1.5 run with classifier-free route guidance")
args = parser.parse_args()

edge = {c: json.loads((OUT / f"eval_{t}.json").read_text())
        for c, t in (("oracle", "oracle_k6"), ("wrong", "fixed_k6"))}
edge_trajs = np.load(OUT / "eval_oracle_k6_trajs.npz")
a15 = json.loads((args.a15_dir / f"{args.a15_tag}.json").read_text())
a15_trajs = np.load(args.a15_dir / f"{args.a15_tag}_trajs.npz")
cfg_path = args.a15_dir / f"{args.a15_cfg_tag}.json"
a15_cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else None
clips = [r["clip_id"] for r in edge["oracle"]["rows"]]
assert [r["clip_id"] for r in a15["rows"]["oracle"]] == clips, "clip order differs"
assert edge["wrong"]["args"]["fixed_nav"] == "Turn right in 30m"
print(f"{len(clips)} clips; ground-truth endpoint agreement between the two runs: "
      f"max {a15['gt_endpoint_max_diff_m']:.4f} m")


def column(rows, key):
    return np.array([r[key] for r in rows])


METRICS = ("ade_3_first", "min_ade_3", "ade_6.4_first", "min_ade_6.4", "fde_6.4_first",
           "min_fde_6.4")
runs = {
    "Edge, oracle route": edge["oracle"]["rows"],
    "A1.5, oracle route": a15["rows"]["oracle"],
    "A1.5, no route": a15["rows"]["none"],
    "Edge, wrong route": edge["wrong"]["rows"],
    "A1.5, wrong route": a15["rows"]["wrong"],
}
if a15_cfg is not None:
    runs["A1.5, oracle + CFG"] = a15_cfg["rows"]["oracle_cfg"]
    runs["A1.5, wrong + CFG"] = a15_cfg["rows"]["wrong_cfg"]
    assert [r["clip_id"] for r in runs["A1.5, oracle + CFG"]] == clips

banner("1. Medians (means) over the same clips, metres")
print(f"  {'run':22s}" + "".join(f"{m:>17s}" for m in METRICS))
table = {}
for name, rows in runs.items():
    table[name] = {m: {"median": float(np.median(column(rows, m))),
                       "mean": float(column(rows, m).mean())} for m in METRICS}
    print(f"  {name:22s}" + "".join(
        f"{table[name][m]['median']:8.3f} ({table[name][m]['mean']:5.2f})" for m in METRICS))

banner("2. Paired per-clip comparison (Wilcoxon signed-rank, two-sided)")
pairs = [("Edge, oracle route", "A1.5, oracle route"),
         ("Edge, oracle route", "A1.5, no route"),
         ("A1.5, oracle route", "A1.5, no route")]
if a15_cfg is not None:
    pairs.append(("Edge, oracle route", "A1.5, oracle + CFG"))
paired = {}
for a, b in pairs:
    for metric in ("ade_6.4_first", "min_ade_6.4", "ade_3_first"):
        x, y = column(runs[a], metric), column(runs[b], metric)
        p = float(wilcoxon(x, y).pvalue)
        wins = int((x < y).sum())
        key = f"{a} vs {b} / {metric}"
        paired[key] = {"wins_first": wins, "n": len(x), "median_diff": float(np.median(x - y)),
                       "p": p}
        print(f"  {a} vs {b:20s} {metric:14s} first better on {wins:2d}/{len(x)}, "
              f"median diff {np.median(x - y):+.3f} m, p = {p:.3f}")

banner("3. Route sensitivity: wrong route minus oracle route, per clip")
sensitivity = {}
SENS = {"Edge": ("Edge, wrong route", "Edge, oracle route"),
        "A1.5": ("A1.5, wrong route", "A1.5, oracle route")}
if a15_cfg is not None:
    SENS["A1.5 + CFG"] = ("A1.5, wrong + CFG", "A1.5, oracle + CFG")
for model, (wrong_run, oracle_run) in SENS.items():
    diff = column(runs[wrong_run], "ade_6.4_first") - column(runs[oracle_run], "ade_6.4_first")
    moved = int((diff > 0.5).sum())
    sensitivity[model] = {"median_increase": float(np.median(diff)), "clips_worse_by_0.5m": moved}
    print(f"  {model:10s} median ADE@6.4 increase {np.median(diff):+.3f} m; "
          f"worse by > 0.5 m on {moved}/{len(diff)} clips")
a15_route_changes_cot = sum(
    o["cot_first"] != w["cot_first"] for o, w in zip(a15["rows"]["oracle"], a15["rows"]["wrong"]))
edge_route_changes_cot = sum(
    o["cot_first"] != w["cot_first"] for o, w in zip(edge["oracle"]["rows"], edge["wrong"]["rows"]))
print(f"  candidate-1 CoT text changes between oracle and wrong route: Edge "
      f"{edge_route_changes_cot}/{len(clips)}, A1.5 {a15_route_changes_cot}/{len(clips)}")
if a15_cfg is not None:
    cfg_s = float(np.median(column(runs["A1.5, oracle + CFG"], "infer_s")))

banner("4. Cost on this machine (6 candidates, RTX 3090)")
edge_s = float(np.median(column(edge["oracle"]["rows"], "infer_s")))
a15_s = a15["summary"]["oracle"]["median_infer_s"]
print(f"  Edge model call median {edge_s:.2f} s (eager, fully on GPU)")
print(f"  A1.5 model call median {a15_s:.2f} s (eager, {a15['attn']}, part of the weights "
      f"offloaded to CPU; peak {a15['peak_gib']:.1f} GiB) - offload dominates, not a fair "
      "speed comparison")

# ---------------------------------------------------------------- figures
fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
for ax, metric, title in ((axes[0], "ade_6.4_first", "ADE@6.4 s, candidate 1"),
                          (axes[1], "min_ade_6.4", "minADE@6.4 s over 6")):
    x = column(runs["A1.5, oracle route"], metric)
    y = column(runs["Edge, oracle route"], metric)
    ax.scatter(x, y, color="#2a78d6", s=28, label="oracle route")
    xn = column(runs["A1.5, no route"], metric)
    ax.scatter(xn, y, color="#eb6834", s=18, marker="x", label="A1.5 without route")
    lim = [min(x.min(), y.min(), xn.min()) * 0.8, max(x.max(), y.max(), xn.max()) * 1.2]
    ax.plot(lim, lim, color="#8a8984", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Alpamayo 1.5 (m)")
    ax.set_ylabel("Alpamayo 2 Edge, oracle route (m)")
    key = f"Edge, oracle route vs A1.5, oracle route / {metric}"
    ax.set_title(f"{title}\nbelow the diagonal = Edge better on that clip "
                 f"({paired[key]['wins_first']}/{len(clips)}, p={paired[key]['p']:.2f})",
                 fontsize=10)
    ax.grid(color="#e1e0d9", which="both")
    ax.legend(fontsize=8)
ax = axes[2]
for i, (model, (wrong_run, oracle_run)) in enumerate(SENS.items()):
    diff = column(runs[wrong_run], "ade_6.4_first") - column(runs[oracle_run], "ade_6.4_first")
    ax.scatter(np.full(len(diff), i) + np.random.default_rng(0).uniform(-0.12, 0.12, len(diff)),
               diff, color=("#2a78d6", "#eb6834", "#c98a00")[i], s=22)
    ax.hlines(np.median(diff), i - 0.25, i + 0.25, color="k", lw=2)
ax.axhline(0, color="#8a8984", lw=1)
ax.set_xticks(range(len(SENS)), [m.replace("A1.5", "Alpamayo 1.5") for m in SENS])
ax.set_yscale("symlog", linthresh=0.5)
ax.set_ylabel("ADE@6.4 s: wrong route − oracle route (m)")
ax.set_title("Route sensitivity: 'Turn right in 30m' on every clip\n(bar = median)",
             fontsize=10)
ax.grid(color="#e1e0d9")
fig.suptitle(f"Alpamayo 2 Edge vs Alpamayo 1.5 — {len(clips)} PhysicalAI-AV clips, t0 = 5.1 s, "
             "6 candidates each, same ground truth")
fig.tight_layout()
fig.savefig(OUT / "compare_a15.png", dpi=105)
print("wrote study/outputs/compare_a15.png")

columns = 7
nrows = -(-len(clips) // columns)
fig, axes = plt.subplots(nrows, columns, figsize=(3.1 * columns, 4.0 * nrows))
for ax in axes.flat:
    ax.set_axis_off()
for ax, clip_id, er, ar in zip(axes.flat, clips, runs["Edge, oracle route"],
                               runs["A1.5, oracle route"]):
    ax.set_axis_on()
    gt, hist = edge_trajs[f"{clip_id}/gt"], edge_trajs[f"{clip_id}/hist"]
    e_c, a_c = edge_trajs[f"{clip_id}/cands"], a15_trajs[f"{clip_id}/oracle"]
    ax.plot(hist[:, 1], hist[:, 0], "-.", color="#52514e", lw=1)
    for k in range(len(e_c)):
        ax.plot(a_c[k, :, 1], a_c[k, :, 0], color="#eb6834", lw=1.6 if k == 0 else 0.6,
                alpha=1 if k == 0 else 0.4)
        ax.plot(e_c[k, :, 1], e_c[k, :, 0], color="#2a78d6", lw=1.6 if k == 0 else 0.6,
                alpha=1 if k == 0 else 0.4)
    ax.plot(gt[:, 1], gt[:, 0], "k--", lw=1.3)
    allx = np.concatenate([e_c[..., 0].ravel(), a_c[..., 0].ravel(), gt[:, 0], hist[:, 0]])
    span = max(np.ptp(allx), 20) / 2 + 3
    cx = (allx.max() + allx.min()) / 2
    ax.set_ylim(cx - span, cx + span)
    ax.set_xlim(span * 0.6, -span * 0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=6)
    ax.set_title(f"{clip_id[:8]}  route: {er['nav']}\nEdge ADE {er['ade_6.4_first']:.2f} / "
                 f"min {er['min_ade_6.4']:.2f}\nA1.5 ADE {ar['ade_6.4_first']:.2f} / "
                 f"min {ar['min_ade_6.4']:.2f}", fontsize=6.5, loc="left")
fig.suptitle("Oracle route: Edge (blue) vs Alpamayo 1.5 (orange); candidate 1 bold, others "
             "faint; logged future dashed", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "compare_a15_bev.png", dpi=90)
print("wrote study/outputs/compare_a15_bev.png")

save_json("compare_a15.json", {
    "clips": clips, "table": table, "paired": paired, "route_sensitivity": sensitivity,
    "cot_changes_with_wrong_route": {"Edge": edge_route_changes_cot, "A1.5": a15_route_changes_cot},
    "median_infer_s": {"Edge": edge_s, "A1.5": a15_s,
                       "A1.5 + CFG": cfg_s if a15_cfg is not None else None},
})
