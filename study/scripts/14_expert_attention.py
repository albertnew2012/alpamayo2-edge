"""Session 5d: where does the flow-matching expert look? Attention mass by input segment.

    python study/scripts/14_expert_attention.py

The expert's 64 action tokens attend to the whole VLM cache (prompt + CoT + anchor) plus
themselves. This script loads the model with eager attention (which returns attention
weights), hooks the 28 expert attention layers during the 10 Euler steps of one normal
inference, and sums each query's attention over key segments:

  first token        the first prompt token (attention sinks usually land here)
  images (per cam)   the 2,892 <|image_pad|> embeddings, split by camera
  ego history        the 57 history bin tokens
  route sentence     "Navigation input is: Turn right in 30m"
  other prompt text  system prompt, camera names, time labels, vision markers, task, chat markup
  reasoning (CoT)    the generated reasoning tokens
  anchor             <|traj_future_start|>
  own action tokens  the expert's 64 tokens (keys written this step, cropped afterwards)

Mass = mean over heads, query tokens, Euler steps; "per token" divides by the segment size.
Eager attention samples a slightly different CoT than SDPA; the one used is printed.
Writes study/outputs/expert_attention.png and expert_attention.json.
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
    banner,
    example,
    load_edge,
    load_sample,
    predict,
    prepare,
    save_json,
)

from alpamayo2_edge.chat_template import CAMERA_DISPLAY_NAMES  # noqa: E402
from alpamayo2_edge.models.utils import fuse_traj_tokens  # noqa: E402
from alpamayo2_edge.sample import canonicalize_sample  # noqa: E402

model, _ = load_edge(attn="eager")
cfg, tok = model.config, model.tokenizer
spec = example()
data = load_sample(**spec)
inputs = prepare(model, data)
td = inputs["tokenized_data"]

captured = []  # one [heads, 64, keys] tensor (mean over batch) per expert layer call


def hook(module, args, output):
    weights = output[1]
    if weights is not None:
        captured.append(weights.float().mean(0).cpu())


handles = [layer.self_attn.register_forward_hook(hook) for layer in model.expert.expert.layers]
try:
    _, _, extra = predict(model, inputs)
finally:
    for handle in handles:
        handle.remove()
n_layers = len(model.expert.expert.layers)
assert len(captured) % n_layers == 0, len(captured)
steps = len(captured) // n_layers
cot = extra["cot"].reshape(-1)[0]
print(f"CoT (eager attention): {cot}")
print(f"captured {len(captured)} attention maps = {steps} Euler steps x {n_layers} layers, "
      f"shape {tuple(captured[0].shape)}")

# ---------------------------------------------------------------- key segments
prompt = fuse_traj_tokens(model.history_traj_tokenizer, model.future_traj_tokenizer,
                          td["input_ids"], {k: inputs[k] for k in ("ego_history_xyz",
                                                                   "ego_history_rot")},
                          cfg.traj_ids)[0].cpu()
generated = extra["generated_ids"][0].cpu()
generated = generated[generated >= 0]
n_prompt, n_gen, n_keys = len(prompt), len(generated), captured[0].shape[-1]
cache_len = n_keys - 64
assert cache_len == n_prompt + n_gen, (cache_len, n_prompt, n_gen)
labels = np.full(n_keys, "other prompt text", dtype=object)
labels[0] = "first token"

image_positions = (prompt == cfg.vlm_config.image_token_id).nonzero().flatten().numpy()
cameras = canonicalize_sample(data)["camera_indices"].tolist()
sizes = (td["image_grid_thw"].prod(-1) // 4).tolist()
cursor = 0
for camera, size in zip(cameras, sizes, strict=True):
    labels[image_positions[cursor:cursor + size]] = f"images: {CAMERA_DISPLAY_NAMES[camera]}"
    cursor += size
lo, hi = cfg.traj_ids["history_id0"], cfg.traj_ids["history_id0"] + cfg.history_vocab_size
labels[((prompt >= lo) & (prompt < hi)).nonzero().flatten().numpy()] = "ego history"


def find(sequence: list[int], start: int = 0) -> int:
    ids = prompt.tolist()
    for i in range(start, len(ids) - len(sequence) + 1):
        if ids[i:i + len(sequence)] == sequence:
            return i
    raise ValueError(f"{tok.decode(sequence)!r} not found in the prompt")


nav_start = find(tok("Navigation input is:", add_special_tokens=False)["input_ids"])
nav_end = find(tok("Your task:", add_special_tokens=False)["input_ids"], nav_start)
labels[nav_start:nav_end] = "route sentence"
labels[n_prompt:n_prompt + n_gen - 1] = "reasoning (CoT)"
labels[n_prompt + n_gen - 1] = "anchor"
labels[cache_len:] = "own action tokens"
print(f"route sentence tokens: {tok.decode(prompt[nav_start:nav_end].tolist())!r}")

ORDER = ["first token"] + [f"images: {CAMERA_DISPLAY_NAMES[c]}" for c in (1, 6, 0, 2, 3, 5)] + [
    "ego history", "route sentence", "other prompt text", "reasoning (CoT)", "anchor",
    "own action tokens"]
masks = {name: torch.from_numpy(labels == name) for name in ORDER}
counts = {name: int(mask.sum()) for name, mask in masks.items()}

# mass[layer, segment] averaged over heads, queries and Euler steps
mass = np.zeros((n_layers, len(ORDER)))
for index, weights in enumerate(captured):
    per_key = weights.mean(dim=(0, 1))  # mean over heads and the 64 queries -> [keys]
    for j, name in enumerate(ORDER):
        mass[index % n_layers, j] += float(per_key[masks[name]].sum()) / steps
overall = mass.mean(0)

banner("Where the expert's attention goes (mean over 28 layers, 16 heads, 64 queries, "
       f"{steps} steps)")
print(f"  {'segment':32s} {'tokens':>7s} {'share of keys':>14s} {'attention mass':>15s} "
      f"{'x per token vs uniform':>23s}")
for j, name in enumerate(ORDER):
    share = counts[name] / n_keys
    print(f"  {name:32s} {counts[name]:7d} {share * 100:13.2f}% {overall[j] * 100:14.2f}% "
          f"{overall[j] / share:23.1f}")
groups = {"images (all)": [n for n in ORDER if n.startswith("images")]}
img = sum(overall[ORDER.index(n)] for n in groups["images (all)"])
print(f"  {'images (all six cameras)':32s} {sum(counts[n] for n in groups['images (all)']):7d} "
      f"{'':14s} {img * 100:14.2f}%")

thirds = [range(0, 9), range(9, 19), range(19, n_layers)]
print("\n  by layer depth (attention mass %):")
print(f"  {'segment':32s}" + "".join(f"{f'layers {r.start}-{r.stop - 1}':>14s}" for r in thirds))
for j, name in enumerate(ORDER):
    print(f"  {name:32s}" + "".join(f"{mass[list(r), j].mean() * 100:14.2f}" for r in thirds))

# Which individual keys absorb the most attention (sinks show up here).
per_key_all = torch.stack([w.mean(dim=(0, 1)) for w in captured]).mean(0)  # [keys]
all_ids = torch.cat([prompt, generated]).tolist()
print("\n  top 12 single keys by attention mass:")
for index in per_key_all.argsort(descending=True)[:12].tolist():
    text = repr(tok.decode([all_ids[index]])) if index < cache_len else f"action token {index - cache_len}"
    print(f"    key {index:5d}  {per_key_all[index] * 100:5.2f}%  {labels[index]:20s} {text}")
other = masks["other prompt text"]
top_other = per_key_all[other].sort(descending=True).values
print(f"  'other prompt text': top 10 of {int(other.sum())} tokens hold "
      f"{top_other[:10].sum() / top_other.sum() * 100:.0f}% of its mass")

colors = ["#52514e", "#eb6834", "#f2a27a", "#c95a2a", "#f6c9b1", "#a8562e", "#d98b5f",
          "#4466aa", "#d4a017", "#bdbdbd", "#1baf7a", "#0b6e4f", "#8a4fbf"]
fig, ax = plt.subplots(figsize=(13, 6))
bottom = np.zeros(n_layers)
for j, name in enumerate(ORDER):
    ax.bar(np.arange(n_layers), mass[:, j] * 100, bottom=bottom, color=colors[j], label=name,
           width=0.85)
    bottom += mass[:, j] * 100
ax.set_xlabel("expert layer (each attends VLM layer i's cache + its own 64 tokens)")
ax.set_ylabel("attention mass (%)")
ax.set_xlim(-0.6, n_layers - 0.4)
ax.set_ylim(0, 100)
ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
ax.set_title(f"Flow-matching expert: attention by input segment ({counts['reasoning (CoT)']} "
             f"CoT tokens of {n_keys} keys), clip {spec['clip_id'][:8]}", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "expert_attention.png", dpi=110)
print("wrote study/outputs/expert_attention.png")
save_json("expert_attention.json", {
    "cot": cot, "keys": n_keys, "counts": counts, "order": ORDER,
    "mass_overall": dict(zip(ORDER, overall.tolist(), strict=True)),
    "mass_by_layer": mass.tolist(),
})
