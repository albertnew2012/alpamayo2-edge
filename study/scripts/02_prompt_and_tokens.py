"""Session 3: how 22 images + 20 poses + one sentence become one token sequence.

    python study/scripts/02_prompt_and_tokens.py

Prints, for the README example:
  * every image: camera, time label, raw and resized size, patch grid, LLM tokens
  * the prompt with each image's <|image_pad|> run collapsed to one marker
  * the 57 ego-history tokens before and after ``fuse_traj_tokens``
  * the 3-D M-RoPE positions at segment boundaries and the rope delta
Writes study/outputs/token_layout.json and study/outputs/prompt_collapsed.txt.
"""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import OUT, banner, example, load_edge, load_sample, save_json  # noqa: E402

from alpamayo2_edge.chat_template import CAMERA_DISPLAY_NAMES, relative_time_labels  # noqa: E402
from alpamayo2_edge.helper import prepare_model_inputs  # noqa: E402
from alpamayo2_edge.models.utils import fuse_traj_tokens  # noqa: E402
from alpamayo2_edge.sample import canonicalize_sample  # noqa: E402

spec = example()
data = load_sample(**spec)
model, _ = load_edge(device="cpu")  # tokenizer, processor and config only; no GPU needed
config, tok = model.config, model.tokenizer

banner("1. Raw sample (load_physical_aiavdataset)")
for key in ("image_frames", "camera_indices", "absolute_timestamps", "ego_history_xyz",
            "ego_history_rot"):
    print(f"  {key:22s} {str(data[key].dtype):14s} {tuple(data[key].shape)}")
print("  nav_text              ", data["nav_text"])

prepared = prepare_model_inputs(data, config, tok)
td = prepared["tokenized_data"]
input_ids = td["input_ids"][0]
grid = td["image_grid_thw"]
canonical = canonicalize_sample(data)
labels = relative_time_labels(canonical["absolute_timestamps"])

banner("2. Per-image token budget (camera-major, then capture time)")
images = []
H, W = data["image_frames"].shape[-2:]
print(f"  raw frames: {H} x {W}   patch 16, merge 2x2 -> one LLM token per 32x32 pixels")
print(f"  {'#':>2} {'camera':24s} {'label':>10s} {'resized':>11s} {'grid(t,h,w)':>14s} "
      f"{'patches':>8s} {'tokens':>7s}")
for index, (camera, label, thw) in enumerate(
    zip(canonical["camera_indices"].tolist(), labels, grid.tolist(), strict=True)
):
    t, h, w = thw
    tokens = t * h * w // 4
    images.append({"camera": camera, "name": CAMERA_DISPLAY_NAMES[camera], "label": label,
                   "resized_hw": [h * 16, w * 16], "grid_thw": thw, "patches": t * h * w,
                   "llm_tokens": tokens})
    print(f"  {index:2d} {CAMERA_DISPLAY_NAMES[camera]:24s} {label:>10s} {h * 16:4d} x {w * 16:<4d} "
          f"{str(tuple(thw)):>14s} {t * h * w:8d} {tokens:7d}")
image_tokens = int((input_ids == config.vlm_config.image_token_id).sum())
print(f"  total image tokens: {image_tokens}   pixel_values {tuple(td['pixel_values'].shape)} "
      f"(patches x 16*16*3)")

banner("3. Token accounting")
history_slots = int((input_ids == config.traj_ids["history_pad"]).sum())
counts = {
    "prompt_length": len(input_ids),
    "image_tokens": image_tokens,
    "history_placeholders": history_slots,
    "other_text_tokens": len(input_ids) - image_tokens - history_slots,
}
for key, value in counts.items():
    print(f"  {key:22s} {value:6d}")
mm = td["mm_token_type_ids"][0]
print("  mm_token_type_ids values:", sorted(set(mm.tolist())), "(0 text, 1 image)")

banner("4. Prompt text, image pads collapsed")
pieces = []
for is_image, group in itertools.groupby(input_ids.tolist(),
                                         key=lambda i: i == config.vlm_config.image_token_id):
    group = list(group)
    if is_image:
        pieces.append(f"[{len(group)} x <|image_pad|>]")
    else:
        pieces.append(tok.decode(group, skip_special_tokens=False))
collapsed = "".join(pieces)
collapsed = collapsed.replace("<|traj_history|>" * history_slots,
                              f"[{history_slots} x <|traj_history|>]")
print(collapsed)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "prompt_collapsed.txt").write_text(collapsed + "\n")

banner("5. Ego history -> 57 discrete tokens (fuse_traj_tokens)")
traj = {k: prepared[k] for k in ("ego_history_xyz", "ego_history_rot")}
fused = fuse_traj_tokens(model.history_traj_tokenizer, model.future_traj_tokenizer,
                         td["input_ids"], traj, config.traj_ids)[0]
slots = (input_ids == config.traj_ids["history_pad"]).nonzero().flatten()
codes = (fused[slots] - config.traj_ids["history_id0"]).reshape(-1, 3)
xyz = traj["ego_history_xyz"][0, 0]
deltas = (xyz[1:] - xyz[:-1]).numpy()
print("  history xyz (t0 frame) first / last:", xyz[0].numpy().round(3), xyz[-1].numpy().round(3))
print("  19 deltas -> 3 bins each in [0, 999] over [-4, 4] m (xy) and [-10, 10] m (z)")
print(f"  {'step':>4} {'dx':>8} {'dy':>8} {'dz':>8}   {'bins':>16}")
for step in (0, 1, 9, 17, 18):
    print(f"  {step:4d} {deltas[step][0]:8.4f} {deltas[step][1]:8.4f} {deltas[step][2]:8.4f}   "
          f"{str(codes[step].tolist()):>16s}")
bin_m = (8.0 / 999, 20.0 / 999)
print(f"  quantization step: {bin_m[0] * 100:.2f} cm (xy) / {bin_m[1] * 100:.2f} cm (z) per 0.1 s"
      f"  => speed resolution {bin_m[0] * 10:.3f} m/s")
decoded, _, _ = model.history_traj_tokenizer.decode(xyz[None], None, codes.reshape(1, -1))
err = (decoded[0] - xyz).abs().max().item()
print(f"  decode(encode(history)) max |error| = {err * 100:.2f} cm")

banner("6. Multimodal RoPE positions (get_rope_index)")
position_ids, rope_deltas = model.vlm.model.get_rope_index(
    td["input_ids"], mm_token_type_ids=td["mm_token_type_ids"], image_grid_thw=grid,
    attention_mask=td["attention_mask"])
pos = position_ids[:, 0]
first_image = int((input_ids == config.vlm_config.image_token_id).nonzero()[0])
h0, w0 = grid[0, 1].item() // 2, grid[0, 2].item() // 2
print(f"  first image starts at token {first_image}, LLM grid {h0} x {w0}")
for label, index in (("token before image 0", first_image - 1),
                     ("image 0, first token", first_image),
                     ("image 0, end of row 0", first_image + w0 - 1),
                     ("image 0, start of row 1", first_image + w0),
                     ("image 0, last token", first_image + h0 * w0 - 1),
                     ("token after image 0", first_image + h0 * w0),
                     ("last prompt token", len(input_ids) - 1)):
    print(f"  {label:26s} seq idx {index:5d}  (t, h, w) = {tuple(pos[:, index].tolist())}")
print(f"  rope_delta = max position + 1 - length = {rope_deltas.item()}  "
      f"(an image of h x w tokens advances the position by max(h, w), not h*w)")

save_json("token_layout.json", {
    "sample": {k: spec[k] for k in ("clip_id", "t0_us", "nav_text")},
    "raw_hw": [H, W],
    "images": images,
    "counts": counts,
    "rope_delta": rope_deltas.item(),
    "history_codes_first_last": [codes[0].tolist(), codes[-1].tolist()],
    "history_roundtrip_max_err_m": err,
})
