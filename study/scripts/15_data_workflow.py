"""Session 3b: the data workflow, from a PhysicalAI-AV clip id to the model's input tensors.

    python study/scripts/15_data_workflow.py [--clip-id ... --t0-us ... --nav-text ...]

Every stage calls the repository's own code and checks what it produced:

  1  dataset    clip id -> chunk -> zip files on the Hub; readers for 6 cameras + egomotion
  2  cameras    requested vs actually decoded frame times (the frame at or before each request)
  3  history    20 egomotion poses, world frame -> ego frame at t0
  4  sample     the in-memory dict returned by load_physical_aiavdataset, validation, sort order
  5  messages   the chat conversation: camera names, time labels, images, history slots, route
  6  template   the chat template text: one <|image_pad|> per image before expansion
  7  resize     per-camera pixel budget -> smart_resize -> bicubic resize in uint8
  8  processor  input_ids (image pads expanded), pixel_values patches, image_grid_thw, types
  9  history    ego history -> 57 delta tokens fused into the 57 <|traj_history|> slots
 10  summary    the final dict handed to sample_trajectories_from_data

Writes study/outputs/data_workflow.json. Write-up: study/10_DATA_WORKFLOW.md.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.functional import InterpolationMode, resize
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATASET_REVISION, avdi, banner, example, load_edge, save_json  # noqa: E402

from alpamayo2_edge.chat_template import CAMERA_DISPLAY_NAMES, build_conversation  # noqa: E402
from alpamayo2_edge.helper import get_processor, prepare_model_inputs  # noqa: E402
from alpamayo2_edge.load_physical_aiavdataset import (  # noqa: E402
    _CAMERAS,
    _FRONT_OFFSETS_US,
    _OTHER_OFFSETS_US,
    load_physical_aiavdataset,
)
from alpamayo2_edge.models.utils import fuse_traj_tokens  # noqa: E402
from alpamayo2_edge.sample import canonical_frame_order, canonicalize_sample  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--clip-id")
parser.add_argument("--t0-us", type=int)
parser.add_argument("--nav-text")
args = parser.parse_args()
spec = example()
if args.clip_id:
    spec = {"clip_id": args.clip_id, "t0_us": args.t0_us or 5_100_000,
            "nav_text": args.nav_text or "Continue straight"}
clip_id, t0_us = spec["clip_id"], spec["t0_us"]
report = {"sample": spec, "dataset_revision": DATASET_REVISION}


class CachedAVDI:
    """The SDK interface with get_clip_feature memoized, so stage 2 reuses stage 1's readers."""

    def __init__(self, inner) -> None:
        self.inner, self.features, self.revision = inner, inner.features, inner.revision
        self.cache = {}

    def get_clip_feature(self, clip, feature, maybe_stream=False):
        if (clip, feature) not in self.cache:
            self.cache[(clip, feature)] = self.inner.get_clip_feature(clip, feature, maybe_stream)
        return self.cache[(clip, feature)]


# ---------------------------------------------------------------- 1. dataset
banner("1. Dataset: clip id -> chunk -> files on the Hub")
interface = CachedAVDI(avdi())
inner = interface.inner
chunk = int(inner.clip_index.at[clip_id, "chunk"])
print(f"  repo nvidia/PhysicalAI-Autonomous-Vehicles @ {DATASET_REVISION[:10]}; "
      f"{len(inner.clip_index):,} clips; clip {clip_id} -> chunk {chunk}")
files = {}
for camera_id, name in _CAMERAS:
    feature = getattr(inner.features.CAMERA, name)
    path = inner.features.get_chunk_feature_filename(chunk, feature)
    reader = interface.get_clip_feature(clip_id, feature, maybe_stream=True)
    stamps = np.asarray(reader.timestamps)
    files[name] = {"zip": path, "in_zip": inner.features.get_clip_files_in_zip(clip_id, feature),
                   "cached_locally": bool(inner.is_file_cached(path)), "fps": float(reader.fps),
                   "frames": int(len(stamps)), "first_us": int(stamps[0]), "last_us": int(stamps[-1])}
    print(f"  cam {camera_id} {feature:26s} {path}  (local: {files[name]['cached_locally']})")
    print(f"        in zip: {list(files[name]['in_zip'].values())}  "
          f"{files[name]['frames']} frames @ {files[name]['fps']:.0f} fps, "
          f"{stamps[0] / 1e6:.2f}..{stamps[-1] / 1e6:.2f} s")
ego_feature = inner.features.LABELS.EGOMOTION
ego = interface.get_clip_feature(clip_id, ego_feature, maybe_stream=True)
print(f"  egomotion: {inner.features.get_chunk_feature_filename(chunk, ego_feature)}, "
      f"time range {ego.time_range[0] / 1e6:.2f}..{ego.time_range[1] / 1e6:.2f} s")
report["files"] = files

# ---------------------------------------------------------------- 2. cameras
banner(f"2. Camera frames: requested vs decoded (t0 = {t0_us / 1e6:.1f} s)")
data = load_physical_aiavdataset(clip_id, t0_us, nav_text=spec["nav_text"], avdi=interface)
print(f"  {'#':>2} {'camera':24s} {'requested':>10s} {'decoded':>10s} {'lag (ms)':>9s}")
frames = []
requested_all = np.concatenate([t0_us + (_FRONT_OFFSETS_US if c in (1, 6) else _OTHER_OFFSETS_US)
                                for c, _ in _CAMERAS])
for i, (camera, actual, requested) in enumerate(zip(
        data["camera_indices"].tolist(), data["absolute_timestamps"].tolist(), requested_all,
        strict=True)):
    lag = (requested - actual) / 1000
    assert 0 <= lag < 1000 / 30 + 1, "the decoder returns the frame at or before the request"
    frames.append({"camera": camera, "requested_us": int(requested), "decoded_us": int(actual),
                   "lag_ms": lag})
    print(f"  {i:2d} {CAMERA_DISPLAY_NAMES[camera]:24s} {(requested - t0_us) / 1e6:+9.2f}s "
          f"{(actual - t0_us) / 1e6:+9.3f}s {lag:9.1f}")
print(f"  image_frames {tuple(data['image_frames'].shape)} uint8, "
      f"{data['image_frames'].numel() / 1e6:.0f} MB; lag = how much older the frame is than requested")
report["frames"] = frames

# ---------------------------------------------------------------- 3. history
banner("3. Ego history: 20 poses at 0.1 s, world frame -> ego frame at t0")
times = t0_us + np.arange(-1_900_000, 1, 100_000)
world = ego(times).pose
world_xyz = np.asarray(world.translation)
xyz = data["ego_history_xyz"][0].numpy()
rot = data["ego_history_rot"][0].numpy()
print(f"  world xyz first {np.round(world_xyz[0], 2)}  last (t0) {np.round(world_xyz[-1], 2)}")
print(f"  t0 frame xyz first {np.round(xyz[0], 3)}  last {np.round(xyz[-1], 6)}")
assert np.allclose(xyz[-1], 0, atol=1e-5) and np.allclose(rot[-1], np.eye(3), atol=1e-5)
speed = np.linalg.norm(xyz[-1, :2] - xyz[-2, :2]) * 10
yaw_change = np.degrees(np.arctan2(rot[0, 1, 0], rot[0, 0, 0]))
print(f"  last pose is the origin with identity rotation; speed at t0 {speed:.2f} m/s; "
      f"heading 1.9 s ago {yaw_change:+.2f} deg relative to now")
report["history"] = {"speed_t0_mps": float(speed), "first_xyz": xyz[0].tolist()}

# ---------------------------------------------------------------- 4. sample
banner("4. The sample dict (load_physical_aiavdataset) and its canonical order")
for key, value in data.items():
    desc = f"{str(value.dtype):14s} {tuple(value.shape)}" if isinstance(value, torch.Tensor) else repr(value)[:90]
    print(f"  {key:22s} {desc}")
order = canonical_frame_order(data["camera_indices"], data["absolute_timestamps"])
print(f"  canonical order (camera, then time) = loader order: {torch.equal(order, torch.arange(22))}")
canonical = canonicalize_sample(data)

# ---------------------------------------------------------------- 5. messages
banner("5. Chat messages (build_conversation)")
model, _ = load_edge(device="cpu")  # config, tokenizer and processor; no GPU needed here
messages = build_conversation(canonical, num_tokens_per_history_traj=57, include_camera_ids=True)
user = messages[1]["content"]
kinds = [entry["type"] for entry in user]
print(f"  system: {messages[0]['content'][0]['text']!r}")
print(f"  user: {len(user)} entries = {kinds.count('image')} images + {kinds.count('text')} texts")
for entry in user[:4] + user[-6:]:
    print("    ", entry["type"], repr(entry["text"])[:80] if entry["type"] == "text" else
          f"[3, 1080, 1920] bounds {entry['min_pixels']}..{entry['max_pixels']}")
reference = canonical["absolute_timestamps"].reshape(-1)[-1]
print(f"  time labels are relative to the LAST frame in camera-major order (front telephoto, "
      f"{(reference - t0_us) / 1e6:+.3f} s from t0), rounded to 0.1 s")

# ---------------------------------------------------------------- 6. template
banner("6. Chat template text (before the processor expands image pads)")
processor = get_processor(model.tokenizer, model.config)
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                     add_vision_id=False, enable_thinking=False)
print(f"  {len(text)} characters, {text.count('<|image_pad|>')} x <|image_pad|> (one per image), "
      f"ends with {text[-30:]!r}")

# ---------------------------------------------------------------- 7. resize
banner("7. Per-camera pixel budget -> smart_resize -> bicubic resize in uint8")
image_processor = processor.image_processor
sizes = {}
for camera in (1, 6, 0, 2, 3, 5):
    lower, upper = canonical["camera_pixel_bounds"][camera]
    h, w = smart_resize(1080, 1920, factor=32, min_pixels=lower, max_pixels=upper)
    sizes[camera] = (h, w)
    print(f"  {CAMERA_DISPLAY_NAMES[camera]:24s} budget {lower:,}..{upper:,} px -> {h} x {w} "
          f"= {h * w:,} px -> {h // 16} x {w // 16} patches -> {h * w // 1024} tokens")
report["resized"] = {CAMERA_DISPLAY_NAMES[c]: list(s) for c, s in sizes.items()}

# ---------------------------------------------------------------- 8. processor
banner("8. Processor output (prepare_model_inputs)")
prepared = prepare_model_inputs(data, model.config, model.tokenizer)
td = prepared["tokenized_data"]
for key, value in td.items():
    print(f"  {key:22s} {str(value.dtype):14s} {tuple(value.shape)}")
ids, grid = td["input_ids"][0], td["image_grid_thw"]
pad = model.config.vlm_config.image_token_id
runs, current = [], 0
for is_pad in (ids == pad).tolist() + [False]:
    if is_pad:
        current += 1
    elif current:
        runs.append(current)
        current = 0
assert runs == (grid.prod(-1) // 4).tolist(), "each image's pad run = its patches / 4"
print(f"  {len(runs)} runs of <|image_pad|> in input_ids, lengths = patches / 4: {runs[:4]} ...")
print(f"  pixel_values: {grid.prod(-1).sum().item()} patch rows = sum of t*h*w over images; "
      f"768 = 16*16*3; range [{td['pixel_values'].min():.1f}, {td['pixel_values'].max():.1f}]")
blocks = torch.split(td["pixel_values"], grid.prod(-1).tolist())
t, h, w = grid[3].tolist()
rebuilt = blocks[3].reshape(h // 2, w // 2, 2, 2, 16, 16, 3).permute(0, 2, 4, 1, 3, 5, 6)
rebuilt = ((rebuilt.reshape(h * 16, w * 16, 3) * 0.5 + 0.5) * 255).round().to(torch.uint8)
direct = resize(canonical["image_frames"][3, 0], [h * 16, w * 16],
                interpolation=InterpolationMode.BICUBIC).permute(1, 2, 0)
diff = int((rebuilt.int() - direct.int()).abs().max())
print(f"  image 3 rebuilt from its patch rows == direct bicubic resize: max |diff| = {diff}")
start, ranges = 0, []
for i, n in enumerate(grid.prod(-1).tolist()):
    ranges.append([start, start + n - 1])
    start += n
report["pixel_value_rows"] = ranges
report["counts"] = {"prompt_tokens": len(ids), "image_tokens": int((ids == pad).sum()),
                    "patch_rows": int(grid.prod(-1).sum())}

# ---------------------------------------------------------------- 9. history tokens
banner("9. Ego history -> 57 tokens (done inside the model by fuse_traj_tokens)")
fused = fuse_traj_tokens(model.history_traj_tokenizer, model.future_traj_tokenizer, td["input_ids"],
                         {k: prepared[k] for k in ("ego_history_xyz", "ego_history_rot")},
                         model.config.traj_ids)[0]
slots = (ids == model.config.traj_ids["history_pad"]).nonzero().flatten()
bins = (fused[slots] - model.config.traj_ids["history_id0"]).reshape(-1, 3)
print(f"  {len(slots)} <|traj_history|> slots at prompt positions {slots[0].item()}..{slots[-1].item()}"
      f" -> bins (dx, dy, dz) first {bins[0].tolist()} last {bins[-1].tolist()}")

# ---------------------------------------------------------------- 10. summary
banner("10. What sample_trajectories_from_data receives")
print(f"  tokenized_data: input_ids [1, {len(ids)}] ({report['counts']['image_tokens']} image pads, "
      f"57 history slots), attention_mask, mm_token_type_ids, pixel_values "
      f"[{report['counts']['patch_rows']}, 768], image_grid_thw [22, 3]")
print(f"  ego_history_xyz {tuple(prepared['ego_history_xyz'].shape)}, ego_history_rot "
      f"{tuple(prepared['ego_history_rot'].shape)}  (batch, trajectory group, 20 poses)")
print("  next (inside the model): fuse history tokens -> vision tower -> projector -> scatter "
      "into the image pads -> M-RoPE -> prefill   (study/03_INFERENCE_WALKTHROUGH.md)")
save_json("data_workflow.json", report)
