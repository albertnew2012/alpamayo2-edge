# 10 — The data workflow: from a clip id to the model's input tensors

Everything that happens to the data *before* the model's first layer: how a PhysicalAI-AV clip
is found and streamed, which camera frames and poses are taken, how they become a chat
prompt, and how 22 camera images turn into `pixel_values [11568, 768]`.

| | |
|---|---|
| script | [`scripts/15_data_workflow.py`](scripts/15_data_workflow.py) — runs every stage below with the repository's own code and checks each result |
| debug config | **3-3b. Data workflow: clip id -> model inputs, stage by stage** (`.vscode/launch.json`, `justMyCode` + `stopOnEntry`) |
| output | [`outputs/data_workflow.json`](outputs/data_workflow.json) |
| sample | README clip `ea7bbd31-b7a5-4972-8dbd-7089e6b53de4`, t0 = 4.0 s, route "Turn right in 30m" |
| runtime | ~1–2 min (streams the clip's six videos and egomotion from the Hub); no GPU |

Related: [01 §3](01_MODEL_STRUCTURE.md) (token layout and M-RoPE), [02_prompt_and_tokens.py](scripts/02_prompt_and_tokens.py)
(config 3-3, the prompt in detail), [03](03_INFERENCE_WALKTHROUGH.md) (what the model does with these tensors).

---

## 0. The whole workflow on one page

```
 clip id + t0 + route sentence
        │
 ① DATASET  nvidia/PhysicalAI-Autonomous-Vehicles (Hugging Face, gated), revision pinned
        │   clip_index.parquet: clip id -> chunk (ea7bbd31 -> chunk 727)
        │   one zip per (feature, chunk): camera/<cam>/<cam>.chunk_0727.zip, labels/egomotion/...
        │   physical_ai_av SDK: local HF cache if present, else stream byte ranges from the Hub
        ▼
 ② CAMERAS   6 cameras, mp4 at 30 fps + per-frame timestamps            load_physical_aiavdataset
        │   front wide (1), front tele (6): t0 −2, −1, −0.5, −0.1, 0 s  → 5 frames each
        │   cross L/R (0, 2), rear L/R (3, 5): t0 −1, −0.1, 0 s         → 3 frames each
        │   decoder returns the frame AT OR BEFORE each request, with its real timestamp
        │   image_frames [22, 1, 3, 1080, 1920] uint8, camera_indices [22], absolute_timestamps [22]
        ▼
 ③ HISTORY   egomotion interpolated at t0 −1.9 … 0 s (20 poses, 0.1 s apart)
        │   world frame → ego frame at t0:  xyz' = R0⁻¹ (xyz − xyz0),  R' = R0⁻¹ R
        │   ego_history_xyz [1, 20, 3], ego_history_rot [1, 20, 3, 3]   (last pose = origin, identity)
        ▼
 ④ SAMPLE    validate (counts, dtypes, shapes, finite), sort camera-major then by time
        ▼                                                                 prepare_model_inputs ↓
 ⑤ MESSAGES  system prompt + user turn: "Front left camera: " <t=-1.0s> [image] <t=-0.1s> [image] …
        │   <|traj_history_start|> 57 × <|traj_history|> <|traj_history_end|>
        │   "Navigation input is: Turn right in 30m"  "Your task: …"
        ▼
 ⑥ TEMPLATE  chat template, enable_thinking=False → text ending "<|im_start|>assistant\n<think></think>"
        │   one <|vision_start|><|image_pad|><|vision_end|> per image (22 pads)
        ▼
 ⑦ RESIZE    per-camera pixel budget → smart_resize (multiples of 32) → bicubic resize IN uint8
        │   front/tele 320 × 576 (184,320 px) · side/rear 224 × 416 (93,184 px)
        ▼
 ⑧ PROCESSOR patchify 16 × 16 (block-major 2 × 2), expand each <|image_pad|> to patches/4 tokens,
        │   normalize (x/255 − 0.5)/0.5
        │   input_ids [1, 3261] · attention_mask · mm_token_type_ids · image_grid_thw [22, 3]
        │   pixel_values [11568, 768] float32
        ▼
 ⑨ HISTORY TOKENS (inside the model) 19 deltas × (dx, dy, dz) → 57 bins → into the 57 slots
        ▼
    sample_trajectories_from_data   → vision tower → projector → scatter → M-RoPE → prefill …
```

Stages ①–③ are [`load_physical_aiavdataset`](../src/alpamayo2_edge/load_physical_aiavdataset.py);
④–⑧ are [`prepare_model_inputs`](../src/alpamayo2_edge/helper.py); ⑨ runs at the start of
[`sample_trajectories_from_data`](../src/alpamayo2_edge/models/alpamayo2_edge.py).

---

## ① Dataset: where the bytes come from

**Code:** the `physical_ai_av` SDK (`PhysicalAIAVDatasetInterface`, `get_clip_feature`,
`open_file`), called from `load_physical_aiavdataset`.

- The dataset is a **gated** Hugging Face repo: `nvidia/PhysicalAI-Autonomous-Vehicles`. The
  study scripts pin revision `2ac2e9c0…` (the notebook's), so every run sees the same data.
- `clip_index.parquet` lists 306,151 clips and maps each clip to one of 3,146 **chunks**
  (≤ 100 clips each; chunk 727 has 99). Data is packed as one **zip per (feature, chunk)**.
- For our clip (chunk 727), each camera zip holds three files:
  `<clip>.<camera>.mp4`, `<clip>.<camera>.timestamps.parquet` (one int64 µs time per frame) and
  `<clip>.<camera>.blurred_boxes.parquet` (privacy-blur boxes). Egomotion is
  `labels/egomotion/egomotion.chunk_0727.zip`.
- `get_clip_feature(..., maybe_stream=True)` opens the zip **from the local HF cache if
  present, otherwise streams it from the Hub by byte ranges** (`HfFileSystem`), reads only this
  clip's members, and returns:
  - a `SeekVideoReader` per camera: the clip's mp4 bytes held in memory plus its timestamps;
  - an egomotion interpolator: poses at arbitrary times.

Measured for the README clip:

| camera | frames | fps | time range |
|---|---|---|---|
| front wide / tele / cross L/R / rear L/R | 605 each | 30 | ≈ −0.1 … 20.0 s |
| egomotion | — | (interpolated) | −0.20 … 104.94 s |

Every clip is ~20 s of video; egomotion runs longer. That bounds t0: it must leave 2 s of
camera history (t0 ≥ 2.0 s, checked by the loader), and for evaluation the logged future
needs t0 + 6.4 s of egomotion.

## ② Camera frames: what is requested, what comes back

**Code:** [`_load_camera_frames`](../src/alpamayo2_edge/load_physical_aiavdataset.py#L59) →
`camera.decode_images_from_timestamps(requested)` (line 84).

| cameras | requested times (relative to t0) | frames |
|---|---|---|
| 1 front wide, 6 front tele | −2.0, −1.0, −0.5, −0.1, 0.0 s | 5 each |
| 0 cross left, 2 cross right, 3 rear left, 5 rear right | −1.0, −0.1, 0.0 s | 3 each |

The decoder picks, for each requested time, **the latest frame at or before it**
(`searchsorted(timestamps, requested, side="right") − 1` in the SDK), so a frame is never from
the future. It returns the frame's **real** timestamp, and the loader keeps that, not the
requested one. Measured lag (how much older the frame is than requested):

| cameras | lag |
|---|---|
| front wide, front tele, rear left, rear right | 10.6 ms |
| cross left, cross right | 30.9 ms |

The cameras are not synchronized with each other (at 30 fps a frame lasts 33 ms). The model is
told the real times through the text labels (stage ⑤), rounded to 0.1 s.

Result, in loader order (cameras 0, 1, 2, 3, 5, 6, each oldest → newest):

| key | dtype | shape |
|---|---|---|
| `image_frames` | uint8 | `[22, 1, 3, 1080, 1920]` (137 MB, RGB, contiguous, CPU) |
| `camera_indices` | int64 | `[22]` |
| `absolute_timestamps` | int64 | `[22]` µs, clip-relative, real capture times |

## ③ Ego history: 20 poses in the ego frame at t0

**Code:** `load_physical_aiavdataset`, lines 165–180.

```python
history_timestamps = t0 + arange(-1.9 s, 0, 0.1 s)       # 20 times
pose = egomotion(history_timestamps)                      # world frame
local_xyz = R0⁻¹ (xyz − xyz[-1])                          # R0 = rotation at t0
local_rot = R0⁻¹ · R
```

After the transform the last pose (t0) is exactly the origin with the identity rotation;
x points forward, y left, z up. Measured: first pose (−1.9 s) at (−13.85, 0.04, −0.14) m,
speed at t0 7.63 m/s, heading 1.9 s ago −0.19° relative to now (driving straight).

**Not loaded:** future poses (the README demo never sees the answer), camera calibration.
The study scripts load the logged future separately, only to plot and score.

## ④ The sample dict, validation and order

`load_physical_aiavdataset` returns:

| key | value |
|---|---|
| `image_frames`, `camera_indices`, `absolute_timestamps` | from ② |
| `ego_history_xyz` `[1, 20, 3]`, `ego_history_rot` `[1, 20, 3, 3]` | from ③ |
| `nav_text` | `["Turn right in 30m"]` — must be supplied; the dataset has no route text |
| `camera_pixel_bounds` | `{1: (163840, 196608), 6: (…), 0/2/3/5: (81920, 98304)}` |
| `clip_id`, `t0_us`, `provenance` (dataset, clip, t0, revision), `input_profile` `"camera_subsampled"` | bookkeeping |

[`validate_sample`](../src/alpamayo2_edge/sample.py#L100) checks camera ids and frame counts
(5/5/3/3/3/3), dtypes, shapes and finite poses, and refuses an empty route.
[`canonicalize_sample`](../src/alpamayo2_edge/sample.py#L132) sorts frames **by camera, then
by capture time** (two stable sorts). For the loader's output this order is already the
loader order.

**You can build this dict yourself** (another dataset, your own car): the contract is the table
above. `prepare_model_inputs` does not care where the frames came from.

## ⑤ Messages: the chat conversation

**Code:** [`build_conversation`](../src/alpamayo2_edge/chat_template.py#L64),
[`_image_content`](../src/alpamayo2_edge/chat_template.py#L35).

- **system:** "You are a driving assistant that generates safe and accurate actions."
- **user:** 56 entries = 22 images + 34 texts, in this order:
  - per camera, its name once (`"Front left camera: "`), then for each frame a time label
    (`"<t=-1.0s>"`) and the image, which carries its camera's `min_pixels` / `max_pixels`;
  - `"\n"`, then `<|traj_history_start|>` + 57 × `<|traj_history|>` + `<|traj_history_end|>`;
  - `"\n"`, `"Navigation input is: Turn right in 30m"`, `"\n"`, and the task sentence
    "Your task: output the chain-of-thought reasoning of the driving process, then output
    the future trajectory."

**Time labels** are `timestamp − timestamp of the last frame in camera-major order`, i.e.
relative to the **newest front-telephoto frame** (here −0.011 s from t0), formatted with one
decimal (`z` flag turns −0.0 into 0.0). So "0.0s" means "the newest frame", not exactly t0.

## ⑥ Chat template

`processor.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)`
renders the checkpoint's own `chat_template.jinja`: 2,519 characters with **one**
`<|vision_start|><|image_pad|><|vision_end|>` per image (22 pads), ending in
`<|im_start|>assistant\n<think></think>`. `prepare_model_inputs` refuses a template that
does not end with the empty think block, because the model was trained to write its driving
reasoning right after it.

## ⑦ Resize: per-camera pixel budget

**Code:** [`_tokenize_messages`](../src/alpamayo2_edge/helper.py#L87), lines 111–117.

For each image, `smart_resize(1080, 1920, factor=32, min_pixels, max_pixels)` picks a size
inside the camera's budget that keeps the aspect ratio and makes both sides multiples of 32
(16-pixel patches × 2×2 merge). Then the **uint8** frame is resized bicubically; normalization
comes later (resizing in uint8 matches training).

| camera | budget (pixels) | resized | patches | LLM tokens |
|---|---|---|---|---|
| front wide, front tele | 163,840 – 196,608 | **320 × 576** = 184,320 | 20 × 36 = 720 | 180 |
| cross L/R, rear L/R | 81,920 – 98,304 | **224 × 416** = 93,184 | 14 × 26 = 364 | 91 |

How 320 × 576 comes out: 1080 × 1920 = 2.07 M px > 196,608, so scale by
√(2.07 M / 196,608) ≈ 3.25 → 332 × 591 → round down to multiples of 32 → 320 × 576.

## ⑧ Processor: tokens and patches

The processor is called with `do_resize=False, do_rescale=False, do_normalize=False` (the
helper already resized, and normalizes itself afterwards):

| key | dtype | shape | what |
|---|---|---|---|
| `input_ids` | int64 | `[1, 3261]` | each `<|image_pad|>` expanded to patches/4 copies: runs of 91 or 180, 2,892 in total |
| `attention_mask` | int64 | `[1, 3261]` | all ones (one sample, no padding) |
| `mm_token_type_ids` | int64 | `[1, 3261]` | 1 at image tokens, 0 elsewhere (M-RoPE needs it) |
| `image_grid_thw` | int64 | `[22, 3]` | per image (t, h, w) in patches: (1, 20, 36) or (1, 14, 26) |
| `pixel_values` | float32 | `[11568, 768]` | every 16 × 16 patch of every image, normalized to [−1, 1] |

**`pixel_values` layout** (verified: rebuilding image 3 from its rows equals a direct bicubic
resize, max difference 0):

- **rows = patches**, image after image in canonical order (camera, then time):
  front left 0–1091 · front 1092–4691 · front right 4692–5783 · rear left 5784–6875 ·
  rear right 6876–7967 · front tele 7968–11567. The current front frame is rows 3972–4691.
  `torch.split(pixel_values, image_grid_thw.prod(-1).tolist())` gives the 22 blocks.
- **inside an image**, 2 × 2 **block-major**: the four patches that the projector merges into
  one token are consecutive rows; blocks go left to right, top to bottom.
- **768 columns** = 16 × 16 pixels × 3 colours, pixel by pixel (R, G, B together).
- **normalization:** `(x / 255 − 0.5) / 0.5` → [−1, 1] (checkpoint mean = std = 0.5).

Rebuild an image from its rows:

```python
pv, g = td["pixel_values"], td["image_grid_thw"]
t, h, w = g[3].tolist()
x = torch.split(pv, g.prod(-1).tolist())[3].reshape(h // 2, w // 2, 2, 2, 16, 16, 3)
x = x.permute(0, 2, 4, 1, 3, 5, 6).reshape(h * 16, w * 16, 3)          # [320, 576, 3]
img = ((x * 0.5 + 0.5) * 255).round().clamp(0, 255).to(torch.uint8)
```

## ⑨ Ego history tokens (inside the model)

`prepare_model_inputs` returns the history as `[1, 1, 20, 3]` / `[1, 1, 20, 3, 3]` (batch,
trajectory group, poses). At the start of `sample_trajectories_from_data`,
[`fuse_traj_tokens`](../src/alpamayo2_edge/models/utils.py#L154) runs the
`DeltaTrajectoryTokenizer`: 19 consecutive deltas × (dx, dy, dz), each in 1000 bins over ±4 m
(xy) and ±10 m (z), offset to ids `<i0>`… (131,072 …), and writes them into the 57
`<|traj_history|>` slots (prompt positions 3161–3217) with `masked_scatter`. Measured bins:
first `[583, 499, 500]`, last `[595, 499, 500]` (≈ 0.67 → 0.76 m per 0.1 s forward).

## ⑩ What the model receives, and what happens next

```
data = {"tokenized_data": {input_ids, attention_mask, mm_token_type_ids, pixel_values, image_grid_thw},
        "ego_history_xyz": [1, 1, 20, 3], "ego_history_rot": [1, 1, 20, 3, 3]}
```

Inside the model: fuse history tokens (⑨) → vision tower on `pixel_values` (per image) →
projector merges each 4 consecutive rows into one 2048-d token (11,568 → 2,892) → those
replace the 2,892 `<|image_pad|>` embeddings → M-RoPE positions → prefill → reasoning →
expert. See [03](03_INFERENCE_WALKTHROUGH.md).

---

## Variants

**Freeform VQA** ([`load_physical_aiavdataset_vqa`](../src/alpamayo2_edge/load_physical_aiavdataset.py#L211),
[`prepare_vqa_inputs`](../src/alpamayo2_edge/helper.py)): 6 cameras × 4 frames at −0.3, −0.2,
−0.1, 0 s; **all** cameras at the front budget (320 × 576, 180 tokens), so 24 images = 4,320
image tokens and a ~4,630-token prompt (depends on the question); no ego history, no route; the prompt ends with the
question and no `<think></think>`.

**What the CLI saves** ([`inference.py`](../src/alpamayo2_edge/inference.py) `run_data`):
`packed.safetensors` = the exact model inputs (`tokenized_data.*` with the history tokens
already fused, `traj_data.ego_history_*`); `prompt.txt` = the decoded prompt with all special
tokens; `output.json` records the clip, t0, route and dataset revision.

**Caches used by the study scripts** (not part of the repo's workflow):
`tmp/sample_cache/drive_<clip>_<t0>.pt` stores the ④ dict (137 MB each) so debugger restarts
skip streaming; the demo videos wrap the SDK so each clip's videos are fetched once and every
0.1 s tick re-runs ②–⑧ from memory (inputs verified identical to a fresh load).

## Things that bite

1. **The route is required and not in the dataset.** Whatever you write becomes part of the
   input; metrics depend on how you choose it ([09](09_REASONING_ACTION_STUDY.md), [07](07_HEAD_TO_HEAD_VS_A15.md)).
2. **Frame times are real, not requested**, and cameras differ by ~20 ms. If you build the dict
   yourself, pass real capture times.
3. **"0.0s" is the newest telephoto frame**, not t0.
4. **Resize happens in uint8, before normalization.** Normalizing first and resizing floats
   gives slightly different pixels than training saw.
5. **Order matters.** Frames must be camera-major then by time; the time labels, the pad runs in
   `input_ids` and the rows of `pixel_values` all follow that order. `canonicalize_sample`
   enforces it.
6. **CPU cost:** stages ⑦–⑧ take ~1.4 s on this machine (resizing 22 full-HD frames), more than
   the accelerated model call ([05](05_ACCELERATION.md)).

## Debugging it (config 3-3b)

| line in `15_data_workflow.py` | inspect |
|---|---|
| 81 | `reader.timestamps[:5]`, `reader.fps` — one camera's frame clock |
| 98 | step **into** `load_physical_aiavdataset` (justMyCode keeps you in repo code) |
| 121 | `xyz`, `rot[-1]` — the history in the t0 frame |
| 137 | `order` — the canonical permutation |
| 144 | `messages[1]["content"][:6]` — names, labels, images |
| 159 | `text[:400]`, `text.count("<|image_pad|>")` |
| 178 | step into `prepare_model_inputs` → `_tokenize_messages` (resize at helper.py:111) |
| 195 | `blocks[3].shape`, the rebuilt image `rebuilt` vs `direct` |
| 213 | `fused[3161:3170]` — the history bins in place |

Another clip: `python study/scripts/15_data_workflow.py --clip-id <uuid> --t0-us 5100000
--nav-text "Continue straight"`.
