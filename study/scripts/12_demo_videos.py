"""Demo videos: Alpamayo 2 Edge driving through PhysicalAI-AV clips.

    python study/scripts/12_demo_videos.py drive --clip-id ea7bbd31-b7a5-4972-8dbd-7089e6b53de4
    python study/scripts/12_demo_videos.py route --clip-id ea7bbd31-b7a5-4972-8dbd-7089e6b53de4
    python study/scripts/12_demo_videos.py flow

drive  one inference every 0.1 s of the clip (10 Hz = real time at 10 fps): the six input
       cameras with the predicted plans projected into them, a bird's-eye view, and the
       reasoning text. Route = an oracle navigation sentence re-derived at every tick from
       the logged path ("Turn right in 40m" ... "Continue straight").
route  the same ticks run twice, with the oracle route and with left/right swapped, side by
       side: how the sentence steers the plan.
flow   one tick, the 10 Euler steps of flow matching for six candidates, animated from noise
       to the final plans (Euler steps are straight segments in action space, so frames
       between steps are exact points on the integration path).

Inputs at each tick are built by the repository's own ``load_physical_aiavdataset`` and
``prepare_model_inputs``; the dataset interface is wrapped so each clip's videos are
streamed once and kept in memory. Videos go to study/outputs/videos/.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False  # model text may contain '$'
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import OUT, avdi, derive_nav_text, load_edge, predict  # noqa: E402

from alpamayo2_edge.helper import prepare_model_inputs, to_device  # noqa: E402
from alpamayo2_edge.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402

VIDEOS = OUT / "videos"
README_CLIP = "ea7bbd31-b7a5-4972-8dbd-7089e6b53de4"
# Edge camera id -> SDK sensor name, and the panel layout (row, column).
CAMERAS = {
    0: ("camera_cross_left_120fov", "Front left", (0, 0)),
    1: ("camera_front_wide_120fov", "Front", (0, 1)),
    2: ("camera_cross_right_120fov", "Front right", (0, 2)),
    3: ("camera_rear_left_70fov", "Rear left", (1, 0)),
    6: ("camera_front_tele_30fov", "Front telephoto", (1, 1)),
    5: ("camera_rear_right_70fov", "Rear right", (1, 2)),
}
PALETTE = ["#00e5ff", "#ff8a3d", "#7dff6b", "#ffd23d", "#ff6bd6", "#b18cff"]
BEV_PALETTE = ["#1f77d4", "#eb6834", "#1baf7a", "#d49a00", "#d64fa8", "#7b5cd6"]
DISPLAY_W, DISPLAY_H = 640, 360
MIN_DEPTH_M = 0.5


class CachedAVDI:
    """The SDK interface with ``get_clip_feature`` memoized: each clip feature is fetched once.

    ``SeekVideoReader`` keeps the decoded container open over in-memory bytes, so repeated
    ``decode_images_from_timestamps`` calls (one set per tick) need no further streaming.
    """

    def __init__(self, inner) -> None:
        self.inner, self.features, self.revision = inner, inner.features, inner.revision
        self.cache = {}

    def get_clip_feature(self, clip_id, feature, maybe_stream=False):
        key = (clip_id, feature)
        if key not in self.cache:
            self.cache[key] = self.inner.get_clip_feature(clip_id, feature, maybe_stream=maybe_stream)
        return self.cache[key]


class Clip:
    """Calibration, egomotion and per-tick inputs for one clip."""

    def __init__(self, clip_id: str, interface: CachedAVDI) -> None:
        self.id, self.avdi = clip_id, interface
        f = interface.features
        self.ego = interface.get_clip_feature(clip_id, f.LABELS.EGOMOTION, maybe_stream=True)
        intrinsics = interface.get_clip_feature(clip_id, f.CALIBRATION.CAMERA_INTRINSICS, True)
        extrinsics = interface.get_clip_feature(clip_id, f.CALIBRATION.SENSOR_EXTRINSICS, True)
        dims = interface.get_clip_feature(clip_id, f.CALIBRATION.VEHICLE_DIMENSIONS, True)
        self.length, self.width = float(dims.length), float(dims.width)
        self.rear_to_center = float(dims.rear_axle_to_bbox_center)
        self.cams = {
            cid: (intrinsics.camera_models[name], extrinsics.sensor_poses[name].inv())
            for cid, (name, _, _) in CAMERAS.items()
        }
        readers = [interface.get_clip_feature(clip_id, getattr(f.CAMERA, name.upper()), True)
                   for name, _, _ in CAMERAS.values()]
        # Every camera must cover the whole input schedule of a tick.
        self.cam_range = (max(int(r.timestamps[0]) for r in readers),
                          min(int(r.timestamps[-1]) for r in readers))

    def tick_range(self, t_start: float | None, t_end: float | None, stride: float):
        lo = max(2.0, self.cam_range[0] / 1e6 + 2.0) if t_start is None else t_start
        hi = (self.cam_range[1] / 1e6 - 0.01) if t_end is None else t_end
        return [int(round(t * 1e6)) for t in np.arange(lo, hi + 1e-9, stride)]

    def path(self, t0_us: int, seconds: float) -> np.ndarray:
        """Logged positions t0+0.1 ... t0+seconds in the t0 ego frame (x fwd, y left)."""
        end = min(t0_us + int(seconds * 1e6), int(self.ego.time_range[1]))
        times = np.arange(t0_us, end + 1, 100_000)
        poses = self.ego(times).pose
        xyz = np.asarray(poses.translation)
        return poses.rotation[0].inv().apply(xyz - xyz[0])[1:]

    def route(self, t0_us: int) -> str:
        """Oracle navigation sentence from the next 10 s of the logged path."""
        return derive_nav_text(self.path(t0_us, 10.0))

    def load(self, t0_us: int, nav_text: str) -> dict:
        return load_physical_aiavdataset(self.id, t0_us, nav_text=nav_text, avdi=self.avdi)


def project(cam, xyz: np.ndarray):
    """Ego-frame points -> display pixels and a visibility mask."""
    model, ego_to_cam = cam
    pts = ego_to_cam.apply(np.asarray(xyz, dtype=np.float64))
    pixels = model.ray2pixel(pts)
    visible = (pts[:, 2] > MIN_DEPTH_M) & (pixels[:, 0] > -model.width) & (
        pixels[:, 0] < 2 * model.width) & (pixels[:, 1] > -model.height) & (pixels[:, 1] < 2 * model.height)
    scale = np.array([DISPLAY_W / model.width, DISPLAY_H / model.height])
    return pixels * scale, visible


def ribbon(xy: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    d = np.gradient(xy, axis=0)
    d /= np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-6)
    normal = np.stack([-d[:, 1], d[:, 0]], axis=-1)
    z = np.zeros((len(xy), 1))
    return (np.hstack([xy + normal * width / 2, z]), np.hstack([xy - normal * width / 2, z]))


def latest_frames(data: dict) -> dict[int, np.ndarray]:
    """The newest frame of every camera, downscaled for display."""
    frames = {}
    cams, stamps = data["camera_indices"].numpy(), data["absolute_timestamps"].reshape(-1).numpy()
    for cid in CAMERAS:
        index = np.nonzero(cams == cid)[0]
        index = index[np.argmax(stamps[index])]
        image = data["image_frames"][index, 0].permute(1, 2, 0).numpy()
        frames[cid] = np.asarray(Image.fromarray(image).resize((DISPLAY_W, DISPLAY_H), Image.BILINEAR))
    return frames


def draw_camera(ax, clip: Clip, cid: int, image, cands, logged, hist, title, ribbon_first=True):
    ax.imshow(image, extent=(0, DISPLAY_W, DISPLAY_H, 0))
    ax.set_xlim(0, DISPLAY_W)
    ax.set_ylim(DISPLAY_H, 0)
    ax.set_axis_off()
    cam = clip.cams[cid]
    if hist is not None:
        px, vis = project(cam, np.hstack([hist[:, :2], np.zeros((len(hist), 1))]))
        if vis.sum() > 1:
            ax.plot(px[vis, 0], px[vis, 1], color="#d0d0d0", lw=1.6, ls="-.")
    if logged is not None:
        px, vis = project(cam, np.hstack([logged[:, :2], np.zeros((len(logged), 1))]))
        if vis.sum() > 1:
            ax.plot(px[vis, 0], px[vis, 1], color="white", lw=1.8, ls="--")
    for k in range(len(cands) - 1, -1, -1):
        xy = cands[k, :, :2]
        if k == 0 and ribbon_first:
            left, right = ribbon(xy, clip.width)
            pl, vl = project(cam, left)
            pr, vr = project(cam, right)
            vis = vl & vr
            if vis.sum() > 1:
                poly = np.concatenate([pl[vis], pr[vis][::-1]])
                ax.fill(poly[:, 0], poly[:, 1], color=PALETTE[0], alpha=0.35, lw=0)
        px, vis = project(cam, np.hstack([xy, np.zeros((len(xy), 1))]))
        if vis.sum() > 1:
            ax.plot(px[vis, 0], px[vis, 1], color=PALETTE[k % 6], lw=2.4 if k == 0 else 1.3)
    ax.text(6, 18, title, color="white", fontsize=10, weight="bold",
            bbox=dict(facecolor="black", alpha=0.55, pad=2, lw=0))


def draw_bev(ax, clip: Clip, cands, logged, hist, extent, labels=True, palette=BEV_PALETTE):
    (x_lo, x_hi), y_half = extent
    ax.set_facecolor("#f7f6f2")
    ax.plot(hist[:, 1], hist[:, 0], "-.", color="#52514e", lw=2, label="history" if labels else None)
    if logged is not None:
        ax.plot(logged[:, 1], logged[:, 0], "k--", lw=2, label="logged future" if labels else None)
    for k in range(len(cands) - 1, -1, -1):
        ax.plot(cands[k, :, 1], cands[k, :, 0], color=palette[k % 6], lw=2.6 if k == 0 else 1.4,
                label=(f"candidate {k + 1}" if labels else None))
    ax.add_patch(Rectangle((-clip.width / 2, clip.rear_to_center - clip.length / 2), clip.width,
                           clip.length, facecolor="#2b2b2b", edgecolor="none", zorder=5))
    ax.set_xlim(y_half, -y_half)
    ax.set_ylim(x_lo, x_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#dddcd5", lw=0.8)
    ax.tick_params(labelsize=8, colors="#cccccc")
    ax.set_xlabel("y (m), left +", fontsize=9, color="#cccccc")
    ax.set_ylabel("x (m), forward", fontsize=9, color="#cccccc")
    if labels:  # candidates were drawn last-to-first for z-order; list them first-to-last
        handles, names = ax.get_legend_handles_labels()
        order = sorted(range(len(names)), key=lambda i: (not names[i].startswith(("hist", "logged")),
                                                         names[i]))
        ax.legend([handles[i] for i in order], [names[i] for i in order], fontsize=8,
                  loc="lower left", framealpha=0.85)


def bev_extent(clip: Clip, ticks: list[int]) -> tuple[tuple[float, float], float]:
    paths = [clip.path(t, 6.4) for t in ticks[:: max(1, len(ticks) // 20)]]
    fwd = max(float(np.abs(p[:, 0]).max()) for p in paths)
    lat = max(float(np.abs(p[:, 1]).max()) for p in paths)
    x_hi = float(np.clip(fwd + 8, 35, 220))
    y_half = float(np.clip(max(lat + 8, 0.36 * (x_hi + 22)), 14, 120))
    return (-22.0, x_hi), y_half


def to_array(fig) -> np.ndarray:
    fig.canvas.draw()
    array = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return array


class Writer:
    """Raw RGB frames -> H.264 MP4 through ffmpeg."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium", "-movflags", "+faststart",
             str(path)], stdin=subprocess.PIPE)
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.frames += 1

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()
        print(f"wrote {self.path.relative_to(OUT.parents[1])} ({self.frames} frames, "
              f"{self.path.stat().st_size / 1e6:.1f} MB)")


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width)) or text


def run_model(model, inputs, candidates: int):
    torch.cuda.synchronize()
    start = time.perf_counter()
    xyz, _, extra = predict(model, inputs, num_traj_sets=candidates)
    torch.cuda.synchronize()
    cands = xyz[0].reshape(-1, 64, 3).float().cpu().numpy()
    return cands, list(extra["cot"].reshape(-1)), time.perf_counter() - start


def ade(cands, logged):
    if logged is None or len(logged) < 64:
        return None
    return np.linalg.norm(cands[:, :, :2] - logged[None, :64, :2], axis=-1).mean(-1)


# ---------------------------------------------------------------------------------- drive
def render_drive(clip, t0_us, data, cands, cots, logged, route, seconds, extent, accel):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#101010")
    frames = latest_frames(data)
    hist = data["ego_history_xyz"][0].numpy()
    cam_w, cam_h, left, top = 0.237, 0.237 * 19.2 / 10.8 * 9 / 16, 0.008, 0.935
    for cid, (_, name, (row, col)) in CAMERAS.items():
        ax = fig.add_axes((left + col * (cam_w + 0.004), top - (row + 1) * cam_h - row * 0.008,
                           cam_w, cam_h))
        draw_camera(ax, clip, cid, frames[cid], cands, logged if cid in (0, 1, 2, 6) else None,
                    hist if cid in (3, 5) else None, name)
    ax = fig.add_axes((0.762, 0.07, 0.23, 0.865))
    draw_bev(ax, clip, cands, logged, hist, extent)
    speed = float(np.linalg.norm(hist[-1, :2] - hist[-2, :2]) * 10)
    errors = ade(cands, logged)
    fig.text(0.008, 0.965, f"Alpamayo 2 Edge  ·  clip {clip.id[:8]}  ·  t0 = {t0_us / 1e6:5.1f} s"
             f"  ·  {speed * 3.6:5.1f} km/h", color="white", fontsize=17, weight="bold")
    fig.text(0.762, 0.965, "bird's-eye view, ego frame at t0", color="#bbbbbb", fontsize=12)
    y = top - 2 * cam_h - 0.04
    fig.text(0.01, y, f"Route:  {route}", color="#ffd23d", fontsize=15, weight="bold")
    cot = cots[0].strip()
    shown = "<NOMINAL>  (nothing that needs explaining: adaptive reasoning skips the CoT)" \
        if cot == "<NOMINAL>" else cot
    fig.text(0.01, y - 0.045, "Reasoning (candidate 1):", color="#9ad7ff", fontsize=12)
    fig.text(0.01, y - 0.06, wrap(shown, 105), color="white", fontsize=15, va="top")
    others = [c for c in dict.fromkeys(c.strip() for c in cots[1:]) if c != cot][:3]
    if others:
        fig.text(0.01, y - 0.175, "Other candidates:\n" + "\n".join(
            "• " + (c if len(c) < 135 else c[:132] + "...") for c in others),
            color="#bbbbbb", fontsize=10.5, va="top")
    stats = (f"{len(cands)} candidates  ·  model call {seconds:.2f} s "
             f"({'torch.compile + CUDA graphs' if accel else 'eager'}, RTX 3090)")
    if errors is not None:
        stats += f"  ·  ADE vs logged path: candidate 1 {errors[0]:.2f} m, best {errors.min():.2f} m"
    fig.text(0.01, 0.025, stats, color="#9a9a9a", fontsize=11)
    fig.text(0.01, 0.005, "Cameras: predicted plans (candidate 1 = filled ribbon, vehicle width), "
             "logged future dashed white, ego history dash-dot in the rear views.  Route is "
             "an oracle sentence derived from the logged path.", color="#6f6f6f", fontsize=9)
    return to_array(fig)


def prefetching(clip: Clip, model, ticks: list[int], routes):
    """Yield (t0, route, data, prepared) with the next tick's CPU work running in a thread.

    The worker does CPU work only (decode, resize, tokenize). The host-to-GPU copy happens
    here, on the main thread: a CUDA call from another thread would break CUDA-graph capture.
    """

    def work(t0):
        route = routes(t0)
        data = clip.load(t0, route if isinstance(route, str) else route[0])
        prepared = [prepare_model_inputs({**data, "nav_text": [text]}, model.config, model.tokenizer)
                    for text in ([route] if isinstance(route, str) else route)]
        return t0, route, data, prepared

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(work, ticks[0])
        for index in range(len(ticks)):
            t0, route, data, prepared = future.result()
            if index + 1 < len(ticks):
                future = pool.submit(work, ticks[index + 1])
            yield t0, route, data, [to_device(p, device="cuda") for p in prepared]


def maybe_accelerate(model, enabled: bool):
    if not enabled:
        return None
    from alpamayo2_edge.accel import AccelConfig, enable_acceleration, resolve_kernels

    if resolve_kernels("auto", "cuda")["decode"] is None:
        print("flash-attn not importable: running eager")
        return None
    return enable_acceleration(model, AccelConfig(mode="tuned"))


def drive(args):
    clip = Clip(args.clip_id, CachedAVDI(avdi()))
    ticks = clip.tick_range(args.t_start, args.t_end, args.stride)
    extent = bev_extent(clip, ticks)
    print(f"clip {clip.id}: {len(ticks)} ticks {ticks[0] / 1e6:.1f}..{ticks[-1] / 1e6:.1f} s")
    model, _ = load_edge()
    accel = maybe_accelerate(model, args.accel)
    name = args.out or f"drive_{clip.id[:8]}.mp4"
    writer = Writer(VIDEOS / name, args.fps, (1920, 1080))
    start = time.perf_counter()
    for index, (t0, route, data, prepared) in enumerate(prefetching(clip, model, ticks, clip.route)):
        cands, cots, seconds = run_model(model, prepared[0], args.candidates)
        logged = clip.path(t0, 6.4)
        writer.write(render_drive(clip, t0, data, cands, cots, logged, route, seconds, extent,
                                  accel is not None))
        if index % 10 == 0:
            print(f"  t0 {t0 / 1e6:5.1f} s  {route!r:24s} {seconds:5.2f} s  {cots[0][:80]}"
                  f"   [{time.perf_counter() - start:6.0f} s elapsed]", flush=True)
    writer.close()


# ---------------------------------------------------------------------------------- route
def swap(route: str) -> str:
    if "right" in route:
        return route.replace("right", "left")
    if "left" in route:
        return route.replace("left", "right")
    return "Turn left in 30m" if route == "Continue straight" else route


def render_route(clip, t0_us, data, results, logged, extent):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#101010")
    frames = latest_frames(data)
    hist = data["ego_history_xyz"][0].numpy()
    fig.text(0.5, 0.965, f"Same images, same history, different route sentence  ·  clip "
             f"{clip.id[:8]}  ·  t0 = {t0_us / 1e6:5.1f} s", color="white", fontsize=17,
             weight="bold", ha="center")
    for side, (route, cands, cots) in enumerate(results):
        x0 = 0.01 + side * 0.5
        color = "#ffd23d" if side == 0 else "#ff8a3d"
        fig.text(x0, 0.915, ("Oracle route:  " if side == 0 else "Swapped route:  ") + route,
                 color=color, fontsize=15, weight="bold")
        ax = fig.add_axes((x0, 0.6, 0.29, 0.29))  # 0.29 of width x 0.29 of height = 16:9
        draw_camera(ax, clip, 1, frames[1], cands, logged, None, "Front")
        ax = fig.add_axes((x0, 0.30, 0.29, 0.29))
        draw_camera(ax, clip, 6, frames[6], cands, logged, None, "Front telephoto")
        ax = fig.add_axes((x0 + 0.302, 0.06, 0.183, 0.83))
        draw_bev(ax, clip, cands, logged, hist, extent, labels=False)
        ax.set_ylabel("")  # would overlap the telephoto panel; the x axis is labelled
        fig.text(x0, 0.265, "Reasoning (candidate 1):", color="#9ad7ff", fontsize=11)
        fig.text(x0, 0.25, wrap(cots[0].strip(), 70), color="white", fontsize=12.5, va="top")
    fig.text(0.01, 0.01, "Plans projected into the front and telephoto cameras; logged future "
             "dashed white (BEV: dashed black). Left: route derived from the logged path. Right: "
             "the same sentence with left/right swapped.", color="#7a7a7a", fontsize=9.5)
    return to_array(fig)


def route_video(args):
    clip = Clip(args.clip_id, CachedAVDI(avdi()))
    ticks = clip.tick_range(args.t_start, args.t_end, args.stride)
    extent = bev_extent(clip, ticks)
    extent = (extent[0], max(extent[1], 20.0))
    print(f"clip {clip.id}: {len(ticks)} ticks")
    model, _ = load_edge()
    maybe_accelerate(model, args.accel)
    writer = Writer(VIDEOS / (args.out or f"route_{clip.id[:8]}.mp4"), args.fps, (1920, 1080))
    for index, (t0, routes, data, prepared) in enumerate(
            prefetching(clip, model, ticks, lambda t: (clip.route(t), swap(clip.route(t))))):
        results = []
        for route, inputs in zip(routes, prepared, strict=True):
            cands, cots, _ = run_model(model, inputs, args.candidates)
            results.append((route, cands, cots))
        writer.write(render_route(clip, t0, data, results, clip.path(t0, 6.4), extent))
        if index % 10 == 0:
            print(f"  t0 {t0 / 1e6:5.1f} s  {routes}  {results[0][2][0][:50]} | "
                  f"{results[1][2][0][:50]}", flush=True)
    writer.close()


# ---------------------------------------------------------------------------------- flow
def flow_video(args):
    clip_id = args.clip_id or README_CLIP
    clip = Clip(clip_id, CachedAVDI(avdi()))
    t0 = int(round((args.t_start or 4.0) * 1e6))
    route = args.route or ("Turn right in 30m" if clip_id == README_CLIP else clip.route(t0))
    model, _ = load_edge()
    data = clip.load(t0, route)
    inputs = to_device(prepare_model_inputs(data, model.config, model.tokenizer), device="cuda")
    diffusion, captured = model.expert.diffusion, {}
    original = diffusion.sample

    def sample_all_steps(*a, **k):
        k["return_all_steps"] = True
        steps, times = original(*a, **k)
        captured["x"], captured["t"] = steps, times
        return steps[:, -1]

    diffusion.sample = sample_all_steps
    try:
        _, cots, _ = run_model(model, inputs, args.candidates)
    finally:
        diffusion.sample = original
    xs = captured["x"].float().cpu()  # [K, steps + 1, 64, 2]
    n_steps = xs.shape[1] - 1
    hist_xyz = inputs["ego_history_xyz"][:, -1].repeat_interleave(xs.shape[0], 0)
    hist_rot = inputs["ego_history_rot"][:, -1].repeat_interleave(xs.shape[0], 0)
    logged = clip.path(t0, 6.4)
    hist = data["ego_history_xyz"][0].numpy()
    frames = latest_frames(data)
    extent = ((-22.0, 70.0), 32.0)
    writer = Writer(VIDEOS / (args.out or f"flow_{clip_id[:8]}.mp4"), 20, (1920, 1080))
    sub = 8
    schedule = [0.0] * 15 + [i / sub for i in range(n_steps * sub + 1)] + [float(n_steps)] * 30
    seconds = np.arange(1, 65) * 0.1
    for position in schedule:
        low = min(int(np.floor(position)), n_steps - 1)
        frac = position - low
        x = xs[:, low] + frac * (xs[:, low + 1] - xs[:, low])  # exact: Euler segments are linear
        with torch.no_grad():
            xyz, _ = model.expert.action_space.action_to_traj(x.to(hist_xyz.device), hist_xyz,
                                                              hist_rot)
        cands = xyz.float().cpu().numpy()
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#101010")
        t = position / n_steps
        fig.text(0.01, 0.955, f"Flow matching: noise → plan  ·  t = {t:.2f}  ·  Euler step "
                 f"{min(int(position) + (frac > 0), n_steps)}/{n_steps}", color="white",
                 fontsize=20, weight="bold")
        fig.text(0.01, 0.915, f"clip {clip_id[:8]} at {t0 / 1e6:.1f} s · route {route!r} · "
                 f"{len(cands)} candidates, each with its own CoT and its own noise · "
                 "x ← x + 0.1·v(x, t), 10 steps", color="#bbbbbb", fontsize=12.5)
        ax = fig.add_axes((0.01, 0.42, 0.47, 0.47 * 19.2 / 10.8 * 9 / 16))
        draw_camera(ax, clip, 1, frames[1], cands, logged, None, "Front", ribbon_first=t > 0.6)
        ax = fig.add_axes((0.51, 0.06, 0.24, 0.83))
        draw_bev(ax, clip, cands, logged, hist, extent)
        for row, (label, channel) in enumerate((("normalized acceleration", 0),
                                                ("normalized curvature", 1))):
            ax = fig.add_axes((0.775, 0.55 - row * 0.45, 0.215, 0.34))
            ax.set_facecolor("#f7f6f2")
            for k in range(len(cands) - 1, -1, -1):
                ax.plot(seconds, x[k, :, channel].numpy(), color=BEV_PALETTE[k % 6],
                        lw=2 if k == 0 else 1)
            final = xs[:, -1, :, channel]
            ax.set_ylim(float(final.min()) - 2.5, float(final.max()) + 2.5)
            ax.set_title(label + " (what the expert denoises)", color="white", fontsize=11)
            ax.set_xlabel("seconds after t0", color="#bbbbbb", fontsize=9)
            ax.tick_params(colors="#bbbbbb", labelsize=8)
            ax.grid(color="#dddcd5", lw=0.6)
        fig.text(0.01, 0.35, "Reasoning (candidate 1):", color="#9ad7ff", fontsize=12)
        fig.text(0.01, 0.335, wrap(cots[0].strip(), 70), color="white", fontsize=14, va="top")
        fig.text(0.01, 0.03, "Each frame decodes the current x_t through the unicycle action "
                 "space. Between Euler steps the frames are exact points on the integration "
                 "path (the update is linear in x).", color="#7a7a7a", fontsize=10)
        writer.write(to_array(fig))
    writer.close()


parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["drive", "route", "flow"])
parser.add_argument("--clip-id", default=None)
parser.add_argument("--t-start", type=float, default=None, help="seconds into the clip")
parser.add_argument("--t-end", type=float, default=None)
parser.add_argument("--stride", type=float, default=0.1, help="seconds between inferences")
parser.add_argument("--fps", type=float, default=10.0)
parser.add_argument("--candidates", type=int, default=None)
parser.add_argument("--route", default=None, help="flow mode: route sentence")
parser.add_argument("--no-accel", dest="accel", action="store_false")
parser.add_argument("--out", default=None)
args = parser.parse_args()
if args.mode in ("drive", "route") and args.clip_id is None:
    args.clip_id = README_CLIP
args.candidates = args.candidates or {"drive": 6, "route": 3, "flow": 6}[args.mode]
{"drive": drive, "route": route_video, "flow": flow_video}[args.mode](args)
