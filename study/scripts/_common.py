"""Shared helpers for the study scripts.

Everything here is a thin convenience layer over the repository's own API
(``alpamayo2_edge.*``); nothing reimplements the model.

* paths: repo root, local checkpoint, study outputs, sample cache
* ``load_sample`` / ``load_vqa_sample``: PhysicalAI-AV loaders, cached in ``tmp/``
  so a debugger restart does not re-stream 22 camera frames
* ``load_logged_future``: the path the car actually drove (reference only, never a
  model input)
* ``load_edge``: ``checkpoint.load_model`` on the local checkpoint
* ``derive_nav_text``: an *oracle* route instruction derived from the logged future
"""

from __future__ import annotations

import functools
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.environ.get("EDGE_MODEL", REPO / ".work" / "Alpamayo2-Edge"))
OUT = REPO / "study" / "outputs"
CACHE = REPO / "tmp" / "sample_cache"
MANIFEST = REPO / "examples" / "nav_samples.json"
# The notebook pins this PhysicalAI-AV commit; pinning it keeps every script on the same data.
DATASET_REVISION = "2ac2e9c056e0c8426face5bfad64b0961a295b28"
# CLI defaults (inference.py ``settings``).
DEFAULT_SETTINGS = dict(
    seed=43,
    top_p=0.98,
    top_k=None,
    temperature=0.6,
    num_traj_sets=1,
    num_traj_samples=1,
    max_generation_length=256,
    diffusion_kwargs={"inference_step": 10},
)
# The Alpamayo 1.5 repo's test clip (alpamayo1.5/src/alpamayo1_5/test_inference.py).
A15_GOLDEN_CLIP = ("030c760c-ae38-49aa-9ad8-f5650a545d26", 5_100_000)


def example() -> dict:
    """The README example: clip ea7bbd31..., t0 = 4.0 s, "Turn right in 30m"."""
    from alpamayo2_edge.inference import load_clip_manifest

    return load_clip_manifest(MANIFEST, 0)


@functools.cache
def avdi():
    import physical_ai_av

    return physical_ai_av.PhysicalAIAVDatasetInterface(revision=DATASET_REVISION)


def _cached(name: str, build):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if path.is_file():
        return torch.load(path, weights_only=False)
    value = build()
    torch.save(value, path)
    return value


def load_sample(clip_id: str, t0_us: int, nav_text: str) -> dict:
    """``load_physical_aiavdataset`` with an on-disk cache (nav text is not cached)."""
    from alpamayo2_edge.load_physical_aiavdataset import load_physical_aiavdataset

    data = _cached(
        f"drive_{clip_id}_{t0_us}.pt",
        lambda: load_physical_aiavdataset(clip_id, t0_us, nav_text="placeholder", avdi=avdi()),
    )
    return {**data, "nav_text": [nav_text]}


def load_vqa_sample(clip_id: str, t0_us: int) -> dict:
    from alpamayo2_edge.load_physical_aiavdataset import load_physical_aiavdataset_vqa

    return _cached(
        f"vqa_{clip_id}_{t0_us}.pt",
        lambda: load_physical_aiavdataset_vqa(clip_id, t0_us, avdi=avdi()),
    )


def load_logged_future(clip_id: str, t0_us: int, steps: int = 64) -> np.ndarray | None:
    """Logged ego path ``[steps, 3]`` at t0 + 0.1 s ... in the t0 ego frame (x fwd, y left)."""

    def build():
        ego = avdi().get_clip_feature(clip_id, avdi().features.LABELS.EGOMOTION, maybe_stream=True)
        times = t0_us + np.arange(0, steps * 100_000 + 1, 100_000)
        if times[-1] > ego.time_range[1]:
            return None
        poses = ego(times).pose
        xyz = np.asarray(poses.translation)
        return poses.rotation[0].inv().apply(xyz - xyz[0])[1:].astype(np.float32)

    return _cached(f"future_{clip_id}_{t0_us}_{steps}.pt", build)


def load_logged_future_pose(clip_id: str, t0_us: int, steps: int = 64):
    """Logged future ``(xyz [steps, 3], rot [steps, 3, 3])`` in the t0 ego frame, or None."""

    def build():
        ego = avdi().get_clip_feature(clip_id, avdi().features.LABELS.EGOMOTION, maybe_stream=True)
        times = t0_us + np.arange(0, steps * 100_000 + 1, 100_000)
        if times[-1] > ego.time_range[1]:
            return None
        poses = ego(times).pose
        xyz = np.asarray(poses.translation)
        inverse = poses.rotation[0].inv()
        local_xyz = inverse.apply(xyz - xyz[0])[1:].astype(np.float32)
        local_rot = (inverse * poses.rotation).as_matrix()[1:].astype(np.float32)
        return local_xyz, local_rot

    return _cached(f"future_pose_{clip_id}_{t0_us}_{steps}.pt", build)


def derive_nav_text(future_xyz: np.ndarray) -> str:
    """An oracle route instruction from the logged future, in the manifest's phrasing.

    A real stack gets this from the navigation system. Here the ground-truth path
    stands in for the route, so any metric computed with it is *route-conditioned*:
    the model is told which way the car turned, not where it drove.
    """
    xy = future_xyz[:, :2].astype(np.float64)
    seg = np.diff(np.vstack([np.zeros(2), xy]), axis=0)
    heading = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0] + 1e-9))
    moving = np.linalg.norm(seg, axis=1) > 0.05
    if moving.sum() < 5:
        return "Stop"
    heading = heading - heading[moving][0]
    arc = np.cumsum(np.linalg.norm(seg, axis=1))
    turned = np.nonzero(moving & (np.abs(heading) > np.deg2rad(30)))[0]
    if len(turned) == 0:
        return "Continue straight"
    side = "left" if heading[turned[0]] > 0 else "right"
    # Distance to where the turn starts (heading passes 10 degrees), rounded to 10 m.
    start = np.nonzero(np.abs(heading) > np.deg2rad(10))[0][0]
    meters = int(max(10, round(arc[start] / 10.0) * 10))
    return f"Turn {side} in {meters}m"


def load_edge(device: str = "cuda", attn: str = "sdpa", dtype: str = "bfloat16"):
    from alpamayo2_edge.checkpoint import load_model

    model, info = load_model(str(MODEL_DIR), device=device, dtype=dtype, attn_implementation=attn)
    return model, info


def prepare(model, data: dict, device: str = "cuda") -> dict:
    from alpamayo2_edge.helper import prepare_model_inputs, to_device

    return to_device(prepare_model_inputs(data, model.config, model.tokenizer), device=device)


def predict(model, inputs: dict, device: str = "cuda", **overrides):
    """``inference.predict`` with CLI defaults and keyword overrides."""
    from alpamayo2_edge.inference import predict as _predict

    settings = {**DEFAULT_SETTINGS, **overrides}
    return _predict(model, inputs, settings, device)


def ade(pred_xyz: np.ndarray, gt_xyz: np.ndarray, horizon: int | None = None) -> np.ndarray:
    """Per-candidate average displacement (xy) over the first ``horizon`` waypoints."""
    pred = pred_xyz.reshape(-1, pred_xyz.shape[-2], 3)[:, :horizon, :2]
    return np.linalg.norm(pred - gt_xyz[None, :horizon, :2], axis=-1).mean(-1)


def fde(pred_xyz: np.ndarray, gt_xyz: np.ndarray, horizon: int | None = None) -> np.ndarray:
    pred = pred_xyz.reshape(-1, pred_xyz.shape[-2], 3)[:, :horizon, :2]
    index = (horizon or pred.shape[1]) - 1
    return np.linalg.norm(pred[:, index] - gt_xyz[None, index, :2], axis=-1)


@contextmanager
def cuda_timer(label: str, sink: dict | None = None, device: str = "cuda"):
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    yield
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if sink is not None:
        sink[label] = elapsed
    print(f"  [{elapsed * 1e3:9.1f} ms] {label}")


def save_json(name: str, payload) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name

    def default(value):
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, (np.ndarray, torch.Tensor)):
            return value.tolist()
        return str(value)

    path.write_text(json.dumps(payload, indent=2, default=default) + "\n")
    print(f"wrote {path.relative_to(REPO)}")
    return path


def banner(text: str) -> None:
    print("\n" + "=" * 100 + f"\n{text}\n" + "=" * 100)
