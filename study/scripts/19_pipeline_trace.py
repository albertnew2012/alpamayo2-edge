"""Pipeline 3-1, traced: every tensor between the camera frames and the 6 trajectories.

    python study/scripts/19_pipeline_trace.py [--candidates 6]

Runs exactly the call config 3-1 makes in ``06_demo_bev.py`` §2
(``predict(model, inputs, num_traj_sets=6)``: README clip, seed 43, SDPA, bf16) and
records shapes, cache sizes, masks and timings at each stage. Nothing is reimplemented:
the model's own methods are wrapped on the instance (plus two helpers in the model's
module namespace), the wrappers only observe, and the CoTs are checked against
``study/outputs/demo.json`` from the 3-1 run.

One untraced warm-up call runs first (``predict`` re-seeds, so it does not change the
result). The wrappers synchronize CUDA around every call they watch, which inflates the
decode timings a little. Writes study/outputs/pipeline_trace.json; the doc built from it
is study/12_PIPELINE_3-1.md.
"""

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import torch

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

import alpamayo2_edge.models.alpamayo2_edge as edge_module  # noqa: E402
from alpamayo2_edge.helper import prepare_model_inputs  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--candidates", type=int, default=6)
args = parser.parse_args()
N = args.candidates
HISTORY_BINS = (131_072, 132_072)  # <i0> ... <i999>

spec = example()
data = load_sample(**spec)
model, _ = load_edge()
start = time.perf_counter()
prepare_model_inputs(data, model.config, model.tokenizer)
cpu_prepare_ms = (time.perf_counter() - start) * 1e3
inputs = prepare(model, data)

banner("warm-up (untraced)")
predict(model, inputs, num_traj_sets=N)

calls: dict[str, list] = {}


def shape(value):
    return list(value.shape) if isinstance(value, torch.Tensor) else None


def cache_info(cache) -> dict:
    keys = cache.layers[0].keys
    nbytes = sum(t.numel() * t.element_size() for layer in cache.layers for t in (layer.keys, layer.values))
    return {"layers": len(cache.layers), "layer0_keys": shape(keys), "dtype": str(keys.dtype),
            "mib": nbytes / 2**20}


def watch(label, owner, name, record):
    """Wrap ``owner.name``; after each call store ``record(args, kwargs, out)`` plus its time."""
    original = getattr(owner, name)

    @functools.wraps(original)
    def wrapper(*a, **kw):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = original(*a, **kw)
        torch.cuda.synchronize()
        entry = record(a, kw, out)
        entry["ms"] = (time.perf_counter() - t0) * 1e3
        calls.setdefault(label, []).append(entry)
        return out

    setattr(owner, name, wrapper)


def rec_backbone(a, kw, out):
    ids = kw["input_ids"]
    entry = {"input_ids": shape(ids), "has_pixel_values": kw.get("pixel_values") is not None,
             "last_hidden_state": shape(out.last_hidden_state)}
    if entry["has_pixel_values"]:  # the shared prefill (alpamayo2_edge.py:139)
        entry.update(
            image_tokens=int((kw["mm_token_type_ids"] == 1).sum()),
            history_tokens=int(((ids >= HISTORY_BINS[0]) & (ids < HISTORY_BINS[1])).sum()),
            attention_mask=shape(kw["attention_mask"]),
            pixel_values=shape(kw["pixel_values"]),
            cache_after=cache_info(out.past_key_values),
        )
    return entry


def rec_image_features(a, kw, out):
    pooled = out.pooler_output
    grid = a[1]
    return {"pixel_values": shape(a[0]), "image_grid_thw": grid.tolist(),
            "cu_seqlens": [0] + (grid[:, 1] * grid[:, 2]).cumsum(0).tolist(),
            "images": len(pooled), "tokens_per_image": [int(p.shape[0]) for p in pooled],
            "image_embeds": [sum(int(p.shape[0]) for p in pooled), int(pooled[0].shape[1])]}


def rec_in_out(a, kw, out):
    hidden = getattr(out, "last_hidden_state", out)
    return {"in": shape(a[0]), "out": shape(hidden), "out_dtype": str(hidden.dtype)}


def rec_rope(a, kw, out):
    position_ids, deltas = out
    p = position_ids[:, 0]
    return {"position_ids": shape(position_ids), "max_position": int(position_ids.max()),
            "rope_deltas": deltas.flatten().tolist(),
            "distinct_values_on_t_axis": int(p[0].unique().numel()),
            "tokens_whose_t_h_w_differ": int(((p[0] != p[1]) | (p[0] != p[2])).sum()),
            "positions_tokens_30_to_51": p[:, 30:52].tolist()}


def rec_generate(a, kw, out):
    return {"input_ids": shape(kw["input_ids"]), "attention_mask": shape(kw["attention_mask"]),
            "sequences": shape(out.sequences),
            "rope_deltas_on_model": model.vlm.model.rope_deltas.flatten().tolist(),
            "cache_after": cache_info(kw["past_key_values"])}


def rec_find_eos(a, kw, out):
    return {"searched": shape(a[0]), "first_anchor_index_plus_1": out.tolist()}


def rec_build_mask(a, kw, out):
    position_ids, mask = out
    prefix = kw["kv_cache_seq_len"]
    blocked = (mask[:, 0, 0, :prefix] < 0).sum(-1)
    return {"offset": kw["offset"].tolist(), "rope_deltas": kw["rope_deltas"].flatten().tolist(),
            "kv_cache_seq_len": prefix, "n_diffusion_tokens": kw["n_diffusion_tokens"],
            "position_ids": shape(position_ids),
            "first_position_per_row": position_ids[0, :, 0].tolist(),
            "last_position_per_row": position_ids[0, :, -1].tolist(),
            "same_on_t_h_w": bool((position_ids[0] == position_ids[1]).all()
                                  and (position_ids[0] == position_ids[2]).all()),
            "attention_mask": shape(mask), "mask_dtype": str(mask.dtype),
            "blocked_cache_columns_per_row": blocked.tolist(),
            "visible_cache_columns_per_row": (prefix - blocked).tolist(),
            "own_action_columns_all_visible": bool((mask[:, 0, :, prefix:] == 0).all())}


def rec_diffusion(a, kw, out):
    return {"batch_size": kw["batch_size"], "inference_step": kw.get("inference_step"),
            "out": shape(out)}


def rec_in_proj(a, kw, out):
    x, t = a[0], a[1]
    return {"x": shape(x), "t": shape(t), "t_value": round(float(t.flatten()[0]), 4),
            "x_std": float(x.float().std()), "out": shape(out), "out_dtype": str(out.dtype)}


def rec_expert(a, kw, out):
    return {"inputs_embeds": shape(kw["inputs_embeds"]), "position_ids": shape(kw["position_ids"]),
            "attention_mask": shape(kw["attention_mask"]),
            "cache_len_after_forward": kw["past_key_values"].get_seq_length(),
            "last_hidden_state": shape(out.last_hidden_state)}


def rec_to_traj(a, kw, out):
    return {"action": shape(a[0]), "history_xyz_last": shape(a[1]), "history_rot_last": shape(a[2]),
            "xyz": shape(out[0]), "rot": shape(out[1])}


vlm, expert = model.vlm.model, model.expert
watch("backbone", vlm, "forward", rec_backbone)
watch("image_features", vlm, "get_image_features", rec_image_features)
watch("vision_tower", vlm.visual, "forward", rec_in_out)
watch("projector", vlm.projector, "forward", rec_in_out)
watch("rope", vlm, "get_rope_index", rec_rope)
watch("generate", model.vlm, "generate", rec_generate)
watch("find_eos_offset", edge_module, "find_eos_offset", rec_find_eos)
watch("expert_geometry", edge_module, "build_expert_pos_ids_and_attn_mask", rec_build_mask)
watch("flow_matching", expert.diffusion, "sample", rec_diffusion)
watch("action_in_proj", expert.action_in_proj, "forward", rec_in_proj)
watch("expert", expert.expert, "forward", rec_expert)
watch("action_out_proj", expert.action_out_proj, "forward", rec_in_out)
watch("action_to_traj", expert.action_space, "action_to_traj", rec_to_traj)

banner(f"traced call: predict(model, inputs, num_traj_sets={N})   (06_demo_bev.py:89)")
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
t0 = time.perf_counter()
xyz, rot, extra = predict(model, inputs, num_traj_sets=N)
torch.cuda.synchronize()
model_call_ms = (time.perf_counter() - t0) * 1e3

prefill = [c for c in calls["backbone"] if c["has_pixel_values"]]
decode = [c for c in calls["backbone"] if not c["has_pixel_values"]]
assert len(prefill) == 1 and len(calls["expert"]) == len(calls["action_in_proj"])
trace = {
    "sample": spec,
    "candidates": N,
    "cpu_prepare_ms": cpu_prepare_ms,
    "model_call_ms": model_call_ms,
    "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
    "image_features": calls["image_features"][0],
    "vision_tower": calls["vision_tower"][0],
    "projector": calls["projector"][0],
    "rope": calls["rope"][0],
    "prefill": prefill[0],
    "generate": calls["generate"][0],
    "decode": {"steps": len(decode), "input_per_step": decode[0]["input_ids"],
               "ms_total": sum(c["ms"] for c in decode),
               "ms_per_step": sum(c["ms"] for c in decode) / len(decode)},
    "find_eos_offset": calls["find_eos_offset"][0],
    "expert_geometry": calls["expert_geometry"][0],
    "flow_matching": calls["flow_matching"][0],
    "action_in_proj": {k: v for k, v in calls["action_in_proj"][0].items() if k not in ("t_value", "x_std")},
    "expert_forward": calls["expert"][0],
    "action_out_proj": calls["action_out_proj"][0],
    "euler_steps": [{"t": i["t_value"], "x_std": i["x_std"], "expert_ms": e["ms"],
                     "cache_len_during_forward": e["cache_len_after_forward"]}
                    for i, e in zip(calls["action_in_proj"], calls["expert"], strict=True)],
    "expert_ms_total": sum(c["ms"] for c in calls["expert"]),
    "action_to_traj": calls["action_to_traj"][0],
    "outputs": {"pred_xyz": shape(xyz), "pred_rot": shape(rot), "extra_keys": sorted(extra),
                "generated_ids": shape(extra["generated_ids"]),
                "cot_tokens_incl_anchor": (extra["generated_ids"] >= 0).sum(1).tolist(),
                "cot": [str(c) for c in extra["cot"].reshape(-1)]},
}
demo = OUT / "demo.json"
if demo.is_file():
    reference = [c["cot"] for c in json.loads(demo.read_text())["candidates"]]
    trace["cot_matches_3-1_demo_json"] = reference == trace["outputs"]["cot"]
save_json("pipeline_trace.json", trace)

banner("summary")
p, g, m = trace["prefill"], trace["generate"], trace["expert_geometry"]
rows = [
    ("CPU prepare_model_inputs", f"{cpu_prepare_ms:.0f} ms"),
    ("vision tower", f"{trace['vision_tower']['in']} -> {trace['vision_tower']['out']}  "
                     f"{trace['vision_tower']['ms']:.1f} ms"),
    ("projector", f"{trace['projector']['in']} -> {trace['projector']['out']}  {trace['projector']['ms']:.1f} ms"),
    ("prefill (incl. vision)", f"ids {p['input_ids']}  cache {p['cache_after']['layer0_keys']} "
                               f"{p['cache_after']['mib']:.0f} MiB  {p['ms']:.1f} ms"),
    ("M-RoPE", f"max position {trace['rope']['max_position']}  rope_deltas {trace['rope']['rope_deltas']}"),
    ("generate", f"sequences {g['sequences']}  cache {g['cache_after']['layer0_keys']} "
                 f"{g['cache_after']['mib']:.0f} MiB  {g['ms']:.1f} ms, {trace['decode']['steps']} steps"),
    ("first anchor (+1) per row", f"{trace['find_eos_offset']['first_anchor_index_plus_1']}"),
    ("expert positions", f"{m['first_position_per_row']} ... {m['last_position_per_row']}"),
    ("expert mask", f"{m['attention_mask']}  blocked per row {m['blocked_cache_columns_per_row']}"),
    ("expert, 10 Euler steps", f"{trace['expert_ms_total']:.1f} ms"),
    ("outputs", f"xyz {trace['outputs']['pred_xyz']}  rot {trace['outputs']['pred_rot']}"),
    ("model call", f"{model_call_ms:.0f} ms   peak {trace['peak_gpu_gib']:.2f} GiB"),
    ("CoTs equal 3-1 demo.json", f"{trace.get('cot_matches_3-1_demo_json')}"),
]
for label, value in rows:
    print(f"  {label:28s} {value}")
