"""Session 8: where the time and memory go - eager vs ``--accel tuned``, SDPA vs FA2.

    python study/scripts/09_latency_vram.py [--repeats 5]

Model-call latency only (inputs already prepared and on the GPU); the CPU-side
``prepare_model_inputs`` is timed separately because it is not small. For the eager
path, the call is split into vision+prefill, CoT decode and the expert by wrapping the
model's own methods on the instance. Writes study/outputs/latency.json.
"""

import argparse
import functools
import gc
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import banner, example, load_edge, load_sample, predict, save_json  # noqa: E402

from alpamayo2_edge.helper import prepare_model_inputs, to_device  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--repeats", type=int, default=5)
args = parser.parse_args()
spec = example()
data = load_sample(**spec)
results = {"gpu": torch.cuda.get_device_name(0)}


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - start


def wrap_stages(model, sink):
    """Time vision+prefill (first vlm.model call), decode (vlm.generate) and the expert."""
    originals = {}

    def install(owner, name, label):
        original = getattr(owner, name)
        originals[(owner, name)] = original

        @functools.wraps(original)
        def wrapper(*a, **k):
            if label == "prefill" and sink.get("_prefill_done"):
                return original(*a, **k)
            out, seconds = timed(lambda: original(*a, **k))
            sink.setdefault(label, []).append(seconds)
            if label == "prefill":
                sink["_prefill_done"] = True
            return out

        setattr(owner, name, wrapper)

    install(model.vlm.model, "forward", "prefill")
    install(model.vlm, "generate", "decode")
    install(model, "_sample_actions", "expert")

    def remove():
        for (owner, name), _ in originals.items():
            delattr(owner, name) if name in vars(owner) else None

    return remove


def run(model, inputs, label, sets, repeats, stages=False):
    predict(model, inputs, num_traj_sets=sets)  # warm-up / graph capture
    times, tokens, split = [], [], {}
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        sink = {}
        remove = wrap_stages(model, sink) if stages else (lambda: None)
        (xyz, _, extra), seconds = timed(lambda: predict(model, inputs, num_traj_sets=sets))
        remove()
        times.append(seconds)
        ids = extra["generated_ids"]
        tokens.append(int((ids >= 0).sum(1).max()))
        for key in ("prefill", "decode", "expert"):
            if key in sink:
                split.setdefault(key, []).append(sum(sink[key]))
    row = {
        "median_s": statistics.median(times), "min_s": min(times),
        "cot_tokens_max": statistics.median(tokens),
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    if split:
        row["stages_median_s"] = {k: statistics.median(v) for k, v in split.items()}
        row["decode_ms_per_token"] = 1e3 * row["stages_median_s"]["decode"] / row["cot_tokens_max"]
    results[label] = row
    print(f"  {label:34s} median {row['median_s'] * 1e3:7.0f} ms   min {row['min_s'] * 1e3:7.0f} ms"
          f"   CoT {row['cot_tokens_max']:4.0f} tok   peak {row['peak_gb']:.2f} GB")
    if split:
        print("      " + "   ".join(f"{k} {v * 1e3:.0f} ms" for k, v in row["stages_median_s"].items())
              + f"   ({row['decode_ms_per_token']:.1f} ms/token)")
    return row


banner("CPU preprocessing")
model, _ = load_edge()
_, prep = timed(lambda: prepare_model_inputs(data, model.config, model.tokenizer))
results["prepare_model_inputs_s"] = prep
print(f"  prepare_model_inputs (sort, bicubic resize of 22 x 1080p, normalize, tokenize): "
      f"{prep * 1e3:.0f} ms")
inputs = to_device(prepare_model_inputs(data, model.config, model.tokenizer), device="cuda")
print(f"  weights resident: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

banner("Eager, SDPA (the default path)")
run(model, inputs, "eager_sdpa_1cot", 1, args.repeats, stages=True)
run(model, inputs, "eager_sdpa_6cot", 6, max(2, args.repeats // 2), stages=True)

banner("--accel tuned (torch.compile + CUDA graphs, FA2 decode/expert)")
from alpamayo2_edge.accel import AccelConfig, enable_acceleration  # noqa: E402

accel = enable_acceleration(model, AccelConfig(mode="tuned"))
_, build = timed(lambda: accel.warmup(inputs))
print(f"  first call (compile + capture): {build:.1f} s")
run(model, inputs, "accel_tuned_1cot", 1, args.repeats)
run(model, inputs, "accel_tuned_6cot", 6, max(2, args.repeats // 2))
results["accel_describe"] = {k: v for k, v in accel.describe().items() if k != "builds"}
accel.remove()
del model, inputs
gc.collect()
torch.cuda.empty_cache()

banner("Eager, flash_attention_2 attention backend")
model, _ = load_edge(attn="flash_attention_2")
inputs = to_device(prepare_model_inputs(data, model.config, model.tokenizer), device="cuda")
run(model, inputs, "eager_fa2_1cot", 1, args.repeats, stages=True)

banner("Summary")
base = results["eager_sdpa_1cot"]["median_s"]
for key in ("eager_sdpa_1cot", "eager_fa2_1cot", "accel_tuned_1cot", "eager_sdpa_6cot",
            "accel_tuned_6cot"):
    print(f"  {key:20s} {results[key]['median_s'] * 1e3:7.0f} ms   x{base / results[key]['median_s']:.2f}"
          f" vs eager 1-CoT")
save_json("latency.json", results)
