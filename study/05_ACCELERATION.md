# 05 — Why eager inference is slow, and what `--accel` does about it

Read with [`docs/acceleration.md`](../docs/acceleration.md) (the upstream guide) open.
Measured by [`09_latency_vram.py`](scripts/09_latency_vram.py) (launch config **3-8**) on
the README example, RTX 3090, model call only (inputs already on the GPU). Full numbers in
`study/outputs/latency.json`.

## 1. Where an eager call spends its time

| part | eager, 1 CoT (23 tokens) | what bounds it |
|---|---|---|
| CPU `prepare_model_inputs` | **1 442 ms** (not in the call) | bicubic resize of 22 × 1080p frames, processor, tokenization |
| vision tower + prefill (3 260 tokens) | 295 ms | compute: ≈10 TFLOP of dense matmul at roughly the 3090's BF16 tensor-core peak |
| CoT decode | 425 ms = **18.5 ms/token** | Python + kernel-launch overhead: the weights (3.4 GB incl. lm_head) + 0.36 GB cache can be read in ≈4 ms, the remaining ≈14 ms is launching hundreds of small eager kernels per token from Python |
| expert, 10 Euler steps | 196 ms = 19.6 ms/step | memory traffic: every step re-concatenates the whole cache (`DynamicCache.update`) in FP32 (the expert's activations are FP32 under autocast) and expands KV heads for SDPA |
| **model call** | **934 ms** median | |

With 6 CoTs (33 tokens, the longest CoT sets the length): 1 455 ms; prefill is unchanged
(shared), decode 630 ms, expert 515 ms; peak memory 9.21 GB.

## 2. What `enable_acceleration` installs

`alpamayo2_edge.accel` replaces two methods **on the model instance** (nothing global,
`handle.remove()` restores them):

| part | replaces | mechanism |
|---|---|---|
| **decode** (`DecodeAccelerator`) | `_generate_with_shared_prefill` | a static, flash-layout `FlashKVCache` `[rows, max_len, 8, 128]` per layer; each token = one CUDA-graph replay of [compiled decoder step → the same logits processors → sampling → stop bookkeeping], driven through `generate(custom_generate=GraphSampler)` |
| **prefill** | (inside decode) | one CUDA-graph replay of the 28 layers over the prompt right-padded to a 256-token bucket, writing K/V straight into the static cache (only for a single CoT) |
| **expert** (`ExpertAccelerator`) | `_sample_actions` | the whole 10-step Euler loop is one graph; each step attends the static cache in place through `flash_attn_with_kvcache` instead of concatenating |

Kernels (`accel/kernels.py`): decode and expert use FlashAttention-2
`flash_attn_with_kvcache` wrapped as a `torch.library.custom_op` (so `torch.compile`
treats it as one opaque op); prefill uses SDPA, or FA3 on Hopper. On this 3090:
`{'prefill': 'sdpa', 'decode': 'fa2', 'expert': 'fa2'}`.

Determinism tricks worth reading in the code:

- **RNG inside graphs.** CUDA-graph RNG is Philox-offset based; `GraphedCall` saves and
  restores the RNG state around warm-up and capture, so a replay draws exactly what eager
  sampling would at that state.
- **`MaskTokenIdsLogitsProcessor` → `MaskColumns`.** `scores[:, ids] = -inf` copies the
  index list to the GPU on every call, which cannot be captured; `index_fill_` is the same
  write with a pre-allocated index.
- **Shapes are keys.** Graphs are cached per (rows, bucketed max length, sampler settings,
  autocast state). A prompt crossing a 256-token bucket, another resolution or another
  candidate count builds new graphs on first use.

## 3. Measured

| configuration | model call (median) | speedup | peak memory |
|---|---|---|---|
| eager, SDPA, 1 CoT (the default) | 934 ms | 1.00× | 6.18 GB |
| eager, `flash_attention_2`, 1 CoT | 1 061 ms | **0.88×** | 11.55 GB |
| **`--accel tuned`, 1 CoT** | **420 ms** | **2.22×** | 6.21 GB |
| eager, SDPA, 6 CoTs | 1 455 ms | 1.00× (6-CoT baseline) | 9.21 GB |
| **`--accel tuned`, 6 CoTs** | **898 ms** | **1.62×** | 9.13 GB |

Eager FA2 is *slower* than SDPA on this GPU (decode 23.5 vs 18.5 ms/token) and peaks almost
twice as high; FA2 only pays off inside the graphs, where it is called as
`flash_attn_with_kvcache` on the static cache. (Why HF's eager FA2 path costs more here was
not investigated.) With 6 CoTs the speedup is smaller because the prefill runs eagerly and
the expert graph replays only when all CoTs end at the same length (`fallbacks` in
`latency.json`). The notebook's own timing cell agrees: 1.45 s default, 0.90 s accelerated,
identical CoT text for all six candidates.

First accelerated call (compile + capture) with a warm inductor cache in `tmp/`: 10.9 s;
cold (the first CLI run): 67 s of graph builds. The CLI run with `--accel tuned` produced
the **same CoT text** as eager on the example, as the upstream doc predicts is likely but
not guaranteed (fused kernels round BF16 differently; a near-tied token can flip).

## 4. What is still on the table

- **CPU preprocessing (1.4 s)** is now 3× the accelerated model call. Resizing on the GPU
  (or in the camera pipeline) is the largest remaining win.
- **Fewer Euler steps**: one step is within 0.10 m ADE of ten on the example
  ([04 §6](04_ACTION_SPACE_AND_FLOW.md)); the expert graph is 10 sequential steps.
- **Streaming prefill**: at 10 Hz, most images in the prompt were already encoded one tick
  earlier. The unmerged `toyin/streaming-prefill-simulation` branch simulates reusing
  their K/V (Qwen3-VL era; not used here).
- **TensorRT / Thor**: out of scope for this repo; the README points to
  `NVlabs/alpamayo-recipes`.
