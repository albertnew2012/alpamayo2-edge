# 03 — One inference call, stage by stage (and proven bit-exact)

`Alpamayo2Edge.sample_trajectories_from_data` is ~180 lines that hand most of the work to
HF `generate`. [`scripts/04_walkthrough.py`](scripts/04_walkthrough.py) performs the same
computation as nine explicit stages using the model's own submodules, then compares against
the library call made with the same seed. Launch config **3-4** runs it under the debugger.

## The verdict first

```
prefill identical       True     (last hidden state and all 56 K/V tensors, torch.equal)
CoT tokens identical    True     (23 tokens incl. the anchor)
trajectory identical    True     (max |dxyz| 0.000e+00 m)   rotations True
```

So the stage list below is not a paraphrase: it *is* what the library computes, down to
the bit. Every stage is a place to put a breakpoint.

## The stages, with measured shapes and times (RTX 3090, eager, SDPA)

| # | stage | code | output | time |
|---|---|---|---|---|
| 0 | inputs | `helper.prepare_model_inputs` (CPU) | `input_ids [1, 3261]`, `pixel_values [11568, 768]`, `image_grid_thw [22, 3]`, `mm_token_type_ids` | **1 365 ms** |
| 1 | fuse history | `models.utils.fuse_traj_tokens` | 57 ids replaced (`131655, 131571, 131572, …`) | 0.5 ms |
| 2 | vision tower | `vlm.model.visual(patches, grid_thw)` | `[11568, 1152]` fp32 | 144 ms |
| 3 | projector | `vlm.model.projector` | `[2892, 2048]` | 4 ms |
| 4 | scatter | `embed_tokens` + `masked_scatter` | `inputs_embeds [1, 3260, 2048]` | 0.4 ms |
| 5 | M-RoPE | `vlm.model.get_rope_index` | `position_ids [3, 1, 3260]`, `rope_delta −2556` | 5 ms |
| 6 | prefill | `vlm.model.language_model(...)` | cache 28 × `[1, 8, 3260, 128]` (357 MiB) | 139 ms |
| 7 | decode | loop: `language_model` 1 token → `lm_head` → processors → `multinomial` | 23 tokens, cache 3 284 | 393 ms (17.1 ms/token) |
| 8 | expert | 10 × (`action_in_proj` → `expert.expert` → `crop` → `action_out_proj` → Euler) | `x [1, 64, 2]` | 188 ms (18.8 ms/step) |
| 9 | integrate | `action_space.action_to_traj` | `pred_xyz [1, 64, 3]`, `pred_rot [1, 64, 3, 3]` | 2 ms |

GPU stages total **0.87 s**; CPU preprocessing is **1.37 s**. On this machine the CPU
side (bicubic resize of 22 × 1080p frames in uint8, processor, tokenization) costs more than
the whole model. That is the first thing to move to the GPU in a deployment.

## Details that are easy to get wrong

### The prefill stops one token short
The library prefills `input_ids[:, :-1]` (`mm_token_type_ids[:, :-1]` too) and lets
`generate` forward the last prompt token (`</think>`), whose logits give the first CoT
token. All image tokens precede it, so the vision tower runs once at batch 1.

### Decode positions are `cache_length + rope_delta`
After the prompt, every new token is text, so all three M-RoPE components equal
`cache_length + rope_delta` (= 704, 705, …). The walkthrough passes exactly that as a
`[3, 1, 1]` tensor. This is also what `accel.decode.DecodeStep` computes inside its graph.

### The sampler's processor order matters for bit-exactness
HF merges the custom processors first and appends the warpers after:

```python
scores[:, 131072:135072] = -inf      # MaskDiscreteTrajectoryLogitsProcessor ×2 (history, future bins)
scores[:, [11]]          = -inf      # MaskTokenIdsLogitsProcessor: <|im_end|>
scores = scores / 0.6                # TemperatureLogitsWarper
# TopPLogitsWarper: sort ascending, cumsum(softmax) <= 1-0.98 → -inf, always keep the top token
next = torch.multinomial(softmax(scores), 1)
```

`predict()` calls `torch.manual_seed(seed)` before the call and nothing else draws from
the global CUDA RNG before the first `multinomial`, so seeding the same way reproduces the
same tokens.

### Why generation runs one step past the anchor
`generate` stores in the cache the token it *feeds* into the model, not the token it just
*sampled*. When `<|traj_future_start|>` is sampled it is not cached yet; `StopAfterEOS`
therefore lets one more step run (which feeds the anchor and samples a throwaway token).
The walkthrough does the same thing explicitly: once the anchor has been sampled, forward
it and stop. The expert's 64 action tokens are positioned **right after the anchor**
(positions 728…791 here).

### The expert's cache is cropped after every step
`DynamicCache.update` appends the expert's 64 K/V to each layer; `cache.crop(prefix)`
removes them before the next step, so step *k* never sees step *k−1*'s action tokens.
Under eager attention this means re-concatenating the full prompt cache 10 times (and in
FP32, because the expert's activations are FP32 under autocast). That, not FLOPs, is why an
expert step costs 18.8 ms; the accelerated path attends the static cache in place (see
[05](05_ACCELERATION.md)).

### `num_traj_sets` × `num_traj_samples`
Both just multiply the number of independent CoTs (`n_samples_total = sets × samples`);
the shapes `[B, sets, samples, 64, 3]` are a grouping convention. Every candidate gets its
own CoT **and** its own noise. Prefill is shared; decode and the expert run at batch
`B × n_samples_total`. CoTs of different length are handled by the 4-D mask (cache columns
beyond a row's own anchor are −inf) and by padding the tokens after each anchor.

### The third return value is a placeholder
`logprob` is `zeros_like(pred_xyz[..., 0])`. The README says so; do not rank candidates
with it.

## Where to put breakpoints (launch config 3-4)

| line in `04_walkthrough.py` | what to inspect |
|---|---|
| `[2] vision tower` | `vis.shape`, `td["image_grid_thw"]` |
| `[5] M-RoPE` | `position_ids[:, 0, 30:130]`, the jump after image 0 |
| `[6] prefill` | `cache.layers[0].keys.shape`, `same_prefill` |
| inside `next_token` | `scores.isinf().sum()` (= 4001 masked), the top-p cutoff |
| inside the Euler loop | `x` after each step, `cache.get_seq_length()` before and after `crop` |
| `[9]` | `xyz[0, ::10]`, compare with `ref_xyz` |

Next: [04 — action space and flow matching](04_ACTION_SPACE_AND_FLOW.md).
