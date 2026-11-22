"""Session 4: one inference call, unrolled stage by stage and checked bit-exact.

    python study/scripts/04_walkthrough.py [--steps 10] [--seed 43]

``Alpamayo2Edge.sample_trajectories_from_data`` is ~180 lines spread over HF
``generate``. This script performs the same computation as nine explicit stages, calling
the model's own submodules (no kernel is reimplemented), prints every tensor shape and
the time per stage, and finally compares against the library call made with the same
seed:

  0  inputs           prepare_model_inputs (CPU): resize, normalize, tokenize
  1  fuse history     57 <|traj_history|> placeholders -> discrete delta tokens
  2  vision tower     SigLIP2, 27 blocks, per-image attention
  3  projector        2x2 merge + MLP -> 2048-d LLM tokens
  4  scatter          image embeddings replace <|image_pad|> embeddings
  5  M-RoPE           3-D (t, h, w) positions + rope delta
  6  prefill          28 decoder layers over the prompt -> KV cache
  7  decode           sample the CoT token by token (own sampler, HF's processor order)
  8  expert           10 Euler steps of the flow-matching expert reading the KV cache
  9  action -> traj   unicycle integration -> 64 poses

Expected result: the CoT tokens and the 64 x 3 trajectory are *identical* (torch.equal)
to the library's. Set a breakpoint anywhere and inspect.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    banner,
    cuda_timer,
    example,
    load_edge,
    load_sample,
    predict,
    save_json,
)

from alpamayo2_edge.helper import prepare_model_inputs, to_device  # noqa: E402
from alpamayo2_edge.models.expert_utils import build_expert_pos_ids_and_attn_mask  # noqa: E402
from alpamayo2_edge.models.utils import fuse_traj_tokens  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=10, help="flow-matching Euler steps")
parser.add_argument("--seed", type=int, default=43)
parser.add_argument("--max-new-tokens", type=int, default=256)
args = parser.parse_args()
DEVICE = "cuda"
TOP_P, TEMPERATURE = 0.98, 0.6

model, _ = load_edge(DEVICE)
cfg = model.config
spec = example()
data = load_sample(**spec)
T = {}  # stage timings


def show(name: str, tensor: torch.Tensor) -> None:
    print(f"      {name:34s} {str(tuple(tensor.shape)):22s} {str(tensor.dtype)}")


banner("Reference: the library call (inference.predict, CLI defaults)")
inputs = to_device(prepare_model_inputs(data, cfg, model.tokenizer), device=DEVICE)
with cuda_timer("library sample_trajectories_from_data", T):
    ref_xyz, ref_rot, ref_extra = predict(model, inputs, seed=args.seed,
                                          diffusion_kwargs={"inference_step": args.steps})
print("  CoT:", ref_extra["cot"].reshape(-1)[0])

banner("Unrolled")
vlm, text, expert = model.vlm, model.vlm.model.language_model, model.expert
ctx = (torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16))
with ctx[0], ctx[1]:
    # ---------------------------------------------------------------- 0. inputs
    print("\n[0] prepare_model_inputs  (CPU: sort, resize uint8, normalize, tokenize)")
    with cuda_timer("0 prepare_model_inputs", T):
        prepared = to_device(prepare_model_inputs(data, cfg, model.tokenizer), device=DEVICE)
    td = prepared["tokenized_data"]
    for key in ("input_ids", "attention_mask", "mm_token_type_ids", "pixel_values",
                "image_grid_thw"):
        show(key, td[key])

    # ---------------------------------------------------------------- 1. fuse history
    print("\n[1] fuse_traj_tokens: 57 placeholders -> history delta bins (<i0>..<i999>)")
    traj = {k: prepared[k] for k in ("ego_history_xyz", "ego_history_rot")}
    with cuda_timer("1 fuse history tokens", T):
        input_ids = fuse_traj_tokens(model.history_traj_tokenizer, model.future_traj_tokenizer,
                                     td["input_ids"], traj, cfg.traj_ids)
    changed = (input_ids != td["input_ids"]).sum().item()
    print(f"      {changed} ids replaced; e.g. {input_ids[0, (td['input_ids'][0] == cfg.traj_ids['history_pad']).nonzero()[:6, 0]].tolist()}")
    prompt_len = input_ids.shape[1]

    # The library prefills all but the last prompt token; generate() feeds the last one
    # and produces the first CoT logits from it.
    pre_ids = input_ids[:, :-1]
    pre_mask = td["attention_mask"][:, :-1]
    pre_types = td["mm_token_type_ids"][:, :-1]

    # ---------------------------------------------------------------- 2. vision tower
    print("\n[2] vision tower: SigLIP2 ViT, 27 blocks x 1152, attention within each image")
    visual = vlm.model.visual
    with cuda_timer("2 vision tower", T):
        patches = td["pixel_values"].type(visual.dtype)
        vis = visual(patches, grid_thw=td["image_grid_thw"], return_dict=True).last_hidden_state
    show("pixel_values (patches x 16*16*3)", patches)
    show("vision last_hidden_state", vis)

    # ---------------------------------------------------------------- 3. projector
    print("\n[3] projector: LayerNorm -> group 2x2 patches (4 x 1152 = 4608) -> MLP -> 2048")
    with cuda_timer("3 projector", T):
        image_embeds = vlm.model.projector(vis)
    show("image_embeds", image_embeds)

    # ---------------------------------------------------------------- 4. scatter
    print("\n[4] token embeddings, image rows overwritten (masked_scatter)")
    with cuda_timer("4 embed + scatter", T):
        embeds = text.embed_tokens(pre_ids)
        image_mask = (pre_ids == cfg.vlm_config.image_token_id)[..., None]
        embeds = embeds.masked_scatter(image_mask, image_embeds.to(embeds.dtype))
    show("inputs_embeds", embeds)
    print(f"      image rows {int(image_mask.sum())} of {pre_ids.shape[1]}")

    # ---------------------------------------------------------------- 5. M-RoPE
    print("\n[5] 3-D rotary positions: text advances 1/token, an h x w image advances max(h, w)")
    with cuda_timer("5 get_rope_index", T):
        position_ids, rope_delta = vlm.model.get_rope_index(
            pre_ids, mm_token_type_ids=pre_types, image_grid_thw=td["image_grid_thw"],
            attention_mask=pre_mask)
    show("position_ids (t, h, w)", position_ids)
    print(f"      rope_delta {rope_delta.item()}; last prompt position "
          f"{position_ids[0, 0, -1].item() + 1}")

    # ---------------------------------------------------------------- 6. prefill
    print("\n[6] prefill: 28 decoder layers over the prompt, writing the KV cache")
    with cuda_timer("6 prefill (28 layers)", T):
        out = text(inputs_embeds=embeds, position_ids=position_ids, attention_mask=pre_mask,
                   use_cache=True)
    cache = out.past_key_values
    show("last_hidden_state", out.last_hidden_state)
    show("cache layer 0 keys [B, kv, L, d]", cache.layers[0].keys)
    print(f"      cache: 28 layers x (K, V), {cache.get_seq_length()} tokens, "
          f"{sum(lay.keys.numel() + lay.values.numel() for lay in cache.layers) * 2 / 2**20:.0f} MiB")

    # Library prefill on the same inputs, for an exact check of stages 2-6.
    lib = vlm.model(input_ids=pre_ids, attention_mask=pre_mask, use_cache=True,
                    pixel_values=td["pixel_values"], image_grid_thw=td["image_grid_thw"],
                    mm_token_type_ids=pre_types)
    same_prefill = torch.equal(lib.last_hidden_state, out.last_hidden_state) and all(
        torch.equal(a.keys, b.keys) and torch.equal(a.values, b.values)
        for a, b in zip(lib.past_key_values.layers, cache.layers, strict=True))
    print(f"      == vlm.model(...) prefill, hidden state and all 56 K/V tensors: {same_prefill}")
    del lib

    # ---------------------------------------------------------------- 7. decode
    print("\n[7] decode: sample the CoT until <|traj_future_start|> (the expert's anchor)")
    anchor = cfg.traj_ids["future_start"]
    blocked = [cfg.vlm_config.text_config.eos_token_id]  # <|im_end|> cannot end the CoT
    lo, hi = cfg.traj_ids["history_id0"], cfg.traj_ids["future_id0"] + cfg.future_vocab_size

    def next_token(logits: torch.Tensor) -> torch.Tensor:
        # Same order as HF generate: custom processors, then temperature, then top-p.
        scores = logits.float()
        scores[:, lo:hi] = -torch.inf  # 4000 trajectory bins (two processors in the library)
        scores[:, blocked] = -torch.inf
        scores = scores / TEMPERATURE
        sorted_scores, order = torch.sort(scores, descending=False)
        remove = sorted_scores.softmax(-1).cumsum(-1) <= 1 - TOP_P
        remove[..., -1:] = False
        scores = scores.masked_fill(remove.scatter(1, order, remove), -torch.inf)
        return torch.multinomial(scores.softmax(-1), num_samples=1)

    torch.manual_seed(args.seed)  # predict() seeds the global RNG the same way
    token = input_ids[:, -1:]
    generated = []
    with cuda_timer("7 decode loop", T):
        for _ in range(args.max_new_tokens):
            length = cache.get_seq_length()
            positions = (length + rope_delta).view(1, 1, 1).expand(3, 1, 1)
            step = text(inputs_embeds=text.embed_tokens(token), position_ids=positions,
                        attention_mask=torch.ones(1, length + 1, dtype=torch.long, device=DEVICE),
                        past_key_values=cache, use_cache=True)
            if generated and generated[-1] == anchor:
                break  # the anchor is now cached: that is all the expert needs
            token = next_token(vlm.lm_head(step.last_hidden_state[:, -1:])[:, -1])
            generated.append(token.item())
    n_cot = len(generated)
    print(f"      {n_cot} tokens generated (incl. anchor); cache now {cache.get_seq_length()} long")
    print("      CoT:", model.tokenizer.decode(generated[:-1]).replace("<think></think>", ""))
    ref_ids = ref_extra["generated_ids"][0]
    ref_ids = ref_ids[ref_ids >= 0].tolist()
    same_cot = ref_ids == generated
    print(f"      == library generated_ids: {same_cot}")
    print(f"      per token: {T['7 decode loop'] / n_cot * 1e3:.1f} ms")

    # ---------------------------------------------------------------- 8. expert
    print(f"\n[8] expert: {args.steps} Euler steps, 64 action tokens attend [prompt + CoT] + themselves")
    prefix = cache.get_seq_length()
    offset = torch.tensor([prefix], device=DEVICE)
    expert_pos, expert_mask = build_expert_pos_ids_and_attn_mask(
        offset=offset, rope_deltas=rope_delta, kv_cache_seq_len=prefix, n_diffusion_tokens=64,
        b_star=1, device=DEVICE, prefix_mask=torch.ones(1, prompt_len, device=DEVICE))
    show("expert position_ids", expert_pos)
    show("expert attention_mask (0 = visible)", expert_mask)
    print(f"      action tokens sit at positions {expert_pos[0, 0, 0].item()}.."
          f"{expert_pos[0, 0, -1].item()}, right after the anchor")
    generator = torch.Generator(device=DEVICE).manual_seed(args.seed)
    x = torch.randn(1, 64, 2, device=DEVICE, generator=generator)  # t = 0: pure noise
    grid = torch.linspace(0.0, 1.0, args.steps + 1, device=DEVICE)
    with cuda_timer("8 expert (all steps)", T):
        for i in range(args.steps):
            dt = (grid[i + 1] - grid[i]).view(1, 1, 1)
            t = grid[i].view(1, 1, 1)
            tokens = expert.action_in_proj(x, t).reshape(1, 64, -1)  # Fourier(x, t) -> MLP -> 512
            hidden = expert.expert(inputs_embeds=tokens, position_ids=expert_pos,
                                   past_key_values=cache, attention_mask=expert_mask,
                                   use_cache=True, is_causal=False).last_hidden_state
            cache.crop(prefix)  # drop the 64 action K/V again: every step sees only the CoT
            velocity = expert.action_out_proj(hidden).reshape(-1, 64, 2)
            x = x + dt * velocity
            if i in (0, args.steps - 1):
                show(f"step {i}: tokens / hidden / v", tokens)
    print(f"      per step: {T['8 expert (all steps)'] / args.steps * 1e3:.1f} ms")
    show("action (normalized accel, curvature)", x)

    # ---------------------------------------------------------------- 9. integrate
    print("\n[9] action_to_traj: unicycle integration from the t0 speed and course")
    with cuda_timer("9 action_to_traj", T):
        xyz, rot = expert.action_space.action_to_traj(
            x, traj["ego_history_xyz"][:, -1], traj["ego_history_rot"][:, -1])
    show("pred_xyz", xyz)
    show("pred_rot", rot)

banner("Verdict")
same_xyz = torch.equal(xyz, ref_xyz.reshape(xyz.shape))
same_rot = torch.equal(rot, ref_rot.reshape(rot.shape))
diff = (xyz - ref_xyz.reshape(xyz.shape)).abs().max().item()
print(f"  prefill identical       {same_prefill}")
print(f"  CoT tokens identical    {same_cot}")
print(f"  trajectory identical    {same_xyz} (max |dxyz| {diff:.3e} m)   rotations {same_rot}")
print(f"  6.4 s endpoint          x {xyz[0, -1, 0]:.2f} m forward, y {xyz[0, -1, 1]:.2f} m left")
model_time = sum(v for k, v in T.items() if k[0] in "123456789")
print(f"  unrolled GPU stages     {model_time:.2f} s   (library call {T['library sample_trajectories_from_data']:.2f} s)")
save_json("walkthrough.json", {
    "timings_s": T, "cot_tokens": n_cot, "prompt_length": prompt_len,
    "cache_length_for_expert": prefix, "identical": {"prefill": same_prefill, "cot": same_cot,
                                                     "xyz": same_xyz, "rot": same_rot},
    "max_abs_diff_m": diff, "endpoint_xy": xyz[0, -1, :2].tolist(),
})
