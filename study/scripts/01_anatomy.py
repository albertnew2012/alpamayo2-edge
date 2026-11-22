"""Session 2: where do the 2.65 B parameters live? Dumped from the loaded checkpoint.

    python study/scripts/01_anatomy.py [--device cpu]

Prints the module tree with parameter counts, one layer of every stack with its
tensor shapes, the tokenizer layout (base vocab, 4000 trajectory bins, driving
specials), the KV-geometry contract between VLM and expert, and the constants that
are rebuilt from config instead of stored. Writes study/outputs/anatomy.json.
"""

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_DIR, banner, load_edge, save_json  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

model, loading_info = load_edge(device=args.device)
total = sum(p.numel() for p in model.parameters())
print(f"total parameters: {total:,}  ({total / 1e9:.4f} B)")


def count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def line(name: str, n: int, note: str = "", depth: int = 0) -> dict:
    pct = 100.0 * n / total
    print(f"{'  ' * depth}{name:<{58 - 2 * depth}} {n / 1e6:10.3f} M  {pct:6.2f} %  {note}")
    return {"name": name, "params": n, "percent": round(pct, 3), "note": note}


banner("1. Module tree")
vlm, expert = model.vlm, model.expert
visual, projector, text = vlm.model.visual, vlm.model.projector, vlm.model.language_model
tree = []
tree.append(line("vlm  (Cosmos3EdgeForConditionalGeneration)", count(vlm)))
tree.append(line("model.visual  (SigLIP2 ViT)", count(visual), depth=1))
tree.append(line("embeddings.patch_embedding", count(visual.embeddings.patch_embedding),
                 "Linear 768 -> 1152 (16x16x3 patch)", depth=2))
tree.append(line("embeddings.position_embedding", count(visual.embeddings.position_embedding),
                 "16x16 learned grid, resized per image", depth=2))
tree.append(line(f"encoder.layers x {len(visual.encoder.layers)}", count(visual.encoder),
                 f"{count(visual.encoder.layers[0]) / 1e6:.3f} M each", depth=2))
tree.append(line("post_layernorm", count(visual.post_layernorm), depth=2))
tree.append(line("model.projector  (2x2 patch merger + MLP)", count(projector), depth=1))
tree.append(line("model.language_model  (decoder)", count(text), depth=1))
tree.append(line("embed_tokens", count(text.embed_tokens),
                 f"{tuple(text.embed_tokens.weight.shape)}", depth=2))
tree.append(line(f"layers x {len(text.layers)}", sum(count(layer) for layer in text.layers),
                 f"{count(text.layers[0]) / 1e6:.3f} M each", depth=2))
tree.append(line("norm", count(text.norm), depth=2))
tree.append(line("lm_head  (untied)", count(vlm.lm_head), f"{tuple(vlm.lm_head.weight.shape)}",
                 depth=1))
tree.append(line("expert  (ExpertModel)", count(expert)))
tree.append(line("expert  (Cosmos3EdgeTextModel, no embed_tokens)", count(expert.expert), depth=1))
tree.append(line(f"layers x {len(expert.expert.layers)}",
                 sum(count(layer) for layer in expert.expert.layers),
                 f"{count(expert.expert.layers[0]) / 1e6:.3f} M each", depth=2))
tree.append(line("norm", count(expert.expert.norm), depth=2))
tree.append(line("action_in_proj  (PerWaypointActionInProjV2)", count(expert.action_in_proj),
                 depth=1))
tree.append(line("action_out_proj  (Linear 512 -> 2)", count(expert.action_out_proj), depth=1))
tree.append(line("action_space / diffusion", count(expert.action_space) + count(expert.diffusion),
                 "no learned parameters", depth=1))


def shapes(module: torch.nn.Module) -> "OrderedDict[str, list[int]]":
    return OrderedDict((name, list(p.shape)) for name, p in module.named_parameters())


banner("2. One layer of each stack")
per_layer = {}
for label, layer in (
    ("vision encoder layer 0", visual.encoder.layers[0]),
    ("projector", projector),
    ("VLM decoder layer 0", text.layers[0]),
    ("expert decoder layer 0", expert.expert.layers[0]),
    ("action_in_proj", expert.action_in_proj),
    ("action_out_proj", expert.action_out_proj),
):
    print(f"\n--- {label}: {count(layer) / 1e6:.3f} M")
    per_layer[label] = shapes(layer)
    for name, shape in per_layer[label].items():
        print(f"    {name:44s} {shape}")
    if label == "action_in_proj":
        print("    ", layer)

banner("3. Configuration that shapes the computation")
vlm_text = model.config.vlm_config.text_config
vision = model.config.vlm_config.vision_config
expert_text = model.config.expert_config.llm_config
geometry = {}
for label, cfg in (("vlm text", vlm_text), ("expert", expert_text)):
    geometry[label] = {
        key: getattr(cfg, key)
        for key in (
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "intermediate_size",
            "hidden_act",
            "vocab_size",
        )
    }
    geometry[label]["rope_parameters"] = cfg.rope_parameters
    print(label, json.dumps(geometry[label]))
geometry["vision"] = {
    key: getattr(vision, key)
    for key in ("hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size",
                "patch_size", "spatial_merge_size", "num_patches", "hidden_act")
}
print("vision", json.dumps(geometry["vision"]))
print("projector_hidden_size", model.config.vlm_config.projector_hidden_size)
q_dim = expert_text.num_attention_heads * expert_text.head_dim
kv_dim = expert_text.num_key_value_heads * expert_text.head_dim
print(
    f"\nKV contract: expert width {expert_text.hidden_size} projects to q {q_dim} and k/v {kv_dim}, "
    f"exactly the VLM's {vlm_text.num_key_value_heads} KV heads x {vlm_text.head_dim}. "
    f"Expert layer i attends VLM layer i's cache (both {vlm_text.num_hidden_layers} layers)."
)
kv_bytes_per_token = 2 * vlm_text.num_hidden_layers * kv_dim * 2
print(f"VLM KV cache: {kv_bytes_per_token / 1024:.0f} KiB per token in bf16 (K+V, all layers)")

banner("4. Tokenizer and trajectory vocabulary")
tok = model.tokenizer
ids = model.config.traj_ids
vocab = tok.get_vocab()
tokenizer_info = {
    "len(tokenizer)": len(tok),
    "embedding rows": text.embed_tokens.weight.shape[0],
    "eos": (tok.eos_token, tok.eos_token_id),
    "pad": (tok.pad_token, tok.pad_token_id),
    "history bins": [ids["history_id0"], ids["history_id0"] + model.config.history_vocab_size - 1],
    "future bins": [ids["future_id0"], ids["future_id0"] + model.config.future_vocab_size - 1],
    "traj_ids": ids,
}
for name in ("<i0>", "<i999>", "<i1000>", "<i3999>", "<|traj_history_start|>", "<|traj_history|>",
             "<|traj_history_end|>", "<|traj_future_start|>", "<|image_pad|>", "<think>",
             "</think>", "<|im_start|>", "<|im_end|>"):
    tokenizer_info[name] = vocab.get(name)
for key, value in tokenizer_info.items():
    print(f"  {key:26s} {value}")

banner("5. Constants rebuilt from config (non-persistent buffers)")
constants = {}
for module_name, module in (("action_space", expert.action_space),
                            ("action_in_proj", expert.action_in_proj)):
    for name, buffer in module.named_buffers():
        constants[f"{module_name}.{name}"] = (
            buffer.flatten()[:4].float().tolist() if buffer.numel() > 1 else buffer.item()
        )
for key, value in constants.items():
    print(f"  {key:44s} {value}")
print("  action space:", type(expert.action_space).__name__,
      "dims", expert.action_space.get_action_space_dims(),
      "dt", expert.action_space.dt, "slip_gradient", getattr(expert.action_space, "slip_gradient", None))
print("  diffusion:", type(expert.diffusion).__name__, "x_dims", expert.diffusion.x_dims,
      "default steps", expert.diffusion.num_inference_steps,
      "CFG", expert.diffusion.use_classifier_free_guidance)

banner("6. Checkpoint on disk")
index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]
by_prefix = {}
for key in index:
    prefix = ".".join(key.split(".")[:3])
    by_prefix[prefix] = by_prefix.get(prefix, 0) + 1
print(f"  {len(index)} tensors in {len(set(index.values()))} shards")
for prefix, n in sorted(by_prefix.items()):
    print(f"    {prefix:40s} {n:4d} tensors")
problems = {k: v for k, v in loading_info.items() if v}
print("  load_model loading_info non-empty fields:", problems or "none")

save_json(
    "anatomy.json",
    {
        "total_params": total,
        "tree": tree,
        "per_layer_shapes": per_layer,
        "geometry": geometry,
        "projector_hidden_size": model.config.vlm_config.projector_hidden_size,
        "tokenizer": tokenizer_info,
        "constants": constants,
        "checkpoint_tensors": len(index),
    },
)
