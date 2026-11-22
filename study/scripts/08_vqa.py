"""Session 6b: the same weights as a plain VLM - freeform VQA about the scene.

    python study/scripts/08_vqa.py [--question "..."] [--max-new-tokens 512]

Uses ``load_physical_aiavdataset_vqa`` (6 cameras x 4 frames at -0.3..0 s, all at the
front-camera pixel budget) and ``text_tasks.generate_vqa_answer`` (greedy, stops at
<|im_end|> or <|traj_future_start|>, trajectory bins masked). No ego history, no
navigation, no expert. Writes study/outputs/vqa.json.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import banner, example, load_edge, load_vqa_sample, save_json  # noqa: E402

from alpamayo2_edge.helper import prepare_vqa_inputs, to_device  # noqa: E402
from alpamayo2_edge.text_tasks import generate_vqa_answer  # noqa: E402

DEFAULT_QUESTIONS = [
    "How should the ego vehicle drive in this scene? "
    "Describe the relevant road users, traffic signals, and hazards.",
    "Which traffic signs are visible, and what do they require of the ego vehicle?",
    "Is there a vehicle directly ahead of the ego vehicle? What is it doing?",
    "What can be seen in the rear cameras?",
]
parser = argparse.ArgumentParser()
parser.add_argument("--question", action="append", help="repeatable; default: four questions")
parser.add_argument("--max-new-tokens", type=int, default=512)
args = parser.parse_args()
questions = args.question or DEFAULT_QUESTIONS

spec = example()
data = load_vqa_sample(spec["clip_id"], spec["t0_us"])
print("VQA frames:", tuple(data["image_frames"].shape), "cameras",
      sorted(set(data["camera_indices"].tolist())))
model, _ = load_edge()
answers = []
for question in questions:
    banner(question)
    inputs = to_device(prepare_vqa_inputs(data, model.config, model.tokenizer, question),
                       device="cuda")
    length = inputs["tokenized_data"]["input_ids"].shape[1]
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        answer = generate_vqa_answer(model, inputs, max_new_tokens=args.max_new_tokens)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    n_tokens = len(model.tokenizer(answer)["input_ids"])
    print(answer)
    print(f"\n  prompt {length} tokens, answer ~{n_tokens} tokens, {seconds:.2f} s")
    answers.append({"question": question, "answer": answer, "prompt_tokens": length,
                    "seconds": seconds})
save_json("vqa.json", {"sample": {k: spec[k] for k in ("clip_id", "t0_us")}, "answers": answers})
