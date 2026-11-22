"""Render study/diagrams/alpamayo2_edge_architecture.svg from the measured anatomy.

    python study/scripts/01_anatomy.py          # writes study/outputs/anatomy.json
    python study/scripts/make_architecture_svg.py

Parameter counts come from anatomy.json; token counts and timings are the values
measured on the README example (study/06_LOCAL_SETUP_AND_RESULTS.md).
"""

import json
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import OUT, REPO  # noqa: E402

anatomy = json.loads((OUT / "anatomy.json").read_text())
TREE = [(row["name"].split("  ")[0].strip(), row["params"]) for row in anatomy["tree"]]
P = {}
for name, n in TREE:  # first occurrence wins, so "layers x 28" is the VLM's
    P.setdefault(name, n)
P["expert"] = next(n for name, n in TREE if name == "expert")
P["expert layer"] = [n for name, n in TREE if name == "layers x 28"][1] / 28
TOTAL = anatomy["total_params"]
W, H = 1400, 1640
C = {  # light palette; the SVG carries its own white background
    "ink": "#1d1d1b", "muted": "#5c5b57", "line": "#8a8984", "bg": "#ffffff",
    "input": "#eef4fb", "input_s": "#2a78d6", "prep": "#f5f3ee", "prep_s": "#8a8984",
    "vision": "#fdf0e8", "vision_s": "#eb6834", "vlm": "#e9f6f0", "vlm_s": "#1baf7a",
    "expert": "#fbf3df", "expert_s": "#c98a00", "out": "#f3ecf7", "out_s": "#8a4fbf",
    "cache": "#e8eef8", "cache_s": "#4466aa",
}
parts = []


def rect(x, y, w, h, fill, stroke, rx=10, dash=None, sw=1.6):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=13, weight="normal", color=None, anchor="start", italic=False):
    style = ' font-style="italic"' if italic else ""
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
                 f'fill="{color or C["ink"]}" text-anchor="{anchor}"{style}>{escape(s)}</text>')


def lines(x, y, rows, size=12.5, gap=17, color=None, weight="normal"):
    for i, row in enumerate(rows):
        text(x, y + i * gap, row, size=size, color=color, weight=weight)


def arrow(x1, y1, x2, y2, color=None, dash=None, sw=1.8):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color or C["line"]}" '
                 f'stroke-width="{sw}" marker-end="url(#arrow)"{d}/>')


def box(x, y, w, h, kind, title, rows, size=12.5, title_size=14.5):
    rect(x, y, w, h, C[kind], C[kind + "_s"])
    text(x + 14, y + 24, title, size=title_size, weight="bold", color=C[kind + "_s"])
    lines(x + 14, y + 45, rows, size=size)


M = 1e6
parts.append(f'<rect width="{W}" height="{H}" fill="{C["bg"]}"/>')
parts.append('<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
             'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
             f'fill="{C["line"]}"/></marker></defs>')
text(W / 2, 38, "Alpamayo 2 Edge — one inference call (measured on the README example, RTX 3090)",
     size=21, weight="bold", anchor="middle")
text(W / 2, 62, f"{TOTAL:,} parameters · BF16 · Cosmos3-Edge VLM (2.45 B) + flow-matching "
     "action expert (0.20 B)", size=14, color=C["muted"], anchor="middle")

# ---- inputs
y = 85
box(40, y, 470, 108, "input", "6 cameras · 22 frames (uint8 1080×1920)", [
    "front wide (1), front tele (6): 5 frames at −2, −1, −0.5, −0.1, 0 s",
    "cross L/R (0, 2), rear L/R (3, 5): 3 frames at −1, −0.1, 0 s",
    "actual capture times kept (sensor jitter is part of the input)",
])
box(530, y, 360, 108, "input", "Ego history", [
    "20 poses, −1.9 … 0 s at 10 Hz",
    "xyz [20, 3] + rot [20, 3, 3], ego frame at t0",
    "(x forward, y left, z up)",
])
box(910, y, 450, 108, "input", "Navigation instruction", [
    '"Turn right in 30m"   — plain prompt text',
    "no route tokens, no classifier-free guidance",
    "required: the dataset has no route, the caller supplies it",
])
for x in (275, 710, 1135):
    arrow(x, y + 108, x, y + 140)

# ---- preprocessing
y = 228
box(40, y, 470, 118, "prep", "prepare_model_inputs (CPU, ≈1.4 s)", [
    "sort by camera, then capture time; label each <t=−1.0s>",
    "bicubic resize in uint8 to the camera's pixel budget:",
    "  front/tele 320×576 (163–197 k px) · others 224×416 (82–98 k px)",
    "normalize (x/255 − 0.5)/0.5 · patchify 16×16 → 11,568 × 768",
], size=12)
box(530, y, 360, 118, "prep", "DeltaTrajectoryTokenizer", [
    "19 steps × (dx, dy, dz) → 57 bins",
    "1000 bins: xy ±4 m (0.8 cm), z ±10 m",
    "ids <i0>…<i999> = 131072…132071",
    "masked_scatter into 57 <|traj_history|> slots",
], size=12)
box(910, y, 450, 118, "prep", "Chat template (enable_thinking=False)", [
    "system: “You are a driving assistant …”",
    "user: camera name, time label, image × 22, history,",
    "      “Navigation input is: …”, task sentence",
    "assistant: <think></think>   ← reasoning starts here",
], size=12)
arrow(275, y + 118, 275, y + 150)
arrow(870, y + 118, 870, y + 323)
arrow(1135, y + 118, 1135, y + 323)

# ---- vision
y = 380
box(40, y, 470, 122, "vision", f"SigLIP2 vision tower · {P['model.visual'] / M:.2f} M", [
    "patch_embedding Linear 768 → 1152 + learned 16×16 position",
    "  grid, bilinearly resized to every image's own patch grid",
    "27 blocks × 1152 · 16 heads · MLP 4304 (gelu-tanh) · LayerNorm",
    "attention only within one image (cu_seqlens) · temporal patch 1",
], size=12)
box(530, y, 320, 122, "vision", f"Projector · {P['model.projector'] / M:.2f} M", [
    "LayerNorm(1152) per patch",
    "group 2×2 neighbours → 4608",
    "Linear 4608 → 11520 → GELU → 2048",
    "11,568 patches → 2,892 LLM tokens",
], size=12)
arrow(510, y + 60, 530, y + 60)
arrow(690, y + 122, 690, y + 167)

# ---- token sequence
y = 535
text(40, y + 6, "Prompt token sequence: 3,261 tokens", size=14.5, weight="bold")
segments = [("sys", 34, "#d9d8d2"), ("6 cameras × frames: 2,892 image tokens (180 per front/tele, 91 per side/rear) + labels", 3109 - 34, "#f6c9b1"),
            ("57 hist", 57, "#bcd6f3"), ("nav + task", 95, "#d9d8d2")]
x0, total_w, tokens = 40, 1320, 3261
x = x0
for label, n, color in segments:
    w = total_w * n / tokens
    rect(x, y + 16, w, 30, color, "#ffffff", rx=3, sw=1)
    if w > 60:
        text(x + w / 2, y + 36, label, size=11.5, anchor="middle")
    x += w
text(x0 + total_w - 2, y + 64, "…<|im_start|>assistant\\n<think></think>", size=11.5, anchor="end",
     color=C["muted"])
text(40, y + 64, "3-D M-RoPE (t, h, w): text +1 per token, an h×w image advances max(h, w) → "
     "rope_delta = −2556; last prompt position 704", size=11.5, color=C["muted"])
arrow(350, y + 72, 350, y + 100)

# ---- VLM decoder and expert
y = 640
vlm_h = 330
rect(40, y, 620, vlm_h, C["vlm"], C["vlm_s"])
text(54, y + 25, f"Cosmos3-Edge text decoder · {P['model.language_model'] / M:.0f} M + lm_head "
     f"{P['lm_head'] / M:.0f} M", size=14.5, weight="bold", color=C["vlm_s"])
lines(54, y + 47, [
    f"embed_tokens 135,168 × 2048 ({P['embed_tokens'] / M:.1f} M) · lm_head untied (same size)",
    "vocab: 131,072 base (Tekken-style) + 4,000 trajectory bins + 30 driving specials",
], size=12)
for i in range(6):  # a stack of layer cards
    rect(70 + i * 5, y + 92 + i * 5, 470, 118, "#ffffff", C["vlm_s"], rx=6, sw=1)
lx, ly = 100, y + 142
text(lx, ly, f"× 28 decoder layers · {P['layers x 28'] / 28 / M:.2f} M each", size=13, weight="bold")
lines(lx, ly + 21, [
    "RMSNorm → GQA self-attention",
    "   q 2048→2048 (16 heads × 128) · k, v 2048→1024 (8 × 128)",
    "   interleaved M-RoPE, sections [24, 20, 20], θ = 1e8 · causal",
    "RMSNorm → MLP 2048 → 9216 → relu² → 2048 (no gate, no bias)",
], size=12)
lines(54, y + 258, [
    "Prefill: prompt[:−1] once (138 ms); every CoT reuses it (shared prefill)",
    "Decode: top-p 0.98, T 0.6; 4000 trajectory bins and <|im_end|> → −∞;",
    "stop one step after <|traj_future_start|> so the anchor is cached",
], size=12)
text(54, y + 314, "17 ms / token eager · ~22 CoT tokens on the example", size=12, color=C["muted"],
     italic=True)

# KV cache column
cx, cw = 672, 80
rect(cx, y, cw, vlm_h, C["cache"], C["cache_s"], rx=8)
for i, row in enumerate(["KV", "cache", "", "28 ×", "(K, V)", "", "[8, L,", "128]", "",
                         "L =", "3,284", "", "112 KiB", "/ token"]):
    text(cx + cw / 2, y + 30 + i * 19, row, size=12, anchor="middle", color=C["cache_s"],
         weight="bold" if i < 2 else "normal")

ex = 765
rect(ex, y, 595, vlm_h, C["expert"], C["expert_s"])
text(ex + 14, y + 25, f"Action expert · {P['expert'] / M:.2f} M", size=14.5,
     weight="bold", color=C["expert_s"])
lines(ex + 14, y + 47, ["Cosmos3EdgeTextModel without embed_tokens, width 512",
                        "non-causal over [cached prompt + CoT ; its own 64 action tokens]"],
      size=12)
for i in range(6):
    rect(ex + 25 + i * 5, y + 92 + i * 5, 470, 118, "#ffffff", C["expert_s"], rx=6, sw=1)
lx = ex + 55
text(lx, ly, f"× 28 expert layers · {P['expert layer'] / M:.2f} M each", size=13,
     weight="bold")
lines(lx, ly + 21, [
    "RMSNorm → attention, q 512→2048 (16 × 128)",
    "   k, v 512→1024 (8 × 128) = the VLM's KV geometry",
    "   layer i attends VLM layer i's cache + its own K/V",
    "RMSNorm → MLP 512 → 3840 → relu² → 512",
], size=12)
lines(ex + 14, y + 258, [
    "64 action tokens at positions offset … offset+63 (right after the anchor)",
    "4-D float mask hides padding / other rows' CoT tails",
    "cache.crop(prefix) after every step: each step sees only the CoT",
], size=12)
text(ex + 14, y + 314, "18.8 ms / Euler step eager · 10 steps", size=12, color=C["muted"],
     italic=True)
for k in range(4):  # layer i -> expert layer i
    yy = y + 150 + k * 20
    arrow(cx + cw, yy, ex + 25, yy, color=C["cache_s"], sw=1.4)
    arrow(560, yy, cx, yy, color=C["vlm_s"], sw=1.4)

# ---- below: CoT text and expert I/O
y = 1000
box(40, y, 620, 96, "out", "Reasoning text (extra[\"cot\"])", [
    '"Slow down and turn right at the intersection to follow our route,',
    '  as the lead vehicle has cleared our path."',
    "generated_ids keep the sampled tokens; the trajectory never comes from text",
], size=12)
arrow(350, 970, 350, 1000)
box(ex, y, 595, 96, "expert", f"action_in_proj ({P['action_in_proj'] / M:.2f} M) / action_out_proj", [
    "x_t [64, 2], t → Fourier features (10 log-freqs, sin+cos) per channel + t = 60",
    "→ MLP 60→512→512→512 (SiLU, RMSNorm) → LayerNorm → 64 tokens × 512",
    "out: Linear 512 → 2 = velocity v(x_t, t) for every waypoint",
], size=12)
arrow(1062, 970, 1062, 1000)
arrow(1062, 1096, 1062, 1126)

y = 1126
box(ex, y, 595, 110, "expert", "Flow matching (rectified flow, Euler)", [
    "x₀ ~ N(0, I) [64, 2] (own torch.Generator)",
    "for t in 0, 0.1, …, 0.9:  x ← x + 0.1 · v(x, t)",
    "x₁ = normalized (accel, curvature) per 0.1 s step; no CFG",
    "path is nearly straight: 1 step lands within 0.10 m ADE of 10",
], size=12)
arrow(1062, 1236, 1062, 1266)

y = 1266
box(ex, y, 595, 128, "expert", "UnicycleAccelCurvatureCourseActionSpace", [
    "de-normalize: a = 0.68·x[…,0] + 0.03 m/s², κ = 0.026·x[…,1] 1/m",
    "v0 and course θ0 estimated from the history (sideslip-aware)",
    "v ← v + a·dt;  θ ← θ + κ·v·dt;  x, y by trapezoid along the course",
    "yaw = course − θ0 + 2.5e-3 · Δa_lat  (sideslip model) → rot",
    "drivable by construction: |a| ≤ 9.8 m/s², |κ| ≤ 0.33 1/m",
], size=12)
arrow(1062, 1394, 1062, 1420)

y = 1420
box(40, y, 1320, 70, "out", "Outputs", [
    "pred_xyz [B, sets, samples, 64, 3] and pred_rot [B, sets, samples, 64, 3, 3] — 64 poses at 0.1 s = 6.4 s, ego frame at t0 · "
    "cot text per candidate"], size=12.5)
parts.append(f'<path d="M 350 1096 L 350 1440" stroke="{C["line"]}" stroke-width="1.8" '
             'fill="none" marker-end="url(#arrow)" stroke-dasharray="5 4"/>')

# ---- parameter bar
y = 1525
text(40, y, "Where the 2.65 B parameters are", size=14.5, weight="bold")
bars = [("vision 412.7 M", P["model.visual"], C["vision_s"]),
        ("proj", P["model.projector"], "#f2a27a"),
        ("embed 276.8 M", P["embed_tokens"], "#7fd3b0"),
        ("28 decoder layers 1,409.4 M", P["layers x 28"], C["vlm_s"]),
        ("lm_head 276.8 M", P["lm_head"], "#7fd3b0"),
        ("expert 198.8 M", P["expert"], C["expert_s"])]
x = 40
for label, n, color in bars:
    w = 1320 * n / TOTAL
    rect(x, y + 14, w, 34, color, "#ffffff", rx=3, sw=1)
    if w > 70:
        text(x + w / 2, y + 36, label, size=11.5, anchor="middle", color="#ffffff", weight="bold")
    x += w
text(40, y + 72, "VLM 92.5 % (vision 15.6 · projector 2.9 · embeddings 10.4 · layers 53.2 · "
     "lm_head 10.4) · expert 7.5 % · the action-specific input/output projections are 0.56 M",
     size=12, color=C["muted"])

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       'font-family="Inter, Helvetica, Arial, sans-serif">\n' + "\n".join(parts) + "\n</svg>\n")
target = REPO / "study" / "diagrams" / "alpamayo2_edge_architecture.svg"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(svg)
print("wrote", target.relative_to(REPO))
