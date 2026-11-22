# 06 — Local setup and every result on this machine

What was installed, what was run, what came out. Everything here was produced on
2026-11-10/11 from `main` @ `02f55a7` and checkpoint revision `f8951d09`.

## The machine

| | |
|---|---|
| GPUs | 2 × RTX 3090 24 GB (sm_86, BF16). **cuda:0 used**; cuda:1 drives the desktop |
| driver | 580.178.04 (CUDA 13.0 capable); PyTorch wheels bring CUDA 12.8 |
| OS | Ubuntu 24.04.5, glibc 2.39 |
| RAM | 78 GB |
| Python / uv | 3.12.3 / uv 0.12.5 |

## Install (what was actually run)

```bash
cd ~/Desktop/alpamayo2-edge
mkdir -p tmp && echo 'tmp/' >> .gitignore           # scratch files stay in the repo, untracked

# 1. environment: locked deps + the notebook extra (includes physical-ai-av, jupyter, matplotlib) + dev
UV_PROJECT_ENVIRONMENT=.venv uv sync --locked --extra notebook --dev

# 2. FlashAttention 2, prebuilt wheel pinned by hash (docs/acceleration.md)
echo "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl --hash=sha256:f25da18657a87fc83dc1bfb8b7751b82246e9db355510226b674fd437c34b5fb" > tmp/flash-attn.txt
uv pip install --python .venv --no-deps --require-hashes -r tmp/flash-attn.txt

# 3. weights, pinned revision, into .work/ (gitignored)
.venv/bin/python study/scripts/download_weights.py        # = README snapshot_download recipe
```

Resulting versions: torch **2.8.0+cu128**, torchvision 0.23.0, transformers **5.17.0**,
flash-attn **2.8.3**, physical-ai-av 0.2.0, triton 3.4.0. `.venv` is 8.2 GB.

HF account `albertnew2012` has access to both `nvidia/Alpamayo2-Edge` and
`nvidia/PhysicalAI-Autonomous-Vehicles` (checked with `list_repo_refs`, config **0a**).

Checkpoint `.work/Alpamayo2-Edge`: `00000.safetensors` 4.28 GB + `00001.safetensors`
1.03 GB + tokenizer/processor/config; `validate_checkpoint` passes; `load_model` reports no
missing, unexpected or mismatched keys.

**To keep FlashAttention** across later syncs, use the task "env: uv sync" (it adds
`--inexact`); a plain `uv sync` removes the wheel because it is not in `uv.lock`.

## Results

### Tests (config 0c)

```
ALPAMAYO2_EDGE_CHECKPOINT=.work/Alpamayo2-Edge python -m pytest
158 passed, 1 warning in 40.15s        (0 skipped: checkpoint and FlashAttention tests ran)
```

### README CLI demos (configs 1a–1c)

| run | result |
|---|---|
| **1a** default (1 CoT) | `pred_xyz [1,1,1,64,3]`, finite. CoT: *"Slow down and turn right at the intersection to follow our route, as the lead vehicle has cleared our path."* `elapsed_seconds` 3.14 (includes CPU preprocessing and first-call warm-up); 36 s wall incl. loading |
| **1b** six candidates | `[1,1,6,64,3]`; six CoTs, all "turn right", three mention the yield sign, one the lead vehicle turning left; 3.65 s |
| **1c** `--accel tuned` | **same CoT as 1a**; `elapsed_seconds` 1.85; 3 graphs built in 67 s (cold inductor cache), no fallbacks; 96 s wall |

Each run writes `outputs/<name>/<clip>_<t0>/{output.json, output.safetensors,
packed.safetensors, prompt.txt}` and `outputs/<name>/report.json` (versions, settings,
loading info, acceleration description).

### Notebook (config 2a)

`notebooks/inference.ipynb` executed headless on the local weights in **67 s**, all 18
cells, saved with outputs to [`outputs/inference_executed.ipynb`](outputs/inference_executed.ipynb)
plus four PNGs (`notebook_fig*.png`). 2.65 B parameters; prompt 3 261 tokens; six CoTs
(identical to config 3-1's, same seed); peak 9.2 GB; model call 1.45 s default vs 0.90 s
accelerated with identical CoT text; VQA prompt 4 636 tokens.

### Study scripts

| config | headline result |
|---|---|
| 3-2 anatomy | 2,651,163,378 parameters (= model card); VLM 92.5 %, expert 7.5 %; 907 tensors |
| 3-3 tokens | prompt 3 261 = 2 892 image + 57 history + 312 text; `rope_delta −2556`; history round trip 6.8 cm |
| **3-4 walkthrough** | **prefill, 23 CoT tokens and the trajectory identical to the library (`torch.equal`)**; GPU stages 0.87 s, CPU prep 1.37 s |
| 3-5a action space | course round trip FDE 0.24 cm vs yaw 1.84 cm; discrete tokens FDE 0.15 cm; bf16 constants shift 3.1 cm |
| 3-5b flow | path nearly straight; 1 Euler step within 0.096 m ADE of 10 |
| 3-1 demo | minADE@6 0.313 m, candidate 1 ADE 0.70 m; diversity: 6 CoTs 1.34 m vs 1 CoT × 6 noises 1.70 m |
| 3-6a route ablation | right → completes the turn (y −4.7 m); left → yields, slight left drift; straight/stop → slow yielding approach; the first CoT of each still says "turning left" |
| 3-6b VQA | 4 answers, 0.8–1.4 s each; without a route the model invents one ("turn left … to follow our route") |
| 3-7a eval, oracle route | 21 clips: ADE@6.4 s candidate 1 median **0.671 m** (mean 1.502), minADE@6 median **0.388 m**, ADE@3 s median 0.152 m; `<NOMINAL>` on 13/21 |
| 3-7b eval, wrong route | "Turn right in 30m" everywhere: median ADE@6.4 s **2.626 m**, minADE@6 0.988 m |
| 3-8 latency | eager 934 ms, FA2 eager 1 061 ms, `--accel tuned` **420 ms** (2.22×); 6 CoTs 1 455 → 898 ms |
| 4a–4c demo videos | 3 drive-alongs (180 ticks each, one 6-candidate accelerated inference per 0.1 s: 0.46 s with `<NOMINAL>`, ~0.85 s with reasoning), a route split-screen (91 ticks × 2 routes), a flow-matching animation → `study/outputs/videos/`, [FIGURES.md](FIGURES.md) |
| 3-7c vs Alpamayo 1.5 | same 21 clips: oracle-route medians Edge 0.671 / A1.5 1.019 / A1.5 + CFG 1.017 m (ADE@6.4 s cand. 1), paired p = 0.89 / 0.92; wrong route +1.434 m Edge vs −0.016 m A1.5 (−0.017 m with CFG) → [07](07_HEAD_TO_HEAD_VS_A15.md) |

### The 21-clip evaluation in detail

Clips: the Alpamayo 1.5 test clip `030c760c` first, then 20 drawn with seed 0 from the
306 151 valid clips with all six cameras and egomotion; t0 = 5.1 s; 6 candidates;
route = `derive_nav_text(logged future)` (19 × "Continue straight", 1 × "Turn left in 10m",
1 × "Turn left in 20m"). Per-clip rows, CoTs and timings are in `outputs/eval_oracle_k6.json`;
the picture is [`outputs/eval_oracle_k6.png`](outputs/eval_oracle_k6.png).

| metric (m) | mean | median |
|---|---|---|
| ADE@3 s, candidate 1 | 0.330 | 0.152 |
| minADE@3 s over 6 | 0.195 | 0.101 |
| ADE@6.4 s, candidate 1 | 1.502 | 0.671 |
| minADE@6.4 s over 6 | 0.830 | 0.388 |
| FDE@6.4 s, candidate 1 | 4.213 | 2.279 |
| minFDE@6.4 s over 6 | 2.173 | 1.088 |

Worst clips: `e92fce02` (roundabout at 2.9 m/s, ADE 8.9 m: the route said "Continue
straight" but the car went round the roundabout), `2bb07f05` (30 m/s, 4.2 m),
`cf1fabe6` (stop sign, 2.8 m), `d8db2176` (curve, 2.6 m but minADE 0.39 m). Highway clips
at 24–30 m/s are among the best (`eeed08e5` ADE 0.25 m, minADE 0.04 m).

**Caveats.** 21 clips is a sanity check. The route is derived from the future. The
`derive_nav_text` heuristic is crude (roundabouts become "Continue straight"). Numbers are
not comparable to anything evaluated with a different route policy, horizon or candidate
count.

## What was added to the repository

Nothing under `src/`, `tests/`, `notebooks/`, `docs/` or `examples/` was modified, and
nothing was committed.

| path | what |
|---|---|
| `.gitignore` | `+ tmp/` |
| `.vscode/launch.json` | 25 debug configs (all `justMyCode: true`, `stopOnEntry: true`) |
| `.vscode/tasks.json`, `settings.json` | shell tasks (uv sync, FA2 wheel, hf login, jupyter lab), interpreter and env |
| `study/*.md` | this course |
| `study/scripts/` | 15 files: `_common.py` helpers, 00–10, download, notebook runner, diagram |
| `study/outputs/` | figures and JSON from the runs above, the executed notebook |
| `study/diagrams/` | the architecture SVG |
| `study/outputs/videos/` | 5 demo MP4s (28 MB total) + poster stills, made by `study/scripts/12_demo_videos.py` |

In `~/Desktop/alpamayo1.5` (for the head-to-head, nothing under its `src/` changed, nothing
committed): `study/scripts/21_eval_vs_edge.py`, launch configs **29** and **29b** in its
`.vscode/launch.json`, `tmp/` added to its `.gitignore`; results in its (gitignored)
`study/outputs/eval_vs_edge*.json|npz`; its 21-clip sample cache in its `tmp/`.

Untracked, gitignored: `.venv` (8.2 GB), `.work` (5.0 GB checkpoint), `outputs/` (CLI
results, 103 MB), `tmp/` (3.1 GB: sample cache of 21 clips at 137 MB each, inductor/Triton
caches, logs). Deleting `tmp/` is safe; the scripts rebuild it.
