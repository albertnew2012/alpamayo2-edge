"""Session 0: is this machine ready? Versions, GPUs, FlashAttention, checkpoint, HF access.

    python study/scripts/00_env_check.py
"""

import importlib
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATASET_REVISION, MODEL_DIR, banner  # noqa: E402


def pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


banner("Python / packages")
print("python      ", sys.version.split()[0], sys.executable)
for name in ("torch", "torchvision", "transformers", "flash-attn", "physical-ai-av", "triton"):
    print(f"{name:12s}", pkg(name))

banner("CUDA")
print("cuda runtime", torch.version.cuda, "| available", torch.cuda.is_available())
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    free, total = torch.cuda.mem_get_info(index)
    print(
        f"  cuda:{index} {props.name}  sm_{props.major}{props.minor}  "
        f"bf16={torch.cuda.is_bf16_supported()}  free {free / 2**30:.1f} / {total / 2**30:.1f} GiB"
    )

banner("Acceleration kernels (docs/acceleration.md)")
from alpamayo2_edge.accel import resolve_kernels  # noqa: E402

for module in ("flash_attn", "flash_attn_interface"):
    try:
        importlib.import_module(module)
        print(f"  {module:22s} importable")
    except Exception as exc:  # noqa: BLE001
        print(f"  {module:22s} NOT importable ({type(exc).__name__})")
if torch.cuda.is_available():
    print("  resolve_kernels('auto') ->", resolve_kernels("auto", "cuda"))

banner("Checkpoint")
from alpamayo2_edge.checkpoint import validate_checkpoint  # noqa: E402

print("  EDGE_MODEL =", MODEL_DIR)
try:
    validate_checkpoint(Path(MODEL_DIR))
    size = sum(p.stat().st_size for p in Path(MODEL_DIR).glob("*.safetensors"))
    print(f"  complete: {size / 1e9:.2f} GB of safetensors")
except FileNotFoundError as exc:
    print("  INCOMPLETE:", exc)
    print("  -> run the launch config '0b. Download weights' first")

banner("Hugging Face gated access")
from huggingface_hub import HfApi  # noqa: E402

api = HfApi()
for repo, kind in (
    ("nvidia/Alpamayo2-Edge", "model"),
    ("nvidia/PhysicalAI-Autonomous-Vehicles", "dataset"),
):
    try:
        api.list_repo_refs(repo, repo_type=kind)
        print(f"  GRANTED  {repo}")
    except Exception as exc:  # noqa: BLE001
        print(f"  BLOCKED  {repo}  ({type(exc).__name__})")
print("  dataset revision pinned by the study scripts:", DATASET_REVISION)
