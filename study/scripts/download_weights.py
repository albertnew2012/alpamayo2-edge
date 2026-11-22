"""Download the pinned Alpamayo 2 Edge checkpoint into .work/Alpamayo2-Edge (README recipe).

    python study/scripts/download_weights.py

5.32 GB: two safetensors shards plus tokenizer/processor/config. Re-running is a no-op
once the files are present.
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_DIR  # noqa: E402

REVISION = "f8951d095de3589b3face1dec3f4394c8ecedb38"

path = snapshot_download(
    repo_id="nvidia/Alpamayo2-Edge", revision=REVISION, local_dir=str(MODEL_DIR)
)
print("checkpoint at", path)

from alpamayo2_edge.checkpoint import validate_checkpoint  # noqa: E402

validate_checkpoint(Path(path))
print("validate_checkpoint: OK")
