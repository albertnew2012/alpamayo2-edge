"""Execute notebooks/inference.ipynb headless against the local checkpoint.

    python study/scripts/run_notebook.py [--skip-accel]

The upstream notebook is not modified: a copy with ``MODEL_ID`` pointed at
``$EDGE_MODEL`` (instead of downloading nvidia/Alpamayo2-Edge into the HF cache) is
executed with nbclient from ``notebooks/`` and saved, outputs included, to
study/outputs/inference_executed.ipynb. Figures are also exported as PNGs.
"""

import argparse
import base64
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_DIR, OUT, REPO  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--skip-accel", action="store_true", help="drop the accelerated cell")
parser.add_argument("--timeout", type=int, default=1800, help="per-cell timeout (s)")
args = parser.parse_args()

source = REPO / "notebooks" / "inference.ipynb"
nb = nbformat.read(source, as_version=4)
patched = False
for cell in nb.cells:
    if cell.cell_type == "code" and 'MODEL_ID = "nvidia/Alpamayo2-Edge"' in cell.source:
        cell.source = cell.source.replace('MODEL_ID = "nvidia/Alpamayo2-Edge"',
                                          f'MODEL_ID = "{MODEL_DIR}"')
        patched = True
if not patched:
    raise RuntimeError("MODEL_ID line not found; the notebook changed upstream")
if args.skip_accel:
    nb.cells = [c for c in nb.cells if "enable_acceleration(model" not in c.source]

print(f"executing {len(nb.cells)} cells of {source.relative_to(REPO)} (MODEL_ID -> {MODEL_DIR})")
start = time.perf_counter()
client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3",
                        resources={"metadata": {"path": str(source.parent)}})
client.execute()
print(f"done in {time.perf_counter() - start:.0f} s")

OUT.mkdir(parents=True, exist_ok=True)
target = OUT / "inference_executed.ipynb"
nbformat.write(nb, target)
print("wrote", target.relative_to(REPO))
figure = 0
for index, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            print(f"--- cell {index} ---\n{output['text'].strip()[:3000]}")
        data = output.get("data", {})
        if "image/png" in data:
            figure += 1
            path = OUT / f"notebook_fig{figure}_cell{index}.png"
            path.write_bytes(base64.b64decode(data["image/png"]))
            print("wrote", path.relative_to(REPO))
