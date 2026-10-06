"""Build and execute the documentation notebooks.

    python notebooks/_build/build.py            # all notebooks
    python notebooks/_build/build.py surface    # one of: surface, forecast

Each ``nb_*.py`` module holds a notebook's cells as plain strings, which keeps
them reviewable in diffs; this script turns them into ``notebooks/<TITLE>.ipynb``
and executes them in place so the saved notebook carries its outputs.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = {"surface": "nb_surface", "forecast": "nb_forecast"}


def build(module_name: str, timeout: int = 3600) -> Path:
    sys.path.insert(0, str(ROOT))
    mod = importlib.import_module(f"notebooks._build.{module_name}")
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "vss", "display_name": "stochastic-vol-surface (.venv)",
                                 "language": "python"}
    for kind, src in mod.CELLS:
        src = textwrap.dedent(src).strip("\n")
        nb.cells.append(nbformat.v4.new_markdown_cell(src) if kind == "md"
                        else nbformat.v4.new_code_cell(src))
    out = ROOT / "notebooks" / f"{mod.TITLE}.ipynb"
    t0 = time.time()
    NotebookClient(nb, timeout=timeout, kernel_name="vss",
                   resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
    nbformat.write(nb, out)
    print(f"{out.relative_to(ROOT)} executed in {time.time() - t0:.0f}s")
    return out


if __name__ == "__main__":
    for key in sys.argv[1:] or list(NOTEBOOKS):
        build(NOTEBOOKS[key])
