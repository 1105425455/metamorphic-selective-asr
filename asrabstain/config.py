"""Path constants used by the inference modules.

In the full training repository this module holds the dataset, noise-bank and
builder configuration as well. Inference only needs the four paths below, so
that is all that ships here. Every value can be overridden with
``ASRABSTAIN_OUTPUTS`` / ``ASRABSTAIN_MODEL_DIR``.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directory holding the evaluation artefacts (benchmark manifests, run outputs).
OUTPUTS_DIR = Path(os.environ.get("ASRABSTAIN_OUTPUTS", str(REPO_ROOT / "outputs_final")))

# Local Whisper-small weights. Defaults to the HuggingFace hub id when no local
# copy is configured.
STUDENT_MODEL_DIR = os.environ.get("ASRABSTAIN_MODEL_DIR", "openai/whisper-small")

BENCHMARK_DIR = OUTPUTS_DIR / "benchmark"
