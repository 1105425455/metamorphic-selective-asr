"""Construction configuration for the released checkpoints.

The four causes, the student model directory and the severity ladders used when
building the corpus. Inference needs the cause list and the model directory; the
ladders are kept because the checkpoint metadata refers to them.

Set ``ASRABSTAIN_OUTPUTS`` to point ``config.OUTPUTS_DIR`` at the artefact tree.
"""

from __future__ import annotations

from asrabstain.config import STUDENT_MODEL_DIR

# The student model is Whisper-small; labels match the deployed model.
FINAL_STUDENT_MODEL_DIR = STUDENT_MODEL_DIR

# Four causes: three orthogonal axes plus the overlapping boundary case.
FINAL_REASONS: tuple[str, ...] = (
    "inaudible_noise",
    "bandlimited_muffled",
    "incomplete_speech",
    "overlapping_speech",
)

# Severity ladders, four levels each. The truncated ladder is raised at its
# lightest level so the four causes produce abstentions at comparable rates.
SNR_SEV: tuple[float, ...] = (5.0, 0.0, -5.0, -10.0)
BANDLIMIT_SEV: tuple[int, ...] = (0, 1, 2, 3)
INCOMPLETE_KEEP_SEV: tuple[float, ...] = (0.85, 0.65, 0.45, 0.25)
OVERLAP_RATIO_SEV: tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0)
