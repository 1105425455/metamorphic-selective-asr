"""Encoder-MTL configuration.

A binary action head and a four-way cause head on a Whisper acoustic encoder.
Unlike structured SFT, which is seq2seq and emits a decision token plus a cause
token and a transcript, this model is discriminative: the pooled encoder output
feeds two linear heads. It produces no transcript.

Every hyperparameter is copied from ``sft_config_final.py`` -- epochs=8, batch=32,
warmup=1000, seed=42, bf16, ``freeze_encoder=True``, class balancing on -- except
the learning rate, which is documented where it is defined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from asrabstain.config import OUTPUTS_DIR
from asrabstain.final.config_final import FINAL_STUDENT_MODEL_DIR
from asrabstain.final.train.sft_config_final import DataConfig

RUNS_DIR = OUTPUTS_DIR / "runs"

# Four causes. The order is a contract: the cause head weights in the checkpoint
# are stored this way, so reordering them would silently mislabel every
# prediction. This is not the same order as FINAL_REASONS.
CAUSE_CLASSES: tuple[str, ...] = (
    "inaudible_noise",
    "bandlimited_muffled",
    "overlapping_speech",
    "incomplete_speech",
)
CAUSE_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CAUSE_CLASSES)}


@dataclass
class ModelConfig:
    """Structure of the discriminative head.

    ``pooling``:
      - ``"attention"`` (default): learnable single-query additive attention.
        The evidence for each cause is unevenly distributed in time: truncation
        shows up at the end of the clip, overlap in a short span, while only
        bandlimiting is a global spectral property. Mean pooling dilutes a local
        anomaly across 1500 frames, which structurally hurts the truncation and
        overlap axes.
      - ``"mean"``: average over time, kept as an ablation.
    """

    pooling: str = "attention"
    # Head width. Whisper-small has d_model=768; project to 256 before the logits,
    # A non-linear recombination of the frozen features; a purely linear probe
    hidden_dim: int = 256
    dropout: float = 0.1


@dataclass
class LossConfig:
    """Weighted sum of the binary and four-way cross-entropies.

    ``cause_weight``: the cause head is supervised on ABSTAIN samples only, since
    ANSWER samples carry no cause. That is roughly half the data and four classes
    are harder than two, so the weight stays at 1.0.

    ``balance_*``: identical to the structured SFT loss (answer_share=0.5, the
    causes split evenly within ABSTAIN), implemented with a
    WeightedRandomSampler.
    """

    action_weight: float = 1.0
    cause_weight: float = 1.0
    label_smoothing: float = 0.0
    balance_decision_classes: bool = True
    balance_reason_classes: bool = True
    answer_share: float = 0.5


@dataclass
class OptimConfig:
    freeze_encoder: bool = True
    # lr differs from structured SFT on purpose. 1.5e-5 is the scale for fine-tuning
    # a pretrained decoder; this head is randomly initialised (about 0.5M parameters,
    # encoder fully frozen) and would barely move at that rate over 8 epochs. 1e-3 is
    # the usual scale for a shallow head or linear probe. Every other hyperparameter
    # (epochs, batch, warmup, seed, bf16, freeze) matches structured SFT.
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass
class EncoderMTLConfig:
    model_dir: str = FINAL_STUDENT_MODEL_DIR
    run_name: str = "encoder_mtl"
    seed: int = 42

    epochs: int = 8
    per_device_batch_size: int = 32
    grad_accum_steps: int = 1
    dtype: str = "bf16"

    log_every_steps: int = 20

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    @property
    def run_dir(self) -> Path:
        return RUNS_DIR / self.run_name

    @property
    def ckpt_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def best_file(self) -> Path:
        return self.run_dir / "best.pt"

    @property
    def metrics_file(self) -> Path:
        return self.run_dir / "metrics.jsonl"


DEFAULT = EncoderMTLConfig()
