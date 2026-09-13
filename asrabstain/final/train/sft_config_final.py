"""Configuration for the four-cause SFT checkpoints (Structured SFT and GRPO).

Four cause tokens, Whisper-small as the student, and paths under
``config.OUTPUTS_DIR`` (set ``ASRABSTAIN_OUTPUTS`` to point at the artefact tree).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from asrabstain.config import BENCHMARK_DIR, OUTPUTS_DIR
from asrabstain.final.config_final import FINAL_REASONS, FINAL_STUDENT_MODEL_DIR

RUNS_DIR = OUTPUTS_DIR / "runs"

# Special tokens. The first content token is always the decision token.
DECISION_ANSWER = "<|answer|>"
DECISION_ABSTAIN = "<|abstain|>"
REASON_TOKENS: dict[str, str] = {r: f"<|{r}|>" for r in FINAL_REASONS}
SPECIAL_TOKENS: tuple[str, ...] = (
    DECISION_ANSWER,
    DECISION_ABSTAIN,
    *REASON_TOKENS.values(),
)
# The reason-free checkpoint adds only the decision tokens, so its vocabulary has
# no cause ids at all. That is a structural guarantee: the model cannot emit a
# cause token at any step, unlike masking the cause logits at decode time.
#
# Consequence: its vocabulary size differs from the four-cause checkpoints
# (51867 vs 51871) and the two cannot load each other's state dict. They are
# independent runs, so that is not needed.
HARD_SFT_SPECIAL_TOKENS: tuple[str, ...] = (
    DECISION_ANSWER,
    DECISION_ABSTAIN,
)


@dataclass
class DataConfig:
    benchmark_dir: Path = BENCHMARK_DIR
    train_file: str = "train.jsonl"
    dev_file: str = "dev.jsonl"
    target_sr: int = 16000
    allow_sstable_fallback: bool = True
    max_audio_seconds: float = 30.0
    # Data loading does soundfile reads, resampling and mel extraction on the CPU.
    # With a frozen encoder the decoder forward is fast, so the CPU can starve
    # the GPU; more workers keep it fed.
    num_workers: int = 8


@dataclass
class CurriculumConfig:
    """Degradation-aware curriculum, keyed on |degrade_dist - boundary|."""

    enabled: bool = True
    boundary: float = 0.30
    start_band: float = 0.30
    band_step: float = 0.10
    min_band: float = 0.0


@dataclass
class LossConfig:
    """Decision-weighted loss."""

    decision_token_weight: float = 2.0
    # Four causes but only one supervised token, so the cause loss is weighted
    reason_token_weight: float = 3.0
    label_smoothing: float = 0.0
    balance_decision_classes: bool = True
    balance_reason_classes: bool = True
    answer_share: float = 0.5


@dataclass
class OptimConfig:
    """Optimiser: frozen encoder, slightly lower peak learning rate."""

    freeze_encoder: bool = True
    # The decoder has about 3x the parameters of the base model, so the peak
    lr: float = 1.5e-5
    weight_decay: float = 0.0
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass
class SFTConfig:
    model_dir: str = FINAL_STUDENT_MODEL_DIR
    run_name: str = "sft_final"
    seed: int = 42
    hard_sft: bool = False  # True = reason-free ablation, no cause token or loss

    epochs: int = 8
    # Whisper-small at batch 32 uses roughly 60% of a 24G card; raise via --batch-size.
    per_device_batch_size: int = 32
    grad_accum_steps: int = 1
    dtype: str = "bf16"

    save_every_epoch: bool = True
    keep_last_n: int = 8
    eval_every_epoch: bool = True
    log_every_steps: int = 20

    data: DataConfig = field(default_factory=DataConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    @property
    def run_dir(self) -> Path:
        return RUNS_DIR / self.run_name

    @property
    def ckpt_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def best_dir(self) -> Path:
        return self.run_dir / "best"

    @property
    def metrics_file(self) -> Path:
        return self.run_dir / "metrics.jsonl"

    @property
    def state_file(self) -> Path:
        return self.run_dir / "trainer_state.json"


DEFAULT = SFTConfig()
