"""Encoder-MTL: a frozen Whisper encoder with pooled features and two heads.

    log-mel (B, 80, 3000)
      -> WhisperEncoder             (B, 1500, 768)   frozen by default
      -> pooling (attention | mean) (B, 768)
      -> shared MLP 768->256 + GELU + dropout
      -> action head 256->2         ANSWER / ABSTAIN
      -> cause head  256->4         inaudible_noise / bandlimited_muffled /
                                    overlapping_speech / incomplete_speech

Both heads read the same MLP output, which is the usual multi-task arrangement:
a shared representation with task-specific output layers. Fully independent heads
would be two separate probes rather than multi-task learning.

The model has no decoder and produces no transcript; the reported transcript is
an empty string.
"""

from __future__ import annotations

import torch
from torch import nn

from asrabstain.final.train.encoder_mtl_config import CAUSE_CLASSES, EncoderMTLConfig


class AttentionPooling(nn.Module):
    """Single-query additive attention pooling: a_t = softmax(w2 tanh(W1 h_t)).

    With a frozen encoder this is the only component that can weight the time
    axis. Evidence for truncation sits at the end of the clip and evidence for
    overlap is local, so averaging over 1500 frames dilutes it.
    """

    def __init__(self, dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, hidden)
        self.query = nn.Linear(hidden, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Align the encoder output with this module's weight dtype (bf16 encoder,
        # fp32 head). The attention weights pass through a softmax and are more
        # stable in fp32; the upcast touches activations, not weights.
        x = x.to(self.proj.weight.dtype)
        # x: (B, T, D) → scores: (B, T)
        scores = self.query(torch.tanh(self.proj(x))).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        return torch.bmm(attn.unsqueeze(1), x).squeeze(1)


class EncoderMTL(nn.Module):
    """Discriminative abstention and cause attribution model."""

    def __init__(self, encoder: nn.Module, cfg: EncoderMTLConfig) -> None:
        super().__init__()
        self.encoder = encoder
        self.cfg = cfg
        dim = encoder.config.d_model
        mcfg = cfg.model

        self.pooling_kind = mcfg.pooling
        if mcfg.pooling == "attention":
            self.pool: nn.Module | None = AttentionPooling(dim, mcfg.hidden_dim)
        elif mcfg.pooling == "mean":
            self.pool = None
        else:
            raise ValueError(f"unknown pooling: {mcfg.pooling} (attention|mean)")

        self.trunk = nn.Sequential(
            nn.Linear(dim, mcfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(mcfg.dropout),
        )
        # Binary head: index 0 = ANSWER, 1 = ABSTAIN; p_abstain = softmax(...)[1].
        self.action_head = nn.Linear(mcfg.hidden_dim, 2)
        # Four-way cause head; row order is CAUSE_CLASSES and is fixed.
        self.cause_head = nn.Linear(mcfg.hidden_dim, len(CAUSE_CLASSES))

        if cfg.optim.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def head_parameters(self):
        """Head parameters only (pooling, trunk, both heads), for the optimizer."""
        mods: list[nn.Module] = [self.trunk, self.action_head, self.cause_head]
        if self.pool is not None:
            mods.append(self.pool)
        for m in mods:
            yield from m.parameters()

    def forward(self, input_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (action_logits (B,2), cause_logits (B,4))."""
        if self.cfg.optim.freeze_encoder:
            # No graph for a frozen encoder: it is most of the compute.
            with torch.no_grad():
                hidden = self.encoder(input_features).last_hidden_state
            hidden = hidden.detach()
        else:
            hidden = self.encoder(input_features).last_hidden_state

        # Whisper always pads to 30s and uses no attention mask: padding frames
        # are valid log-mel silence, matching the encoder's own behaviour.
        #
        # dtype boundary
        # The encoder runs in bf16 and is frozen; the pooling, trunk and heads stay
        # fp32. They hold about 0.5M parameters, so the cost is negligible and the
        # gain is numerical stability for softmax, cross-entropy and AdamW.
        # Cast before pooling: the attention Linear would otherwise see bf16
        # input and raise a dtype mismatch on GPU.
        head_dtype = self.trunk[0].weight.dtype
        hidden = hidden.to(head_dtype)
        pooled = hidden.mean(dim=1) if self.pool is None else self.pool(hidden)
        z = self.trunk(pooled)
        return self.action_head(z), self.cause_head(z)


def build_encoder_mtl(cfg: EncoderMTLConfig) -> tuple[EncoderMTL, object]:
    """Build Encoder-MTL on a Whisper encoder. Returns (model, processor).

    The vocabulary is not extended: this model does not go through the tokenizer
    to produce a decision, so the processor is used only for log-mel features.
    """
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(cfg.model_dir)
    full = WhisperForConditionalGeneration.from_pretrained(cfg.model_dir)
    encoder = full.model.encoder
    model = EncoderMTL(encoder, cfg)
    return model, processor
