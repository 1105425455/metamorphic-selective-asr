"""Decoders for the released generative checkpoints.

``AbstainDecoder`` reuses the shared decoding logic and rebuilds only the cause
table, giving the four causes used by Structured SFT and both GRPO checkpoints.

``HardAbstainDecoder`` handles the reason-free checkpoint, whose vocabulary has
no cause tokens. It drops the cause step and returns an empty cause, so the
sequence is fixed to ``[SOT][notimestamps]<decision><transcript...>``.
"""

from __future__ import annotations

import torch

from asrabstain.eval.decode import AbstainDecoder as _BaseDecoder
from asrabstain.eval.decode import DecodeResult  # noqa: F401  re-exported
from asrabstain.final.train.sft_config_final import (
    DECISION_ABSTAIN,
    DECISION_ANSWER,
    REASON_TOKENS,
)


class AbstainDecoder(_BaseDecoder):
    """Decoder whose cause table holds the four released causes."""

    def __init__(self, model, processor, device, dtype) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.dtype = dtype
        tok = processor.tokenizer
        self.answer_id = tok.convert_tokens_to_ids(DECISION_ANSWER)
        self.abstain_id = tok.convert_tokens_to_ids(DECISION_ABSTAIN)
        self.reason_ids = {
            r: tok.convert_tokens_to_ids(t) for r, t in REASON_TOKENS.items()
        }
        self.id2reason = {v: k for k, v in self.reason_ids.items()}
        self.sot_id = tok.convert_tokens_to_ids("<|startoftranscript|>")
        self.notimestamps_id = tok.convert_tokens_to_ids("<|notimestamps|>")
        self.eos_id = tok.eos_token_id


class HardAbstainDecoder(_BaseDecoder):
    """Decoder for the reason-free checkpoint: ANSWER/ABSTAIN only.

    That vocabulary has no cause tokens, so ``AbstainDecoder`` cannot be reused:
    converting the four cause strings would return the unknown id for each, the
    cause step would then pick among unknown ids, and the forced prefix would
    become a sequence the model never saw during training.

    This class reimplements ``decode_batch`` without the cause step. The sequence
    is fixed to:
      ANSWER:  [SOT][notimestamps]<|answer|>  <transcript...>
      ABSTAIN: [SOT][notimestamps]<|abstain|> <transcript...>
    The cause is always "" and reason_probs is empty.
    """

    def __init__(self, model, processor, device, dtype) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.dtype = dtype
        tok = processor.tokenizer
        self.answer_id = tok.convert_tokens_to_ids(DECISION_ANSWER)
        self.abstain_id = tok.convert_tokens_to_ids(DECISION_ABSTAIN)
        # No cause tokens: set them explicitly so misuse surfaces immediately
        self.reason_ids = {}
        self.id2reason = {}
        self.sot_id = tok.convert_tokens_to_ids("<|startoftranscript|>")
        self.notimestamps_id = tok.convert_tokens_to_ids("<|notimestamps|>")
        self.eos_id = tok.eos_token_id

    @torch.no_grad()
    def decode_batch(
        self, waveforms: list, target_sr: int = 16000, max_new: int = 128
    ) -> list[DecodeResult]:
        feats = self.processor.feature_extractor(
            waveforms, sampling_rate=target_sr, return_tensors="pt"
        ).input_features.to(self.device, dtype=self.dtype)
        enc = self.model.model.encoder(feats).last_hidden_state

        bsz = feats.shape[0]
        prefix = torch.tensor(
            [[self.sot_id, self.notimestamps_id]], device=self.device
        ).repeat(bsz, 1)

        # Decision position: same convention as the four-cause decoder.
        out = self.model.model.decoder(input_ids=prefix, encoder_hidden_states=enc)
        logits = self.model.proj_out(out.last_hidden_state[:, -1, :]).float()
        probs = torch.softmax(logits, dim=-1)
        p_abstain = probs[:, self.abstain_id].cpu().numpy()
        is_abstain = (logits[:, self.abstain_id] > logits[:, self.answer_id]).cpu().numpy()

        # Continue straight to the transcript; there is no cause step.
        prefixes = [
            [self.sot_id, self.notimestamps_id,
             self.abstain_id if is_abstain[i] else self.answer_id]
            for i in range(bsz)
        ]
        texts, avg_lps = self._greedy_transcript_batch(enc, prefixes, max_new)

        return [
            DecodeResult(
                decision="ABSTAIN" if is_abstain[i] else "ANSWER",
                transcript=texts[i],
                reason="",           # reason-free: no cause emitted
                p_abstain=float(p_abstain[i]),
                seq_avg_logprob=avg_lps[i],
                reason_probs={},
            )
            for i in range(bsz)
        ]
