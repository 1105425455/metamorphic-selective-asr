"""Native Whisper decoding statistics for the confidence baseline.

Feature extraction for Whisper-Conf. The likelihood and entropy of each greedy
decode are read from ``output_scores``; no fitting happens here, and the logistic
regression coefficients live in the top-level ``whisper_conf_parameters.json``.

Decoding matches the settings used to produce the locked failure labels: greedy
(``num_beams=1``, ``do_sample=False``), English, ``task=transcribe``, at most 128
new tokens.

no-speech probability
---------------------
``generate`` does not expose the no-speech logits, so one explicit forward with
``decoder_input_ids=[SOT]`` supplies them, matching the OpenAI convention.

This checkpoint stores that probability on ``<|nocaptions|>`` (50362), not on the
``<|nospeech|>`` position: measured on one locked clip, p(50257) = 3.6e-8 and
p(50362) = 0.0657. Taking the maximum of the two covers either naming without
letting the unusable term dominate.

entropy
-------
Whisper's suppress-tokens processor sets 90 logits to negative infinity, so a
direct ``p * log p`` yields ``NaN``. The ``0 log 0 = 0`` convention is applied
with ``nan_to_num`` before summing.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# cuDNN is unstable for Whisper's conv1d front end; disable it globally.
torch.backends.cudnn.enabled = False

SR = 16_000

NOSPEECH_ID = 50257  # <|nospeech|> position, near zero for this checkpoint
NOCAPTIONS_ID = 50362  # <|nocaptions|>, which carries the no-speech mass here
SOT_ID = 50258  # <|startoftranscript|>


@dataclass
class ConfFeatures:
    """Raw decoding statistics for one clip, plus provenance fields.

    These are unstandardised; the scaler is fitted separately on development data.
    """

    uid: str
    hyp: str
    # The ten features fed to the logistic regression.
    mean_logprob: float  # mean log-softmax over generated tokens
    min_logprob: float  # least confident token, in logprob
    mean_entropy: float  # mean full-vocabulary entropy per step, in nats
    max_entropy: float  # peak per-step entropy
    no_speech_prob: float  # no-speech probability at the SOT position
    compression_ratio: float  # len(utf8) / len(zlib(utf8)), a repetition detector
    n_tokens: float  # generated tokens, excluding special ones
    duration_s: float  # clip duration in seconds
    tokens_per_sec: float  # n_tokens / duration_s
    seq_avg_logprob: float  # same value as mean_logprob, kept for baseline naming
    # Labels and provenance; not features.
    unsafe: int = -1
    wer: float = -1.0
    numeric_error: int = 0
    cause: str = ""
    reference: str = ""
    origin: str = ""  # in-domain stratum (clean / hard_positive / abstain)
    target: str = ""  # constructive label, never used to build the feature set


FEATURE_NAMES: tuple[str, ...] = (
    "mean_logprob",
    "min_logprob",
    "mean_entropy",
    "max_entropy",
    "no_speech_prob",
    "compression_ratio",
    "n_tokens",
    "duration_s",
    "tokens_per_sec",
    "seq_avg_logprob",
)


def compression_ratio(text: str) -> float:
    """Compression ratio: raw bytes over zlib-compressed bytes.

    OpenAI Whisper uses this to spot degenerate repetition: repeated text
    compresses well, so the ratio rises. Empty text returns 0.0.
    """
    if not text:
        return 0.0
    raw = text.encode("utf-8")
    return float(len(raw) / len(zlib.compress(raw)))


def _load_wav(path: Path) -> np.ndarray:
    """Read a wav and resample to 16 kHz mono."""
    import soundfile as sf

    w, sr = sf.read(str(path), dtype="float32")
    if w.ndim > 1:
        w = w.mean(axis=-1)
    if sr != SR:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(SR, sr)
        w = resample_poly(w, SR // g, sr // g).astype(np.float32)
    return np.asarray(w, dtype="float32")


@torch.no_grad()
def _no_speech_probs(model, feats: torch.Tensor) -> np.ndarray:
    """No-speech probability at the SOT position, from one forward pass.

    Takes max(p[<|nospeech|>], p[<|nocaptions|>]) because checkpoints differ
    in which token carries it; here it is nocaptions and the other is near zero.
    """
    bsz = feats.shape[0]
    dec = torch.full((bsz, 1), SOT_ID, dtype=torch.long, device=feats.device)
    logits = model(input_features=feats, decoder_input_ids=dec).logits[:, 0].float()
    probs = torch.softmax(logits, dim=-1)
    ns = torch.maximum(probs[:, NOSPEECH_ID], probs[:, NOCAPTIONS_ID])
    return ns.cpu().numpy()


@torch.no_grad()
def _decode_batch(model, proc, wavs: list[np.ndarray], max_new: int) -> list[dict]:
    """Greedy decode one batch, returning token-level statistics.

    `output_scores=True` exposes the per-step logits, from which the logprob and
    entropy are computed. Entropy uses the full vocabulary distribution,
    H = -sum p log p in nats, which shows hesitation better than top-1 does.

    Whisper suppresses 90 tokens by setting their logits to -inf, which makes
    `p * log p` produce NaN there. Applying the convention 0 log 0 = 0 and
    replacing the -inf logs with zero keeps the sum well defined.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    feats = proc.feature_extractor(
        wavs, sampling_rate=SR, return_tensors="pt"
    ).input_features.to(device, dtype=dtype)

    ns = _no_speech_probs(model, feats)
    gen = model.generate(
        feats,
        language="en",
        task="transcribe",
        max_new_tokens=max_new,
        do_sample=False,
        num_beams=1,
        output_scores=True,
        return_dict_in_generate=True,
    )
    seqs = gen.sequences  # [B, prefix + T]
    scores = gen.scores  # T tensors of shape [B, V]
    n_new = len(scores)
    gen_tokens = seqs[:, seqs.shape[1] - n_new:]  # aligned with scores

    eos = proc.tokenizer.eos_token_id
    special = set(proc.tokenizer.all_special_ids)

    step_lp = np.zeros((len(wavs), n_new), dtype=np.float64)
    step_ent = np.zeros((len(wavs), n_new), dtype=np.float64)
    for t, logit in enumerate(scores):
        lsm = torch.log_softmax(logit.float(), dim=-1)
        tok = gen_tokens[:, t]
        step_lp[:, t] = lsm.gather(1, tok.unsqueeze(1)).squeeze(1).cpu().numpy()
        p = lsm.exp()
        # 0 log 0 = 0: suppressed tokens give logit -inf, so p is 0.
        safe_lsm = torch.nan_to_num(lsm, neginf=0.0)
        step_ent[:, t] = (-(p * safe_lsm).sum(dim=-1)).cpu().numpy()

    texts = proc.batch_decode(seqs, skip_special_tokens=True)
    out: list[dict] = []
    for b in range(len(wavs)):
        toks = gen_tokens[b].tolist()
        # Valid steps: before EOS and not special tokens.
        valid: list[int] = []
        for t, tid in enumerate(toks):
            if tid == eos:
                break
            if tid not in special:
                valid.append(t)
        if valid:
            lp = step_lp[b, valid]
            ent = step_ent[b, valid]
        else:  # Empty output: fall back to a low-confidence value.
            lp = np.array([-10.0])
            ent = np.array([0.0])
        out.append({
            "hyp": texts[b].strip(),
            "mean_logprob": float(lp.mean()),
            "min_logprob": float(lp.min()),
            "mean_entropy": float(ent.mean()),
            "max_entropy": float(ent.max()),
            "no_speech_prob": float(ns[b]),
            "n_tokens": float(len(valid)),
        })
    return out
