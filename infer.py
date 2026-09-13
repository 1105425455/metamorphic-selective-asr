"""Inference for the metamorphic selective-ASR checkpoints.

The wrapper here loads the released checkpoints and calls the original
implementation modules: ``asrabstain.eval.decode`` and
``asrabstain.final.eval.decode_final`` for the generative interfaces,
``asrabstain.final.train.encoder_mtl_model`` for the discriminative one, and
``final_eval.decode_whisper_conf`` for the native-confidence baseline.

Generative checkpoints are HuggingFace Whisper directories. Encoder-MTL ships a
single head checkpoint (``best.pt``). Whisper-Conf runs the unmodified base
model and applies the coefficients in ``whisper_conf_parameters.json``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "final_eval"))

TARGET_SR = 16000
_DTYPES = {"bf16": "bfloat16", "bfloat16": "bfloat16",
           "fp16": "float16", "fp32": "float32"}


@dataclass
class Prediction:
    decision: str        # "ANSWER" or "ABSTAIN"
    p_abstain: float     # probability of abstaining at the decision position
    cause: str           # cause name when abstaining, otherwise ""
    transcript: str      # best-effort transcript; "" for Encoder-MTL
    seq_avg_logprob: float = 0.0


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _cast(dtype: str):
    return getattr(torch, _DTYPES.get(dtype, dtype))


def read_audio(path: str | Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """Read a wav/flac file as mono float32 at ``target_sr``."""
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if sr != target_sr:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(target_sr, sr)
        wav = resample_poly(wav, target_sr // g, sr // g).astype(np.float32)
    return np.asarray(wav, dtype=np.float32)


class GenerativeInterface:
    """Structured SFT, Reason-free SFT, and both GRPO checkpoints."""

    def __init__(self, model_dir: str | Path, device: str | None = None,
                 dtype: str | None = None):
        from transformers import WhisperProcessor

        from asrabstain.final.eval.decode_final import AbstainDecoder, HardAbstainDecoder

        device = device or default_device()
        self.device = device
        self.dtype = _cast(dtype or _default_dtype(device))
        self.processor = WhisperProcessor.from_pretrained(str(model_dir))
        self.model = _load_whisper(str(model_dir), self.dtype).to(device).eval()
        tok = self.processor.tokenizer
        reason_free = tok.convert_tokens_to_ids("<|inaudible_noise|>") == tok.unk_token_id
        self.reason_free = reason_free
        self._decoder = (HardAbstainDecoder if reason_free else AbstainDecoder)(
            self.model, self.processor, device, self.dtype)

    def predict(self, waveforms: list[np.ndarray], max_new: int = 128) -> list[Prediction]:
        if not waveforms:
            return []
        return [
            Prediction(r.decision, r.p_abstain, r.reason, r.transcript, r.seq_avg_logprob)
            for r in self._decoder.decode_batch(waveforms, TARGET_SR, max_new=max_new)
        ]

    def predict_paths(self, paths: Iterable[str | Path],
                      max_new: int = 128) -> list[Prediction]:
        return self.predict([read_audio(p) for p in paths], max_new=max_new)


class EncoderMTLInterface:
    """Encoder-MTL: action and cause without a decoder."""

    def __init__(self, ckpt: str | Path, whisper_dir: str = "openai/whisper-small",
                 device: str | None = None, fallback_model_dir: str | Path | None = None):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        from asrabstain.final.train.encoder_mtl_config import EncoderMTLConfig
        from asrabstain.final.train.encoder_mtl_model import build_encoder_mtl

        device = device or default_device()
        source = self._resolve_source(whisper_dir, fallback_model_dir)
        cfg = EncoderMTLConfig(model_dir=source)
        self.model, self.processor = build_encoder_mtl(cfg)
        state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        self.model.load_state_dict(state.get("model", state))
        self.model = self.model.to(device).eval()
        self.device = device
        self.encoder_dtype = next(self.model.encoder.parameters()).dtype

    @staticmethod
    def _resolve_source(whisper_dir: str, fallback_model_dir: str | Path | None) -> str:
        if Path(whisper_dir).is_dir():
            return whisper_dir
        if fallback_model_dir is not None:
            fb = Path(fallback_model_dir)
            if (fb / "preprocessor_config.json").exists():
                return str(fb)
        return whisper_dir

    def _features(self, waveforms: list[np.ndarray]):
        return self.processor.feature_extractor(
            waveforms, sampling_rate=TARGET_SR, return_tensors="pt"
        ).input_features.to(self.device, dtype=self.encoder_dtype)

    def predict(self, waveforms: list[np.ndarray]) -> list[Prediction]:
        from asrabstain.final.train.encoder_mtl_config import CAUSE_CLASSES

        if not waveforms:
            return []
        with torch.no_grad():
            action_logits, cause_logits = self.model(self._features(waveforms))
        probs = torch.softmax(action_logits.float(), dim=-1).cpu().numpy()
        causes = cause_logits.float().argmax(dim=-1).cpu().numpy()
        return [
            Prediction("ABSTAIN" if probs[i, 1] > 0.5 else "ANSWER",
                       float(probs[i, 1]),
                       CAUSE_CLASSES[int(causes[i])] if probs[i, 1] > 0.5 else "",
                       "")
            for i in range(len(waveforms))
        ]

    def cause_only(self, waveforms) -> list[str]:
        from asrabstain.final.train.encoder_mtl_config import CAUSE_CLASSES

        if waveforms and isinstance(waveforms[0], (str, Path)):
            waveforms = [read_audio(p) for p in waveforms]
        if not waveforms:
            return []
        with torch.no_grad():
            _, cause_logits = self.model(self._features(waveforms))
        idx = cause_logits.float().argmax(dim=-1).cpu().numpy()
        return [CAUSE_CLASSES[int(i)] for i in idx]

    def predict_paths(self, paths: Iterable[str | Path]) -> list[Prediction]:
        return self.predict([read_audio(p) for p in paths])


class WhisperConfInterface:
    """Naked Whisper-small plus a logistic regression on decoding statistics."""

    def __init__(self, model_dir: str, device: str | None = None,
                 dtype: str | None = None, params: str | Path | None = None):
        from transformers import WhisperProcessor

        device = device or default_device()
        self.device = device
        self.params = json.loads(
            Path(params or _ROOT / "whisper_conf_parameters.json").read_text())
        self.processor = WhisperProcessor.from_pretrained(model_dir)
        self.model = _load_whisper(
            model_dir, _cast(dtype or _default_dtype(device))).to(device).eval()

    def score(self, features: dict[str, float]) -> float:
        p = self.params
        x = np.asarray([float(features[k]) for k in p["features"]], dtype=float)
        mean = np.asarray(p["scaler_mean"], dtype=float)
        scale = np.asarray(p["scaler_scale"], dtype=float)
        coef = np.asarray(p["coefficients"], dtype=float)
        logit = float(((x - mean) / scale) @ coef + p["intercept"])
        return 1.0 / (1.0 + np.exp(-logit))

    def predict_paths(self, paths: Iterable[str | Path]) -> list[Prediction]:
        wavs = [read_audio(p) for p in paths]
        if not wavs:
            return []
        feats = self.features(wavs)
        out = []
        for f in feats:
            p_abstain = self.score(f)
            out.append(Prediction("ABSTAIN" if p_abstain > 0.5 else "ANSWER",
                                  p_abstain, "", f["hyp"],
                                  f["seq_avg_logprob"]))
        return out

    def features(self, wavs: list[np.ndarray]) -> list[dict[str, float]]:
        """The ten decoding statistics used by the classifier.

        ``_decode_batch`` returns the logprob, entropy and no-speech terms for the
        whole batch; the remaining terms are derived here with the same formulas
        the original feature-extraction script used. ``seq_avg_logprob`` mirrors
        ``mean_logprob``, as in that script.
        """
        from decode_whisper_conf import _decode_batch, compression_ratio

        stats = _decode_batch(self.model, self.processor, wavs, 128)
        out = []
        for wav, s in zip(wavs, stats):
            dur = float(len(wav) / TARGET_SR)
            out.append({
                "hyp": s["hyp"],
                "mean_logprob": s["mean_logprob"],
                "min_logprob": s["min_logprob"],
                "mean_entropy": s["mean_entropy"],
                "max_entropy": s["max_entropy"],
                "no_speech_prob": s["no_speech_prob"],
                "compression_ratio": compression_ratio(s["hyp"]),
                "n_tokens": s["n_tokens"],
                "duration_s": dur,
                "tokens_per_sec": float(s["n_tokens"] / max(dur, 1e-6)),
                "seq_avg_logprob": s["mean_logprob"],
            })
        return out


def _default_dtype(device: str) -> str:
    """Default precision per device.

    On GPU this is float16, the precision the released predictions were decoded
    in, so a fresh run matches the stored outputs. On CPU it is float32, since
    float16 and bfloat16 are not supported by every CPU and PyTorch combination.
    Pass ``dtype`` explicitly to override.
    """
    return "fp16" if device.startswith("cuda") else "fp32"


def _load_whisper(model_dir: str, dtype):
    """Load a Whisper checkpoint at the given precision.

    ``torch_dtype`` is accepted by every transformers release; newer versions
    also accept ``dtype``. The older spelling is tried first, and the resulting
    precision is checked in case an unknown keyword was ignored.
    """
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=dtype)
    if next(model.parameters()).dtype != dtype:
        model = WhisperForConditionalGeneration.from_pretrained(model_dir, dtype=dtype)
    return model
