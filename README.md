# Metamorphic Selective ASR — inference code

Inference code for **Metamorphic Acoustic Supervision for Selective ASR:
Transfer Asymmetry and Routing-Aware Composition**.

Each clip gets an *action* (answer or abstain) and, when abstaining, a *cause*.

## Install

```bash
pip install torch transformers numpy soundfile scipy
```

Python 3.10 or newer, with `torch>=2.1` and `transformers>=4.40`. Runs on GPU
when available, otherwise CPU.

## Checkpoints

Run from the repository root; each archive is unpacked into `bigdata/`:

```bash
for f in ckpt_structured_sft ckpt_hard_sft ckpt_grpo_frozen \
         ckpt_grpo_unfrozen ckpt_encoder_mtl; do
  curl -LO "https://github.com/1105425455/metamorphic-selective-asr/releases/download/icassp2027-submission/$f.tar.gz"
  mkdir -p "bigdata/$f" && tar -xzf "$f.tar.gz" -C "bigdata/$f" && rm "$f.tar.gz"
done
```

## Usage

```python
from infer import GenerativeInterface

model = GenerativeInterface("bigdata/ckpt_structured_sft")
pred = model.predict_paths(["examples/audio/bandlimited_sev3.wav"])[0]

pred.decision     # "ANSWER" or "ABSTAIN"
pred.p_abstain    # P(abstain) at the decision position
pred.cause        # cause name when abstaining, otherwise ""
pred.transcript   # best-effort transcript; "" for Encoder-MTL
```

`p_abstain` is the ranking score behind AURC, ECE and Unsafe@60.

### Interfaces

| Class | Checkpoint | Emits transcript |
|---|---|---|
| `GenerativeInterface` | `structured_sft`, `hard_sft`, `grpo_frozen`, `grpo_unfrozen` | yes |
| `EncoderMTLInterface` | `encoder_mtl/best.pt` | no |
| `WhisperConfInterface` | base `whisper-small` | yes |

The generative checkpoints bundle a processor. The other two need a
`whisper-small` directory:

```python
from infer import EncoderMTLInterface, WhisperConfInterface

mtl = EncoderMTLInterface("bigdata/ckpt_encoder_mtl/best.pt",
                          whisper_dir="/path/to/whisper-small")
conf = WhisperConfInterface("/path/to/whisper-small")
```

For `EncoderMTLInterface` only, `fallback_model_dir="bigdata/ckpt_structured_sft"`
supplies the processor and encoder configuration when no local `whisper-small`
is available. The released Encoder-MTL checkpoint contains the encoder weights
used at training time, and these are loaded over whatever the fallback provided,
so the encoder is the same either way.

### Composition

```python
clip = "examples/audio/bandlimited_sev3.wav"
action = conf.predict_paths([clip])[0]
if action.p_abstain > 0.5:
    cause = mtl.cause_only([clip])[0]
```

`cause_only` returns the cause argmax without consulting Encoder-MTL's own
action threshold.

## Causes

`inaudible_noise`, `bandlimited_muffled`, `overlapping_speech`,
`incomplete_speech`. The reason-free checkpoint (`ckpt_hard_sft`) has no cause
vocabulary and always returns an empty cause.

`EncoderMTLInterface` returns names from this set; its head rows are ordered
`inaudible_noise`, `bandlimited_muffled`, `overlapping_speech`,
`incomplete_speech`, which is an internal detail and not part of the API.

## Example clips

One mother utterance: the clean original, plus severity-3 bandlimiting and
truncation. Outputs from `structured_sft`:

```
clean.wav             ANSWER
bandlimited_sev3.wav  ABSTAIN  bandlimited_muffled
truncated_sev3.wav    ABSTAIN  incomplete_speech
```

## Data

The example clips come from LibriSpeech `train-clean-100`:

> V. Panayotov, G. Chen, D. Povey and S. Khudanpur, "LibriSpeech: An ASR corpus
> based on public domain audio books," ICASSP 2015.

LibriSpeech is released under CC BY 4.0 (https://www.openslr.org/12). The
severity-3 variants under `examples/audio/` were derived from one of its clips by
applying bandlimiting and truncation, and are redistributed under the same
licence. GigaSpeech, TED-LIUM 3 and ATCO2 corpora are not redistributed here.

The MIT licence at the repository root covers the source code only.
`openai/whisper-small` is MIT, held by OpenAI.

## Citation

```bibtex
@inproceedings{metamorphic_selective_asr,
  title  = {Metamorphic Acoustic Supervision for Selective ASR:
            Transfer Asymmetry and Routing-Aware Composition},
  author = {Huang, Yongbin and Deng, Xiaoling and Lan, Yubin and Li, Zhen},
  year   = {2027}
}
```
