"""Decision decoding and p_abstain / logprob extraction at evaluation time.

The sequence is fixed by training::

    [SOT][notimestamps] <decision> [<cause>] <transcript...>
    ANSWER : <|answer|> <transcript tokens...>
    ABSTAIN: <|abstain|> <|cause token|> <transcript tokens...>

Per sample this extracts:

* ``p_abstain`` -- full-vocabulary softmax on <|abstain|> at the decision
  position. This is the ranking score behind AURC.
* ``decision`` -- argmax between the answer and abstain logits only, so an
  ordinary token cannot influence the choice.
* transcript -- greedy continuation after the decision, and after the cause when
  abstaining.
* ``seq_avg_logprob`` -- mean token logprob over the generated span.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from asrabstain.train.sft_config import (
    DECISION_ABSTAIN,
    DECISION_ANSWER,
    REASON_TOKENS,
)


@dataclass
class DecodeResult:
    decision: str  # "ANSWER" / "ABSTAIN"
    transcript: str  # transcript when answering, "" when abstaining
    reason: str  # cause name when abstaining, "" when answering
    p_abstain: float  # softmax of <|abstain|> at the decision position
    seq_avg_logprob: float  # mean logprob over the generated span
    # reason_probs: softmax over the reason tokens at the cause position.
    # Read from the logits already produced at that position; the model is not
    # changed and nothing is retrained. Order follows REASON_TOKENS keys.
    # Computed for ANSWER samples too, as a counterfactual.
    reason_probs: dict[str, float] = None  # type: ignore[assignment]


class AbstainDecoder:
    """Decoder for the Whisper-abstain models."""

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
        # Forced prefix [SOT][notimestamps], matching the training labels.
        self.sot_id = tok.convert_tokens_to_ids("<|startoftranscript|>")
        self.notimestamps_id = tok.convert_tokens_to_ids("<|notimestamps|>")
        self.eos_id = tok.eos_token_id

    @torch.no_grad()
    def decode_batch(
        self, waveforms: list[np.ndarray], target_sr: int = 16000, max_new: int = 128
    ) -> list[DecodeResult]:
        feats = self.processor.feature_extractor(
            waveforms, sampling_rate=target_sr, return_tensors="pt"
        ).input_features.to(self.device, dtype=self.dtype)
        enc = self.model.model.encoder(feats).last_hidden_state

        bsz = feats.shape[0]
        prefix = torch.tensor(
            [[self.sot_id, self.notimestamps_id]], device=self.device
        ).repeat(bsz, 1)

        # Step 1: decision position.
        out = self.model.model.decoder(
            input_ids=prefix, encoder_hidden_states=enc
        )
        logits = self.model.proj_out(out.last_hidden_state[:, -1, :]).float()  # (B,V)
        probs = torch.softmax(logits, dim=-1)
        p_abstain = probs[:, self.abstain_id].cpu().numpy()

        # Decision = argmax between the answer and abstain logits only.
        ans_l = logits[:, self.answer_id]
        abs_l = logits[:, self.abstain_id]
        is_abstain = (abs_l > ans_l).cpu().numpy()

        # Step 2: cause position. Computed for the whole batch, restricted to the
        # reason tokens. The step always feeds [...prefix, abstain_id]: for an
        # ABSTAIN sample that is the real position, for an ANSWER sample it is a
        # counterfactual used only for analysis.
        rname_list = list(self.reason_ids.keys())
        rid_list = list(self.reason_ids.values())
        reasons: list[str] = [""] * bsz
        reason_lp = [0.0] * bsz
        reason_probs_all: list[dict[str, float]] = [
            {r: 0.0 for r in rname_list} for _ in range(bsz)]
        ids_all = torch.cat(
            [prefix, torch.full((bsz, 1), self.abstain_id, device=self.device)], dim=1)
        out_r = self.model.model.decoder(
            input_ids=ids_all, encoder_hidden_states=enc)
        rl = self.model.proj_out(out_r.last_hidden_state[:, -1, :]).float()  # (B,V)
        rlp = torch.log_softmax(rl, dim=-1)
        sub = rl[:, rid_list]                       # (B,3) cause logits
        sub_probs = torch.softmax(sub, dim=-1)      # (B,3) normalised over causes
        pick = sub.argmax(dim=-1)                   # (B,)
        for i in range(bsz):
            for j, r in enumerate(rname_list):
                reason_probs_all[i][r] = float(sub_probs[i, j].item())
            if is_abstain[i]:
                reasons[i] = rname_list[int(pick[i])]
                reason_lp[i] = float(rlp[i, rid_list[int(pick[i])]].item())

        # Step 3: decode the transcript with a KV cache, stopping per sample.
        # Forced prefixes: ANSWER gets [..,answer]; ABSTAIN [..,abstain,cause].
        prefixes: list[list[int]] = []
        for i in range(bsz):
            if is_abstain[i]:
                prefixes.append([self.sot_id, self.notimestamps_id, self.abstain_id,
                                 self.reason_ids[reasons[i]]])
            else:
                prefixes.append([self.sot_id, self.notimestamps_id, self.answer_id])
        texts, avg_lps = self._greedy_transcript_batch(enc, prefixes, max_new)

        results: list[DecodeResult] = []
        for i in range(bsz):
            if is_abstain[i]:
                seq_lp = ((reason_lp[i] + avg_lps[i]) / 2) if texts[i] else reason_lp[i]
                results.append(DecodeResult(
                    decision="ABSTAIN", transcript=texts[i], reason=reasons[i],
                    p_abstain=float(p_abstain[i]), seq_avg_logprob=seq_lp,
                    reason_probs=reason_probs_all[i]))
            else:
                results.append(DecodeResult(
                    decision="ANSWER", transcript=texts[i], reason="",
                    p_abstain=float(p_abstain[i]), seq_avg_logprob=avg_lps[i],
                    reason_probs=reason_probs_all[i]))
        return results

    def _greedy_transcript_batch(
        self, enc, prefixes: list[list[int]], max_new: int
    ) -> tuple[list[str], list[float]]:
        """Greedy transcript decoding with a KV cache.

        prefixes[i] is the forced prefix of sample i.
        Samples are grouped by prefix length, so each group is one batched
        forward. Generation stops at eos or after six repeats of one token.
        Returns (texts, avg_logprobs) in the original order.
        """
        n = len(prefixes)
        texts = [""] * n
        avg_lps = [0.0] * n
        groups: dict[int, list[int]] = {}
        for i, p in enumerate(prefixes):
            groups.setdefault(len(p), []).append(i)

        for _plen, idxs in groups.items():
            ids = torch.tensor([prefixes[i] for i in idxs], device=self.device)
            enc_g = enc[idxs]
            g = len(idxs)
            gen_ids: list[list[int]] = [[] for _ in range(g)]
            logps: list[list[float]] = [[] for _ in range(g)]
            done = [False] * g
            repeat_run = [0] * g
            last_tok = [None] * g

            past = None
            cur = ids
            for _ in range(max_new):
                out = self.model.model.decoder(
                    input_ids=cur, encoder_hidden_states=enc_g,
                    past_key_values=past, use_cache=True,
                )
                past = out.past_key_values
                step = self.model.proj_out(out.last_hidden_state[:, -1, :]).float()  # (g,V)
                lp = torch.log_softmax(step, dim=-1)
                nxt = step.argmax(dim=-1)  # (g,)
                nxt_l = nxt.tolist()
                for k in range(g):
                    if done[k]:
                        continue
                    t = nxt_l[k]
                    logps[k].append(float(lp[k, t].item()))
                    if t == self.eos_id:
                        done[k] = True
                        continue
                    repeat_run[k] = repeat_run[k] + 1 if t == last_tok[k] else 0
                    last_tok[k] = t
                    if repeat_run[k] >= 6:
                        done[k] = True
                        continue
                    gen_ids[k].append(t)
                if all(done):
                    break
                cur = nxt.unsqueeze(1)  # cache is warm; feed only the new token

            for k, i in enumerate(idxs):
                texts[i] = self.processor.tokenizer.decode(
                    gen_ids[k], skip_special_tokens=True).strip()
                avg_lps[i] = float(np.mean(logps[k])) if logps[k] else 0.0
        return texts, avg_lps
