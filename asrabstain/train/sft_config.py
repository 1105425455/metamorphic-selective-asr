"""Special-token definitions for the earlier three-cause checkpoints.

The four-cause configuration imports these names; at inference time only the
token strings matter. The sequence contract is::

    [SOT][notimestamps] <decision> [<cause>] <transcript...>

The decision token is always the first content token. Changing a token string
silently produces unknown ids at load time, so keep them in sync with the
released weights.
"""

from __future__ import annotations

DECISION_ANSWER = "<|answer|>"
DECISION_ABSTAIN = "<|abstain|>"
REASON_TOKENS = {
    "inaudible_noise": "<|inaudible_noise|>",
    "overlapping_speech": "<|overlapping_speech|>",
    "incomplete_speech": "<|incomplete_speech|>",
}
SPECIAL_TOKENS: tuple[str, ...] = (
    DECISION_ANSWER,
    DECISION_ABSTAIN,
    *REASON_TOKENS.values(),
)
