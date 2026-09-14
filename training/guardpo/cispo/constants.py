"""Constants shared by data preparation, rewards, and the VERL plugin."""

from __future__ import annotations

LABEL_SAFE = "safe"
LABEL_UNSAFE = "unsafe"
VALID_LABELS = (LABEL_SAFE, LABEL_UNSAFE)

LABEL_TO_INT = {
    LABEL_SAFE: 0,
    LABEL_UNSAFE: 1,
}
INT_TO_LABEL = {value: key for key, value in LABEL_TO_INT.items()}

DEFAULT_LABEL_REWARD_CORRECT = 1.0
DEFAULT_LABEL_REWARD_WRONG = -1.0
DEFAULT_LABEL_REWARD_MALFORMED = -1.25
