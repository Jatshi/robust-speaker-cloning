"""One deterministic speaker split shared by training, selection and final test."""
from __future__ import annotations

import random


def split_speakers(
    speakers: list[str], seed: int = 42, train_ratio: float = 0.85, selection_ratio: float = 0.05
) -> dict[str, set[str]]:
    ordered = sorted(speakers); random.Random(seed).shuffle(ordered)
    train_end = max(1, int(len(ordered) * train_ratio))
    selection_end = max(train_end + 1, int(len(ordered) * (train_ratio + selection_ratio)))
    if selection_end >= len(ordered):
        selection_end = len(ordered) - 1
    result = {
        "train": set(ordered[:train_end]),
        "selection": set(ordered[train_end:selection_end]),
        "test": set(ordered[selection_end:]),
    }
    if any(result[left] & result[right] for left, right in (("train", "selection"), ("train", "test"), ("selection", "test"))):
        raise AssertionError("speaker split leakage")
    return result
