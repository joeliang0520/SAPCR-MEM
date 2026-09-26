"""Turn accumulated label votes into a downstream-facing label decision."""

from __future__ import annotations

import math


def label_with_candidates(
    label_scores: dict[str, float],
    *,
    max_aliases: int = 2,
    min_alias_score: float = 2.0,
    min_alias_fraction: float = 0.10,
) -> tuple[str, list[str], list[dict[str, str | float]]]:
    """Rank accumulated evidence and expose scored runner-up labels.

    Each observation contributes 1.0 for its primary label and may contribute
    smaller relative scores for alternatives. ``label_candidates`` contains
    every runner-up with positive evidence; ``label_aliases`` is the compact,
    credible subset retained for readers that only understand strings.
    """
    if not label_scores:
        raise ValueError("An object needs label evidence")
    if max_aliases < 0:
        raise ValueError("max_aliases cannot be negative")
    if not math.isfinite(min_alias_score) or min_alias_score < 0:
        raise ValueError("min_alias_score must be finite and non-negative")
    if not math.isfinite(min_alias_fraction) or not 0 <= min_alias_fraction <= 1:
        raise ValueError("min_alias_fraction must be finite and in [0, 1]")
    for label, score in label_scores.items():
        if not isinstance(label, str) or not label:
            raise ValueError("Labels must be non-empty strings")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("Label scores must be numbers")
        if not math.isfinite(score) or score < 0:
            raise ValueError("Label scores must be finite and non-negative")
    if not any(score > 0 for score in label_scores.values()):
        raise ValueError("At least one label score must be positive")

    winner = max(label_scores, key=label_scores.get)
    winning_score = label_scores[winner]
    ranked = sorted(
        (
            (position, label, float(score))
            for position, (label, score) in enumerate(label_scores.items())
            if label != winner and score > 0
        ),
        key=lambda item: (-item[2], item[0]),
    )
    minimum = max(min_alias_score, winning_score * min_alias_fraction)
    aliases = [
        label for _, label, score in ranked if score >= minimum
    ][:max_aliases]
    total = float(sum(label_scores.values()))
    candidates = [
        {"label": label, "support": score / total}
        for _, label, score in ranked
    ]
    return winner, aliases, candidates
