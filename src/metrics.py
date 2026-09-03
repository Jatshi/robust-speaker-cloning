"""Objective metrics and paired statistics for robust cloning evaluation."""
from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def mel_lsd_db(predicted_log_power: np.ndarray, clean_log_power: np.ndarray) -> float:
    """Mel-domain log-spectral distance in dB (not waveform/full-STFT LSD)."""
    predicted = np.asarray(predicted_log_power, dtype=np.float64)
    clean = np.asarray(clean_log_power, dtype=np.float64)
    if predicted.shape != clean.shape or predicted.ndim < 2:
        raise ValueError("predicted and clean log-mel arrays must have the same [..., mel, frame] shape")
    difference_db = (10.0 / math.log(10.0)) * (predicted - clean)
    return float(np.sqrt(np.mean(difference_db**2, axis=-2)).mean())


def _bootstrap_mean_ci(values: np.ndarray, seed: int = 42, repetitions: int = 5000) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    means = generator.choice(values, size=(repetitions, values.size), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def paired_summary(
    frame: pd.DataFrame,
    baseline_column: str = "baseline_xvector_similarity",
    robust_column: str = "robust_xvector_similarity",
    group_column: str = "degradation",
) -> dict:
    """Produce effect sizes, bootstrap CIs and per-group paired Wilcoxon tests."""
    required = {baseline_column, robust_column, group_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing evaluation columns: {sorted(missing)}")
    clean = frame.dropna(subset=list(required)).copy()
    clean["delta"] = clean[robust_column] - clean[baseline_column]
    rows: list[dict] = []
    groups = [("overall", clean), *((str(name), group) for name, group in clean.groupby(group_column, sort=True))]
    raw_p_values: list[float] = []
    for index, (name, group) in enumerate(groups):
        delta = group["delta"].to_numpy(dtype=float)
        low, high = _bootstrap_mean_ci(delta, seed=42 + index)
        if len(delta) >= 2 and np.any(np.abs(delta) > 0):
            p_value = float(wilcoxon(delta, zero_method="pratt", alternative="two-sided").pvalue)
        else:
            p_value = 1.0
        raw_p_values.append(p_value)
        rows.append({
            "group": name,
            "n": int(len(group)),
            "baseline_mean": float(group[baseline_column].mean()),
            "robust_mean": float(group[robust_column].mean()),
            "mean_delta": float(delta.mean()),
            "median_delta": float(np.median(delta)),
            "ci95_low": low,
            "ci95_high": high,
            "win_rate": float(np.mean(delta > 0)),
            "wilcoxon_p": p_value,
        })
    adjusted = _holm_adjust(raw_p_values[1:]) if len(raw_p_values) > 1 else []
    for row, value in zip(rows[1:], adjusted):
        row["holm_adjusted_p"] = value
    rows[0]["holm_adjusted_p"] = rows[0]["wilcoxon_p"]
    return {"protocol": "paired_two_sided_wilcoxon+percentile_bootstrap", "comparisons": rows}
