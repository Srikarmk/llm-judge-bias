"""Bootstrap statistics that resample items, never single calls.

Adapted from prompt-brittleness (itself from memory-rot and agent-interface-lab). The unit is the item:
both orders of a pair, and every condition of an item, share the item's question and answers, so an
item's rows move together. When tasks are pooled, items are resampled within each task so every draw
keeps each task's share.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def boot_means(values: np.ndarray, groups: np.ndarray | None = None, n_boot: int = C.BOOT,
               seed: int = 0) -> np.ndarray:
    """Bootstrap distribution of the mean of per-item values, stratified by `groups`."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.zeros(len(values), dtype=int)
    groups = np.asarray(groups)
    sums = np.zeros(n_boot)
    for g in np.unique(groups):
        v = values[groups == g]
        idx = rng.integers(0, len(v), size=(n_boot, len(v)))
        sums += v[idx].sum(axis=1)
    return sums / len(values)


def mean_ci(values, groups=None, n_boot: int = C.BOOT, seed: int = 0) -> dict:
    values = np.asarray(values, dtype=float)
    keep = ~np.isnan(values)
    values = values[keep]
    if groups is not None:
        groups = np.asarray(groups)[keep]
    if len(values) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    b = boot_means(values, groups, n_boot, seed)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"mean": float(values.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(values))}


def item_ci(df: pd.DataFrame, value: str, unit: str = "id", n_boot: int = C.BOOT, seed: int = 0) -> dict:
    """Mean of `value` over items (averaging an item's rows first), item bootstrap stratified by task."""
    if df.empty:
        return mean_ci([])
    per = df.groupby(["task", unit])[value].mean().reset_index()
    return mean_ci(per[value].values, per["task"].values, n_boot, seed)


def paired_ci(df: pd.DataFrame, value: str, ref: pd.DataFrame, ref_value: str | None = None, unit: str = "id",
              n_boot: int = C.BOOT, seed: int = 0) -> dict:
    """Mean over items of (item mean of `value` in df) minus (item mean in ref), paired by item."""
    if df.empty or ref.empty:
        return mean_ci([])
    a = df.groupby(["task", unit])[value].mean()
    b = ref.groupby(["task", unit])[ref_value or value].mean()
    m = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").reset_index()
    d = (m["a"] - m["b"]).values
    return mean_ci(d, m["task"].values, n_boot, seed)


def above_zero(r: dict) -> bool:
    return bool(r and r.get("n") and r["lo"] > 0)


def below_zero(r: dict) -> bool:
    return bool(r and r.get("n") and r["hi"] < 0)
