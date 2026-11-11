"""Preserve historical operators and add genuinely uniform-original random.

Historical uniform_random is quota-balanced random, not equal probability over
originals at each draw. The added baseline makes those meanings explicit.
"""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'baseline'))
from acquisition import acquire as legacy_acquire


def acquire(strategy, scores, rows, selected, budget, selection_seed):
    if strategy != 'original_uniform_random':
        return legacy_acquire(strategy, scores, rows, selected, budget, selection_seed)
    if not isinstance(budget, int) or not len(selected) <= budget <= len(rows):
        raise ValueError('Invalid cumulative budget')
    if len(set(selected)) != len(selected):
        raise ValueError('Duplicate prefix')
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(row['group'], []).append(index)
    names = sorted(groups)
    rng = np.random.default_rng(selection_seed)
    # Replaying the fixed draw stream makes all budgets cumulative. Draw a
    # remaining original uniformly, then a remaining patch within it uniformly.
    out = []
    while len(out) < budget:
        name_index = int(rng.integers(len(names)))
        name = names[name_index]
        patch_index = int(rng.integers(len(groups[name])))
        out.append(groups[name].pop(patch_index))
        if not groups[name]:
            names.pop(name_index)
    if out[:len(selected)] != list(selected):
        raise ValueError('Prefix does not match frozen uniform-original stream')
    return out
