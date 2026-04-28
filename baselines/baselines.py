"""
baselines/baselines.py
======================
Baseline explainability methods for comparison with CTA.

All baselines share the same interface:
    model = BaselineClass(trajectory, categorical_vars)
    scores = model.compute()          # returns {variable: importance_score}

Methods implemented:
    SHAPWrapper           — KernelSHAP via shap library (falls back to manual)
    PermutationImportance — Classical shuffle-and-measure
    Ablation              — Best-value ablation per variable
    FANOVA                — Functional ANOVA variance decomposition
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _get_levels(trajectory, categorical_vars):
    """Collect all observed category levels per variable."""
    levels = {}
    for v in categorical_vars:
        seen = sorted(set(x.get(v) for x, _ in trajectory if v in x))
        levels[v] = seen
    return levels


def _build_design_matrix(trajectory, categorical_vars):
    """
    One-hot encode trajectory configurations.
    Returns (X, y) where y is the objective vector.
    """
    levels = _get_levels(trajectory, categorical_vars)
    rows, targets = [], []
    for x, f in trajectory:
        row = []
        for v in categorical_vars:
            for cat in levels[v]:
                row.append(1.0 if x.get(v) == cat else 0.0)
        rows.append(row)
        targets.append(f)
    return np.array(rows), np.array(targets), levels


def _col_map(categorical_vars, levels):
    """Map each variable to its one-hot column indices."""
    idx, col_map = 0, {}
    for v in categorical_vars:
        n = len(levels[v])
        col_map[v] = list(range(idx, idx + n))
        idx += n
    return col_map


# ─────────────────────────────────────────────────────────────────────────────
# SHAP Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class SHAPWrapper:
    """
    KernelSHAP applied to a Ridge surrogate fitted on trajectory data.

    Falls back to a manual Shapley implementation when the `shap` library
    is not installed (which is the default — no extra install required).

    Note: SHAP assumes i.i.d. features and ignores temporal structure,
    which is the key limitation CatExplain addresses.
    """

    def __init__(self, trajectory, categorical_vars):
        self.trajectory = trajectory
        self.categorical_vars = categorical_vars

    def compute(self):
        X, y, levels = _build_design_matrix(self.trajectory, self.categorical_vars)
        col_map = _col_map(self.categorical_vars, levels)

        try:
            import shap
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0).fit(X, y)
            explainer = shap.LinearExplainer(model, X)
            shap_vals = np.abs(explainer.shap_values(X)).mean(axis=0)
            return {v: float(shap_vals[col_map[v]].sum())
                    for v in self.categorical_vars}
        except ImportError:
            return self._manual(y, levels)

    def _manual(self, y, levels):
        """
        Manual variable-level Shapley approximation via marginal means.
        Used when the shap library is not installed.
        """
        global_mean = float(np.mean(y))
        var_means = {}
        for v in self.categorical_vars:
            vd = defaultdict(list)
            for x, f in self.trajectory:
                if v in x:
                    vd[x[v]].append(f)
            var_means[v] = {
                val: float(np.mean(vals)) - global_mean
                for val, vals in vd.items()
            }
        phi = {}
        for v in self.categorical_vars:
            contributions = [abs(var_means[v].get(x.get(v), 0.0))
                             for x, _ in self.trajectory]
            phi[v] = float(np.mean(contributions))
        return phi


# ─────────────────────────────────────────────────────────────────────────────
# Permutation Importance
# ─────────────────────────────────────────────────────────────────────────────

class PermutationImportance:
    """
    Classical permutation importance: shuffle each variable independently
    and measure the degradation in mean objective.

    Limitation: completely destroys the temporal structure of the trajectory,
    violating Axiom A4.
    """

    def __init__(self, trajectory, categorical_vars=None, n_repeats=30):
        self.trajectory = trajectory
        self.categorical_vars = categorical_vars or list(
            {k for x, _ in trajectory for k in x})
        self.n_repeats = n_repeats

    def compute(self):
        objs = np.array([f for _, f in self.trajectory])
        baseline = float(np.mean(objs))
        phi = {}
        rng = np.random.default_rng(42)

        for v in self.categorical_vars:
            vals = [x.get(v) for x, _ in self.trajectory]
            drops = []
            for _ in range(self.n_repeats):
                perm_vals = rng.permutation(vals)
                # Estimate performance degradation by perturbing each point
                # Use the objective values directly (surrogate-free)
                perm_objs = []
                for (x, f), new_val in zip(self.trajectory, perm_vals):
                    # Find closest trajectory point with this variable value
                    matching = [f2 for x2, f2 in self.trajectory
                                if x2.get(v) == new_val]
                    if matching:
                        perm_objs.append(float(np.mean(matching)))
                    else:
                        perm_objs.append(f)
                perm_mean = float(np.mean(perm_objs))
                drops.append(abs(perm_mean - baseline))
            phi[v] = float(np.mean(drops))
        return phi


# ─────────────────────────────────────────────────────────────────────────────
# Ablation
# ─────────────────────────────────────────────────────────────────────────────

class Ablation:
    """
    Ablation importance: compare the best objective with vs without each
    variable fixed to its best-observed value.

    Limitation: ignores interaction effects and has no axiom guarantees.
    """

    def __init__(self, trajectory, categorical_vars):
        self.trajectory = trajectory
        self.categorical_vars = categorical_vars

    def compute(self):
        # Best value per variable (minimisation)
        best_vals = {}
        for v in self.categorical_vars:
            vd = defaultdict(list)
            for x, f in self.trajectory:
                if v in x:
                    vd[x[v]].append(f)
            if vd:
                best_vals[v] = min(vd, key=lambda c: float(np.mean(vd[c])))

        best_f = min(f for _, f in self.trajectory)
        phi = {}
        for v in self.categorical_vars:
            best_without = float("inf")
            for x, f in self.trajectory:
                if x.get(v) != best_vals.get(v):
                    best_without = min(best_without, f)
            if best_without == float("inf"):
                best_without = best_f
            phi[v] = max(0.0, best_without - best_f)
        return phi


# ─────────────────────────────────────────────────────────────────────────────
# FANOVA
# ─────────────────────────────────────────────────────────────────────────────

class FANOVA:
    """
    Functional ANOVA: variance-based main-effect decomposition over
    trajectory data.

    Assigns importance as the between-group variance of the objective
    when partitioned by each variable's value.

    Limitation: assumes i.i.d. sampling (not satisfied by adaptive optimizers).
    """

    def __init__(self, trajectory, categorical_vars):
        self.trajectory = trajectory
        self.categorical_vars = categorical_vars

    def compute(self):
        global_mean = float(np.mean([f for _, f in self.trajectory]))
        phi = {}
        for v in self.categorical_vars:
            vd = defaultdict(list)
            for x, f in self.trajectory:
                if v in x:
                    vd[x[v]].append(f)
            if not vd:
                phi[v] = 0.0
                continue
            # Between-group variance (main effect)
            group_var = float(np.mean([
                (float(np.mean(vals)) - global_mean) ** 2
                for vals in vd.values()
            ]))
            phi[v] = group_var
        return phi
