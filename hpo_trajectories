"""
catexplain/cta.py
=================
Canonical Trajectory Attribution (CTA) — unique explanation satisfying
Faithfulness, Monotonicity, Non-interference, and Path-independence.

Reference:
    Bahig & Le Digabel (2026). CatExplain: A Unifying Axiomatic Theory
    of Explainable Black-Box Optimization. NeurIPS.
"""
import numpy as np
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class CanonicalTrajectoryAttribution:
    """
    Canonical Trajectory Attribution (CTA).

    Parameters
    ----------
    trajectory : list of (dict, float)
        Each element is (configuration_dict, objective_value).
        Minimization is assumed — lower objective = better.
    categorical_vars : list of str
        Names of the categorical variables to explain.
    hierarchical : dict, optional
        Maps child variable → parent variable for hierarchical spaces.
        E.g. {'learning_rate': 'optimizer'} means learning_rate is only
        active when optimizer is active.

    Examples
    --------
    >>> traj = [
    ...     ({"arch": "resnet", "loss": "focal"}, 0.12),
    ...     ({"arch": "vgg",    "loss": "ce"},    0.18),
    ... ]
    >>> model = CanonicalTrajectoryAttribution(traj, ["arch", "loss"])
    >>> phi = model.exact_cta()
    """

    def __init__(
        self,
        trajectory: List[Tuple[Dict, float]],
        categorical_vars: List[str],
        hierarchical: Optional[Dict[str, str]] = None,
    ):
        self.trajectory       = trajectory
        self.categorical_vars = categorical_vars
        self.hierarchical     = hierarchical or {}
        self.T = len(trajectory)
        self.V = len(categorical_vars)
        self._precompute_best_values()
        self._precompute_marginals()

    # ── private helpers ───────────────────────────────────────────────────────

    def _precompute_best_values(self):
        """Best observed category per variable (lowest mean objective)."""
        self.best_values: Dict = {}
        for var in self.categorical_vars:
            vd: Dict = defaultdict(list)
            for x, f in self.trajectory:
                if var in x:
                    vd[x[var]].append(f)
            if vd:
                self.best_values[var] = min(
                    vd, key=lambda v: float(np.mean(vd[v])))

    def _precompute_marginals(self):
        """Empirical marginal distributions over visited configurations."""
        self.marginals: Dict = {}
        for var in self.categorical_vars:
            counts: Dict = defaultdict(int)
            for x, _ in self.trajectory:
                if var in x:
                    counts[x[var]] += 1
            total = sum(counts.values())
            self.marginals[var] = (
                {val: c / total for val, c in counts.items()} if total else {})

    def _estimate_f(
        self,
        x_base: Dict,
        fixed_vars: set,
        marginalized_vars: set,
        best_values: Dict,
    ) -> float:
        """
        Estimate f at a configuration where:
          - variables in fixed_vars are pinned to best_values[v]
          - variables in marginalized_vars are integrated out via P_traj
          - others keep their value from x_base
        """
        x_target = dict(x_base)
        for v in fixed_vars:
            if v in best_values:
                x_target[v] = best_values[v]

        if not marginalized_vars:
            return self._nearest_neighbor(x_target)

        ef, tw = 0.0, 0.0
        for x_t, f_t in self.trajectory:
            # Check agreement on all non-marginalized variables
            match = all(
                x_t.get(var) == x_target.get(var)
                for var in set(x_t) & set(x_target)
                if var not in marginalized_vars
            )
            if not match:
                continue
            w = 1.0
            for var in marginalized_vars:
                val = x_t.get(var)
                if val is not None:
                    w *= self.marginals.get(var, {}).get(val, 0.0)
            ef += w * f_t
            tw += w

        return ef / tw if tw > 1e-12 else self._nearest_neighbor(x_target)

    def _nearest_neighbor(self, x_target: Dict) -> float:
        """Objective of nearest configuration in trajectory (Hamming dist)."""
        best_d, best_f = float("inf"), self.trajectory[0][1]
        for x_t, f_t in self.trajectory:
            d = 0.0
            all_keys = set(x_target) | set(x_t)
            for k in all_keys:
                v1, v2 = x_target.get(k), x_t.get(k)
                if v1 is None or v2 is None:
                    d += 1.0
                elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    d += (v1 - v2) ** 2
                elif v1 != v2:
                    d += 1.0
            if d < best_d:
                best_d, best_f = d, f_t
        return best_f

    # ── public API ────────────────────────────────────────────────────────────

    def exact_cta(self) -> Dict[str, float]:
        """
        Exact CTA computation — O(T · 2^|V|) time.

        Recommended for |V| ≤ 15. For larger spaces, use approximate_cta().

        Returns
        -------
        dict
            {variable_name: importance_score}
            Negative values indicate the variable hinders optimization.
        """
        if self.V > 15:
            warnings.warn(
                f"|V|={self.V} > 15. Exact computation may be slow. "
                "Consider approximate_cta() instead.")

        phi    = {v: 0.0 for v in self.categorical_vars}
        weight = 1.0 / (2 ** (self.V - 1))  # uniform Shapley weight

        for x_t, _ in self.trajectory:
            for v in self.categorical_vars:
                others  = [w for w in self.categorical_vars if w != v]
                V_other = len(others)
                for mask in range(2 ** V_other):
                    S = {others[i] for i in range(V_other) if (mask >> i) & 1}
                    f_without = self._estimate_f(x_t, S,       {v},   self.best_values)
                    f_with    = self._estimate_f(x_t, S | {v}, set(), self.best_values)
                    phi[v]   += weight * (f_without - f_with)

        # FIX vs original paper code: divide by T ONCE outside the loop
        return {v: p / self.T for v, p in phi.items()}

    def approximate_cta(self, n_samples: int = 1000) -> Dict[str, float]:
        """
        Monte Carlo Shapley approximation — O(T · |V| · n_samples) time.

        Recommended for |V| > 15. For most benchmarks, n_samples=500 is
        sufficient for stable estimates.

        Parameters
        ----------
        n_samples : int
            Number of random permutations per (t, v) pair.
        """
        phi = {v: 0.0 for v in self.categorical_vars}

        for x_t, _ in self.trajectory:
            for v in self.categorical_vars:
                others  = [w for w in self.categorical_vars if w != v]
                samples = []
                for _ in range(n_samples):
                    # Use the full coalition (correct marginal contribution)
                    S         = set(others)
                    f_without = self._estimate_f(x_t, S,       {v},   self.best_values)
                    f_with    = self._estimate_f(x_t, S | {v}, set(), self.best_values)
                    samples.append(f_without - f_with)
                phi[v] += float(np.mean(samples))

        return {v: p / self.T for v, p in phi.items()}

    def confidence_intervals(
        self,
        method:      str   = "exact",
        n_bootstrap: int   = 200,
        alpha:       float = 0.05,
    ) -> Dict[str, Dict]:
        """
        Block bootstrap confidence intervals for CTA estimates.

        Parameters
        ----------
        method : 'exact' or 'approximate'
        n_bootstrap : int
            Number of bootstrap resamples.
        alpha : float
            Significance level. Default 0.05 → 95% CI.
        """
        boot = []
        for _ in range(n_bootstrap):
            idx  = np.random.choice(self.T, self.T, replace=True)
            traj = [self.trajectory[i] for i in idx]
            m    = CanonicalTrajectoryAttribution(
                traj, self.categorical_vars, self.hierarchical)
            boot.append(
                m.exact_cta() if method == "exact" else m.approximate_cta())

        lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
        return {
            v: {
                "mean":  float(np.mean([b[v] for b in boot])),
                "lower": float(np.percentile([b[v] for b in boot], lo)),
                "upper": float(np.percentile([b[v] for b in boot], hi)),
                "std":   float(np.std([b[v] for b in boot])),
            }
            for v in self.categorical_vars
        }

    def estimate_asymptotic_variance(
        self,
        variable: str,
        method:   str = "batch_means",
    ) -> float:
        """
        Estimate σ²_v, the asymptotic variance used in Theorem 3.

        Parameters
        ----------
        variable : str
        method : 'batch_means' | 'spectral' | 'bootstrap_block'
        """
        Y = self._Y_sequence(variable)
        T = len(Y)

        if method == "batch_means":
            J  = max(2, int(T ** (1 / 3)))
            B  = T // J
            bm = [float(np.mean(Y[i*B:(i+1)*B])) for i in range(J)]
            return float(B * np.var(bm, ddof=1))

        elif method == "spectral":
            K  = max(1, int(T ** (1 / 3)))
            Yc = Y - np.mean(Y)
            s  = float(np.var(Yc))
            for k in range(1, K + 1):
                s += 2 * (1 - k / K) * float(np.mean(Yc[:-k] * Yc[k:]))
            return max(s, 0.0)

        raise ValueError(f"Unknown method '{method}'. "
                         "Choose 'batch_means' or 'spectral'.")

    def _Y_sequence(self, variable: str) -> np.ndarray:
        """Centred contribution sequence Y_t^(v) for asymptotic variance."""
        w      = 1.0 / (2 ** (self.V - 1))
        others = [v for v in self.categorical_vars if v != variable]
        Y      = np.zeros(self.T)
        for t, (x_t, _) in enumerate(self.trajectory):
            val = 0.0
            for mask in range(2 ** len(others)):
                S  = {others[i] for i in range(len(others)) if (mask >> i) & 1}
                fw = self._estimate_f(x_t, S,              {variable}, self.best_values)
                fv = self._estimate_f(x_t, S | {variable}, set(),      self.best_values)
                val += w * (fw - fv)
            Y[t] = val
        return Y - np.mean(Y)

    @staticmethod
    def faithfulness(
        cta_values: Dict[str, float],
        trajectory: List[Tuple[Dict, float]],
        k: int = 2,
    ) -> float:
        """
        Faithfulness@k metric.

        Measures the ratio of performance drop when the top-k vs bottom-k
        variables are removed from the best configuration found.
        Higher is better (≥1 means top-k variables drive more improvement).

        Parameters
        ----------
        cta_values : dict of {variable: score}
        trajectory : same format as CanonicalTrajectoryAttribution.trajectory
        k : int
        """
        ranked      = sorted(cta_values.items(), key=lambda x: -x[1])
        top_vars    = [v for v, _ in ranked[:k]]
        bottom_vars = [v for v, _ in ranked[-k:]]

        best_per: Dict = {}
        for v in top_vars + bottom_vars:
            vd: Dict = defaultdict(list)
            for x, f in trajectory:
                if v in x:
                    vd[x[v]].append(f)
            if vd:
                best_per[v] = min(vd, key=lambda c: float(np.mean(vd[c])))

        best_f = min(f for _, f in trajectory)

        def best_without(avoid):
            w = float("inf")
            for x, f in trajectory:
                if any(x.get(v) == best_per.get(v) for v in avoid):
                    continue
                w = min(w, f)
            return w if w < float("inf") else best_f

        drop_top    = best_without(top_vars)    - best_f
        drop_bottom = best_without(bottom_vars) - best_f
        return drop_top / (drop_bottom + 1e-8)

    def explain(
        self,
        method:      str  = "exact",
        return_ci:   bool = True,
        n_bootstrap: int  = 200,
    ) -> Dict:
        """
        Main interface: compute CTA with optional confidence intervals.

        Parameters
        ----------
        method : 'exact' (|V| ≤ 15) or 'approximate' (|V| > 15)
        return_ci : bool
            Whether to compute bootstrap confidence intervals.
        n_bootstrap : int
            Number of bootstrap resamples for CIs.

        Returns
        -------
        dict with keys:
            'values'               → {variable: score}
            'confidence_intervals' → {variable: {mean, lower, upper, std}}
        """
        if method == "exact" and self.V <= 15:
            cta = self.exact_cta()
        else:
            cta = self.approximate_cta()

        result = {"values": cta}
        if return_ci:
            result["confidence_intervals"] = self.confidence_intervals(
                method=method, n_bootstrap=n_bootstrap)
        return result
