"""
Canonical Trajectory Attribution (CTA)
Bahig & Le Digabel (2026) — NeurIPS
"""
import numpy as np
import warnings
from collections import defaultdict


class CanonicalTrajectoryAttribution:
    def __init__(self, trajectory, categorical_vars, hierarchical=None):
        self.trajectory = trajectory
        self.categorical_vars = categorical_vars
        self.hierarchical = hierarchical or {}
        self.T = len(trajectory)
        self.V = len(categorical_vars)
        self._precompute_best_values()
        self._precompute_marginals()

    def _precompute_best_values(self):
        self.best_values = {}
        for var in self.categorical_vars:
            vd = defaultdict(list)
            for x, f in self.trajectory:
                if var in x:
                    vd[x[var]].append(f)
            if vd:
                self.best_values[var] = min(vd, key=lambda v: float(np.mean(vd[v])))

    def _precompute_marginals(self):
        self.marginals = {}
        for var in self.categorical_vars:
            counts = defaultdict(int)
            for x, _ in self.trajectory:
                if var in x:
                    counts[x[var]] += 1
            total = sum(counts.values())
            self.marginals[var] = {v: c/total for v, c in counts.items()} if total else {}

    def _estimate_f(self, x_base, fixed_vars, marginalized_vars, best_values):
        x_target = dict(x_base)
        for v in fixed_vars:
            if v in best_values:
                x_target[v] = best_values[v]
        if not marginalized_vars:
            return self._nearest_neighbor(x_target)
        ef, tw = 0.0, 0.0
        for x_t, f_t in self.trajectory:
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

    def _nearest_neighbor(self, x_target):
        best_d, best_f = float("inf"), self.trajectory[0][1]
        for x_t, f_t in self.trajectory:
            d = sum(
                (v1 - v2)**2 if isinstance(v1, (int, float)) and isinstance(v2, (int, float))
                else (0 if v1 == v2 else 1)
                for k in set(x_target) | set(x_t)
                for v1, v2 in [(x_target.get(k), x_t.get(k))]
            )
            if d < best_d:
                best_d, best_f = d, f_t
        return best_f

    def exact_cta(self):
        """Exact CTA — O(T · 2^|V|). Use for |V| <= 15."""
        if self.V > 15:
            warnings.warn(f"|V|={self.V} > 15. Consider approximate_cta().")
        phi = {v: 0.0 for v in self.categorical_vars}
        weight = 1.0 / (2 ** (self.V - 1))
        for x_t, _ in self.trajectory:
            for v in self.categorical_vars:
                others = [w for w in self.categorical_vars if w != v]
                for mask in range(2 ** len(others)):
                    S = {others[i] for i in range(len(others)) if (mask >> i) & 1}
                    fw = self._estimate_f(x_t, S,       {v},   self.best_values)
                    fv = self._estimate_f(x_t, S | {v}, set(), self.best_values)
                    phi[v] += weight * (fw - fv)
        # FIX: divide by T once, outside the loop
        return {v: p / self.T for v, p in phi.items()}

    def approximate_cta(self, n_samples=1000):
        """Monte Carlo Shapley — O(T · |V| · n). For |V| > 15."""
        phi = {v: 0.0 for v in self.categorical_vars}
        for x_t, _ in self.trajectory:
            for v in self.categorical_vars:
                others = [w for w in self.categorical_vars if w != v]
                samples = []
                for _ in range(n_samples):
                    perm = list(np.random.permutation(others))
                    S = set(perm)  # full coalition minus v
                    fw = self._estimate_f(x_t, S,       {v},   self.best_values)
                    fv = self._estimate_f(x_t, S | {v}, set(), self.best_values)
                    samples.append(fw - fv)
                phi[v] += float(np.mean(samples))
        return {v: p / self.T for v, p in phi.items()}

    def confidence_intervals(self, method="exact", n_bootstrap=200, alpha=0.05):
        boot = []
        for _ in range(n_bootstrap):
            idx  = np.random.choice(self.T, self.T, replace=True)
            traj = [self.trajectory[i] for i in idx]
            m    = CanonicalTrajectoryAttribution(traj, self.categorical_vars)
            boot.append(m.exact_cta() if method == "exact" else m.approximate_cta())
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

    @staticmethod
    def faithfulness(cta_values, trajectory, k=2):
        ranked = sorted(cta_values.items(), key=lambda x: -x[1])
        top    = [v for v, _ in ranked[:k]]
        bottom = [v for v, _ in ranked[-k:]]
        best_per = {}
        for v in top + bottom:
            vd = defaultdict(list)
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
        return (best_without(top) - best_f) / (best_without(bottom) - best_f + 1e-8)

    def explain(self, method="exact", return_ci=True, n_bootstrap=200):
        cta = (self.exact_cta() if method == "exact" and self.V <= 15
               else self.approximate_cta())
        result = {"values": cta}
        if return_ci:
            result["confidence_intervals"] = self.confidence_intervals(
                method=method, n_bootstrap=n_bootstrap)
        return result
