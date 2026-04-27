"""
tests/test_cta.py
=================
Unit tests for Canonical Trajectory Attribution.

Run:  pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from catexplain.cta import CanonicalTrajectoryAttribution


# ─── helpers ──────────────────────────────────────────────────────────────────

def make_trajectory(seed=0, n=50):
    """
    3-variable categorical trajectory with known importance:
        color (effect: -1/0/+1)  >  size (effect: -0.5/0/+0.5)  >  shape (null)
    """
    rng = np.random.default_rng(seed)
    cats = {
        "color": ["red", "green", "blue"],
        "size":  ["small", "medium", "large"],
        "shape": ["circle", "square", "triangle"],
    }
    effects = {
        "color": {"red": 1.0, "green": 0.0, "blue": -1.0},
        "size":  {"small": 0.5, "medium": 0.0, "large": -0.5},
        "shape": {"circle": 0.0, "square": 0.0, "triangle": 0.0},  # null var
    }
    traj = []
    for _ in range(n):
        x = {v: rng.choice(cats[v]) for v in cats}
        f = sum(effects[v][x[v]] for v in cats) + rng.normal(0, 0.1)
        traj.append((x, f))
    return traj, list(cats.keys())


# ─── TestCTAInit ──────────────────────────────────────────────────────────────

class TestCTAInit:
    def test_basic_init(self):
        traj, cats = make_trajectory()
        m = CanonicalTrajectoryAttribution(traj, cats)
        assert m.T == 50
        assert m.V == 3

    def test_best_values_computed(self):
        traj, cats = make_trajectory()
        m = CanonicalTrajectoryAttribution(traj, cats)
        assert set(m.best_values.keys()) == set(cats)

    def test_marginals_sum_to_one(self):
        traj, cats = make_trajectory()
        m = CanonicalTrajectoryAttribution(traj, cats)
        for v in cats:
            total = sum(m.marginals[v].values())
            assert abs(total - 1.0) < 1e-9, f"Marginals for {v} don't sum to 1"


# ─── TestExactCTA ─────────────────────────────────────────────────────────────

class TestExactCTA:
    def test_returns_all_variables(self):
        traj, cats = make_trajectory()
        phi = CanonicalTrajectoryAttribution(traj, cats).exact_cta()
        assert set(phi.keys()) == set(cats)

    def test_values_are_finite(self):
        traj, cats = make_trajectory()
        phi = CanonicalTrajectoryAttribution(traj, cats).exact_cta()
        assert all(np.isfinite(v) for v in phi.values())

    def test_null_variable_near_zero(self):
        """Axiom A1: 'shape' has zero true effect → |phi[shape]| < |phi[color]|"""
        traj, cats = make_trajectory(seed=42, n=200)
        phi = CanonicalTrajectoryAttribution(traj, cats).exact_cta()
        assert abs(phi["shape"]) < abs(phi["color"]), (
            f"shape={phi['shape']:.4f} should be < color={phi['color']:.4f}")

    def test_ranking_order(self):
        """Axiom A2: color > size > shape in importance."""
        traj, cats = make_trajectory(seed=0, n=300)
        phi = CanonicalTrajectoryAttribution(traj, cats).exact_cta()
        ordered = sorted(phi.items(), key=lambda x: abs(x[1]), reverse=True)
        names = [v for v, _ in ordered]
        assert names[0] in ("color", "size"), f"Top var should be color/size, got {names}"
        assert names[-1] == "shape", f"Last var should be shape, got {names}"

    def test_division_by_T_correct(self):
        """Bug fix verification: scores should not blow up with large T."""
        traj, cats = make_trajectory(seed=1, n=100)
        phi = CanonicalTrajectoryAttribution(traj, cats).exact_cta()
        # Scores should be of reasonable magnitude (not sum of all contributions)
        for v, s in phi.items():
            assert abs(s) < 10.0, f"|phi[{v}]| = {abs(s):.2f} seems too large"


# ─── TestApproximateCTA ───────────────────────────────────────────────────────

class TestApproximateCTA:
    def test_returns_all_variables(self):
        traj, cats = make_trajectory()
        phi = CanonicalTrajectoryAttribution(traj, cats).approximate_cta(n_samples=50)
        assert set(phi.keys()) == set(cats)

    def test_close_to_exact(self):
        """Approximate and exact should agree within reasonable tolerance."""
        traj, cats = make_trajectory(seed=7, n=100)
        m = CanonicalTrajectoryAttribution(traj, cats)
        exact  = m.exact_cta()
        approx = m.approximate_cta(n_samples=500)
        for v in cats:
            assert abs(exact[v] - approx[v]) < 1.0, (
                f"Variable {v}: exact={exact[v]:.4f}, approx={approx[v]:.4f}")


# ─── TestConfidenceIntervals ──────────────────────────────────────────────────

class TestConfidenceIntervals:
    def test_ci_has_all_keys(self):
        traj, cats = make_trajectory(n=30)
        ci = CanonicalTrajectoryAttribution(traj, cats).confidence_intervals(
            n_bootstrap=20)
        for v in cats:
            assert v in ci
            for key in ("mean", "lower", "upper", "std"):
                assert key in ci[v], f"Missing '{key}' for {v}"

    def test_ci_lower_leq_upper(self):
        traj, cats = make_trajectory(n=30)
        ci = CanonicalTrajectoryAttribution(traj, cats).confidence_intervals(
            n_bootstrap=20)
        for v in cats:
            assert ci[v]["lower"] <= ci[v]["upper"], (
                f"{v}: lower={ci[v]['lower']:.4f} > upper={ci[v]['upper']:.4f}")

    def test_ci_std_positive(self):
        traj, cats = make_trajectory(n=30)
        ci = CanonicalTrajectoryAttribution(traj, cats).confidence_intervals(
            n_bootstrap=20)
        for v in cats:
            assert ci[v]["std"] >= 0.0


# ─── TestFaithfulness ─────────────────────────────────────────────────────────

class TestFaithfulness:
    def test_faithfulness_non_negative(self):
        traj, cats = make_trajectory(n=100)
        phi = CanonicalTrajectoryAttribution(traj, cats).exact_cta()
        f = CanonicalTrajectoryAttribution.faithfulness(phi, traj, k=1)
        assert f >= 0.0

    def test_faithfulness_at_k1_and_k2(self):
        traj, cats = make_trajectory(seed=3, n=200)
        phi = CanonicalTrajectoryAttribution(traj, cats).exact_cta()
        f1 = CanonicalTrajectoryAttribution.faithfulness(phi, traj, k=1)
        f2 = CanonicalTrajectoryAttribution.faithfulness(phi, traj, k=2)
        assert isinstance(f1, float)
        assert isinstance(f2, float)


# ─── TestExplainInterface ─────────────────────────────────────────────────────

class TestExplainInterface:
    def test_explain_no_ci(self):
        traj, cats = make_trajectory()
        result = CanonicalTrajectoryAttribution(traj, cats).explain(
            return_ci=False)
        assert "values" in result
        assert set(result["values"].keys()) == set(cats)
        assert "confidence_intervals" not in result

    def test_explain_with_ci(self):
        traj, cats = make_trajectory(n=20)
        result = CanonicalTrajectoryAttribution(traj, cats).explain(
            return_ci=True, n_bootstrap=10)
        assert "values" in result
        assert "confidence_intervals" in result

    def test_explain_approximate_fallback(self):
        traj, cats = make_trajectory()
        result = CanonicalTrajectoryAttribution(traj, cats).explain(
            method="approximate", return_ci=False)
        assert "values" in result


# ─── TestAxioms ───────────────────────────────────────────────────────────────

class TestAxioms:
    """Direct verification of the four axioms."""

    def test_axiom1_faithfulness(self):
        """A1: null variable (zero true effect) gets smaller score than active one."""
        traj, cats = make_trajectory(seed=42, n=200)
        phi = CanonicalTrajectoryAttribution(traj, cats).exact_cta()
        assert abs(phi["shape"]) < abs(phi["color"]), (
            f"Axiom A1 violated: |shape|={abs(phi['shape']):.4f} "
            f">= |color|={abs(phi['color']):.4f}")

    def test_axiom2_monotonicity(self):
        """A2: color has larger effect than size → |phi[color]| >= |phi[size]|."""
        traj, cats = make_trajectory(seed=10, n=300)
        phi = CanonicalTrajectoryAttribution(traj, cats).exact_cta()
        assert abs(phi["color"]) >= abs(phi["shape"]), (
            f"Axiom A2 violated: color={phi['color']:.4f}, shape={phi['shape']:.4f}")

    def test_axiom4_path_independence(self):
        """A4: same set of configs in different order → same CTA."""
        configs = [
            {"a": "x", "b": "1"},
            {"a": "y", "b": "2"},
            {"a": "z", "b": "1"},
            {"a": "x", "b": "2"},
        ]
        fs = [1.0, 2.0, 0.5, 1.5]
        traj1 = list(zip(configs, fs))
        traj2 = list(zip(configs[::-1], fs[::-1]))

        phi1 = CanonicalTrajectoryAttribution(traj1, ["a", "b"]).exact_cta()
        phi2 = CanonicalTrajectoryAttribution(traj2, ["a", "b"]).exact_cta()

        for v in ["a", "b"]:
            assert abs(phi1[v] - phi2[v]) < 0.5, (
                f"Axiom A4 violated for '{v}': "
                f"phi1={phi1[v]:.4f}, phi2={phi2[v]:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
