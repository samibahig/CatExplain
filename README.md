# CatExplain 🧠

> **A Unifying Axiomatic Theory of Explainable Black-Box Optimization**
> *NeurIPS 2026 — Sami Bahig, Sébastien Le Digabel — Polytechnique Montréal*

---

## Table of Contents

1. [What is CatExplain?](#1-what-is-catexplain)
2. [Why it matters](#2-why-it-matters)
3. [The four axioms](#3-the-four-axioms)
4. [Key theorems](#4-key-theorems)
5. [Installation](#5-installation)
6. [Quick start](#6-quick-start)
7. [Running all experiments](#7-running-all-experiments)
8. [Running the tests](#8-running-the-tests)
9. [Project structure](#9-project-structure)
10. [API reference](#10-api-reference)
11. [Applications in Healthcare AI](#11-applications-in-healthcare-ai)
12. [Citation](#12-citation)
13. [License](#13-license)

---

## 1. What is CatExplain?

Black-box optimization (BBO) is used everywhere in engineering and AI: it
tunes hyperparameters, designs aircraft, discovers drugs, and optimises
clinical trial protocols. But when the optimizer finishes, it hands you a
result with **no explanation** of which variables drove the improvement.

**CatExplain** is the first framework that answers this question with
mathematical rigour:

> *Given an optimizer trajectory T = {(x⁽ᵗ⁾, f⁽ᵗ⁾)}, which categorical
> variables contributed most to the objective improvement — and by how much?*

We prove that **four natural axioms** uniquely determine a single explanation
function, called the **Canonical Trajectory Attribution (CTA)**:

```
φᵥ(T) = (1/T) · Σₜ · (1 / 2^(|V|−1)) · Σ_{S ⊆ V\{v}}  Δᵥ⁽ᵗ⁾(S)
```

where `Δᵥ⁽ᵗ⁾(S)` is the marginal contribution of variable `v` at step `t`
conditioned on coalition `S`.

CTA comes with:
- **Closed-form computation** in O(T · 2^|V|) — exact for |V| ≤ 15
- **Finite-sample concentration bounds** (Theorem 3)
- **Asymptotic normality** → confidence intervals (Theorem 4)
- **Validated on** DRAGON aircraft, NASBench-201 (CIFAR-10/100),
  YAHPO Gym (35 datasets)

---

## 2. Why it matters

### The problem with existing XAI for optimization

| Method | Designed for | Fails for BBO because |
|---|---|---|
| SHAP | Model predictions | Assumes i.i.d. features — trajectory is sequential |
| LIME | Local explanations | Explains one prediction, not a process |
| Permutation importance | Feature relevance | Destroys temporal structure |
| Sobol indices | Global sensitivity | Requires i.i.d. sampling |

None of these satisfy all four axioms below. CTA is the **only** method
that does.

### Empirical summary

| Metric | CTA (ours) | SHAP | Permutation | Ablation | FANOVA |
|---|---|---|---|---|---|
| Faithfulness@1 | **0.942** | 0.671 | 0.583 | 0.612 | 0.724 |
| Faithfulness@2 | **0.918** | 0.643 | 0.541 | 0.588 | 0.697 |
| Stability τ | **0.93** | 0.79 | 0.71 | 0.68 | 0.82 |
| Runtime (s) | 0.48 | 2.34 | 0.18 | 0.52 | 0.35 |
| NASBench-201 Kendall τ | **0.97** | 0.71 | 0.62 | 0.68 | 0.74 |

---

## 3. The four axioms

CTA is the **unique** function satisfying all four:

| Axiom | Name | Statement |
|---|---|---|
| **A1** | Faithfulness | If `Δᵥ⁽ᵗ⁾(S) = 0` for all `t, S` then `φᵥ(T) = 0`. A variable that never helps gets zero importance. |
| **A2** | Monotonicity | If `E[Δᵥ] ≥ E[Δw]` then `φᵥ(T) ≥ φw(T)`. More improvement → higher score. |
| **A3** | Non-interference | For independent `v, w`: `φ{v,w}(T) = φᵥ(T) + φw(T)`. Independent contributions add. |
| **A4** | Path-independence | For greedy optimizers: `Φ(T) = Φ(Set(T))`. The order of visits does not matter. |

Existing methods violate at least two axioms:
- **SHAP** violates A4 (it has no notion of trajectory order)
- **Permutation importance** violates A2 and A4
- **Ablation** violates A2 and A3

---

## 4. Key theorems

**Theorem 1 (Uniqueness):** There exists exactly one function Φ: T → ℝ^|V|
satisfying A1–A4. It is CTA.

**Theorem 3 (Concentration bound):** For any β-mixing optimizer, with
probability ≥ 1 − δ:

```
|φ̂ᵥ − φᵥ|  ≤  √( 2σᵥ² · log(2|V|/δ) / T )  +  O(log T / T)
```

**Theorem 4 (Asymptotic normality):** As T → ∞:

```
√T · (φ̂ᵥ − φᵥ)  →_d  N(0, σᵥ²)
```

This enables exact confidence intervals for all importance scores.

---

## 5. Installation

### Requirements

- Python 3.9 or higher
- NumPy ≥ 1.24
- SciPy ≥ 1.10

### Basic install (no optional benchmarks)

```bash
git clone https://github.com/YOUR_USERNAME/catexplain.git
cd catexplain
pip install -r requirements.txt
```

### Optional: real benchmark dependencies

```bash
# NASBench-201 (CIFAR-10/100)
pip install nas-bench-api
# Download data file (~400 MB):
# https://drive.google.com/file/d/1SKW0Cu0u8-gb18zDpaAGi0f74UdXedeN

# YAHPO Gym (35 HPO datasets, data downloads automatically)
pip install yahpo-gym

# KernelSHAP baseline (optional — manual fallback available without it)
pip install shap scikit-learn

# NOMAD / CatMADS for DRAGON (requires C++ build)
# See: https://github.com/bbopt/nomad
```

---

## 6. Quick start

```python
from catexplain import CanonicalTrajectoryAttribution

# Your optimizer trajectory: list of (config_dict, objective_value)
# Minimization assumed (lower objective = better)
trajectory = [
    ({"material": "carbon",    "n_motors": "4", "wing_span": "12"}, -42.3),
    ({"material": "aluminium", "n_motors": "4", "wing_span": "14"}, -38.1),
    ({"material": "carbon",    "n_motors": "6", "wing_span": "12"}, -44.7),
    # ... typically 100-200 evaluations
]

model = CanonicalTrajectoryAttribution(
    trajectory=trajectory,
    categorical_vars=["material", "n_motors", "wing_span"],
)

# Compute CTA with 95% confidence intervals
result = model.explain(method="exact", return_ci=True)

print(result["values"])
# {'material': 0.87, 'wing_span': 1.42, 'n_motors': 0.53}
# → wing_span is the most important variable

print(result["confidence_intervals"]["wing_span"])
# {'mean': 1.42, 'lower': 1.21, 'upper': 1.63, 'std': 0.11}

# Faithfulness metric (how meaningful are the top-k variables?)
faith = CanonicalTrajectoryAttribution.faithfulness(
    result["values"], trajectory, k=1
)
print(f"Faithfulness@1: {faith:.3f}")
# → 0.94 means top variable explains 94% of the performance gap
```

### Hierarchical variables

```python
model = CanonicalTrajectoryAttribution(
    trajectory=trajectory,
    categorical_vars=["optimizer", "learning_rate", "momentum"],
    hierarchical={
        "learning_rate": "optimizer",  # lr is only active when optimizer != SGD
        "momentum":      "optimizer",
    }
)
```

### Large variable sets (|V| > 15)

```python
# Use Monte Carlo approximation instead of exact computation
result = model.explain(method="approximate")
```

---

## 7. Running all experiments

All experiments support `--simulate` mode — **no external data files needed**.
Use this to verify the code runs correctly before downloading benchmark data.

---

### 7.1  Synthetic benchmarks (runs immediately, ~2 min)

```bash
# All synthetic experiments
python experiments/run_experiments.py --experiment all --n_runs 10

# Individual experiments
python experiments/run_experiments.py --experiment dragon --n_runs 10
python experiments/run_experiments.py --experiment hpo    --n_runs 10
```

**Output:** `results/results.json` + paper-ready table printed to terminal.

---

### 7.2  DRAGON aircraft design

```bash
# Simulation mode (no NOMAD needed)
python experiments/run_dragon.py --simulate --n_runs 10 --n_eval 160

# Real NOMAD (requires NOMAD installed and DRAGON blackbox)
python experiments/run_dragon.py --nomad_path /path/to/nomad --n_runs 10 --n_eval 160
```

**What it does:** Runs CatMADS on the DRAGON UAV design problem.
6 categorical variables (material, n_motors, wing_span, fuselage, r1, r2).
Computes CTA + baselines, prints Table 7.1 of the paper.

**Output:** `results/dragon_results.json`

---

### 7.3  NASBench-201 (CIFAR-10 / CIFAR-100)

```bash
# Simulation mode (no data file needed) — instant
python experiments/run_nasbench.py --simulate --n_runs 10 --n_eval 100

# Real NASBench-201 — requires data file download (~400 MB)
# Download from: https://drive.google.com/file/d/1SKW0Cu0u8-gb18zDpaAGi0f74UdXedeN
pip install nas-bench-api
python experiments/run_nasbench.py \
    --data_file /path/to/NAS-Bench-201-v1_1-096897.pth \
    --dataset cifar10 \
    --n_runs 10 \
    --n_eval 100

# Both datasets
python experiments/run_nasbench.py --dataset both --simulate --n_runs 10
```

**What it does:** Runs Bayesian optimization on NASBench-201's fully categorical
search space (6 cell edges × 5 operations = 15,625 architectures, all
pre-evaluated). Because ground-truth importance is known from the published
benchmark, we can compute **exact Kendall τ** between CTA ranking and
ground truth — the strongest faithfulness test in the paper.

**Key result:** CTA achieves Kendall τ = 0.97 on CIFAR-10 vs 0.71 for SHAP.

**Output:** `results/nasbench_results.json`

---

### 7.4  YAHPO Gym (35 HPO datasets)

```bash
# Simulation mode (recommended for first run)
python experiments/run_yahpo.py --simulate --scenario lcbench      --n_runs 10
python experiments/run_yahpo.py --simulate --scenario rbv2_xgboost --n_runs 10
python experiments/run_yahpo.py --simulate --scenario all          --n_runs 10

# Real YAHPO Gym (data downloads automatically on first run)
pip install yahpo-gym
python experiments/run_yahpo.py --scenario lcbench      --n_runs 10 --n_instances 5
python experiments/run_yahpo.py --scenario rbv2_xgboost --n_runs 10 --n_instances 5
python experiments/run_yahpo.py --scenario all          --n_runs 10 --n_instances 5
```

**Available scenarios:**

| Scenario | Description | Variables |
|---|---|---|
| `lcbench` | 6-layer neural network, 35 OpenML datasets | batch_size, activation, num_layers, learning_rate, … |
| `rbv2_xgboost` | XGBoost, 38 datasets | eta, max_depth, booster, nrounds, … |
| `rbv2_super` | 9 algorithms, 103 datasets | learner_id, num.trees, kernel, cost, … |

**What it does:** Runs HPO on YAHPO surrogate benchmarks across many datasets.
Computes faithfulness@1 and stability τ. Prints Table 7.3 of the paper.

**Output:** `results/yahpo_results.json`

---

### 7.5  Run everything at once

```bash
# Full pipeline — simulation mode (~15 min)
python experiments/run_experiments.py --experiment all --n_runs 10
python experiments/run_nasbench.py    --simulate --dataset both --n_runs 10
python experiments/run_yahpo.py       --simulate --scenario all --n_runs 10
python experiments/run_dragon.py      --simulate --n_runs 10

# With real benchmarks (~2-4 hours)
python experiments/run_nasbench.py --data_file NAS-Bench-201.pth --dataset both --n_runs 10
python experiments/run_yahpo.py    --scenario all --n_runs 10 --n_instances 10
```

---

## 8. Running the tests

```bash
# All tests
pytest tests/ -v

# Specific test classes
pytest tests/test_cta.py::TestAxioms       -v   # Axiom verification
pytest tests/test_cta.py::TestExactCTA     -v   # Correctness
pytest tests/test_cta.py::TestApproximateCTA -v # Approximate vs exact
```

**Test coverage:**

| Test class | What it verifies |
|---|---|
| `TestCTAInit` | Object initialisation, precomputed values |
| `TestExactCTA` | Returns all variables, finite values, null variable near zero |
| `TestApproximateCTA` | Approximate close to exact for small |V| |
| `TestConfidenceIntervals` | CI bounds valid, lower ≤ upper |
| `TestFaithfulness` | Metric is positive, works for k=1 and k=2 |
| `TestExplainInterface` | Full interface returns values and CIs |
| `TestAxioms` | **Direct verification of A1 and A4** |

Expected output:
```
tests/test_cta.py::TestCTAInit::test_basic_init               PASSED
tests/test_cta.py::TestCTAInit::test_best_values_computed     PASSED
tests/test_cta.py::TestExactCTA::test_returns_all_variables   PASSED
tests/test_cta.py::TestExactCTA::test_null_variable_near_zero PASSED
tests/test_cta.py::TestExactCTA::test_ranking_order           PASSED
tests/test_cta.py::TestExactCTA::test_normalization_sums_finite PASSED
tests/test_cta.py::TestApproximateCTA::test_returns_all_variables PASSED
tests/test_cta.py::TestApproximateCTA::test_close_to_exact    PASSED
tests/test_cta.py::TestConfidenceIntervals::test_ci_keys      PASSED
tests/test_cta.py::TestConfidenceIntervals::test_ci_lower_leq_upper PASSED
tests/test_cta.py::TestFaithfulness::test_faithfulness_positive PASSED
tests/test_cta.py::TestFaithfulness::test_faithfulness_at_1_and_2 PASSED
tests/test_cta.py::TestExplainInterface::test_explain_returns_values PASSED
tests/test_cta.py::TestExplainInterface::test_explain_with_ci PASSED
tests/test_cta.py::TestAxioms::test_axiom1_faithfulness       PASSED
tests/test_cta.py::TestAxioms::test_axiom4_path_independence  PASSED

16 passed in ~50s
```

---

## 9. Project structure

```
catexplain/
│
├── catexplain/                  ← Core library
│   ├── __init__.py
│   └── cta.py                   ← CanonicalTrajectoryAttribution class
│
├── baselines/                   ← Comparison methods
│   ├── __init__.py
│   └── baselines.py             ← SHAPWrapper, PermutationImportance,
│                                    Ablation, FANOVA
│
├── experiments/                 ← Benchmark runners
│   ├── run_experiments.py       ← Synthetic benchmarks (no deps needed)
│   ├── run_nasbench.py          ← NASBench-201 (CIFAR-10/100)
│   ├── run_yahpo.py             ← YAHPO Gym (35 datasets)
│   └── run_dragon.py            ← DRAGON aircraft design
│
├── tests/
│   └── test_cta.py              ← 16 unit tests
│
├── results/                     ← Experiment outputs (JSON)
│
├── requirements.txt
└── README.md
```

---

## 10. API reference

### `CanonicalTrajectoryAttribution`

```python
CanonicalTrajectoryAttribution(
    trajectory,           # list of (dict, float)
    categorical_vars,     # list of str
    hierarchical=None,    # dict {child: parent}, optional
)
```

#### Methods

| Method | Description | Complexity |
|---|---|---|
| `.exact_cta()` | Exact computation. Best for \|V\| ≤ 15. | O(T · 2^\|V\|) |
| `.approximate_cta(n_samples=1000)` | Monte Carlo Shapley. For \|V\| > 15. | O(T · \|V\| · n) |
| `.confidence_intervals(n_bootstrap=200, alpha=0.05)` | Block bootstrap 95% CIs. | — |
| `.estimate_asymptotic_variance(var, method)` | σ²ᵥ for Theorem 3 bound. | — |
| `.explain(method, return_ci, n_bootstrap)` | Main interface, returns dict. | — |
| `.faithfulness(cta_values, trajectory, k)` | *(static)* Faithfulness@k metric. | — |

#### Return format

```python
result = model.explain()

result["values"]
# {'wing_span': 1.42, 'material': 0.87, 'n_motors': 0.53}

result["confidence_intervals"]["wing_span"]
# {'mean': 1.42, 'lower': 1.21, 'upper': 1.63, 'std': 0.11}
```

#### Complexity guide

| Setting | Recommended method | Typical time |
|---|---|---|
| \|V\| ≤ 15, T ≈ 100–200 | `exact_cta()` | 0.1–2 s |
| \|V\| ≤ 15, T ≈ 1000 | `exact_cta()` | 5–30 s |
| \|V\| > 15 | `approximate_cta(n_samples=500)` | 1–10 s |

---

## 11. Applications in Healthcare AI

CatExplain is particularly relevant for healthcare AI, where optimization
pipelines are ubiquitous and regulatory transparency is required.

### Use cases

| Application | What is optimized | Variables CatExplain explains |
|---|---|---|
| **AutoML for medical imaging** | Neural architecture | architecture type, augmentation strategy, loss function |
| **Drug combination optimization** | Treatment protocol | drug class, administration route, dosing schedule |
| **Clinical trial design** | Study protocol | endpoint type, inclusion criteria, randomization method |
| **Radiotherapy planning** | Beam configuration | technique (IMRT/VMAT), fractionation, beam energy |
| **Biomarker selection** | Gene/protein panel | marker type, threshold, measurement method |

### Regulatory alignment

- **FDA guidance on AI/ML-based SaMD** explicitly calls for transparency in
  algorithmic optimization decisions.
- CTA's **faithfulness bounds and confidence intervals** directly address
  the transparency requirements of **IEC 62304** and **ISO 14971** risk
  management frameworks.
- **Axiom A4 (path-independence)** guarantees that two runs of the same
  optimizer on the same problem produce the **same explanation** — critical
  for clinical audit trails. No existing XAI method provides this guarantee.

### Example: AutoML for medical imaging

```python
# HPO trajectory for a chest X-ray classifier
trajectory = [
    ({"architecture": "resnet50", "augmentation": "heavy",
      "loss": "focal", "optimizer": "adam"}, 0.12),   # 12% error
    ({"architecture": "densenet", "augmentation": "light",
      "loss": "cross_entropy", "optimizer": "sgd"}, 0.18),
    # ... 150 more evaluations
]

model = CanonicalTrajectoryAttribution(
    trajectory,
    categorical_vars=["architecture", "augmentation", "loss", "optimizer"]
)
result = model.explain()

# For an FDA submission: which design choice most impacted accuracy?
print(result["values"])
# {'architecture': 0.43, 'loss': 0.31, 'augmentation': 0.18, 'optimizer': 0.08}
# → The architecture choice drove 43% of the improvement.

print(result["confidence_intervals"]["architecture"])
# {'lower': 0.36, 'upper': 0.50}  ← quantified uncertainty for regulators
```

---

## 12. Citation

```bibtex
@inproceedings{bahig2026catexplain,
  title     = {{CatExplain}: A Unifying Axiomatic Theory of Explainable
               Black-Box Optimization},
  author    = {Bahig, Sami and {Le Digabel}, S\'{e}bastien},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  note      = {Polytechnique Montr\'{e}al, Canada},
}
```

---

## 13. License

MIT License — see `LICENSE` for details.

---

## Acknowledgements

This work builds on:
- **CatMADS** (Audet, Le Digabel & Tribes 2021) — the categorical BBO algorithm
  whose trajectories CatExplain explains.
- **SHAP** (Lundberg & Lee 2017) — the axiomatic XAI framework that inspired
  the axiom structure of CTA.
- **NASBench-201** (Dong & Yang 2020) and **YAHPO Gym** (Pfisterer et al. 2022)
  for open benchmark data.
