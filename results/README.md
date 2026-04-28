# CatExplain — Datasets

This folder contains all trajectory datasets used in the paper experiments.

---

## Files

| File | Benchmark | Runs | Evals/run | Size |
|---|---|---|---|---|
| `dragon_trajectories.json` | DRAGON aircraft design | 10 | 160 | ~434 KB |
| `hpo_trajectories.json` | HPO XGBoost / CIFAR-10 | 10 | 120 | ~311 KB |
| `nasbench_trajectories.json` | NASBench-201 CIFAR-10 | 10 | 100 | ~329 KB |
| `yahpo_trajectories.json` | YAHPO Gym (3 scenarios) | 5 | 120 | ~766 KB |

---

## Format

Every file follows this structure:

```json
{
  "benchmark": "DRAGON",
  "description": "...",
  "categorical_vars": ["var1", "var2", ...],
  "ground_truth_rank": {"var1": 1, "var2": 2, ...},
  "n_runs": 10,
  "n_eval_per_run": 160,
  "runs": [
    {
      "run_id": 0,
      "categorical_vars": [...],
      "n_eval": 160,
      "trajectory": [
        {"config": {"var1": "val_a", "var2": "val_b"}, "objective": -42.3},
        ...
      ]
    },
    ...
  ]
}
```

`objective` is always **minimization** (lower = better). If the original
benchmark maximizes (e.g. accuracy), values are negated.

---

## Loading a dataset

```python
import json
from catexplain import CanonicalTrajectoryAttribution

# Load
with open("results/dragon_trajectories.json") as f:
    data = json.load(f)

# Use one run
run       = data["runs"][0]
cat_vars  = run["categorical_vars"]
trajectory = [(item["config"], item["objective"])
              for item in run["trajectory"]]

# Compute CTA
model  = CanonicalTrajectoryAttribution(trajectory, cat_vars)
result = model.explain()
print(result["values"])
```

---

## Note on synthetic vs real data

These datasets are **synthetic surrogates** that mirror the statistical
properties of the real benchmarks:

- **DRAGON**: effects calibrated from published CatMADS results on UAV design.
- **HPO XGBoost**: effects calibrated from YAHPO lcbench/rbv2_xgboost data.
- **NASBench-201**: operation effects and edge importance weights match the
  published NASBench-201 ground-truth ranking (Dong & Yang 2020, Table 3).
- **YAHPO**: surrogates of YAHPO Gym lcbench, rbv2_xgboost, rbv2_super.

To run with **real** benchmark data, see the instructions in the main README:
- NASBench-201: `pip install nas-bench-api` + download data file
- YAHPO Gym: `pip install yahpo-gym` (data downloads automatically)
- DRAGON: requires NOMAD/CatMADS installation
