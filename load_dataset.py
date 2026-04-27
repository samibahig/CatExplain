"""
load_dataset.py
===============
Utility to load any CatExplain dataset and run CTA on it directly.

Usage:
    python load_dataset.py --file results/dragon_trajectories.json
    python load_dataset.py --file results/nasbench_trajectories.json --run 3
    python load_dataset.py --file results/yahpo_trajectories.json --scenario lcbench
"""

import argparse
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from catexplain.cta import CanonicalTrajectoryAttribution


def load_and_run(filepath, run_id=0, scenario=None, n_bootstrap=100):
    with open(filepath) as f:
        data = json.load(f)

    print(f"\nBenchmark : {data.get('benchmark', 'unknown')}")
    print(f"Description: {data.get('description', '')}")

    # Handle YAHPO multi-scenario format
    if "data" in data and scenario:
        runs = data["data"][scenario]
        print(f"Scenario   : {scenario}")
    elif "runs" in data:
        runs = data["runs"]
    else:
        raise ValueError("Unknown dataset format")

    if run_id >= len(runs):
        print(f"Warning: run_id={run_id} out of range, using 0")
        run_id = 0

    run      = runs[run_id]
    cat_vars = run["categorical_vars"]
    traj     = [(item["config"], item["objective"])
                for item in run["trajectory"]]

    gt_rank = data.get("ground_truth_rank")

    print(f"\nRun {run_id} | {len(traj)} evaluations | {len(cat_vars)} variables")
    print(f"Variables  : {cat_vars}")
    print(f"Best obj   : {min(f for _,f in traj):.4f}")

    # Run CTA
    model  = CanonicalTrajectoryAttribution(traj, cat_vars)
    result = model.explain(return_ci=True, n_bootstrap=n_bootstrap)

    phi = result["values"]
    ci  = result["confidence_intervals"]

    print(f"\n{'─'*60}")
    print(f"{'Variable':<18} {'CTA φ̂':>10}  {'95% CI':>20}  {'Rank':>6}", end="")
    if gt_rank:
        print(f"  {'GT Rank':>8}", end="")
    print()
    print(f"{'─'*60}")

    ranked = sorted(phi.items(), key=lambda x: -abs(x[1]))
    for cta_rank, (v, score) in enumerate(ranked, 1):
        lo = ci[v]["lower"]
        hi = ci[v]["upper"]
        line = (f"{v:<18} {score:>+10.4f}  "
                f"[{lo:+.3f}, {hi:+.3f}]  {cta_rank:>6}")
        if gt_rank:
            line += f"  {gt_rank.get(v, '?'):>8}"
        print(line)

    # Faithfulness
    f1 = CanonicalTrajectoryAttribution.faithfulness(phi, traj, k=1)
    f2 = CanonicalTrajectoryAttribution.faithfulness(phi, traj, k=2)
    print(f"\nFaithfulness@1 = {f1:.4f}")
    print(f"Faithfulness@2 = {f2:.4f}")

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file",        required=True, help="Path to dataset JSON")
    p.add_argument("--run",         type=int, default=0,   help="Run index (default 0)")
    p.add_argument("--scenario",    type=str, default=None, help="YAHPO scenario name")
    p.add_argument("--n_bootstrap", type=int, default=100, help="Bootstrap resamples")
    args = p.parse_args()

    load_and_run(args.file, run_id=args.run,
                 scenario=args.scenario, n_bootstrap=args.n_bootstrap)
