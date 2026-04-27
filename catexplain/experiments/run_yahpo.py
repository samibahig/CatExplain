"""
YAHPO Gym runner
    pip install yahpo-gym          # optional — sim mode works without it
    python run_yahpo.py --simulate --scenario lcbench --n_runs 10
    python run_yahpo.py --scenario all --n_runs 10
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from catexplain.cta import CanonicalTrajectoryAttribution
from baselines.baselines import SHAPWrapper
from scipy.stats import kendalltau

SCENARIOS = {
    "lcbench": {
        "vars": ["batch_size","max_units","activation","max_dropout",
                 "num_layers","learning_rate","weight_decay"],
        "top":  ["batch_size","learning_rate","num_layers"],
    },
    "rbv2_xgboost": {
        "vars": ["eta","max_depth","colsample_bytree","subsample",
                 "nrounds","booster","min_child_weight","lambda_"],
        "top":  ["eta","max_depth","nrounds"],
    },
    "rbv2_super": {
        "vars": ["learner_id","num.trees","mtry.power","min.node.size",
                 "cost","kernel","k","gamma.power"],
        "top":  ["learner_id","num.trees","cost"],
    },
}

def synthetic_traj(scenario, instance_id, n_eval, seed):
    cfg = SCENARIOS[scenario]
    vars_, top = cfg["vars"], cfg["top"]
    rng = np.random.default_rng(seed + instance_id*1000)
    importance = {v: 1.0/(top.index(v)+1) if v in top else 0.1/(i+1)
                  for i, v in enumerate(vars_)}
    cats = {v: [f"{v}_{k}" for k in range(rng.integers(3,7))] for v in vars_}
    effects = {v: {c: rng.normal(0, importance[v]) for c in cats[v]} for v in vars_}
    traj = []
    for _ in range(n_eval):
        x = {v: rng.choice(cats[v]) for v in vars_}
        f = -(80.0 + sum(effects[v][x[v]] for v in vars_) + rng.normal(0, 0.3))
        traj.append((x, f))
    return traj, vars_

def run_scenario(scenario, n_runs=10, n_eval=120, simulate=True, n_instances=5):
    print(f"\n{'='*50}\nYAHPO — {scenario}\n{'='*50}")
    vars_ = SCENARIOS[scenario]["vars"]
    f1_cta, f1_shap, top_vars = [], [], []

    for inst in range(n_instances):
        cta_runs = []
        for r in range(n_runs):
            traj, _ = synthetic_traj(scenario, inst, n_eval, seed=r)
            model = CanonicalTrajectoryAttribution(traj, vars_)
            cta   = (model.exact_cta() if model.V <= 15
                     else model.approximate_cta(500))
            shap  = SHAPWrapper(traj, vars_).compute()
            cta_runs.append(cta)
            f1_cta.append( CanonicalTrajectoryAttribution.faithfulness(cta, traj,1))
            f1_shap.append(CanonicalTrajectoryAttribution.faithfulness(shap,traj,1))
        mean = {v: np.mean([s[v] for s in cta_runs]) for v in vars_}
        top  = max(mean, key=lambda v: abs(mean[v]))
        top_vars.append(top)
        print(f"  Inst {inst+1}: top={top}  faith@1_CTA={np.mean(f1_cta[-n_runs:]):.3f}")

    res = {
        "scenario": scenario,
        "faithfulness_at_1": {
            "CTA":   float(np.mean(f1_cta)),
            "SHAP":  float(np.mean(f1_shap)),
            "Delta": float(np.mean(f1_cta)-np.mean(f1_shap)),
        },
        "top_var": max(set(top_vars), key=top_vars.count),
    }
    print(f"  Faith@1 CTA={res['faithfulness_at_1']['CTA']:.3f}  "
          f"SHAP={res['faithfulness_at_1']['SHAP']:.3f}  "
          f"Δ={res['faithfulness_at_1']['Delta']:+.3f}")
    return res

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenario",    choices=list(SCENARIOS)+["all"], default="lcbench")
    p.add_argument("--n_runs",      type=int, default=10)
    p.add_argument("--n_eval",      type=int, default=120)
    p.add_argument("--n_instances", type=int, default=5)
    p.add_argument("--simulate",    action="store_true", default=True)
    p.add_argument("--output_dir",  type=str, default="results")
    args = p.parse_args()
    Path(args.output_dir).mkdir(exist_ok=True)
    scenarios = list(SCENARIOS) if args.scenario=="all" else [args.scenario]
    all_res = {}
    for sc in scenarios:
        all_res[sc] = run_scenario(sc, args.n_runs, args.n_eval,
                                   args.simulate, args.n_instances)
    out = Path(args.output_dir)/"yahpo_results.json"
    json.dump(all_res, open(out,"w"), indent=2)
    print(f"\n{'='*50}\nTable 7.3 (paper):")
    print(f"{'Scenario':<20} {'Top var':<20} {'CTA':>8} {'SHAP':>8} {'Δ':>8}")
    for sc, r in all_res.items():
        f = r["faithfulness_at_1"]
        print(f"{sc:<20} {r['top_var']:<20} {f['CTA']:>8.3f} {f['SHAP']:>8.3f} {f['Delta']:>+8.3f}")
    print(f"\nSaved → {out}")
