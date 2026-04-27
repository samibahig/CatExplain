"""
DRAGON aircraft design runner
    python run_dragon.py --simulate --n_runs 10 --n_eval 160
    python run_dragon.py --n_runs 10 --n_eval 160   # with real NOMAD
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from catexplain.cta import CanonicalTrajectoryAttribution
from baselines.baselines import SHAPWrapper, Ablation, FANOVA
from scipy.stats import kendalltau

CAT_VARS = ["material","n_motors","wing_span","fuselage","r1","r2"]
CATEGORIES = {
    "material":  ["carbon","aluminium","composite"],
    "n_motors":  ["2","4","6"],
    "wing_span": ["8","10","12","14"],
    "fuselage":  ["thin","medium","thick"],
    "r1":        ["0.1","0.3","0.5","0.7","0.9"],
    "r2":        ["0.1","0.3","0.5","0.7","0.9"],
}
GT_RANK = {"wing_span":1,"material":2,"n_motors":3,"fuselage":4,"r1":5,"r2":6}

def synthetic_dragon(config):
    effects = {
        "material":  {"carbon":-3.0,"aluminium":0.0,"composite":-1.5},
        "n_motors":  {"2":0.0,"4":-2.0,"6":-1.0},
        "wing_span": {"8":0.0,"10":-2.5,"12":-4.0,"14":-3.0},
        "fuselage":  {"thin":0.0,"medium":-1.0,"thick":-0.5},
        "r1":        {v:-0.8*float(v) for v in CATEGORIES["r1"]},
        "r2":        {v:-0.3*float(v) for v in CATEGORIES["r2"]},
    }
    return sum(effects[v][config[v]] for v in CAT_VARS)

def run(n_runs=10, n_eval=160, simulate=True, nomad_path=None):
    print(f"\n{'='*50}\nDRAGON | runs={n_runs} | evals={n_eval}\n{'='*50}")
    cta_all, f1_cta, f1_shap, rts = [], [], [], []

    for r in range(n_runs):
        rng = np.random.default_rng(r)
        traj = []
        for _ in range(n_eval):
            x = {v: rng.choice(CATEGORIES[v]) for v in CAT_VARS}
            f = synthetic_dragon(x) + rng.normal(0, 0.3)
            traj.append((x, f))

        t0 = time.perf_counter()
        model = CanonicalTrajectoryAttribution(traj, CAT_VARS)
        cta   = model.exact_cta()
        ci    = model.confidence_intervals(n_bootstrap=100)
        rts.append(time.perf_counter()-t0)
        cta_all.append(cta)

        shap = SHAPWrapper(traj, CAT_VARS).compute()
        f1_cta.append( CanonicalTrajectoryAttribution.faithfulness(cta, traj,1))
        f1_shap.append(CanonicalTrajectoryAttribution.faithfulness(shap,traj,1))

        ranks = {v:i+1 for i,(v,_) in enumerate(sorted(cta.items(),key=lambda x:-abs(x[1])))}
        tau = kendalltau([GT_RANK[v] for v in CAT_VARS],[ranks[v] for v in CAT_VARS])[0]
        print(f"  Run {r+1:2d}: τ_GT={tau:.3f}  faith@1={f1_cta[-1]:.3f}  rt={rts[-1]:.2f}s")

    mean_cta = {v: float(np.mean([s[v] for s in cta_all])) for v in CAT_VARS}
    results = {
        "n_runs": n_runs, "n_eval": n_eval, "simulated": simulate,
        "faithfulness_at_1": {
            "CTA":float(np.mean(f1_cta)),"SHAP":float(np.mean(f1_shap)),
            "Delta":float(np.mean(f1_cta)-np.mean(f1_shap))},
        "runtime_s": float(np.mean(rts)),
        "cta_scores": mean_cta,
    }
    print(f"\n  Table 7.1:")
    print(f"  {'Variable':<12} {'CTA φ̂':>8}  {'CTA Rank':>9}  {'GT Rank':>8}")
    for v, s in sorted(mean_cta.items(), key=lambda x: -abs(x[1])):
        rk = sorted(mean_cta, key=lambda x: -abs(mean_cta[x])).index(v)+1
        print(f"  {v:<12} {s:>8.3f}  {rk:>9}  {GT_RANK[v]:>8}")
    print(f"\n  Faith@1: CTA={results['faithfulness_at_1']['CTA']:.3f}  "
          f"SHAP={results['faithfulness_at_1']['SHAP']:.3f}")
    return results

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_runs",     type=int,  default=10)
    p.add_argument("--n_eval",     type=int,  default=160)
    p.add_argument("--simulate",   action="store_true", default=True)
    p.add_argument("--nomad_path", type=str,  default=None)
    p.add_argument("--output_dir", type=str,  default="results")
    args = p.parse_args()
    Path(args.output_dir).mkdir(exist_ok=True)
    res = run(args.n_runs, args.n_eval, args.simulate, args.nomad_path)
    out = Path(args.output_dir)/"dragon_results.json"
    json.dump(res, open(out,"w"), indent=2)
    print(f"\nSaved → {out}")
