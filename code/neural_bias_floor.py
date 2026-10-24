"""V11 diagnostic: bias of the sampled point target with the true metric as ground cost."""
from pathlib import Path
import json

import numpy as np
from scipy.optimize import linear_sum_assignment

from finite_batch import solve
from run_verified_study import ring

ROOT = Path(__file__).resolve().parents[1]


def main():
    gamma = .6
    model = ring(6, gamma)
    metric = solve(model.reward, model.transition, gamma)["truth"][0]
    rng = np.random.default_rng(np.random.SeedSequence([71000]))
    out = {"note": "post hoc diagnostic; true metric as ground cost; 3000 random latent pairs per m",
           "rows": []}
    for m in [8, 32, 128]:
        same, bias = [], []
        for _ in range(3000):
            s, t = rng.integers(6, size=2)
            q = []
            for a in range(2):
                xs = rng.choice(6, size=m, p=model.transition[s, a])
                ys = rng.choice(6, size=m, p=model.transition[t, a])
                rs = (rng.random(m) < model.reward[s, a]).mean()
                rt = (rng.random(m) < model.reward[t, a]).mean()
                cost = metric[np.ix_(xs, ys)]
                rows, cols = linear_sum_assignment(cost)
                q.append((1-gamma)*abs(rs-rt)+gamma*cost[rows, cols].mean())
            if s == t:
                same.append(max(q))
            else:
                bias.append(max(q)-metric[s, t])
        out["rows"].append({"successors": m, "same_latent_target_mean": float(np.mean(same)),
                            "distinct_pair_bias_mean": float(np.mean(bias)), "same_pairs": len(same)})
        print(out["rows"][-1], flush=True)
    (ROOT/"results"/"neural"/"bias_floor.json").write_text(json.dumps(out, indent=2)+"\n", encoding="utf8")


if __name__ == "__main__":
    main()
