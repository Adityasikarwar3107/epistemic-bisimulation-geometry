"""V1 (calibration), V2 (propagation), V3 (ablation), and V4 (budgets)."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import platform
import time

import numpy as np
import scipy
from scipy.stats import beta as beta_distribution

from ebg import (Model, confidence_radii, reach_policy, goal_policy,
                 evaluate_goal_policy)
from finite_batch import solve, PairTransport

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results"/"verified"
SPEC = {
    "version": "finite-study-v1", "nominal_delta": .1,
    "calibration_seeds": list(range(10000, 10500)),
    "calibration_checkpoints_per_row": [0, 128, 512, 2048, 8192, 32768, 131072],
    "calibration_models": [[6, .6], [8, .9]],
    "propagation_seeds": list(range(31000, 31100)),
    "propagation_counts_per_uncertain_row": [0, 32, 128, 512, 2048, 8192],
    "propagation_escape_probabilities": [.02, .10, .30],
    "ablation_seeds": list(range(51000, 51032)),
    "ablation_tasks": ["corridor", "branching_return"],
    "ablation_priorities": ["local", "propagated"],
    "ablation_navigation": ["random", "optimistic"],
    "ablation_retained_quota": 64, "ablation_refresh_episodes": 128,
    "ablation_step_budget": 250000, "bootstrap_resamples": 5000,
    "numeric_tolerance": 2e-11,
    "scope": "Finite-state experiments; no image-based or continuous-control benchmark",
}


def csv_write(name, rows):
    with (OUT/name).open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval_probability(successes, total):
    return [float(beta_distribution.ppf(.025, successes, total-successes+1))
            if successes else 0.,
            float(beta_distribution.ppf(.975, successes+1, total-successes))
            if successes < total else 1.]


def bootstrap_mean(values, seed=899):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values),
                                 (SPEC["bootstrap_resamples"], len(values)))].mean(1)
    return {"mean": float(values.mean()),
            "ci95": np.quantile(samples, [.025, .975]).tolist()}


def ring(states, gamma):
    reward = np.c_[np.linspace(.08, .82, states),
                   .12+.68*((np.arange(states)*3) % states)/max(1, states-1)]
    transition = np.zeros((states, 2, states))
    for s in range(states):
        for a in range(2):
            transition[s, a, s] = .25
            transition[s, a, (s+(1 if a else -1)) % states] = .75
    return Model(reward, transition, gamma)


def sampled_prefixes(model, seeds, checkpoints, tag):
    """Exact binomial sufficient statistics of nested, independent row streams."""
    states, actions = model.reward.shape
    batch = len(seeds)
    indices = np.argsort(-model.transition, axis=-1, kind="stable")[..., :2]
    success_probability = np.take_along_axis(model.transition,
                                             indices[..., :1], axis=-1)[..., 0]
    reward_rng = [np.random.default_rng(np.random.SeedSequence([seed, tag, 0]))
                  for seed in seeds]
    transition_rng = [np.random.default_rng(np.random.SeedSequence([seed, tag, 1]))
                      for seed in seeds]
    reward_sum = np.zeros((batch, states, actions), dtype=np.int64)
    successor_sum = np.zeros_like(reward_sum)
    previous = 0
    for n in checkpoints:
        increment = n-previous
        if increment:
            for b in range(batch):
                reward_sum[b] += reward_rng[b].binomial(increment, model.reward)
                successor_sum[b] += transition_rng[b].binomial(increment, success_probability)
        rhat = reward_sum/max(n, 1)
        phat = np.zeros((batch, states, actions, states))
        for s in range(states):
            for a in range(actions):
                p = successor_sum[:, s, a]/n if n else np.full(batch, .5)
                phat[:, s, a, indices[s, a, 0]] = p
                phat[:, s, a, indices[s, a, 1]] += 1-p
        yield n, rhat, phat, reward_sum.copy(), successor_sum.copy()
        previous = n


def calibration():
    rows, summaries = [], []
    for model_id, (states, gamma) in enumerate(SPEC["calibration_models"]):
        model = ring(states, gamma)
        truth = solve(model.reward, model.transition, gamma)["truth"][0]
        seeds = SPEC["calibration_seeds"]
        whole_run = np.ones(len(seeds), dtype=bool)
        for n, rh, ph, reward_sum, successor_sum in sampled_prefixes(
                model, seeds, SPEC["calibration_checkpoints_per_row"], model_id+1):
            br, bp = confidence_radii(np.full((states, 2), n), 2, SPEC["nominal_delta"])
            computed = solve(rh, ph, gamma, br=br, bp=bp)
            lo, hi, majorant = (computed[k] for k in ["lower", "upper", "majorant"])
            width = hi-lo
            covered = ((lo <= truth+2e-9) & (truth <= hi+2e-9)).all((1, 2))
            whole_run &= covered
            error = np.max(np.abs((lo+hi)/2-truth), axis=(1, 2))
            sup_width = width.max((1, 2))
            violation = np.maximum(width-majorant, 0).max((1, 2))
            triangle = np.maximum(hi[:, :, :, None]-hi[:, :, None, :]
                                  -hi[:, None, :, :], 0).max((1, 2, 3))
            assert np.all(error[covered] <= sup_width[covered]/2+2e-9)
            assert np.max(violation) < 2e-8 and np.max(triangle) < 2e-8
            if n == 0:
                assert np.max(error) <= .5+2e-9
            for b, seed in enumerate(seeds):
                rows.append({"model": f"ring{states}_g{gamma}", "seed": seed,
                    "states": states, "actions": 2, "gamma": gamma,
                    "n_per_row": n, "all_pairs_covered": int(covered[b]),
                    "sup_width": float(sup_width[b]), "midpoint_sup_error": float(error[b]),
                    "majorant_violation": float(violation[b]),
                    "upper_triangle_violation": float(triangle[b]),
                    "lower_residual": float(computed["lower_residual"][b]),
                    "upper_residual": float(computed["upper_residual"][b])})
            summaries.append({"model": f"ring{states}_g{gamma}", "n_per_row": n,
                "mean_width": float(sup_width.mean()), "mean_error": float(error.mean()),
                "coverage": float(covered.mean()), "max_majorant_violation": float(violation.max())})
            np.savez_compressed(OUT/f"calibration_{model_id}_n{n}.npz",
                seeds=seeds, reward_estimates=rh, transition_estimates=ph,
                reward_success_counts=reward_sum, successor_first_counts=successor_sum,
                true_reward=model.reward, true_transition=model.transition, true_metric=truth,
                lower=lo, upper=hi, pair_majorant=majorant, gamma=gamma, n=n)
            print(f"V1 ring{states} gamma={gamma} n={n}: width={sup_width.mean():.4f}, "
                  f"error={error.mean():.4f}, coverage={covered.mean():.3f}", flush=True)
        successes = int(whole_run.sum())
        summaries.append({"model": f"ring{states}_g{gamma}", "whole_run_successes": successes,
                          "runs": len(seeds), "coverage_ci95": interval_probability(successes, len(seeds))})
    csv_write("calibration.csv", rows)
    return summaries


def propagation():
    rows = []
    gamma = .8
    checkpoints = SPEC["propagation_counts_per_uncertain_row"]
    seeds = SPEC["propagation_seeds"]
    for eta in SPEC["propagation_escape_probabilities"]:
        p = np.eye(4)[:, None, :]
        p[0, 0] = [1-eta, 0, eta, 0]
        p[1, 0] = [0, 1-eta, 0, eta]
        rewards = np.array([[.1], [.1], [.3], [.7]])
        truth = solve(rewards, p, gamma)["truth"][0]
        total = np.zeros((len(seeds), 2), dtype=int)
        rngs = [np.random.default_rng(np.random.SeedSequence([seed, 2])) for seed in seeds]
        previous = 0
        for n in checkpoints:
            if n > previous:
                for b in range(len(seeds)):
                    total[b] += rngs[b].binomial(n-previous, [.3, .7])
            rh = np.broadcast_to(rewards, (len(seeds), 4, 1)).copy()
            rh[:, 2:, 0] = total/max(n, 1)
            br = np.zeros((4, 1))
            br[2:, 0] = confidence_radii(np.full((4, 1), n), 1, .1)[0][2:, 0]
            bp = np.zeros((4, 1))  # all transition rows are declared known in V2
            result = solve(rh, np.broadcast_to(p, (len(seeds), 4, 1, 4)), gamma, br=br, bp=bp)
            width = result["upper"]-result["lower"]
            global_bound = min(1., float(2*result["bonus"].max()/(1-gamma)))
            for b, seed in enumerate(seeds):
                bound = result["majorant"][b, 0, 1]
                assert width[b, 0, 1] <= bound+2e-8
                rows.append({"seed": seed, "eta": eta, "gamma": gamma,
                    "n_per_uncertain_row": n, "local_source_R": 0.,
                    "true_distance_R": float(truth[0, 1]),
                    "width_R": float(width[b, 0, 1]), "pair_majorant_R": float(bound),
                    "global_bound": global_bound, "known_transition_model": 1,
                    "covered_R": int(result["lower"][b, 0, 1] <= truth[0, 1]+2e-9
                                     and truth[0, 1] <= result["upper"][b, 0, 1]+2e-9)})
            previous = n
    csv_write("propagation.csv", rows)
    return {"rows": len(rows), "scope": "Known current rows/dynamics; sampled Bernoulli successor rewards"}


def navigation_model(name):
    states = 6 if name == "corridor" else 7
    p = np.zeros((states, 2, states))
    if name == "corridor":
        for s in range(states):
            p[s, 0, 0] = 1
            p[s, 1, min(s+1, states-1)] += .85
            p[s, 1, s] += .15
        horizon = 9
    else:
        for s in range(states):
            for a in range(2):
                if s < 3:
                    p[s, a, 2*s+1+a] = .8
                    p[s, a, s] = .2
                else:
                    p[s, a, 0] = 1
        horizon = 7
    reward = np.c_[.05+.6*np.arange(states)/(states-1),
                   .1+.7*((np.arange(states)*3) % states)/(states-1)]
    return Model(reward, p, .6), horizon


def ablation_run(task, seed, priority_mode, navigation_mode):
    model, horizon = navigation_model(task)
    states, actions = model.reward.shape
    task_id = SPEC["ablation_tasks"].index(task)+1
    transition_rng = [[np.random.default_rng(np.random.SeedSequence([seed, task_id, s, a, 0]))
                       for a in range(actions)] for s in range(states)]
    reward_rng = [[np.random.default_rng(np.random.SeedSequence([seed, task_id, s, a, 1]))
                   for a in range(actions)] for s in range(states)]
    policy_rng = np.random.default_rng(np.random.SeedSequence([seed, task_id, 91]))
    counts = np.zeros((states, actions), dtype=int)
    outcomes = np.zeros((states, actions, states))
    reward_sum = np.zeros((states, actions))
    quota, budget = SPEC["ablation_retained_quota"], SPEC["ablation_step_budget"]
    steps = episodes = 0
    priorities = np.ones((states, actions))
    while counts.min() < quota and steps < budget:
        phat = np.eye(states)[:, None, :].repeat(actions, axis=1)
        np.divide(outcomes, np.maximum(counts, 1)[:, :, None], out=phat,
                  where=(counts > 0)[:, :, None])
        unknown = counts < quota
        if episodes % SPEC["ablation_refresh_episodes"] == 0:
            br, bp = confidence_radii(counts, 2, .1)
            rh = reward_sum/np.maximum(counts, 1)
            fit = solve(rh, phat, model.gamma, br=br, bp=bp)
            score = 2*fit["bonus"][0]
            if priority_mode == "propagated":
                operator = PairTransport(phat)
                score += model.gamma*operator.apply(
                    fit["majorant"], operator.coefficients(fit["lower_raw"]))[0]
            score = score.max(axis=1)
            maximum = score[unknown].max()
            priorities = np.ones((states, actions))
            if maximum > 0:
                priorities[unknown] = .5+.5*score[unknown]/maximum
        policy = reach_policy(phat, unknown, horizon, priorities)[0]
        state = 0
        episodes += 1
        for remaining in range(horizon, 0, -1):
            action = int(policy_rng.integers(actions)) if navigation_mode == "random" else \
                int(policy[remaining, state])
            value = transition_rng[state][action].random()
            successor = int(np.searchsorted(np.cumsum(model.transition[state, action]), value,
                                            side="right"))
            steps += 1
            if unknown[state, action]:
                counts[state, action] += 1
                outcomes[state, action, successor] += 1
                reward_sum[state, action] += reward_rng[state][action].binomial(1, model.reward[state, action])
                break
            state = successor
            if steps >= budget:
                break
    completed = counts.min() >= quota
    phat = np.eye(states)[:, None, :].repeat(actions, axis=1)
    np.divide(outcomes, np.maximum(counts, 1)[:, :, None], out=phat, where=(counts > 0)[:, :, None])
    goal_horizon = horizon-1
    reaching = min(float(evaluate_goal_policy(model.transition,
        goal_policy(phat, goal, goal_horizon)[0], goal, goal_horizon)[0]) for goal in range(states))
    return {"task": task, "seed": seed, "priority": priority_mode,
        "navigation": navigation_mode, "steps": steps, "episodes": episodes,
        "completed": int(completed), "step_budget": budget, "retained_quota": quota,
        "retained_samples": int(counts.sum()), "goal_horizon": goal_horizon,
        "min_goal_reach_probability": reaching,
        "empirical_model_hash": hashlib.sha256(phat.tobytes()+reward_sum.tobytes()).hexdigest()}


def ablations():
    rows, summaries, contrasts = [], [], []
    for task in SPEC["ablation_tasks"]:
        for priority in SPEC["ablation_priorities"]:
            for navigation in SPEC["ablation_navigation"]:
                group = [ablation_run(task, seed, priority, navigation)
                         for seed in SPEC["ablation_seeds"]]
                rows.extend(group)
                summary = {"task": task, "priority": priority, "navigation": navigation,
                    "completed": sum(r["completed"] for r in group), "runs": len(group),
                    **bootstrap_mean([r["steps"] for r in group])}
                summaries.append(summary)
                print("V3", task, priority, navigation, summary, flush=True)
        for navigation in SPEC["ablation_navigation"]:
            local = {r["seed"]: r for r in rows if r["task"] == task
                     and r["navigation"] == navigation and r["priority"] == "local"}
            propagated = {r["seed"]: r for r in rows if r["task"] == task
                          and r["navigation"] == navigation and r["priority"] == "propagated"}
            differences = [propagated[s]["steps"]-local[s]["steps"] for s in SPEC["ablation_seeds"]]
            contrasts.append({"task": task, "navigation": navigation,
                              "contrast": "propagated minus local, interaction steps",
                              **bootstrap_mean(differences)})
        # Every completed variant consumes the same first quota outcomes per row.
        for seed in SPEC["ablation_seeds"]:
            complete = [r for r in rows if r["task"] == task and r["seed"] == seed and r["completed"]]
            assert len({r["empirical_model_hash"] for r in complete}) <= 1
    csv_write("ablation.csv", rows)
    return {"groups": summaries, "paired_priority_contrasts": contrasts,
            "endpoint": "Retain 64 samples per row; not a metric-resolution or theorem event"}


def budgets():
    rows = []
    for dimension in [1, 2, 3]:
        for m in [4, 6, 8, 10, 12]:
            points = m**dimension
            epsilon = 1/(32*m)
            quota = math.ceil(8/epsilon**2*math.log(2*points/.1))
            rows.append({"kind": "calculated theorem quota, not observed stopping time",
                         "d": dimension, "m": m, "M": points, "epsilon": epsilon,
                         "cover_queries": points+1, "n_per_unknown_row": quota,
                         "resolution_query_budget": points*quota+1})
    csv_write("clock_budgets.csv", rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["all", "calibration", "propagation", "ablations", "budgets"], default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    spec_path = OUT/"experiment_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != SPEC:
        raise RuntimeError("The frozen specification differs; use a new version/output folder")
    spec_path.write_text(json.dumps(SPEC, indent=2)+"\n", encoding="utf8")
    target = OUT/"summary.json"
    summary = json.loads(target.read_text()) if target.exists() else {}
    started = time.time()
    for name, runner in [("calibration", calibration), ("propagation", propagation),
                         ("ablations", ablations), ("budgets", budgets)]:
        if args.part in ["all", name]:
            summary[name] = runner()
            target.write_text(json.dumps(summary, indent=2)+"\n", encoding="utf8")
    manifest = {"python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "platform": platform.platform(),
        "elapsed_seconds_this_invocation": time.time()-started,
        "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [Path(__file__), ROOT/"code/finite_batch.py", ROOT/"code/ebg.py"]},
        "result_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(OUT.glob("*")) if p.suffix in [".csv", ".npz"]},
        "access": "generative row samples for V1; partial known model for V2; reset interaction for V3",
        "unverified_legacy_aggregates_used_as_new_results": False}
    (OUT/"run_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf8")
    print("Completed", args.part, "in", round(time.time()-started, 2), "seconds", flush=True)


if __name__ == "__main__":
    main()
