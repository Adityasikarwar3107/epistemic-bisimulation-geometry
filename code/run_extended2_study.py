"""V8 (refined certificates), V9 (RiverSwim), and V10 (adjoint priorities)."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import csv
import hashlib
import json
import platform
import time

import numpy as np
import scipy

from ebg import Model, goal_policy, evaluate_goal_policy, greedy_cover, embedding_distance
from finite_batch import solve
from finite_general import (solve_refined, radii, kl_reward_interval, support_indices,
                            transport_for)
from run_verified_study import (ring, sampled_prefixes, interval_probability,
                                bootstrap_mean, navigation_model)
from run_extended_study import clustered_model, eager_reach_policy

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT/"results"/"verified"
OUT = ROOT/"results"/"extended2"
SPEC = {
    "version": "extended-study-v2", "nominal_delta": .1,
    "methods": ["baseline", "global", "support"],
    "method_definitions": {
        "baseline": "Hoeffding rewards, Weissman TV with Gamma, transport range R_max (Lemma 2)",
        "global": "KL reward intervals, Weissman TV with Gamma, transport range iterated to ||d+||",
        "support": "KL reward intervals, Weissman TV with each row's known support size, "
                   "transport range iterated to the largest d+ among the row's known successors"},
    "v1_models": [[6, .6], [8, .9]], "v1_seeds": list(range(10000, 10500)),
    "v1_checkpoints": [0, 128, 512, 2048, 8192, 32768, 131072],
    "v6_states": [16, 32, 64, 128], "v6_seeds": list(range(80000, 80100)),
    "v6_checkpoints": [2048, 8192, 32768, 131072], "v6_anchor_radius": .2,
    "decision_epsilon": .15, "v6_bootstrap_states": [16, 32],
    "bootstrap_replicates": 200, "bootstrap_level": .9,
    "riverswim": {"states": 6, "gamma": .9, "seeds": list(range(90000, 90100)),
                  "checkpoints": [512, 2048, 8192, 32768, 131072],
                  "definition": "Osband et al. six-state variant: left deterministic; right at 0 stays .4/moves .6, "
                                "interior left .05/stay .6/right .35, at 5 left .4/stay .6; "
                                "Bernoulli rewards .005 for left at 0 and 1 for right at 5"},
    "navigation": {"task": "riverswim", "gamma": .9, "horizon": 20, "quota": 256,
                   "cap": 1000000, "seeds": list(range(92100, 92132)),
                   "rules": ["random", "optimistic"], "ties": "toward unresolved rows"},
    "priorities": {"tasks": ["corridor", "branching_return", "riverswim"],
                   "rules": ["uniform", "source", "propagated", "adjoint"],
                   "seeds": list(range(63100, 63132)), "quota": 4096, "refresh": 512,
                   "budgets": {"corridor": [5000, 10000, 20000, 40000, 80000],
                               "branching_return": [5000, 10000, 20000, 40000, 80000],
                               "riverswim": [20000, 40000, 80000, 160000, 320000]},
                   "riverswim_gamma": .6, "riverswim_horizon": 20, "cap": 2000000,
                   "ties": "toward unresolved rows", "certificate": "baseline"},
    "targeted": {"tasks": {"riverswim": [4, 5], "corridor": [4, 5]},
                 "rules": ["count", "adjoint_target"],
                 "seeds": list(range(64100, 64132)), "gamma": .6, "riverswim_horizon": 20,
                 "stop": "baseline lower endpoint at the target pair exceeds zero",
                 "priority_map": "v = 1/2 + score/(2 max) over rows, as in eq. (43)",
                 "count_score": "row radius term (1-gamma) beta_r + gamma R beta_P, geometry free",
                 "planner": "at each step either sample the row (payoff v) or, if the row has at least "
                            "traversable_after samples, move on under its empirical law; every executed "
                            "transition is retained; ties favor sampling",
                 "traversable_after": 16,
                 "retirement": "adjoint_target only: a row with zero influence on the target pair stops "
                               "being sampled once it has traversable_after samples; the count rule samples "
                               "every row; ties are broken by a 1e-3 multiple of the row radius",
                 "refresh": 32, "check_every": 16, "cap": 200000, "certificate": "baseline"},
    "numeric_tolerance": 2e-11, "bootstrap_resamples": 5000,
    "pilot_note": "Designs of the navigation, priority and targeted parts were checked on pilot seeds 92000, 63000, 64000 and 64001, which are excluded from the reported seed ranges; the targeted design (count baseline, floored priorities, retirement of non-influential rows) was fixed after those pilots",
}


def csv_write(name, rows):
    # Tasks record different budget columns.
    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with (OUT/name).open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def covered(lower, upper, truth):
    return ((lower <= truth+2e-9) & (truth <= upper+2e-9)).all((-2, -1))


def certificates(rh, ph, n, gamma, structure, methods=None):
    """Endpoints for each method from empirical rewards rh and transitions ph at count n."""
    states, actions = rh.shape[-2:]
    gamma_bound = int(np.count_nonzero(structure > 0, -1).max())
    counts = np.full((states, actions), n)
    br, bp, ell = radii(counts, gamma_bound, SPEC["nominal_delta"])
    row_support = np.count_nonzero(structure > 0, -1)
    _, bps, _ = radii(counts, row_support, SPEC["nominal_delta"])
    lo, hi = kl_reward_interval(rh, np.broadcast_to(counts, rh.shape), ell)
    out = {}
    for method in methods or SPEC["methods"]:
        if method == "baseline":
            out[method] = solve_refined(rh-br, rh+br, rh, ph, gamma, bp)
        elif method == "global":
            out[method] = solve_refined(lo, hi, rh, ph, gamma, bp, diameter="global")
        else:
            out[method] = solve_refined(lo, hi, rh, ph, gamma, bps, diameter="support",
                                        support_index=support_indices(structure))
    return out


def decisions(lower, upper, truth, epsilon):
    """Certified merge/split decisions and their errors against the true metric."""
    mask = ~np.eye(truth.shape[-1], dtype=bool)
    zero = (truth < 1e-9) & mask
    far = (truth > epsilon) & mask
    merge = (upper <= epsilon) & mask
    split = (lower > 1e-9) & mask
    decided = (merge | split) & mask
    record = {"decided_fraction": decided.sum((1, 2))/mask.sum(),
              "false_merge": (merge & far).sum((1, 2)), "false_split": (split & zero).sum((1, 2))}
    if zero.any():
        record["zero_pairs_merged"] = (merge & zero).sum((1, 2))/zero.sum()
    if far.any():
        record["far_pairs_split"] = (split & far).sum((1, 2))/far.sum()
    return record


def _v1_job(job):
    model_id, n = job
    states, gamma = SPEC["v1_models"][model_id]
    model = ring(states, gamma)
    archive = np.load(FROZEN/f"calibration_{model_id}_n{n}.npz")
    truth = archive["true_metric"]
    out = certificates(archive["reward_estimates"], archive["transition_estimates"], n, gamma,
                       model.transition, ["global", "support"])
    out["baseline"] = {"lower": archive["lower"], "upper": archive["upper"]}
    rows = []
    for method, r in out.items():
        ok = covered(r["lower"], r["upper"], truth)
        width = (r["upper"]-r["lower"]).max((1, 2))
        error = np.abs((r["lower"]+r["upper"])/2-truth).max((1, 2))
        for b, seed in enumerate(archive["seeds"]):
            rows.append({"model": f"ring{states}_g{gamma}", "seed": int(seed), "n_per_row": n,
                         "method": method, "all_pairs_covered": int(ok[b]),
                         "sup_width": float(width[b]), "midpoint_sup_error": float(error[b]),
                         "stages": int(r.get("stages", 1))})
    return rows


def refine_v1():
    jobs = [(m, n) for m in range(len(SPEC["v1_models"])) for n in SPEC["v1_checkpoints"]]
    with ProcessPoolExecutor(max_workers=7) as pool:
        rows = [row for part in pool.map(_v1_job, jobs) for row in part]
    csv_write("v1_refined.csv", rows)
    summary = []
    for model in sorted({r["model"] for r in rows}):
        for method in SPEC["methods"]:
            group = [r for r in rows if r["model"] == model and r["method"] == method]
            runs = {}
            for r in group:
                runs.setdefault(r["seed"], []).append(r["all_pairs_covered"])
            whole = sum(all(v) for v in runs.values())
            for n in SPEC["v1_checkpoints"]:
                cell = [r for r in group if r["n_per_row"] == n]
                summary.append({"model": model, "method": method, "n_per_row": n,
                                "coverage": float(np.mean([r["all_pairs_covered"] for r in cell])),
                                "mean_sup_width": float(np.mean([r["sup_width"] for r in cell])),
                                "mean_error": float(np.mean([r["midpoint_sup_error"] for r in cell]))})
            summary.append({"model": model, "method": method, "whole_run_successes": whole,
                            "runs": len(runs), "coverage_ci95": interval_probability(whole, len(runs))})
            print("V8-V1", model, method, whole, flush=True)
    return summary


def band_from_bootstrap(rh, first, n, gamma, structure, seeds, tag, mask):
    """Simultaneous bootstrap band for two-successor rows, as in V5."""
    states = rh.shape[-2]
    replicates, level = SPEC["bootstrap_replicates"], SPEC["bootstrap_level"]
    indices = np.argsort(-structure, axis=-1, kind="stable")[..., :2]
    phat = np.zeros(rh.shape+(states,))
    for s in range(states):
        for a in range(2):
            phat[:, s, a, indices[s, a, 0]] = first[:, s, a]
            phat[:, s, a, indices[s, a, 1]] += 1-first[:, s, a]
    plug = solve(rh, phat, gamma)["truth"]
    lo, hi = np.zeros_like(plug), np.zeros_like(plug)
    chunk = max(1, 4000//replicates)
    for start in range(0, len(seeds), chunk):
        stop = min(start+chunk, len(seeds))
        draws_r, draws_p = [], []
        for b in range(start, stop):
            rng = np.random.default_rng(np.random.SeedSequence([int(seeds[b]), tag, n, 5]))
            draws_r.append(rng.binomial(n, rh[b], (replicates, states, 2))/n)
            draws_p.append(rng.binomial(n, first[b], (replicates, states, 2))/n)
        rstar, pstar = np.concatenate(draws_r), np.concatenate(draws_p)
        pp = np.zeros(pstar.shape+(states,))
        for s in range(states):
            for a in range(2):
                pp[:, s, a, indices[s, a, 0]] = pstar[:, s, a]
                pp[:, s, a, indices[s, a, 1]] += 1-pstar[:, s, a]
        boot = solve(rstar, pp, gamma, tolerance=1e-10)["truth"].reshape(stop-start, replicates, states, states)
        critical = np.quantile(np.abs(boot-plug[start:stop, None]).max((2, 3)), level, axis=1)[:, None, None]
        lo[start:stop] = np.where(mask, np.maximum(0, plug[start:stop]-critical), 0)
        hi[start:stop] = np.where(mask, np.minimum(1, plug[start:stop]+critical), 0)
    return lo, hi


def _v6_job(states):
    gamma, radius, eps = .6, SPEC["v6_anchor_radius"], SPEC["decision_epsilon"]
    model, block = clustered_model(states, gamma)
    truth = solve(model.reward, model.transition, gamma)["truth"][0]
    seeds = SPEC["v6_seeds"]
    mask = ~np.eye(states, dtype=bool)
    rows = []
    for n, rh, ph, reward_sum, successor_sum in sampled_prefixes(model, seeds, SPEC["v6_checkpoints"], 100+states):
        out = certificates(rh, ph, n, gamma, model.transition)
        if states in SPEC["v6_bootstrap_states"]:
            lo, hi = band_from_bootstrap(rh, successor_sum/n, n, gamma, model.transition, seeds, 200+states, mask)
            out["bootstrap_band"] = {"lower": lo, "upper": hi}
        for method, r in out.items():
            lo, hi = r["lower"], r["upper"]
            ok = covered(lo, hi, truth)
            width = (hi-lo).max((1, 2))
            dec = decisions(lo, hi, truth, eps)
            for b, seed in enumerate(seeds):
                record = {"states": states, "seed": seed, "n_per_row": n, "method": method,
                          "all_pairs_covered": int(ok[b]), "sup_width": float(width[b]),
                          "decided_fraction": float(dec["decided_fraction"][b]),
                          "zero_pairs_merged": float(dec["zero_pairs_merged"][b]),
                          "far_pairs_split": float(dec["far_pairs_split"][b]),
                          "false_merge": int(dec["false_merge"][b]), "false_split": int(dec["false_split"][b]),
                          "anchors": ""}
                if method != "bootstrap_band" and width[b] < radius:
                    centers = greedy_cover(hi[b], radius)
                    distortion = float(np.abs(embedding_distance(hi[b][:, centers])-truth).max())
                    assert distortion <= max(width[b], 2*radius)+2e-9
                    record["anchors"] = len(centers)
                rows.append(record)
        print("V8-V6", states, n, flush=True)
    return rows


def refine_v6():
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = [row for part in pool.map(_v6_job, SPEC["v6_states"][::-1]) for row in part]
    csv_write("v6_refined.csv", rows)
    summary = []
    for states in SPEC["v6_states"]:
        for method in SPEC["methods"]+["bootstrap_band"]:
            group = [r for r in rows if r["states"] == states and r["method"] == method]
            if not group:
                continue
            runs = {}
            for r in group:
                runs.setdefault(r["seed"], []).append(r["all_pairs_covered"])
            for n in SPEC["v6_checkpoints"]:
                cell = [r for r in group if r["n_per_row"] == n]
                anchors = [r["anchors"] for r in cell if r["anchors"] != ""]
                summary.append({"states": states, "method": method, "n_per_row": n,
                                "coverage": float(np.mean([r["all_pairs_covered"] for r in cell])),
                                "mean_sup_width": float(np.mean([r["sup_width"] for r in cell])),
                                "decided_fraction": float(np.mean([r["decided_fraction"] for r in cell])),
                                "zero_pairs_merged": float(np.mean([r["zero_pairs_merged"] for r in cell])),
                                "far_pairs_split": float(np.mean([r["far_pairs_split"] for r in cell])),
                                "runs_with_false_merge": sum(r["false_merge"] > 0 for r in cell),
                                "runs_with_false_split": sum(r["false_split"] > 0 for r in cell),
                                "anchor_runs": len(anchors),
                                "mean_anchors": float(np.mean(anchors)) if anchors else None})
            summary.append({"states": states, "method": method,
                            "whole_run_successes": sum(all(v) for v in runs.values()), "runs": len(runs)})
    return summary


def riverswim(states=6, gamma=.9):
    p, r = np.zeros((states, 2, states)), np.zeros((states, 2))
    for s in range(states):
        p[s, 0, max(s-1, 0)] = 1
        if s == 0:
            p[s, 1, 0], p[s, 1, 1] = .4, .6
        elif s == states-1:
            p[s, 1, s-1], p[s, 1, s] = .4, .6
        else:
            p[s, 1, s-1], p[s, 1, s], p[s, 1, s+1] = .05, .6, .35
    r[0, 0], r[states-1, 1] = .005, 1.
    return Model(r, p, gamma)


def multinomial_prefixes(model, seeds, checkpoints, tag):
    """Nested exact sufficient statistics for rows with up to three successors."""
    states, actions = model.reward.shape
    reward_sum = np.zeros((len(seeds), states, actions), dtype=np.int64)
    counts = np.zeros((len(seeds), states, actions, states), dtype=np.int64)
    reward_rng = [np.random.default_rng(np.random.SeedSequence([s, tag, 0])) for s in seeds]
    transition_rng = [np.random.default_rng(np.random.SeedSequence([s, tag, 1])) for s in seeds]
    previous = 0
    for n in checkpoints:
        inc = n-previous
        for b in range(len(seeds)):
            reward_sum[b] += reward_rng[b].binomial(inc, model.reward)
            for s in range(states):
                for a in range(actions):
                    counts[b, s, a] += transition_rng[b].multinomial(inc, model.transition[s, a])
        yield n, reward_sum/n, counts/n, reward_sum.copy(), counts.copy()
        previous = n


def riverswim_generative():
    cfg = SPEC["riverswim"]
    model = riverswim(cfg["states"], cfg["gamma"])
    gamma, seeds = cfg["gamma"], cfg["seeds"]
    truth = solve_refined(model.reward, model.reward, model.reward, model.transition, gamma,
                          np.zeros_like(model.reward), compute_majorant=False)["upper"][0]
    mask = ~np.eye(cfg["states"], dtype=bool)
    rows = []
    for n, rh, ph, reward_sum, counts in multinomial_prefixes(model, seeds, cfg["checkpoints"], 9):
        out = certificates(rh, ph, n, gamma, model.transition)
        replicates, level = SPEC["bootstrap_replicates"], SPEC["bootstrap_level"]
        plug = solve_refined(rh, rh, rh, ph, gamma, np.zeros_like(rh), compute_majorant=False)["upper"]
        lo_b, hi_b = np.zeros_like(plug), np.zeros_like(plug)
        for start in range(0, len(seeds), 2):
            stop = min(start+2, len(seeds))
            rs, ps = [], []
            for b in range(start, stop):
                rng = np.random.default_rng(np.random.SeedSequence([seeds[b], 9, n, 5]))
                rs.append(rng.binomial(n, rh[b], (replicates,)+rh[b].shape)/n)
                ps.append(np.stack([[[rng.multinomial(n, ph[b, s, a]) for a in range(2)]
                                     for s in range(cfg["states"])] for _ in range(replicates)])/n)
            rstar, pstar = np.concatenate(rs), np.concatenate(ps)
            boot = solve_refined(rstar, rstar, rstar, pstar, gamma, np.zeros_like(rstar),
                                 compute_majorant=False, tolerance=1e-10)["upper"]
            boot = boot.reshape(stop-start, replicates, cfg["states"], cfg["states"])
            crit = np.quantile(np.abs(boot-plug[start:stop, None]).max((2, 3)), level, axis=1)[:, None, None]
            lo_b[start:stop] = np.where(mask, np.maximum(0, plug[start:stop]-crit), 0)
            hi_b[start:stop] = np.where(mask, np.minimum(1, plug[start:stop]+crit), 0)
        out["bootstrap_band"] = {"lower": lo_b, "upper": hi_b}
        for method, r in out.items():
            lo, hi = r["lower"], r["upper"]
            ok = covered(lo, hi, truth)
            width = (hi-lo).max((1, 2))
            error = np.abs((lo+hi)/2-truth).max((1, 2))
            split = ((lo > 1e-9) & mask).sum((1, 2))/mask.sum()
            if method != "bootstrap_band":
                assert np.max(hi-lo-r["majorant"]) < 2e-8
            for b, seed in enumerate(seeds):
                rows.append({"seed": seed, "n_per_row": n, "method": method, "all_pairs_covered": int(ok[b]),
                             "sup_width": float(width[b]), "midpoint_sup_error": float(error[b]),
                             "certified_split_fraction": float(split[b])})
        print("V9 riverswim", n, {m: round(float(np.mean([r["sup_width"] for r in rows if r["n_per_row"] == n and r["method"] == m])), 4)
                                  for m in out}, flush=True)
    csv_write("riverswim_generative.csv", rows)
    summary = []
    for method in SPEC["methods"]+["bootstrap_band"]:
        group = [r for r in rows if r["method"] == method]
        runs = {}
        for r in group:
            runs.setdefault(r["seed"], []).append(r["all_pairs_covered"])
        for n in cfg["checkpoints"]:
            cell = [r for r in group if r["n_per_row"] == n]
            summary.append({"method": method, "n_per_row": n,
                            "coverage": float(np.mean([r["all_pairs_covered"] for r in cell])),
                            "mean_sup_width": float(np.mean([r["sup_width"] for r in cell])),
                            "mean_error": float(np.mean([r["midpoint_sup_error"] for r in cell])),
                            "certified_split_fraction": float(np.mean([r["certified_split_fraction"] for r in cell]))})
        whole = sum(all(v) for v in runs.values())
        summary.append({"method": method, "whole_run_successes": whole, "runs": len(runs),
                        "coverage_ci95": interval_probability(whole, len(runs))})
    return {"true_diameter": float(truth.max()), "cells": summary}


def task_model(task, gamma_riverswim):
    if task == "riverswim":
        return riverswim(6, gamma_riverswim), SPEC["priorities"]["riverswim_horizon"]
    return navigation_model(task)


def adjoint_scores(fit, transport, counts, gamma, target=None):
    """Share of the aggregate (or target-pair) majorant attributable to each row, per extra sample."""
    majorant, source = fit["majorant"][0], fit["source"][0]
    states, _, actions = source.shape
    weights = transport.coefficients(fit["lower_raw"])
    future = transport.apply(majorant, weights)[0]
    active = np.argmax(source+gamma*future, axis=-1)
    kernel = np.zeros((states*states, states*states))
    flows = weights
    if hasattr(transport, "i0"):
        left = [transport.i0[0], transport.i0[0], transport.i1[0], transport.i1[0]]
        right = [transport.i0[0], transport.i1[0], transport.i0[0], transport.i1[0]]
        for s in range(states):
            for t in range(states):
                if s == t:
                    continue
                a = active[s, t]
                for k in range(4):
                    x, y = left[k][s, a], right[k][t, a]
                    kernel[s*states+t, x*states+y] += flows[k][0, s, t, a]
    else:
        idx = transport.index[0]
        for s in range(states):
            for t in range(states):
                if s == t:
                    continue
                a = active[s, t]
                for e in range(9):
                    x, y = idx[s, a, e//3], idx[t, a, e % 3]
                    kernel[s*states+t, x*states+y] += flows[0, s, t, a, e]
    diag = [i*states+i for i in range(states)]
    kernel[:, diag] = 0
    if target is None:
        weight = (~np.eye(states, dtype=bool)).ravel().astype(float)
    else:
        weight = np.zeros(states*states)
        weight[target[0]*states+target[1]] = weight[target[1]*states+target[0]] = 1.
    weight /= weight.sum()
    occupancy = np.linalg.solve(np.eye(states*states)-gamma*kernel.T, weight).reshape(states, states)
    occupancy = occupancy+occupancy.T
    share = np.zeros((states, actions))
    for s in range(states):
        for t in range(states):
            if s != t:
                share[s, active[s, t]] += occupancy[s, t]*source[s, t, active[s, t]]/2
    return share/(counts+1)


def acquisition_run(task, seed, rule, budgets, quota, cap, refresh, horizon_override=None,
                    navigation="optimistic", gamma_riverswim=.6, record_widths=True, stream_tag=31):
    model, horizon = task_model(task, gamma_riverswim)
    horizon = horizon_override or horizon
    states, actions = model.reward.shape
    gamma_bound = int(np.count_nonzero(model.transition > 0, -1).max())
    task_id = ["corridor", "branching_return", "riverswim"].index(task)+stream_tag
    transition_rng = [[np.random.default_rng(np.random.SeedSequence([seed, task_id, s, a, 0]))
                       for a in range(actions)] for s in range(states)]
    reward_rng = [[np.random.default_rng(np.random.SeedSequence([seed, task_id, s, a, 1]))
                   for a in range(actions)] for s in range(states)]
    policy_rng = np.random.default_rng(np.random.SeedSequence([seed, task_id, 91]))
    counts = np.zeros((states, actions), dtype=int)
    outcomes = np.zeros((states, actions, states))
    reward_sum = np.zeros((states, actions))
    pending = list(budgets)
    snapshots, steps, episodes = {}, 0, 0
    priorities = np.ones((states, actions))

    def empirical():
        phat = np.eye(states)[:, None, :].repeat(actions, axis=1)
        np.divide(outcomes, np.maximum(counts, 1)[:, :, None], out=phat, where=(counts > 0)[:, :, None])
        return phat

    def snapshot():
        br, bp, _ = radii(counts, gamma_bound, SPEC["nominal_delta"])
        rh = reward_sum/np.maximum(counts, 1)
        fit = solve_refined(rh-br, rh+br, rh, empirical(), model.gamma, bp, compute_majorant=False)
        width = fit["upper"][0]-fit["lower"][0]
        return float(width.max()), float(width[~np.eye(states, dtype=bool)].mean())
    while counts.min() < quota and steps < cap:
        while record_widths and pending and steps >= pending[0]:
            snapshots[pending.pop(0)] = snapshot()
        phat = empirical()
        unknown = counts < quota
        if rule != "uniform" and episodes % refresh == 0:
            br, bp, _ = radii(counts, gamma_bound, SPEC["nominal_delta"])
            rh = reward_sum/np.maximum(counts, 1)
            fit = solve_refined(rh-br, rh+br, rh, phat, model.gamma, bp)
            source = fit["source"][0]
            if rule == "source":
                score = source.max(axis=1)
            elif rule == "propagated":
                transport = transport_for(phat)
                score = (source+model.gamma*transport.apply(
                    fit["majorant"], transport.coefficients(fit["lower_raw"]))[0]).max(axis=1)
            else:
                score = adjoint_scores(fit, transport_for(phat), counts, model.gamma)
            maximum = score[unknown].max()
            priorities = np.ones((states, actions))
            if maximum > 0:
                priorities[unknown] = .5+.5*score[unknown]/maximum
        policy = eager_reach_policy(phat, unknown, horizon, priorities)[0]
        state = 0
        episodes += 1
        for remaining in range(horizon, 0, -1):
            if navigation == "random":
                action = int(policy_rng.integers(actions))
            else:
                action = int(policy[remaining, state])
            value = transition_rng[state][action].random()
            successor = int(np.searchsorted(np.cumsum(model.transition[state, action]), value, side="right"))
            steps += 1
            if unknown[state, action]:
                counts[state, action] += 1
                outcomes[state, action, successor] += 1
                reward_sum[state, action] += reward_rng[state][action].binomial(1, model.reward[state, action])
                break
            state = successor
            if steps >= cap:
                break
    completed = int(counts.min() >= quota)
    record = {"task": task, "seed": seed, "rule": rule, "navigation": navigation,
              "steps": steps, "completed": completed, "retained": int(counts.sum()),
              "empirical_model_hash": hashlib.sha256(outcomes.tobytes()+reward_sum.tobytes()).hexdigest()}
    if record_widths:
        final = snapshot()
        for budget in budgets:
            snap = snapshots.get(budget, final)
            record[f"sup_width_at_{budget}"], record[f"mean_width_at_{budget}"] = snap
    return record


def _nav_job(job):
    seed, rule = job
    cfg = SPEC["navigation"]
    return acquisition_run("riverswim", seed, "uniform", [], cfg["quota"], cfg["cap"], 10**9,
                           horizon_override=cfg["horizon"], navigation=rule,
                           gamma_riverswim=cfg["gamma"], record_widths=False, stream_tag=41)


def riverswim_navigation():
    cfg = SPEC["navigation"]
    jobs = [(seed, rule) for rule in cfg["rules"] for seed in cfg["seeds"]]
    with ProcessPoolExecutor(max_workers=7) as pool:
        rows = list(pool.map(_nav_job, jobs))
    csv_write("riverswim_navigation.csv", rows)
    out = []
    for rule in cfg["rules"]:
        group = [r for r in rows if r["navigation"] == rule]
        out.append({"navigation": rule, "completed": sum(r["completed"] for r in group), "runs": len(group),
                    "retained_samples": bootstrap_mean([r["retained"] for r in group]),
                    **bootstrap_mean([r["steps"] for r in group])})
        print("V9 navigation", out[-1], flush=True)
    return out


def _priority_job(job):
    task, seed, rule = job
    cfg = SPEC["priorities"]
    return acquisition_run(task, seed, rule, cfg["budgets"][task], cfg["quota"], cfg["cap"], cfg["refresh"],
                           gamma_riverswim=cfg["riverswim_gamma"])


def priorities():
    cfg = SPEC["priorities"]
    jobs = [(task, seed, rule) for task in cfg["tasks"] for rule in cfg["rules"] for seed in cfg["seeds"]]
    partial = OUT/"priorities_partial.jsonl"
    done = {}
    if partial.exists():
        for line in partial.read_text(encoding="utf8").splitlines():
            r = json.loads(line)
            done[(r["task"], r["seed"], r["rule"])] = r
    todo = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    # Append each finished run so that an interrupted study can resume.
    with ProcessPoolExecutor(max_workers=7) as pool, partial.open("a", encoding="utf8") as stream:
        for r in pool.map(_priority_job, todo):
            done[(r["task"], r["seed"], r["rule"])] = r
            stream.write(json.dumps(r)+"\n")
            stream.flush()
    results = [done[(j[0], j[1], j[2])] for j in jobs]
    csv_write("priorities.csv", results)
    by = {(r["task"], r["seed"], r["rule"]): r for r in results}
    groups, contrasts = [], []
    for task in cfg["tasks"]:
        endpoints = [f"mean_width_at_{b}" for b in cfg["budgets"][task]]+["steps"]
        for rule in cfg["rules"]:
            group = [by[(task, s, rule)] for s in cfg["seeds"]]
            groups.append({"task": task, "rule": rule, "completed": sum(r["completed"] for r in group),
                           "runs": len(group), **{e: bootstrap_mean([r[e] for r in group]) for e in endpoints}})
        for first, second in [("adjoint", "source"), ("adjoint", "uniform"), ("source", "uniform"),
                              ("propagated", "source")]:
            for e in endpoints:
                diff = [by[(task, s, first)][e]-by[(task, s, second)][e] for s in cfg["seeds"]]
                contrasts.append({"task": task, "contrast": f"{first} minus {second}", "endpoint": e,
                                  **bootstrap_mean(diff)})
        for seed in cfg["seeds"]:
            done = [by[(task, seed, rule)] for rule in cfg["rules"] if by[(task, seed, rule)]["completed"]]
            assert len({r["empirical_model_hash"] for r in done}) <= 1
        print("V10", task, [(g["rule"], round(g["steps"]["mean"]), round(g[endpoints[-2]]["mean"], 3))
                            for g in groups if g["task"] == task], flush=True)
    return {"groups": groups, "contrasts": contrasts}


def targeted_run(task, seed, rule):
    """Sample until the target pair is certified distinct (lower endpoint above zero)."""
    cfg = SPEC["targeted"]
    target = cfg["tasks"][task]
    model, horizon = task_model(task, cfg["gamma"])
    if task == "riverswim":
        horizon = cfg["riverswim_horizon"]
    states, actions = model.reward.shape
    gamma_bound = int(np.count_nonzero(model.transition > 0, -1).max())
    task_id = ["corridor", "branching_return", "riverswim"].index(task)+51
    transition_rng = [[np.random.default_rng(np.random.SeedSequence([seed, task_id, s, a, 0]))
                       for a in range(actions)] for s in range(states)]
    reward_rng = [[np.random.default_rng(np.random.SeedSequence([seed, task_id, s, a, 1]))
                   for a in range(actions)] for s in range(states)]
    counts = np.zeros((states, actions), dtype=int)
    outcomes = np.zeros((states, actions, states))
    reward_sum = np.zeros((states, actions))
    steps = episodes = 0
    priorities = np.ones((states, actions))
    samplable = np.ones((states, actions), dtype=bool)
    certified = False

    def plan(phat):
        """Stop-or-continue recursion: sampling a row pays v; a traversable row may be crossed."""
        value = np.zeros(states)
        choice = np.zeros((horizon+1, states), dtype=int)
        stop = np.zeros((horizon+1, states), dtype=bool)
        cross = counts >= cfg["traversable_after"]
        for h in range(1, horizon+1):
            go = np.where(cross, np.einsum("sak,k->sa", phat, value), -np.inf)
            pay = np.where(samplable, priorities, -np.inf)
            q = np.maximum(pay, go)
            choice[h] = q.argmax(1)
            best = q[np.arange(states), choice[h]]
            stop[h] = pay[np.arange(states), choice[h]] >= go[np.arange(states), choice[h]]-1e-12
            value = best
        return choice, stop
    while steps < cfg["cap"]:
        phat = np.eye(states)[:, None, :].repeat(actions, axis=1)
        np.divide(outcomes, np.maximum(counts, 1)[:, :, None], out=phat, where=(counts > 0)[:, :, None])
        br, bp, _ = radii(counts, gamma_bound, SPEC["nominal_delta"])
        rh = reward_sum/np.maximum(counts, 1)
        if episodes % cfg["check_every"] == 0:
            fit = solve_refined(rh-br, rh+br, rh, phat, model.gamma, bp, compute_majorant=False)
            if fit["lower"][0, target[0], target[1]] > 1e-9:
                certified = True
                break
        if rule != "uniform" and episodes % cfg["refresh"] == 0:
            fit = solve_refined(rh-br, rh+br, rh, phat, model.gamma, bp)
            radius = (1-model.gamma)*br+model.gamma*bp
            tie = 1e-3*radius/radius.max()
            if rule == "count":
                priorities = .5+.5*radius/radius.max()
            else:
                score = adjoint_scores(fit, transport_for(phat), counts, model.gamma, target)
                relevance = score/score.max() if score.max() > 0 else np.zeros_like(score)
                priorities = .5+.5*relevance+tie
                # Rows with no influence on the target retire once they can be traversed.
                samplable = (counts < cfg["traversable_after"]) | (relevance > 1e-9)
        choice, stop = plan(phat)
        state = 0
        episodes += 1
        for remaining in range(horizon, 0, -1):
            action = int(choice[remaining, state])
            value = transition_rng[state][action].random()
            successor = int(np.searchsorted(np.cumsum(model.transition[state, action]), value, side="right"))
            steps += 1
            counts[state, action] += 1
            outcomes[state, action, successor] += 1
            reward_sum[state, action] += reward_rng[state][action].binomial(1, model.reward[state, action])
            if stop[remaining, state]:
                break
            state = successor
    return {"task": task, "seed": seed, "rule": rule, "certified": int(certified), "steps": steps,
            "episodes": episodes, "retained": int(counts.sum()),
            "target_row_samples": int(counts[target[0]].sum()+counts[target[1]].sum())}


def _targeted_job(job):
    return targeted_run(*job)


def targeted():
    cfg = SPEC["targeted"]
    jobs = [(task, seed, rule) for task in cfg["tasks"] for rule in cfg["rules"] for seed in cfg["seeds"]]
    with ProcessPoolExecutor(max_workers=7) as pool:
        rows = list(pool.map(_targeted_job, jobs))
    csv_write("targeted.csv", rows)
    by = {(r["task"], r["seed"], r["rule"]): r for r in rows}
    groups, contrasts = [], []
    for task in cfg["tasks"]:
        for rule in cfg["rules"]:
            group = [by[(task, s, rule)] for s in cfg["seeds"]]
            groups.append({"task": task, "rule": rule, "certified": sum(r["certified"] for r in group),
                           "runs": len(group), "steps": bootstrap_mean([r["steps"] for r in group]),
                           "target_share": bootstrap_mean([r["target_row_samples"]/max(r["retained"], 1) for r in group])})
        for first, second in [("adjoint_target", "count")]:
            diff = [by[(task, s, first)]["steps"]-by[(task, s, second)]["steps"] for s in cfg["seeds"]]
            contrasts.append({"task": task, "contrast": f"{first} minus {second}", **bootstrap_mean(diff)})
        print("V10-targeted", task, [(g["rule"], g["certified"], round(g["steps"]["mean"]))
                                     for g in groups if g["task"] == task], flush=True)
    return {"groups": groups, "contrasts": contrasts}


def main():
    parser = argparse.ArgumentParser()
    parts = {"v1": refine_v1, "v6": refine_v6, "riverswim": riverswim_generative,
             "navigation": riverswim_navigation, "priorities": priorities, "targeted": targeted}
    parser.add_argument("--part", choices=list(parts)+["manifest"], required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    spec_path = OUT/"experiment_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != SPEC:
        raise RuntimeError("The frozen specification differs; use a new version/output folder")
    spec_path.write_text(json.dumps(SPEC, indent=2)+"\n", encoding="utf8")
    started = time.time()
    if args.part in parts:
        result = {args.part: parts[args.part](), "elapsed_seconds": time.time()-started}
        (OUT/f"summary_{args.part}.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf8")
    manifest = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                "platform": platform.platform(),
                "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                [Path(__file__), ROOT/"code/finite_general.py", ROOT/"code/finite_batch.py",
                                 ROOT/"code/ebg.py", ROOT/"code/run_extended_study.py",
                                 ROOT/"code/run_verified_study.py"]},
                "frozen_inputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(FROZEN.glob("calibration_*.npz"))},
                "result_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(OUT.glob("*.csv"))+sorted(OUT.glob("summary_*.json"))}}
    (OUT/"run_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf8")
    print("Completed", args.part, "in", round(time.time()-started, 1), "seconds", flush=True)


if __name__ == "__main__":
    main()
