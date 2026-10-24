"""V5 (interval baselines), V6 (scaling), and V7 (anytime priorities)."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import time

import numpy as np
import scipy
from concurrent.futures import ProcessPoolExecutor

from ebg import Model, confidence_radii, reach_policy, greedy_cover, embedding_distance
from finite_batch import solve, PairTransport
from run_verified_study import (ring, sampled_prefixes, interval_probability,
                                bootstrap_mean, navigation_model)

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT/"results"/"verified"
OUT = ROOT/"results"/"extended"
SPEC = {
    "version": "extended-study-v1", "nominal_delta": .1,
    "baseline_models": [[6, .6], [8, .9]],
    "baseline_checkpoints_per_row": [128, 512, 2048, 8192, 32768, 131072],
    "baseline_seeds": list(range(10000, 10500)),
    "baseline_bootstrap_replicates": 200,
    "baseline_bootstrap_level": .9,
    "scaling_states": [16, 32, 64, 128], "scaling_blocks": 4, "scaling_gamma": .6,
    "scaling_seeds": list(range(80000, 80100)),
    "scaling_checkpoints_per_row": [2048, 8192, 32768, 131072],
    "scaling_anchor_radius": .2, "scaling_timing_repeats": 5,
    "anytime_tasks": ["corridor", "branching_return"],
    "anytime_priorities": ["uniform", "local", "propagated"],
    "anytime_seeds": list(range(61000, 61032)),
    "anytime_retained_quota": 4096, "anytime_refresh_episodes": 512,
    "anytime_step_budgets": [5000, 10000, 20000, 40000, 80000],
    "anytime_step_cap": 1000000, "bootstrap_resamples": 5000,
    "numeric_tolerance": 2e-11,
    "scope": "Finite-state generative and reset-interaction studies; no image benchmark",
}


def csv_write(name, rows):
    with (OUT/name).open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def off_diagonal(states):
    return ~np.eye(states, dtype=bool)


def covered(lower, upper, truth):
    return ((lower <= truth+2e-9) & (truth <= upper+2e-9)).all((-2, -1))


def _bootstrap_chunk(job):
    """Bootstrap replicate metrics for seeds start..stop of one V1 checkpoint."""
    model_id, n, start, stop = job
    states, gamma = SPEC["baseline_models"][model_id]
    level, replicates = SPEC["baseline_bootstrap_level"], SPEC["baseline_bootstrap_replicates"]
    model = ring(states, gamma)
    indices = np.argsort(-model.transition, axis=-1, kind="stable")[..., :2]
    archive = np.load(FROZEN/f"calibration_{model_id}_n{n}.npz")
    rh, ph = archive["reward_estimates"], archive["transition_estimates"]
    first = archive["successor_first_counts"]/n
    # The full-batch plug-in solve matches the centre used by the caller bit for bit.
    plug = solve(rh, ph, gamma)["truth"][start:stop]
    draws_r, draws_p = [], []
    for b in range(start, stop):
        rng = np.random.default_rng(np.random.SeedSequence([int(archive["seeds"][b]), model_id, n, 5]))
        draws_r.append(rng.binomial(n, rh[b], (replicates, states, 2))/n)
        draws_p.append(rng.binomial(n, first[b], (replicates, states, 2))/n)
    rstar = np.concatenate(draws_r)
    pstar = np.concatenate(draws_p)
    phat = np.zeros(pstar.shape+(states,))
    for s in range(states):
        for a in range(2):
            phat[:, s, a, indices[s, a, 0]] = pstar[:, s, a]
            phat[:, s, a, indices[s, a, 1]] += 1-pstar[:, s, a]
    boot = solve(rstar, phat, gamma, tolerance=1e-10)["truth"]
    boot = boot.reshape(stop-start, replicates, states, states)
    tail = (1-level)/2
    deviation = np.abs(boot-plug[:, None]).max((2, 3))
    critical = np.quantile(deviation, level, axis=1)[:, None, None]
    return (start, stop, np.quantile(boot, tail, axis=1), np.quantile(boot, 1-tail, axis=1), critical)


def baselines():
    """Certified, global, and bootstrap intervals on identical V1 samples."""
    rows, summaries = [], []
    level = SPEC["baseline_bootstrap_level"]
    replicates = SPEC["baseline_bootstrap_replicates"]
    for model_id, (states, gamma) in enumerate(SPEC["baseline_models"]):
        model = ring(states, gamma)
        indices = np.argsort(-model.transition, axis=-1, kind="stable")[..., :2]
        mask = off_diagonal(states)
        whole = {name: np.ones(len(SPEC["baseline_seeds"]), dtype=bool)
                 for name in ["certified", "global", "bootstrap_pointwise", "bootstrap_band"]}
        for n in SPEC["baseline_checkpoints_per_row"]:
            archive = np.load(FROZEN/f"calibration_{model_id}_n{n}.npz")
            assert list(archive["seeds"]) == SPEC["baseline_seeds"]
            truth = archive["true_metric"]
            rh, ph = archive["reward_estimates"], archive["transition_estimates"]
            first = archive["successor_first_counts"]/n
            plug = solve(rh, ph, gamma)["truth"]
            br, bp = confidence_radii(np.full((states, 2), n), 2, SPEC["nominal_delta"])
            source = float(np.max((1-gamma)*2*br+gamma*2*bp))
            radius = min(1., source/(1-gamma))
            global_lo = np.where(mask, np.maximum(0, plug-radius), 0)
            global_hi = np.where(mask, np.minimum(1, plug+radius), 0)
            point_lo, point_hi = np.zeros_like(plug), np.zeros_like(plug)
            band_lo, band_hi = np.zeros_like(plug), np.zeros_like(plug)
            chunk = 25
            jobs = [(model_id, n, start, min(start+chunk, len(plug)))
                    for start in range(0, len(plug), chunk)]
            # Chunks keep the batch composition of a serial run, so results do not depend on workers.
            with ProcessPoolExecutor(max_workers=7) as pool:
                for start, stop, lo_q, hi_q, critical in pool.map(_bootstrap_chunk, jobs):
                    point_lo[start:stop], point_hi[start:stop] = lo_q, hi_q
                    band_lo[start:stop] = np.where(mask, np.maximum(0, plug[start:stop]-critical), 0)
                    band_hi[start:stop] = np.where(mask, np.minimum(1, plug[start:stop]+critical), 0)
            methods = {"certified": (archive["lower"], archive["upper"]),
                       "global": (global_lo, global_hi),
                       "bootstrap_pointwise": (point_lo, point_hi),
                       "bootstrap_band": (band_lo, band_hi)}
            plug_error = np.abs(plug-truth).max((1, 2))
            for name, (lo, hi) in methods.items():
                ok = covered(lo, hi, truth)
                whole[name] &= ok
                width = (hi-lo).max((1, 2))
                mean_width = ((hi-lo)*mask).sum((1, 2))/mask.sum()
                for b, seed in enumerate(SPEC["baseline_seeds"]):
                    rows.append({"model": f"ring{states}_g{gamma}", "seed": seed, "n_per_row": n,
                                 "method": name, "all_pairs_covered": int(ok[b]),
                                 "sup_width": float(width[b]), "mean_pair_width": float(mean_width[b]),
                                 "plug_in_sup_error": float(plug_error[b])})
                summaries.append({"model": f"ring{states}_g{gamma}", "n_per_row": n, "method": name,
                                  "coverage": float(ok.mean()), "covered_runs": int(ok.sum()),
                                  "coverage_ci95": interval_probability(int(ok.sum()), len(ok)),
                                  "mean_sup_width": float(width.mean()),
                                  "mean_pair_width": float(mean_width.mean())})
                print(f"V5 ring{states} n={n} {name}: coverage={ok.mean():.3f} "
                      f"width={width.mean():.4f}", flush=True)
            summaries.append({"model": f"ring{states}_g{gamma}", "n_per_row": n,
                              "method": "plug_in_point", "mean_sup_error": float(plug_error.mean()),
                              "global_radius": radius})
        for name, ok in whole.items():
            summaries.append({"model": f"ring{states}_g{gamma}", "method": name,
                              "whole_run_successes": int(ok.sum()), "runs": len(ok),
                              "coverage_ci95": interval_probability(int(ok.sum()), len(ok))})
    csv_write("baselines.csv", rows)
    return summaries


LATENT_REWARD = [[.1+.25*k, .85-.25*((3*k) % 4)] for k in range(4)]
LATENT_SUCCESSORS = [[(k, .7, (k+1) % 4), ((k+2) % 4, .6, (k+3) % 4)] for k in range(4)]


def clustered_model(states, gamma):
    """Exactly lumpable random MDP with four declared latent blocks."""
    blocks = SPEC["scaling_blocks"]
    assert states % blocks == 0
    block = np.arange(states)*blocks//states
    members = [np.flatnonzero(block == k) for k in range(blocks)]
    rng = np.random.default_rng(np.random.SeedSequence([states, 7]))
    reward = np.zeros((states, 2))
    transition = np.zeros((states, 2, states))
    for s in range(states):
        for a, (k1, p1, k2) in enumerate(LATENT_SUCCESSORS[block[s]]):
            transition[s, a, rng.choice(members[k1])] += p1
            transition[s, a, rng.choice(members[k2])] += 1-p1
            reward[s, a] = LATENT_REWARD[block[s]][a]
    return Model(reward, transition, gamma), block


def scaling():
    rows, summaries = [], []
    gamma, radius = SPEC["scaling_gamma"], SPEC["scaling_anchor_radius"]
    seeds = SPEC["scaling_seeds"]
    final = SPEC["scaling_checkpoints_per_row"][-1]
    for states in SPEC["scaling_states"]:
        model, block = clustered_model(states, gamma)
        truth = solve(model.reward, model.transition, gamma)["truth"][0]
        same = block[:, None] == block[None, :]
        assert np.max(truth[same]) < 1e-9
        separation = float(truth[~same].min())
        mask = off_diagonal(states)
        whole = np.ones(len(seeds), dtype=bool)
        for n, rh, ph, _, _ in sampled_prefixes(model, seeds, SPEC["scaling_checkpoints_per_row"],
                                               100+states):
            br, bp = confidence_radii(np.full((states, 2), n), 2, SPEC["nominal_delta"])
            computed = solve(rh, ph, gamma, br=br, bp=bp)
            lo, hi, majorant = computed["lower"], computed["upper"], computed["majorant"]
            ok = covered(lo, hi, truth)
            whole &= ok
            width = (hi-lo).max((1, 2))
            mean_width = ((hi-lo)*mask).sum((1, 2))/mask.sum()
            error = np.abs((lo+hi)/2-truth).max((1, 2))
            violation = np.maximum(hi-lo-majorant, 0).max((1, 2))
            assert np.max(violation) < 2e-8
            for b, seed in enumerate(seeds):
                triangle = float(np.maximum(hi[b][:, :, None]-hi[b][:, None, :]-hi[b][None, :, :], 0).max())
                assert triangle < 2e-8
                record = {"states": states, "seed": seed, "n_per_row": n, "all_pairs_covered": int(ok[b]),
                          "sup_width": float(width[b]), "mean_pair_width": float(mean_width[b]),
                          "midpoint_sup_error": float(error[b]), "majorant_violation": float(violation[b]),
                          "upper_triangle_violation": triangle, "anchors": "", "anchor_distortion": "",
                          "cover_radius": ""}
                if n == final:
                    assert width[b] < radius
                    centers = greedy_cover(hi[b], radius)
                    distance = embedding_distance(hi[b][:, centers])
                    distortion = float(np.abs(distance-truth).max())
                    cover = float(np.min(hi[b][:, centers], axis=1).max())
                    assert cover <= radius+2e-9 and distortion <= max(width[b], 2*radius)+2e-9
                    assert len(centers) <= SPEC["scaling_blocks"]
                    record.update(anchors=len(centers), anchor_distortion=distortion, cover_radius=cover)
                rows.append(record)
            summary = {"states": states, "n_per_row": n, "coverage": float(ok.mean()),
                       "mean_sup_width": float(width.mean()), "mean_pair_width": float(mean_width.mean()),
                       "mean_error": float(error.mean()), "true_block_separation": separation}
            if n == final:
                final_rows = [r for r in rows if r["states"] == states and r["n_per_row"] == n]
                summary.update(mean_anchors=float(np.mean([r["anchors"] for r in final_rows])),
                               max_anchors=int(max(r["anchors"] for r in final_rows)),
                               max_anchor_distortion=float(max(r["anchor_distortion"] for r in final_rows)))
                timings = []
                for b in range(SPEC["scaling_timing_repeats"]):
                    started = time.perf_counter()
                    solve(rh[b], ph[b], gamma, br=br, bp=bp)
                    timings.append(time.perf_counter()-started)
                summary.update(median_certificate_seconds=float(np.median(timings)),
                               certificate_seconds=timings)
            summaries.append(summary)
            print(f"V6 S={states} n={n}: coverage={ok.mean():.3f} width={width.mean():.4f} "
                  f"pair={mean_width.mean():.4f} error={error.mean():.4f}", flush=True)
        summaries.append({"states": states, "whole_run_successes": int(whole.sum()), "runs": len(seeds),
                          "coverage_ci95": interval_probability(int(whole.sum()), len(seeds))})
    csv_write("scaling.csv", rows)
    return summaries


def snapshot_width(model, counts, outcomes, reward_sum):
    states, actions = counts.shape
    phat = np.eye(states)[:, None, :].repeat(actions, axis=1)
    np.divide(outcomes, np.maximum(counts, 1)[:, :, None], out=phat, where=(counts > 0)[:, :, None])
    br, bp = confidence_radii(counts, 2, SPEC["nominal_delta"])
    fit = solve(reward_sum/np.maximum(counts, 1), phat, model.gamma, br=br, bp=bp,
                compute_majorant=False)
    width = fit["upper"][0]-fit["lower"][0]
    mask = off_diagonal(states)
    return float(width.max()), float(width[mask].mean())


# Post hoc diagnostic, added after V7 was analyzed; kept out of the frozen SPEC.
POSTHOC = {"version": "anytime-posthoc-v1",
           "reason": "V7 cost differences were traced to ties in the stopped planner; "
                     "this diagnostic breaks ties toward executing an unresolved row",
           "tie_tolerance": 1e-12, "priorities": ["uniform", "local", "propagated"]}


def eager_reach_policy(transition, unknown, horizon, priorities):
    """The planner of eq. (44), with ties broken toward executing an unresolved row."""
    s, a, _ = transition.shape
    value = np.zeros(s)
    policies = np.zeros((horizon+1, s), dtype=int)
    for h in range(1, horizon+1):
        q = np.einsum("sak,k->sa", transition, value)
        q[unknown] = priorities[unknown]
        best = q.max(axis=1, keepdims=True)
        tied = q >= best-POSTHOC["tie_tolerance"]
        preferred = tied & unknown
        choice = np.where(preferred.any(axis=1), preferred.argmax(axis=1), tied.argmax(axis=1))
        policies[h] = choice
        value = q[np.arange(s), choice]
    return policies, value


def anytime_run(task, seed, priority_mode, eager=False):
    model, horizon = navigation_model(task)
    states, actions = model.reward.shape
    task_id = SPEC["anytime_tasks"].index(task)+11
    transition_rng = [[np.random.default_rng(np.random.SeedSequence([seed, task_id, s, a, 0]))
                       for a in range(actions)] for s in range(states)]
    reward_rng = [[np.random.default_rng(np.random.SeedSequence([seed, task_id, s, a, 1]))
                   for a in range(actions)] for s in range(states)]
    counts = np.zeros((states, actions), dtype=int)
    outcomes = np.zeros((states, actions, states))
    reward_sum = np.zeros((states, actions))
    quota, cap = SPEC["anytime_retained_quota"], SPEC["anytime_step_cap"]
    budgets = list(SPEC["anytime_step_budgets"])
    snapshots = {}
    steps = episodes = 0
    priorities = np.ones((states, actions))
    while counts.min() < quota and steps < cap:
        while budgets and steps >= budgets[0]:
            snapshots[budgets.pop(0)] = snapshot_width(model, counts, outcomes, reward_sum)
        phat = np.eye(states)[:, None, :].repeat(actions, axis=1)
        np.divide(outcomes, np.maximum(counts, 1)[:, :, None], out=phat,
                  where=(counts > 0)[:, :, None])
        unknown = counts < quota
        if priority_mode != "uniform" and episodes % SPEC["anytime_refresh_episodes"] == 0:
            br, bp = confidence_radii(counts, 2, SPEC["nominal_delta"])
            fit = solve(reward_sum/np.maximum(counts, 1), phat, model.gamma, br=br, bp=bp)
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
        planner = eager_reach_policy if eager else reach_policy
        policy = planner(phat, unknown, horizon, priorities)[0]
        state = 0
        episodes += 1
        for remaining in range(horizon, 0, -1):
            action = int(policy[remaining, state])
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
            if steps >= cap:
                break
    completed = int(counts.min() >= quota)
    final = snapshot_width(model, counts, outcomes, reward_sum)
    for budget in budgets:
        snapshots[budget] = final
    record = {"task": task, "seed": seed, "priority": priority_mode, "steps_to_quota": steps,
              "completed": completed, "final_sup_width": final[0], "final_mean_width": final[1],
              "empirical_model_hash": hashlib.sha256(outcomes.tobytes()+reward_sum.tobytes()).hexdigest()}
    for budget in SPEC["anytime_step_budgets"]:
        record[f"sup_width_at_{budget}"] = snapshots[budget][0]
        record[f"mean_width_at_{budget}"] = snapshots[budget][1]
    return record


def _anytime_job(job):
    return anytime_run(*job)


def anytime(eager=False):
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(task, seed, priority, eager) for task in SPEC["anytime_tasks"]
            for priority in SPEC["anytime_priorities"] for seed in SPEC["anytime_seeds"]]
    with ProcessPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(_anytime_job, jobs))
    finished = {(r["task"], r["seed"], r["priority"]): r for r in results}
    rows, summaries, contrasts = [], [], []
    for task in SPEC["anytime_tasks"]:
        for priority in SPEC["anytime_priorities"]:
            group = [finished[(task, seed, priority)] for seed in SPEC["anytime_seeds"]]
            rows.extend(group)
            summary = {"task": task, "priority": priority,
                       "completed": sum(r["completed"] for r in group), "runs": len(group),
                       "steps_to_quota": bootstrap_mean([r["steps_to_quota"] for r in group])}
            for budget in SPEC["anytime_step_budgets"]:
                summary[f"mean_width_at_{budget}"] = bootstrap_mean([r[f"mean_width_at_{budget}"] for r in group])
                summary[f"sup_width_at_{budget}"] = bootstrap_mean([r[f"sup_width_at_{budget}"] for r in group])
            summaries.append(summary)
            print("V7", task, priority, {k: (v["mean"] if isinstance(v, dict) else v)
                                         for k, v in summary.items()}, flush=True)
        by = {(r["priority"], r["seed"]): r for r in rows if r["task"] == task}
        endpoints = [f"mean_width_at_{budget}" for budget in SPEC["anytime_step_budgets"]]
        for first, second in [("local", "uniform"), ("propagated", "uniform"), ("propagated", "local")]:
            for endpoint in endpoints+["steps_to_quota"]:
                differences = [by[(first, s)][endpoint]-by[(second, s)][endpoint]
                               for s in SPEC["anytime_seeds"]]
                contrasts.append({"task": task, "endpoint": endpoint,
                                  "contrast": f"{first} minus {second}", **bootstrap_mean(differences)})
        for seed in SPEC["anytime_seeds"]:
            complete = [r for r in rows if r["task"] == task and r["seed"] == seed and r["completed"]]
            assert len({r["empirical_model_hash"] for r in complete}) <= 1
    csv_write("anytime_posthoc.csv" if eager else "anytime.csv", rows)
    return {"groups": summaries, "paired_contrasts": contrasts}


def anytime_posthoc():
    (OUT/"posthoc_spec.json").write_text(json.dumps(POSTHOC, indent=2)+"\n", encoding="utf8")
    return anytime(eager=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["all", "baselines", "scaling", "anytime",
                                           "anytime_posthoc", "manifest"],
                        default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    spec_path = OUT/"experiment_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != SPEC:
        raise RuntimeError("The frozen specification differs; use a new version/output folder")
    spec_path.write_text(json.dumps(SPEC, indent=2)+"\n", encoding="utf8")
    started = time.time()
    for name, runner in [("baselines", baselines), ("scaling", scaling), ("anytime", anytime),
                         ("anytime_posthoc", anytime_posthoc)]:
        if args.part == name or (args.part == "all" and name != "anytime_posthoc"):
            part_start = time.time()
            result = {name: runner(), "elapsed_seconds": time.time()-part_start}
            (OUT/f"summary_{name}.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf8")
    manifest = {"python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "platform": platform.platform(),
        "processor": platform.processor(),
        "elapsed_seconds_this_invocation": time.time()-started,
        "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [Path(__file__), ROOT/"code/finite_batch.py", ROOT/"code/ebg.py",
                         ROOT/"code/run_verified_study.py"]},
        "frozen_inputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(FROZEN.glob("calibration_*.npz"))},
        "result_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(OUT.glob("*.csv"))+sorted(OUT.glob("summary_*.json"))
                         +sorted(OUT.glob("posthoc_spec.json"))}}
    (OUT/"run_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf8")
    print("Completed", args.part, "in", round(time.time()-started, 2), "seconds", flush=True)


if __name__ == "__main__":
    main()
