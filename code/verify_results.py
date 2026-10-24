"""Checks the proof constructions, the solvers, and every stored record against its manifest."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import os
import platform
import time

import numpy as np
from ebg import (Model, transport, metric_fixed_point, confidence_radii, intervals,
                 pair_majorant, certificate_quota, learn_with_resets, goal_policy,
                 evaluate_goal_policy)
from finite_batch import PairTransport, solve
import theory_checks as original

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"results"/"verified"
PROVENANCE = json.loads((ROOT/"results"/"code_provenance.json").read_text())
CHECKS = []


def logic_fingerprint(path):
    """Hash of the syntax tree without docstrings; comments never reach the tree."""
    tree = ast.parse(path.read_text(encoding="utf8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return hashlib.sha256(ast.dump(tree).encode("utf8")).hexdigest()


def check(name, function):
    function()
    CHECKS.append(name)
    print("PASS", name, flush=True)


def independent_transport():
    rng = np.random.default_rng(991)
    for gamma in [.2, .8]:
        s, a = 4, 2
        p = np.zeros((s, a, s))
        for i in range(s):
            for act in range(a):
                support = rng.choice(s, 2, replace=False)
                p[i, act, support] = rng.dirichlet(np.ones(2))
        r = rng.uniform(.1, .9, (s, a))
        model = Model(r, p, gamma)
        h = rng.uniform(0, 1, (s, s))
        h = (h+h.T)/2
        np.fill_diagonal(h, 0)
        batched = PairTransport(p).apply(h)[0]
        for i in range(s):
            for j in range(s):
                for act in range(a):
                    scalar = transport(p[i, act], p[j, act], h)
                    assert abs(scalar-batched[i, j, act]) < 1e-10
        reference = metric_fixed_point(model)[0]
        assert np.max(abs(reference-solve(r, p, gamma)["truth"][0])) < 2e-9
        br = np.full_like(r, .015)
        bp = np.full_like(r, .01)
        lo, hi, _ = intervals(model, br, bp)
        result = solve(r, p, gamma, br=br, bp=bp)
        assert np.max(abs(lo-result["lower"][0])) < 2e-9
        assert np.max(abs(hi-result["upper"][0])) < 2e-9


def bounded_majorant():
    gamma, eta = .9, .05
    p = np.eye(4)[:, None, :]
    p[0, 0] = [1-eta, 0, eta, 0]
    p[1, 0] = [0, 1-eta, 0, eta]
    reward = np.array([[.1], [.1], [.3], [.7]])
    br = np.array([[0.], [0.], [1.], [1.]])
    bp = np.zeros_like(br)
    model = Model(reward, p, gamma)
    result = solve(reward, p, gamma, br=br, bp=bp)
    operator = PairTransport(p)
    weights = operator.coefficients(result["lower_raw"])
    unbounded = np.zeros((1, 4, 4))
    for _ in range(1000):
        updated = np.max(2*result["bonus"]+gamma*operator.apply(unbounded, weights), axis=-1)
        updated[:, np.arange(4), np.arange(4)] = 0
        if np.max(abs(updated-unbounded)) < 1e-12:
            unbounded = updated
            break
        unbounded = updated
    unbounded = unbounded[0]
    bounded = result["majorant"][0]
    width = result["upper"][0]-result["lower"][0]
    assert np.max(width-bounded) < 2e-8
    assert np.max(bounded-np.minimum(1., unbounded)) < 2e-8
    assert bounded[0, 1] < .8 and min(1., unbounded[0, 1]) > .9


def quota_and_horizon():
    parameters = (3, 2, 2, .6, 1., .2, 6, .5, .1)
    quota, alpha = certificate_quota(*parameters)
    for n, expected in [(quota, True), (quota-1, False)]:
        term = (2*np.log(2)+np.log(8*3*2*np.pi**2/(3*.1))+2*np.log(n))/(2*n)
        assert bool(term <= alpha**2) == expected
    p = np.array([[[0., 1.]], [[0., 1.]]])
    model = Model(np.array([[.2], [.8]]), p, .5)
    assert evaluate_goal_policy(p, goal_policy(p, 1, 0)[0], 1, 0)[0] == 0
    assert evaluate_goal_policy(p, goal_policy(p, 1, 1)[0], 1, 1)[0] == 1
    _, report = learn_with_resets(model, 1, 2, 89, metric_priority_epoch=None)
    assert report["goal_horizon"] == 1 and report["min_true_goal_probability"] == 1


def individual_data():
    manifest = json.loads((DATA/"run_manifest.json").read_text())
    spec = json.loads((DATA/"experiment_spec.json").read_text())
    assert hashlib.sha256((DATA/"experiment_spec.json").read_bytes()).hexdigest() == manifest["spec_sha256"]
    for name, digest in manifest["result_files"].items():
        if f"verified/{name}" in PROVENANCE["withheld_results"]:
            continue
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest() == digest, name
    rows = list(csv.DictReader((DATA/"calibration.csv").open()))
    assert len(rows) == 2*500*7
    lookup = {(r["model"], int(r["seed"]), int(r["n_per_row"])): r for r in rows}
    assert len(lookup) == len(rows)
    for model_id, (states, gamma) in enumerate(spec["calibration_models"]):
        for n in spec["calibration_checkpoints_per_row"]:
            arrays = np.load(DATA/f"calibration_{model_id}_n{n}.npz")
            lo, hi, truth = arrays["lower"], arrays["upper"], arrays["true_metric"]
            assert np.max(np.abs(lo-lo.transpose(0, 2, 1))) < 2e-9
            assert np.max(np.abs(hi-hi.transpose(0, 2, 1))) < 2e-9
            assert np.min(lo) >= 0 and np.max(hi) <= 1
            assert np.max(lo-truth) < 2e-9 and np.max(truth-hi) < 2e-9
            widths = np.max(hi-lo, axis=(1, 2))
            errors = np.max(np.abs((lo+hi)/2-truth), axis=(1, 2))
            assert np.all(errors <= widths/2+2e-9)
            assert np.max(hi-lo-arrays["pair_majorant"]) < 2e-8
            if n == 0:
                assert np.max(errors) <= .5+2e-9
            else:
                assert np.max(abs(arrays["reward_estimates"]-arrays["reward_success_counts"]/n)) < 1e-12
            for index, seed in enumerate(arrays["seeds"]):
                row = lookup[(f"ring{states}_g{gamma}", int(seed), n)]
                assert int(row["all_pairs_covered"]) == 1
                assert abs(float(row["sup_width"])-widths[index]) < 1e-12
                assert abs(float(row["midpoint_sup_error"])-errors[index]) < 1e-12
    propagation = list(csv.DictReader((DATA/"propagation.csv").open()))
    assert len(propagation) == 100*3*6
    assert all(int(r["covered_R"]) == 1 and float(r["local_source_R"]) == 0 for r in propagation)
    assert all(float(r["width_R"]) <= float(r["pair_majorant_R"])+2e-8 for r in propagation)
    ablation = list(csv.DictReader((DATA/"ablation.csv").open()))
    assert len(ablation) == 2*32*4
    for task in spec["ablation_tasks"]:
        for seed in spec["ablation_seeds"]:
            group = [r for r in ablation if r["task"] == task and int(r["seed"]) == seed]
            assert len(group) == 4
            assert all(int(r["steps"]) <= int(r["step_budget"]) for r in group)
            assert len({r["empirical_model_hash"] for r in group if int(r["completed"])}) == 1
    for row in csv.DictReader((DATA/"clock_budgets.csv").open()):
        m, d = int(row["m"]), int(row["d"])
        M, epsilon = m**d, 1/(32*m)
        n = int(np.ceil(8/epsilon**2*np.log(2*M/.1)))
        assert int(row["resolution_query_budget"]) == M*n+1


def extended_data():
    """V5-V7 records against their manifest and invariants."""
    extended = ROOT/"results"/"extended"
    manifest = json.loads((extended/"run_manifest.json").read_text())
    spec = json.loads((extended/"experiment_spec.json").read_text())
    assert hashlib.sha256((extended/"experiment_spec.json").read_bytes()).hexdigest() == manifest["spec_sha256"]
    for name, digest in manifest["result_files"].items():
        assert hashlib.sha256((extended/name).read_bytes()).hexdigest() == digest, name
    for name, digest in manifest["frozen_inputs"].items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest() == digest, name
    baseline = list(csv.DictReader((extended/"baselines.csv").open()))
    assert len(baseline) == 2*len(spec["baseline_checkpoints_per_row"])*len(spec["baseline_seeds"])*4
    assert all(int(r["all_pairs_covered"]) == 1 for r in baseline if r["method"] in ["certified", "global"])
    scaling = list(csv.DictReader((extended/"scaling.csv").open()))
    assert len(scaling) == len(spec["scaling_states"])*len(spec["scaling_seeds"])*len(spec["scaling_checkpoints_per_row"])
    for r in scaling:
        assert float(r["majorant_violation"]) < 2e-8 and float(r["upper_triangle_violation"]) < 2e-8
        if r["anchors"]:
            assert int(r["anchors"]) <= spec["scaling_blocks"]
            assert float(r["anchor_distortion"]) <= max(float(r["sup_width"]), 2*spec["scaling_anchor_radius"])+2e-9
    anytime = list(csv.DictReader((extended/"anytime.csv").open()))
    assert len(anytime) == len(spec["anytime_tasks"])*len(spec["anytime_priorities"])*len(spec["anytime_seeds"])
    for task in spec["anytime_tasks"]:
        for seed in spec["anytime_seeds"]:
            group = [r for r in anytime if r["task"] == task and int(r["seed"]) == seed]
            assert len({r["empirical_model_hash"] for r in group if int(r["completed"])}) <= 1


def refined_operators():
    """Exact three-successor transport, baseline equivalence, and the refined sandwich."""
    from finite_general import (SupportTransport, solve_refined, kl_reward_interval,
                                support_indices, radii)
    rng = np.random.default_rng(1234)
    s, a = 5, 2
    p = np.zeros((s, a, s))
    for i in range(s):
        for act in range(a):
            k = rng.integers(1, 4)
            p[i, act, rng.choice(s, k, replace=False)] = rng.dirichlet(np.ones(k))
    h = rng.uniform(0, 1, (s, s))
    h = (h+h.T)/2
    np.fill_diagonal(h, 0)
    values = SupportTransport(p).apply(h)[0]
    for i in range(s):
        for j in range(s):
            for act in range(a):
                assert abs(values[i, j, act]-transport(p[i, act], p[j, act], h)) < 1e-10
    r = rng.uniform(.1, .9, (s, a))
    br, bp = np.full_like(r, .03), np.full_like(r, .02)
    base = solve_refined(r-br, r+br, r, p, .8, bp)
    lo_ref, hi_ref, _ = intervals(Model(r, p, .8), br, bp)
    assert np.max(abs(base["upper"][0]-hi_ref)) < 2e-8 and np.max(abs(base["lower"][0]-lo_ref)) < 2e-8
    truth = solve_refined(r, r, r, p, .8, np.zeros_like(r), compute_majorant=False)["upper"][0]
    # Empirical models inside the radii must keep the true metric between refined endpoints.
    for trial in range(20):
        noise = rng.uniform(-1, 1, r.shape)*.03
        phat = p.copy()
        for i in range(s):
            for act in range(a):
                support = np.flatnonzero(p[i, act] > 0)
                if len(support) > 1:
                    shift = min(.02, p[i, act, support].min())*rng.uniform(-1, 1)
                    phat[i, act, support[0]] += shift
                    phat[i, act, support[1]] -= shift
        rh = np.clip(r+noise, 0, 1)
        counts = np.full((s, a), 4096)
        _, _, ell = radii(counts, 3, .1)
        lo, hi = kl_reward_interval(r, counts, ell)
        lo, hi = np.minimum(lo, rh-.0301), np.maximum(hi, rh+.0301)
        lo, hi = np.minimum(lo, r), np.maximum(hi, r)
        tv = np.full_like(r, .02)
        for mode in ["global", "support"]:
            out = solve_refined(lo, hi, rh, phat, .8, tv, diameter=mode, support_index=support_indices(p))
            assert np.all(out["lower"][0] <= truth+2e-9) and np.all(truth <= out["upper"][0]+2e-9)
            assert np.max(out["upper"]-out["lower"]-out["majorant"]) < 2e-8


def extended2_data():
    """V8-V11 records against their manifests and invariants."""
    folder = ROOT/"results"/"extended2"
    manifest = json.loads((folder/"run_manifest.json").read_text())
    assert hashlib.sha256((folder/"experiment_spec.json").read_bytes()).hexdigest() == manifest["spec_sha256"]
    for name, digest in manifest["result_files"].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest, name
    for name, digest in manifest["frozen_inputs"].items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest() == digest, name
    for name in ["v1_refined.csv", "riverswim_generative.csv"]:
        rows = list(csv.DictReader((folder/name).open()))
        assert all(int(r["all_pairs_covered"]) == 1 for r in rows if r["method"] != "bootstrap_band"), name
    rows = list(csv.DictReader((folder/"v6_refined.csv").open()))
    certified = [r for r in rows if r["method"] != "bootstrap_band"]
    assert all(int(r["all_pairs_covered"]) == 1 and int(r["false_merge"]) == 0 and int(r["false_split"]) == 0
               for r in certified)
    neural = ROOT/"results"/"neural"
    manifest = json.loads((neural/"run_manifest.json").read_text())
    assert hashlib.sha256((neural/"experiment_spec.json").read_bytes()).hexdigest() == manifest["spec_sha256"]
    for name, digest in manifest["result_files"].items():
        assert hashlib.sha256((neural/name).read_bytes()).hexdigest() == digest, name


def code_provenance():
    """The released code matches the code recorded in every run manifest."""
    for study in ["verified", "extended", "extended2", "neural"]:
        manifest = json.loads((ROOT/"results"/study/"run_manifest.json").read_text())
        for name, digest in manifest["code_sha256"].items():
            entry = PROVENANCE["files"][name]
            assert entry["run_sha256"] == digest, (study, name)
            path = ROOT/"code"/name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["release_sha256"], name
            assert logic_fingerprint(path) == entry["release_logic"], name
            if entry["logic_identical"]:
                assert entry["release_logic"] == entry["run_logic"], name


def neural_algebra():
    import torch
    from neural_primitives import SymmetricWidthCritic, surrogate_endpoints, width_target, representation_loss
    torch.manual_seed(43)
    critic = SymmetricWidthCritic(3, 2)
    z, other = torch.randn(6, 3), torch.randn(6, 3)
    actions = torch.tensor([0, 1, 0, 1, 0, 1])
    a, b = critic(z, other, actions), critic(other, z, actions)
    assert torch.allclose(a, b) and torch.all((a >= 0) & (a <= 1))
    # Equal embeddings at distinct observations must not force zero uncertainty.
    assert torch.all(critic(z, z, actions) > 0)
    mask = torch.tensor([True, False, False, False, False, False])
    assert critic(z, z, actions, mask)[0] == 0
    point, width = torch.tensor([.02, .5, .98]), torch.tensor([.8, .2, .8])
    endpoints = surrogate_endpoints(point, width)
    assert torch.all(endpoints["interval_width"] <= width+1e-7)
    assert endpoints["midpoint"][0] > point[0]
    p, q = torch.tensor([[.4, .6]]), torch.tensor([[.7, .3]])
    plan = p[:, :, None]*q[:, None, :]
    successor = torch.tensor([[[.1, .3], [.5, .7]]])
    target = width_target(torch.tensor([.03]), plan, successor, .8, p=p, q=q)
    assert torch.allclose(target, .06+.8*(plan*successor).sum((1, 2)))
    distance = torch.tensor([.1, .4, .7], requires_grad=True)
    temporal = torch.tensor([True, False, False])
    loss = representation_loss(distance, endpoints["midpoint"], endpoints["interval_width"], temporal)
    scaled = representation_loss(distance, 7*endpoints["midpoint"], 7*endpoints["interval_width"], temporal, reward_max=7.)
    assert torch.allclose(loss, scaled)
    loss.backward()
    assert torch.isfinite(distance.grad).all()


if __name__ == "__main__":
    import importlib.util
    start = time.time()
    functions = [original.arbitrary_cost_transport, original.sandwich_and_majorant,
            original.lower_cost_counterexample, original.certified_anchors,
            original.path_intersection, original.scale_and_schedule,
            original.controlled_grid, original.reachability_error_bound,
            original.reward_geometry_counterexample, original.weighted_exploration_and_quota,
            independent_transport, bounded_majorant, quota_and_horizon,
            individual_data, extended_data, refined_operators, extended2_data, code_provenance]
    skipped = []
    if importlib.util.find_spec("torch"):
        functions.append(neural_algebra)
    else:
        skipped.append({"check": "neural_algebra", "reason": "Optional PyTorch is not installed"})
    for function in functions:
        check(function.__name__, function)
    report = {"passed": len(CHECKS), "checks": CHECKS, "skipped": skipped, "seconds": time.time()-start,
              "cpu_identifier": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
              "scope": "Construction, implementation, provenance, and individual-data verification"}
    (ROOT/"reports").mkdir(exist_ok=True)
    (ROOT/"reports"/"verification.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf8")
    print(json.dumps(report, indent=2))
