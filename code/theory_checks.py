"""Deterministic checks of the constructions used in the proofs."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import linprog
from ebg import (Model, transport, metric_fixed_point, metric_operator, intervals,
                 pair_majorant, triangle_violation, anchor_embedding,
                 embedding_distance, greedy_cover, monotone_envelope,
                 certainty_weight, reach_policy, metric_row_priorities,
                 certificate_quota, confidence_radii)

ROOT = Path(__file__).resolve().parents[1]
CHECKS = []


def check(name, fn):
    fn()
    CHECKS.append(name)
    print("PASS:", name, flush=True)


def fixture():
    p = np.array([[[.8, .2, 0], [0, .7, .3]],
                  [[.2, .8, 0], [0, .2, .8]],
                  [[.1, 0, .9], [.3, 0, .7]]])
    return Model(np.array([[.1, .7], [.4, .2], [.8, .5]]), p, .65)


def arbitrary_cost_transport():
    rng = np.random.default_rng(29)
    for _ in range(12):
        p = rng.dirichlet(np.ones(2)); q = rng.dirichlet(np.ones(2))
        c = rng.random((2, 2))
        val, plan = transport(p, q, c, True)
        ae = np.array([[1, 1, 0, 0], [0, 0, 1, 1],
                       [1, 0, 1, 0], [0, 1, 0, 1]])
        ref = linprog(c.ravel(), A_eq=ae, b_eq=np.r_[p, q],
                      bounds=(0, None), method="highs")
        assert abs(val-ref.fun) < 1e-9
        assert np.allclose(plan.sum(axis=1), p)
        assert np.allclose(plan.sum(axis=0), q)
        ph = .95*p+.05*np.array([1., 0.])
        qh = .9*q+.1*np.array([0., 1.])
        error = abs(val-transport(ph, qh, c))
        tv = .5*(np.sum(abs(p-ph))+np.sum(abs(q-qh)))
        assert error <= tv+1e-9
        cp = np.clip(c+.01, 0, 1)
        assert abs(val-transport(p, q, cp)) <= np.max(abs(c-cp))+1e-9


def sandwich_and_majorant():
    true = fixture()
    rh = np.clip(true.reward+.01*np.array([[1, -1], [-1, 1], [1, -1]]), 0, 1)
    ph = .98*true.transition+.02*np.ones_like(true.transition)/3
    fit = Model(rh, ph, true.gamma)
    br = abs(rh-true.reward)
    bp = .5*np.sum(abs(ph-true.transition), axis=2)
    exact, _, _ = metric_fixed_point(true)
    lo, hi, _ = intervals(fit, br, bp)
    assert np.all(lo <= exact+1e-8) and np.all(exact <= hi+1e-8)
    assert triangle_violation(hi) < 1e-8
    assert np.max(abs((lo+hi)/2-exact)) <= np.max(hi-lo)/2+1e-8
    u, b, plans = pair_majorant(fit, lo, br, bp)
    delta = hi-lo
    rhs = np.max(2*b+true.gamma*np.einsum("ijaxy,xy->ija", plans, delta), axis=2)
    assert np.all(delta <= rhs+1e-8)
    assert np.all(delta <= u+1e-8)
    assert np.allclose(plans.sum(axis=(3, 4)), 1)
    k = [0, 1]
    bk = np.max(b[np.ix_(k, k, range(2))])
    eta = np.max(fit.transition[k, :, 2])
    bound = (2*bk+2*true.gamma*eta)/(1-true.gamma)
    assert np.max(delta[np.ix_(k, k)]) <= bound+1e-8


def lower_cost_counterexample():
    r = np.array([[0.], [.5], [1.]])
    p = np.eye(3)[:, None, :]
    m = Model(r, p, gamma=0)
    lo, _, _ = intervals(m, np.array([[0.], [.5], [0.]]), np.zeros((3, 1)))
    assert lo[0, 1] == lo[1, 2] == 0 and lo[0, 2] == 1
    assert triangle_violation(lo) == 1


def certified_anchors():
    model = fixture()
    exact, _, _ = metric_fixed_point(model)
    lo, hi, _ = intervals(model, np.full((3, 2), .001), np.zeros((3, 2)))
    width = np.max(hi-lo)
    z = anchor_embedding(hi, list(range(3)))
    assert np.allclose(embedding_distance(z), hi)
    assert np.max(abs(embedding_distance(z)-exact)) <= width+1e-8
    radius = max(.2, 2*width)
    centers = greedy_cover(hi, radius)
    assert np.max(np.min(hi[:, centers], axis=1)) <= radius+1e-9
    distance = embedding_distance(anchor_embedding(hi, centers))
    assert np.max(abs(distance-exact)) <= max(width, 2*radius)+1e-8


def path_intersection():
    u1 = np.array([[0., 1, 3], [1, 0, 2], [3, 2, 0]])
    u2 = np.array([[0., 2, 3], [2, 0, 1], [3, 1, 0]])
    truth = np.array([[0., .5, 1], [.5, 0, .5], [1, .5, 0]])
    assert triangle_violation(np.minimum(u1, u2)) == 1
    lo, hi = monotone_envelope([truth], [u1, u2])
    assert np.all(lo <= truth) and np.all(truth <= hi)
    assert triangle_violation(hi) == 0
    assert hi[0, 2] == 2


def scale_and_schedule():
    model = fixture()
    br = np.full((3, 2), .01); bp = np.full((3, 2), .02)
    lo, hi, _ = intervals(model, br, bp)
    scaled = Model(model.reward*7, model.transition, model.gamma, 7.)
    ls, hs, _ = intervals(scaled, br*7, bp)
    assert np.allclose(ls, lo*7, atol=1e-8)
    assert np.allclose(hs, hi*7, atol=1e-8)
    weights = [certainty_weight(w, .2) for w in [0, .1, 1, 10]]
    assert np.all(np.diff(weights) <= 0)
    assert certainty_weight(0, 0) == 1
    assert certainty_weight(.4, .2) == certainty_weight(4, 2)
    v1, v2, tau, y = 2, 3, .4, .8
    biased = (v1*tau+v2)*y/(v1+v2)
    assert biased < y


def controlled_grid():
    m, d = 3, 2
    eps = 1/(32*m)
    points = np.array([(x, y) for x in range(m) for y in range(m)])
    rewards = np.zeros((len(points)+1, d+1))
    rewards[1:, :d] = .25+points/(4*m)
    rewards[1:, d] = .75+2*eps*np.where(np.arange(len(points)) % 2, 1, -1)
    p = np.eye(len(rewards))[:, None, :].repeat(d+1, axis=1)
    model = Model(rewards, p, .6)
    exact, _, _ = metric_fixed_point(model)
    expected = np.max(abs(rewards[:, None, :]-rewards[None, :, :]), axis=2)
    assert np.allclose(exact, expected, atol=1e-8)
    geometry = np.max(abs(points[:, None, :]-points[None, :, :]), axis=2)/(4*m)
    assert np.allclose(exact[1:, 1:], geometry, atol=1e-8)
    assert len(greedy_cover(exact, eps)) == len(rewards)


def reachability_error_bound():
    model = fixture()
    ph = .99*model.transition+.01*np.ones_like(model.transition)/3
    unknown = np.zeros((3, 2), dtype=bool); unknown[2, 1] = True
    h = 8
    policy, fitted = reach_policy(ph, unknown, h)
    _, optimal_true = reach_policy(model.transition, unknown, h)
    value = np.zeros(3)
    for remaining in range(1, h+1):
        nxt = np.zeros(3)
        for s in range(3):
            a = policy[remaining, s]
            nxt[s] = 1 if unknown[s, a] else model.transition[s, a] @ value
        value = nxt
    alpha = np.max(.5*np.sum(abs(ph-model.transition), axis=2))
    assert value[0] >= optimal_true[0]-2*h*alpha-1e-9


def reward_geometry_counterexample():
    r = np.array([[0., 1.], [0., 1.]])
    p = np.eye(2)[:, None, :].repeat(2, axis=1)
    m = Model(r, p, .8)
    exact, _, _ = metric_fixed_point(m)
    assert np.max(exact) == 0
    assert abs(0-1/(1-m.gamma)) > exact[0, 1]/(1-m.gamma)


def weighted_exploration_and_quota():
    model = fixture()
    unknown = np.zeros((3, 2), dtype=bool)
    unknown[1, 0] = unknown[2, 1] = True
    weights = metric_row_priorities(model, np.full((3, 2), .02),
                                   np.full((3, 2), .03), unknown)
    assert np.all((weights[unknown] >= .5) & (weights[unknown] <= 1))
    _, weighted_value = reach_policy(model.transition, unknown, 8, weights)
    _, hit_value = reach_policy(model.transition, unknown, 8)
    assert np.all(weighted_value >= .5*hit_value-1e-9)
    assert np.all(weighted_value <= hit_value+1e-9)
    n, alpha = certificate_quota(3, 2, 3, .65, 1., .2, 8, .5, .05)
    br, bp = confidence_radii(np.full((3, 2), n), 3, .05/2)
    assert np.max(br) <= alpha+1e-12 and np.max(bp) <= alpha+1e-12
    assert 4*alpha/(1-model.gamma) <= .2/8+1e-12


if __name__ == "__main__":
    for name, fn in [
        ("Arbitrary-cost primal OT and stability", arbitrary_cost_transport),
        ("Sandwich, pair Bellman bound, and escape bound", sandwich_and_majorant),
        ("Nonmetric lower-endpoint counterexample", lower_cost_counterexample),
        ("Certified anchor and compressed representations", certified_anchors),
        ("Safe monotone path intersection", path_intersection),
        ("Reward scaling and monotone loss schedule", scale_and_schedule),
        ("Exact controlled-grid lower-bound construction", controlled_grid),
        ("Stopped empirical-navigation error bound", reachability_error_bound),
        ("Arbitrary-policy value-bound counterexample", reward_geometry_counterexample),
        ("Metric-priority floor and certificate sampling quota", weighted_exploration_and_quota)]:
        check(name, fn)
    report = {"passed": len(CHECKS), "checks": CHECKS,
              "scope": "Implementation and construction checks, not manuscript experiments."}
    (ROOT/"reports").mkdir(exist_ok=True)
    (ROOT/"reports"/"theory_checks.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf8")
