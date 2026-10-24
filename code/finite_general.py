"""Exact transport for rows with up to three successors and the refined endpoints.

SupportTransport enumerates the 81 spanning-tree bases of the 3x3 transportation
polytope. solve_refined implements Proposition 12 (KL rewards, diameter refinement).
"""
from itertools import combinations

import numpy as np

from finite_batch import PairTransport


def fixed_point(operator, shape, gamma, tolerance=2e-11, max_iterations=10000, initial=None):
    """As finite_batch.fixed_point, optionally warm-started; the stopping rule is unchanged."""
    value = np.zeros(shape) if initial is None else np.array(initial, dtype=float)
    for iteration in range(max_iterations):
        updated = operator(value)
        step = float(np.max(np.abs(updated-value)))
        value = updated
        if step/(1-gamma) <= tolerance:
            residual = np.max(np.abs(operator(value)-value), axis=(-2, -1))
            return value, residual, iteration+1
    raise RuntimeError("Contraction did not converge within the numerical budget")


def _tree_maps():
    edges = [(i, j) for i in range(3) for j in range(3)]
    maps = []
    for chosen in combinations(range(9), 5):
        parent = list(range(6))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        acyclic = True
        for e in chosen:
            i, j = edges[e]
            a, b = find(i), find(3+j)
            if a == b:
                acyclic = False
                break
            parent[a] = b
        if not acyclic:
            continue
        system = np.zeros((5, 5))
        for k, e in enumerate(chosen):
            i, j = edges[e]
            system[i, k] = 1
            if j < 2:
                system[3+j, k] = 1
        inverse = np.linalg.inv(system)
        full = np.zeros((9, 6))
        for k, e in enumerate(chosen):
            full[e, :5] = inverse[k]
        maps.append(full)
    assert len(maps) == 81
    return np.array(maps)


TREE_MAPS = _tree_maps()


class SupportTransport:
    """Exact transport between rows whose successors lie in at most three labels."""

    def __init__(self, transition, tolerance=1e-12):
        p = np.asarray(transition, dtype=float)
        if p.ndim == 3:
            p = p[None]
        if np.max(np.count_nonzero(p > 0, axis=-1)) > 3:
            raise ValueError("SupportTransport supports at most three successors")
        self.batch, self.states, self.actions, _ = p.shape
        index = np.argsort(-p, axis=-1, kind="stable")[..., :3]
        mass = np.take_along_axis(p, index, axis=-1)
        self.index = index
        left = mass[:, :, None, :, :]
        right = mass[:, None, :, :, :]
        marginals = np.concatenate([np.broadcast_to(left, (self.batch, self.states, self.states, self.actions, 3)),
                                    np.broadcast_to(right, (self.batch, self.states, self.states, self.actions, 3))], -1)
        self.flows = np.einsum("tef,bijaf->bijate", TREE_MAPS, marginals)
        self.feasible = (self.flows >= -tolerance).all(-1)
        self.flows = np.maximum(self.flows, 0)
        b = np.arange(self.batch)[:, None, None, None, None, None]
        rows = index[:, :, None, :, :, None]
        cols = index[:, None, :, :, None, :]
        self.rows = np.broadcast_to(rows, (self.batch, self.states, self.states, self.actions, 3, 3))
        self.cols = np.broadcast_to(cols, (self.batch, self.states, self.states, self.actions, 3, 3))
        self.b = b

    def _entries(self, cost):
        h = np.asarray(cost)
        if h.ndim == 2:
            h = h[None]
        return h[self.b, self.rows, self.cols].reshape(self.batch, self.states, self.states, self.actions, 9)

    def coefficients(self, cost):
        values = np.einsum("bijate,bijae->bijat", self.flows, self._entries(cost))
        values = np.where(self.feasible, values, np.inf)
        best = values.argmin(-1)
        return np.take_along_axis(self.flows, best[..., None, None], axis=-2)[..., 0, :]

    def apply(self, cost, coefficients=None):
        weights = self.coefficients(cost) if coefficients is None else coefficients
        return (weights*self._entries(cost)).sum(-1)


def transport_for(transition):
    p = np.asarray(transition)
    if np.max(np.count_nonzero(p > 0, axis=-1)) <= 2:
        return PairTransport(p)
    return SupportTransport(p)


def kl_bernoulli(p, q):
    p = np.clip(p, 0, 1)
    q = np.clip(q, 1e-15, 1-1e-15)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p > 0, p*np.log(p/q), 0.)
        b = np.where(p < 1, (1-p)*np.log((1-p)/(1-q)), 0.)
    return a+b


def kl_reward_interval(mean, counts, ell, reward_max=1.):
    """Chernoff-Hoeffding (KL) interval for [0, R]-bounded rewards, same tail budget as Hoeffding."""
    m = np.clip(np.asarray(mean, float)/reward_max, 0, 1)
    n = np.maximum(np.asarray(counts), 1)
    level = ell/n
    lo, hi = np.zeros_like(m), m.copy()
    up_lo, up_hi = m.copy(), np.ones_like(m)
    for _ in range(60):
        mid = (lo+hi)/2
        outside = kl_bernoulli(m, mid) > level
        lo, hi = np.where(outside, mid, lo), np.where(outside, hi, mid)
        mid = (up_lo+up_hi)/2
        outside = kl_bernoulli(m, mid) > level
        up_lo, up_hi = np.where(outside, up_lo, mid), np.where(outside, mid, up_hi)
    lower, upper = lo, up_hi
    zero = np.asarray(counts) == 0
    lower, upper = np.where(zero, 0., lower), np.where(zero, 1., upper)
    return lower*reward_max, upper*reward_max


def radii(counts, support, delta, reward_max=1.):
    """Hoeffding/Weissman radii of Lemma 2; support may be a scalar or a per-row array."""
    s, a = counts.shape[-2:]
    n = np.maximum(counts, 1)
    ell = np.log(4*s*a*np.pi**2*n**2/(3*delta))
    br = reward_max*np.minimum(1, np.sqrt(ell/(2*n)))
    bp = np.minimum(1, np.sqrt((np.asarray(support)*np.log(2)+ell)/(2*n)))
    bp = np.where(np.asarray(support) <= 1, 0., bp)
    br = np.where(counts == 0, reward_max, br)
    bp = np.where((counts == 0) & (np.asarray(support) > 1), 1., bp)
    return br, bp, ell


def solve_refined(reward_lo, reward_hi, reward_hat, transition, gamma, tv, reward_max=1.,
                  diameter="none", support_index=None, tolerance=2e-11, max_stages=40,
                  compute_majorant=True):
    """Endpoints from reward intervals and TV radii, with optional diameter refinement.

    diameter: "none" uses R_max; "global" iterates D = ||d^+||; "support" uses, for each row,
    the largest upper-endpoint distance among its known successor labels (support_index).
    """
    rl, rh = (np.asarray(x, float) for x in (reward_lo, reward_hi))
    if rl.ndim == 2:
        rl, rh = rl[None], rh[None]
    tv = np.broadcast_to(np.asarray(tv, float), rl.shape)
    transport = transport_for(transition)
    batch, states, actions = rl.shape
    diagonal = np.arange(states)
    upper_gap = np.maximum(rh[:, :, None, :]-rl[:, None, :, :], rh[:, None, :, :]-rl[:, :, None, :])
    lower_gap = np.maximum(0, np.maximum(rl[:, :, None, :]-rh[:, None, :, :], rl[:, None, :, :]-rh[:, :, None, :]))
    row_diameter = np.full((batch, states, actions), float(reward_max))
    clip = np.full(batch, float(reward_max))
    stages = []
    previous_upper = None
    for stage in range(max_stages):
        perturb = gamma*(row_diameter*tv)
        pert = perturb[:, :, None, :]+perturb[:, None, :, :]
        cap = clip[:, None, None]

        def upper_op(cost):
            q = (1-gamma)*upper_gap+gamma*transport.apply(cost)+pert
            value = np.minimum(cap, q.max(-1))
            value[:, diagonal, diagonal] = 0
            return value

        def lower_op(cost):
            q = (1-gamma)*lower_gap+np.maximum(gamma*transport.apply(cost)-pert, 0)
            value = q.max(-1)
            value[:, diagonal, diagonal] = 0
            return value
        # Warm start; the residual padding keeps the endpoints valid.
        warm_upper = None if stage == 0 else upper
        warm_lower = None if stage == 0 else lower
        upper, ur, ui = fixed_point(upper_op, (batch, states, states), gamma, tolerance, initial=warm_upper)
        lower, lr, li = fixed_point(lower_op, (batch, states, states), gamma, tolerance, initial=warm_lower)
        hi_error, lo_error = ur/(1-gamma), lr/(1-gamma)
        padded_hi = np.minimum(cap, upper+hi_error[:, None, None])
        padded_lo = np.maximum(0, lower-lo_error[:, None, None])
        padded_hi[:, diagonal, diagonal] = padded_lo[:, diagonal, diagonal] = 0
        stages.append(float(np.mean(padded_hi.max((1, 2)))))
        if diameter == "none":
            break
        if previous_upper is not None and np.max(np.abs(padded_hi-previous_upper)) < 1e-9:
            break
        previous_upper = padded_hi
        # Each refinement uses only an upper endpoint already valid on the same event.
        clip = np.minimum(clip, padded_hi.max((1, 2)))
        if diameter == "global":
            new = np.broadcast_to(clip[:, None, None], row_diameter.shape)
        else:
            idx = support_index
            sub = padded_hi[:, idx[:, :, :, None], idx[:, :, None, :]]
            new = sub.max((-2, -1))
        row_diameter = np.minimum(row_diameter, new)
    result = {"lower": padded_lo, "upper": padded_hi, "lower_raw": lower, "stages": len(stages),
              "stage_mean_sup_upper": stages, "row_diameter": row_diameter,
              "lower_residual": lr, "upper_residual": ur}
    if compute_majorant:
        weights = transport.coefficients(lower)
        source = (1-gamma)*(upper_gap-lower_gap)+2*pert+2*gamma*lo_error[:, None, None, None]
        source[:, diagonal, diagonal] = 0

        def width_operator(width):
            out = np.minimum(reward_max, np.max(source+gamma*transport.apply(width, weights), axis=-1))
            out[:, diagonal, diagonal] = 0
            return out
        majorant, wr, _ = fixed_point(width_operator, (batch, states, states), gamma, tolerance)
        padding = wr/(1-gamma)+lo_error+hi_error
        majorant = np.minimum(reward_max, majorant+padding[:, None, None])
        majorant[:, diagonal, diagonal] = 0
        result.update(majorant=majorant, source=source)
    return result


def support_indices(transition):
    """Known successor labels of each row, padded by repetition to three entries."""
    p = np.asarray(transition)
    s, a, _ = p.shape
    out = np.zeros((s, a, 3), dtype=int)
    for i in range(s):
        for j in range(a):
            labels = list(np.flatnonzero(p[i, j] > 0))
            assert 1 <= len(labels) <= 3
            out[i, j] = (labels+[labels[0]]*3)[:3]
    return out
