"""Vectorized exact transport for rows with at most two successors.

The 2x2 transport polytope is a segment, so the optimal plan is one of its endpoints.
"""
import numpy as np


class PairTransport:
    def __init__(self, transition):
        p = np.asarray(transition, dtype=float)
        if p.ndim == 3:
            p = p[None]
        if np.any(p < 0) or not np.allclose(p.sum(-1), 1, atol=1e-12):
            raise ValueError("Invalid probability rows")
        if np.max(np.count_nonzero(p > 0, axis=-1)) > 2:
            raise ValueError("The vectorized solver supports at most two successors")
        self.batch, self.states, self.actions, _ = p.shape
        indices = np.argsort(-p, axis=-1, kind="stable")[..., :2]
        if self.states == 1:
            indices = np.repeat(indices, 2, axis=-1)
        self.i0, self.i1 = indices[..., 0], indices[..., 1]
        probability = np.take_along_axis(p, indices[..., :1], axis=-1)[..., 0]
        self.p = probability[:, :, None, :]
        self.q = probability[:, None, :, :]
        self.low = np.maximum(0., self.p+self.q-1)
        self.high = np.minimum(self.p, self.q)
        self.batch_index = np.arange(self.batch)[:, None, None, None]
        self.diagonal = np.arange(self.states)

    def entries(self, cost):
        h = np.asarray(cost)
        if h.ndim == 2:
            h = h[None]
        b = self.batch_index
        return [h[b, left[:, :, None, :], right[:, None, :, :]]
                for left, right in [(self.i0, self.i0), (self.i0, self.i1),
                                    (self.i1, self.i0), (self.i1, self.i1)]]

    def coefficients(self, cost):
        c00, c01, c10, c11 = self.entries(cost)
        z = np.where(c00-c01-c10+c11 >= 0, self.low, self.high)
        return [z, self.p-z, self.q-z, 1-self.p-self.q+z]

    def apply(self, cost, coefficients=None):
        entries = self.entries(cost)
        weights = self.coefficients(cost) if coefficients is None else coefficients
        return sum(w*c for w, c in zip(weights, entries))


def fixed_point(operator, shape, gamma, tolerance=2e-11, max_iterations=10000):
    value = np.zeros(shape)
    for iteration in range(max_iterations):
        updated = operator(value)
        step = float(np.max(np.abs(updated-value)))
        value = updated
        if step/(1-gamma) <= tolerance:
            residual = np.max(np.abs(operator(value)-value), axis=(-2, -1))
            return value, residual, iteration+1
    raise RuntimeError("Contraction did not converge within the numerical budget")


def solve(reward, transition, gamma, reward_max=1., br=None, bp=None,
          tolerance=2e-11, compute_majorant=True):
    reward = np.asarray(reward, dtype=float)
    if reward.ndim == 2:
        reward = reward[None]
    transport = PairTransport(transition)
    batch, states, actions = reward.shape
    assert batch == transport.batch
    br = np.zeros_like(reward) if br is None else np.broadcast_to(br, reward.shape)
    bp = np.zeros_like(reward) if bp is None else np.broadcast_to(bp, reward.shape)
    gaps = np.abs(reward[:, :, None, :]-reward[:, None, :, :])
    er = br[:, :, None, :]+br[:, None, :, :]
    ep = bp[:, :, None, :]+bp[:, None, :, :]
    bonus = (1-gamma)*er+gamma*reward_max*ep
    diagonal = transport.diagonal
    bonus[:, diagonal, diagonal] = 0

    def operator(cost, kind):
        ot = transport.apply(cost)
        if kind == "lower":
            q = (1-gamma)*np.maximum(gaps-er, 0)+gamma*np.maximum(ot-reward_max*ep, 0)
        else:
            q = (1-gamma)*gaps+gamma*ot
            if kind == "upper":
                q = q+bonus
        value = np.max(q, axis=-1)
        if kind == "upper":
            value = np.minimum(reward_max, value)
        value[:, diagonal, diagonal] = 0
        return value

    shape = (batch, states, states)
    if br.max() == 0 and bp.max() == 0:
        truth, residual, iterations = fixed_point(
            lambda h: operator(h, "true"), shape, gamma, tolerance)
        return {"truth": truth, "residual": residual, "iterations": iterations}
    lower, lr, li = fixed_point(lambda h: operator(h, "lower"), shape, gamma, tolerance)
    upper, ur, ui = fixed_point(lambda h: operator(h, "upper"), shape, gamma, tolerance)
    lo_error, hi_error = lr/(1-gamma), ur/(1-gamma)
    padded_lo = np.maximum(0, lower-lo_error[:, None, None])
    padded_hi = np.minimum(reward_max, upper+hi_error[:, None, None])
    padded_lo[:, diagonal, diagonal] = padded_hi[:, diagonal, diagonal] = 0
    result = {"lower": padded_lo, "upper": padded_hi, "lower_raw": lower,
              "lower_residual": lr, "upper_residual": ur,
              "iterations": max(li, ui), "bonus": bonus}
    if compute_majorant:
        weights = transport.coefficients(lower)
        source = 2*bonus+2*gamma*lo_error[:, None, None, None]
        source[:, diagonal, diagonal] = 0

        def width_operator(width):
            out = np.minimum(reward_max, np.max(
                source+gamma*transport.apply(width, weights), axis=-1))
            out[:, diagonal, diagonal] = 0
            return out

        majorant, wr, wi = fixed_point(width_operator, shape, gamma, tolerance)
        padding = wr/(1-gamma)+lo_error+hi_error
        majorant = np.minimum(reward_max, majorant+padding[:, None, None])
        majorant[:, diagonal, diagonal] = 0
        result.update(majorant=majorant, majorant_residual=wr,
                      majorant_iterations=wi)
    return result
