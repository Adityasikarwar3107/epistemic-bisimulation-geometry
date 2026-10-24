"""Reference implementation of the finite-state confidence operators.

Transport is solved in primal form because the lower cost need not be a
pseudometric. TV(p, q) = ||p - q||_1 / 2.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.optimize import linprog


@lru_cache(maxsize=128)
def _constraints(m, n):
    out = np.zeros((m+n, m*n))
    for i in range(m):
        out[i, i*n:(i+1)*n] = 1
    for j in range(n):
        out[m+j, j::n] = 1
    return out


def transport(p, q, cost, return_plan=False):
    """Optimal coupling for an arbitrary nonnegative ground cost."""
    ip, iq = np.flatnonzero(p > 0), np.flatnonzero(q > 0)
    pp, qq = p[ip].copy(), q[iq].copy()
    pp /= pp.sum()
    qq /= qq.sum()
    cc = cost[np.ix_(ip, iq)]
    m, n = cc.shape
    if m == 1 or n == 1 or np.ptp(cc) < 1e-14:
        plan = np.outer(pp, qq)
    elif m == n == 2:
        lo, hi = max(0., pp[0]+qq[0]-1.), min(pp[0], qq[0])
        slope = cc[0, 0]-cc[0, 1]-cc[1, 0]+cc[1, 1]
        z = lo if slope >= 0 else hi
        plan = np.array([[z, pp[0]-z],
                         [qq[0]-z, 1.-pp[0]-qq[0]+z]])
    else:
        opt = linprog(cc.ravel(), A_eq=_constraints(m, n),
                      b_eq=np.r_[pp, qq], bounds=(0, None), method="highs")
        if not opt.success:
            raise RuntimeError(opt.message)
        plan = opt.x.reshape(m, n)
    value = float(np.sum(cc*plan))
    if not return_plan:
        return value
    full = np.zeros(cost.shape)
    full[np.ix_(ip, iq)] = plan
    return value, full


@dataclass
class Model:
    reward: np.ndarray
    transition: np.ndarray
    gamma: float = .75
    reward_max: float = 1.

    def __post_init__(self):
        s, a = self.reward.shape
        assert self.transition.shape == (s, a, s)
        assert 0 <= self.gamma < 1
        assert np.min(self.reward) >= 0
        assert np.max(self.reward) <= self.reward_max+1e-12
        assert np.min(self.transition) >= -1e-12
        assert np.allclose(self.transition.sum(axis=2), 1)


def confidence_radii(counts, support, delta, reward_max=1.):
    """Anytime visit-count confidence, including an explicit n=0 convention."""
    s, a = counts.shape
    n = np.maximum(counts, 1)
    ell = np.log(4*s*a*np.pi**2*n**2/(3*delta))
    br = reward_max*np.minimum(1, np.sqrt(ell/(2*n)))
    bp = np.minimum(1, np.sqrt((support*np.log(2)+ell)/(2*n)))
    br[counts == 0] = reward_max
    bp[counts == 0] = 1.
    return br, bp


def metric_operator(model, cost, kind="true", br=None, bp=None):
    s, a = model.reward.shape
    out = np.zeros((s, s))
    br = np.zeros((s, a)) if br is None else br
    bp = np.zeros((s, a)) if bp is None else bp
    g, rmax = model.gamma, model.reward_max
    for i in range(s):
        for j in range(i+1, s):
            values = []
            for act in range(a):
                rd = abs(model.reward[i, act]-model.reward[j, act])
                ot = transport(model.transition[i, act],
                               model.transition[j, act], cost)
                er, ep = br[i, act]+br[j, act], bp[i, act]+bp[j, act]
                if kind == "upper":
                    val = (1-g)*(rd+er)+g*(ot+rmax*ep)
                elif kind == "lower":
                    val = (1-g)*max(rd-er, 0)+g*max(ot-rmax*ep, 0)
                elif kind == "true":
                    val = (1-g)*rd+g*ot
                else:
                    raise ValueError(kind)
                values.append(val)
            val = max(values)
            if kind == "upper":
                val = min(rmax, val)
            out[i, j] = out[j, i] = val
    return out


def metric_fixed_point(model, kind="true", br=None, bp=None, tolerance=1e-10):
    s = model.reward.shape[0]
    d = np.zeros((s, s))
    for iteration in range(10000):
        nxt = metric_operator(model, d, kind, br, bp)
        residual = float(np.max(abs(nxt-d)))
        d = nxt
        if residual/(1-model.gamma) <= tolerance:
            residual = float(np.max(abs(metric_operator(model, d, kind, br, bp)-d)))
            return d, residual, iteration+1
    raise RuntimeError("metric iteration did not converge")


def intervals(model, br, bp, tolerance=1e-10):
    lo, lr, li = metric_fixed_point(model, "lower", br, bp, tolerance)
    hi, hr, hi_iter = metric_fixed_point(model, "upper", br, bp, tolerance)
    lo = np.maximum(0, lo-lr/(1-model.gamma))
    hi = np.minimum(model.reward_max, hi+hr/(1-model.gamma))
    np.fill_diagonal(lo, 0)
    np.fill_diagonal(hi, 0)
    return lo, hi, {"lower_residual": lr, "upper_residual": hr,
                    "iterations": max(li, hi_iter)}


def pair_majorant(model, lower, br, bp, tolerance=1e-10):
    s, a = model.reward.shape
    g, rmax = model.gamma, model.reward_max
    plans = np.zeros((s, s, a, s, s))
    bonus = np.zeros((s, s, a))
    for i in range(s):
        for act in range(a):
            plans[i, i, act] = np.diag(model.transition[i, act])
    for i in range(s):
        for j in range(i+1, s):
            for act in range(a):
                _, cp = transport(model.transition[i, act],
                                  model.transition[j, act], lower, True)
                plans[i, j, act] = cp
                plans[j, i, act] = cp.T
                b = (1-g)*(br[i, act]+br[j, act])
                b += g*rmax*(bp[i, act]+bp[j, act])
                bonus[i, j, act] = bonus[j, i, act] = b
    u = np.zeros((s, s))
    for _ in range(10000):
        nxt = np.max(2*bonus+g*np.einsum("ijaxy,xy->ija", plans, u), axis=2)
        if np.max(abs(nxt-u))/(1-g) <= tolerance:
            return nxt, bonus, plans
        u = nxt
    raise RuntimeError("pair uncertainty iteration did not converge")


def triangle_violation(cost):
    return float(max(0., np.max(cost[:, :, None]
                                -cost[:, None, :]-cost[None, :, :])))


def monotone_envelope(lowers, uppers):
    """Safe intersection; path closure restores the upper triangle inequality."""
    lo = np.maximum.reduce(lowers)
    hi = np.minimum.reduce(uppers).copy()
    for k in range(len(hi)):
        hi = np.minimum(hi, hi[:, k, None]+hi[None, k, :])
    return lo, hi


def anchor_embedding(upper, centers):
    return upper[:, centers]


def embedding_distance(z):
    return np.max(abs(z[:, None, :]-z[None, :, :]), axis=2)


def greedy_cover(upper, radius):
    centers = [0]
    while True:
        distances = np.min(upper[:, centers], axis=1)
        idx = int(np.argmax(distances))
        if distances[idx] <= radius+1e-12:
            return centers
        centers.append(idx)


def certainty_weight(mean_width, mean_distance, floor=.1):
    if mean_width+mean_distance == 0:
        return 1.
    return float(np.clip(mean_distance/(mean_distance+mean_width), floor, 1.))


def metric_row_priorities(model, br, bp, unknown):
    """Bounded [1/2,1] priorities from the exact propagated pair majorant."""
    lo, _, _ = intervals(model, br, bp)
    u, b, plans = pair_majorant(model, lo, br, bp)
    scores = np.max(2*b+model.gamma*np.einsum("ijaxy,xy->ija", plans, u), axis=1)
    weights = np.ones_like(scores)
    maximum = float(np.max(scores[unknown])) if np.any(unknown) else 0.
    if maximum > 0:
        weights[unknown] = .5+.5*scores[unknown]/maximum
    return weights


def certificate_quota(states, actions, support, gamma, reward_max,
                      epsilon, horizon, reach_probability, delta):
    if not (states >= 1 and actions >= 1 and 1 <= support <= states
            and 0 <= gamma < 1 and reward_max > 0 and epsilon > 0
            and horizon >= 1 and 0 < reach_probability <= 1 and 0 < delta < 1):
        raise ValueError("Invalid certificate parameters")
    alpha = min((1-gamma)*epsilon/(32*reward_max),
                reach_probability/(8*horizon))
    def sufficient(n):
        ell = np.log(8*states*actions*np.pi**2/(3*delta))+2*np.log(n)
        return (support*np.log(2)+ell)/(2*n) <= alpha**2
    lo, hi = 1, 2
    while not sufficient(hi):
        hi *= 2
    while lo < hi:
        mid = (lo+hi)//2
        if sufficient(mid):
            hi = mid
        else:
            lo = mid+1
    return lo, alpha


def reach_policy(transition, unknown, horizon, priorities=None):
    """Optimistic reach-to-unknown DP, stopped at the first unknown action."""
    s, a, _ = transition.shape
    if priorities is None:
        priorities = np.ones((s, a))
    assert priorities.shape == (s, a)
    assert np.all((priorities[unknown] >= .5) & (priorities[unknown] <= 1.))
    value = np.zeros(s)
    policies = np.zeros((horizon+1, s), dtype=int)
    for h in range(1, horizon+1):
        q = np.einsum("sak,k->sa", transition, value)
        q[unknown] = priorities[unknown]
        policies[h] = np.argmax(q, axis=1)
        value = q[np.arange(s), policies[h]]
    return policies, value


def goal_policy(transition, goal, horizon):
    s = transition.shape[0]
    value = np.eye(s)[goal]
    policies = np.zeros((horizon+1, s), dtype=int)
    for h in range(1, horizon+1):
        q = np.einsum("sak,k->sa", transition, value)
        policies[h] = np.argmax(q, axis=1)
        value = np.max(q, axis=1)
        value[goal] = 1.
    return policies, value


def evaluate_goal_policy(transition, policy, goal, horizon):
    s = transition.shape[0]
    value = np.eye(s)[goal]
    for h in range(1, horizon+1):
        value = np.einsum("sk,k->s", transition[np.arange(s), policy[h]], value)
        value[goal] = 1.
    return value


def learn_with_resets(model, quota, horizon, seed, metric_priority_epoch=128,
                      max_episodes=None, support_bound=None, delta=.05):
    """Interaction only: the planner never reads the true transition kernel."""
    rng = np.random.default_rng(seed)
    s, a = model.reward.shape
    counts = np.zeros((s, a), dtype=int)
    outcomes = np.zeros((s, a, s))
    rewards = np.zeros((s, a))
    episodes = steps = 0
    priorities = None
    while np.min(counts) < quota:
        if max_episodes is not None and episodes >= max_episodes:
            raise RuntimeError("Explicit episode budget exhausted before completion")
        phat = np.full((s, a, s), 1/s)
        np.divide(outcomes, np.maximum(counts, 1)[:, :, None],
                  out=phat, where=(counts > 0)[:, :, None])
        unknown = counts < quota
        if metric_priority_epoch is not None and episodes % metric_priority_epoch == 0:
            br, bp = confidence_radii(counts, support_bound or s, delta, model.reward_max)
            fitted = Model(rewards/np.maximum(counts, 1), phat,
                           model.gamma, model.reward_max)
            priorities = metric_row_priorities(fitted, br, bp, unknown)
        policy, _ = reach_policy(phat, unknown, horizon, priorities)
        state = 0
        episodes += 1
        for h in range(horizon, 0, -1):
            act = policy[h, state]
            successor = rng.choice(s, p=model.transition[state, act])
            reward = rng.binomial(1, model.reward[state, act]/model.reward_max)*model.reward_max
            steps += 1
            if unknown[state, act]:
                counts[state, act] += 1
                outcomes[state, act, successor] += 1
                rewards[state, act] += reward
                break
            state = successor
    phat = outcomes/counts[:, :, None]
    rhat = rewards/counts
    goal_horizon = horizon-1
    min_reach = min(float(evaluate_goal_policy(
        model.transition, goal_policy(phat, goal, goal_horizon)[0], goal, goal_horizon)[0])
        for goal in range(s))
    return Model(rhat, phat, model.gamma, model.reward_max), {
        "steps": steps, "episodes": episodes, "quota": quota,
        "min_true_goal_probability": min_reach, "goal_horizon": goal_horizon,
        "retained_samples": int(counts.sum())}
