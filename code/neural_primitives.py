"""PyTorch building blocks of the neural extension (Appendix F)."""
import torch
from torch import nn


class SymmetricWidthCritic(nn.Module):
    def __init__(self, latent_dim, actions, reward_max=1., hidden=128):
        super().__init__()
        self.reward_max = float(reward_max)
        self.actions = actions
        self.network = nn.Sequential(nn.Linear(2*latent_dim+actions, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, z, other, action, self_pair=None):
        features = torch.cat([(z-other).abs(), (z+other)/2,
            torch.nn.functional.one_hot(action, self.actions).to(z.dtype)], dim=-1)
        value = self.reward_max*self.network(features).squeeze(-1)
        # An embedding collision does not establish identity of observations.
        if self_pair is not None:
            value = value.masked_fill(self_pair, 0.)
        return value


def surrogate_endpoints(point, majorant, reward_max=1., self_pair=None):
    point = point.clamp(0., reward_max)
    majorant = majorant.clamp(0., reward_max)
    lower = (point-majorant/2).clamp(0., reward_max)
    upper = (point+majorant/2).clamp(0., reward_max)
    if self_pair is not None:
        lower = lower.masked_fill(self_pair, 0.)
        upper = upper.masked_fill(self_pair, 0.)
        majorant = majorant.masked_fill(self_pair, 0.)
    return {"lower": lower, "upper": upper, "midpoint": (lower+upper)/2,
            "interval_width": upper-lower, "majorant": majorant}


def width_target(source, coupling, successor_width, gamma, reward_max=1.,
                 plan_suboptimality=0., p=None, q=None):
    if not 0 <= gamma < 1:
        raise ValueError("Discount must lie in [0,1)")
    if torch.any(source < 0) or torch.any(successor_width < 0) or torch.any(successor_width > reward_max):
        raise ValueError("Sources and successor majorants must have their declared bounds")
    if torch.any(coupling < 0) or not torch.allclose(
            coupling.sum((-2, -1)), torch.ones_like(source), atol=1e-6):
        raise ValueError("Coupling must be a nonnegative unit-mass plan")
    if p is not None and not torch.allclose(coupling.sum(-1), p, atol=1e-6):
        raise ValueError("Incorrect row marginal")
    if q is not None and not torch.allclose(coupling.sum(-2), q, atol=1e-6):
        raise ValueError("Incorrect column marginal")
    expectation = (coupling*successor_width).sum((-2, -1))
    return (2*source+gamma*plan_suboptimality+gamma*expectation).clamp(0., reward_max).detach()


def representation_loss(distance, midpoint, interval_width, temporal,
                        reward_max=1., tau=1., zeta=.05):
    if reward_max <= 0 or zeta <= 0 or not 0 <= tau <= 1:
        raise ValueError("Invalid normalized loss parameters")
    target = (midpoint/reward_max).detach()
    width = (interval_width/reward_max).detach()
    residual = (distance-target).square()/(zeta*zeta+width.square())
    loss = distance.sum()*0
    for mask, weight in [(temporal, tau), (~temporal, 1.)]:
        if mask.any():
            loss = loss+weight*residual[mask].mean()
    return loss
